"""
Interfacing with OpenAI models.
"""

import json
import os
import sys
from typing import Literal, cast

from litellm import NotGiven
from loguru import logger
from openai import NOT_GIVEN, BadRequestError, OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion_message_tool_call import (
    Function as OpenaiFunction,
)
from openai.types.chat.chat_completion_tool_choice_option_param import (
    ChatCompletionToolChoiceOptionParam,
)
from openai.types.chat.completion_create_params import ResponseFormat
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from app.data_structures import FunctionCallIntent
from app.log import log_and_always_print
from app.model import common
from app.model.common import Model


def _log_before_retry(retry_state) -> None:
    """Print to stdout before each retry sleep so a stalled call is visible.

    Without this, a failing ``call`` sleeps silently inside tenacity and the
    whole run looks frozen (vllm sits idle with 0 requests waiting).
    """
    exc = retry_state.outcome.exception()
    sleep = getattr(retry_state.next_action, "sleep", 0.0)
    log_and_always_print(
        f"[vllm.call] attempt {retry_state.attempt_number} failed: "
        f"{type(exc).__name__}: {exc}. Retrying in {sleep:.1f}s "
        f"(max {VLLM_MAX_RETRIES} attempts)."
    )


# Local vllm has no rate limits, so retries only help with transient/network
# errors. Keep waits short so a real failure surfaces in seconds, not minutes.
# BadRequestError (e.g. context_length_exceeded) is deterministic: retrying it
# is pointless, so it is excluded and raised immediately.
VLLM_MAX_RETRIES = 3


def _estimate_tokens(text: str) -> int:
    """Cheap, dependency-free token estimate (~chars / ratio).

    Used only for the safety cap below, not for billing. The ratio is
    configurable via GENERIC_VLLM_CHARS_PER_TOKEN; lowering it trims more
    aggressively (safer), raising it trims less.
    """
    ratio = float(os.getenv("GENERIC_VLLM_CHARS_PER_TOKEN", "4.0"))
    return int(len(text) / ratio) if ratio > 0 else len(text)


def _msg_tokens(msg: dict) -> int:
    total = 4  # rough per-message role/formatting overhead
    content = msg.get("content")
    if isinstance(content, str):
        total += _estimate_tokens(content)
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        total += _estimate_tokens(str(fn.get("name", "")) + str(fn.get("arguments", "")))
    return total


def truncate_messages(messages: list[dict]) -> list[dict]:
    """Bound the input size so the conversation cannot outgrow the context window.

    Opt-in and caller-agnostic: controlled entirely by GENERIC_VLLM_MAX_INPUT_TOKEN.
    When that env var is unset (or <= 0) this is a no-op, so behavior is identical
    to not having the cap at all. Set the SAME value across model runs to keep
    them comparable.

    Strategy: keep the leading system message(s) and the first user message (the
    issue statement) as an anchor, drop the oldest middle messages, and keep the
    most recent messages that fit the budget. A placeholder marks the cut.
    """
    budget_env = os.getenv("GENERIC_VLLM_MAX_INPUT_TOKEN")
    if not budget_env:
        return messages
    try:
        budget = int(budget_env)
    except ValueError:
        return messages
    if budget <= 0:
        return messages

    total = sum(_msg_tokens(m) for m in messages)
    if total <= budget:
        return messages

    # Anchor head: leading system messages + the first user message (issue stmt).
    head: list[dict] = []
    i = 0
    while i < len(messages) and messages[i].get("role") == "system":
        head.append(messages[i])
        i += 1
    if i < len(messages) and messages[i].get("role") == "user":
        head.append(messages[i])
        i += 1

    placeholder = {
        "role": "user",
        "content": "[Note: earlier conversation context was truncated to fit the model's context window.]",
    }

    remaining = budget - sum(_msg_tokens(m) for m in head) - _msg_tokens(placeholder)

    # Greedily keep the most recent messages that fit the remaining budget.
    tail: list[dict] = []
    used = 0
    for m in reversed(messages[i:]):
        t = _msg_tokens(m)
        if used + t > remaining:
            break
        tail.append(m)
        used += t
    tail.reverse()

    # Repair tool-call pairing at the junction: a leading "tool" message would be
    # orphaned (its parent assistant with tool_calls was dropped). Drop such.
    while tail and tail[0].get("role") == "tool":
        tail.pop(0)

    result = head + [placeholder] + tail
    new_total = sum(_msg_tokens(m) for m in result)
    log_and_always_print(
        f"[vllm.call] Truncated context: {len(messages)} -> {len(result)} messages, "
        f"~{total} -> ~{new_total} est. input tokens (budget {budget}). "
        "Set GENERIC_VLLM_MAX_INPUT_TOKEN to tune."
    )
    if new_total > budget:
        log_and_always_print(
            "[vllm.call] WARNING: still over budget after truncation; the anchor "
            "messages (system prompt + issue) alone exceed it. The request may be "
            "rejected by vllm."
        )
    return result


class OpenAISKDModel(Model):
    """
    Base class for creating Singleton instances of OpenAISKD models.
    """

    _instances = {}

    def __new__(cls):
        if cls not in cls._instances:
            cls._instances[cls] = super().__new__(cls)
            cls._instances[cls]._initialized = False
        return cls._instances[cls]

    def __init__(
        self,
        name: str,
        max_output_token: int,
        cost_per_input: float,
        cost_per_output: float,
        model_id: str | None = None,
        parallel_tool_call: bool = False,
    ):
        if self._initialized:
            return
        super().__init__(
            name, cost_per_input, cost_per_output, parallel_tool_call
        )
        # max number of output tokens allowed in model response
        # sometimes we want to set a lower number for models with smaller context window,
        # because output token limit consumes part of the context window
        self.max_output_token = max_output_token
        # client for making request
        self.client: OpenAI | None = None
        # pid that created self.client; used to detect fork() and rebuild the
        # client in the child (httpx/httpcore connection pools are NOT fork-safe).
        self._client_pid: int | None = None
        self._initialized = True
        self.model_id = model_id if model_id is not None else name

    def setup(self) -> None:
        """
        Check API key, and initialize OpenAI client.

        Fork-safe: ACR forks worker processes (main.py uses ProcessPoolExecutor
        with the 'fork' start method). An OpenAI/httpx client created in the
        parent and inherited by a forked child shares an unsafe connection pool,
        which can wedge the child in the HTTP layer before the request is ever
        sent. So we (re)build the client whenever the current pid differs from
        the one that created it.
        """
        if self.client is None or self._client_pid != os.getpid():
            base_url = self.check_base_url()
            key = self.check_api_key()
            timeout = float(os.getenv("GENERIC_VLLM_TIMEOUT", "1200"))
            self.client = OpenAI(base_url=base_url, api_key=key, timeout=timeout)
            self._client_pid = os.getpid()

    def check_base_url(self) -> str:
        base_url = os.getenv("OPENAI_BASE_URL")
        if not base_url:
            logger.error(
                "Please set the OPENAI_BASE_URL env var (e.g. https://api.openai.com/v1 or http://localhost:8080/v1)"
            )
            sys.exit(1)
        return base_url

    def check_api_key(self) -> str:
        key = os.getenv("OPENAI_KEY")
        if not key:
            logger.info(
                "OPENAI_KEY env var not set, using dummy key. This will only work if the base URL is set to a local test server that does not require authentication."
            )
            return "dummy"
        return key

    def extract_resp_content(
        self, chat_completion_message: ChatCompletionMessage
    ) -> str:
        """
        Given a chat completion message, extract the content from it.
        """
        content = chat_completion_message.content
        if content is None:
            return ""
        else:
            return content

    def extract_resp_func_calls(
        self,
        chat_completion_message: ChatCompletionMessage,
    ) -> list[FunctionCallIntent]:
        """
        Given a chat completion message, extract the function calls from it.
        Args:
            chat_completion_message (ChatCompletionMessage): The chat completion message.
        Returns:
            List[FunctionCallIntent]: A list of function calls.
        """
        result = []
        tool_calls = chat_completion_message.tool_calls
        if tool_calls is None:
            return result

        call: ChatCompletionMessageToolCall
        for call in tool_calls:
            called_func: OpenaiFunction = call.function
            func_name = called_func.name
            func_args_str = called_func.arguments
            # maps from arg name to arg value
            if func_args_str == "":
                args_dict = {}
            else:
                try:
                    args_dict = json.loads(func_args_str, strict=False)
                except json.decoder.JSONDecodeError:
                    args_dict = {}
            func_call_intent = FunctionCallIntent(
                func_name, args_dict, called_func
            )
            result.append(func_call_intent)

        return result

    # FIXME: the returned type contains OpenAI specific Types, which should be avoided
    @retry(
        retry=retry_if_not_exception_type(BadRequestError),
        wait=wait_random_exponential(min=1, max=10),
        stop=stop_after_attempt(VLLM_MAX_RETRIES),
        before_sleep=_log_before_retry,
    )
    def call(
        self,
        messages: list[dict],
        top_p: float = 1,
        tools: list[dict] | None = None,
        response_format: Literal["text", "json_object"] = "text",
        temperature: float | None = None,
        **kwargs,
    ) -> tuple[
        str,
        list[ChatCompletionMessageToolCall] | None,
        list[FunctionCallIntent],
        float,
        int,
        int,
    ]:
        """
        Calls the openai API to generate completions for the given inputs.
        Assumption: we only retrieve one choice from the API response.

        Args:
            messages (List): A list of messages.
                            Each item is a dict (e.g. {"role": "user", "content": "Hello, world!"})
            top_p (float): The top_p to use. We usually do not vary this, so not setting it as a cmd-line argument. (from 0 to 1)
            tools (List, optional): A list of tools.

        Returns:
            Raw response and parsed components.
            The raw response is to be sent back as part of the message history.
        """
        if temperature is None:
            temperature = common.MODEL_TEMP

        # Bound input size before sending (no-op unless GENERIC_VLLM_MAX_INPUT_TOKEN
        # is set). Keeps the growing search thread from overflowing the window.
        messages = truncate_messages(messages)

        # Rebuild the client if we have been forked into a new process; reusing
        # the parent's httpx connection pool across fork() can hang the request.
        self.setup()
        assert self.client is not None
        try:
            logger.debug(
                "Calling model {} with {} messages, max_output_token={}",
                self.name,
                len(messages),
                self.max_output_token if not self.name.startswith("o1") else NOT_GIVEN,
            )

            if tools is not None and len(tools) == 1:
                # there is only one tool => force the model to use it
                tool_name = tools[0]["function"]["name"]
                tool_choice = {
                    "type": "function",
                    "function": {"name": tool_name},
                }
                response: ChatCompletion = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,  # type: ignore
                    tools=tools,  # type: ignore
                    tool_choice=cast(
                        ChatCompletionToolChoiceOptionParam, tool_choice
                    ),
                    # temperature=(
                    #     temperature if self.name.startswith("o1") else NOT_GIVEN
                    # ),
                    temperature=0,
                    response_format=cast(
                        ResponseFormat, {"type": response_format}
                    ),
                    max_tokens=(
                        self.max_output_token
                        if not self.name.startswith("o1")
                        else NOT_GIVEN
                    ),
                    max_completion_tokens=(
                        self.max_output_token
                        if self.name.startswith("o1")
                        else NOT_GIVEN
                    ),
                    # top_p=top_p,
                    stream=False,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
            else:
                response: ChatCompletion = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,  # type: ignore
                    tools=tools if tools is not None else NOT_GIVEN,  # type: ignore
                    # temperature=(
                    #     temperature if self.name.startswith("o1") else NOT_GIVEN
                    # ),
                    temperature=0,
                    response_format=cast(
                        ResponseFormat, {"type": response_format}
                    ),
                    max_tokens=(
                        self.max_output_token
                        if not self.name.startswith("o1")
                        else NOT_GIVEN
                    ),
                    max_completion_tokens=(
                        self.max_output_token
                        if self.name.startswith("o1")
                        else NOT_GIVEN
                    ),
                    # top_p=top_p,
                    stream=False,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )

            usage_stats = response.usage
            assert usage_stats is not None

            input_tokens = int(usage_stats.prompt_tokens)
            output_tokens = int(usage_stats.completion_tokens)
            cost = self.calc_cost(input_tokens, output_tokens)

            logger.debug(
                "Model {} call completed: input_tokens={}, output_tokens={}, cost=${:.6f}",
                self.name,
                input_tokens,
                output_tokens,
                cost,
            )

            common.thread_cost.process_cost += cost
            common.thread_cost.process_input_tokens += input_tokens
            common.thread_cost.process_output_tokens += output_tokens

            raw_response = response.choices[0].message
            # log_and_print(f"Raw model response: {raw_response}")
            content = self.extract_resp_content(raw_response)
            raw_tool_calls = raw_response.tool_calls
            func_call_intents = self.extract_resp_func_calls(raw_response)
            return (
                content,
                raw_tool_calls,
                func_call_intents,
                cost,
                input_tokens,
                output_tokens,
            )
        except BadRequestError as e:
            logger.debug("BadRequestError ({}): messages={}", e.code, messages)
            num_msgs = len(messages)
            approx_chars = sum(len(str(m.get("content") or "")) for m in messages)
            if e.code == "context_length_exceeded":
                log_and_always_print(
                    f"[vllm.call] Context length exceeded: {num_msgs} messages, "
                    f"~{approx_chars} chars (~{approx_chars // 4} tokens) sent. "
                    "The request was NOT retried. The conversation thread has "
                    "outgrown the vllm context window (max-model-len minus "
                    "max_tokens)."
                )
            else:
                log_and_always_print(
                    f"[vllm.call] BadRequestError ({e.code}): {e}. Not retried."
                )
            raise e


class GenericVLLM(OpenAISKDModel):
    def __init__(self):
        super().__init__(
            "generic-vllm",
            int(os.getenv("GENERIC_VLLM_MAX_OUTPUT_TOKEN", "16384")),
            0.0,
            0.0,
            model_id=os.getenv("GENERIC_VLLM_MODEL_ID", "generic-vllm"),
        )
