import os
from pathlib import Path
import subprocess
import json

_CMD_BIN_MAPPING = {
    "arise_search": "arise_search",
    "arise_get_entity_info": "arise_get_entity_info",
    "arise_get_code_span": "arise_get_code_span",
    "arise_get_enclosing_scopes": "arise_get_enclosing_scopes",
    "arise_traverse_relations": "arise_traverse_relations",
    "arise_get_dataflow_slice": "arise_get_dataflow_slice",
    "arise_build_context_bundle": "arise_build_context_bundle",
    "arise_rank_suspects": "arise_rank_suspects",
    "arise_explain_slice": "arise_explain_slice",
}


class ARISEBinaryShim:
    def __init__(self):
        self.arise_directory = os.getenv("ARISE_DIRECTORY", "")
        self.binary_directory = os.path.join(
            self.arise_directory, "src/arise/swe_agent_bundle/bin"
        )
        if not self.binary_directory:
            raise ValueError(
                "ARISE_BINARY_DIRECTORY environment variable is set but empty. Please provide a valid directory path."
            )

        for cmd_name in _CMD_BIN_MAPPING.keys():
            if not os.path.isfile(
                os.path.join(self.binary_directory, _CMD_BIN_MAPPING[cmd_name])
            ):
                raise FileNotFoundError(
                    f"Required ARISE command binary '{_CMD_BIN_MAPPING[cmd_name]}' not found in directory: {self.binary_directory}"
                )

    def _get_command_path(self, command_name: str):
        if command_name not in _CMD_BIN_MAPPING:
            raise ValueError(f"Unknown command name: {command_name}")
        binary_name = _CMD_BIN_MAPPING[command_name]
        command_path = os.path.join(self.binary_directory, binary_name)
        if not os.path.isfile(command_path):
            raise FileNotFoundError(f"Command binary not found: {command_path}")
        return command_path

    def _call_command(self, command_name: str, args: list[str]):
        command_path = Path(self._get_command_path(command_name)).resolve()

        arise_src = Path(self.arise_directory) / "src"

        env = os.environ.copy()
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            f"{arise_src}{os.pathsep}{existing}" if existing else str(arise_src)
        )

        cmd = ["uv", "run", str(command_path), *args]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                env=env,
            )
            # return json.loads(result.stdout)
            return result.stdout
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Command '{command_name}' failed with exit code {e.returncode}: {e.stderr}"
            ) from e
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Failed to parse JSON output from command '{command_name}': {e.msg}"
            ) from e

    def call_arise(self, command_name: str, args: list[str]):
        print(f"Calling ARISE command '{command_name}' with args: {args}", flush=True)
        return self._call_command(command_name, args)


if __name__ == "__main__":
    # Example usage
    provider = ARISEBinaryShim()
    project_dir = os.path.join(os.getcwd(), "app/arise")
    result = provider.call_arise(
        "arise_search", [project_dir, "get_command_path", "5"]
    )
    print(result)
