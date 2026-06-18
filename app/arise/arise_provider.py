from loguru import logger

from app.arise.arise_shim import ARISEBinaryShim


class ARISEProvider:
    def __init__(self):
        self.shim = ARISEBinaryShim()

    def call_arise(self, command_name: str, args: list[str]):
        logger.debug(f"Calling ARISE command: {command_name} with args: {args}")
        return self.shim.call_arise(command_name, args)

    def arise_search(self, root_dir: str, query: str, top_k: int | None = None):
        """
        Search for code entities (classes, functions, modules) in the repository graph by text query.
        Returns a JSON list of matching entities with their file paths and line numbers, sorted by relevance score.
        Use this to locate where things are defined before reading or editing them.
        """
        args = [root_dir, query]
        if top_k is not None:
            args.append(str(top_k))
        return self.call_arise("arise_search", args)

    def arise_get_entity_info(self, root_dir: str, node_id: str):
        """
        Return metadata and graph-connectivity summary for a specific node by its ID.
        Shows file location, type, name, line range, and edge counts grouped by relation type.
        Use after arise_search to inspect a located entity in detail.
        """
        return self.call_arise("arise_get_entity_info", [root_dir, node_id])

    def arise_get_code_span(
        self, root_dir: str, file_path: str, start_line: int, end_line: int
    ):
        """
        Read a specific range of lines from a file in the repository.
        Returns the code text with its file path and line numbers as JSON.
        Use after arise_search to read the source of a located entity.
        """
        return self.call_arise(
            "arise_get_code_span",
            [root_dir, file_path, str(start_line), str(end_line)],
        )

    def arise_get_enclosing_scopes(
        self, root_dir: str, file_path: str, line: int
    ):
        """
        Return the enclosing scopes (module, class, function/method) for a given line in a file.
        Results are ordered from innermost to outermost.
        Useful for understanding what context a line of code lives in.
        """
        return self.call_arise(
            "arise_get_enclosing_scopes", [root_dir, file_path, str(line)]
        )

    def arise_traverse_relations(
        self,
        root_dir: str,
        node_id: str,
        max_hops: int | None = None,
        direction: str | None = None,
        relation_types: str | None = None,
    ):
        """
        Traverse the repository graph from a seed node, following edges up to max_hops away.
        Returns a JSON subgraph (nodes + edges) reachable from the seed under the given constraints.
        Useful for exploring call graphs, inheritance hierarchies, and import chains.

        direction must be 'out' (default) or 'in'.
        relation_types is an optional comma-separated list of edge types to follow
        (e.g. 'calls,contains'). Valid types: contains, imports, imported_by, calls, called_by, inherits.
        """
        args = [root_dir, node_id]
        if max_hops is not None:
            args.append(str(max_hops))
        if direction is not None:
            args.append(direction)
        if relation_types is not None:
            args.append(relation_types)
        return self.call_arise("arise_traverse_relations", args)

    def arise_get_dataflow_slice(
        self,
        root_dir: str,
        file_path: str,
        line: int,
        variable: str,
        direction: str | None = None,
    ):
        """
        Trace the intra-procedural data-flow slice for a variable at a given line.
        Returns a JSON array of SliceStep objects, each with file_path, start_line, end_line, variable, and role.

        direction can be 'backward' (default, find definitions), 'forward' (find uses), or 'both'.
        Returns an explanatory message if the line has no associated statement node.
        Use this to understand where a variable is defined or consumed within a function.
        """
        args = [root_dir, file_path, str(line), variable]
        if direction is not None:
            args.append(direction)
        return self.call_arise("arise_get_dataflow_slice", args)

    def arise_build_context_bundle(
        self,
        root_dir: str,
        issue_text: str,
        seed_ids: str,
        token_budget: int | None = None,
    ):
        """
        Assemble a ranked set of code spans relevant to the given issue under a token budget.
        Scores candidates by TF-IDF relevance to the issue, structural proximity to seed nodes,
        and data-flow slice membership. Returns a JSON object with 'spans' and 'total_tokens'.

        seed_ids is a comma-separated list of node ID strings (as returned by arise_search).
        """
        args = [root_dir, issue_text, seed_ids]
        if token_budget is not None:
            args.append(str(token_budget))
        return self.call_arise("arise_build_context_bundle", args)

    def arise_rank_suspects(
        self,
        root_dir: str,
        issue_text: str,
        stack_trace: str | None = None,
        top_k: int | None = None,
    ):
        """
        Rank functions and methods by their suspicion score for the given issue.
        Seeds from stack trace frames and TF-IDF search, expands via call graph, then scores
        using relevance, proximity, and data-flow slice membership.
        Returns a JSON array of SuspectRegion objects with node_id, file_path, name, line range, and score.
        """
        args = [root_dir, issue_text]
        if stack_trace is not None:
            args.append(stack_trace)
        if top_k is not None:
            args.append(str(top_k))
        return self.call_arise("arise_rank_suspects", args)


if __name__ == "__main__":
    import os
    from rich import print
    provider = ARISEProvider()
    project_dir = os.path.join(os.getcwd(), "app/arise")
    search_results = provider.arise_search(project_dir, "def arise_search", 5)
    print(search_results)
