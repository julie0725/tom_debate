import itertools
from pathlib import Path
from typing import Iterator, Optional, Union

from core.tom_task import ToMTask


class Proxy:
    """Single entry point for all task sources: file paths and raw natural language."""

    def __init__(self, config: dict):
        self.config = config

    def get_tasks(self, input_str: str, limit: Optional[int] = None) -> Iterator[ToMTask]:
        """Return an iterator of ToMTask objects from either a file or a natural language string.

        - File path that exists  → detect_adapter(path).load()  [DatasetAdapter]
        - Plain string (no file) → TextAdapter(text, config).load()  [TextDatasetAdapter]
        Both types expose load(), so dispatch is uniform.
        """
        adapter = self._resolve(input_str)
        tasks = adapter.load()
        if limit is not None:
            return itertools.islice(tasks, limit)
        return tasks

    def _resolve(self, input_str: str):
        if Path(input_str).exists():
            from data.adapters import detect_adapter
            return detect_adapter(input_str)
        from data.adapters.text_adapter import TextAdapter
        return TextAdapter(input_str, self.config)
