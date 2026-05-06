import re
from abc import ABC, abstractmethod
from typing import Iterator

from core.tom_task import ToMTask


class DatasetAdapter(ABC):
    def __init__(self, path: str):
        self.path = path

    @abstractmethod
    def load(self) -> Iterator[ToMTask]:
        ...

    @staticmethod
    def _strip_options(text: str) -> str:
        """Remove trailing inline-option blocks like 'A) basket B) box' or 'A. x B. y'."""
        stripped = re.sub(r'(?:\s+[A-Z][).]\s+\S+)+\s*$', '', text.strip())
        return stripped.strip() or text.strip()
