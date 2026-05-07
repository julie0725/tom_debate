from abc import ABC, abstractmethod
from typing import Iterator

from core.tom_task import ToMTask


class DatasetAdapter(ABC):
    """파일 기반 어댑터 인터페이스"""

    def __init__(self, path: str):
        self.path = path

    @abstractmethod
    def load(self) -> Iterator[ToMTask]:
        pass


class TextDatasetAdapter(ABC):
    """텍스트 기반 어댑터 인터페이스"""

    def __init__(self, text: str, config: dict):
        self.text = text
        self.config = config

    @abstractmethod
    def load(self) -> Iterator[ToMTask]:
        pass
