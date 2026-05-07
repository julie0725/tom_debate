from pathlib import Path

from data.adapters.base import DatasetAdapter, TextDatasetAdapter
from data.adapters.csv_adapter import CsvAdapter
from data.adapters.json_adapter import JsonAdapter
from data.adapters.text_adapter import TextAdapter
from data.adapters.proxy import Proxy

__all__ = [
    "DatasetAdapter",
    "TextDatasetAdapter",
    "CsvAdapter",
    "JsonAdapter",
    "TextAdapter",
    "Proxy",
    "REGISTRY",
    "get_adapter",
    "detect_adapter",
]

# To add a new dataset: create data/adapters/<name>_adapter.py,
# then add one import and one entry here. No other files need to change.
REGISTRY: dict[str, type[DatasetAdapter]] = {
    "csv": CsvAdapter,
    "json": JsonAdapter,
}


def get_adapter(dataset_name: str, path: str) -> DatasetAdapter:
    cls = REGISTRY.get(dataset_name)
    if cls is None:
        raise ValueError(
            f"No adapter for '{dataset_name}'. Available: {list(REGISTRY)}"
        )
    return cls(path)


def detect_adapter(path: str) -> DatasetAdapter:
    """Auto-detect the correct adapter from file extension."""
    p = Path(path)
    if p.suffix == ".csv":
        return CsvAdapter(path)
    if p.suffix == ".json":
        return JsonAdapter(path)
    raise ValueError(f"Unsupported file format: {p.suffix}")
