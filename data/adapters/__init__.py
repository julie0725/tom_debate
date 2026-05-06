from data.adapters.dataset_adapter import DatasetAdapter
from data.adapters.bigtom_adapter import BigToMAdapter
from data.adapters.hitom_adapter import HiToMAdapter

# To add a new dataset: create data/adapters/<name>_adapter.py,
# then add one import and one entry here. No other files need to change.
REGISTRY: dict[str, type[DatasetAdapter]] = {
    "bigtom": BigToMAdapter,
    "hitom": HiToMAdapter,
}


def get_adapter(dataset_name: str, path: str) -> DatasetAdapter:
    cls = REGISTRY.get(dataset_name)
    if cls is None:
        raise ValueError(
            f"No adapter for '{dataset_name}'. Available: {list(REGISTRY)}"
        )
    return cls(path)
