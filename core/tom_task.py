from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ToMTask:
    """Single unit of work passed from a data adapter into the pipeline."""
    context: str
    question: str
    gold_answer: Optional[str] = None
    dataset_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)
