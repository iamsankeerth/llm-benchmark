"""Model-entry helpers for benchmark queue records.

The queue still crosses script seams as dictionaries for compatibility, but
this module concentrates the field names, status rules, and filename rules in
one place.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ModelEntry:
    queue_id: str
    requested_name: str
    category: str
    source: str
    resolved_runtime: str
    resolved_model_ref: str
    variant_note: str = ""
    ollama_tag: str = ""
    hf_repo: str = ""
    size: str = "?"
    estimated_tps: float = 0
    fit_level: str = ""
    is_moe: bool = False
    status: str = "pending"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelEntry":
        return cls(
            queue_id=data.get("queue_id", ""),
            requested_name=data.get("requested_name", data.get("name", "")),
            category=data.get("category", ""),
            source=data.get("source", ""),
            resolved_runtime=data.get("resolved_runtime", ""),
            resolved_model_ref=data.get("resolved_model_ref", ""),
            variant_note=data.get("variant_note", ""),
            ollama_tag=data.get("ollama_tag", ""),
            hf_repo=data.get("hf_repo", ""),
            size=data.get("size", "?"),
            estimated_tps=data.get("estimated_tps", data.get("tps", 0)),
            fit_level=data.get("fit_level", ""),
            is_moe=bool(data.get("is_moe", data.get("moe", False))),
            status=data.get("status", "pending"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def is_runnable(self) -> bool:
        return self.status == "pending"

    @property
    def is_deferred_vision(self) -> bool:
        return self.status == "deferred_vision" or self.resolved_runtime == "deferred_vision"

    @property
    def is_provider_unsupported(self) -> bool:
        return self.status == "provider_unsupported" or self.source == "provider_unsupported"

    @property
    def runtime_ref(self) -> str:
        return self.ollama_tag or self.resolved_model_ref or self.hf_repo

    @property
    def safe_queue_id(self) -> str:
        return safe_name(self.queue_id)

    @property
    def safe_runtime_ref(self) -> str:
        return safe_name(self.runtime_ref)

    @property
    def safe_requested_name(self) -> str:
        return safe_name(self.requested_name.lower().replace(" ", "-"))

    def is_runtime(self, *runtimes: str) -> bool:
        return self.resolved_runtime in runtimes


def as_model_entry(data: ModelEntry | dict[str, Any]) -> ModelEntry:
    if isinstance(data, ModelEntry):
        return data
    return ModelEntry.from_dict(data)


def safe_name(value: str) -> str:
    return str(value).replace(":", "_").replace("/", "_").replace("\\", "_")
