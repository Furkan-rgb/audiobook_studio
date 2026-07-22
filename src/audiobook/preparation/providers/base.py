"""Provider protocol and errors shared by local and hosted adapters."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..types import PreparationRequest, PreparationResult, ProviderMetadata


@dataclass(frozen=True)
class ProviderDescriptor:
    """What a provider offers and what it needs, known without contacting it.

    A frontend has to build its menus before any provider is instantiated, and
    preflight has to report a missing API key before a run starts.  Both are
    answered from here, so the protocol requires every adapter to declare it:
    an adapter that cannot say which models it serves cannot be offered.
    """

    name: str
    label: str
    models: tuple[str, ...]
    default_model: str = ""
    base_url: str | None = None
    api_key_env: str | None = None
    local: bool = False
    parameters: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Provider descriptor needs a name")
        if not self.models:
            raise ValueError(f"Provider {self.name!r} declares no models")
        if not self.default_model:
            object.__setattr__(self, "default_model", self.models[0])
        if self.default_model not in self.models:
            raise ValueError(
                f"Provider {self.name!r} default model {self.default_model!r} is "
                "not in its model list"
            )

    @property
    def requires_api_key(self) -> bool:
        return bool(self.api_key_env)

    def api_key(self) -> str | None:
        """The configured key, read from the environment rather than config.

        Config is committed; secrets are not.  Adapters name the variable, the
        value only ever lives in the process environment.
        """

        if not self.api_key_env:
            return None
        return os.environ.get(self.api_key_env) or None

    def missing_requirement(self) -> str | None:
        """Why this provider cannot be used yet, or None if it can."""

        if self.requires_api_key and not self.api_key():
            return f"{self.api_key_env} is not set in the environment."
        return None


class ProviderError(RuntimeError):
    """Base error for narration-preparation providers."""


class ProviderUnavailableError(ProviderError):
    """The provider service or requested model is unavailable."""


class ProviderResponseError(ProviderError):
    """The provider returned an invalid or unsuccessful response."""


@runtime_checkable
class NarrationPreparationProvider(Protocol):
    """Minimal interface implemented by Ollama and future hosted providers."""

    @classmethod
    def describe(cls) -> ProviderDescriptor:
        """Menu and requirements, answerable without constructing anything."""
        ...

    @property
    def metadata(self) -> ProviderMetadata: ...

    def check_available(self) -> None: ...

    def prepare(self, request: PreparationRequest) -> PreparationResult: ...

    def close(self) -> None: ...
