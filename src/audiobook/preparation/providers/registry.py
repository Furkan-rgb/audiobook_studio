"""Small registry that keeps workflow orchestration provider-agnostic."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import NarrationPreparationProvider, ProviderDescriptor


ProviderFactory = Callable[..., NarrationPreparationProvider]
_PROVIDERS: dict[str, ProviderFactory] = {}


def register_provider(
    name: str,
    factory: ProviderFactory,
    *,
    replace: bool = False,
) -> None:
    """Register a provider factory under a case-insensitive CLI name.

    Registration is where the protocol is enforced: the descriptor is built
    once, here, so an adapter that cannot describe its models or requirements
    fails at import rather than when a frontend tries to draw its menu.
    """

    key = name.strip().casefold()
    if not key:
        raise ValueError("Provider name cannot be blank")
    if key in _PROVIDERS and not replace:
        raise ValueError(f"Preparation provider already registered: {name}")

    describe = getattr(factory, "describe", None)
    if not callable(describe):
        raise TypeError(
            f"Preparation provider {name!r} must implement describe() -> ProviderDescriptor"
        )
    descriptor = describe()
    if not isinstance(descriptor, ProviderDescriptor):
        raise TypeError(
            f"Preparation provider {name!r}.describe() must return a ProviderDescriptor"
        )

    _PROVIDERS[key] = factory


def available_providers() -> tuple[str, ...]:
    """Return registered provider names in stable display order."""

    return tuple(sorted(_PROVIDERS))


def provider_descriptor(name: str) -> ProviderDescriptor:
    """The menu and requirements a registered provider declares.

    Rebuilt on each call rather than cached at registration: descriptors read
    config and the environment, so a model added to config or a key exported
    into the shell shows up on the next refresh instead of after a restart.
    """

    key = name.strip().casefold()
    try:
        factory = _PROVIDERS[key]
    except KeyError as exc:
        installed = ", ".join(available_providers()) or "none"
        raise ValueError(
            f"Unknown preparation provider: {name}. Installed providers: {installed}"
        ) from exc
    return factory.describe()


def provider_descriptors() -> tuple[ProviderDescriptor, ...]:
    """Descriptors for every registered provider, in display order."""

    return tuple(provider_descriptor(name) for name in available_providers())


def create_provider(name: str, **configuration: Any) -> NarrationPreparationProvider:
    """Construct a registered provider without leaking its response types."""

    key = name.strip().casefold()
    try:
        factory = _PROVIDERS[key]
    except KeyError as exc:
        installed = ", ".join(available_providers()) or "none"
        raise ValueError(
            f"Unknown preparation provider: {name}. Installed providers: {installed}"
        ) from exc
    return factory(**configuration)


__all__ = [
    "ProviderFactory",
    "available_providers",
    "create_provider",
    "provider_descriptor",
    "provider_descriptors",
    "register_provider",
]
