"""Local model metadata for context summarization; no endpoint requests."""

from collections.abc import Mapping

from langchain_core.runnables.base import RunnableBindingBase

DEFAULT_SUMMARIZATION_THRESHOLD = 50000


def model_context_window(model: object) -> int | None:
    """Return the model's context window, or None when it is unknown.

    LangChain profiles supply ``max_input_tokens``. Ollama's configured
    ``num_ctx`` takes precedence because it controls the allocated window.
    """
    kwargs = {}
    while isinstance(model, RunnableBindingBase):
        # Outer bindings override inner bindings.
        kwargs = {**model.kwargs, **kwargs}
        model = model.bound

    model_name = getattr(model, "model_name", None) or getattr(
        model, "model", None
    )
    if any(
        key in kwargs and kwargs[key] != model_name
        for key in ("model", "model_name")
    ):
        # Instance metadata does not describe an overridden model.
        return None

    window = getattr(model, "num_ctx", None)
    options = kwargs.get("options")
    if hasattr(model, "num_ctx") and options is not None:
        # Ollama bound options replace instance options, including num_ctx.
        window = (
            options.get("num_ctx")
            if isinstance(options, Mapping)
            else getattr(options, "num_ctx", None)
        )
    elif window is None:
        profile = getattr(model, "profile", None)
        if isinstance(profile, Mapping):
            window = profile.get("max_input_tokens")

    if isinstance(window, int) and not isinstance(window, bool) and window > 0:
        return window
    return None


def summarization_threshold(
    model: object, fallback: int = DEFAULT_SUMMARIZATION_THRESHOLD
) -> int:
    """Return 70% of the model's context window, or the fixed fallback."""
    window = model_context_window(model)
    if window is None:
        return fallback
    return round(window * 0.7)
