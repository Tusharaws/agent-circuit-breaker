from typing import Any, Callable

from mlx_lm import generate as mlx_generate
from mlx_lm import load as mlx_load

DEFAULT_MODEL_NAME = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
DEFAULT_MAX_TOKENS = 200


class SLMClient:
    """Small-language-model client (Phase 4) -- Qwen2.5-0.5B-Instruct via
    mlx-lm (Apple Silicon native acceleration), chosen for lowest resource
    footprint of the considered options (see backlog item for the full
    cost/latency/accuracy rationale).

    `model`/`tokenizer`/`generate_fn` are all constructor-injectable
    (defaulting to the real `mlx_lm.load`/`generate`), matching the
    dependency-injection pattern already established throughout this
    project -- keeps unit tests fast and network-independent.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        model: Any = None,
        tokenizer: Any = None,
        generate_fn: Callable[..., str] = mlx_generate,
    ) -> None:
        if model is not None and tokenizer is not None:
            self._model, self._tokenizer = model, tokenizer
        else:
            self._model, self._tokenizer = mlx_load(model_name)
        self._generate_fn = generate_fn

    def generate(self, prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
        formatted_prompt = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True
        )
        return self._generate_fn(
            self._model, self._tokenizer, prompt=formatted_prompt, max_tokens=max_tokens, verbose=False
        )
