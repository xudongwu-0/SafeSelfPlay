"""Pure-Python contracts for synchronizing PEFT LoRA tensors to vLLM.

The training process and the rollout process use different model wrappers.  A
checkpoint changing on disk therefore does not prove that rollout loaded the
same adapter.  These helpers keep the tensor-name boundary explicit and reject
partial or malformed A/B updates before vLLM can silently fall back to the base
module.
"""

from __future__ import annotations

import re
from collections.abc import Iterable


_PEFT_PREFIX = "base_model.model."
_LORA_SUFFIXES = {
    ".lora_A.weight": "A",
    ".lora_B.weight": "B",
    ".lora_embedding_A": "A",
    ".lora_embedding_B": "B",
}


def normalize_peft_lora_name(name: str) -> str:
    """Return a vLLM-compatible, canonical PEFT LoRA tensor name.

    PEFT ``named_parameters`` includes the adapter name (normally
    ``.default.``), while a saved PEFT state dict does not.  vLLM 0.8.x also
    assumes the two-component ``base_model.model.`` prefix when parsing tensor
    names.  Keeping that native prefix works on both vLLM 0.8.x and 0.10.x.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("LoRA tensor name must be a non-empty string")

    while name.startswith("module."):
        name = name[len("module.") :]

    parts = name.split(".")
    if len(parts) >= 3 and parts[-3] in {"lora_A", "lora_B"} and parts[-1] == "weight":
        # Drop the PEFT adapter name: lora_A.default.weight -> lora_A.weight.
        del parts[-2]
        name = ".".join(parts)
    elif len(parts) >= 2 and parts[-2] in {
        "lora_embedding_A",
        "lora_embedding_B",
    }:
        # Embedding LoRA parameters end in the adapter name rather than
        # ``.weight``: lora_embedding_A.default -> lora_embedding_A.
        del parts[-1]
        name = ".".join(parts)

    if not name.startswith(_PEFT_PREFIX):
        # This is the form produced by the old synchronization code after it
        # stripped the PEFT prefix.  Restore the prefix instead of passing a
        # version-dependent name to vLLM.
        if name.startswith("model."):
            name = _PEFT_PREFIX + name
        else:
            raise ValueError(f"LoRA tensor name must start with 'base_model.model.' or 'model.': {name!r}")

    return name


def _split_lora_name(name: str) -> tuple[str, str]:
    for suffix, side in _LORA_SUFFIXES.items():
        if name.endswith(suffix):
            module_name = name[len(_PEFT_PREFIX) : -len(suffix)]
            if not module_name:
                break
            return module_name, side
    raise ValueError(f"Unsupported LoRA tensor name: {name!r}")


def _shape_tuple(shape: object) -> tuple[int, ...]:
    try:
        result = tuple(int(dim) for dim in shape)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid LoRA tensor shape: {shape!r}") from exc
    if not result or any(dim <= 0 for dim in result):
        raise ValueError(f"LoRA tensor shape must be positive: {result!r}")
    return result


def validate_lora_tensor_specs(
    specs: Iterable[tuple[str, object]],
    target_modules: Iterable[str] | str | None = None,
) -> list[tuple[str, tuple[int, ...]]]:
    """Normalize and validate one complete linear-LoRA synchronization.

    Every target module must have exactly one two-dimensional A tensor and one
    B tensor, and their rank dimensions must agree.  Normalization collisions
    are rejected so one adapter name cannot overwrite another in the vLLM
    tensor dictionary.
    """
    normalized_specs: list[tuple[str, tuple[int, ...]]] = []
    seen_names: set[str] = set()
    module_sides: dict[str, dict[str, tuple[int, ...]]] = {}

    for raw_name, raw_shape in specs:
        name = normalize_peft_lora_name(raw_name)
        if name in seen_names:
            raise ValueError(f"Duplicate normalized LoRA tensor name: {name}")
        seen_names.add(name)

        shape = _shape_tuple(raw_shape)
        if len(shape) != 2:
            raise ValueError(f"Linear LoRA tensor must be rank 2: {name} {shape}")
        module_name, side = _split_lora_name(name)
        sides = module_sides.setdefault(module_name, {})
        if side in sides:
            raise ValueError(f"Duplicate LoRA {side} tensor for {module_name}")
        sides[side] = shape
        normalized_specs.append((name, shape))

    if not normalized_specs:
        raise ValueError("No LoRA A/B tensors were selected for synchronization")

    for module_name, sides in module_sides.items():
        missing = {"A", "B"} - set(sides)
        if missing:
            raise ValueError(f"Incomplete LoRA pair for {module_name}: missing {sorted(missing)}")
        a_shape, b_shape = sides["A"], sides["B"]
        if a_shape[0] != b_shape[1]:
            raise ValueError(f"LoRA rank mismatch for {module_name}: A{a_shape} vs B{b_shape}")

    if isinstance(target_modules, str):
        if target_modules != "all-linear":
            try:
                target_pattern = re.compile(target_modules)
            except re.error as exc:
                raise ValueError(f"Invalid LoRA target-module regex: {target_modules!r}") from exc
            unmatched = {module_name for module_name in module_sides if target_pattern.fullmatch(module_name) is None}
            if unmatched:
                raise ValueError(f"LoRA modules do not match target-module regex: {sorted(unmatched)[:8]!r}")
    elif target_modules is not None:
        expected = {str(module) for module in target_modules}
        observed = {module_name.rsplit(".", 1)[-1] for module_name in module_sides}
        unexpected = observed - expected
        missing_targets = expected - observed
        if unexpected or missing_targets:
            raise ValueError(
                f"LoRA target-module mismatch: missing={sorted(missing_targets)}, unexpected={sorted(unexpected)}"
            )

    return normalized_specs


def validate_lora_runtime_module_mapping(
    parsed_modules: Iterable[str],
    runtime_modules: Iterable[str],
    packed_module_sources: Iterable[str] = (),
) -> set[str]:
    """Require every parsed adapter module to address a real vLLM layer.

    vLLM can successfully register an adapter even when none of its parsed
    names match the rollout model; activation then resets every LoRA slot and
    generation is indistinguishable from the base model. Packed layers (for
    example Llama's ``qkv_proj``) expose their original q/k/v source names
    separately, so those source names are valid too.
    """
    parsed = {str(module) for module in parsed_modules}
    if not parsed:
        raise ValueError("vLLM parsed no LoRA modules from synchronized tensors")

    available = {str(module) for module in runtime_modules}
    available.update(str(module) for module in packed_module_sources)
    if not available:
        raise ValueError("vLLM exposes no runtime LoRA modules")

    unmatched = parsed - available
    if unmatched:
        raise ValueError(f"Synchronized LoRA modules do not match the vLLM model: {sorted(unmatched)[:8]!r}")
    return parsed
