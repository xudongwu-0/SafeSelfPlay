"""Test that LoRA reset works correctly:
1. before reset  → trained output (lora_B non-zero)
2. after reset   → base model output (lora_B = 0, delta = 0)
3. second reset  → same output as first reset (both are base model)
"""
import math
import torch
import torch.nn.init as nn_init
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "What is 2+2?"
MAX_NEW = 30
DEVICE = "cuda"

def load_model():
    from transformers import AutoModelForCausalLM
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEVICE)
    cfg = LoraConfig(
        r=32, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0, bias="none",
    )
    return get_peft_model(base, cfg)

def reset_lora(model):
    for name, param in model.named_parameters():
        if "lora_A" in name:
            nn_init.kaiming_uniform_(param.data, a=math.sqrt(5))
        elif "lora_B" in name:
            nn_init.zeros_(param.data)

def make_lora_nonzero(model):
    """Simulate a trained state: give lora_B small random values."""
    for name, param in model.named_parameters():
        if "lora_B" in name:
            nn_init.normal_(param.data, std=0.01)

def generate(model, tokenizer):
    model.eval()
    ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False,
                             temperature=1.0, repetition_penalty=1.0)
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

def lora_delta_norm(model):
    """Sum of |lora_B @ lora_A| norms — should be 0 after reset."""
    total = 0.0
    layers = {}
    for name, param in model.named_parameters():
        layer = name.rsplit(".lora_", 1)[0]
        layers.setdefault(layer, {})[name] = param
    for layer, params in layers.items():
        A = next((v for k, v in params.items() if "lora_A" in k), None)
        B = next((v for k, v in params.items() if "lora_B" in k), None)
        if A is not None and B is not None:
            total += (B @ A).norm().item()
    return total


if __name__ == "__main__":
    print(f"Loading model: {MODEL}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = load_model()

    # Simulate trained LoRA (non-zero B)
    make_lora_nonzero(model)
    delta_before = lora_delta_norm(model)
    out_before = generate(model, tok)
    print(f"\n[BEFORE RESET]  delta_norm={delta_before:.4f}")
    print(f"  output: {out_before!r}")

    # First reset
    reset_lora(model)
    delta_after1 = lora_delta_norm(model)
    out_after1 = generate(model, tok)
    print(f"\n[AFTER RESET 1] delta_norm={delta_after1:.6f}")
    print(f"  output: {out_after1!r}")

    # Second reset
    reset_lora(model)
    delta_after2 = lora_delta_norm(model)
    out_after2 = generate(model, tok)
    print(f"\n[AFTER RESET 2] delta_norm={delta_after2:.6f}")
    print(f"  output: {out_after2!r}")

    # Assertions
    print("\n--- RESULTS ---")
    assert delta_before > 1e-3, f"FAIL: expected non-zero delta before reset, got {delta_before}"
    print(f"PASS: lora delta non-zero before reset ({delta_before:.4f})")

    assert delta_after1 < 1e-9, f"FAIL: expected zero delta after reset 1, got {delta_after1}"
    print(f"PASS: lora delta = 0 after reset 1 ({delta_after1})")

    assert delta_after2 < 1e-9, f"FAIL: expected zero delta after reset 2, got {delta_after2}"
    print(f"PASS: lora delta = 0 after reset 2 ({delta_after2})")

    assert out_before != out_after1, "FAIL: output should change after reset"
    print(f"PASS: output changed after reset")

    assert out_after1 == out_after2, f"FAIL: two resets should give same output\n  reset1: {out_after1!r}\n  reset2: {out_after2!r}"
    print(f"PASS: both resets give identical output")

    print("\nAll checks passed!")
