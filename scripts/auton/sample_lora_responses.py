"""Sample responses from base model + LoRA checkpoint."""
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "/zfsauton/scratch/wentsec/hf_cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
LORA_PATH = sys.argv[1] if len(sys.argv) > 1 else None

SYSTEM_PROMPT = (
    "As an expert poker strategist, analyze the following Kuhn Poker hand.\n"
    "Kuhn Poker: 3-card deck (Jack < Queen < King). Each player antes 1 chip.\n"
    "Only two actions exist: Pass or Bet (+1 chip). There is NO raise or fold action.\n"
    "Pass = check/fold depending on context. Bet = bet/call depending on context.\n\n"
    "Game flow:\n"
    "  P1: Pass or Bet\n"
    "  P2: Pass or Bet\n"
    "  If P1 passed and P2 bet → P1 gets one more chance: Pass(fold) or Bet(call)\n"
    "Showdown: higher card wins the pot.\n\n"
    "For each decision: note the pot size and board state, calculate pot odds, then choose.\n"
    "Reply ONLY in this format (nothing after the action tag):\n"
    "<think>your reasoning</think><action>Pass</action>\n"
    "or\n"
    "<think>your reasoning</think><action>Bet</action>\n"
    "Your available actions are:\n"
    "Pass, Bet"
)

CASES = [
    ("P0 Jack open",
     "Your card: Jack. You are Player 1 (first to act).\nPot size: 2 chips (both antes). Board state: no action yet.\nChoose: Pass or Bet."),
    ("P0 Queen open",
     "Your card: Queen. You are Player 1 (first to act).\nPot size: 2 chips (both antes). Board state: no action yet.\nChoose: Pass or Bet."),
    ("P0 King open",
     "Your card: King. You are Player 1 (first to act).\nPot size: 2 chips (both antes). Board state: no action yet.\nChoose: Pass or Bet."),
    ("P1 Jack vs Bet",
     "Your card: Jack. You are Player 2.\nPot size: 3 chips. Board state: Player 1 chose Bet.\nPot odds: 3:1 (call 1 to win 3).\nChoose: Pass or Bet."),
    ("P1 Queen vs Bet",
     "Your card: Queen. You are Player 2.\nPot size: 3 chips. Board state: Player 1 chose Bet.\nPot odds: 3:1 (call 1 to win 3).\nChoose: Pass or Bet."),
    ("P1 King vs Bet",
     "Your card: King. You are Player 2.\nPot size: 3 chips. Board state: Player 1 chose Bet.\nPot odds: 3:1 (call 1 to win 3).\nChoose: Pass or Bet."),
]

print(f"Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

if LORA_PATH:
    import json, pathlib
    cfg_path = pathlib.Path(LORA_PATH) / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text())
    known_keys = {
        "peft_type", "task_type", "base_model_name_or_path", "inference_mode",
        "r", "lora_alpha", "lora_dropout", "bias", "fan_in_fan_out",
        "target_modules", "modules_to_save", "init_lora_weights",
        "layers_to_transform", "layers_pattern", "rank_pattern", "alpha_pattern",
        "use_rslora", "use_dora", "revision",
    }
    cfg = {k: v for k, v in cfg.items() if k in known_keys}
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"Loading LoRA from {LORA_PATH}...")
    model = PeftModel.from_pretrained(model, LORA_PATH)
    model = model.merge_and_unload()
    print("LoRA merged.")
else:
    print("No LoRA — base model only.")

model.eval()

for label, obs in CASES:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": obs}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False, temperature=None, top_p=None)
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\n=== {label} ===\n{response}")

print("\nDone.")
