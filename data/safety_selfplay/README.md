# Safety Self-Play SFT Data

`attacker_rewrite_1180.jsonl` is the cleaned role-specific attacker SFT set
used by the canonical LoRA v2 training path. Each JSONL row contains a
system/user/assistant message sequence and metadata describing its label and
source class.

## Data Summary

- 1,180 rows: 715 harmful and 465 benign.
- 884 rows are based on training sources.
- 296 rows use synthetic seeds derived from holdout/test-like references; the
  original benchmark prompts are not copied into those examples.
- 14 malformed, trivial, or duplicate examples were removed.
- SHA-256:
  `11b860bee147d668ad3645a8c757bdab6b2fbcaeed8e0ac5e2acd108ce13c233`.

See `attacker_rewrite_prompts.md` for generation templates and
`attacker_rewrite_1180_report.md` for the cleaning report. Regenerate the data
with `scripts/abs_sft/generate_selected_attacker_sft_data.py`; API credentials
must be provided through environment variables and are never stored here.
