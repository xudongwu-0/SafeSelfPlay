"""
Patch PreTrainedTokenizerFast to add all_special_tokens_extended.

vLLM 0.10.2 accesses tokenizer.all_special_tokens_extended (originally on the
slow PreTrainedTokenizer). transformers>=4.50 returns the fast tokenizer for
Qwen2, which never had this property. This patch appends it to
tokenization_utils_fast.py so all subprocesses (Ray workers) see the fix.
"""
import inspect
import transformers

fn = inspect.getfile(transformers.PreTrainedTokenizerFast)
content = open(fn).read()

if "all_special_tokens_extended" not in content:
    patch = (
        "\n# vLLM compat: add missing all_special_tokens_extended to fast tokenizer\n"
        "PreTrainedTokenizerFast.all_special_tokens_extended = "
        "property(lambda self: list(self.added_tokens_decoder.values()))\n"
    )
    with open(fn, "a") as f:
        f.write(patch)
    print(f"Patched {fn}")
else:
    print("all_special_tokens_extended already present — no patch needed")
