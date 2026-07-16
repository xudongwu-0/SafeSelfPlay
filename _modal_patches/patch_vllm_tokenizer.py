"""Patch tokenizer classes to add all_special_tokens_extended for vLLM 0.10.2.

Some transformers tokenizers in the image do not expose the
all_special_tokens_extended property that vLLM reads while caching tokenizers.
Patch the installed source files so subprocesses and Ray workers inherit the
compatibility shim.
"""
import inspect
import transformers


def patch_class(cls):
    fn = inspect.getfile(cls)
    marker = f"# ROLL Modal vLLM compat: {cls.__name__}.all_special_tokens_extended"
    content = open(fn).read()
    if marker in content:
        print(f"{cls.__name__} patch already present in {fn}")
        return

    patch = f"""
{marker}
def _roll_all_special_tokens_extended(self):
    tokens = []
    try:
        tokens.extend(list(self.added_tokens_decoder.values()))
    except Exception:
        pass
    try:
        for token in self.all_special_tokens:
            if token not in tokens:
                tokens.append(token)
    except Exception:
        pass
    return tokens
{cls.__name__}.all_special_tokens_extended = property(_roll_all_special_tokens_extended)
"""
    with open(fn, "a") as f:
        f.write(patch)
    print(f"Patched {cls.__name__} in {fn}")


for tokenizer_cls_name in ["PreTrainedTokenizerBase", "PreTrainedTokenizer", "PreTrainedTokenizerFast"]:
    tokenizer_cls = getattr(transformers, tokenizer_cls_name, None)
    if tokenizer_cls is not None:
        patch_class(tokenizer_cls)
