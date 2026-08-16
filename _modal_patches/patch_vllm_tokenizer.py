"""Provide all_special_tokens_extended only when Transformers lacks it.

vLLM 0.10.2 reads this property while caching tokenizers. Preserve the native
Transformers descriptor whenever it exists; an older compatibility shim
recursively depended on the public plain-token property it implemented. The
fallback is written into installed source so
subprocesses and Ray workers inherit it on older Transformers releases.
"""
import inspect
import transformers


def patch_class(cls):
    fn = inspect.getfile(cls)
    marker = (
        "# ROLL Modal vLLM compat v2: "
        f"{cls.__name__}.all_special_tokens_extended"
    )
    content = open(fn).read()
    if marker in content:
        print(f"{cls.__name__} patch already present in {fn}")
        return

    descriptor = inspect.getattr_static(
        cls, "all_special_tokens_extended", None
    )
    descriptor_getter = getattr(descriptor, "fget", None)
    if descriptor is not None and getattr(
        descriptor_getter, "__name__", ""
    ) != "_roll_all_special_tokens_extended":
        print(
            f"{cls.__name__} already provides "
            f"all_special_tokens_extended in {fn}"
        )
        return

    patch = f"""
{marker}
def _roll_all_special_tokens_extended(self):
    all_tokens = []
    seen = set()
    special_tokens_map = getattr(self, "special_tokens_map_extended", {{}})
    for value in special_tokens_map.values():
        tokens_to_add = value if isinstance(value, (list, tuple)) else [value]
        for token in tokens_to_add:
            token_text = str(token)
            if token_text not in seen:
                seen.add(token_text)
                all_tokens.append(token)
    return all_tokens
{cls.__name__}.all_special_tokens_extended = property(_roll_all_special_tokens_extended)
"""
    with open(fn, "a") as f:
        f.write(patch)
    print(f"Patched {cls.__name__} in {fn}")


tokenizer_classes = []
for tokenizer_cls_name in [
    "PreTrainedTokenizerBase",
    "PreTrainedTokenizer",
    "PreTrainedTokenizerFast",
    "TokenizersBackend",
]:
    tokenizer_cls = getattr(transformers, tokenizer_cls_name, None)
    if tokenizer_cls is not None and tokenizer_cls not in tokenizer_classes:
        tokenizer_classes.append(tokenizer_cls)

try:
    from transformers.tokenization_utils_tokenizers import TokenizersBackend
except (ImportError, AttributeError):
    TokenizersBackend = None
if TokenizersBackend is not None and TokenizersBackend not in tokenizer_classes:
    tokenizer_classes.append(TokenizersBackend)

for tokenizer_cls in tokenizer_classes:
    patch_class(tokenizer_cls)
