"""NLA inference: sidecar contract, activation extraction, AV (verbalizer), AR (reconstructor).

Behaviour mirrors the official client `nla_inference.py` from
https://github.com/kitft/natural_language_autoencoders (Apache-2.0), re-implemented for
batched HF `transformers` inference on T4/L4/A100 instead of SGLang. See NOTICE.
"""
