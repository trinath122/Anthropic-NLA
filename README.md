# open-nla-audits

> **Unofficial; not affiliated with Anthropic.** This is independent research that builds on
> Anthropic's paper [*Natural Language Autoencoders Produce Unsupervised Explanations of LLM
> Activations*](https://transformer-circuits.pub/2026/nla/) (Fraser-Taliente, Kantamneni, Ong
> et al., 2026) and its open-source code,
> [kitft/natural_language_autoencoders](https://github.com/kitft/natural_language_autoencoders)
> (Apache-2.0). All credit for NLAs and the released checkpoints goes to the original authors.
> See [NOTICE](NOTICE).

## Research question

Can natural language autoencoders (NLAs) reveal **hidden objectives** in open models, and can
**consistency filtering** reduce NLA confabulation enough to make their explanations
trustworthy?

The work uses the released `kitft/nla-qwen2.5-7b-L20-{av,ar}` checkpoints. It is **not** a
replication: replication ends at a baseline reconstruction check (Step 1).

| Step | What | Status |
|---|---|---|
| 0 | Scaffolding: free-LLM client, NLA inference, resumable jobs, tests | done |
| 1 | Baseline: FVE of the released Qwen2.5-7B L20 NLA (bf16 reference + fp16 agreement) | next |
| 2 | Model organisms: Qwen2.5-7B LoRA with a *harmless* hidden objective (favouring a fictional brand) + controls | planned |
| 3 | Headline: does the NLA expose the hidden **motive** where black-box chat does not? (pre-registered) | planned |
| 4 | Consistency filter: keep claims that agree across samples and nearby tokens; measure precision gain | planned |
| 5 | Optional: AR-derived steering vectors | planned |

Negative results are reported as results. Every number is produced by a script in this repo
from saved artifacts.

## Constraints

* **$0.** Free LLM APIs only (Cerebras, Groq, Gemini free tier, OpenRouter `:free`). No paid
  APIs, no paid cloud.
* **Hardware:** a 6 GB laptop GPU, Kaggle 2×T4, and Colab Pro (already-paid units).
* **Precision:** activations come only from unquantized weights (fp16 on T4, bf16 on
  L4/A100). Each file records its dtype and GPU.
* **No hardcoded NLA constants.** Templates, injection token IDs and scales are read from each
  checkpoint's `nla_meta.yaml` and checked against the live tokenizer.
* **Licences:** training data comes only from Apache/MIT open-weight models. Judges come from
  a different model family than the generators, and every call is logged.

## Setup (WSL/Linux)

```bash
cp .env.example .env      # fill in free API keys (never commit .env)
make install              # uv + Python 3.11, locked deps
make test                 # unit tests (tiny random Qwen + real tokenizer)
make check-providers      # which free LLM providers/models answer with your keys
make hooks                # pre-push secret + attribution scan
```

## Layout

```
src/nla_audit/
  llm_client.py        one OpenAI-compatible client: fallback, backoff, SQLite cache, licence policy, ledger
  nla/sidecar.py       nla_meta.yaml contract + tokenizer drift checks
  nla/extract.py       layer-K activations (= HF hidden_states[K+1]), official position sampling
  nla/av.py            activation verbalizer (batched inputs_embeds generation)
  nla/ar.py            activation reconstructor (truncated LM + value head)
  nla/metrics.py       mse_nrm, FVE (official definition), bootstrap / hierarchical CIs
  runtime/env.py       kaggle / colab / local detection, secrets, dtype, provenance
  runtime/resume.py    shard checkpoints synced to a private HF repo every <=30 min
configs/               YAML configs (llm.yaml: providers, roles, policy)
scripts/               entry points (secret_scan.py, check_providers.py, ...)
tests/                 pytest suite
```

## Citation of the original work

```bibtex
@article{frasertaliente2026nla,
  author  = {Fraser-Taliente, Kit and Kantamneni, Subhash and Ong, Euan and others},
  title   = {Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations},
  journal = {Transformer Circuits Thread},
  year    = {2026},
  url     = {https://transformer-circuits.pub/2026/nla/index.html}
}
```

## License

Apache-2.0 ([LICENSE](LICENSE)). Base models and released checkpoints keep their own licences.
