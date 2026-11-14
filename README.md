# free-model-aggregator-01 v1

One server (Python 3.13, stdlib only) that serves the OpenAI Chat Completions and
Anthropic Messages APIs over ten simulated free-tier providers, in JSON and SSE.

Design: whole-workload planner (quality-ordered, per-provider soft cap, constrained
requests first) + drop-on-first-refusal failover + full three-wire translation
(OpenAI / Anthropic / Gemini) with exact echo, usage and stop-reason fidelity.

Run: `python3 main.py` (reads MAKEFUN_PROMPT_FILE, MAKEFUN_ENV_URL, MAKEFUN_TRIAL_SEED).
`python3 main.py --list` prints the catalog document and exits 0.
