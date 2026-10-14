# Enforce rich-bank scoring and deduplication budget before labels

Before the first rich bank is built, the own-intermediate path now separates its unchanged32-scene candidate generation from official labeling. It checks scene identity, exact serialized FP32 trajectory hash and remaining per-scene capacity before submitting an official request. No new trajectory type, source ordering, weights, score context, scoring seed or optimizer recipe is introduced.

Previously, every1000/3000 raw endpoint was scored before duplicates were discarded. Repeated physical actions could therefore consume official scoring time without adding supervision, and a later fill of the remaining slots could make actual calls exceed the nominal twelve-row budget. The new path does not score those duplicates or over-capacity actions. Generated checkpoint/trajectory identities remain the same; completed official scores still reuse their immutable receipts.

Deployment retains explicit candidate columns. Before committing B_rich, exact duplicate deployment actions are retained only once for pointwise supervision, preserving their first stable identity and all unique deployment actions. Conflicting true labels for an identical physical action are rejected. Original deployment columns still occupy the request-budget slots until this final deduplication, so removing duplicate supervision cannot silently admit more official-label requests. Shorter candidate lists use their actual masks rather than artificial padding actions.

Actual tests:

```text
/root/miniconda3/envs/ddp/bin/python -m pytest -q tests/iqe/specialist_study/test_rich_label_budget.py tests/iqe/specialist_study/test_data_and_banks.py tests/iqe/specialist_study/test_bank_routing.py
16 passed, 1 warning in0.47s
```

Tests verify duplicate/over-capacity scenes never invoke the scoring backend, scene misalignment fails before scoring, and conflicting duplicate component labels cannot be hidden by deduplication. Real avoided-call counts are WAITING_INPUTS until B_rich construction and will be recorded in its provenance. This is not reported as a measured speedup yet. Active expert trainers and their source hashes were not changed.
