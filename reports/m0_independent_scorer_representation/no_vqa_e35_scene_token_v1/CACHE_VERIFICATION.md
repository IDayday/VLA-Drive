# No-VQA scorer cache verification

Status: **PASS**

- Scenes: `103288` unique `103288`
- Candidate count: `64`
- Source shards/chunks: `8` / `808`
- Label workers: `4`
- Invalid scenes: `0`
- Future/evaluator fields in inference cache: `False`
- Checkpoint SHA256: `72c74a113c557df27c86a320f66d4ff2a79fc1a19e678337d5a142a520359309`
- Resolved-config SHA256: `5f70b74293883bebb80fc1feffaf3786556f909645a248374495dfadbf7cd1c3`

The source and label trees have identical relative chunk sets, token/log
order, 64-candidate geometry, and row counts. All tensors are finite, all
offline PDM rows are valid, and cached Base indices equal `argmax(base_scores)`.
