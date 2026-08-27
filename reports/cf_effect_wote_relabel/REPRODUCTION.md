# Reproduction

The run used the fixed first 200 tokens in `relabel_headroom_tokens.txt`, the
released 256×8×3 base-anchor bank, proposal sampling 40×0.1 s, and the pinned
evaluator sources in `EVALUATOR_CONTRACT.json`.

Evaluator contract SHA256: `fa5e264bbd06fac5ef5457cd4ea94484efb1361ef0d24e8ec2aafdb5534fe9f8`.

Run order stopped during G0-R run1 on scene
`0fcede1cbfb15faa` after all 256 candidates were evaluated. Run2,
the published-label audit, G1-R, feature caching, effect construction, and G2-O
were not run because the five-factor score reconstruction error exceeded
`1e-6`.
