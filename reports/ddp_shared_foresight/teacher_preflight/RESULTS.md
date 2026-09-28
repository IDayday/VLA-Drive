# GT MAE initial learning evidence

Actual512updates/16384scene presentations on64TRAIN scenes; no old weights. Full512dim6-layer teacher, valid XY SmoothL1, independent target-mask RNG.42ego queries have peer future; endpoint full-peer ADE0.873218m versus1.347708m with peers hidden but current states retained.245neighbor queries:3.531544m versus4.684739m. These are train-fit and potentially out-of-distribution removal diagnostics, not causal or held-out planning evidence. All309valid actor queries retained;0failures.22ego-only scenes are still included in full summaries.

Teacher same-GPU deterministic continuous4 versus2+2 matched full model tensors, full optimizer, torch/CUDA/mask RNG and data progress exactly. This consumed8real updates in independent runs; no claim about full student/distributed resume.

Full independent teacher has now started from fresh random initialization on101592training scenes, with1696fixed log-disjoint development scenes.30epochs, batch256,397updates/epoch including216scene tail,11910plannedupdates. Exact source d3c05d136949ecc14b1a8312fa12bf966ae03643; full training results pending. Data prep includes103288total train+dev scenes,0failures; no scene deletion for ego-only or missing future labels.

Student initial GPU launch exposed a typed-configuration issue: environment auxiliary weights were strings. Fixed conversion before config validation; that failed launch made0optimizer updates and its ledger is retained. Real VLM checks are rerunning. FLUX access remains unresolved; no visual targets or visual-learning claims exist yet.
