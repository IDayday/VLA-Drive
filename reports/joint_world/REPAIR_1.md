# Repair round1: trajectory coordinates and noise scale

Evidence before changes: absolute-XY randommask1000-step holdout static ADE4.428m versus stationary1.377m; train static2.971m versus1.259m. Generated agent ADE4.904m holdout, low current detection coverage11.46%. Current geometry stays fixed and detector coverage cannot improve in this isolated graph repair.

Hypothesis: predicting noisy absolute world positions spends capacity reconstructing current position and imposes20m agent noise on predominantly small displacements. This produces unnecessary drift with the compact graph and1000-step budget. This is an optimization hypothesis, not a causal explanation established by the observed metrics.

Change: predicted-centre residual coordinates, agent scale5m, ego scale20m. Current geometry encoders remain divided by20m. Predicted centres are detached; no GT centre enters graph input or decode. Train and inference use exactly the same invertible representation. Parameter count/init unchanged. No threshold, object filter, target set, loss weight or matching change.

Fresh matched randommask/allmask64 runs, each1000updates,batch8,seed42. Same cached current inputs, sample order, masks and normal random numbers. Full reports at0/500/1000; all-masked train and complete-log holdout evaluation. No further tuning from holdout, no navtest. Charge2000 additional updates. All old results retained. This consumes ONE of the TWO repair rounds; pure implementation bug fixes are separately logged.
