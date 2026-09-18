# RL budget comparison, verified 2026-09-18

Epoch must identify its unit: a complete dataset traversal, an outer sampling
iteration, and an inner pass over one saved behavior batch are different counts.
No budget, reward or parameter-set change is made on the basis of this comparison.

| Work / source | Supervised phase | RL budget actually stated |
| --- | --- | --- |
| SimWAM paper v1 | 100 epochs, NAVSIM 103,288 scenes | Figure 3 / section 5.4 reports a peak at 15,000 steps and deterioration beyond it; this is not a statement that RL uses 100 dataset epochs. Hard subset, action-expert LoRA. |
| SimWAM released task config | Separate from the paper's experiment | max_steps=2000 optimizer updates; num_inner_epochs=4. num_epochs=1000000 is explicitly ignored when max_steps is set. |
| ReCogDrive paper v2 / ICLR paper | VLM 3 epochs, frozen-VLM diffusion planner 200 epochs | Third stage RL 10 epochs, batch128. Planner-only optimization differs from our trainable language backbone. |
| AutoVLA, NeurIPS2025 paper | 5 epochs | RL fine-tuning 6000 steps, one policy update per batch, LoRA. |
| WAM-Flow paper v1 | nuPlan trajectory SFT 2 epochs, after other pretraining stages | GRPO 0.5 epoch on 103k NAVSIM scenes. Discrete flow / different model. |
| This paired experiment, each of F and U | Existing source step100000 / step120000 respectively | Fixed2000 actual optimizer updates, global16 scenes, inner_epochs2, G8; unchanged original SFT reference. |

SimWAM paper and public code are not an exact matching experiment recipe: the
paper states action LoRA rank32/alpha16 whereas the released task sets rank16/
alpha32, in addition to the different RL budget. We do not assert that the public
2000-step preset reproduces the paper's 15k-step peak.

Our fresh-rollout accounting per variant:

- 2000 / 2 = 1000 new behavior batches.
- 1000 * 16 = 16000 scene draws, versus101592 locked RL training scenes:
  0.15749271596 dataset traversal equivalents.
- 16000 * 8 = 128000 freshly sampled candidate trajectories.
- Reusing each batch twice produces32000 scene-update exposures, or0.31498543192
  traversal equivalents; this does not increase unique scene coverage.
- K10 counts transitions within a trajectory, not new dataset examples.
- `inner_epochs=2` is rollout reuse, not two complete passes over navtrain.
- At this recipe a complete fresh-scene traversal would require approximately
  12700 optimizer updates. The current2000-update budget is kept unchanged.

These exposure counts do not imply convergence or expected reward improvement.
Different trainable parameters, batch semantics, subsets and reward protocols
prevent comparing paper epoch numbers as equal compute or equal data budgets.

Primary sources:

- [SimWAM paper, implementation and RL dynamics](https://arxiv.org/html/2608.07468v1)
- [SimWAM released RL preset, SHA68b426c162827cb7701396895dbb3572d29f3420](https://github.com/H-EmbodVis/SimWAM/blob/68b426c162827cb7701396895dbb3572d29f3420/configs/task/navsim_grpo_action_pdm_384x672_flowgrpo_lora.yaml)
- [ReCogDrive paper, appendix training configuration](https://arxiv.org/html/2506.08052v2)
- [AutoVLA NeurIPS2025 paper, implementation details](https://proceedings.neurips.cc/paper_files/paper/2025/file/2843fccca5bedd369a4764848b9bd546-Paper-Conference.pdf)
- [WAM-Flow paper section3.4](https://arxiv.org/html/2512.06112v1)

The locked upstream image Flow-GRPO code also has outer sampling epochs and
separate inner epochs. Large `num_epochs` defaults are not evidence that a
published run traversed a fixed dataset that many times. The local reference
remains SHA879042cf5707f8b90daa98d147d7deac2317c5da; it supplies the sampler and
surrogate reference, not a verified2000-update convergence budget for DDP.
