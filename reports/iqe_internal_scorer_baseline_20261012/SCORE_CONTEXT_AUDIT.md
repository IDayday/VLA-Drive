# Actual current-input W/A audit

One frozen original VLM; no auxiliary prediction head, external encoder, extra tokens or future labels. Rebatch differences below are numerical observations, not new tolerance gates.

```json
{
  "status": "PASS",
  "evidence": [
    {
      "scene_ids": [
        "9dffd738f7955b17",
        "a94a26bd60635372",
        "6ef131aed5af5f12",
        "e436d91a140e5cc2"
      ],
      "original_vs_K1_max_abs": 0.0,
      "label_shuffle_W_max_abs": 0.0,
      "cache_rebatch_W_max_abs": 0.0,
      "cache_rebatch_A_max_abs": 0.0,
      "cache_rebatch_physical_max_abs": 0.0
    },
    {
      "scene_ids": [
        "1a435b3a12d05632",
        "d627fc50ddec5593",
        "096e941ba39d55d6",
        "07ebca567afd53ee"
      ],
      "original_vs_K1_max_abs": 0.0,
      "label_shuffle_W_max_abs": 0.0,
      "cache_rebatch_W_max_abs": 0.0,
      "cache_rebatch_A_max_abs": 0.0,
      "cache_rebatch_physical_max_abs": 0.0
    },
    {
      "scene_ids": [
        "1a981184e312539b",
        "dce72a2b17b85b3b",
        "e8b923bcc6e65fdb",
        "9fde07748fe4566c"
      ],
      "original_vs_K1_max_abs": 0.0,
      "label_shuffle_W_max_abs": 0.0,
      "cache_rebatch_W_max_abs": 0.0,
      "cache_rebatch_A_max_abs": 0.0,
      "cache_rebatch_physical_max_abs": 0.0
    },
    {
      "scene_ids": [
        "ff9dcd98172a51fd",
        "8205b3f89b1f5bc2",
        "26f0745c5a2c5f19",
        "99df7287e97e5aed"
      ],
      "original_vs_K1_max_abs": 0.0,
      "label_shuffle_W_max_abs": 0.0,
      "cache_rebatch_W_max_abs": 0.0,
      "cache_rebatch_A_max_abs": 0.0,
      "cache_rebatch_physical_max_abs": 0.0
    },
    {
      "scene_ids": [
        "27805397bc4d59e7",
        "00f0292c25055516",
        "446ce5a0fc29506a",
        "c6d9754fe5f050eb"
      ],
      "original_vs_K1_max_abs": 0.0,
      "label_shuffle_W_max_abs": 0.0,
      "cache_rebatch_W_max_abs": 0.0,
      "cache_rebatch_A_max_abs": 0.0,
      "cache_rebatch_physical_max_abs": 0.0
    },
    {
      "scene_ids": [
        "512931de020d5c2c",
        "386a48cdf9de52ec",
        "c41bcbb948115d17",
        "f25badb53644586c"
      ],
      "original_vs_K1_max_abs": 0.0,
      "label_shuffle_W_max_abs": 0.0,
      "cache_rebatch_W_max_abs": 0.0,
      "cache_rebatch_A_max_abs": 0.0,
      "cache_rebatch_physical_max_abs": 0.0
    },
    {
      "scene_ids": [
        "18c6b733242b5b84",
        "5ff6df1b7ba651d6",
        "cc49daefcb0c54eb",
        "95de7ec3005254e3"
      ],
      "original_vs_K1_max_abs": 0.0,
      "label_shuffle_W_max_abs": 0.0,
      "cache_rebatch_W_max_abs": 0.0,
      "cache_rebatch_A_max_abs": 0.0,
      "cache_rebatch_physical_max_abs": 0.0
    },
    {
      "scene_ids": [
        "83bae923a71e5425",
        "26f4068b53255d7f",
        "c5dc5b64b37d5427",
        "f10544e952f95491"
      ],
      "original_vs_K1_max_abs": 0.0,
      "label_shuffle_W_max_abs": 0.0,
      "cache_rebatch_W_max_abs": 0.0,
      "cache_rebatch_A_max_abs": 0.0,
      "cache_rebatch_physical_max_abs": 0.0
    }
  ],
  "frozen_hashes_before": {
    "parameters": "730b397887b6a0ada53e07f88829c8611e19c7275e495d17b0156c032c45ddbc",
    "buffers": "263925f1a53ea02eb16370797b6849fe9981db7a81b1b63617612e30aa9afd50"
  },
  "frozen_hashes_after": {
    "parameters": "730b397887b6a0ada53e07f88829c8611e19c7275e495d17b0156c032c45ddbc",
    "buffers": "263925f1a53ea02eb16370797b6849fe9981db7a81b1b63617612e30aa9afd50"
  },
  "W_shape": [
    4,
    144,
    2048
  ],
  "A_shape": [
    4,
    8,
    2048
  ],
  "W_dtype": "torch.float32",
  "A_dtype": "torch.float32",
  "vlm_calls_per_image_to_selected_trajectory": 1,
  "score_context_projections_per_scene": 1,
  "auxiliary_prediction_heads_executed": 0,
  "GT_in_score_context": false,
  "poisoned_labels_ignored": true,
  "base_shared_parameter_count": 2136787968,
  "base_head_parameter_count": 10964384,
  "new_expert_parameter_count": 10964384,
  "new_scorer_parameter_count": 3301639
}
```
