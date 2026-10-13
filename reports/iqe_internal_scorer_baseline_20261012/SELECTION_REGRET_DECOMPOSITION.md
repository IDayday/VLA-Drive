# Selection evidence

Same locked Base, same fixed development scenes. Positive gains and negative costs are totals over all requested scenes; rates divide by n, switch precision by switched scenes. Oracle capture is unbounded and NA when its denominator vanishes.

```json
{
  "stage_val": {
    "B0_Base": {
      "n": 1024,
      "base": 0.9163406651862211,
      "expert": 0.912002147346718,
      "oracle": 0.9202851330307689,
      "selected": 0.9163406651862211,
      "oracle_gain": 0.003944467844547725,
      "selected_gain": 0.0,
      "selection_regret": 0.003944467844547725,
      "oracle_capture": 0.0,
      "switch_count": 0,
      "switch_rate": 0.0,
      "base_fallback_rate": 1.0,
      "positive_switch_gain_total": 0.0,
      "negative_switch_cost_total": 0.0,
      "positive_switches": 0,
      "negative_switches": 0,
      "improvement_rate": 0.0,
      "degradation_rate": 0.0,
      "tie_rate": 1.0,
      "positive_switch_gain_per_scene": 0.0,
      "negative_switch_cost_per_scene": 0.0,
      "switch_precision": null,
      "high_to_zero": 0,
      "rate_denominator": 1024,
      "high_to_zero_rate_all_scenes": 0.0,
      "base_high_count": 948,
      "high_to_zero_rate_base_high": 0.0,
      "safety": {
        "NC": {
          "selected_violations": 16,
          "base_violations": 16,
          "new_violations": 0,
          "denominator": 1024
        },
        "DAC": {
          "selected_violations": 11,
          "base_violations": 11,
          "new_violations": 0,
          "denominator": 1024
        }
      },
      "large_gap_errors": {
        "0.1": {
          "errors": 3,
          "eligible": 17
        },
        "0.5": {
          "errors": 1,
          "eligible": 8
        }
      },
      "selected_gain_log_bootstrap": {
        "mean": 0.0,
        "lower_95": 0.0,
        "upper_95": 0.0,
        "groups": 53,
        "repetitions": 2000,
        "seed": 11235,
        "reliable_log_count": true
      }
    },
    "B1_Expert": {
      "n": 1024,
      "base": 0.9163406651862211,
      "expert": 0.912002147346718,
      "oracle": 0.9202851330307689,
      "selected": 0.912002147346718,
      "oracle_gain": 0.003944467844547725,
      "selected_gain": -0.004338517839503161,
      "selection_regret": 0.008282985684050885,
      "oracle_capture": -1.0998994060758067,
      "switch_count": 1024,
      "switch_rate": 1.0,
      "base_fallback_rate": 0.0,
      "positive_switch_gain_total": 4.039135072816871,
      "negative_switch_cost_total": 8.481777340468106,
      "positive_switches": 532,
      "negative_switches": 26,
      "improvement_rate": 0.51953125,
      "degradation_rate": 0.025390625,
      "tie_rate": 0.455078125,
      "positive_switch_gain_per_scene": 0.003944467844547725,
      "negative_switch_cost_per_scene": 0.008282985684050885,
      "switch_precision": 0.51953125,
      "high_to_zero": 4,
      "rate_denominator": 1024,
      "high_to_zero_rate_all_scenes": 0.00390625,
      "base_high_count": 948,
      "high_to_zero_rate_base_high": 0.004219409282700422,
      "safety": {
        "NC": {
          "selected_violations": 21,
          "base_violations": 16,
          "new_violations": 5,
          "denominator": 1024
        },
        "DAC": {
          "selected_violations": 14,
          "base_violations": 11,
          "new_violations": 4,
          "denominator": 1024
        }
      },
      "large_gap_errors": {
        "0.1": {
          "errors": 14,
          "eligible": 17
        },
        "0.5": {
          "errors": 7,
          "eligible": 8
        }
      },
      "selected_gain_log_bootstrap": {
        "mean": -0.004338517839503161,
        "lower_95": -0.010783875125933524,
        "upper_95": 0.00035163426908287976,
        "groups": 53,
        "repetitions": 2000,
        "seed": 11235,
        "reliable_log_count": true
      }
    },
    "B2_Oracle_at_2": {
      "n": 1024,
      "base": 0.9163406651862211,
      "expert": 0.912002147346718,
      "oracle": 0.9202851330307689,
      "selected": 0.9202851330307689,
      "oracle_gain": 0.003944467844547725,
      "selected_gain": 0.003944467844547725,
      "selection_regret": 0.0,
      "oracle_capture": 1.0,
      "switch_count": 532,
      "switch_rate": 0.51953125,
      "base_fallback_rate": 0.48046875,
      "positive_switch_gain_total": 4.039135072816871,
      "negative_switch_cost_total": 0.0,
      "positive_switches": 532,
      "negative_switches": 0,
      "improvement_rate": 0.51953125,
      "degradation_rate": 0.0,
      "tie_rate": 0.48046875,
      "positive_switch_gain_per_scene": 0.003944467844547725,
      "negative_switch_cost_per_scene": 0.0,
      "switch_precision": 1.0,
      "high_to_zero": 0,
      "rate_denominator": 1024,
      "high_to_zero_rate_all_scenes": 0.0,
      "base_high_count": 948,
      "high_to_zero_rate_base_high": 0.0,
      "safety": {
        "NC": {
          "selected_violations": 16,
          "base_violations": 16,
          "new_violations": 0,
          "denominator": 1024
        },
        "DAC": {
          "selected_violations": 10,
          "base_violations": 11,
          "new_violations": 0,
          "denominator": 1024
        }
      },
      "large_gap_errors": {
        "0.1": {
          "errors": 0,
          "eligible": 17
        },
        "0.5": {
          "errors": 0,
          "eligible": 8
        }
      },
      "selected_gain_log_bootstrap": {
        "mean": 0.003944467844547725,
        "lower_95": 0.0021456544515358125,
        "upper_95": 0.006194329247937397,
        "groups": 53,
        "repetitions": 2000,
        "seed": 11235,
        "reliable_log_count": true
      }
    },
    "B3_ScratchScorer_argmax": {
      "n": 1024,
      "base": 0.9163406651862211,
      "expert": 0.912002147346718,
      "oracle": 0.9202851330307689,
      "selected": 0.9143612807055517,
      "oracle_gain": 0.003944467844547725,
      "selected_gain": -0.0019793844806695856,
      "selection_regret": 0.005923852325217311,
      "oracle_capture": -0.5018128068671182,
      "switch_count": 947,
      "switch_rate": 0.9248046875,
      "base_fallback_rate": 0.0751953125,
      "positive_switch_gain_total": 3.9913274860789043,
      "negative_switch_cost_total": 6.01821719428456,
      "positive_switches": 509,
      "negative_switches": 23,
      "improvement_rate": 0.4970703125,
      "degradation_rate": 0.0224609375,
      "tie_rate": 0.48046875,
      "positive_switch_gain_per_scene": 0.00389778074812393,
      "negative_switch_cost_per_scene": 0.0058771652287935156,
      "switch_precision": 0.5374868004223865,
      "high_to_zero": 2,
      "rate_denominator": 1024,
      "high_to_zero_rate_all_scenes": 0.001953125,
      "base_high_count": 948,
      "high_to_zero_rate_base_high": 0.002109704641350211,
      "safety": {
        "NC": {
          "selected_violations": 20,
          "base_violations": 16,
          "new_violations": 4,
          "denominator": 1024
        },
        "DAC": {
          "selected_violations": 12,
          "base_violations": 11,
          "new_violations": 2,
          "denominator": 1024
        }
      },
      "large_gap_errors": {
        "0.1": {
          "errors": 11,
          "eligible": 17
        },
        "0.5": {
          "errors": 4,
          "eligible": 8
        }
      },
      "selected_gain_log_bootstrap": {
        "mean": -0.0019793844806695856,
        "lower_95": -0.00770765944752423,
        "upper_95": 0.0021854277978522016,
        "groups": 53,
        "repetitions": 2000,
        "seed": 11235,
        "reliable_log_count": true
      }
    },
    "B4_ScratchScorer_margin": {
      "n": 1024,
      "base": 0.9163406651862211,
      "expert": 0.912002147346718,
      "oracle": 0.9202851330307689,
      "selected": 0.9143612807055517,
      "oracle_gain": 0.003944467844547725,
      "selected_gain": -0.0019793844806695856,
      "selection_regret": 0.005923852325217311,
      "oracle_capture": -0.5018128068671182,
      "switch_count": 947,
      "switch_rate": 0.9248046875,
      "base_fallback_rate": 0.0751953125,
      "positive_switch_gain_total": 3.9913274860789043,
      "negative_switch_cost_total": 6.01821719428456,
      "positive_switches": 509,
      "negative_switches": 23,
      "improvement_rate": 0.4970703125,
      "degradation_rate": 0.0224609375,
      "tie_rate": 0.48046875,
      "positive_switch_gain_per_scene": 0.00389778074812393,
      "negative_switch_cost_per_scene": 0.0058771652287935156,
      "switch_precision": 0.5374868004223865,
      "high_to_zero": 2,
      "rate_denominator": 1024,
      "high_to_zero_rate_all_scenes": 0.001953125,
      "base_high_count": 948,
      "high_to_zero_rate_base_high": 0.002109704641350211,
      "safety": {
        "NC": {
          "selected_violations": 20,
          "base_violations": 16,
          "new_violations": 4,
          "denominator": 1024
        },
        "DAC": {
          "selected_violations": 12,
          "base_violations": 11,
          "new_violations": 2,
          "denominator": 1024
        }
      },
      "large_gap_errors": {
        "0.1": {
          "errors": 11,
          "eligible": 17
        },
        "0.5": {
          "errors": 4,
          "eligible": 8
        }
      },
      "selected_gain_log_bootstrap": {
        "mean": -0.0019793844806695856,
        "lower_95": -0.00770765944752423,
        "upper_95": 0.0021854277978522016,
        "groups": 53,
        "repetitions": 2000,
        "seed": 11235,
        "reliable_log_count": true
      }
    }
  },
  "dev_report": {
    "B0_Base": {
      "n": 1696,
      "base": 0.9060260849778198,
      "expert": 0.9058558999990978,
      "oracle": 0.9116575580141938,
      "selected": 0.9060260849778198,
      "oracle_gain": 0.0056314730363741,
      "selected_gain": 0.0,
      "selection_regret": 0.0056314730363741,
      "oracle_capture": 0.0,
      "switch_count": 0,
      "switch_rate": 0.0,
      "base_fallback_rate": 1.0,
      "positive_switch_gain_total": 0.0,
      "negative_switch_cost_total": 0.0,
      "positive_switches": 0,
      "negative_switches": 0,
      "improvement_rate": 0.0,
      "degradation_rate": 0.0,
      "tie_rate": 1.0,
      "positive_switch_gain_per_scene": 0.0,
      "negative_switch_cost_per_scene": 0.0,
      "switch_precision": null,
      "high_to_zero": 0,
      "rate_denominator": 1696,
      "high_to_zero_rate_all_scenes": 0.0,
      "base_high_count": 1592,
      "high_to_zero_rate_base_high": 0.0,
      "safety": {
        "NC": {
          "selected_violations": 19,
          "base_violations": 19,
          "new_violations": 0,
          "denominator": 1696
        },
        "DAC": {
          "selected_violations": 39,
          "base_violations": 39,
          "new_violations": 0,
          "denominator": 1696
        }
      },
      "large_gap_errors": {
        "0.1": {
          "errors": 7,
          "eligible": 23
        },
        "0.5": {
          "errors": 6,
          "eligible": 15
        }
      },
      "selected_gain_log_bootstrap": {
        "mean": 0.0,
        "lower_95": 0.0,
        "upper_95": 0.0,
        "groups": 16,
        "repetitions": 2000,
        "seed": 11235,
        "reliable_log_count": true
      }
    },
    "B1_Expert": {
      "n": 1696,
      "base": 0.9060260849778198,
      "expert": 0.9058558999990978,
      "oracle": 0.9116575580141938,
      "selected": 0.9058558999990978,
      "oracle_gain": 0.0056314730363741,
      "selected_gain": -0.00017018497872191364,
      "selection_regret": 0.005801658015096014,
      "oracle_capture": -0.030220330919224207,
      "switch_count": 1696,
      "switch_rate": 1.0,
      "base_fallback_rate": 0.0,
      "positive_switch_gain_total": 9.550978269690473,
      "negative_switch_cost_total": 9.83961199360284,
      "positive_switches": 1007,
      "negative_switches": 43,
      "improvement_rate": 0.59375,
      "degradation_rate": 0.02535377358490566,
      "tie_rate": 0.38089622641509435,
      "positive_switch_gain_per_scene": 0.0056314730363741,
      "negative_switch_cost_per_scene": 0.005801658015096014,
      "switch_precision": 0.59375,
      "high_to_zero": 5,
      "rate_denominator": 1696,
      "high_to_zero_rate_all_scenes": 0.00294811320754717,
      "base_high_count": 1592,
      "high_to_zero_rate_base_high": 0.00314070351758794,
      "safety": {
        "NC": {
          "selected_violations": 23,
          "base_violations": 19,
          "new_violations": 4,
          "denominator": 1696
        },
        "DAC": {
          "selected_violations": 38,
          "base_violations": 39,
          "new_violations": 5,
          "denominator": 1696
        }
      },
      "large_gap_errors": {
        "0.1": {
          "errors": 16,
          "eligible": 23
        },
        "0.5": {
          "errors": 9,
          "eligible": 15
        }
      },
      "selected_gain_log_bootstrap": {
        "mean": -0.00017018497872191364,
        "lower_95": -0.006947073901516773,
        "upper_95": 0.004043374907134354,
        "groups": 16,
        "repetitions": 2000,
        "seed": 11235,
        "reliable_log_count": true
      }
    },
    "B2_Oracle_at_2": {
      "n": 1696,
      "base": 0.9060260849778198,
      "expert": 0.9058558999990978,
      "oracle": 0.9116575580141938,
      "selected": 0.9116575580141938,
      "oracle_gain": 0.0056314730363741,
      "selected_gain": 0.0056314730363741,
      "selection_regret": 0.0,
      "oracle_capture": 1.0,
      "switch_count": 1007,
      "switch_rate": 0.59375,
      "base_fallback_rate": 0.40625,
      "positive_switch_gain_total": 9.550978269690473,
      "negative_switch_cost_total": 0.0,
      "positive_switches": 1007,
      "negative_switches": 0,
      "improvement_rate": 0.59375,
      "degradation_rate": 0.0,
      "tie_rate": 0.40625,
      "positive_switch_gain_per_scene": 0.0056314730363741,
      "negative_switch_cost_per_scene": 0.0,
      "switch_precision": 1.0,
      "high_to_zero": 0,
      "rate_denominator": 1696,
      "high_to_zero_rate_all_scenes": 0.0,
      "base_high_count": 1592,
      "high_to_zero_rate_base_high": 0.0,
      "safety": {
        "NC": {
          "selected_violations": 19,
          "base_violations": 19,
          "new_violations": 0,
          "denominator": 1696
        },
        "DAC": {
          "selected_violations": 33,
          "base_violations": 39,
          "new_violations": 0,
          "denominator": 1696
        }
      },
      "large_gap_errors": {
        "0.1": {
          "errors": 0,
          "eligible": 23
        },
        "0.5": {
          "errors": 0,
          "eligible": 15
        }
      },
      "selected_gain_log_bootstrap": {
        "mean": 0.0056314730363741,
        "lower_95": 0.0035393794877183688,
        "upper_95": 0.008163566983369676,
        "groups": 16,
        "repetitions": 2000,
        "seed": 11235,
        "reliable_log_count": true
      }
    },
    "B3_ScratchScorer_argmax": {
      "n": 1696,
      "base": 0.9060260849778198,
      "expert": 0.9058558999990978,
      "oracle": 0.9116575580141938,
      "selected": 0.9057283166442043,
      "oracle_gain": 0.0056314730363741,
      "selected_gain": -0.00029776833361535665,
      "selection_regret": 0.005929241369989457,
      "oracle_capture": -0.05287574524321594,
      "switch_count": 1527,
      "switch_rate": 0.9003537735849056,
      "base_fallback_rate": 0.09964622641509434,
      "positive_switch_gain_total": 9.328119893800855,
      "negative_switch_cost_total": 9.8331349876125,
      "positive_switches": 903,
      "negative_switches": 37,
      "improvement_rate": 0.5324292452830188,
      "degradation_rate": 0.021816037735849055,
      "tie_rate": 0.44575471698113206,
      "positive_switch_gain_per_scene": 0.005500070692099561,
      "negative_switch_cost_per_scene": 0.005797839025714917,
      "switch_precision": 0.5913555992141454,
      "high_to_zero": 5,
      "rate_denominator": 1696,
      "high_to_zero_rate_all_scenes": 0.00294811320754717,
      "base_high_count": 1592,
      "high_to_zero_rate_base_high": 0.00314070351758794,
      "safety": {
        "NC": {
          "selected_violations": 23,
          "base_violations": 19,
          "new_violations": 4,
          "denominator": 1696
        },
        "DAC": {
          "selected_violations": 38,
          "base_violations": 39,
          "new_violations": 5,
          "denominator": 1696
        }
      },
      "large_gap_errors": {
        "0.1": {
          "errors": 16,
          "eligible": 23
        },
        "0.5": {
          "errors": 9,
          "eligible": 15
        }
      },
      "selected_gain_log_bootstrap": {
        "mean": -0.00029776833361535665,
        "lower_95": -0.007103508403354349,
        "upper_95": 0.003875245740447115,
        "groups": 16,
        "repetitions": 2000,
        "seed": 11235,
        "reliable_log_count": true
      }
    },
    "B4_ScratchScorer_margin": {
      "n": 1696,
      "base": 0.9060260849778198,
      "expert": 0.9058558999990978,
      "oracle": 0.9116575580141938,
      "selected": 0.9057283166442043,
      "oracle_gain": 0.0056314730363741,
      "selected_gain": -0.00029776833361535665,
      "selection_regret": 0.005929241369989457,
      "oracle_capture": -0.05287574524321594,
      "switch_count": 1527,
      "switch_rate": 0.9003537735849056,
      "base_fallback_rate": 0.09964622641509434,
      "positive_switch_gain_total": 9.328119893800855,
      "negative_switch_cost_total": 9.8331349876125,
      "positive_switches": 903,
      "negative_switches": 37,
      "improvement_rate": 0.5324292452830188,
      "degradation_rate": 0.021816037735849055,
      "tie_rate": 0.44575471698113206,
      "positive_switch_gain_per_scene": 0.005500070692099561,
      "negative_switch_cost_per_scene": 0.005797839025714917,
      "switch_precision": 0.5913555992141454,
      "high_to_zero": 5,
      "rate_denominator": 1696,
      "high_to_zero_rate_all_scenes": 0.00294811320754717,
      "base_high_count": 1592,
      "high_to_zero_rate_base_high": 0.00314070351758794,
      "safety": {
        "NC": {
          "selected_violations": 23,
          "base_violations": 19,
          "new_violations": 4,
          "denominator": 1696
        },
        "DAC": {
          "selected_violations": 38,
          "base_violations": 39,
          "new_violations": 5,
          "denominator": 1696
        }
      },
      "large_gap_errors": {
        "0.1": {
          "errors": 16,
          "eligible": 23
        },
        "0.5": {
          "errors": 9,
          "eligible": 15
        }
      },
      "selected_gain_log_bootstrap": {
        "mean": -0.00029776833361535665,
        "lower_95": -0.007103508403354349,
        "upper_95": 0.003875245740447115,
        "groups": 16,
        "repetitions": 2000,
        "seed": 11235,
        "reliable_log_count": true
      }
    }
  }
}
```
