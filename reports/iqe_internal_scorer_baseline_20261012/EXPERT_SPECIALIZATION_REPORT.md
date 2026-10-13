# Expert specialization

Raw IL is inherited channel-sum/time-mean L1. All PDMS are [0,1].

```json
{
  "H_train": {
    "n": 2181,
    "base_PDMS": 0.7930600524191329,
    "expert_PDMS": 0.8351289790304971,
    "expert_minus_base": 0.04206892661136414,
    "base_IL": 0.030679507669555052,
    "expert_IL": 0.027547469581675907,
    "log_bootstrap": {
      "mean": 0.04206892661136414,
      "lower_95": 0.03315415178029268,
      "upper_95": 0.05133868092188479,
      "groups": 655,
      "repetitions": 2000,
      "seed": 11235,
      "reliable_log_count": true
    }
  },
  "H_val": {
    "n": 205,
    "base_PDMS": 0.6934243844998749,
    "expert_PDMS": 0.688716179784437,
    "expert_minus_base": -0.004708204715437818,
    "base_IL": 0.1739365088594396,
    "expert_IL": 0.17305591437147885,
    "log_bootstrap": {
      "mean": -0.004708204715437818,
      "lower_95": -0.01859438185727135,
      "upper_95": 0.00610689615595097,
      "groups": 38,
      "repetitions": 2000,
      "seed": 11235,
      "reliable_log_count": true
    }
  },
  "ordinary_train": {
    "n": 2048,
    "base_PDMS": 0.971726239328395,
    "expert_PDMS": 0.9680873108241684,
    "expert_minus_base": -0.003638928504226562,
    "base_IL": 0.029590583634217182,
    "expert_IL": 0.032523139146633184,
    "log_bootstrap": {
      "mean": -0.003638928504226562,
      "lower_95": -0.00688828007755685,
      "upper_95": -0.00114991868965787,
      "groups": 683,
      "repetitions": 2000,
      "seed": 11235,
      "reliable_log_count": true
    }
  }
}
```
