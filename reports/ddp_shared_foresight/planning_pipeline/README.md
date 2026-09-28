# Official planning pipeline validation

Source eebc6abc76e1eefc1e79d78d6ad57fe2903dbd2f. Two real development scenes from a new-campaign four-update startup checkpoint were exported with FP32 masters, retained W, removed auxiliary heads and one ego trajectory. The existing official NAVSIM v1.1 full-environment CPU evaluator completed both scenes with zero failures. No historical driving model, learned scorer, oracle or environment filtering was used.

The numeric scores in these files are plumbing diagnostics only; they are not a development benchmark or evidence of learning. The one-seed aggregation smoke reports every main comparison NOT_RUN. Formal analysis requires five inference seeds42–46 per checkpoint and preserves complete scene/log denominators. Unit tests verify failure averaging and log-cluster uncertainty without treating repeated targets/seeds as independent training repetitions.23focused CPU tests passed.

No full development or Navtest model result exists yet.
