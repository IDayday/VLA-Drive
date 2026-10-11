# Navtest evaluation

For NAVSIM v1 checkpoint inference, parallel PDMS scoring or Navtest scheduling,
read `/mnt/project/skill/navtest-parallel-eval/SKILL.md`. Its versioned source is
`skills/navtest-parallel-eval/SKILL.md` in this repository. Store the active skill
under `/mnt/project/skill`, not under `/root`.

Select hosts and GPU/shard counts from current authorized resources. Read the
dispatch policy and the latest task evaluation override; a past resource plan
does not fix future concurrency. Preserve the registered model, inference and
canonical scoring contracts. The evaluation skill does not authorize changes
to training recipes or resumption of paused research.
