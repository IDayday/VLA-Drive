# C4 paired analysis CPU routing

Only the new C4 adapter parent affinity and its imported monitor limits move from CPUs 0/1 to CPUs 6/7. The actual fixed analyzer, paired population, outputs, statistics and source hashes stay unchanged. The frozen original wrapper is not edited. This avoids competing with the live future metadata writer (0/1) and T1 qualification (2–5).

Fifteen CPU tests passed, including explicit parent/monitor affinity agreement and unchanged child analysis-body AST. The old f8c0ed1 review applies to the prior version only; this resource-only delta needs its own narrow independent review before dispatch. Invoke the new frozen version with taskset -c 6,7, not the older HANDOFF 0,1 command. No analysis has yet been launched.
