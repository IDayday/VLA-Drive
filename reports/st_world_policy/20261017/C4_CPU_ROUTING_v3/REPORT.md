# C4 actual child CPU routing correction

Independent review rejected v2: the reused monitor hardcodes taskset 0,1, so changing its reported budget and parent affinity did not actually route its child. That failed review and all v2 evidence are retained. No analysis was dispatched with v2.

The C4-only adapter now routes the dynamically imported core subprocess.Popen through an exact-prefix proxy. Only the reviewed taskset 0,1 prefix is accepted and mapped to 6,7. The core source, loop, accounting, process ownership, shutdown and scientific analyzer are unchanged. Unknown spawn prefixes are rejected. Parent and reported limits remain 6,7.

Tests include a real CPU child reporting sched_getaffinity==[6,7], binding of the actual core monitor globals to the proxy, wrong-prefix failures, plus the prior fixed-input/analysis-body tests. This correction requires independent delta review before the single actual C4 analysis. It does not touch T1 or the metadata writer.
