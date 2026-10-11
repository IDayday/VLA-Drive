# Live Base and three independent identities

Direct SSH inspection at2026-10-11T23:27:07UTC confirmed the original launcher
PID2124247 and8 workers on `training-vla-zt2-worker-0`, start2026-10-09T23:57:22UTC,
cwd `/mnt/project/DriveVLA-M0-iqe-20261009`, original `iqe.cli train-base --mode full`.
Fresh log observation at `2026-10-11T23:33:45.789966+00:00`: step **40602 /100000**;
recent100 compute mean **3.961879s**; all losses/gradients finite.
Latest inspected complete checkpoint40k: SHA
`f34a68c10ce91a71651ad0878e1d1bb5baadbb2a2da82073ef9bf52e2812e029`,
22,816,333,071 bytes. No signal/restart/hot source edit was made.
The activity and current speed are separate from the accelerated qualification run.

| Identity | Checkpoint and purpose |
|---|---|
| A continuing Base | Original run, latest complete40k at inspection; continues toward100k |
| B locked incremental Base |22k SHA `42320ea3e6fb09c6cd2b955dde7398a4033338a98a68bc4bac111fe8e5e01e93`; unchanged |
| C cache qualification |33k SHA `5bff6d33d7c8651e589f5f13b0150915d9f0ee1daa3b574aa7207ed7477d047a`; independent bounded200-step outputs |

Both33k and40k frozen visual weights/buffers hash:
`0357360ffd0b71c197cbcdee26371be6810ceb2cbaed5d013a4abf43e82f15dd`.
The immutable full plan binds actual90892 training records and fit hash
`c8bbb9b0914efb1ea1bdbdcf234b9a624375786f090bb5481cc6d3daae9e7c7b`.
No hardcoded101592/103288 population was substituted for that checkpoint's fit list.

Existing user-requested navtest monitoring only:14k85.2969,20k87.1803,33k87.9493
(percent scale).33k completed12146 scenes/136 logs without invalid scores.
The50k/100k monitoring schedule is preserved; this task launches no navtest and
uses no navtest cases for training/cache admission/qualification. These results
are not results of the new cache run.
