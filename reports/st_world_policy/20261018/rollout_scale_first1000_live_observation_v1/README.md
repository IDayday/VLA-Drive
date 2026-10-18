# Running consequence policy: first1000 updates

This report reads only the first1000 complete append-only training rows and
small immutable native512 metadata. It does not reload a model, optimizer,
image, feedback payload or metric context and causes zero new score calls.
It records finite losses/gradients and actual sampled FP32 master updates,
not planning improvement. The last100 timing window is the logged slowest
rank update interval; checkpoint/startup/outer-driver costs are excluded.
The formal run continues under its original4096 endpoint, fixed full TRAIN
population and scientific recipe. No intermediate DEV score selected this
checkpoint or changed its training. Final fixed DEV remains pending4096.

The native512 receipt was read and payload file sizes came from its inventory;
this is not an actual native load or a continuous-versus-resumed equivalence
claim. Existing review and recovery evidence remain separately identified.
