# Real Qwen/DDP gradient and deployment check

Source2069df24cf86db39f02477ccf3d0ea1e634670f4. Actual three current cameras and verified generic Qwen3-VL-2B; original802967040-parameter DiT. Zero optimizer updates. Auxiliary test targets are SYNTHETIC and prove only gradient plumbing, never learning or calibrated loss weights.

FM, visual and interaction losses separately produced nonzero W and Qwen layer0 gradients. Frozen native vision had no gradient. Extra poisoned future/teacher fields did not change W or action queries. Deleting both auxiliary heads retained W and yielded bitwise-identical ego for fixed observation/noise. Peak memory16.90GB. Parameter counts in JSON describe the DEPLOYED model after head removal.

Full student train/resume, padding-batch comparison and paired initialization still require their own checks. FullGT teacher is independently training; authorized FLUX VAE access is still pending.
