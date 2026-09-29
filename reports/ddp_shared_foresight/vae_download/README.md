# Exact registered FLUX VAE obtained on 2026-09-29

The BFL repository still requires authentication, but the Diffusers team publicly distributes the identical standalone VAE at `diffusers/FLUX.1-vae`, revision `da548cfb003bdeebaff6da0211fc8fbc67cb563a`. Both the config Git blob and the 167,666,902-byte safetensors SHA256 match the pinned BFL source exactly. This is a download-source correction, not a replacement visual teacher or changed experiment. The earlier access blocker resulted from checking only the BFL download entry.

Public source: https://huggingface.co/diffusers/FLUX.1-vae

Weight SHA256: `f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3`. Config Git blob: `b43183d0f5f0274bccd8054cd0069fc1d5f64586`. Initial transport attempt failed with a transient TLS EOF; a bounded retry completed the pinned download and all hashes passed. No account credentials or access-control bypass were used.

The actual 83,819,683-parameter VAE loaded in the existing Diffusers0.35.2 environment. CPU forward and reconstruction on three real current training camera images were finite. Repeated posterior-mode encodings were bitwise identical and did not consume RNG state; all parameters remained frozen and eval mode remained enforced. Short-side256 preprocessing produces3x256x456 inputs and16x32x57latents. Scaling0.3611, shift0.1159, stride8. This test made zero optimizer updates and used no GPU.

Model files remain in the independent artifact directory and are not committed. Full future-latent caching and formal student training remain subsequent stages.
