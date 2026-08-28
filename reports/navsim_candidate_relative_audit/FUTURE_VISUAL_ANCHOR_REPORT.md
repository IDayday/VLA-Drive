# GT Future Visual Anchor Audit

## Measured data chain

- Audited candidate-target scenes / horizon records: 500 / 2000
- CAM_F0 future file coverage at 0.5/1/2/4 s: 100.000%
- Full synchronization coverage (image + ego pose + annotations + traffic-light field + track tokens + structural target): 100.000%
- Median timestamp resolution error: 0.000262 s
- Rendered visual evidence scenes: 12
- Evidence tokens: `0005d2681afd597b, 008a9f9434c75b99, 00ba15b1edea52fd, 011b69ae584655cc, 01dcab46b55d5e8c, 02018657f0825d92, 0213d7e6fe7b5a41, 025a0d1540ef5632`

The coverage above checks path-backed files for every record. Image decoding, dimensions and pixels are verified on the bounded field-audit sample and every rendered evidence scene; the audit intentionally does not bulk-decode all 2,000 images.

The local data supports the factual synchronization `logged I_GT(t+h) ↔ C_GT,h`.  It does not contain a logged camera image captured from any non-GT candidate pose.  Therefore non-GT candidate images are unavailable as ground truth; reprojected, generated, or synthetic images would be a different supervision class and must be labeled accordingly.

No visual encoder weights were downloaded.  The optional embedding-cache check was intentionally skipped because it is not required to establish the data link.
