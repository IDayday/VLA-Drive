# Independent Relabel Report

G0-R is **FAIL**. The run stopped at the first fixed scene, as required by the
score-reconstruction hard gate.

| Check | Result |
| --- | ---: |
| Requested scenes | 200 |
| Attempted scenes | 1 |
| Completed scenes | 0 |
| Candidates evaluated before stop | 256 |
| Run1 logical SHA256 | NOT_AVAILABLE |
| Run2 logical SHA256 | NOT_RUN |
| Max run-to-run error | NOT_RUN |
| Score reconstruction error | 0.38292398055394494 |
| G0-R | FAIL |

The official scorer includes Driving Direction Compliance as a multiplicative
term. The required independent store contains exactly NC, DAC, EP, TTC and
Comfort, so its required five-factor reconstruction cannot reproduce those
candidate scores within `1e-6`. No factor was merged, dropped, or redefined.
