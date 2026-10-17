"""No scorer calls: reproduce the actual float64 tangent dispatch discrepancy.

Both output layouts are disjoint from the same strided input. NumPy's
conservative overlap test selects scalar libm for the near output and SVML for
the far output on the registered AVX512 NumPy1.26.4 runtime. This probes the
installed operator; it neither patches NumPy nor changes the canonical scorer.
"""
import argparse
import json
import math
from pathlib import Path
import pickle

import numpy as np

from starVLA.model.modules.st_world.consequence import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.replay_root)
    receipt = json.loads((root / "SUMMARY.json").read_text())
    payload = root / "FOUR_REPLAYS.pkl"
    if file_sha256(payload) != receipt["payload_sha256"]:
        raise ValueError("Replay evidence changed")
    with payload.open("rb") as stream:
        replay = pickle.load(stream)
    states = np.stack((replay["reference_states"][3][7], replay["records"][3]["executed_states"][7]))
    storage = np.zeros(64, dtype=np.float64)
    storage[:22].reshape(2, 11)[:] = states
    source = storage[:22].reshape(2, 11)[:, 7]
    saved_input = source.copy()
    cases = []
    for start in (22, 23, 28, 29, 30, 32, 40):
        destination = storage[start:start + 2]
        if np.shares_memory(source, destination):
            raise ValueError("Invalid probe: actual array overlap")
        np.tan(source, out=destination)
        # Exact native expression, using the Pacifica wheelbase already verified
        # in the captured state/model contract. No trajectory is simulated here.
        yaw_rate = states[1, 3] * destination[1] / 3.089
        cases.append({"destination_element_offset": start,
            "actual_memory_overlap": False, "source_stride_bytes": source.strides[0],
            "source_count": len(source), "source_to_destination_bytes": int(destination.ctypes.data - source.ctypes.data),
            "tangent": float(destination[1]), "tangent_hex": float(destination[1]).hex(),
            "yaw_rate": float(yaw_rate), "yaw_rate_hex": float(yaw_rate).hex()})
    if not np.array_equal(source, saved_input):
        raise ValueError("Probe modified its input")
    old = float(.1754005430796875)
    new = float(.17540054307968744)
    reproduced = {case["yaw_rate"] for case in cases} == {old, new}
    result = {"status": "REPRODUCED" if reproduced else "DIFFERENT_CPU_DISPATCH",
        "official_score_calls": 0, "simulation_calls": 0, "numpy": np.__version__,
        "simd": np.core._multiarray_umath.__cpu_features__,
        "captured_replay_sha256": receipt["payload_sha256"],
        "script_sha256": file_sha256(Path(__file__)),
        "input_steering_hex": float(source[1]).hex(),
        "scalar_math_tan_hex": math.tan(float(source[1])).hex(),
        "original_candidate_yaw_rate_hex": old.hex(),
        "fresh_candidate_yaw_rate_hex": new.hex(), "cases": cases}
    out = Path(args.output)
    if out.exists():
        if json.loads(out.read_text()) != result:
            raise ValueError("Immutable layout evidence conflict")
    else:
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "official_score_calls": 0,
                      "different_yaw_rates": sorted({v["yaw_rate"] for v in cases})}))
    if not reproduced:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
