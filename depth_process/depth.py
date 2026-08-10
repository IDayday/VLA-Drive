import os
import json
import pickle
import argparse

import numpy as np
import torch
from tqdm import tqdm
from depth_anything_3.api import DepthAnything3

parser = argparse.ArgumentParser()
parser.add_argument("--split", default="mini")
parser.add_argument("--data_root", default="navsim_dataset")
parser.add_argument("--datalist", default=None)
parser.add_argument("--meta_dir", default=None)
parser.add_argument(
    "--max_samples",
    type=int,
    default=0,
    help="Maximum samples per shard; 0 processes the full shard.",
)
parser.add_argument("--rank", type=int, default=0)
parser.add_argument("--world_size", type=int, default=1)
args = parser.parse_args()


def resolve_navsim_data_path(file_name):
    if file_name is None:
        return None
    file_path = os.fspath(file_name)
    runtime_root = os.environ.get("OPENSCENE_DATA_ROOT", "")
    marker = f"{os.sep}navsim_dataset_raw{os.sep}"
    if runtime_root and marker in file_path:
        relative_path = file_path.split(marker, 1)[1]
        sensor_prefix = "sensor_blobs" + os.sep
        sensor_root = os.environ.get("NAVSIM_SENSOR_BLOBS_ROOT", "")
        if sensor_root and relative_path.startswith(sensor_prefix):
            return os.path.join(sensor_root, relative_path[len(sensor_prefix):])
        return os.path.join(runtime_root, relative_path)
    return file_path
if not 0 <= args.rank < args.world_size:
    parser.error("--rank must be in [0, --world_size)")

if args.datalist is None:
    args.datalist = f"{args.split}_meta.json"
if args.meta_dir is None:
    args.meta_dir = os.path.join(args.data_root, "meta", args.split)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.environ.get("DA3_MODEL", "depth-anything/da3metric-large")
model = DepthAnything3.from_pretrained(model_path)
model = model.to(device=device)
model.eval()

with open(args.datalist, 'rb') as f:
    datas = json.load(f)

np.random.seed(2026)
np.random.shuffle(datas)
datas = datas[args.rank::args.world_size]
if args.max_samples > 0:
    datas = datas[:args.max_samples]

os.makedirs("vis", exist_ok=True)
disable_progress = os.environ.get("TQDM_DISABLE", "").lower() in {
    "1",
    "true",
    "yes",
}
missing_inputs = []
for data_n in tqdm(
    datas,
    desc=f"depth shard {args.rank}/{args.world_size}",
    disable=disable_progress,
):
    data_dir = os.path.join(args.meta_dir, data_n + '.pkl')
    depth_path = data_dir + "-depth.pkl"
    # All three fixed-resolution float32 maps produce a file larger than
    # 400 KiB.  A smaller file is necessarily an interrupted/corrupt write and
    # is regenerated below.  New writes are atomic in the local DA3 exporter.
    if os.path.isfile(depth_path) and os.path.getsize(depth_path) >= 400_000:
        continue
    with open(data_dir, 'rb') as f:
        data = pickle.load(f)
    glo_images = data['glo_images']
    source_images = [
        glo_images['cam_f0']['image_paths'][3],
        glo_images['cam_l0']['image_paths'][3],
        glo_images['cam_r0']['image_paths'][3],
    ]
    keys = ['cam_f0', 'cam_l0', 'cam_r0']
    images = []
    sample_missing = []
    for source_image, key in zip(source_images, keys):
        source_image = resolve_navsim_data_path(source_image)
        if os.path.isfile(source_image):
            images.append(source_image)
            continue
        suffix = os.path.splitext(source_image)[1] or ".jpg"
        still_image = os.path.join(
            args.data_root,
            "navsim_video",
            args.split,
            key,
            data_n + suffix,
        )
        if os.path.isfile(still_image):
            images.append(still_image)
        else:
            sample_missing.append((key, source_image, still_image))
    if sample_missing:
        missing_inputs.append((data_n, sample_missing))
        continue
    model.inference(
        images,
        process_res=252,
        export_dir=(images, data_dir, keys, data_n),
        export_format="depth_vis",
    )

if missing_inputs:
    for data_n, paths in missing_inputs[:20]:
        for key, source_image, still_image in paths:
            print(
                f"[MISSING] {data_n} {key}: "
                f"source={source_image} fallback={still_image}"
            )
    raise RuntimeError(
        f"{len(missing_inputs)} depth samples are waiting for camera stills; "
        "rerun this shard after video preprocessing advances"
    )
