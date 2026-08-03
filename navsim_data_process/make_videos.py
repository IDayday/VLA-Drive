"""Generate DriveDreamer NAVSIM training videos from processed metadata.

This is a resumable, parallel equivalent of ``make_data.py --make_video``.
Each sample produces the same three 9-frame H.264 clips and first-frame stills,
but metadata generation does not need to be repeated.
"""

import argparse
import json
import multiprocessing as mp
import os
import pickle
import shutil
from functools import partial

from moviepy.editor import ImageSequenceClip
from tqdm import tqdm


VIEWS = ("cam_f0", "cam_l0", "cam_r0")
MIN_VIDEO_BYTES = 100_000
MIN_STILL_BYTES = 10_000


def _write_video(
    images: list[str],
    output_path: str,
    *,
    encoder_preset: str = "medium",
) -> None:
    temporary_path = output_path + f".tmp-{os.getpid()}.mp4"
    clip = ImageSequenceClip(images, fps=2)
    try:
        clip.write_videofile(
            temporary_path,
            codec="libx264",
            preset=encoder_preset,
            verbose=False,
            logger=None,
            threads=1,
        )
        os.replace(temporary_path, output_path)
    finally:
        clip.close()
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _is_valid_file(path: str, minimum_bytes: int) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) >= minimum_bytes


def _copy_still(source_path: str, output_path: str) -> None:
    temporary_path = output_path + f".tmp-{os.getpid()}"
    try:
        shutil.copy2(source_path, temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _resolve_image_path(image_path: str, image_root: str | None) -> str:
    if image_root is None:
        return image_path
    prefix, separator, relative_path = image_path.partition("/trainval/")
    if not separator:
        raise ValueError(f"Cannot resolve trainval-relative image path: {image_path}")
    return os.path.join(image_root, relative_path)


def _process_token(
    token: str,
    *,
    meta_dir: str,
    video_dir: str,
    image_root: str | None = None,
    encoder_preset: str = "medium",
) -> tuple[str, str]:
    try:
        with open(os.path.join(meta_dir, token + ".pkl"), "rb") as handle:
            metadata = pickle.load(handle)

        for view in VIEWS:
            images = [
                _resolve_image_path(path, image_root)
                for path in metadata["glo_images"][view]["image_paths"][3:12]
            ]
            view_dir = os.path.join(video_dir, view)
            os.makedirs(view_dir, exist_ok=True)

            video_path = os.path.join(view_dir, token + ".mp4")
            if not _is_valid_file(video_path, MIN_VIDEO_BYTES):
                _write_video(
                    images,
                    video_path,
                    encoder_preset=encoder_preset,
                )

            still_path = os.path.join(view_dir, token + os.path.splitext(images[0])[1])
            if not _is_valid_file(still_path, MIN_STILL_BYTES):
                _copy_still(images[0], still_path)
        return token, ""
    except Exception as exc:
        return token, repr(exc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="mini")
    parser.add_argument("--data_root", default="navsim_dataset")
    parser.add_argument("--datalist", default=None)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--encoder-preset", default="medium")
    args = parser.parse_args()

    datalist = args.datalist or f"{args.split}_meta.json"
    with open(datalist, "r", encoding="utf-8") as handle:
        tokens = json.load(handle)
    if args.max_samples > 0:
        tokens = tokens[: args.max_samples]

    meta_dir = os.path.join(args.data_root, "meta", args.split)
    video_dir = os.path.join(args.data_root, "navsim_video", args.split)
    worker = partial(
        _process_token,
        meta_dir=meta_dir,
        video_dir=video_dir,
        encoder_preset=args.encoder_preset,
    )

    failures: list[tuple[str, str]] = []
    with mp.Pool(processes=args.workers) as pool:
        results = pool.imap_unordered(worker, tokens, chunksize=1)
        for token, error in tqdm(results, total=len(tokens), desc=f"videos {args.split}"):
            if error:
                failures.append((token, error))

    if failures:
        for token, error in failures[:20]:
            print(f"[ERROR] {token}: {error}")
        raise RuntimeError(f"{len(failures)} video samples failed")


if __name__ == "__main__":
    main()
