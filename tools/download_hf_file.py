"""Resumable concurrent HTTP Range downloader for large Hugging Face files."""

import argparse
import concurrent.futures
import json
import os
import threading
import time
from pathlib import Path

import requests
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", required=True, type=int)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--chunk-mb", type=int, default=32)
    parser.add_argument("--retries", type=int, default=30)
    parser.add_argument("--min-kbps", type=int, default=256)
    parser.add_argument("--low-speed-seconds", type=int, default=15)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    state_path = output.with_name(output.name + ".download.json")
    chunk_size = args.chunk_mb * 1024 * 1024
    chunks = [
        (index, start, min(start + chunk_size, args.size) - 1)
        for index, start in enumerate(range(0, args.size, chunk_size))
    ]

    existing_size = output.stat().st_size if output.exists() else 0
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if (
            state.get("url") != args.url
            or state.get("size") != args.size
            or state.get("chunk_size") != chunk_size
        ):
            raise RuntimeError(f"Incompatible download state: {state_path}")
        completed = set(state.get("completed", []))
    else:
        # Reuse only complete leading chunks from a previous sequential download.
        completed = {index for index, _, end in chunks if end < existing_size}
        state = {
            "url": args.url,
            "size": args.size,
            "chunk_size": chunk_size,
            "completed": sorted(completed),
        }

    with output.open("ab"):
        pass
    os.truncate(output, args.size)

    state_lock = threading.Lock()

    def save_state() -> None:
        temp_path = state_path.with_name(state_path.name + ".tmp")
        temp_path.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
        os.replace(temp_path, state_path)

    save_state()

    def download_chunk(item: tuple[int, int, int]) -> int:
        index, start, end = item
        expected = end - start + 1
        headers = {"Range": f"bytes={start}-{end}"}
        last_error: Exception | None = None

        for attempt in range(args.retries):
            try:
                with requests.get(
                    args.url,
                    headers=headers,
                    stream=True,
                    allow_redirects=True,
                    timeout=(30, 120),
                ) as response:
                    response.raise_for_status()
                    content_range = response.headers.get("Content-Range", "")
                    expected_prefix = f"bytes {start}-{end}/"
                    if response.status_code != 206 or not content_range.startswith(expected_prefix):
                        raise RuntimeError(
                            f"Server ignored requested range {start}-{end}: "
                            f"status={response.status_code}, Content-Range={content_range!r}"
                        )

                    fd = os.open(output, os.O_WRONLY)
                    try:
                        offset = start
                        received = 0
                        window_start = time.monotonic()
                        window_bytes = 0
                        for block in response.iter_content(chunk_size=64 * 1024):
                            if not block:
                                continue
                            os.pwrite(fd, block, offset)
                            offset += len(block)
                            received += len(block)
                            window_bytes += len(block)
                            elapsed = time.monotonic() - window_start
                            if elapsed >= args.low_speed_seconds:
                                speed_kbps = window_bytes / elapsed / 1024
                                if speed_kbps < args.min_kbps:
                                    raise RuntimeError(
                                        f"Range {start}-{end} stalled at {speed_kbps:.1f} KiB/s"
                                    )
                                window_start = time.monotonic()
                                window_bytes = 0
                    finally:
                        os.close(fd)

                    if received != expected:
                        raise RuntimeError(
                            f"Short range {start}-{end}: received {received}, expected {expected}"
                        )

                with state_lock:
                    completed.add(index)
                    state["completed"] = sorted(completed)
                    save_state()
                return expected
            except Exception as exc:
                last_error = exc
                time.sleep(min(2 ** min(attempt, 4), 20))

        raise RuntimeError(f"Range {start}-{end} failed after {args.retries} attempts") from last_error

    pending = [item for item in chunks if item[0] not in completed]
    initial_bytes = sum(end - start + 1 for index, start, end in chunks if index in completed)
    with tqdm(total=args.size, initial=initial_bytes, unit="B", unit_scale=True, desc=output.name) as bar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(download_chunk, item) for item in pending]
            for future in concurrent.futures.as_completed(futures):
                bar.update(future.result())

    if len(completed) != len(chunks):
        raise RuntimeError(f"Only {len(completed)}/{len(chunks)} chunks completed")
    if output.stat().st_size != args.size:
        raise RuntimeError(f"Wrong final size: {output.stat().st_size} != {args.size}")
    state_path.unlink()
    print(f"[OK] {output} ({args.size} bytes)")


if __name__ == "__main__":
    main()
