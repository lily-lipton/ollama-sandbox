#!/usr/bin/env python3
# Usage:
#   python3 scripts/jsonl_to_tensorboard.py logs.jsonl --logdir runs/jsonl_import
#   python3 scripts/jsonl_to_tensorboard.py outputs/metrics.jsonl --tag-prefix train/
"""Convert metric JSONL logs into TensorBoard event files for visualization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterator, Tuple

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except ModuleNotFoundError:  # torch not available
    try:
        from tensorboardX import SummaryWriter  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorBoard SummaryWriter が見つかりません。torch または tensorboardX をインストールしてください。"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read metric dictionaries from JSONL and emit TensorBoard summaries."
    )
    parser.add_argument(
        "jsonl",
        type=Path,
        help="Path to a JSONL file containing metric dictionaries (one per line).",
    )
    parser.add_argument(
        "--logdir",
        type=Path,
        default=Path("runs/jsonl_import"),
        help="Destination directory for TensorBoard event files.",
    )
    parser.add_argument(
        "--step-key",
        default="step",
        help="Dictionary key to use as the TensorBoard step (default: step).",
    )
    parser.add_argument(
        "--tag-prefix",
        default="",
        help="Optional prefix to prepend to every TensorBoard tag (e.g. 'train/').",
    )
    parser.add_argument(
        "--no-epoch-fallback",
        action="store_true",
        help="Do not fall back to using the 'epoch' value when step key is missing.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterator[Tuple[int, Dict[str, object]]]:
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {idx}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected dictionary at line {idx}, got {type(record)}")
            yield idx, record


def resolve_step(
    record: Dict[str, object],
    index: int,
    step_key: str,
    allow_epoch_fallback: bool,
) -> int:
    value = record.get(step_key)
    if isinstance(value, (int, float)):
        return int(value)

    if allow_epoch_fallback:
        epoch = record.get("epoch")
        if isinstance(epoch, (int, float)):
            return int(epoch)

    return index


def main() -> None:
    args = parse_args()
    if not args.jsonl.exists():
        raise FileNotFoundError(f"JSONL file not found: {args.jsonl}")

    args.logdir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(args.logdir))
    count = 0

    try:
        for idx, record in iter_jsonl(args.jsonl):
            step = resolve_step(
                record,
                index=idx,
                step_key=args.step_key,
                allow_epoch_fallback=not args.no_epoch_fallback,
            )
            for key, value in record.items():
                if key == args.step_key or not isinstance(value, (int, float)):
                    continue
                tag = f"{args.tag_prefix}{key}"
                writer.add_scalar(tag, value, step)
            count += 1
    finally:
        writer.flush()
        writer.close()

    print(f"Wrote {count} entries to TensorBoard logdir: {args.logdir}")


if __name__ == "__main__":
    main()
