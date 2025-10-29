#!/usr/bin/env python3
# Usage:
#   python3 scripts/show_trainer_logs.py [CHECKPOINT_DIR_OR_STATE_FILE]
#   python3 scripts/show_trainer_logs.py outputs/checkpoint-100 --tail 10 --drop-step
#   python3 scripts/show_trainer_logs.py outputs/checkpoint-50 --min-epoch 75
"""Utility to inspect `trainer_state.json` log history emitted by Hugging Face Trainer."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print log_history entries from a Hugging Face Trainer checkpoint "
            "in a concise dict format."
        )
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="outputs/checkpoint-100",
        help="Path to a checkpoint directory or trainer_state.json file.",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=0,
        help="Limit output to the last N log entries (0 means show all).",
    )
    parser.add_argument(
        "--min-epoch",
        type=float,
        default=None,
        help="Only display entries whose epoch is >= this value.",
    )
    parser.add_argument(
        "--drop-step",
        action="store_true",
        help="Omit the 'step' field from the printed dictionaries.",
    )
    return parser.parse_args()


def load_log_history(checkpoint_path: Path) -> List[Mapping[str, object]]:
    state_path = checkpoint_path
    if checkpoint_path.is_dir():
        state_path = checkpoint_path / "trainer_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"trainer_state.json not found at: {state_path}")

    with state_path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)

    history = state.get("log_history") or []
    # trainer_state keeps summary metrics in log_history already, so just return it.
    return history


def filter_by_epoch(
    logs: Iterable[Mapping[str, object]], min_epoch: float | None
) -> List[Mapping[str, object]]:
    if min_epoch is None:
        return list(logs)
    filtered = []
    for entry in logs:
        epoch = entry.get("epoch")
        if epoch is None or epoch >= min_epoch:
            filtered.append(entry)
    return filtered


def reorder_entry(entry: Mapping[str, object], drop_step: bool) -> Mapping[str, object]:
    preferred_order: Sequence[str] = (
        "loss",
        "grad_norm",
        "learning_rate",
        "eval_loss",
        "eval_runtime",
        "eval_samples_per_second",
        "eval_steps_per_second",
        "train_runtime",
        "train_samples_per_second",
        "train_steps_per_second",
        "train_loss",
        "epoch",
    )

    items = dict(entry)
    if drop_step:
        items.pop("step", None)

    ordered = OrderedDict()
    for key in preferred_order:
        if key in items:
            ordered[key] = items.pop(key)

    for key in sorted(items):
        ordered[key] = items[key]

    return ordered


def maybe_tail(logs: List[Mapping[str, object]], tail_count: int) -> List[Mapping[str, object]]:
    if tail_count <= 0 or tail_count >= len(logs):
        return logs
    return logs[-tail_count:]


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)

    history = load_log_history(checkpoint_path)
    history = filter_by_epoch(history, args.min_epoch)
    history = maybe_tail(history, args.tail)

    for entry in history:
        ordered = reorder_entry(entry, drop_step=args.drop_step)
        print(dict(ordered))


if __name__ == "__main__":
    main()
