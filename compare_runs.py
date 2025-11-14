#!/usr/bin/env python3
"""Summarise distributed training runs by parsing history logs."""
from __future__ import annotations

import argparse
import math
from pathlib import Path


def parse_history(history_path: Path) -> dict | None:
    """Return best validation metrics from a history log."""
    if not history_path.exists():
        return None

    best: dict | None = None
    for line in history_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or not line.startswith("Epoch"):
            continue
        try:
            epoch_part, metrics_part = line.split(",", 1)
            epoch = int(epoch_part.split()[1])
        except (ValueError, IndexError):
            continue

        metrics: dict[str, float] = {}
        for chunk in metrics_part.split(","):
            if "=" not in chunk:
                continue
            key, value = chunk.split("=", 1)
            try:
                metrics[key.strip()] = float(value)
            except ValueError:
                continue

        val_loss = metrics.get("val_loss")
        if val_loss is None:
            continue
        if best is None or val_loss < best["val_loss"]:
            best = {
                "epoch": epoch,
                "val_loss": val_loss,
                "val_mae": metrics.get("val_mae"),
                "val_rmse": metrics.get("val_rmse"),
            }
    return best


def format_metric(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "NA"
    return f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare distributed run histories.")
    parser.add_argument(
        "--run-dir",
        dest="run_dirs",
        action="append",
        required=True,
        help="Path to a run directory containing history_distributed.log.",
    )
    parser.add_argument(
        "--label",
        dest="labels",
        action="append",
        help="Optional label for each run (defaults to directory name).",
    )
    parser.add_argument(
        "--summary",
        required=True,
        help="Path to write TSV summary data.",
    )
    parser.add_argument(
        "--report",
        required=True,
        help="Path to write plain-text comparison table.",
    )
    args = parser.parse_args()

    run_dirs = [Path(p).expanduser() for p in args.run_dirs]
    labels = args.labels or []
    if labels and len(labels) != len(run_dirs):
        raise SystemExit("Number of --label entries must match --run-dir entries.")

    entries = []
    for idx, run_dir in enumerate(run_dirs):
        label = labels[idx] if labels else run_dir.name
        history = parse_history(run_dir / "history_distributed.log")
        entry = {
            "model": label,
            "output": str(run_dir),
            "epoch": str(history["epoch"]) if history else "NA",
            "val_loss": history["val_loss"] if history else math.nan,
            "val_mae": (
                history["val_mae"]
                if history and history.get("val_mae") is not None
                else math.nan
            ),
            "val_rmse": (
                history["val_rmse"]
                if history and history.get("val_rmse") is not None
                else math.nan
            ),
        }
        entries.append(entry)

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as fh:
        fh.write("model\tbest_epoch\tbest_val_loss\tbest_val_mae\tbest_val_rmse\toutput_dir\n")
        for entry in entries:
            fh.write(
                f"{entry['model']}\t{entry['epoch']}\t"
                f"{format_metric(entry['val_loss'])}\t"
                f"{format_metric(entry['val_mae'])}\t"
                f"{format_metric(entry['val_rmse'])}\t"
                f"{entry['output']}\n"
            )

    def sort_key(item: dict) -> tuple[float, str]:
        return (math.inf if math.isnan(item["val_loss"]) else item["val_loss"], item["model"])

    sorted_entries = sorted(entries, key=sort_key)
    lines: list[str] = [
        "Model comparison (sorted by best val_loss):",
        "model\tbest_epoch\tval_loss\tval_mae\tval_rmse\toutput_dir",
    ]
    for entry in sorted_entries:
        lines.append(
            f"{entry['model']}\t{entry['epoch']}\t"
            f"{format_metric(entry['val_loss'])}\t"
            f"{format_metric(entry['val_mae'])}\t"
            f"{format_metric(entry['val_rmse'])}\t"
            f"{entry['output']}"
        )

    report_text = "\n".join(lines)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text + "\n", encoding="utf-8")
    print(report_text)


if __name__ == "__main__":
    main()
