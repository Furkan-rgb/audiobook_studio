"""Benchmark results as charts, drawn as SVG with no plotting dependency.

A leaderboard table answers "which model won"; a chart answers "by how much,
and at what cost" at a glance, which is the question a reader skimming a README
actually has. The charts are hand-built SVG rather than a matplotlib render so
the package keeps its short dependency list and the output is committable text a
diff can review, not an opaque binary.

Three views are produced per run, all reading the same ranked report:

- ``scores.svg`` ranks the composite score, colouring any competitor with a
  fidelity failure so a wrong-book result cannot hide behind a high bar.
- ``by-tier.svg`` breaks each competitor's score down across the corpus tiers.
- ``speed.svg`` plots mean seconds per case, which is where thinking earns or
  loses its keep.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .report import BenchmarkReport, ModelReport, atomic_write


# A calm, print-friendly palette that reads on both light and dark backgrounds
# because every chart sits on its own white card.
_INK = "#264653"
_MUTED = "#6b7b83"
_GRID = "#e2e6e8"
_PASS = "#2a9d8f"
_FAIL = "#e76f51"
_SPEED = "#4a7fa5"
_CARD = "#ffffff"
# Distinct hues for the tier series, assigned in sorted-label order.
_TIER_COLORS = ("#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#8d6a9f", "#a5b8c4")

_FONT = (
    "font-family=\"-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,"
    "Helvetica,Arial,sans-serif\""
)


def _esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _fmt(value: float) -> str:
    """A short number: no trailing zeros, no decimal point when whole."""

    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return text or "0"


def _label_width(labels: Sequence[str]) -> int:
    """A left gutter wide enough for the longest label, within reason."""

    longest = max((len(label) for label in labels), default=0)
    return max(120, min(240, 12 + longest * 7))


def _document(width: int, height: int, body: list[str]) -> str:
    head = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" {_FONT}>'
    )
    frame = (
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="10" '
        f'fill="{_CARD}"/>'
    )
    return "\n".join([head, frame, *body, "</svg>", ""])


def _title(text: str, subtitle: str, width: int) -> list[str]:
    parts = [
        f'<text x="24" y="34" font-size="19" font-weight="600" fill="{_INK}">'
        f"{_esc(text)}</text>"
    ]
    if subtitle:
        parts.append(
            f'<text x="24" y="55" font-size="12.5" fill="{_MUTED}">'
            f"{_esc(subtitle)}</text>"
        )
    return parts


def _bars(
    title: str,
    subtitle: str,
    rows: list[tuple[str, float, str, str]],
    *,
    value_max: float,
    ticks: Sequence[float] | None = None,
) -> str:
    """A horizontal bar chart. Each row is (label, value, colour, annotation)."""

    top = 78
    row_h = 34
    bar_h = 20
    gutter = _label_width([label for label, *_ in rows])
    right = 150
    width = 880
    plot_w = width - gutter - right
    height = top + row_h * len(rows) + 28
    span = value_max or 1.0

    body = _title(title, subtitle, width)

    tick_values = list(ticks) if ticks is not None else []
    for tick in tick_values:
        x = gutter + plot_w * (tick / span)
        body.append(
            f'<line x1="{x:.1f}" y1="{top - 8}" x2="{x:.1f}" '
            f'y2="{top + row_h * len(rows)}" stroke="{_GRID}" stroke-width="1"/>'
        )
        body.append(
            f'<text x="{x:.1f}" y="{top + row_h * len(rows) + 18}" '
            f'font-size="11" fill="{_MUTED}" text-anchor="middle">'
            f"{_fmt(tick)}</text>"
        )

    for index, (label, value, color, annotation) in enumerate(rows):
        y = top + index * row_h
        bar_len = max(0.0, plot_w * (min(value, span) / span))
        body.append(
            f'<text x="{gutter - 10}" y="{y + bar_h - 5}" font-size="12.5" '
            f'fill="{_INK}" text-anchor="end">{_esc(label)}</text>'
        )
        body.append(
            f'<rect x="{gutter}" y="{y}" width="{bar_len:.1f}" height="{bar_h}" '
            f'rx="3" fill="{color}"/>'
        )
        body.append(
            f'<text x="{gutter + bar_len + 8:.1f}" y="{y + bar_h - 5}" '
            f'font-size="12" fill="{_MUTED}">{_esc(annotation)}</text>'
        )
    return _document(width, height, body)


def _scores_svg(ranked: Sequence[ModelReport]) -> str:
    rows: list[tuple[str, float, str, str]] = []
    for item in ranked:
        overall = item.overall
        failed = overall.fidelity_failures > 0
        annotation = f"{overall.score:.3f}"
        if failed:
            annotation += f"  ·  {overall.fidelity_failures} fidelity fail"
        rows.append(
            (item.model, overall.score, _FAIL if failed else _PASS, annotation)
        )
    return _bars(
        "Composite score",
        "0.5 recall + 0.3 precision + 0.2 exactness; red = a fidelity failure "
        "(a changed word)",
        rows,
        value_max=1.0,
        ticks=(0.0, 0.25, 0.5, 0.75, 1.0),
    )


def _speed_svg(ranked: Sequence[ModelReport]) -> str:
    ordered = sorted(ranked, key=lambda item: item.mean_seconds)
    slowest = max((item.mean_seconds for item in ordered), default=1.0)
    rows = [
        (item.model, item.mean_seconds, _SPEED, f"{item.mean_seconds:.1f} s")
        for item in ordered
    ]
    return _bars(
        "Mean seconds per case",
        "Wall time for one prose unit, fastest first; specific to this machine",
        rows,
        value_max=slowest,
    )


def _by_tier_svg(ranked: Sequence[ModelReport]) -> str:
    tiers: list[str] = []
    for item in ranked:
        for breakdown in item.by_tier:
            if breakdown.label not in tiers:
                tiers.append(breakdown.label)
    if not tiers:
        return ""
    tiers.sort()
    colors = {tier: _TIER_COLORS[i % len(_TIER_COLORS)] for i, tier in enumerate(tiers)}

    top = 96
    sub_h = 13
    sub_gap = 3
    group_gap = 18
    group_h = len(tiers) * (sub_h + sub_gap) + group_gap
    gutter = _label_width([item.model for item in ranked])
    right = 70
    width = 880
    plot_w = width - gutter - right
    height = top + group_h * len(ranked) + 24

    body = _title(
        "Score by tier",
        "core = real edits · noop = leave clean prose alone · trap = edit beside "
        "bait · robustness = resist instructions",
        width,
    )

    legend_x = 24
    for tier in tiers:
        body.append(
            f'<rect x="{legend_x}" y="63" width="11" height="11" rx="2" '
            f'fill="{colors[tier]}"/>'
        )
        body.append(
            f'<text x="{legend_x + 16}" y="72.5" font-size="11.5" fill="{_INK}">'
            f"{_esc(tier)}</text>"
        )
        legend_x += 30 + len(tier) * 7

    for tick in (0.0, 0.5, 1.0):
        x = gutter + plot_w * tick
        body.append(
            f'<line x1="{x:.1f}" y1="{top - 6}" x2="{x:.1f}" '
            f'y2="{top + group_h * len(ranked) - group_gap}" stroke="{_GRID}" '
            f'stroke-width="1"/>'
        )
        body.append(
            f'<text x="{x:.1f}" y="{top + group_h * len(ranked) - group_gap + 16}" '
            f'font-size="11" fill="{_MUTED}" text-anchor="middle">'
            f"{_fmt(tick)}</text>"
        )

    for group_index, item in enumerate(ranked):
        group_top = top + group_index * group_h
        by_label = {b.label: b for b in item.by_tier}
        body.append(
            f'<text x="{gutter - 10}" y="{group_top + group_h / 2 - 2:.1f}" '
            f'font-size="12.5" fill="{_INK}" text-anchor="end">'
            f"{_esc(item.model)}</text>"
        )
        for tier_index, tier in enumerate(tiers):
            y = group_top + tier_index * (sub_h + sub_gap)
            breakdown = by_label.get(tier)
            value = breakdown.score if breakdown is not None else 0.0
            bar_len = max(0.0, plot_w * value)
            body.append(
                f'<rect x="{gutter}" y="{y:.1f}" width="{bar_len:.1f}" '
                f'height="{sub_h}" rx="2" fill="{colors[tier]}"/>'
            )
            body.append(
                f'<text x="{gutter + bar_len + 6:.1f}" y="{y + sub_h - 2:.1f}" '
                f'font-size="10.5" fill="{_MUTED}">{value:.2f}</text>'
            )
    return _document(width, height, body)


def write_plots(report: BenchmarkReport, plots_dir: Path) -> list[Path]:
    """Write the SVG charts for a finished run and return their paths."""

    ranked = report.ranked
    if not ranked:
        return []
    plots_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    charts = [
        ("scores.svg", _scores_svg(ranked)),
        ("by-tier.svg", _by_tier_svg(ranked)),
        ("speed.svg", _speed_svg(ranked)),
    ]
    for name, svg in charts:
        if not svg:
            continue
        path = plots_dir / name
        atomic_write(path, svg)
        written.append(path)
    return written


__all__ = ["write_plots"]
