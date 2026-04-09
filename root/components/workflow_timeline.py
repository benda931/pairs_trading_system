# -*- coding: utf-8 -*-
# root/components/workflow_timeline.py
"""
Pipeline Workflow Timeline Component
======================================
Visual timeline bar showing the status of each pipeline stage.

Stage statuses:
  pending   — not yet run (muted)
  running   — currently executing (blue pulse)
  success   — completed successfully (green)
  skipped   — deliberately skipped (grey)
  warning   — completed with non-fatal issues (yellow)
  failed    — hard failure (red)

Usage
-----
from root.components.workflow_timeline import render_workflow_timeline

stages = [
    {"name": "data_ingest",  "status": "success",  "duration_s": 4.2,  "timestamp": "06:00:04"},
    {"name": "quant_engine", "status": "success",  "duration_s": 18.7, "timestamp": "06:00:23"},
    {"name": "scoring",      "status": "warning",  "duration_s": 0.9,  "timestamp": "06:00:24",
     "message": "3 pairs dropped due to stale data"},
    {"name": "risk",         "status": "running",  "duration_s": None, "timestamp": None},
    {"name": "artifacts",    "status": "pending",  "duration_s": None, "timestamp": None},
]
render_workflow_timeline(stages, title="PIPELINE STATUS")
"""

from __future__ import annotations

from typing import Optional
import streamlit as st

from root.design_system import DS


# ---------------------------------------------------------------------------
# Status configuration
# ---------------------------------------------------------------------------

_STATUS_CONFIG = {
    "pending": {
        "color": "rgba(255,255,255,0.15)",
        "bg":    "rgba(255,255,255,0.04)",
        "icon":  "○",
    },
    "running": {
        "color": DS.BRAND,
        "bg":    f"{DS.BRAND}22",
        "icon":  "◉",
    },
    "success": {
        "color": DS.POSITIVE,
        "bg":    f"{DS.POSITIVE}18",
        "icon":  "●",
    },
    "skipped": {
        "color": DS.TEXT_MUTED,
        "bg":    "rgba(255,255,255,0.04)",
        "icon":  "◌",
    },
    "warning": {
        "color": DS.CAUTION,
        "bg":    f"{DS.CAUTION}18",
        "icon":  "◆",
    },
    "failed": {
        "color": DS.CRITICAL,
        "bg":    f"{DS.CRITICAL}22",
        "icon":  "✕",
    },
}


def _stage_chip(
    name: str,
    status: str,
    duration_s: Optional[float],
    timestamp: Optional[str],
    is_last: bool,
) -> str:
    cfg   = _STATUS_CONFIG.get(status, _STATUS_CONFIG["pending"])
    color = cfg["color"]
    bg    = cfg["bg"]
    icon  = cfg["icon"]

    dur_text = ""
    if duration_s is not None:
        dur_text = f" {duration_s:.1f}s" if duration_s < 60 else f" {duration_s/60:.1f}m"

    ts_block = f'<div style="font-size:9px;color:{DS.TEXT_MUTED};">{timestamp or ""}{dur_text}</div>'

    chip = f"""
    <div style="display:flex;flex-direction:column;align-items:center;min-width:80px;">
      <div style="
          display:flex;align-items:center;justify-content:center;
          width:28px;height:28px;
          background:{bg};
          border:1.5px solid {color};
          border-radius:50%;
          font-size:{DS.FONT_MD};
          color:{color};
          z-index:1;
          position:relative;
      ">{icon}</div>
      <div style="font-size:10px;color:{DS.TEXT_SECONDARY};margin-top:4px;
                  text-align:center;white-space:nowrap;max-width:80px;
                  overflow:hidden;text-overflow:ellipsis;">
        {name.replace('_', ' ')}
      </div>
      {ts_block}
    </div>
    """

    connector = ""
    if not is_last:
        connector = (
            '<div style="flex:1;height:1.5px;background:rgba(255,255,255,0.08);'
            'margin-top:-20px;min-width:16px;"></div>'
        )

    return chip + connector


# ---------------------------------------------------------------------------
# render_workflow_timeline
# ---------------------------------------------------------------------------

def render_workflow_timeline(
    stages: list[dict],
    *,
    title: Optional[str] = None,
    compact: bool = False,
) -> None:
    """
    Render the pipeline workflow timeline.

    Parameters
    ----------
    stages:  List of stage dicts with keys:
             name       — display name (underscores → spaces)
             status     — pending / running / success / skipped / warning / failed
             duration_s — float seconds elapsed (None if not run)
             timestamp  — string timestamp (None if not run)
             message    — optional detail shown in expander on failure/warning
    title:   Optional section title.
    compact: Suppress timestamps (denser layout).
    """
    if not stages:
        return

    if title:
        st.markdown(
            f'<div style="font-size:{DS.FONT_XS};font-weight:700;'
            f'color:{DS.TEXT_MUTED};text-transform:uppercase;'
            f'letter-spacing:0.08em;margin-bottom:{DS.SPACE_2};">{title}</div>',
            unsafe_allow_html=True,
        )

    total    = len(stages)
    counts   = {}
    for s in stages:
        st_ = s.get("status", "pending")
        counts[st_] = counts.get(st_, 0) + 1

    done     = counts.get("success", 0) + counts.get("skipped", 0)
    failed   = counts.get("failed", 0)
    warnings = counts.get("warning", 0)
    running  = counts.get("running", 0)
    pct      = int(done / total * 100) if total else 0

    prog_color = (
        DS.CRITICAL if failed
        else DS.CAUTION if warnings
        else DS.BRAND if running
        else DS.POSITIVE if pct == 100
        else DS.TEXT_MUTED
    )

    chips = "".join(
        _stage_chip(
            name=s.get("name", f"stage_{i}"),
            status=s.get("status", "pending"),
            duration_s=None if compact else s.get("duration_s"),
            timestamp=None if compact else s.get("timestamp"),
            is_last=(i == total - 1),
        )
        for i, s in enumerate(stages)
    )

    summary_parts = [f'<span style="color:{DS.TEXT_SECONDARY};">{done}/{total} done</span>']
    if running:
        summary_parts.append(f'<span style="color:{DS.BRAND};">{running} running</span>')
    if warnings:
        summary_parts.append(f'<span style="color:{DS.CAUTION};">{warnings} warning</span>')
    if failed:
        summary_parts.append(f'<span style="color:{DS.CRITICAL};">{failed} failed</span>')
    summary_html = " · ".join(summary_parts)

    html = f"""
    <div style="
        background:{DS.BG_SURFACE};
        border:1px solid rgba(255,255,255,0.06);
        border-radius:{DS.RADIUS_MD};
        padding:{DS.SPACE_3} {DS.SPACE_4} {DS.SPACE_2};
        margin-bottom:{DS.SPACE_3};
    ">
      <div style="height:2px;background:rgba(255,255,255,0.06);border-radius:1px;
                  margin-bottom:{DS.SPACE_3};">
        <div style="height:2px;width:{pct}%;background:{prog_color};border-radius:1px;"></div>
      </div>
      <div style="display:flex;align-items:flex-start;overflow-x:auto;gap:0;
                  padding-bottom:{DS.SPACE_2};">
        {chips}
      </div>
      <div style="font-size:{DS.FONT_XS};margin-top:{DS.SPACE_1};
                  padding-top:{DS.SPACE_2};border-top:1px solid rgba(255,255,255,0.05);">
        {summary_html}
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

    for s in stages:
        if s.get("status") == "failed" and s.get("message"):
            with st.expander(f"Error detail — {s['name']}", expanded=True):
                st.code(s.get("message", ""), language="text")

    for s in stages:
        if s.get("status") == "warning" and s.get("message"):
            with st.expander(f"Warning detail — {s['name']}"):
                st.caption(s.get("message", ""))
