"""FastAPI application exposing the subtitle segmentation service."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from . import config
from .models import (
    ReflowRequest,
    SegmentOut,
    SegmentRequest,
    SegmentResponse,
)
from .reflow import reflow_words
from .segmenter import SegmentationError, SegmentationResult, segment_words
from .srt import to_srt

app = FastAPI(
    title="Subtitle Segmenter",
    version="1.0.0",
    description=(
        "Split a word-level transcript into subtitles with a dynamic-programming "
        "optimiser. Total cost = sum of squared deviations of subtitle durations "
        "(seconds) from the target duration, plus a 4-point penalty for every "
        "crossed sentence-ending-punctuation or long-pause boundary; when "
        "max_line_chars is given it also includes a two-line typesetting cost "
        "(squared line-length difference divided by the squared line width; "
        "single-line subtitles cost 0). Ties are broken in favour of fewer "
        "subtitles. Optional required_break_after and keep_together fields "
        "add hard, human-edited segment boundaries that the optimiser must "
        "respect."
    ),
)


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok"}


@app.get("/", tags=["meta"])
def root() -> dict:
    return {
        "service": "subtitle-segmenter",
        "docs": "/docs",
        "endpoints": {
            "segment_json": "POST /segment",
            "segment_srt": "POST /segment/srt",
            "reflow_json": "POST /reflow",
            "reflow_srt": "POST /reflow/srt",
            "health": "GET /health",
        },
        "defaults": {
            "max_chars": config.DEFAULT_MAX_CHARS,
            "min_duration_ms": config.DEFAULT_MIN_DURATION_MS,
            "max_duration_ms": config.DEFAULT_MAX_DURATION_MS,
            "target_duration_ms": config.DEFAULT_TARGET_DURATION_MS,
            "pause_threshold_ms": config.DEFAULT_PAUSE_THRESHOLD_MS,
        },
    }


def _run(req: SegmentRequest) -> SegmentationResult:
    try:
        return segment_words(
            req.words,
            max_chars=req.max_chars,
            min_duration_ms=req.min_duration_ms,
            max_duration_ms=req.max_duration_ms,
            target_duration_ms=req.target_duration_ms,
            pause_threshold_ms=req.pause_threshold_ms,
            max_line_chars=req.max_line_chars,
            required_break_after=req.required_break_after,
            keep_together=req.keep_together,
        )
    except SegmentationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/segment", response_model=SegmentResponse, tags=["segment"])
def segment(req: SegmentRequest) -> SegmentResponse:
    """Segment the transcript and return the subtitles as JSON."""
    result = _run(req)
    return SegmentResponse(
        segment_count=len(result.segments),
        total_cost=round(result.total_cost, 6),
        segments=[SegmentOut(**vars(seg)) for seg in result.segments],
    )


@app.post(
    "/segment/srt",
    response_class=PlainTextResponse,
    tags=["segment"],
)
def segment_srt(req: SegmentRequest) -> PlainTextResponse:
    """Segment the transcript and return the subtitles as an SRT document."""
    result = _run(req)
    return PlainTextResponse(to_srt(result.segments))


def _run_reflow(
    req: ReflowRequest,
) -> tuple[list, float]:
    try:
        return reflow_words(
            req.words,
            req.deletions,
            max_chars=req.max_chars,
            min_duration_ms=req.min_duration_ms,
            max_duration_ms=req.max_duration_ms,
            target_duration_ms=req.target_duration_ms,
            pause_threshold_ms=req.pause_threshold_ms,
            max_line_chars=req.max_line_chars,
        )
    except SegmentationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/reflow", response_model=SegmentResponse, tags=["reflow"])
def reflow(req: ReflowRequest) -> SegmentResponse:
    """Delete video spans, shift timings, and re-segment (JSON output).

    The returned ``word_start`` / ``word_end`` indices still refer to the
    word positions in the request body; ``start_ms`` / ``end_ms`` are on the
    shortened post-deletion timeline.
    """
    segments, total_cost = _run_reflow(req)
    return SegmentResponse(
        segment_count=len(segments),
        total_cost=round(total_cost, 6),
        segments=[SegmentOut(**vars(seg)) for seg in segments],
    )


@app.post(
    "/reflow/srt",
    response_class=PlainTextResponse,
    tags=["reflow"],
)
def reflow_srt(req: ReflowRequest) -> PlainTextResponse:
    """Delete video spans, shift timings, and re-segment (SRT document)."""
    segments, _ = _run_reflow(req)
    return PlainTextResponse(to_srt(segments))
