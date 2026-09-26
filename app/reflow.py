"""Subtitle re-segmentation after deleting video spans.

Pure Python (no web dependencies) so it can be unit-tested in isolation.

Given the same word-level transcript and segmentation parameters as
:func:`app.segmenter.segment_words`, plus a list of *deletion ranges* given
in the **original** video's millisecond timeline:

* deletion ranges must be time-ascending, non-overlapping and satisfy
  ``start_ms < end_ms`` (ranges touching at an endpoint, e.g. ``[100, 200]``
  and ``[200, 300]``, are accepted and merged);
* a word whose ``[start_ms, end_ms]`` lies completely inside a deletion
  range is discarded (endpoint-equal counts as "inside", so a word ending
  exactly where the cut starts survives);
* a word that merely *intersects* a deletion boundary without being fully
  covered — its audio would be partially cut — is rejected with
  :class:`SegmentationError` (HTTP 422), so is a request whose ranges drop
  every word;
* every surviving word's times are shifted earlier by the total length of
  the deletion ranges that end at or before the word's start, producing a
  gap-free timeline;
* each deletion range that separates two surviving words forces a
  **mandatory subtitle edge** between them, so the two sides of a cut can
  never share a subtitle — the surviving words are split into continuous
  fragments and every fragment is optimised independently with the existing
  character-count, duration and two-line-typesetting rules.

The returned segments keep the **original request word indices** in
``word_start`` / ``word_end``; ``start_ms`` / ``end_ms`` live on the
shortened post-deletion timeline, which is also what the SRT export uses.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Sequence

from .segmenter import (
    SegmentationError,
    SegmentResult,
    WordLike,
    segment_words,
    validate_words,
)


@dataclass(frozen=True)
class DeletionRange:
    """A half-open-over-interesting ``[start_ms, end_ms)``-style cut.

    Containment is closed on both ends (``w.start >= start and w.end <=
    end`` drops the word); a word merely touching the range at one point is
    not considered intersecting it.
    """

    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class _ShiftedWord:
    text: str
    start_ms: int
    end_ms: int
    original_index: int
    orig_start_ms: int
    orig_end_ms: int


def _normalise_deletions(
    raw: Sequence[Sequence[int]],
) -> tuple[DeletionRange, ...]:
    """Validate, sort-check and merge touching deletion ranges.

    The ranges must be given in ascending order and may not overlap; ranges
    that touch at an endpoint are fused (a word spanning the touching point
    is covered by both and must count as fully deleted, not as a
    boundary-crossing partial overlap).
    """
    ranges: list[DeletionRange] = []
    for pos, pair in enumerate(raw):
        if len(pair) != 2:
            raise SegmentationError(
                f"deletion range {pair!r} at position {pos} must be a "
                "[start_ms, end_ms] pair"
            )
        lo, hi = pair
        if lo < 0 or hi < 0:
            raise SegmentationError(
                f"deletion range [{lo}, {hi}] at position {pos} must use "
                "non-negative millisecond offsets"
            )
        if lo >= hi:
            raise SegmentationError(
                f"deletion range [{lo}, {hi}] at position {pos} must satisfy "
                "start_ms < end_ms"
            )
        if ranges and lo < ranges[-1].start_ms:
            raise SegmentationError(
                "deletion ranges must be sorted by time in ascending order; "
                f"range [{lo}, {hi}] at position {pos} starts before "
                f"[{ranges[-1].start_ms}, {ranges[-1].end_ms}]"
            )
        if ranges and lo < ranges[-1].end_ms:
            raise SegmentationError(
                "deletion ranges must not overlap; range "
                f"[{lo}, {hi}] at position {pos} overlaps "
                f"[{ranges[-1].start_ms}, {ranges[-1].end_ms}]"
            )
        # Fuse a touching range: previous end == this start.  Both cuts are
        # contiguous on the original timeline.
        if ranges and lo == ranges[-1].end_ms:
            ranges[-1] = DeletionRange(ranges[-1].start_ms, hi)
        else:
            ranges.append(DeletionRange(lo, hi))
    return tuple(ranges)


def _classify_words(
    words: Sequence[WordLike],
    deletions: Sequence[DeletionRange],
) -> list[_ShiftedWord]:
    """Drop fully-deleted words, reject boundary-crossers, shift the rest.

    A word ``w`` and a range ``r`` overlap (interiors, touching excluded)
    when ``w.start_ms < r.end_ms`` and ``r.start_ms < w.end_ms``; among the
    ranges that overlap the word it is fully deleted iff one contains
    ``[w.start_ms, w.end_ms]``.
    """
    del_starts = [r.start_ms for r in deletions]
    kept: list[_ShiftedWord] = []
    for idx, w in enumerate(words):
        # Candidate ranges whose start is strictly below the word's end
        # (touching at r.start == w.end excluded).  Anything starting at or
        # after the word's end cannot intersect it.
        cand = bisect_right(del_starts, w.end_ms - 1, hi=len(deletions))
        covered = False
        for k in range(cand):
            r = deletions[k]
            if w.start_ms >= r.end_ms:
                continue  # range already ended before the word (touch ok)
            if w.start_ms >= r.start_ms and w.end_ms <= r.end_ms:
                covered = True
                break
            raise SegmentationError(
                f"word {idx} ({w.text!r}, {w.start_ms}..{w.end_ms} ms) "
                f"intersects deletion range [{r.start_ms}, {r.end_ms}] "
                "without being fully contained in it; word boundaries may "
                "not cross a deletion edge — extend the cut to a word "
                "boundary or adjust the range"
            )
        if covered:
            continue
        # Shift by the total deleted length at or before the word's start.
        shift = 0
        for r in deletions:
            if r.end_ms <= w.start_ms:
                shift += r.end_ms - r.start_ms
            else:
                break
        kept.append(
            _ShiftedWord(
                text=w.text,
                start_ms=w.start_ms - shift,
                end_ms=w.end_ms - shift,
                original_index=idx,
                orig_start_ms=w.start_ms,
                orig_end_ms=w.end_ms,
            )
        )
    return kept


def _split_fragments(
    kept: Sequence[_ShiftedWord],
    deletions: Sequence[DeletionRange],
) -> list[list[_ShiftedWord]]:
    """Cut between two consecutive kept words when a deletion sits between.

    Given surviving neighbours ``a`` (index t-1) and ``b`` (index t) on the
    original timeline, they are separated when some range satisfies
    ``a.orig_end_ms <= r.start_ms`` and ``r.end_ms <= b.orig_start_ms``.
    Each side of every such cut must end up in a different subtitle.
    """
    fragments: list[list[_ShiftedWord]] = [[kept[0]]]
    del_starts = [r.start_ms for r in deletions]
    del_ends = [r.end_ms for r in deletions]
    for a, b in zip(kept, kept[1:]):
        # Largest range index whose start <= b's original start: any range
        # separating the pair must be found at or below it.
        k = bisect_right(del_starts, b.orig_start_ms, hi=len(deletions)) - 1
        separated = False
        while k >= 0 and del_ends[k] >= a.orig_end_ms:
            if del_ends[k] <= b.orig_start_ms and del_starts[k] >= a.orig_end_ms:
                separated = True
                break
            k -= 1
        if separated:
            fragments.append([b])
        else:
            fragments[-1].append(b)
    return fragments


def reflow_words(
    words: Sequence[WordLike],
    deletions: Sequence[Sequence[int]],
    *,
    max_chars: int,
    min_duration_ms: int,
    max_duration_ms: int,
    target_duration_ms: int,
    pause_threshold_ms: int,
    max_line_chars: int | None = None,
) -> tuple[list[SegmentResult], float]:
    """Delete spans, shift times, and re-segment each surviving fragment.

    Returns the segments (with original request word indices and shifted
    timestamps, re-indexed from 0) and the summed optimiser cost across the
    independently optimised fragments.
    """
    if len(words) == 0:
        raise SegmentationError("words must contain at least one word")
    validate_words(
        words,
        max_chars=max_chars,
        max_duration_ms=max_duration_ms,
        max_line_chars=max_line_chars,
    )
    ranges = _normalise_deletions(deletions)
    kept = _classify_words(words, ranges)
    if not kept:
        raise SegmentationError(
            "every word is fully contained in a deletion range; at least one "
            "word must survive for a subtitle reflow to exist"
        )

    fragments = _split_fragments(kept, ranges)
    segments: list[SegmentResult] = []
    total_cost = 0.0
    for frag in fragments:
        result = segment_words(
            frag,
            max_chars=max_chars,
            min_duration_ms=min_duration_ms,
            max_duration_ms=max_duration_ms,
            target_duration_ms=target_duration_ms,
            pause_threshold_ms=pause_threshold_ms,
            max_line_chars=max_line_chars,
        )
        total_cost += result.total_cost
        for seg in result.segments:
            first = frag[seg.word_start]
            last = frag[seg.word_end]
            segments.append(
                SegmentResult(
                    index=len(segments),
                    text=seg.text,
                    start_ms=first.start_ms,
                    end_ms=last.end_ms,
                    duration_ms=last.end_ms - first.start_ms,
                    word_start=first.original_index,
                    word_end=last.original_index,
                    crossed_boundaries=seg.crossed_boundaries,
                )
            )
    return segments, total_cost
