"""Unit tests for the delete-and-reflow pipeline.

Run with either:
    python3 -m unittest discover -s tests
    pytest
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.reflow import reflow_words
from app.segmenter import SegmentationError
from app.srt import to_srt


def W(text: str, start_ms: int, end_ms: int) -> SimpleNamespace:
    return SimpleNamespace(text=text, start_ms=start_ms, end_ms=end_ms)


OPTS = dict(
    max_chars=100,
    min_duration_ms=0,
    max_duration_ms=100_000,
    target_duration_ms=2000,
    pause_threshold_ms=1000,
)


# Four words: 0..500, 500..1000, then a 1000 ms gap, 2000..2500, 2500..3000.
BASE = [
    W("a", 0, 500),
    W("b", 500, 1000),
    W("c", 2000, 2500),
    W("d", 2500, 3000),
]


class TestReflow(unittest.TestCase):
    def test_fully_contained_words_are_dropped(self):
        # Delete exactly word b ([500, 1000]); a touches the start and must
        # survive; c/d survive on the other side.
        segs, _ = reflow_words(BASE, [[500, 1000]], **OPTS)
        words = {w for s in segs for w in range(s.word_start, s.word_end + 1)}
        self.assertEqual(words, {0, 2, 3})

    def test_surviving_words_are_shifted_earlier(self):
        # Gap-only cut [1000, 2000] deletes no audio words; every timestamp
        # from 2000 on moves earlier by 1000 ms.
        segs, _ = reflow_words(BASE, [[1000, 2000]], **OPTS)
        self.assertEqual(segs[0].start_ms, 0)
        self.assertEqual(segs[0].end_ms, 1000)
        last = segs[-1]
        self.assertEqual(last.start_ms, 1000)  # was 2000
        self.assertEqual(last.end_ms, 2000)  # was 3000

    def test_shift_accumulates_over_multiple_deletions(self):
        # [500,1000] drops b (500 ms) and [2000,2500] drops c (500 ms);
        # d keeps its original index 3 and is shifted by 1000 ms total.
        segs, _ = reflow_words(BASE, [[500, 1000], [2000, 2500]], **OPTS)
        d_seg = next(s for s in segs if s.word_start == 3)
        self.assertEqual(d_seg.start_ms, 1500)  # 2500 - 1000
        self.assertEqual(d_seg.end_ms, 2000)  # 3000 - 1000

    def test_deletion_forces_separation_of_its_two_sides(self):
        segs, _ = reflow_words(BASE, [[1000, 2000]], **OPTS)
        self.assertGreaterEqual(len(segs), 2)
        # no segment contains words from both sides of the cut
        for s in segs:
            self.assertFalse(
                s.word_end <= 1 and s.word_start >= 2
            )
        self.assertTrue(any(s.word_end == 1 for s in segs))
        self.assertTrue(any(s.word_start == 2 for s in segs))

    def test_word_indices_are_original_request_indices(self):
        segs, _ = reflow_words(BASE, [[500, 1000]], **OPTS)
        self.assertEqual(segs[0].word_start, 0)
        self.assertEqual(segs[0].word_end, 0)
        # the second fragment's first segment still reports index 2, not 1
        self.assertEqual(segs[-1].word_start, 2)
        self.assertEqual(segs[-1].word_end, 3)

    def test_each_fragment_uses_existing_segmentation_rules(self):
        # A fragment that cannot meet min_duration_ms makes the whole reflow
        # impossible -> 422 from the existing optimiser.
        with self.assertRaises(SegmentationError) as ctx:
            reflow_words(BASE, [[1000, 2000]], **{**OPTS, "min_duration_ms": 2000})
        self.assertIn("no legal segmentation", str(ctx.exception))

    def test_422_when_word_crosses_deletion_start(self):
        with self.assertRaises(SegmentationError) as ctx:
            reflow_words(BASE, [[800, 2200]], **OPTS)
        msg = str(ctx.exception)
        self.assertIn("intersects deletion range", msg)
        self.assertIn("word 1", msg)

    def test_422_when_word_crosses_deletion_end(self):
        with self.assertRaises(SegmentationError) as ctx:
            reflow_words(BASE, [[1000, 2400]], **OPTS)
        self.assertIn("intersects deletion range", str(ctx.exception))
        self.assertIn("word 2", str(ctx.exception))

    def test_422_on_unsorted_deletions(self):
        with self.assertRaises(SegmentationError) as ctx:
            reflow_words(BASE, [[2600, 2700], [2000, 2100]], **OPTS)
        self.assertIn("ascending", str(ctx.exception))

    def test_422_on_overlapping_deletions(self):
        with self.assertRaises(SegmentationError) as ctx:
            reflow_words(BASE, [[1000, 2000], [1500, 2100]], **OPTS)
        self.assertIn("overlap", str(ctx.exception))

    def test_422_when_start_ge_end(self):
        for bad in ([[1000, 1000]], [[2000, 1000]]):
            with self.assertRaises(SegmentationError) as ctx:
                reflow_words(BASE, bad, **OPTS)
            self.assertIn("start_ms < end_ms", str(ctx.exception))

    def test_422_when_all_words_deleted(self):
        with self.assertRaises(SegmentationError) as ctx:
            reflow_words(BASE, [[0, 3000]], **OPTS)
        self.assertIn("every word", str(ctx.exception))

    def test_touching_deletion_ranges_are_merged(self):
        # b=[500,1000] is deleted by the first range; merging keeps d's
        # shift equal to the sum and does not split the surviving pair c/d.
        segs, _ = reflow_words(
            BASE, [[500, 1000], [1000, 2000]], **OPTS
        )
        words = {w for s in segs for w in range(s.word_start, s.word_end + 1)}
        self.assertEqual(words, {0, 2, 3})
        d_seg = next(s for s in segs if s.word_start == 2)
        self.assertEqual(d_seg.start_ms, 500)  # 2000 - 1500

    def test_word_spanning_the_touching_point_counts_as_deleted(self):
        words = [W("a", 0, 500), W("b", 500, 2000), W("c", 2000, 2500)]
        segs, _ = reflow_words(
            words, [[500, 1500], [1500, 2000]], **OPTS
        )
        self.assertEqual(
            [s.word_start for s in segs],
            [0, 2],
        )

    def test_deletion_on_exact_word_boundary_keeps_touching_word(self):
        # range ends exactly where c starts: c survives and stays shifted.
        segs, _ = reflow_words(BASE, [[0, 2000]], **OPTS)
        self.assertEqual([s.word_start for s in segs], [2])
        self.assertEqual(segs[0].start_ms, 0)  # 2000 - 2000

    def test_deletion_outside_the_words_is_harmless(self):
        # no word is dropped or shifted; indices/timings match a plain cut
        from app.segmenter import segment_words

        segs, _ = reflow_words(BASE, [[3000, 4000]], **OPTS)
        plain_result = segment_words(BASE, **OPTS)
        self.assertEqual(
            [(s.word_start, s.word_end, s.start_ms, s.end_ms)
             for s in plain_result.segments],
            [(s.word_start, s.word_end, s.start_ms, s.end_ms)
             for s in segs],
        )

    def test_no_deletions_rejected_by_core(self):
        # The HTTP layer requires >= 1 range; calling the core with none
        # is a plain segmentation.
        segs, _ = reflow_words(BASE, [], **OPTS)
        self.assertEqual(
            [(s.word_start, s.word_end) for s in segs],
            [(0, 1), (2, 3)],
        )

    def test_total_cost_sums_per_fragment_costs(self):
        from app.segmenter import segment_words

        left = segment_words(BASE[:2], **OPTS)
        right = segment_words(BASE[2:], **OPTS)
        _, cost = reflow_words(BASE, [[1000, 2000]], **OPTS)
        self.assertAlmostEqual(
            cost, left.total_cost + right.total_cost, places=9
        )

    def test_max_line_chars_applies_within_fragments(self):
        # Three words per fragment joined with spaces are 11 chars, so width
        # 7 forces a two-line layout; the cut splits them into two fragments.
        words = [
            W("aaa", 0, 300),
            W("bbb", 300, 600),
            W("ccc", 600, 900),
            W("ddd", 2000, 2300),
            W("eee", 2300, 2600),
            W("fff", 2600, 2900),
        ]
        segs, _ = reflow_words(
            words, [[900, 2000]], **{**OPTS, "max_line_chars": 7}
        )
        self.assertEqual(len(segs), 2)
        for s in segs:
            self.assertEqual(len(s.text.split("\n")), 2)
            for line in s.text.split("\n"):
                self.assertLessEqual(len(line), 7)

    def test_srt_export_uses_shifted_timeline(self):
        segs, _ = reflow_words(BASE, [[1000, 2000]], **OPTS)
        srt = to_srt(segs)
        self.assertIn("00:00:00,000 --> 00:00:01,000", srt)
        self.assertIn("00:00:01,000 --> 00:00:02,000", srt)
        # blocks are numbered consecutively across fragments
        self.assertTrue(srt.startswith("1\n"))
        self.assertIn("\n2\n", srt)

    def test_overlapping_words_still_validated(self):
        words = [W("a", 0, 1000), W("b", 900, 1900)]
        with self.assertRaises(SegmentationError) as ctx:
            reflow_words(words, [[1000, 1500]], **OPTS)
        self.assertIn("non-overlapping", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
