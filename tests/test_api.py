"""API integration tests (require fastapi + httpx).

Run with either:
    python3 -m unittest discover -s tests
    pytest
"""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app.main import app
from app.srt import format_timestamp


def _word(text, start_ms, end_ms):
    return {"text": text, "start_ms": start_ms, "end_ms": end_ms}


BODY = {
    "words": [
        _word("Hello", 0, 500),
        _word("world.", 500, 1400),
        _word("How", 2000, 2400),
        _word("are", 2400, 2700),
        _word("you", 2700, 3200),
        _word("doing", 3200, 3800),
        _word("today?", 3800, 4600),
        _word("Fine", 5200, 5600),
        _word("thanks.", 5600, 6400),
    ],
    "max_chars": 42,
    "min_duration_ms": 1000,
    "max_duration_ms": 7000,
    "target_duration_ms": 3000,
    "pause_threshold_ms": 500,
}


class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"status": "ok"})

    def test_segment_json(self):
        r = self.client.post("/segment", json=BODY)
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(data["segment_count"], 3)
        segs = data["segments"]
        self.assertEqual(segs[0]["start_ms"], 0)
        self.assertEqual(segs[-1]["end_ms"], 6400)
        # natural boundaries (punctuation or >=500 ms gaps) should land on
        # segment edges, not be crossed
        self.assertEqual(sum(s["crossed_boundaries"] for s in segs), 0)
        # every word covered once, in order
        self.assertEqual(segs[0]["word_start"], 0)
        for a, b in zip(segs, segs[1:]):
            self.assertEqual(b["word_start"], a["word_end"] + 1)
        self.assertEqual(segs[-1]["word_end"], len(BODY["words"]) - 1)
        for s in segs:
            self.assertLessEqual(len(s["text"]), 42)
            self.assertGreaterEqual(s["duration_ms"], 1000)
            self.assertLessEqual(s["duration_ms"], 7000)

    def test_segment_srt(self):
        r = self.client.post("/segment/srt", json=BODY)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.headers["content-type"].startswith("text/plain"))
        srt = r.text
        self.assertTrue(srt.startswith("1\n"))
        self.assertIn("00:00:00,000 --> 00:00:01,400", srt)
        self.assertIn("Hello world.", srt)
        self.assertIn("Fine thanks.", srt)

    def test_422_on_overlong_word(self):
        body = {**BODY, "words": [_word("x" * 50, 0, 500)]}
        r = self.client.post("/segment", json=body)
        self.assertEqual(r.status_code, 422)
        self.assertIn("max_chars", r.json()["detail"])

    def test_422_on_overlapping_words(self):
        body = {
            **BODY,
            "words": [_word("a", 0, 500), _word("b", 400, 900)],
        }
        r = self.client.post("/segment", json=body)
        self.assertEqual(r.status_code, 422)
        self.assertIn("non-overlapping", r.json()["detail"])

    def test_422_on_impossible_min_duration(self):
        body = {
            **BODY,
            "words": [_word("a", 0, 300)],
            "min_duration_ms": 1000,
        }
        r = self.client.post("/segment", json=body)
        self.assertEqual(r.status_code, 422)
        self.assertIn("no legal segmentation", r.json()["detail"])

    def test_422_on_schema_violation(self):
        body = {**BODY, "words": [_word("a", 500, 100)]}  # end < start
        r = self.client.post("/segment", json=body)
        self.assertEqual(r.status_code, 422)

    def test_defaults_applied_when_parameters_omitted(self):
        r = self.client.post(
            "/segment",
            json={"words": BODY["words"]},
        )
        self.assertEqual(r.status_code, 200, r.text)

    def test_max_line_chars_lays_text_out_on_two_lines(self):
        words = [
            _word("aaa", 0, 450),
            _word("bbb", 450, 900),
            _word("ccc", 900, 1350),
            _word("ddd", 1350, 1800),
        ]
        body = {
            "words": words,
            "max_chars": 42,
            "min_duration_ms": 0,
            "max_duration_ms": 7000,
            "target_duration_ms": 2000,
            "pause_threshold_ms": 500,
            "max_line_chars": 7,
        }
        r = self.client.post("/segment", json=body)
        self.assertEqual(r.status_code, 200, r.text)
        segs = r.json()["segments"]
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0]["text"], "aaa bbb\nccc ddd")
        for line in segs[0]["text"].split("\n"):
            self.assertLessEqual(len(line), 7)

    def test_max_line_chars_srt_uses_newline(self):
        words = [
            _word("aaa", 0, 450),
            _word("bbb", 450, 900),
            _word("ccc", 900, 1350),
            _word("ddd", 1350, 1800),
        ]
        body = {
            "words": words,
            "max_chars": 42,
            "min_duration_ms": 0,
            "max_duration_ms": 7000,
            "target_duration_ms": 2000,
            "pause_threshold_ms": 500,
            "max_line_chars": 7,
        }
        r = self.client.post("/segment/srt", json=body)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertIn("aaa bbb\nccc ddd\n", r.text)

    def test_422_when_line_width_makes_segmentation_impossible(self):
        words = [
            _word("aaaa", 0, 400),
            _word("bbbb", 400, 800),
            _word("cccc", 800, 1200),
        ]
        body = {
            "words": words,
            "max_chars": 42,
            "min_duration_ms": 1000,
            "max_duration_ms": 7000,
            "target_duration_ms": 2000,
            "pause_threshold_ms": 500,
            "max_line_chars": 4,
        }
        r = self.client.post("/segment", json=body)
        self.assertEqual(r.status_code, 422)
        self.assertIn("no legal segmentation", r.json()["detail"])

    def test_max_line_chars_must_be_positive(self):
        body = {**BODY, "max_line_chars": 0}
        r = self.client.post("/segment", json=body)
        self.assertEqual(r.status_code, 422)

    def test_required_break_after_forces_segment_edge(self):
        # word 0 lasts only 500 ms < the default 1000 ms minimum, so relax it
        # for this request.
        body = {**BODY, "min_duration_ms": 0, "required_break_after": [0]}
        r = self.client.post("/segment", json=body)
        self.assertEqual(r.status_code, 200, r.text)
        segs = r.json()["segments"]
        self.assertEqual(segs[0]["word_start"], 0)
        self.assertEqual(segs[0]["word_end"], 0)
        # coverage is preserved
        self.assertEqual(segs[-1]["word_end"], len(BODY["words"]) - 1)
        for a, b in zip(segs, segs[1:]):
            self.assertEqual(b["word_start"], a["word_end"] + 1)

    def test_keep_together_forbids_internal_edges(self):
        # The free optimum cuts after word 1 ("world."), which is the left
        # edge of [1, 3] -> words 1..3 must instead share one subtitle.
        body = {**BODY, "keep_together": [[1, 3]]}
        r = self.client.post("/segment", json=body)
        self.assertEqual(r.status_code, 200, r.text)
        segs = r.json()["segments"]
        span = next(
            s for s in segs if s["word_start"] <= 1 <= s["word_end"]
        )
        self.assertGreaterEqual(span["word_end"], 3)
        ends = [s["word_end"] for s in segs]
        self.assertNotIn(1, ends)
        self.assertNotIn(2, ends)

    def test_combined_edits(self):
        body = {
            **BODY,
            "min_duration_ms": 0,
            "required_break_after": [0, 6],
            "keep_together": [[2, 4]],
        }
        r = self.client.post("/segment", json=body)
        self.assertEqual(r.status_code, 200, r.text)
        segs = r.json()["segments"]
        ends = [s["word_end"] for s in segs]
        self.assertIn(0, ends)
        self.assertIn(6, ends)
        self.assertNotIn(2, ends)
        self.assertNotIn(3, ends)
        # words 2..4 share a subtitle
        self.assertTrue(
            any(s["word_start"] <= 2 and s["word_end"] >= 4 for s in segs)
        )

    def test_422_on_required_break_out_of_range(self):
        n = len(BODY["words"])
        body = {**BODY, "required_break_after": [n - 1]}  # no edge after last word
        r = self.client.post("/segment", json=body)
        self.assertEqual(r.status_code, 422)
        self.assertIn("required_break_after", r.json()["detail"])

    def test_422_on_keep_interval_too_small(self):
        body = {**BODY, "keep_together": [[2, 2]]}
        r = self.client.post("/segment", json=body)
        self.assertEqual(r.status_code, 422)
        self.assertIn("at least two", r.json()["detail"])

    def test_422_on_overlapping_keep_intervals(self):
        body = {**BODY, "keep_together": [[1, 4], [3, 6]]}
        r = self.client.post("/segment", json=body)
        self.assertEqual(r.status_code, 422)
        self.assertIn("overlap", r.json()["detail"])

    def test_422_on_break_conflicting_with_keep_interval(self):
        body = {
            **BODY,
            "required_break_after": [3],
            "keep_together": [[2, 5]],
        }
        r = self.client.post("/segment", json=body)
        self.assertEqual(r.status_code, 422)
        self.assertIn("required_break_after", r.json()["detail"])

    def test_422_when_edits_make_segmentation_impossible(self):
        # Mandatory first subtitle is a single 500 ms word < 1000 ms minimum.
        body = {**BODY, "required_break_after": [0]}
        r = self.client.post("/segment", json=body)
        self.assertEqual(r.status_code, 422)
        self.assertIn("no legal segmentation", r.json()["detail"])

    def test_422_on_malformed_keep_pair(self):
        body = {**BODY, "keep_together": [[1, 2, 3]]}
        r = self.client.post("/segment", json=body)
        self.assertEqual(r.status_code, 422)

    def test_422_on_negative_edit_index(self):
        body = {**BODY, "required_break_after": [-1]}
        r = self.client.post("/segment", json=body)
        self.assertEqual(r.status_code, 422)

    def test_srt_and_json_share_the_same_edited_result(self):
        params = {
            "min_duration_ms": 0,
            "required_break_after": [0, 6],
            "keep_together": [[2, 4]],
        }
        rj = self.client.post("/segment", json={**BODY, **params})
        rs = self.client.post("/segment/srt", json={**BODY, **params})
        self.assertEqual(rj.status_code, 200)
        self.assertEqual(rs.status_code, 200)
        segs = rj.json()["segments"]
        srt = rs.text
        # one numbered SRT block per JSON segment, same texts and timings
        blocks = [b for b in srt.strip().split("\n\n")]
        self.assertEqual(len(blocks), len(segs))
        for k, (block, seg) in enumerate(zip(blocks, segs), start=1):
            lines = block.split("\n")
            self.assertEqual(lines[0], str(k))
            self.assertIn(str(seg["text"]), block)
            self.assertIn(
                f"{format_timestamp(seg['start_ms'])} --> "
                f"{format_timestamp(seg['end_ms'])}",
                lines[1],
            )

    def test_omitting_edit_fields_keeps_old_behavior(self):
        r1 = self.client.post("/segment", json=BODY)
        r2 = self.client.post(
            "/segment",
            json={
                **BODY,
                "required_break_after": None,
                "keep_together": None,
            },
        )
        self.assertEqual(r1.status_code, 200)
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(
            r1.json()["segments"], r2.json()["segments"]
        )


class TestReflowAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    REFLOW_BODY = {
        **BODY,
        "min_duration_ms": 0,
        "deletions": [[1400, 5200]],
    }

    def test_reflow_json(self):
        r = self.client.post("/reflow", json=self.REFLOW_BODY)
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        segs = data["segments"]
        self.assertEqual(data["segment_count"], 2)
        # middle words dropped; original request indices are preserved
        self.assertEqual(segs[0]["word_start"], 0)
        self.assertEqual(segs[0]["word_end"], 1)
        self.assertEqual(segs[1]["word_start"], 7)
        self.assertEqual(segs[1]["word_end"], 8)
        # the second fragment is shifted earlier by 3800 ms
        self.assertEqual(segs[0]["start_ms"], 0)
        self.assertEqual(segs[0]["end_ms"], 1400)
        self.assertEqual(segs[1]["start_ms"], 1400)
        self.assertEqual(segs[1]["end_ms"], 2600)
        # the two sides of the cut never share a subtitle
        self.assertFalse(
            any(s["word_start"] <= 1 and s["word_end"] >= 7 for s in segs)
        )

    def test_reflow_srt(self):
        r = self.client.post("/reflow/srt", json=self.REFLOW_BODY)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertTrue(r.headers["content-type"].startswith("text/plain"))
        srt = r.text
        blocks = srt.strip().split("\n\n")
        self.assertEqual(len(blocks), 2)
        self.assertIn("00:00:00,000 --> 00:00:01,400", srt)
        self.assertIn("00:00:01,400 --> 00:00:02,600", srt)
        self.assertIn("Hello world.", srt)
        self.assertIn("Fine thanks.", srt)

    def test_reflow_srt_and_json_share_one_result(self):
        rj = self.client.post("/reflow", json=self.REFLOW_BODY)
        rs = self.client.post("/reflow/srt", json=self.REFLOW_BODY)
        self.assertEqual(rj.status_code, 200)
        self.assertEqual(rs.status_code, 200)
        blocks = rs.text.strip().split("\n\n")
        segs = rj.json()["segments"]
        self.assertEqual(len(blocks), len(segs))
        for k, (block, seg) in enumerate(zip(blocks, segs), start=1):
            lines = block.split("\n")
            self.assertEqual(lines[0], str(k))
            self.assertIn(seg["text"], block)
            self.assertIn(
                f"{format_timestamp(seg['start_ms'])} --> "
                f"{format_timestamp(seg['end_ms'])}",
                lines[1],
            )

    def test_422_when_word_crosses_deletion_boundary(self):
        # word 1 ("world.", 500..1400) is only partly covered.
        body = {**self.REFLOW_BODY, "deletions": [[1000, 5200]]}
        r = self.client.post("/reflow", json=body)
        self.assertEqual(r.status_code, 422)
        self.assertIn("intersects deletion range", r.json()["detail"])

    def test_422_on_unsorted_deletions(self):
        body = {**self.REFLOW_BODY, "deletions": [[5200, 5300], [1400, 1500]]}
        r = self.client.post("/reflow", json=body)
        self.assertEqual(r.status_code, 422)
        self.assertIn("ascending", r.json()["detail"])

    def test_422_on_overlapping_deletions(self):
        body = {**self.REFLOW_BODY, "deletions": [[1400, 3000], [2500, 5200]]}
        r = self.client.post("/reflow", json=body)
        self.assertEqual(r.status_code, 422)
        self.assertIn("overlap", r.json()["detail"])

    def test_422_when_start_ge_end(self):
        body = {**self.REFLOW_BODY, "deletions": [[1400, 1400]]}
        r = self.client.post("/reflow", json=body)
        self.assertEqual(r.status_code, 422)
        self.assertIn("start_ms < end_ms", r.json()["detail"])

    def test_422_when_all_words_deleted(self):
        body = {**self.REFLOW_BODY, "deletions": [[0, 6400]]}
        r = self.client.post("/reflow", json=body)
        self.assertEqual(r.status_code, 422)
        self.assertIn("every word", r.json()["detail"])

    def test_422_when_fragment_has_no_legal_segmentation(self):
        # Delete word 1 ("world.") and words 2..6 (the two ranges touch at
        # 1400 and merge).  The first surviving fragment is then "Hello"
        # alone at 500 ms, below the 1000 ms default minimum -> impossible.
        body = {**BODY, "deletions": [[500, 1400], [1400, 5200]]}
        r = self.client.post("/reflow", json=body)
        self.assertEqual(r.status_code, 422)
        self.assertIn("no legal segmentation", r.json()["detail"])

    def test_422_on_missing_deletions(self):
        body = {k: v for k, v in self.REFLOW_BODY.items() if k != "deletions"}
        r = self.client.post("/reflow", json=body)
        self.assertEqual(r.status_code, 422)

    def test_422_on_empty_deletions(self):
        r = self.client.post(
            "/reflow", json={**self.REFLOW_BODY, "deletions": []}
        )
        self.assertEqual(r.status_code, 422)

    def test_422_on_malformed_deletion_pair(self):
        r = self.client.post(
            "/reflow", json={**self.REFLOW_BODY, "deletions": [[1400]]}
        )
        self.assertEqual(r.status_code, 422)

    def test_422_on_negative_deletion_endpoint(self):
        r = self.client.post(
            "/reflow", json={**self.REFLOW_BODY, "deletions": [[-1, 5200]]}
        )
        self.assertEqual(r.status_code, 422)

    def test_max_line_chars_applies_to_reflow(self):
        words = [
            _word("aaa", 0, 450),
            _word("bbb", 450, 900),
            _word("ccc", 2000, 2450),
            _word("ddd", 2450, 2900),
        ]
        body = {
            "words": words,
            "deletions": [[900, 2000]],
            "max_chars": 42,
            "min_duration_ms": 0,
            "max_duration_ms": 7000,
            "target_duration_ms": 2000,
            "pause_threshold_ms": 500,
            "max_line_chars": 7,
        }
        r = self.client.post("/reflow", json=body)
        self.assertEqual(r.status_code, 200, r.text)
        for s in r.json()["segments"]:
            for line in s["text"].split("\n"):
                self.assertLessEqual(len(line), 7)


if __name__ == "__main__":
    unittest.main()
