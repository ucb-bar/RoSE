"""Completion contract: exit (fast path) -> sentinel -> timeout (backstop)."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from fq import completion


class TestPolicy(unittest.TestCase):
    def test_default_is_exit(self):
        """A well-behaved guest ends the sim itself (Zephyr sys_reboot ->
        HTIF exit -> runworkload returns). That is the expected path."""
        p = completion.from_spec({}, default_timeout_s=600)
        self.assertEqual(p.mode, "exit")
        self.assertFalse(p.watches_uart)
        self.assertEqual(p.timeout_s, 600)
        self.assertFalse(p.timeout_is_success)

    def test_sentinel_is_inferred_from_a_regex(self):
        p = completion.from_spec({"done_regex": "Hello World"})
        self.assertEqual(p.mode, "sentinel")
        self.assertTrue(p.watches_uart)

    def test_structured_block_wins_over_flat_fields(self):
        p = completion.from_spec({
            "done_regex": "flat",
            "completion": {"mode": "sentinel", "sentinel_regex": "structured",
                           "timeout_s": 42}})
        self.assertEqual(p.sentinel_regex, "structured")
        self.assertEqual(p.timeout_s, 42)

    def test_timeout_mode_makes_expiry_a_success(self):
        p = completion.from_spec({"completion": {"mode": "timeout",
                                                 "timeout_s": 300}})
        self.assertTrue(p.timeout_is_success)

    def test_exit_mode_does_not_poll_uart(self):
        """No sentinel => no ssh round trip every poll."""
        self.assertFalse(completion.from_spec(
            {"completion": {"mode": "exit"}}).watches_uart)

    def test_fail_regex_alone_still_watches(self):
        p = completion.from_spec({"fail_regex": "PANIC"})
        self.assertTrue(p.watches_uart)


class TestValidation(unittest.TestCase):
    def test_bad_mode(self):
        errs = completion.validate({"completion": {"mode": "whenever"}})
        self.assertTrue(any("completion.mode" in e for e in errs))

    def test_sentinel_mode_needs_a_regex(self):
        errs = completion.validate({"completion": {"mode": "sentinel"}})
        self.assertTrue(any("needs a sentinel_regex" in e for e in errs))

    def test_bad_regex_is_caught_at_submit(self):
        errs = completion.validate({"done_regex": "([unclosed"})
        self.assertTrue(any("not a valid regex" in e for e in errs))

    def test_good_spec_is_clean(self):
        self.assertEqual(completion.validate(
            {"completion": {"mode": "sentinel",
                            "sentinel_regex": "done"}}), [])


class TestAdvisory(unittest.TestCase):
    def test_warns_when_only_the_timeout_can_end_the_job(self):
        msg = completion.advisory({}, 3600)
        self.assertIsNotNone(msg)
        self.assertIn("sys_reboot", msg)
        self.assertIn("3600", msg)

    def test_no_warning_with_a_sentinel(self):
        self.assertIsNone(completion.advisory({"done_regex": "ok"}, 3600))


if __name__ == "__main__":
    unittest.main()
