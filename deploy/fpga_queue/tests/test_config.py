"""Pool config validation — the checks that stop destructive misconfiguration."""

import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from fq.config import ConfigError, load_pool


def write(d: dict) -> str:
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(d, f)
    f.close()
    return f.name


def pool(lanes, **kw):
    base = {"state_dir": tempfile.mkdtemp(), "backend": "mock",
            "lanes": lanes}
    base.update(kw)
    return write(base)


class TestValidation(unittest.TestCase):
    def test_minimal_hosts_pool_loads(self):
        cfg = load_pool(pool([{"name": "a", "hosts": ["h1"]}]))
        self.assertEqual(len(cfg.lanes), 1)
        self.assertEqual(cfg.lanes[0].mode, "hosts")
        self.assertEqual(cfg.lanes[0].capacity, 1)

    def test_shared_run_farm_tag_is_rejected(self):
        """Two lanes on one tag lets one job terminate another's instances."""
        with self.assertRaises(ConfigError) as cm:
            load_pool(pool([
                {"name": "a", "mode": "tagged", "tag": "shared"},
                {"name": "b", "mode": "tagged", "tag": "shared"},
            ]))
        self.assertIn("share run_farm_tag", str(cm.exception))

    def test_shared_host_is_rejected(self):
        """`firesim kill` pkills host-wide and infrasetup reflashes every
        slot, so two lanes may never name the same host."""
        with self.assertRaises(ConfigError) as cm:
            load_pool(pool([
                {"name": "a", "hosts": ["192.168.0.8"]},
                {"name": "b", "hosts": ["192.168.0.8", "192.168.0.9"]},
            ]))
        self.assertIn("must belong to exactly one", str(cm.exception))

    def test_forbidden_tag_is_rejected(self):
        with self.assertRaises(ConfigError):
            load_pool(pool([{"name": "a", "mode": "tagged",
                             "tag": "mainrunfarm"}]))

    def test_hosts_mode_requires_hosts(self):
        with self.assertRaises(ConfigError) as cm:
            load_pool(pool([{"name": "a", "mode": "hosts"}]))
        self.assertIn("lists no 'hosts'", str(cm.exception))

    def test_tagged_mode_requires_tag(self):
        with self.assertRaises(ConfigError):
            load_pool(pool([{"name": "a", "mode": "tagged"}]))

    def test_manage_hosts_requires_tagged_mode(self):
        """ExternallyProvisioned physically cannot launch or terminate."""
        with self.assertRaises(ConfigError) as cm:
            load_pool(pool([{"name": "a", "hosts": ["h"],
                             "manage_hosts": True}]))
        self.assertIn("cannot launch or terminate", str(cm.exception))

    def test_duplicate_lane_name_rejected(self):
        with self.assertRaises(ConfigError):
            load_pool(pool([{"name": "a", "hosts": ["h1"]},
                            {"name": "a", "hosts": ["h2"]}]))

    def test_quota_overcommit_rejected(self):
        with self.assertRaises(ConfigError) as cm:
            load_pool(pool([{"name": f"l{i}", "hosts": [f"h{i}"]}
                            for i in range(20)], fpga_quota=16))
        self.assertIn("fpga_quota", str(cm.exception))

    def test_bad_reclaim_policy_rejected(self):
        with self.assertRaises(ConfigError):
            load_pool(pool([{"name": "a", "hosts": ["h"]}],
                           reclaim={"policy": "yolo"}))

    def test_capacity_defaults_to_host_count(self):
        cfg = load_pool(pool([{"name": "a", "hosts": ["h1", "h2", "h3"]}]))
        self.assertEqual(cfg.lanes[0].capacity, 3)

    def test_shipped_example_pool_is_valid(self):
        example = (pathlib.Path(__file__).resolve().parent.parent
                   / "examples" / "pool.example.yaml")
        cfg = load_pool(example)
        self.assertEqual(len(cfg.lanes), 8)
        # Eight independent lanes, one host each, no tags involved.
        self.assertTrue(all(l.mode == "hosts" for l in cfg.lanes))
        self.assertTrue(all(l.tag is None for l in cfg.lanes))
        hosts = [h for l in cfg.lanes for h in l.hosts]
        self.assertEqual(len(hosts), len(set(hosts)))


if __name__ == "__main__":
    unittest.main()
