"""Smoke coverage for the ctypes binding against a fresh engine."""

import tempfile
import unittest

from simvec import Engine, NotFoundError


class BindingSmokeTest(unittest.TestCase):
    def test_lifecycle(self):
        with tempfile.TemporaryDirectory() as root:
            with Engine(root) as engine:
                config = {"vector_size": 4, "distance": "Cosine", "index": "Flat"}
                engine.create_collection("docs", config)
                self.assertTrue(engine.collection_exists("docs"))
                self.assertEqual(engine.list_collections(), ["docs"])

                engine.upsert("docs", "1", [1.0, 0.0, 0.0, 0.0], {"title": "hello"})
                engine.upsert_batch(
                    "docs",
                    [{"id": "2", "vector": [0.0, 1.0, 0.0, 0.0], "payload": {"title": "world"}}],
                )
                self.assertEqual(engine.count("docs"), 2)
                self.assertEqual(engine.last_code, 0)

                hits = engine.search_vector("docs", [1.0, 0.0, 0.0, 0.0], 5)
                self.assertEqual(len(hits), 2)
                self.assertEqual(str(hits[0]["id"]), "1")
                self.assertEqual(hits[0]["payload"]["title"], "hello")

                hits = engine.search(
                    "docs", {"vector": [1.0, 0.0, 0.0, 0.0], "limit": 5}
                )
                self.assertEqual(len(hits), 2)

                point = engine.get("docs", "1")
                self.assertEqual(point["vector"], [1.0, 0.0, 0.0, 0.0])

                engine.merge_payload("docs", "1", {"lang": "en"})
                point = engine.get("docs", "1")
                self.assertEqual(point["payload"]["lang"], "en")
                engine.set_payload_field("docs", "1", "topic", "intro")
                self.assertEqual(engine.get("docs", "1")["payload"]["topic"], "intro")
                engine.delete_payload_keys("docs", "1", ["topic"])
                self.assertNotIn("topic", engine.get("docs", "1")["payload"])

                engine.create_payload_index("docs", "title", "keyword")
                self.assertEqual(
                    engine.list_payload_indexes("docs"),
                    [{"field": "title", "schema": "keyword"}],
                )
                self.assertTrue(engine.delete_payload_index("docs", "title"))
                self.assertFalse(engine.delete_payload_index("docs", "title"))

                page = engine.scroll("docs", 10)
                self.assertEqual(len(page["points"]), 2)
                self.assertEqual(page["next_offset"], "2")
                tail = engine.scroll("docs", 10, offset=page["next_offset"])
                self.assertEqual(tail["points"], [])

                info = engine.collection_info("docs")
                self.assertEqual(info["vector_count"], 2)
                self.assertEqual(engine.collection_config("docs")["vector_size"], 4)
                self.assertIsInstance(engine.collection_metrics("docs"), dict)

                self.assertFalse(engine.has_index("docs"))
                self.assertFalse(engine.has_quantization("docs"))
                engine.build_index("docs")
                engine.build_quantization("docs")
                engine.drain_pending("docs")
                engine.drop_published_index("docs")
                engine.drop_index("docs")
                engine.run_maintenance_sweep()

                engine.delete("docs", "2")
                self.assertEqual(engine.count("docs"), 1)
                with self.assertRaises(NotFoundError):
                    engine.get("docs", "2")
                self.assertNotEqual(engine.last_code, 0)
                self.assertTrue(engine.last_message)

                engine.delete_collection("docs")
                self.assertFalse(engine.collection_exists("docs"))

    def test_batch_and_filtered_deletes(self):
        with tempfile.TemporaryDirectory() as root:
            with Engine(root) as engine:
                engine.create_collection("batch", {"vector_size": 3, "distance": "Cosine", "index": "Flat"})
                engine.upsert_batch(
                    "batch",
                    [
                        {"id": "a", "vector": [1, 0, 0], "payload": {"color": "red"}},
                        {"id": "b", "vector": [0, 1, 0], "payload": {"color": "blue"}},
                        {"id": "c", "vector": [0, 0, 1], "payload": {"color": "red"}},
                    ],
                )
                self.assertEqual(engine.count("batch"), 3)

                engine.apply_txn(
                    [
                        {"op": "upsert", "collection": "batch",
                         "point": {"id": "d", "vector": [1, 1, 0]}},
                        {"op": "delete", "collection": "batch", "id": "d"},
                    ]
                )
                self.assertEqual(engine.count("batch"), 3)

                deleted = engine.delete_by_filter(
                    "batch", {"must": [{"field": "color", "condition": {"Match": {"value": "red"}}}]}
                )
                self.assertEqual(deleted, 2)
                self.assertEqual(engine.count("batch"), 1)

                engine.delete_batch("batch", ["b"])
                self.assertEqual(engine.count("batch"), 0)
                engine.compact("batch")

                engine.delete_collection("batch")
                with self.assertRaises(NotFoundError):
                    engine.get("batch", "b")

    def test_promote_and_defaults(self):
        with tempfile.TemporaryDirectory() as root:
            with Engine(root) as engine:
                engine.create_collection("live", {"vector_size": 2, "distance": "Cosine", "index": "Flat"})
                engine.upsert("live", "1", [1.0, 0.0])
                engine.create_collection("live.rebuild-7", {"vector_size": 2, "distance": "Cosine", "index": "Flat"})
                engine.upsert("live.rebuild-7", "2", [0.0, 1.0])
                self.assertIn("live.rebuild-7", engine.list_temp_collections())

                engine.promote_temp_collection("live", "live.rebuild-7", ".old-7")
                self.assertEqual(engine.get("live", "2")["vector"], [0.0, 1.0])
                self.assertEqual(engine.list_old_backups("live"), ["live.old-7"])
                backup = engine.restore_promote_backup("live")
                self.assertEqual(backup, "live.old-7")
                self.assertEqual(engine.get("live", "1")["vector"], [1.0, 0.0])

                self.assertIsNone(engine.get_default_hnsw_config())
                self.assertIsNone(engine.get_default_ivf_config())
                self.assertIsNone(engine.get_default_quantization_config())
                engine.set_default_hnsw_config({"m": 8, "ef_construct": 32})
                engine.set_default_ivf_config({
                    "lists": None,
                    "min_build_points": 1000,
                    "sample_limit": 1024,
                    "kmeans_max_iter": 10,
                    "drift_threshold": 0.1,
                    "drift_check_interval": 1000,
                    "default_nprobe": 4,
                    "auto_promotion": False,
                    "max_probes": None,
                })
                engine.set_default_quantization_config(
                    {"quant_type": {"Binary": {"always_ram": True}}}
                )
                self.assertEqual(engine.get_default_hnsw_config()["m"], 8)
                self.assertIsNotNone(engine.get_default_ivf_config())
                self.assertIsNotNone(engine.get_default_quantization_config())


if __name__ == "__main__":
    unittest.main()
