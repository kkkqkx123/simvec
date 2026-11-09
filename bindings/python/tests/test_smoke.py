"""Smoke coverage for the ctypes binding against a fresh engine."""

import json
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

                page = engine.scroll("docs", 10)
                self.assertEqual(len(page["points"]), 2)
                self.assertEqual(page["next_offset"], "2")
                tail = engine.scroll("docs", 10, offset=page["next_offset"])
                self.assertEqual(tail["points"], [])

                info = engine.collection_info("docs")
                self.assertEqual(info["vector_count"], 2)
                self.assertEqual(engine.collection_config("docs")["vector_size"], 4)
                self.assertIsInstance(engine.collection_metrics("docs"), dict)

                engine.delete("docs", "2")
                self.assertEqual(engine.count("docs"), 1)
                with self.assertRaises(NotFoundError):
                    engine.get("docs", "2")

                engine.delete_collection("docs")
                self.assertFalse(engine.collection_exists("docs"))


if __name__ == "__main__":
    unittest.main()
