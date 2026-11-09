import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { AlreadyExistsError, Engine, NotFoundError } from "../src/index.js";

describe("simvec js binding smoke", () => {
  let root;
  let engine;

  before(() => {
    root = fs.mkdtempSync(path.join(os.tmpdir(), "simvec-js-"));
    engine = new Engine(root);
  });

  after(() => {
    engine?.close();
    fs.rmSync(root, { recursive: true, force: true });
  });

  it("covers collection lifecycle, writes, reads and maintenance", () => {
    const config = { vector_size: 4, distance: "Cosine", index: "Flat" };
    engine.createCollection("docs", config);
    assert.equal(engine.collectionExists("docs"), true);
    assert.deepEqual(engine.listCollections(), ["docs"]);
    assert.throws(() => engine.createCollection("docs", config), AlreadyExistsError);

    engine.upsert("docs", "1", [1, 0, 0, 0], { title: "hello" });
    engine.upsertBatch("docs", [
      { id: "2", vector: [0, 1, 0, 0], payload: { title: "world" } },
    ]);
    assert.equal(engine.count("docs"), 2);

    const hits = engine.searchVector("docs", [1, 0, 0, 0], 5);
    assert.equal(hits.length, 2);
    assert.equal(String(hits[0].id), "1");
    assert.equal(hits[0].payload.title, "hello");

    const queryHits = engine.search("docs", { vector: [1, 0, 0, 0], limit: 5 });
    assert.equal(queryHits.length, 2);

    const point = engine.get("docs", "1");
    assert.deepEqual(point.vector, [1, 0, 0, 0]);

    engine.setPayload("docs", "1", { title: "hello", lang: "en" });
    assert.equal(engine.get("docs", "1").payload.lang, "en");
    engine.mergePayload("docs", "1", { topic: "intro" });
    assert.equal(engine.get("docs", "1").payload.topic, "intro");
    engine.deletePayloadKeys("docs", "1", ["topic"]);
    assert.equal(engine.get("docs", "1").payload.topic, undefined);

    const page = engine.scroll("docs", 10);
    assert.equal(page.points.length, 2);
    const tail = engine.scroll("docs", 10, page.next_offset);
    assert.equal(tail.points.length, 0);

    const info = engine.collectionInfo("docs");
    assert.equal(info.vector_count, 2);
    assert.equal(engine.collectionConfig("docs").vector_size, 4);
    assert.equal(typeof engine.collectionMetrics("docs"), "object");

    engine.createPayloadIndex("docs", "title", "keyword");
    assert.deepEqual(engine.listPayloadIndexes("docs"), [{ field: "title", schema: "keyword" }]);
    assert.equal(engine.deletePayloadIndex("docs", "title"), true);
    assert.equal(engine.deletePayloadIndex("docs", "title"), false);

    assert.equal(engine.hasIndex("docs"), false);
    assert.equal(engine.hasQuantization("docs"), false);
    engine.buildIndex("docs");
    engine.buildQuantization("docs");
    engine.drainPending("docs");
    engine.dropIndex("docs");
    engine.runMaintenanceSweep();

    engine.delete("docs", "2");
    assert.equal(engine.count("docs"), 1);
    assert.throws(() => engine.get("docs", "2"), NotFoundError);

    engine.deleteCollection("docs");
    assert.equal(engine.collectionExists("docs"), false);
  });

  it("covers batch deletes, filtered deletes and compaction", () => {
    engine.createCollection("batch", { vector_size: 3, distance: "Cosine", index: "Flat" });
    engine.upsertBatch("batch", [
      { id: "a", vector: [1, 0, 0], payload: { color: "red" } },
      { id: "b", vector: [0, 1, 0], payload: { color: "blue" } },
      { id: "c", vector: [0, 0, 1], payload: { color: "red" } },
    ]);
    assert.equal(engine.count("batch"), 3);

    const deleted = engine.deleteByFilter("batch", {
      must: [{ field: "color", condition: { Match: { value: "red" } } }],
    });
    assert.equal(deleted, 2);
    assert.equal(engine.count("batch"), 1);

    engine.deleteBatch("batch", ["b"]);
    assert.equal(engine.count("batch"), 0);
    assert.equal(engine.compact("batch"), 0);

    engine.deleteCollection("batch");
    assert.throws(() => engine.get("batch", "b"), NotFoundError);
  });
});
