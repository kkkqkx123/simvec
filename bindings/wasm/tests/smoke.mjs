import { test } from "node:test";
import assert from "node:assert/strict";

test("wasm binding requires a wasm-pack build", async () => {
  let Engine = null;
  try {
    ({ Engine } = await import("../src/index.js"));
  } catch (error) {
    assert.match(
      String(error),
      /pkg\/simvec/,
      "missing pkg should mention the wasm-pack output directory",
    );
    return;
  }
  const engine = await Engine.create();
  engine.createCollection("docs", { vector_size: 2, distance: "Dot" });
  engine.upsert("docs", "a", [1, 0], { color: "red" });
  engine.upsertBatch("docs", [{ id: "b", vector: [0, 1], payload: { color: "blue" } }]);
  assert.equal(engine.count("docs"), 2);
  assert.equal(engine.collectionConfig("docs").vector_size, 2);

  const results = engine.search("docs", { vector: [1, 0], limit: 10 });
  assert.equal(results[0].id, "a");
  const shortcut = engine.searchVector("docs", [1, 0], 10);
  assert.equal(shortcut[0].id, "a");

  engine.mergePayload("docs", "a", { size: "large" });
  engine.setPayloadField("docs", "b", "size", "small");
  assert.equal(engine.get("docs", "a").payload.size, "large");

  engine.createPayloadIndex("docs", "color", "keyword");
  assert.equal(engine.deletePayloadIndex("docs", "color"), true);

  const page = engine.scroll("docs", 1);
  assert.equal(page.points.length, 1);
  assert.ok(page.next_offset);

  assert.equal(engine.deleteByFilter("docs", { must: [] }), 0);
  engine.deleteBatch("docs", ["b"]);
  assert.equal(engine.count("docs"), 1);

  engine.applyTxn([{ op: "upsert", collection: "docs", point: { id: "c", vector: [1, 0] } }]);
  assert.equal(engine.count("docs"), 2);

  assert.equal(engine.buildIndex("docs"), false);
  assert.equal(engine.hasIndex("docs"), false);
  assert.equal(engine.buildQuantization("docs"), false);
  assert.equal(engine.hasQuantization("docs"), false);
  assert.equal(engine.compact("docs"), 2);
  engine.drainPending("docs");
  engine.runMaintenanceSweep();
  assert.ok(engine.collectionMetrics("docs"));

  const snapshot = engine.exportSnapshot();
  const restored = await Engine.create();
  restored.importSnapshot(snapshot);
  assert.equal(restored.count("docs"), 2);

  const before = engine.revision();
  engine.upsert("docs", "rev-check", [1, 0]);
  assert.ok(engine.revision() > before);
  const entry = engine.exportCollection("docs");
  assert.equal(entry.name, "docs");
  const shard = await Engine.create();
  shard.createCollection("docs", { vector_size: 2, distance: "Dot" });
  shard.importCollection(entry);
  assert.equal(shard.count("docs"), engine.count("docs"));
  shard.close();

  assert.throws(() => restored.importSnapshot({ version: 999, collections: [] }), /\[simvec:16\]/);
  assert.throws(() => restored.importSnapshot("not json"), /\[simvec:20\]/);
  assert.equal(restored.count("docs"), 2);

  const { saveSnapshotToOpfs, loadSnapshotToOpfs } = await import("../src/persist.js");
  await assert.rejects(saveSnapshotToOpfs(engine), /OPFS/);
  await assert.rejects(loadSnapshotToOpfs(engine), /OPFS/);

  assert.throws(() => engine.get("docs", "missing"), /\[simvec:12\]/);
  engine.close();
  engine.close();
  assert.equal(engine.closed, true);
  assert.throws(() => engine.count("docs"), /\[simvec:20\]/);
  restored.close();
  restored.close();
  assert.equal(restored.closed, true);
});
