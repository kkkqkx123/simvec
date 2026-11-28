# 延后存储重构收尾：任务清单与修改方案

本文承接 `deferred-storage-refactor.md`（七阶段方案）与 `post-refactor-followups.md`（阶段八设计），把**逐行核对后仍然成立的缺口**整理成本轮可执行清单。本文是临时执行清单，不作为项目事实源；落地后凡仍然成立的事实写回 `AGENTS.md`，完成项从本文删除。

核对基线：`342ba6d`。缺口识别以静态阅读为主；T1–T8 落地后已跑 `cargo check`（lib / tests / c_api / CLI）与 `cargo test`（单元 + 集成），结果见 §7。未跑 bench、语言绑定与 wasm 目标。

## 一、缺口清单

| ID | 优先级 | 缺口 | 证据 | 本轮状态 |
| --- | --- | --- | --- | --- |
| T1 | P1 | `bitmask.bin` 无校验，位翻转只能靠上层重建兜底 | `payload_store/bitmask.rs` 写 `magic/version/几何/位图`，无 CRC | **已落地** §2.1 |
| T2 | P1 | id tracker 无 LSN 绑定，无门刷盘阶段发布存在丢数据窗口 | `store_maintenance.rs` 不关队列门批量刷盘时即 `write_id_tracker`；`IdTracker` 只有 `{generation, entries}` | **已落地** §2.2 |
| T3 | P2 | `select_for_compaction` 在单段集合会选中 appendable 尾段，与注释和 `AGENTS.md` 矛盾 | `segments.rs` 过滤条件 `e.start != appendable \|\| entries.len() == 1` | **已落地（改文档、改注释，不改行为）** §2.3 |
| T4 | P2 | `segments.json` 条目缺 capacity/applied lsn，manifest 缺 generation，`persisted_lsn` 由全局 watermark 统一盖章 | `SegmentEntry` 五个字段；`derive_entries` 单值入参 | **已落地** §2.4 |
| T5 | P2 | `wal/blobs.bin` 体积不可观测（只有 WAL 记录字节进 checkpoint 触发） | `wal.rs::appended_bytes` 不统计 blob；`MaintenanceStatus` 无该字段 | **已落地** §2.5 |
| T6 | P3 | 依赖与死代码残留 | `crc32fast` 常规依赖只服务于一个无效单测；`WalBlobs::path` `#[allow(dead_code)]`；`TombstoneBits::load` 仅单测；`Wal::truncate` 注释仍称 compaction 在用 | **已落地** §2.6 |
| T7 | P2 | named 空间 ANN 契约未声明：仅 HNSW、无量化，IVF 静默降级 exact | `space_ann.rs::hnsw_config` 只认 HNSW；`quant: None` | 文档化 §2.7；是否在 create 时拒绝 IVF 仍待定（见 §4 G5 一节的配置校验项） |
| T8 | P2 | `AGENTS.md` 三处过期断言 | L34 "until physical segment files exist"、L35 "named/sparse 只走 exact scan"、尾段规则 | **已落地** §2.7 |
| T9 | P0 | 压实仍整代重写并全局重排 slot；slot 由 `next_slot` 单调分配，无 free list；blob 集中在 `wal/blobs.bin` 且永不回收 | `compaction.rs::plan_slots` + `meta.next_slot = plan.live_count`；`store_write.rs::resolve_upsert_slot_locked`；`wal_blobs.rs` 仅 `Wal::truncate` 复位 | **未落地**，设计见 §3 |
| T10 | P2 | 阶段八验证矩阵未开始 | `post-refactor-followups.md` 的 G1–G5 对应测试均不存在 | **部分落地**：G2 三项已补，G1/G3/G4/G5 见 §4 |

## 二、本轮已落地项的修改方案

### 2.1 T1 — `bitmask.bin` 统一到 CRC32C

格式（版本 1；读到其他版本一律 `CorruptData`）：

```text
[0..4)   magic        "VBMP"
[4..8)   version      u32 = 1
[8..16)  total_blocks u64
[16..20) block_size   u32
[20..24) page_blocks  u32
[24..28) crc32c       u32，覆盖 [0..24) + 位图字节
[28..)   bitmap       ceil(total_blocks/8) 字节
```

打开：版本或 CRC 不匹配即 `CorruptData`；`MmapBitmask::open_or_rebuild` 已有的"打开失败 → 按 tracker 指针重建并重写"路径承担恢复，因此位图仍属派生数据，**不需要**格式迁移。位图长度必须与 `ceil(total_blocks/8)` 严格相等，截断/多写都判损坏。

不变量：位图损坏不得导致两个 live payload 共用块；重建后必须与 tracker 指针自洽（沿用 `needs_rebuild`）。

### 2.2 T2 — id tracker 绑定 LSN，改为 phase-2 发布

问题窗口：checkpoint 的批量刷盘不关队列门（这是缩短停写窗口的设计），`flush_segment` 在此阶段无条件重写 `id_tracker.bin`，其内容取自**当前**内存态。掉电后重开时，tracker 里 `point_lsn` 已推进到新值，但该点的 vectors/payload 字节未 fsync，`stale_for_id` 会跳过重放记录，点以旧/空字节复活。

方案：

1. `IdTracker` 增加 `lsn`：该 tracker 内容对应的**段级 applied 前沿**。
2. `segments::write_id_tracker` 只写 `segments/<start>/id_tracker.staged`（同信封、同 fsync），不覆盖已发布文件。
3. `segments::publish_id_trackers(dir, segment_slots, count)` 在队列门内、re-flush 之后、`meta.bin` 之前统一 rename `.staged → id_tracker.bin` 并 fsync 段目录。
4. `read_id_trackers` 读已发布文件，并用 `tracker.lsn` 给 `point_lsn` 封顶（`min(point_lsn, lsn)`），`delete_lsn` 保持原值。

不变量：**tracker 一旦可见，该段所有字节已经 durable**。崩溃点：

- 批量刷盘后崩：只有 `.staged`，open 忽略；已发布 tracker 仍是上一轮的一致状态，replay 从旧 floor 补齐。
- publish 后、`meta.bin`/generation lsn 前崩：tracker 已一致且数据已 durable，replay 从旧 floor 重放，`stale_for_id` 跳过重复应用，结果幂等。
- 部分段 publish 完成：每段的可见状态都自洽，`persisted` 由 `min_persisted` 决定，ACK 不会越过未刷新的段。

代价：每段多一次 rename + 目录 fsync，仍在队列门内，但不含大文件 fsync。

### 2.3 T3 — 尾段选择规则（结论：行为正确，注释与文档有错）

初稿曾计划删掉 `|| entries.len() == 1` 例外，实测被三个用例否决：
`storage::tests::test_upsert_then_delete_then_reinsert`、
`storage::queue::tests::compaction_with_a_non_empty_queue_keeps_every_transaction`、
`engine::tests::test_deletes_schedule_background_compaction` —— 单段集合删除后垃圾比很高，
若尾段不可选则 `needs_compaction()` 永远为假，后台压实再也不会触发，小集合的垃圾无法回收。

因此结论反转：**代码行为保留**，修正点改为"注释与文档说错了"。函数注释和 `AGENTS.md` 都写了
"appendable tail is never selected"，实际规则是"前面还有冻结段时排除尾段；集合只有尾段时可选"。
本轮修的是这两处措辞，并补一个同时覆盖两种情形的单测
（`the_appendable_tail_is_selected_only_as_the_sole_segment`）。

### 2.4 T4 — `segments.json` 条目补齐

- `Manifest { version, generation, segment_slots, next_slot, segments }`
- `SegmentEntry { start, end, capacity, live_count, tombstone_count, applied_lsn, persisted_lsn }`
- `derive_entries(next_slot, segment_slots, is_tombstoned, watermarks)`，`watermarks: impl Fn(usize) -> (u64, u64)` 返回 `(applied, persisted)`；compaction 用 `(watermark, watermark)`。
- `read_entries(dir, generation, segment_slots, next_slot)`：generation/几何/前沿任一不匹配即视为过期（返回空，由调用方回落到 generation floor）。

**不做**每段 `文件版本/长度/CRC`：CRC 需要全量扫描段文件，而快照侧本来就逐文件计算 len/CRC；段级文件版本的收益要等 T9 的段级压实（段文件在两次压实之间不再变化）才成立。此项随 T9 一起补，避免现在写一份没人消费的元数据。

### 2.5 T5 — blob 体积可观测

- `WalBlobs::len()`（`blobs.bin` 当前长度）/ `Wal::blob_bytes()`。
- `MaintenanceStatus` 增加 `wal_blob_bytes`，经既有 JSON 通道暴露给 C ABI 与绑定。

**范围声明**：本项只让增长可见，不回收。回收方式取决于 T9 的 blob 归属决策（每段 `blobs.bin`），在此之前不要给 blob 增加 checkpoint 触发权重——checkpoint 不回收 blob，加重触发只会放大 I/O。

### 2.6 T6 — 依赖与死代码

- 删除 `crc32fast`，**不移到 dev-dependencies**。它唯一的消费者是 `snapshot.rs` 的
  `legacy_crc32_envelope_is_rejected`，而该测试是无效的：它把 magic 写成 `SVSP`，
  真实常量是 `SVPC`，`read` 在 magic 分支就返回 `None`，从未走到 CRC 比较，所以
  "旧 CRC32 会被拒绝"从未被真正断言。CRC 不匹配这条分支已由
  `storage::tests::test_corrupt_dtype_header_is_rejected` 覆盖（正确信封 + 翻转 body
  一个字节 → `read_strict` 返回 `CorruptData`），删掉旧测试不损失任何覆盖。
  依赖图与两个 `Cargo.lock` 中的 `crc32fast` 条目一并消失。
- 删除 `WalBlobs::path`（含 `#[allow(dead_code)]`）。
- `TombstoneBits::load` / `save` 保留：`load_segments` 与 `save_segment` 内部就调用它们，不是整代遗留加载器，不能收为 `#[cfg(test)]`（登记于 `storage-remediation-tasks.md` C3）。
- `Wal::truncate` 注释改为"仅测试/整库重建辅助，生产 checkpoint 与压实都走 `truncate_through`"。

### 2.7 T7 / T8 — 文档事实修正

`AGENTS.md`：

- L34："compaction 仍整代重写…until physical segment files exist" → 物理段文件已经存在，前提句作废；改为如实描述"整代重写 + 全局重排 slot"，并指向本文 T9。
- L34："the appendable tail is never selected on its own" → 与代码一致（单段也不选）。
- L35："ANN and quantization apply only to the single default dense space — named/sparse spaces use exact scan" → 改为"named dense 空间有 per-space HNSW（`spaces/<space>/ann/hnsw.bin`），IVF 与量化未实现，配置为 IVF 时按 exact scan 处理；sparse 用 posting + exact，multi 用 MaxSim"。
- 磁盘布局补一句：量化码仍在 generation 级 `quant.bin`（内存按段 mmap），`quant_seg_<start>.bin` 是派生副本。
- 维护章节记录 T5 新增的 `wal_blob_bytes` 与 T2 的 tracker 发布顺序。

`src/storage/wal.rs`、`src/storage/segments.rs` 的模块注释同步到物理段 + tracker 现状。

## 三、T9 — 段级压实、free list 与 blob 归属（未落地）

### 3.1 目标与不变量

- 段 `start` 与 `capacity` 恒定；压实只重排**被选中段内**的 live 点到最小 local offset，后续段 slot 范围不动。
- 一个 id 在一个 generation 内只有一个权威段归属；该归属由段 tracker 的 `offset` 决定。
- 干净段不进重写路径：先 flush，再按字节复制（或先复制后 flush 再校验），不重排 slot、不重建 payload 页、不重建 ANN。
- 旧段文件在读者 guard 归零前不得删除。

### 3.2 slot 分配改为 free list

- tracker 持久化 `free_offsets: Vec<u32>`（升序）。分配取**最小可用 offset**；live 与 replay 走同一函数，保证日志顺序与槽位一致。
- 新 id 进 appendable 尾段；删除后重新 upsert 在原段内复用 free offset（`delete_lsn` 保留，用于拒绝旧记录复活）。
- `meta.next_slot` 降级为"尾段高水位"，不再是全局分配器；段级容量由 `segment_slots` 与段数决定。

### 3.3 压实流程

1. 规划：按垃圾比选段（排除尾段），生成"段 → 段内重排映射"。
2. staging：未选中段逐文件复制（vectors/keys/tombstones/payloads_store/ANN/quant），选中段重排 live 点并重建其内部文件；`segments.json` 记录每段文件版本/长度/CRC。
3. 发布：generation manifest → 切换内存句柄 → 旧 guard 归零后删除旧目录。失败序列与现状一致（发布后失败进 poison）。

### 3.4 blob 归属与回收

- `segments/<start>/blobs.bin`，引用含 `segment_start`，避免压实后旧引用误指新内容。
- 压实只复制 live blob；被覆盖/删除的 blob 随段重写回收。
- 每段 blob 的回收空间计入 checkpoint 触发与 `wal_blob_bytes` 的同口径统计。

### 3.5 验收

- 持续写入下反复压实 + 重开，逐点比较 id→slot、payload、filter、搜索与 scroll。
- 持有旧 guard 时确认旧段文件未被提前删除。
- 只选中一段时，未选中段的文件 mtime/inode 不变（字节复制或直接复用），ANN 不重建。
- 破坏单段 ANN/量化文件，其它段仍走 ANN，坏段回落 exact。

### 3.6 顺序与风险

T9 必须一次做完"段级压实 + free list"，只做一半会出现"slot 稳定但分配仍单调"或"分配可复用但压实重排"的自相矛盾状态。风险集中在 generation 发布与 poison 序列，落地时必须跑完整 `cargo test -p simvec` 与 `tests/recovery_test.rs`，并按 §3.5 补崩溃注入。

## 四、T10 — 阶段八验证矩阵（未落地）

沿用 `post-refactor-followups.md` 的编号，补齐可回归用例：

1. **G1 绑定端到端**：python/js/go/java/r 的 smoke 各加 `maintenance_status` + 四个 snapshot API 的调用链；`src/c_api/tests.rs` 补 `maintenance_status_c_abi` 断言 `segments` 数组。
2. **G2 快照拒绝**：`test_snapshot_verification_rejects_missing_file`、`test_snapshot_rejects_existing_target`、`test_snapshot_verification_rejects_wrong_version`。
3. **G3 WAL/blob 崩溃注入**：blob fsync 前/后、WAL 后 apply 前、ACK 后 truncate 中四窗口；连同本轮 T2 新增的"批量刷盘后崩 / publish 后崩"两窗口。
4. **G4 性能基线**：`wal_bytes_vs_checkpoint` quick 档；注意 `benches/quick.rs` 是 `autobenches = false` 的共享助手模块，新档必须落到具体 bench 文件（原文档的落点写错了）。
5. **G5 Turbo 配置拒绝**：`VectorDataType::Turbo` 在 create 时报 `InvalidConfig`。

## 五、本轮不改变的不变量

- generation manifest 仍是唯一提交点；发布后失败进 poison，等待重开。
- WAL 仍是事务顺序权威；blob/页/字段索引/ANN 都是派生数据。
- 公开分数高分优先，内部距离低分优先；`naive` 是唯一正确性基线。
- 不引入旧格式读取路径；无法识别的目录 `CorruptData`，由重建或重新导入恢复。
- 不改变 C ABI 既有函数的签名与语义；本轮的 ABI 变更仅为 `MaintenanceStatus` 的新增 JSON 字段。

## 六、完成定义

- `cargo check --lib`、`cargo check --features c_api` 零警告。
- `cargo test -p simvec` 全绿（含本轮修改涉及的 bitmask/tracker/segments/checkpoint 用例）。
- `AGENTS.md` 与代码一致，`docs/plan/post-refactor-followups.md` 的"已完整落地"结论改为指向本文。

## 七、本轮落地结果（自检记录）

| 检查 | 结果 |
| --- | --- |
| `cargo check --lib --ignore-rust-version` | 通过，0 警告 |
| `cargo check --features c_api --ignore-rust-version` | 通过，0 警告（未改 C ABI 签名，`include/simvec.h` 无需重生成） |
| `cargo check --tests --ignore-rust-version` | 通过 |
| `cargo test --lib --ignore-rust-version` | 420 passed / 0 failed |
| `cargo test --tests --ignore-rust-version` | 13 个集成套件全通过（commit_queue 3、concurrency 8、engine_quarantine 1、filter_bitmap 9、filtered_count_scroll 5、hnsw 10、ivf 11、payload_page_boundary 2、quantization_recall 5、recovery 11、search 11、storage 10、tombstone_checkpoint 4） |
| `cargo check --manifest-path cli/Cargo.toml` | 通过；两个 `Cargo.lock` 随依赖图重生成：`crc32fast` 条目从 `Cargo.lock` 与 `cli/Cargo.lock` 中消失，并补上此前 `cli/Cargo.lock` 里缺失的 `lz4_flex` / `twox-hash` |

新增用例：

- `payload_store::bitmask::tests::a_flipped_bit_fails_the_checksum`（T1）
- `payload_store::bitmask::tests::a_version_one_file_is_rejected`（T1）
- `storage::segments::tests::the_appendable_tail_is_selected_only_as_the_sole_segment`（T3）
- `storage::tests::test_staged_id_tracker_is_invisible_until_published`（T2）
- `storage::tests::test_snapshot_verification_rejects_missing_file`（T10-G2）
- `storage::tests::test_snapshot_rejects_existing_target`（T10-G2）
- `storage::tests::test_snapshot_verification_rejects_wrong_version`（T10-G2）

删除用例：`storage::snapshot::tests::legacy_crc32_envelope_is_rejected`（T6）。该测试的 magic 写成 `SVSP`（真值 `SVPC`），在 magic 分支就返回 `None`，从未验证 CRC，属于无效断言；其意图已由 `test_corrupt_dtype_header_is_rejected` 覆盖。

T10 的 G2 三项已随本轮补上（行为本来就正确，缺的是回归保护），其余 G1/G3/G4/G5 仍待做。

一处计划反转（见 §2.3）：T3 原计划删掉单段例外，被三个既有用例否决，最终改为修改注释与文档、补测试锁定行为。
未运行：`cargo bench`（无性能改动）、语言绑定测试（需各自工具链）、wasm 目标（缺 `wasm32-unknown-unknown`）。
