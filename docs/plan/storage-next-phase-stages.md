# 存储下一阶段：分阶段实施方案（S1–S8）

本文是 [`storage-next-phase-impl.md`](./storage-next-phase-impl.md) 的**执行顺序版**：把那 11 个条目拆成 8 个可独立提交、可独立回退的阶段，并记录本轮实施后**必须修正的两条结论**（§3.2、§3.3）。

- **事实源**：`AGENTS.md` 的存储章节。与本文冲突时以 `AGENTS.md` 为准。
- **上游**：`storage-next-phase.md`（任务与取舍）、`storage-next-phase-impl.md`（函数级细节与 8 条修订 R1–R8）。
- **本文新增的修订**：R9（N2 的"去门"降级为"门内只剩发布"）、R10（N1b 的段内空洞表示）。

---

## 0. 状态总览

| 阶段 | 内容 | 状态 | 提交 |
| --- | --- | --- | --- |
| S1 | **N5 崩溃注入骨架** + 六窗口矩阵 | **已完成** | `storage: add SIMVEC_FAULT crash-window hooks and the six-window matrix (N5 base)` |
| S2 | **N2 checkpoint 去二次刷盘**（tracker 自封顶 + `gate_wait_ms`） | **已完成** | `checkpoint: publish a self-capping tracker and drop the gated re-flush (N2)` |
| S3 | N1b-1 payload 段内追加（B1） | 待实施，3–4 人日 | — |
| S4 | N1d 退役整代 `quant.bin` | 待实施，2–3 人日 | — |
| S5 | N1b-2 段级压实 + `segments.json` 指纹 | 待实施，5–6 人日 | — |
| S6 | N1c blob 归属到段 | 待实施，3 人日 | — |
| S7 | N3 快照去门 | 待实施（依赖 S3/S4） | — |
| S8 | N4b 五绑定冒烟 | 待实施，2 人日 | — |

累计剩余约 15–18 人日；本轮交付 S1+S2 约 4 人日。

**为什么 S1 排在 S2 之前**：S2 改的是"崩溃窗口里谁负责补齐"的契约。没有 S1，这类改动只能靠"改完跑一遍全量测试"来判断，而全量测试不会恰好停在 fsync 与 rename 之间。S1 让每个窗口都能被**确定性地**打开（`SIMVEC_FAULT=<point>` → `SIGABRT`），S2 的验收才有可能写成断言而不是祈祷。

---

## 1. 分阶段原则

1. **先补可验证性，再动写路径**：每个阶段先落地"能被观测的东西"（计数、状态字段、注入点、bench），再改行为。
2. **一个阶段只引入一个不变量**，并且这个不变量必须有一条用例**直接**断言它（不是间接"查询结果看起来对"）。
3. **每个阶段可独立回退**：回退方式写在阶段末尾的"回退"小节，不依赖后续阶段。
4. **不改变对外语义的改动单独提交**：例如 S2 只改"何时发布"，不改任何可见结果。

---

## 2. S1（已完成）崩溃注入骨架

### 2.1 形态

`src/storage/fault.rs`：

```rust
pub(crate) fn fault_point(name: &'static str);   // 命中即 std::process::abort()
```

- `SIMVEC_FAULT=a,b` 在首次调用时解析并缓存（`OnceLock<Option<Vec<String>>>`），未布防的进程每个点只付一次 relaxed load + 分支；
- 命中用 `abort()` 而不是 `exit()`：否则析构函数会把被测窗口里的状态刷盘，测试可能"因为清理而通过"；
- 点名是测试契约，`POINTS` 常量表 + `debug_assert!` 保证调用点不会写错名字。

### 2.2 注入点表

| 点名 | 位置 | 窗口含义 | 发生后必须成立 |
| --- | --- | --- | --- |
| `wal_after_append` | `queue::run_batch`，`append_batch` 之后、apply 之前 | 记录已持久化、未应用 | 重开按日志补齐，且只补一次 |
| `apply_before_ack` | `queue::run_batch`，`apply_queued_txn` 之后、`settle` 之前 | 已应用、调用方仍在等 | 重开不重复计数 |
| `checkpoint_after_flush` | `store_maintenance::checkpoint_inner`，`flush_data_files` 之后 | 数据已 fsync、tracker 未发布 | 计数不变，全部可重放 |
| `checkpoint_after_tracker_publish` | 同上，`publish_id_trackers` 之后 | tracker 可见、manifest 落后 | open 接受"tracker 领先 manifest" |
| `wal_before_truncate` | `wal::truncate_through` 入口 | 日志即将被裁切 | 重开可读，日志里的记录不丢 |
| `compaction_after_publish` | `compaction::commit_generation`，`publish` 之后 | 新 generation 已发布、内存未切换 | 重开按新 manifest 加载 |

### 2.3 验收

`tests/fault_injection_test.rs`：每个窗口在**子进程**里跑（`current_exe --exact crash_child --ignored`，`SIMVEC_FAULT=<point>`），父进程只断言重开后的状态，并断言子进程死于 `SIGABRT`（`signal() == Some(6)`）——这样"点没命中而正常退出"不会被误判为通过。

```text
cargo test --test fault_injection_test        # 6 passed, 1 ignored（child）
```

### 2.4 后续阶段的用法

- S2 的每个契约变更都对应上表某个窗口，直接加一条父进程断言即可；
- S5（段级压实）复用 `compaction_after_publish`，并新增 `compaction_after_link`（硬链接完成、manifest 之前）；
- S6（blob 按段）新增 `blob_written_before_wal`。

---

## 3. S2（已完成）checkpoint：门内只剩发布

### 3.1 改动

| 位置 | 改动 |
| --- | --- |
| `segments::write_id_tracker` | **条目按本段前沿封顶**：`version > lsn` 的 id 以"未知"发布（`free: true`、`point_lsn/delete_lsn = 0`），交由重放重建 |
| `store_maintenance::checkpoint_inner` | 删除 `flush_raced_segments`（门内二次刷盘）与 `raced_segments` 助手；保留门，但门内只剩 `publish_id_trackers` + `meta.save` + manifest + marker + truncate |
| 同上 | 发布前沿改用 `durable = inner.min_persisted(last_applied)`：`generation.set_lsn(durable)`、`truncate_through(wal_pins.clamp(durable))` |
| `storage.rs` | 新增 `last_gate_wait_us`，`MaintenanceStatus::gate_wait_ms` 上报 |
| `store_open::replay_wal` | 打开时把 `segment_applied` 从**代际前沿**起算（原先从 0 起算） |

### 3.2 R9：为什么门不能整个摘掉

原文 §3.4 的目标是"checkpoint 不再关门"。实施后确认这只做对了一半，必须修正：

- **能摘的是第二次刷盘**。原实现关门后重刷"被写入者碰过"的段，是为了让 tracker 命名的字节一定已落盘。现在 tracker 自己按段前沿封顶，未落盘的条目以"未知"发布，重放负责补齐——于是门内不再需要任何 fsync。
- **不能摘的是"发布"本身**。三件事必须与写入者互斥：
  1. `meta.bin` / `segments.json` / marker 的写入顺序是恢复语义的一部分（marker 用 `next_lsn` 取号，若队首还有更小 LSN 的记录未 append，日志顺序会倒退，段名连续性断裂）；
  2. `publish_id_trackers` 的 rename 与 `meta.save` 必须成对出现，否则 open 会看到"tracker 已可见、manifest 未前进"之外的第三种中间态；
  3. `truncate_through` 必须在"没有事务处于已分配 LSN 未入日志"的状态下执行。
- 结论：**N2 的收益口径改为"门内不含 fsync"**，用 `gate_wait_ms` 度量（门内等待只来自在途事务 drain，量级应为毫秒；若回到秒，说明有人把刷盘搬回门内）。门本身保留，注释与 `AGENTS.md` 已按此更新。

### 3.3 R10：`generation.lsn` 必须是"可发布前沿"而不是"已应用前沿"

封顶带来的新约束：checkpoint 写入的 `generation.lsn` 必须 ≤ 每个段真正落盘的前沿，否则 open 的 `meta.last_applied_txn.min(generation.lsn())` 会把重放起点抬到未落盘记录之上，被标记为"未知"的 id 就再也没人重建了。因此：

```text
generation.lsn = durable = min(segment_persisted)     // 不再是 meta.last_applied_txn
truncate_through(wal_pins.clamp(durable))
```

内存里的 `meta.last_applied_txn` 仍是真实的已应用前沿（不回落），只是"这一代文件能担保到哪里"由 `generation.lsn` 单独表达。`store_open` 打开时 `segment_applied` 也按 `floor` 起算，否则"刚打开就 checkpoint"会以 0 为前沿，把整库发布成未知（该缺陷已由 `test_checkpoint_marker_at_full_segment_reopens_cleanly` 捕获）。

### 3.4 验收

| 用例 | 断言 |
| --- | --- |
| `test_checkpoint_flushes_each_segment_once`（改名自 `..._resyncs_only_segments_a_writer_touched`） | 每次 checkpoint 每段恰好刷一次，没有第二遍 |
| `test_checkpoint_racing_writers_keeps_every_acknowledged_point`（新增） | 4 写线程 × 3 次 checkpoint：重开后 `count`、`get`、`scroll` 三者一致；`gate_wait_ms` 有界（< 5 s） |
| `tests/fault_injection_test.rs` 的 `checkpoint_after_flush` / `checkpoint_after_tracker_publish` | 两个窗口在新契约下仍可恢复 |
| 既有 `checkpoint_with_a_non_empty_queue_keeps_every_transaction` | 门仍会等在途事务，不丢事务 |

### 3.5 回退

把 `write_id_tracker` 的封顶分支删掉、`flush_raced_segments` 恢复即可；`gate_wait_ms` 与 `segment_applied` 起点修复是纯增量，保留无害。

---

## 4. S3（待实施）N1b-1：payload 段内追加（B1）

> 这是 S5 的**硬前置**：硬链接共享 inode 的合法性只有一条——同一 generation 内已发布段的字节只能追加。向量侧 N1a 已保证，payload 侧没有。

### 4.1 现状与问题

- payload 页已按段隔离：`segments/<start>/payloads_store/{config.json, tracker.bin, tracker.journal.bin, bitmask.bin, pages/page_NNNN.bin}`（`segment_payloads.rs`）；
- 但 `PayloadStore::put_at` 是**原地改**：先 `bitmask.free_blocks(旧块)`，再 `bitmask.allocate` 拿新块（可能立刻复用刚释放的块），然后 `pages.write_value` 覆盖该块，最后 `tracker.set_and_record`。删除（`put_at(None)`）与 `merge_at`/`delete_keys_at` 走的都是这条路。

后果：若 S5 把未选中段的 `payloads_store/` 目录硬链接进新 generation，老 generation 的存活读者会读到被别的槽位复用的页（错 payload），或读到撕裂页（CRC 失败）。因此 S5 之前必须让"已冻结段的 payload 字节只追加"。

### 4.2 目标不变量

> **P1**：段 `s` 被冻结（`s < tail_base`）之后，其 `payloads_store/` 内任何**页数据**不再被改写；页只在该段仍可追加时增长。

### 4.3 实现

1. **分配器改为按段单调**（`payload_store::bitmask.rs`）：新增 `allocate_append(blocks)`——从该段"已用块水位"向后取，绝不回填空洞；`free_blocks` 只更新位图与**段内垃圾计数**，不再让位图驱动分配。
2. **垃圾回收交给压实**：段级压实（S5）在重写选中段时按 live 槽位重放 payload，天然回收；未选中段不回收（其垃圾在下次选中时回收）。段内垃圾计数进入 `SegmentEntry`，作为选中判据的一部分（`garbage_ratio = (tombstones + payload_dead_blocks)/total`）。
3. **冻结判定**：写入路径拿 `segments::tail_base(next_slot, segment_slots)` 与目标槽比较；落在冻结段内的 `put_at` 只允许"置空"（记为垃圾，不改页），落在尾段则走 `allocate_append`。
4. **页大小余量**：`DEFAULT_PAGE_SIZE` 视为上限，段内垃圾会让页数增长；`ensure_page` 已按需创建，无需改动，但 `memory_estimate` 与 `StoreConfig` 注释要写明"页占用上界 = live + 段内垃圾"。
5. **原子性**沿用现有顺序：页写 → tracker 记录（journal append）→ 水位；单条 tracker 记录 ≤ 12 B，落在同一 cache line。

### 4.4 验收

| 用例 | 断言 |
| --- | --- |
| `test_frozen_segment_payload_pages_are_append_only` | 冻结段内 `put_at(Some)` 后页文件字节不变（逐页 hash） |
| `test_overwrite_in_frozen_segment_leaves_page_bytes_intact` | N1a 的覆盖（删旧槽 + 新槽追加）不改变冻结段任一页 |
| `test_payload_garbage_is_reclaimed_by_compaction` | 压实后 `pages/` 总字节下降，`get` 全部正确 |
| `test_append_only_allocation_survives_reopen` | 重开后水位 = 压实前 live 块的连续性 |

### 4.5 回退

`PayloadStore` 增加 `StoreConfig::append_only: bool`（默认 false），S5 依赖它时必须为 true；回退即改为 false 并恢复位图分配。

---

## 5. S4（待实施）N1d：退役整代 `quant.bin`

### 5.1 现状

两套文件：整代 `quant.bin`（`slot_capacity * bytes_per_vector`，`write_slot` 直接按全局槽位偏移写）+ 每段派生副本 `segments/<start>/quant_seg_<start>.bin`（`write_segment_file`，只被段图使用）。压实写整代文件（`write_compacted_file`），于是"只重写选中段"在任何量化集合上都被顶回全量（R5）。

### 5.2 目标

`quant_seg_<start>.bin` 成为**唯一**副本：

| 动作 | 新行为 |
| --- | --- |
| `write_slot(slot, vector)` | 解析段 → 打开/扩展该段文件 → 写 `local * bpp` |
| `grow_to(n)` | 按段扩展（缺段文件即"该段无量化码"，退回精确扫描） |
| `open` | 逐段加载；缺一段不报错，只让该段退回精确路径；`QuantMeta` 上移到 generation 根 |
| 压实 | 只写选中段的 `quant_seg_*.bin`；未选中段硬链接 |
| `set_quantization_config` | 允许中途改变配置，未重训的段直接删除文件（可重建） |

### 5.3 验收

`test_quant_codes_live_in_segment_files`（不存在 `quant.bin`）、`test_dropping_one_segment_quant_file_keeps_recall`（删一段只影响该段召回，检索仍返回 top-k）、`test_quant_segment_files_shrink_after_compaction`、磁盘占用前后对比（同集合近似减半）。

### 5.4 回退

保留 `QuantStore::open` 对整代文件的兼容读取（存在 `quant.bin` 时按整代装载，删除 `QUANT_FILE` 后走新路径），一个里程碑后删除兼容分支。

---

## 6. S5（待实施）N1b-2：段级压实

### 6.1 选择

```rust
pub(crate) struct SegmentPlan {
    pub base: u32,
    pub width: u32,        // 段槽宽（= segment_slots，末段可能被 slot_capacity 截断）
    pub live: u32,         // 压实后段内条数，落在 [base, base+live)
    pub local_map: Vec<u32>, // 长度 = width，u32::MAX = 丢弃
    pub rewrite: bool,
}
```

- **显式 `compact()`**：`rewrite = tombstone_count > 0`（含尾段）。
- **后台**：沿用 `select_for_compaction`（尾段仅当它是唯一段时入选）。

### 6.2 R10：段内空洞的表示

段基址不变（读路径 `slot / segment_slots` 不用改），所以压实后的段是"稠密前缀 + 尾部空洞"。空洞**不能**记成墓碑：那会让该段 `garbage_ratio` 立刻再次越阈值，形成压实自旋，也会污染 `tombstone_count`。表示方式：

1. `keys.bin` 重写时整段清零，只写 `[0, live)` 的 key → 空洞槽没有 key，扫描自然跳过；
2. `tombstones.bin` 的位保持 0（不是垃圾，是未使用）；
3. `segments.json` 的该段 `end = base + live`，`derive_entries` 按 `end` 而非 `next_slot` 计账 → 空洞不计入 live，也不计入 garbage；
4. `next_slot` 只在**尾段被选中**时下调为 `tail_base + live_tail`（唯一允许下调的场景）；`slot_capacity` 不变（地址空间不回收，`Vectors::open` 的长度校验因此不受影响）。

### 6.3 每段文件动作

| 文件 | 选中段 | 未选中段 |
| --- | --- | --- |
| `vectors.bin` | 段内重排写（长度不变） | 硬链接 |
| `keys.bin` | 整段清零 + 写 `[0, live)` | 硬链接 |
| `tombstones.bin` | 写全 0 位图 | 硬链接 |
| `vector_dtype.bin` | 复制 | 硬链接 |
| `payloads_store/` | 按 `local_map` 重放（S3 之后可硬链接页 + 只改 tracker） | S3 之后硬链接；之前整树 `fs::copy` |
| `id_tracker.bin` | **必须重写**（绑定 generation id） | **必须重写**，内容不变 |
| `quant_seg_<start>.bin` | 只重建该段（S4 之后） | 硬链接（S4 之后） |
| `hnsw_seg_<start>.bin` / `ivf_seg_<start>.bin` | 只重建该段，`local` 用新编号 | 硬链接，并**复用内存里的 `SegmentHnsw`**（槽号未变） |
| 整代 `quant.bin` | S4 之前必须整份重写（R5） | 同左 |

`meta` 语义：`slot_capacity` 不变；`next_slot` 见 §6.2；`live_count`/`tombstone_count` 按压实后重算。

### 6.4 `segments.json` 指纹

```rust
pub(crate) struct SegmentFileEntry { pub path: String, pub len: u64, pub crc32: u32 }
// SegmentEntry 追加 files，#[serde(default)]，FORMAT_VERSION 仍为 1
```

- 写入点：checkpoint 与压实写 manifest 之前；**只对 `persisted_lsn` 前进过的段重算**，未变化的段从旧 manifest 继承，避免 checkpoint 变成 O(全库) 读；
- 读取点：open 只校验"长度"，CRC 校验放在 `verify` 类诊断路径（每次 open 读全库 CRC 不可接受）；
- 断言未选中段"新旧 generation 字节相同"的直接依据。

### 6.5 发布与内存切换

沿用现有序列（`staged` → `DataFiles::open` 预校验 → manifest rename → `files.swap` → poison 标志），差异：

1. **硬链接在 fsync 之前**：`hard_link(old, new)` 不产生数据块，但目录项要 `File::open(dir)?.sync_all()`；
2. `reverse` 只对选中段重映射（可由 `local_map` 推出，O(选中段)），未选中段不动；空洞槽不入 `reverse`；
3. `filter_bitmap`/`payload_indexes` 先复用现有全量 `rebuild`（O(live)，风险最低），后续再优化为按段 `remap`；
4. `pending` 按 `local_map` 重映射并丢弃判死槽；
5. 旧 generation GC：`retire()` 删目录即可——未选中段 inode 因新链接存活。

### 6.6 验收

| 用例 | 断言 |
| --- | --- |
| `test_segment_compaction_keeps_untouched_segments` | 未选中段每文件 `ino`、`mtime`、字节三者全不变（`MetadataExt`） |
| `test_segment_compaction_costs_only_selected_bytes` | 新 generation 中 inode 不在旧集合的文件总长 ≤ `Σ选中段字节 + 元数据` |
| `test_segment_compaction_preserves_queries` | 200 点 + 随机过滤：`count/get/search/scroll/filter` 逐点一致 |
| `test_segment_compaction_cycles_stay_bounded` | 100 轮"写→删→压实→重开"：`live_count` 正确，目录占用单调不增 |
| `test_segment_compaction_hole_slots_are_not_garbage` | 压实后该段 `tombstone_count == 0` 且 `needs_compaction() == false` |
| 新增 `fault_point("compaction_after_link")` | 硬链接完成、manifest 之前崩溃：旧 generation 仍可服务，重开忽略 staging |

### 6.7 回退

`StorageConfig::segment_compaction: bool`（默认 true）保留今天的整代实现作为降级路径，一个里程碑后删除。

---

## 7. S6（待实施）N1c：blob 归属到段

- `segments/<start>/blobs.bin`；`BlobRefSpec` 增加 `segment_start`；归属规则 = 事务里第一个点所在段；
- 压实只复制 live 点引用的 blob，被覆盖/删除的 blob 随段重写回收；跨段引用按引用计数复制；
- `wal_blob_bytes` 语义改为"各段 blob 之和"（字段不变）；
- 验收：`test_blob_files_are_segment_scoped`、`test_compaction_reclaims_unreferenced_blobs`、`test_upsert_after_reopen_resolves_segment_blob`。

---

## 8. S7（待实施）N3：快照去门

前置：S3（payload 不可变）、S4（量化码按段）。步骤（`storage.rs::snapshot_capture_inner`）：

1. `pinned = generation.lsn()`（S2 之后它已经是"可发布前沿"），`pin_wal_ack(pinned)`；
2. 不变文件走硬链接/FICLONE（`copy_snapshot_file` 已有增量复用），尾段只复制 `≤ pinned` 的前缀（`copy_prefix` 已有）；
3. 复制 `≥ pinned` 的 WAL 段，恢复时按 `pinned` 重放补齐；
4. 唯一保留的临界区缩短为"取 generation + 写清单"，`compact_mutex` 仍持。

验收：`test_snapshot_window_allows_writes`、`test_snapshot_restore_replays_to_the_export_point`（同时断言"恢复点 = pinned"和"重放后 = 导出时刻"两种期望）、`test_verify_snapshot_accepts_prefix_copy`。

---

## 9. S8（待实施）N4b：五绑定冒烟

`bindings/{python,js,go,java,r}` 各一条 `create → upsert → maintenance_status → snapshot → snapshot_incremental → verify → restore` 冒烟，断言 `maintenance_status` 含 `segments`/`gate_wait_ms`。CI 顺序固定为 `cargo build --features c_api` → `SIMVEC_LIB=target/debug/libsimvec.so` → 各绑定测试。

---

## 10. 依赖图与风险

```text
S1(N5 注入) ──> S2(N2 自封顶 tracker) ──> S3(payload 只追加) ──> S5(段级压实) ──> S7(快照去门)
                                          └──> S4(quant 按段) ──┘
S6(blob 按段) 独立，但收益依赖 S5 的选中逻辑
S8 独立
```

| 风险 | 影响 | 缓解 |
| --- | --- | --- |
| S5 段内空洞被误当垃圾（R10） | 压实自旋、`tombstone_count` 失真 | 用 `end = base + live` 计账 + 专门用例 |
| S3 之前直接做 S5 | 跨代错读 payload | 顺序强制；S5 的未选中段 payload 先复制不硬链接 |
| 硬链接跨文件系统 | `hard_link` 失败 | 回退为 `fs::copy` 并 `tracing::warn!` |
| 压实期间并发写 | 计划失效 | 沿用 `mutations` 计数 + 有界重试 + 关门兜底 |
| `gate_wait_ms` 被误读成"门越小越好" | 把 fsync 搬回门内以压低数字 | 注释写明该数字"只应包含在途事务 drain"；S5 之后复核 |

---

## 11. 验收命令

```text
cargo check --all-targets                      # 0 错误 0 警告
cargo test                                     # 全绿（当前 531 项）
cargo test --features c_api
cargo test --manifest-path cli/Cargo.toml --target-dir target
cargo test --test fault_injection_test         # 六窗口矩阵
QUICK=1 cargo bench --bench compaction_io_vs_garbage   # S5 后 bytes_written 应随垃圾量而非库容增长
```

> 本机工具链 `rustc 1.93` 低于 `rust-version = "1.97"`，本地执行需追加 `--ignore-rust-version`；CI 仍用 `+1.97.0`。
