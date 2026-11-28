# 存储下一阶段实施细则（N1a–N7）

本文是 [`storage-next-phase.md`](./storage-next-phase.md) 的**代码级实施细则**：把原文的 11 个任务条目落到具体文件、函数、数据结构、测试用例与验收命令，并记录本次实施中发现、需要修订原文的 8 处结论。

- **事实源**：`AGENTS.md` 的存储章节。本文与 `AGENTS.md` 冲突时以 `AGENTS.md` 为准。
- **上游文档**：`storage-next-phase.md`（任务与取舍）、`deferred-storage-completion.md`（T9 遗留项）。
- **工作方式**：每个任务一个独立提交，先补齐"可验证性"（用例、计数、注入钩子），再改动写路径，最后改 `AGENTS.md`。
- **执行顺序**：本文按"任务"组织，按"阶段"落地见 [`storage-next-phase-stages.md`](./storage-next-phase-stages.md)（S1-S8，含已完成标记与实施后新增的修订 R9/R10）。

---

## 0. 代码审计：原文各条目的真实起点

进入本阶段时仓库只有一条历史提交（`storage remain 2`），没有可用的 diff，因此逐条对照源码核实：

| 编号 | 原文假设 | 审计结论 | 证据（改动前） |
| --- | --- | --- | --- |
| N1a | 未做，upsert 原地覆盖 | **本次完成** | `store_write.rs::resolve_upsert_slot_locked` 对已存在 id 直接返回旧 slot；`apply_upsert_locked` 原地 `vectors.write_slot` |
| N1b | 未做 | 未做 | `compaction.rs::plan_slots` 全局重排；`stage_generation` 逐段重写全部文件；`meta.next_slot = plan.live_count` |
| N1c | 未做 | 未做 | `wal_blobs.rs` 单文件只追加，仅整体 `reset()` |
| N1d | 未做 | 未做 | `quant.bin` 整代可写 + `quant_seg_<start>.bin` 派生副本双份 |
| N2 | 未做 | 未做 | `store_maintenance.rs::checkpoint_inner` 仍 `queue.quiesce()` |
| N3 | 未做 | 未做 | `storage.rs::snapshot_capture_inner` 仍 `queue.quiesce()` |
| N4 | 缺端到端用例 | Rust 侧补完，绑定冒烟未做 | `src/c_api/tests.rs` 无 `maintenance_status` 断言 |
| N5 | 未做 | 未做 | 全仓无 `SIMVEC_*` 环境变量注入钩子 |
| N6 | 未做 | 未做 | `benches/` 无 `compaction_io_vs_garbage`，无写入字节计数 |
| N7 | 未做 | **本次完成** | `CollectionStore::create_with_storage_and_slots` 不校验 `dtype`，`Turbo` 直到首次 `write_slot` 才报错 |

`segments.json` 也确认没有原文 §2.5 要求的每段 `len`/`crc32c`（`SegmentEntry` 只有 `start/end/capacity/live_count/tombstone_count/applied_lsn/persisted_lsn`）。

---

## 1. 本次已落地

### 1.1 N1a 点级追加写 + 派生的尾段 free list

**提交**：`storage: append-on-overwrite with a derived tail free list (N1a)`

改动点：

| 位置 | 改动 |
| --- | --- |
| `store_write.rs::apply_upsert_locked` | 已存在 id 不再原地写：分配新 slot → 旧 slot 走 `apply_delete_slot_locked`（墓碑 + 索引注销 + 图治愈）→ 新 slot 写 key/向量/payload |
| `store_write.rs::allocate_slot_locked` | 新分配器：先回收**可追加尾段**里的墓碑槽，再从 `meta.next_slot` 取新槽并 `ensure_capacity` |
| `store_write.rs::reclaim_slot_locked` | 复用墓碑槽：清位 + `tombstone_count -= 1` |
| `store_write.rs::publish_slot_locked` | 写 `keys.bin`、`reverse`、`live_count += 1`（取代 `init_slot_locked`） |
| `tombstones.rs::first_free_in` | 按 slot 顺序找第一个墓碑位（跨 64K 位块扫描） |
| `segments.rs::tail_base` | 尾段基址 = `floor(next_slot / segment_slots) * segment_slots`（与 `immutable_end` 的"首段未填满"特例不同） |
| `storage.rs::StoreInner` | 新增 `free_cursor`（扫描提示）与 `replaying`（重放标记） |
| `store_maintenance.rs` | checkpoint 落盘后重置 `free_cursor`；`ensure_capacity` 的 `max_points` 改判 `live_count` |
| `compaction.rs::commit_generation` | 重排后重置 `free_cursor` |
| `store_open.rs` | 初始化两个新字段；`validate_admission_locked` 改判 `live_count`；重放期置 `replaying` |
| `index/hnsw.rs` | 删除 `note_overwrite_delta`（追加写不再有"节点向量被改写"这条 staleness 腿，墓碑治愈已覆盖） |

**不变量与取舍**（已写入 `AGENTS.md`）：

1. 已发布段（填满的段）只接受追加，活槽字节永不改写 —— 由 `test_overwrite_never_rewrites_a_complete_segment` 直接比对 `vectors.bin`/`keys.bin`/`tombstones.bin` 字节。
2. free list **派生不持久化**：由墓碑表 + `free_cursor` 复算；`free_cursor` 在每个持久化边界（open / checkpoint / compaction）归零，使实时应用与重放在同一起点出发。
3. `meta.next_slot` 是**地址空间高水位**而非存活数：准入与扩容都改判 `live_count`。

**验收用例**（均已通过）：

| 用例 | 断言 |
| --- | --- |
| `test_upsert_overwrite_appends_and_tombstones_the_old_slot` | 新 slot ≠ 旧 slot、旧 slot 有墓碑、`count()` 不变、旧槽字节不变、重开后一致 |
| `test_repeated_overwrites_reclaim_the_dead_tail_slot` | 同一 id 200 次更新后 `next_slot <= 2`、`tombstone_count <= 1`、`count() == 1` |
| `test_frozen_segment_slots_are_never_reclaimed` | 冻结段里的墓碑槽不被回收，新插入走尾段 |
| `test_overwrite_never_rewrites_a_complete_segment` | 完整段三份文件字节完全不变 |
| `test_replay_keeps_the_directory_consistent_after_an_overwrite` | 重开后 count/向量/槽位唯一性 + `keys.bin` 与 `reverse` 逐槽一致 |
| `test_upsert_then_delete_then_reinsert`（改写） | 删除后重插回收尾段死槽，不再产生待压实垃圾 |

### 1.2 N7 `Turbo` 在 create 期拒绝

**提交**：`storage: reject element dtypes without a distance mapping at create (N7)`

- 新增 `CollectionConfig::validate_element_types`（`src/types/config.rs`）：只校验各 dense space 的 `dtype × distance` 映射，**不**携带 `validate_kind` 的跨字段镜像检查（否则会误伤只调 `config.distance` 的调用方）。
- `CollectionStore::create_with_storage_and_slots` 在建目录之前调用它，因此失败的 create 不落任何文件。
- 用例：`test_turbo_dtype_is_rejected_at_create` —— 断言 `InvalidConfig` 且目标目录不存在，并用同一路径以 `F16` 重建成功。

### 1.3 N4（Rust 侧）`maintenance_status` 的 C ABI 形状断言

**提交**：`c_api: assert the maintenance_status segment array shape (N4)`

用例 `c_api::tests::maintenance_status_reports_segment_shape`：建集合 → 批量 upsert → `simvec_maintenance_status` → 解析 JSON，断言 `healthy`/`persisted_lsn`/`wal_blob_bytes` 类型，以及 `segments` 是非空数组、每项含数值型 `start`/`applied_lsn`/`persisted_lsn` 且 `start` 递增、`error` 为 null。

### 1.4 N6 压实 I/O 计量骨架

**提交**：`bench: account compaction bytes against garbage volume (N6)`

`benches/compaction_io_vs_garbage.rs`（已在 `Cargo.toml` 注册，`harness = false`）不改动生产代码，用 inode 差集计量：

```text
bytes_written = Σ{新 generation 中 inode 不在旧 generation 的文件长度}
```

未复用（未硬链接）的文件才计数，因此该指标本身就是 N1b 的验收口径。`QUICK=1` 的当前基线（live=2000、dim=64、segment_slots=1024、第一段集中垃圾）：

| 垃圾量 | live_bytes | bytes_written | write/live |
| --- | --- | --- | --- |
| 200 | 729 473 | 671 890 | **0.92** |
| 600 | 729 473 | 665 499 | **0.91** |
| 1000 | 729 472 | 374 414 | 0.51 |

"删 10% 与删 30% 写同样多的字节"正是原文 §2.1 要解决的问题；N1b 落地后这张表的 `bytes_written` 必须转向跟垃圾列走、对 `live_bytes` 不敏感。

### 1.5 本次全量验证

```text
cargo check --all-targets --ignore-rust-version     # 0 error 0 warning
cargo test  --ignore-rust-version                   # lib 432 绿；13 个集成测试二进制 + doc-tests 全绿
cargo test  --ignore-rust-version --features c_api --lib   # 441 绿
cargo test  --manifest-path cli/Cargo.toml --target-dir target --ignore-rust-version  # 4 绿
```

> 本机只有 `rustc 1.93`，低于 `rust-version = "1.97"`，因此所有命令需要 `--ignore-rust-version`；CI 仍按原文用 `+1.97.0`。

---

## 2. 对原文的修订（8 条，均有代码依据）

| # | 原文结论 | 修订 | 依据 |
| --- | --- | --- | --- |
| R1 | §2.4 free list 覆盖"段内已墓碑且 `slot < segment_next_slot`"的所有槽位 | **收窄为只回收可追加尾段** | 冻结段字节不可变是 N1b 硬链接复用的前提；另外 `index_lifecycle::register_slot` 对已发布段内的槽走 `index.insert`，把一个新向量塞进已建好的图会污染邻接表，尾段槽走 `pending` 精确扫描则无此问题 |
| R2 | §2.4 验收"1000 次同 id 更新 → 1000 个墓碑" | 改为"`next_slot` 有界、`live_count == 1`、`tombstone_count <= 1`" | 复用型 free list 与"每次更新留一个墓碑"自相矛盾；实际实现是两槽交替 |
| R3 | §2.4 验收"实时应用与崩溃重放两边 `id → slot` 完全相同" | **不成立**，改为"目录、向量、计数一致；槽号可能漂移" | 重放从 tracker 的 `point_lsn` 封顶值出发，未落盘的追加写在重放里无法复原原位；`apply_upsert_locked` 因此在 `replaying` 时对已知 id 原地重写以保证幂等 |
| R4 | §2.5 顺序为 N1b → N1c | **N1c 的"payload 段内不可变"必须早于 N1b 的硬链接**（见 §3.1） | `PayloadStore::put_at` 释放旧块后重新分配并原地改写槽位指针，跨 generation 共享 inode 会让老 generation 的读者读到别的槽位复用的页或撕裂页 |
| R5 | §2.5 只列 `vectors/keys/payloads/tombstones/tracker/ANN` | 还要处理 `quant.bin`：N1b 若不同步做 N1d，紧压实仍整份重写量化码 | `compaction.rs::stage_generation` 里 `q.write_compacted_file(dir.join("quant.bin"), ...)` 是整代写 |
| R6 | §2.5 "显式 `compact()` 保持全选" vs §7 #3 "尾段不参与压实" | 二者冲突。建议：显式 `compact()` 选**所有含墓碑的段（含尾段）**，尾段压到 `tail_base + live_tail` 并允许下调 `next_slot`；后台维护按阈值选择且不选尾段 | 否则显式压实永远留下尾段垃圾 |
| R7 | §2.5 "未选中段逐文件硬链接" | tracker 不能硬链接：它绑定 generation id，`read_id_trackers` 对不同 generation 的 tracker 直接报 `CorruptData` | `segments.rs::write_id_tracker`/`read_id_trackers` |
| R8 | §9 验收命令 `cargo +1.97.0 ...` | 补充：本地工具链低于 MSRV 时用 `--ignore-rust-version`，否则 cargo 在 MSRV 检查处失败 | 本机 `cargo 1.93.0` |

---

## 3. 剩余任务细化方案

### 3.1 N1b 段级压实（P0，6–8 人日）

#### 3.1.1 前置：payload 的段内不可变（原 N1c 的写路径部分，2–3 人日）

硬链接复用的合法性只有一条：**同一 generation 内，已发布段的字节只能追加**。向量侧已由 N1a 保证，payload 侧没有：

- `PayloadStore::put_at`（`payload_store.rs`）对已存在的槽释放旧块再分配，可能复用同一页，并原地改写槽位指针表；
- `PayloadStore::delete_keys_at` / `merge_at` 同样原地改页；
- 结论：跨 generation 共享 `payloads_store/` 的 inode 会让**老 generation 的存活读者**读到被别的槽位复用的页（错 payload）或撕裂页（JSON 解析失败）。

两条可选路线：

| 路线 | 做法 | 代价 | 建议 |
| --- | --- | --- | --- |
| B1 | payload 改为段内追加写：`put_at` 不再复用已释放块，槽位指针表改为“追加一条 block 描述 + 原子更新指针表项”，旧块成为垃圾由压实回收 | 段内垃圾增长更快（和向量一样靠压实回收），`blocks_per_page` 需留 GC 余量 | **推荐**，与 N1a 同构，且顺带解决 N3 |
| B2 | N1b 只硬链接 `vectors.bin`/`keys.bin`/`hnsw_seg_*`/`ivf_seg_*`/`quant_seg_*`，`payloads_store/` 与 `tombstones.bin` 逐段字节复制 | 收益被 payload 体积顶住；payload 大的集合几乎无收益 | 作为 B1 之前的过渡，可单独交付 |

B1 的原子性：指针表项是一个固定长度记录（`blob_offset + len + flags`），单条记录 ≤ 12 字节，落在同一 cache line 内，仍需保证"页写在前、指针更新在后 + checkpoint 时 fsync 页"，沿用 `append_blob` 的既有顺序（blob 字节 → 记录 → header 水位）。

#### 3.1.2 选择与规划

```rust
// segments.rs 新增
pub(crate) struct SegmentPlan {
    pub base: u32,            // 段基址，等于目录名
    pub live: u32,            // 压实后的 local 数
    pub local_map: Vec<u32>,  // old local -> new local, u32::MAX = 丢弃
}

pub(crate) fn plan_segment(base: u32, capacity: u32, is_live: impl Fn(usize) -> bool) -> SegmentPlan
```

- 显式 `compact()`：`targets = 所有 tombstone_count > 0 的段`（含尾段，见 R6）。
- 后台：`select_for_compaction(entries, threshold)` 的现存规则（尾段仅当它是唯一段时入选）。
- 每段的 `local_map` 只在段内重排，段基址不变 → 读路径 `slot / segment_slots` 完全不用改。

#### 3.1.3 每段文件动作

| 文件 | 选中段 | 未选中段 |
| --- | --- | --- |
| `vectors.bin` | 只写 live local，长度 `live * dim * 4`（`write_vectors_segment`：`plan_slots` 的段内版本） | 硬链接 |
| `keys.bin` | `write_dir_file`，`rec_capacity` 仍为 `segment_slots`（读路径的 `local` 上界不变），只写 live local 的 key | 硬链接（槽位未变，key 位置不变） |
| `tombstones.bin` | 写全 0 位图（压实后本段无墓碑） | 硬链接 |
| `payloads_store/` | `SegmentPayloadStore::compact_segment_to(dest, base, &local_map)`（新增，按段复制 live 槽的页） | B1 后可硬链接；B2 前逐段复制 |
| `id_tracker.bin` | **必须重写**（generation id 变了）：live 条目 `offset = new local`，被丢弃 id 保留 `free = true` 条目（`offset = 0`，保留 `delete_lsn`） | **必须重写**，内容不变但 generation 更新（R7） |
| `vector_dtype.bin` | 复制（`snapshot::write`） | 硬链接 |
| `hnsw_seg_<start>.bin` / `ivf_seg_<start>.bin` / `quant_seg_<start>.bin` | 只重建该段（`stage_hnsw_segments`/`stage_ivf_segments` 的段内版本，`local` 用新编号） | 硬链接（`segment_hnsw::file_path` 在 generation 根，注意路径不在段目录） |

`meta` 语义（必须写进 `AGENTS.md`）：

- `slot_capacity` 不变（地址空间不回收）；
- `next_slot`：尾段未入选时不变；尾段入选时 `= tail_base + live_tail`（可下调，这是唯一会下调的场景）；
- `live_count`/`tombstone_count` 按压实后的实际值重算。

#### 3.1.4 `segments.json` 补每段文件指纹

```rust
pub(crate) struct SegmentFileEntry { pub path: String, pub len: u64, pub crc32: u32 }
// SegmentEntry 新增字段，保持 #[serde(default)]，FORMAT_VERSION 仍为 1
pub files: Vec<SegmentFileEntry>,
```

用途：open 校验（长度/CRC 不符 = `CorruptData`）、快照与诊断不再扫全量、以及**断言未选中段在新旧 generation 里字节相同**。写入点：`segments::write_with_entries` 的调用方（checkpoint 与 compaction）在写 manifest 前对每段文件求 `(len, crc32)`；未选中段可从旧 manifest 直接继承，避免重复读盘。

#### 3.1.5 发布与内存切换

沿用现有序列（`staged` → `DataFiles::open` 预校验 → manifest rename → `files.swap` → poison 标志），差异：

1. **硬链接要在 fsync 之前**：`hard_link(old, new)` 不产生新数据块，但目录项本身要 `sync`（`File::open(dir)?.sync_all()`）。
2. **内存重建按段**：只对选中段把 `reverse` 的槽位从 `base + old_local` 改成 `base + new_local`；未选中段不动。为实现这点，`commit_generation` 需要 `Vec<(PointId, old_slot, new_slot)>`（可由 `plan.points` + `local_map` 推出）。
3. `filter_bitmap`/`payload_indexes`：先复用现有全量 `rebuild`（O(live)，风险最低），后续再优化为按段 `remap(base, &local_map)`。
4. `pending`：按 `local_map` 重映射选中段，丢弃被判死的槽。
5. 旧 generation GC：`retire()` 删旧目录即可 —— 未选中段 inode 因新链接存活，选中段旧 inode 只被旧 generation 引用。

#### 3.1.6 验收

| 用例 | 断言 |
| --- | --- |
| `test_segment_compaction_keeps_untouched_segments` | 未选中段每文件 **inode、mtime、字节** 全部不变（`MetadataExt::ino()`） |
| `test_segment_compaction_costs_only_selected_bytes` | 新 generation 中 inode 不在旧集合的文件总长度 ≤ `Σ选中段字节 + 元数据` |
| `test_segment_compaction_preserves_queries` | 随机 200 点 + 随机过滤条件，`count/get/search/scroll/filter` 逐点一致 |
| `test_segment_compaction_cycles_stay_bounded` | 100 轮"写 → 删 → 压实 → 重开"，`live_count` 正确、目录占用单调不增（WAL 波动除外） |
| `test_segments_json_records_file_fingerprints` | manifest 的 `len/crc32` 与实际文件一致；篡改一字节后 open 报 `CorruptData` |
| `test_segment_compaction_replay_after_crash` | 发布后强杀（配合 §3.6 钩子），重开按新 manifest 加载，`count` 一致 |

#### 3.1.7 回退

N1b 是**单次切换**任务：保留 `compaction_full_generation()`（今天的实现）作为配置开关 `StorageConfig::segment_compaction = false` 的降级路径，一个里程碑后再删除。

### 3.2 N1c blob 归属到段（P1，3–4 人日）

- 文件：`segments/<start>/blobs.bin`，`BlobRefSpec` 增加 `segment_start: u32`（`wal_blobs.rs`、`WalRecord::Upsert` 的引用路径）。
- 归属规则（原文 §7 #4 选项 A）：**事务里第一个点所在的段**；压实只复制 live 点引用的 blob，被覆盖/删除的 blob 随段重写回收。
- `wal_blob_bytes` 改为按段求和（`MaintenanceStatus` 字段不变，语义改为"各段 blob 之和"）。
- 校验：`blob_ref` 的 `segment_start` 与实际段不匹配 = `CorruptData`；跨段引用按引用计数复制（一个 blob 被多个段引用时各段各一份，回收按引用计数）。
- 验收：`test_blob_files_are_segment_scoped`、`test_compaction_reclaims_unreferenced_blobs`、`test_upsert_after_reopen_resolves_segment_blob`。

### 3.3 N1d 退役整代 `quant.bin`（P1，2–3 人日）

- `quant_seg_<start>.bin` 升为唯一副本；删除 `quant.bin`/`quant_meta.bin` 的整代写路径（`QuantStore::write_slot` 改写到 `segment(start)`，`grow_to` 按段扩容）。
- `QuantStore::open` 按段加载；缺一段 = 该段无量化码（回退精确扫描），整代缺失才重建。
- 与 N1b 的耦合：N1b 的"只重写选中段"依赖本项，否则 `quant.bin` 整份重写把 I/O 上界顶回全量（R5）。**建议 N1b 与 N1d 同里程碑**。
- 验收：`test_quant_codes_live_in_segment_files`、`test_dropping_generation_quant_file_keeps_recall`、磁盘占用对比（同一集合前后近似减半）。

### 3.4 N2 摘掉 checkpoint 的队列门（P1，4–5 人日）

现门保护两件事（`store_maintenance.rs::checkpoint_inner`）：WAL 不被切掉排队事务；可见 tracker 必须指向已 fsync 字节。方案：

1. **冻结前沿**：`flush_data_files` 结束时取 `frozen = meta.last_applied_txn`，向下传给 `flush_raced_segments`/`flush_segment`。
2. **tracker 按冻结前沿封顶**：`write_id_tracker` 的每个条目 `point_lsn = min(point_lsn, frozen)`，`tracker.lsn = frozen`。之后新写入落在 `frozen` 之上，不属于这份 tracker。
3. **不再需要门**：`publish_id_trackers` 的 rename 变成安全操作（tracker 已封顶，它命名的字节在 `frozen` 之前且已 fsync）。
4. **截断点**：`effective = wal_pins.clamp(inner.min_persisted(frozen))`（现有 `min_persisted` + `clamp` 已具备）。
5. `meta.save` + marker append 仍在 `inner.write()` 内，微秒级；段删除（`truncate_through`）按段首 LSN 规则天然保留排队事务所在段。

验收：`MaintenanceStatus` 新增 `gate_wait_ms: u64`（默认 0，checkpoint 期间为 0）；用例 `test_checkpoint_does_not_block_writers`（8 线程持续 upsert 期间 checkpoint，断言无事务等待超时、`gate_wait_ms == 0`）、`test_checkpoint_without_gate_survives_crash_windows`（配合 §3.6 的三个窗口）。

风险：`tracker.lsn` 落后于实时状态后，重放可能给 id 分配新槽（R3 已接受该语义），需要专门用例覆盖"未关门时 checkpoint 与写入并发"。

### 3.5 N3 摘掉快照的队列门（P2，3–4 人日）

前置：N1a（向量不可变）、§3.1.1 B1（payload 不可变）、N1d（量化码按段）。步骤（`storage.rs::snapshot_capture_inner`）：

1. 取 `pinned = last_applied_txn`，`pin_wal_ack(pinned)`（已有）。
2. 复制 generation 文件树：不变文件走 `FICLONE`/硬链接（`copy_snapshot_file` 已有增量复用），尾段只复制 `≤ pinned` 的前缀（`copy_prefix` 已有）。
3. 复制 `≥ pinned` 的 WAL 段，恢复时按 `pinned` 重放补齐。
4. 唯一保留的门缩短为"取 lsn + 写清单"这一段（`compact_mutex` 仍持，避免 compaction 换代）。

验收：`test_snapshot_window_allows_writes`（窗口内 upsert 不阻塞、无 gate 等待）、`test_snapshot_restore_replays_to_the_export_point`（恢复后先断言 `= pinned` 的 count，再断言重放后 = 导出时刻最新值，两种期望都写清）、`test_verify_snapshot_accepts_prefix_copy`。

### 3.6 N4b / N5 / N6 验证与配置缺口

**N4b 绑定冒烟（2 人日）**：`bindings/{python,js,go,java,r}` 各一条 `create → upsert → maintenance_status → snapshot → snapshot_incremental → verify → restore` 冒烟；Python 的 `tests/test_smoke.py` 已具备骨架，其余按同一顺序补。CI 顺序固定为 `cargo build --features c_api` → `SIMVEC_LIB=target/debug/libsimvec.so` → 各绑定测试。

**N5 崩溃注入矩阵（4–6 人日）**：新增 `src/storage/fault.rs`：

```rust
pub(crate) fn fault_point(name: &'static str);   // 命中即 std::process::abort()
```

`SIMVEC_FAULT=<point>`（逗号分隔可多处）在首次调用时解析并缓存；`fault_point` 在 hot path 上只是一次 `Relaxed` 原子读 + `Option` 判断（默认关闭时代价可忽略，与 `lock-metrics` 同级思路）。注入点表：

| 窗口 | 点位（调用位置） | 期望 |
| --- | --- | --- |
| blob 写后、WAL append 前 | `apply_upsert_locked` 后 / `wal.append` 前 | 重开忽略孤儿 blob，不报 `CorruptData` |
| WAL fsync 后、apply 前 | `queue::flush_batch` 返回后 | 重放补上，count 正确 |
| apply 后、ACK 前 | `apply_queued_txn` 返回后 | 重放幂等，不重复计数 |
| ACK 后、truncate 中 | `wal.truncate_through` 内 | 段与 tracker 自洽，可重入 |
| checkpoint 批量刷盘后 | `flush_data_files` 后 | 只有 `.staged` tracker，open 忽略 |
| tracker 发布后、manifest 前 | `publish_id_trackers` 后 | `point_lsn` 被封顶，重放补齐 |
| 压实发布后、内存切换前 | `commit_generation` 的 `publish` 后 | poison → 重开按新 manifest 加载 |

每条用例在子进程里跑（`std::process::Command` 起自身带 `SIMVEC_FAULT`），主进程只断言重开后的状态。

**N6 性能基线（1–2 人日，计量骨架已完成，见 §1.4）**：`benches/compaction_io_vs_garbage.rs`（`autobenches = false`，已在 `Cargo.toml` 注册 `[[bench]]`），口径不依赖内部计数：

```text
bytes_written = Σ{新 generation 中 inode 不在旧集合的文件长度}
```

（`std::os::unix::fs::MetadataExt::ino()` + `data_dir()` 公开接口即可）。三个垃圾比例（0.1/0.3/0.5）各跑一次压实并打印 `bytes_written` 与 `Σ垃圾段字节`，QUICK=1 时缩小到 2k 点。**通过标准**：`bytes_written` 随垃圾段字节线性增长、不随集合总量增长；N1b 之前该 bench 就是"反面基线"，其数值应写入提交说明。

---

## 4. 修订后的里程碑

```text
M1  N1a(已完成) ──> N1b-1 payload 段内追加(B1) ──> N1b-2 段级压实 + segments.json 指纹
                                                    └──> N1d（同里程碑，否则 I/O 收益被 quant.bin 顶住）
M2  N5 注入钩子(先做) ──> N2 checkpoint 去门 ──> N5 七窗口补齐
M3  N1c blob 按段 ──> N3 快照去门
M4  N4b 绑定冒烟、N6 bench、N7(已完成)
```

工作量合计（含已完成的 N1a/N7/N4a）：约 21–31 人日；本次交付覆盖 N1a + N7 + N4(Rust 侧) 约 6 人日。

## 5. 风险清单

| 风险 | 影响 | 缓解 |
| --- | --- | --- |
| payload 原地改写（R4） | 硬链接后老 generation 读者错读 payload | B1 先行；B2 过渡方案只硬链接向量/目录/图 |
| 重放槽号漂移（R3） | 外部若把 `slot_of` 当稳定 id 会失效 | `slot_of` 不在公共文档承诺范围；`AGENTS.md` 已写明；用例锁定"目录一致"而非"槽号一致" |
| `next_slot` 语义变更 | 依赖 `next_slot == live_count` 的运维脚本/监控会有偏差 | 文档 + `MaintenanceStatus` 不暴露 `next_slot`；容量类字段统一改用 `live_count` |
| 硬链接跨文件系统 | `hard_link` 失败（staging 与 generation 同目录，理论上不会） | 失败回退为字节复制并记 `tracing::warn!` |
| 压实期间并发写 | 计划失效 | 沿用 `mutations` 计数器 + 有界重试 + 关门兜底 |

## 6. 验收命令（本阶段目标态）

```text
cargo +1.97.0 check --all-targets            # 0 错误 0 警告（本地自 1.93 起需 --ignore-rust-version）
cargo +1.97.0 test                            # 全绿，含 §1、§3 各用例
cargo +1.97.0 test --features c_api
cargo +1.97.0 test --manifest-path cli/Cargo.toml --target-dir target
SIMVEC_FAULT=<point> cargo test --test recovery_test   # 七窗口矩阵
QUICK=1 cargo bench --bench compaction_io_vs_garbage   # 字节数与垃圾量成正比
```
