# 存储剩余任务（E1–E4）设计与落地方案

> **已被 `docs/plan/storage-next-phase.md` 取代**（作为下一阶段任务文档）。本文保留为 E1–E4 的阶段记录；其中 §E1 关于“段目录（catalog）是段级压实前置条件”的结论已被推翻：段地址固定 + 段内紧凑即可，无需 catalog，推导见 `storage-next-phase.md` §2.3。

本文接 `docs/plan/storage-remediation-tasks.md`：A/B/C/D 四组（缺陷修复、遗留清理、文档订正）已落地，剩下 E1–E4 四项能力缺口。事实源仍然是 `AGENTS.md` 的存储章节；本文只描述**改什么、为什么这样改、改完怎么验收**。

## 结论速览

| 编号 | 内容 | 本轮 | 说明 |
| --- | --- | --- | --- |
| E1 | 压实整代重写 + slot 全局重排 | **只出设计，不动代码** | 前置条件（已发布段不可变）不成立，任何"只重写脏段"的实现都会把并发写泄漏进新代或退化成等价拷贝 |
| E2 | 关门后全量 re-flush | **已实施**（窗口从"全库"降到"竞态段"） | gate 仍在；去掉 gate 的前置条件见 E2b |
| E3 | 快照全程持队列门 | **已实施**（读锁移除 + reflink 快路径） | gate 仍在，理由与 E1 同源 |
| E4 | CLI 缺 `maintenance_status` 与快照命令 | **已实施** | 5 个命令，含 CLI 冒烟用例 |

---

## E1 压实仍整代重写、slot 全局重排

### 今天的机制与代价

`CollectionStore::compact`（`src/storage/compaction.rs`）的做法是：把全部存活点按旧 slot 升序压成 `0..live_count` 的连续编号，然后在 `.g*-staging/` 里**整代重建**所有文件——每个段的 `vectors.bin`、`keys.bin`、`payloads_store/`、`tombstones.bin`、`id_tracker.bin`，整代的 `quant.bin`/`quant_meta.bin`、`segments.json`、`meta.bin`、字段索引，以及每个段的 HNSW/IVF 图。

代价有两层：

1. **写放大与总大小成正比**，与垃圾量无关。删掉一个点也触发整代重写（`compact` 只要 `tombstone_count > 0` 就会走完全流程）。
2. **重排让所有派生结构失效**：HNSW/IVF 要按新编号重建，字段索引、filter bitmap、pending 队列、`reverse` 映射全部重建。

### 为什么"只重写有垃圾的段"不能直接做

这是本轮最关键的结论，写在这里避免下一轮再走一遍弯路。直觉方案是"未变化的段直接复用（硬链接）进新代"，但它在当前布局下**不成立**：

- **已发布的段不是不可变的。** `apply_upsert_locked`（`src/storage/store_write.rs:340`）对已存在的 id 直接 `vectors.write_slot(slot, ..)`——原地改写**任意**段的向量区；`write_payload_at` 同样原地改 payload 页；`quant.bin` 更是整代唯一的可写代码数组（`AGENTS.md` 已登记，T9 才拆分）。因此一旦把某段硬链接进新代，之后对旧代的写会同时出现在新代里——新代不再是"某一时刻的快照"，manifest 会指向被并发改写的字节。
- **改成复制就没有收益。** 复制一遍全部段 = 今天的整代重写，成本不变。
- **段边界是 slot 地址的一部分。** `slot = 段起点 + 段内偏移`，段起点恒为 `index * segment_slots`。跳过某个段、或让段长度变化，都会让 `slot → (段, 偏移)` 的换算失效，读路径、payload store 分片、ANN 的 `slot_base` 全部要改。

结论：E1 不是"在 compaction.rs 里加个 if"，它要求**点级原地写改为追加写**（点的新版本落到新位置，旧位置只留墓碑），再引入"段目录"把逻辑区间映射到物理段，之后压实才能退化为"合并/丢弃物理段"。这是格式级改造，必须先设计评审再动代码，本轮不写半成品——半成品压实正是最难在事后发现的那类缺陷（只有在并发写 + 崩溃重启的组合下才暴露）。

### 需要的格式变更

| 变更 | 内容 | 影响面 |
| --- | --- | --- |
| 段目录（catalog） | 新增 generation 内的 `segments/catalog.bin`：有序条目 `{slot_base, slot_len, physical_id, tombstone_ratio}`，读路径按 `slot_base` 二分定位物理段，取代 `slot / segment_slots` 的算术换算 | `segments.rs`、`vectors.rs`、`keys.rs`、`segment_payloads.rs`、`segment_hnsw.rs`、`segment_ivf.rs`、`compaction.rs` |
| 物理段可换名 | 段目录里的 `physical_id` 与目录名解耦，压实可以写 `seg-<id>/` 新目录再原子改目录，无需整代重建 | `generation.rs`（提交点改为"目录 + manifest 双写"） |
| 追加写 | upsert 命中已有点时不再 `write_slot` 原地改写，而是分配新位置 + 旧位置墓碑；`id_tracker` 记录最新位置 | 写路径、`tombstones`、`free list`（当前不存在，需要一并引入） |
| 段级回收 | 段内存活率低于阈值时，只重写该段（`slot_base`/`slot_len` 不变，段内紧凑）；整段死亡则从目录摘除并删目录 | `compaction.rs` 重写为 `compact_segments(selection)` |
| 退役 `quant.bin` | 删除整代可写的 `quant.bin`，只保留 `quant_seg_<start>.bin`（AGENTS.md 已记为 T9 的重复存储） | `quant.rs` |

其中"追加写 + free list"是前置条件：没有它，段永远是脏的，段级压实就没有可复用的输入。

### 分阶段落地

| 阶段 | 内容 | 完成判据 |
| --- | --- | --- |
| E1a | 引入段目录与 `physical_id`，读写路径全部走目录；一次性切换，不留算术换算法（当前无旧数据需要兼容） | 全部现有测试通过；`segment_slots` 只作为"新段默认长度"存在 |
| E1b | 点级追加写 + free list；`write_slot` 从写路径消失 | upsert 已有点不修改任何已发布段的字节（可用文件 mtime/inode + 内容哈希断言） |
| E1c | `compact()` 改为段级：只重写目录里被选中的段，未选中段的文件**硬链接**进新代 | 压实 I/O 上界 = 选中段字节数；未选中段的 inode 在新代中保持不变 |
| E1d | 退役 `quant.bin`，量化只保留按段副本 | 量化集合的磁盘占用减半，`quant_meta.bin` 只描述段副本 |

### 验收与不变量（E1 整体）

- 压实期间允许并发写：提交前用 `mutations` 计数器 + 目录版本校验，竞态则重排一次（沿用现有 `COMMIT_ATTEMPTS` 结构）。
- 崩溃一致性：目录与 manifest 的提交点仍是单个原子 rename；任何时刻打开只能看到"旧目录 + 旧 manifest"或"新目录 + 新 manifest"。
- 段级 I/O 上界：一次压实触碰的字节数不超过 `Σ(选中段字节数) + 目录/manifest`。
- 反例必须被拒绝：目录条目重叠、`slot_base` 不连续增长、`physical_id` 重复，一律 `CorruptData`。

### 风险

- 这是本仓库存储层最大的一次结构变更，无法用"加开关"的方式灰度（禁止版本分支），只能一次性切换。建议独立分支 + 独立评审 + 专门的并发/崩溃测试，不塞进当前的整改节奏。

---

## E2 关门后全量 re-flush

### 已实施

`checkpoint_inner`（`src/storage/store_maintenance.rs`）本来就是两段式：不关门时全量刷盘，关 gate 后"再刷一遍"再发布水印。问题在于关门后调的还是 `flush_data_files`——**全量**，于是关门的窗口里又跑了一遍每段的 fsync。

本轮改动：

- 新增 `raced_segments(applied, persisted, count, floor)`：返回 `applied > persisted` 的段。缺条目按"脏"处理（`persisted` 缺省 0，`applied` 缺省取 `last_applied_txn`），宁可多刷一次，不会漏刷。
- 新增 `flush_raced_segments`：关门复刷改调它。没有竞态段时直接返回，关门窗口只剩"写锁 + `meta.bin` + 标记 append + truncate"。
- `flush_segment` 增加 `#[cfg(test)]` 计数 `segment_flushes`，用例 `test_checkpoint_second_phase_flushes_only_raced_segments` 断言"相邻两次 checkpoint 各自只花一段一次刷盘"，回归到全量 re-flush 会直接失败。

为什么安全：`segment_applied` 在 apply 事务的同一个 `inner.write()` 临界区里推进（`store_write.rs:316`），`persisted` 只在 `flush_segment` 成功后推进。gate 关闭后不再有 apply，所以 `applied == persisted` 的段在批量刷盘之后**没有任何未落盘的写入**；`persisted < applied` 的段正好是"批量刷盘期间被写到的段"。

### E2b（未做）：去掉 gate 本身

gate 现在保护两件事：WAL truncate（不能把排队中的事务切出日志）与 tracker 的可见性（可见的 tracker 必须指向已 fsync 的字节）。要做到完全不关门，需要：

1. **冻结前沿**：checkpoint 取一个 `frozen = last_applied`，tracker 的 `lsn` 记为 `frozen`，之后新写入不进该 tracker（现在的 `write_id_tracker` 读的是实时状态）。
2. **truncate 只由 `min_persisted` 决定**：已有 `wal_pins.clamp(min_persisted)`，但还要保证"排队中事务所在段不会被删"——`truncate_through` 已按段首 LSN 判断，尾部段永远保留，成立。
3. **marker 与 worker 的 LSN 顺序**：marker 在 `inner.write()` 内分配并 append，worker 也在这个锁下分配 LSN，顺序天然一致。

风险点集中在第 1 条：tracker 一旦允许"落后于实时状态"，open 侧的 `tracker.lsn` 上限裁剪（已实现，`read_id_trackers` 把 `point_lsn` 裁到 `tracker.lsn`）必须配合"重放会重新分配 slot"的语义，需要新的并发测试。**列为独立的下一轮任务，不在本轮改。**

---

## E3 快照全程持队列门

### 已实施

`snapshot_capture` 原来同时持有 `compact_mutex`、`queue.quiesce()` 和 `inner.read()`，三者都覆盖整个复制窗口。本轮：

- **移除 store 读锁**。复制读的是文件树，从不读 `inner`；而所有能移动这些文件的操作（压实、checkpoint、索引/量化构建）都先拿 `compact_mutex`（`compaction.rs:374`、`store_maintenance.rs`、`index_lifecycle.rs:531/602/669`），gate 又停住了 applier。读锁在这里唯一的实际作用是把**读请求**一起挡住——一个大集合的快照会让 search/scroll 冻结整个复制时长。
- **新增 reflink 快路径** `copy_snapshot_bytes`/`clone_file`（`FICLONE`）。关门窗口的长短等于复制耗时，而在支持 reflink 的文件系统（btrfs/xfs/部分配置的 ext4）上克隆是元数据操作，几 GB 段几乎瞬时完成；失败时删掉空目标文件并回落到 `std::fs::copy`。`restore_snapshot` 也改走同一路径（恢复不受 gate 约束，但快照/恢复对称更简单）。
- 用例 `test_snapshot_does_not_hold_the_store_read_lock`：在复制窗口内用 `snapshot_with_probe` 断言读锁空闲、`count()`/`scroll()` 可用（持锁时会死锁或失败）。

### E3b（未做）：把 gate 也摘掉

仍然不成立，原因与 E1 同源：generation 内的每个段都可以被原地改写（`write_slot` / payload 页 / `quant.bin`）。关掉 gate 后，复制过程中"某点被 upsert 原地覆盖"会让快照出现 `lsn` 之后的字节，且 WAL 副本里没有对应事务，重放补不回来。彻底解决要等 E1b（点级追加写）之后：届时"已发布段不可变"，快照可以只锁极短的元数据窗口，其余全部无锁复制。

---

## E4 CLI 缺 `maintenance_status` 与快照命令

### 已实施

`cli/src/commands.rs` 新增 5 个命令，直接调用公共 Rust API（CLI 不经过 C ABI，见 `AGENTS.md` 的 CLI 边界）：

| 命令 | 映射 | 输出 |
| --- | --- | --- |
| `maintenance_status <collection>` | `LocalVectorEngine::maintenance_status` | `MaintenanceStatus` JSON（`healthy` / `persisted_lsn` / `wal_blob_bytes` / `error` / `segments`） |
| `snapshot <collection> <dest>` | `snapshot_collection` | `SnapshotInfo` JSON（`generation` / `lsn` / `reused_files` / `files`） |
| `snapshot_incremental <collection> <base> <dest>` | `snapshot_collection_incremental` | 同上，`reused_files` 反映硬链接复用数 |
| `verify_snapshot <src>` | `storage::verify_snapshot` | 同上；失败时非零退出并带 CRC/路径错误 |
| `restore_snapshot <src> <dest>` | `restore_snapshot` | 无输出（成功即 0） |

设计约定：

- 命令名与 C ABI 的 `simvec_*` 一一对应，`maintenance_status` 与 `metrics` 并列，避免把"状态"和"指标"混在一个输出里。
- 快照路径一律是**引擎根目录之外**的绝对/相对路径：根目录下任何带 `generation.bin` 的目录都会在下次 `LocalVectorEngine::open` 时被当成集合加载（`engine/core.rs:54`）。这条写进了 CLI 冒烟用例的注释，避免使用者把快照放进数据目录。
- `restore_snapshot` 的目标必须不存在（实现里 `dest.exists()` 即报错），这是"不把两代数据和两套 WAL 混在一个锁文件下"的硬约束。
- CLI 命令表 `COMMANDS` 与 `.help` 自动同步；`docs/plan/standalone-cli-shell.md` 的命令映射表已补齐。

用例 `cli/tests/smoke.rs::snapshot_commands_round_trip`：create → upsert → `maintenance_status` → `snapshot` → `verify_snapshot` → `snapshot_incremental` → `restore_snapshot` → 新进程 `count` 恢复出的集合。

### 仍未做的部分

`docs/plan/standalone-cli-shell.md` 里规划但尚未实现的命令（如集合重命名、`open`/`close` 语义、批量导入导出）仍按该文档的框架排期，不在 E4 范围内。

---

## 验收方式与结果

| 命令 | 结果 |
| --- | --- |
| `cargo +1.97.0 check --all-targets` | 通过，无警告 |
| `cargo +1.97.0 check --features c_api --all-targets` | 通过（C ABI 未受影响） |
| `cargo +1.97.0 test --lib storage::` | 184 通过（含本轮 2 个新用例） |
| `cargo +1.97.0 test` | 426 lib + 12 个集成测试文件全部通过 |
| `cargo +1.97.0 test`（`cli/`） | 11 通过（lib 5 + lexer 2 + smoke 4，含 `snapshot_commands_round_trip`） |

工具链仍是仓库声明的 1.97，本机默认 `rustc` 更低，命令需显式 `+1.97.0`。

## 落地清单（本轮实际改动）

| 文件 | 改动 |
| --- | --- |
| `src/storage/store_maintenance.rs` | E2：`raced_segments`、`flush_raced_segments`，关门复刷改调增量刷盘；`segment_flushes` 计数；`read_lock_free`、`segment_flushes` 测试钩子 |
| `src/storage.rs` | E2：`segment_flushes` 字段（`cfg(test)`）；E3：移除快照窗口的 store 读锁、`snapshot_with_probe`、`copy_snapshot_bytes`/`clone_file`（reflink），`restore_snapshot` 复用同一复制路径 |
| `src/storage/store_open.rs` | E2：两处构造点补 `segment_flushes` |
| `src/storage/tests.rs` | E2/E3 回归用例 |
| `cli/src/commands.rs` | E4：5 个命令 + `COMMANDS` 列表 |
| `cli/tests/smoke.rs` | E4：快照全链路冒烟用例 |
| `docs/plan/standalone-cli-shell.md` | E4：命令映射表补齐 |
| `AGENTS.md` | checkpoint 关门复刷为增量；快照不持读锁 + reflink |
