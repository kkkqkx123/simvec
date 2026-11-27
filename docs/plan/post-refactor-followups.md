# 延后存储重构：完整性评估与下一阶段设计

本文对 `deferred-storage-refactor.md` 七个阶段的落地情况做逐项核对，给出完整性结论，并把**尚未覆盖的验收项**整理成阶段八的可执行任务。本阶段只补强验证与绑定端到端能力，**不改动已落地的数据格式与核心语义**。

## 一、完整性评估结论

### 1.1 已完整落地（无需再动）

| 阶段 | 核心实现 | 验证状态 |
| --- | --- | --- |
| 一 CRC32C + HNSW 反向邻接 | 统一 CRC32C 信封；`Node.reverse` + 加载派生 + 删除清理全部入边 | `test_healing_removes_inbound_edges_absent_from_deleted_adjacency`；旧 CRC32 `legacy_crc32_envelope_is_rejected` |
| 二 dense 物理段 + tracker | `segments/<start>/` + 段级 `id_tracker.bin`；删除 `dense_snapshot.rs` | `test_physical_segments_own_their_files`、`test_dense_snapshot_restores_id_directory`、`test_mismatched_id_tracker_is_rejected` |
| 三 段级 flush + ACK | rayon 并行 flush；段水位独立；`MaintenanceStatus` 经 C ABI 暴露 | `test_segment_wal_ack_stops_at_a_live_pin`、`test_checkpoint_flush_runs_without_the_write_lock` |
| 四 引用式 WAL + blob | `wal_blobs.rs` + `WalRecordOnDisk`；先 blob fsync 后写引用 | `test_corrupt_blob_is_reported`、`test_apply_txn_rejects_invalid_before_wal_append` |
| 五 named/sparse 段级 + ANN | `spaces.json` + `SegmentedBlobs` + per-space `SpaceAnn` + 段级 postings | `named_dense_space_builds_and_uses_ann_after_checkpoint`、`multi_recall_scores_with_max_sim` |
| 六 类型 + multivector | `VectorDataType` + `distance/dtype.rs` naive/SIMD + MaxSim + dense sidecar | `f16_dtype_is_applied_and_persisted`、`simd_kernel_matches_naive_on_decoded_dtypes`、`maxsim_matches_exact_baseline` |
| 七 快照清单 + C ABI | WAL 段按首 lsn 绑定 version；cbindgen 重生成头文件；5 语言绑定同步 | `snapshot_c_abi_roundtrip`、`test_incremental_snapshot_reuses_unchanged_files`、`binding-matrix.md` |

全量测试：414 单元测试 + 13 个集成套件全部通过（0 failed）；`cargo check` 与 `--features c_api` 均零警告。

### 1.2 剩余缺口（本阶段处理）

按优先级排列，均为**验证/矩阵补全**，不是实现缺失：

| 编号 | 缺口 | 规划出处 | 影响 |
| --- | --- | --- | --- |
| G1 | 五个语言绑定的 smoke 测试均未覆盖新增的 `maintenance_status` 与四个 snapshot API | 阶段七"C ABI 快照可从 Rust 集合恢复并搜索" | 绑定层 API 存在但与 C ABI 的一致性无端到端验证；头文件已同步但调用链未测 |
| G2 | 快照"缺文件"与"目标已存在"缺少专门拒绝测试；"版本不匹配"仅有内存快照测试 | 阶段七验收"位翻转/路径穿越/缺文件/版本不匹配/目标已存在全部拒绝" | 三类拒绝分支无回归保护 |
| G3 | 引用式 WAL 四类崩溃注入未完整覆盖（blob fsync 前/后、WAL 后 apply 前、ACK 后 truncate 中） | 阶段四验收 + 崩溃矩阵 | blob 成为恢复依据后的关键窗口缺少注入测试 |
| G4 | WAL 体积与 checkpoint 时间缺少 quick bench 相对数据 | 阶段四/WAL"用 quick bench 做相对比较" | 无性能回归基线 |
| G5 | `VectorDataType::Turbo` 在 `distance_typed` 中返回 `InvalidConfig` | 阶段六"turbo 仅在语义可明确映射时开放" | 语义不可明确映射 → 当前按配置拒绝，符合规划；需在配置校验明确记录并测试该拒绝路径（非缺陷，属文档缺口） |

明确**不重做**：数据格式兼容层、跨进程多写者、对象存储、格式迁移框架（规划"明确不做"）。

---

## 二、阶段八设计：验证矩阵补全与绑定端到端

### 2.1 目标

在不改变任何 on-disk 格式与公开语义的前提下，把七个阶段的验收矩阵补到可回归的完整状态：绑定层端到端覆盖新增 API、快照五类拒绝、WAL 四类崩溃注入、性能基线。

### 2.2 不变量

- 不修改 `src/storage/*`、`src/distance/*` 中已落地的格式与编码逻辑；本阶段只新增测试与绑定测试。
- 若测试暴露真实实现缺陷，则修复实现并同步 `AGENTS.md`，不允许通过放宽断言绕过。
- 所有新测试沿用 `cargo test --ignore-rust-version -p simvec`；绑定测试用各自工具链，不引入新依赖。

### 2.3 模块职责与改动清单

**G1 绑定端到端测试**

每个绑定的 smoke 测试新增一段：创建集合 → 写入 → `maintenance_status` 断言段水位结构 → `snapshot` → `verify` → 新目录 `restore` → 搜索还原点 → `incremental_snapshot` 断言复用。

| 文件 | 新增覆盖 |
| --- | --- |
| `bindings/python/tests/test_smoke.py` | `maintenance_status`、`snapshot_collection`、`verify_snapshot`、`restore_snapshot`、`snapshot_collection_incremental` |
| `bindings/js/tests/smoke.test.mjs` | `maintenanceStatus`、`snapshotCollection`、`verifySnapshot`、`restoreSnapshot`、`snapshotCollectionIncremental` |
| `bindings/go/engine_test.go` | `MaintenanceStatus`、`SnapshotCollection`、`VerifySnapshot`、`RestoreSnapshot`、`SnapshotCollectionIncremental` |
| `bindings/java/src/test/java/io/simvec/EngineSmokeTest.java` | 同上 Java 方法名 |
| `bindings/r/tests/smoke.R` | `simvec_maintenance_status` 等五个 R 函数 |

C ABI 侧 `src/c_api/tests.rs` 已有 `snapshot_c_abi_roundtrip`；补一个 `maintenance_status_c_abi` 断言 JSON 含 `segments` 数组。

**G2 快照拒绝测试**

在 `src/storage/tests.rs` 新增（复用 `snapshot.rs` 的 `checked_path` / `verify_snapshot`）：

- `test_snapshot_verification_rejects_missing_file`：快照清单删除一个 `files[]` 指向的文件 → `verify_snapshot` 报 `CorruptData`/I/O 错误。
- `test_snapshot_rejects_existing_target`：`restore_snapshot` 目标目录已存在且非空 → 拒绝，不覆盖。
- `test_snapshot_verification_rejects_wrong_version`：篡改文件清单 version（与 generation 不符，且非 WAL 段）→ `CorruptData`。

**G3 WAL 崩溃注入矩阵**

在 `src/storage/tests.rs` / `tests/recovery_test.rs` 新增四个窗口，逐点比较 id→slot、payload、filter、搜索：

- `test_crash_before_blob_fsync_leaves_no_wal_reference`：模拟 blob 写完未 fsync → WAL 无引用。
- `test_orphan_blob_after_wal_failure_is_ignored_on_reopen`：孤立 blob 存在但无 WAL 引用 → reopen 忽略。
- `test_crash_after_wal_before_apply_rebuilds_from_blob`：WAL 引用已落盘、内存未 apply → replay 从 blob 重建。
- `test_crash_after_ack_during_truncate_keeps_contiguous_suffix`：ACK 后截断中断 → 存活 WAL 段仍是连续后缀。

**G4 性能基线**

在 `benches/quick.rs` 增加 `wal_bytes_vs_checkpoint` quick 档：比较引用式 WAL 与分段 checkpoint 的相对体积/耗时，仅记录相对数，不与完整 bench 或跨机器数据比较。

**G5 Turbo 配置拒绝**

在 `src/types/config.rs` 增加配置校验测试：声明 `VectorDataType::Turbo` 的集合创建时报 `InvalidConfig`；在 `AGENTS.md` dtype 段落补一句"Turbo 语义不可明确映射，配置拒绝"。

### 2.4 验收

- `cargo test --ignore-rust-version -p simvec` 全绿（含新增测试）。
- 五个语言 smoke 测试各自通过（若本机缺工具链，至少静态核对函数存在并在 binding-matrix 标注未执行）。
- `docs/binding-matrix.md` 增加"新增 API 端到端测试"一列并逐语言勾选。
- `AGENTS.md` 记录阶段八新增的验证能力与 Turbo 配置拒绝事实。

### 2.5 明确不做

本阶段不新增文件格式、不改变 WAL 编码、不新增语言绑定函数、不引入兼容读取。任何"补测"发现的缺陷按真实缺陷修复，不通过放宽断言消除。
