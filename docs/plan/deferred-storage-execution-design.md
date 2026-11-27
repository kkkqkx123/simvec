# 延后存储重构执行设计

本文是 `deferred-storage-refactor.md` 的落地执行蓝本：逐阶段给出**现状、目标、数据布局、模块职责、状态转换、不变量、改动清单与验收**，作为代码改动的直接依据。规划文档是"做什么"，本文是"怎么改"。落地后仍成立的事实回写 `AGENTS.md`。

**本版为完整实现版**：阶段五/六/七的每一项都必须真正落地，不允许以"元数据骨架""留作后续"等方式绕过。任何一项无法完成时，必须显式报失败，而不是静默降级。

## 全局不变量（任何阶段都不得破坏）

1. **generation manifest 是唯一提交点**；发布后失败进入 poison，重开恢复。
2. **WAL 是事务顺序权威**；LSN 由 commit-queue 单一分配、无空洞。
3. **外部 id 是身份，slot 只是段内物理位置**；日志与快照不依赖未提交的绝对 slot。
4. **一个查询用一个段 guard**；一个 id 在一个 generation 内只有一个权威段归属。
5. **派生数据可丢弃重建**（字段索引、ANN、量化、postings、payload 页）；提交数据（generation/manifest/WAL/spaces.json/tracker）不可重建。
6. 所有 kernel 与 `naive` 在 1e-4 内一致；公开分数高分优先、内部距离低分优先。
7. 格式变更一次性生效；不识别的目录报 `CorruptData`，不叠兼容读分支。

## 依赖顺序与当前基线

一(CRC32C+HNSW) → 二(物理段+tracker) → 三(段级 flush+ACK) → 四(引用式 WAL+blob) → 五(named/sparse 段级+ANN) → 六(类型+multi) → 七(快照清单+C ABI)。

基线：阶段一~四已完成并通过全量测试。阶段五进行中，阶段六/七未实施。

---

## 阶段二 — dense 物理段与段级 id tracker（已完成）

id 目录权威从整代 `dense_snapshot.bin` 迁移到段级 `id_tracker.bin`；`store_open.rs` 用 `read_id_trackers` 重建目录；tracker 缺失/CRC 失败且与 generation 不符时按 `CorruptData` 拒绝；删除整代 snapshot 写入路径与模块。验收通过。

---

## 阶段三 — 段级 flush 与 WAL ACK（已完成）

删除 `note_segments_persisted` 抬平逻辑，段独立推进水位；`flush_data_files` 用 rayon 并行；`MaintenanceStatus` 增加每段 applied/persisted 并经 C ABI 暴露。验收通过。

---

## 阶段四 — 引用式 WAL 与 blob 持久化（已完成）

`wal_blobs.rs` 每段 `blobs.bin`，8 字节对齐 `[crc32c][length][version][kind][payload]`；WAL 磁盘格式改为 `WalRecordOnDisk` 引用变体，append 先 fsync blob 再写引用，replay 校验 CRC/版本/kind 后还原；`truncate()` 重置 blob；四类测试（roundtrip/损坏/存在性/段数）通过。

---

## 阶段五 — named/sparse 段级存储与 ANN（完整实现）

### 目标
把 `kind_store`/`named_store` 从"根 blob + 全量 exact"升级为**复用段基础设施的一等空间**：每空间有段目录、段级 id tracker、`spaces.json` 权威元数据，named dense 段可独立 ANN，sparse 段用 CSR + posting DF 规划。

### 数据布局
```
generations/g<id>/
  spaces.json                       # 权威：每空间 kind/distance/dim/multi/index/quant + generation/lsn
  spaces/<space>/segments/<start>/
    points.bin                      # 该段 point blob（VectorData 编码）
    payloads.bin                    # 该段 JSON payload blob
    id_tracker.bin                  # 段内 id->offset/version/free 权威目录
    postings/manifest.bin,<bucket>  # 该段 sparse posting（DF 派生）
    hnsw.bin                        # named dense 段 ANN（派生，损坏回退 exact）
```
根 `kind.bin`/`named.bin` 降级为**类型标记**（仅 `collection` + `marker`），配置权威迁到 `spaces.json`。

### 模块职责
- **新模块 `src/storage/spaces_meta.rs`**：`SpacesManifest { version, generation, lsn, spaces: BTreeMap<String, SpaceMeta> }`，`SpaceMeta { kind, distance, dim, multi_config, index, quantization }`，CRC 信封（magic `SVPC` 复用 `snapshot::write/read`）。open 校验 `generation`/`lsn`，不符报 `CorruptData`。
- **新模块 `src/storage/space_segments.rs`**：`SegmentSet` —— 段宽 `segment_slots`、`slot -> (space, start, local_offset)` 映射、每段 `SpaceSegment { points, payloads, postings, tracker, hnsw }` guard、段目录创建/打开。一个查询只取一个段 guard。
- **`kind_store`/`named_store`**：`Inner` 增加 `segments: SegmentSet`；`apply_upsert`/`replay_txn` 用段 tracker 的 version 水位并按 slot 路由到段；`checkpoint` 逐段 sync + 写 tracker + 写 `spaces.json`；`compact` 逐段重写 live 到新 generation；`read`/`score` 按段 guard 走 exact 或 ANN。
- **`postings.rs`**：posting 记录 term → `(local_offset, weight)`，manifest 存 term DF（派生，损坏重建）；`prefer_posting_scan` 用 DF/IDF 规划。
- **`config.rs`**：完整校验组合 —— dense 支持 HNSW/IVF/量化，sparse 仅 posting+exact，multi 仅 MaxSim；不支持组合在创建时拒绝。

### 状态转换
`upsert → blob 追加到目标段 → WAL 引用 → apply 段 tracker`；`checkpoint → 逐段 sync(blob/posting/tracker/ANN) → spaces.json → manifest → ACK → WAL truncate`；`open → spaces.json → 逐段 tracker → WAL replay(按 version 水位) → 派生索引按需重建`。段 ANN/posting 损坏只降级该段到 exact，不改候选资格。

### 不变量
每空间配置以 `spaces.json` 为唯一权威；一个 id 在一段内只有一个权威 offset；段 tracker 的 version 是 WAL replay 水位；派生 ANN/posting 可丢弃重建，tracker/spaces.json 不可。

### 验收
named/sparse 与 exact baseline 逐点一致；混合过滤与全量后置过滤一致；每空间独立 checkpoint/重开；posting/ANN 损坏只失加速；`spaces.json` 校验位翻转/paths 越界报 `CorruptData`。

---

## 阶段六 — 向量类型与 multivector（完整实现）

### 目标
类型标签进入段 blob 头、段向量文件头与 WAL 引用，支持 `f32/f16/uint8/binary/turbo/multi`；`naive` 为唯一正确性基线；multivector 用 MaxSim。

### 数据布局
- 新增 `src/types/config/dtype.rs`：`VectorDataType { F32, F16, Uint8, Binary, Turbo, Multi }`，`as_name()`。
- blob 记录头：`[magic][version][dtype u8][dim u32][scale f32][zero_point f32][payload]`；CRC 覆盖全头+数据。
- `VectorSpaceConfig` 增加 `#[serde(default)] dtype: VectorDataType`，默认 `F32`；`VectorData`/`VectorView` 携带 dtype。
- WAL 引用 blob 的 `kind` 字段扩展携带 dtype。

### 模块职责
- **`distance/naive.rs`**：新增 `distance_typed(metric, dtype, a: &[u8], b: &[u8]) -> Result<f32>` —— f16 半精度转 f32、uint8 反量化 `(q - zero_point) * scale`、binary 用 popcount(Hamming→按 metric 映射)、turbo 仅在语义可映射时按 f32、multi 走 MaxSim；所有输出与 f32 基准在 1e-4 内。
- **`distance/{avx2,avx512,neon,portable}.rs`**：按类型新增 kernel；不满足 1e-4 一致性时该类型 kernel 关闭并回落 naive。保留 `kernel::all_kernels` 交叉一致性测试。
- **`distance.rs`**：`distance_typed` 按 dtype 分发；`to_score` 不变。
- **`config.rs`**：按空间声明允许的 dtype×metric 组合，创建时拒绝不支持组合。
- **`kind_store`/`named_store`/`dense`**：读取时先按 blob 头 dtype 解码再进 kernel；multivector 子向量数写入 tracker，MaxSim 走 typed kernel。

### 不变量
`f32` 缺省且行为不变；`naive` 是所有类型基准；类型标签损坏报 `CorruptData`；公开分数方向不变。

### 验收
每 dtype×每 metric 的 naive 对照；SIMD 与 naive 1e-4；multivector MaxSim 与 exact 一致；dtype 标签位翻转报 `CorruptData`。

---

## 阶段七 — 快照清单与 C ABI（完整实现）

### 目标
`SnapshotInfo` 聚合段级文件清单与 WAL 水位；C ABI 四函数与全部绑定完整可用。

### 数据布局
- `SnapshotFile` 增加 `space: Option<String>`、`segment: Option<u32>`、`generation`（每段文件绑定 generation，WAL 文件绑定其 generation 与首 lsn）。
- `SnapshotInfo` 增加 `segments: Vec<SegmentSnapshotInfo>`（每段 `start`/`space`/文件清单/`persisted_lsn`）、`wal_floor: u64`。
- 文件 `version` 来自 `segments.json` 与各段 manifest，不再用 generation id 统一代替。

### 模块职责
- **`storage.rs`**：`snapshot_capture` 遍历 generation + 每空间每段的 manifest 收集 `{version,len,crc}`；`snapshot_to_incremental` 按 (path,version,len,crc) 四元组硬链接；`verify_snapshot`/`restore_snapshot` 校验 CRC 与路径穿越。
- **`c_api/snapshots.rs`**：四函数保持不变；`include/simvec.h` 用 cbindgen 重生成（不可用则手工同步并注明）。
- **绑定**：`engine.pyi`/`engine.d.ts` 补四函数声明；python/js/go/java/r 实现已在 matrix 中登记，逐一核对存在；wasm 不暴露。

### 验收
位翻转/路径穿越/缺文件/版本不匹配/目标已存在全部拒绝；第二次快照只硬链接未变段；C ABI 快照可从 Rust 集合恢复并搜索。

---

## 实施顺序与验证节奏

每阶段：改代码 → `cargo test --ignore-rust-version -p simvec` → 修复 → 进入下一阶段。全部阶段结束后：全量测试 + `cargo build --features c_api` + 回写 `AGENTS.md` 与 `docs/binding-matrix.md`。任何阶段不允许半落地或以"后续"名义绕过；确无法完成时显式报错并停止。
