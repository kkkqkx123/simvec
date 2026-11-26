# 存储正确性修复计划（11）

本文件把存储层评审（`/workspace/simvec-存储结构评审.md`，对照 `ref/qdrant` 的逐结构比对）收敛为本仓库内可执行的问题清单与修改方案。所有改动遵循 `docs/plan/README.md` 的实施约束：**在现有模块内原地重构、不新建 v2 替代模块、不做向后兼容与旧数据迁移、编译与测试集中执行**。

排序原则：先修"静默损坏数据"的缺陷，再修"空间与启动开销"，最后收敛演进能力；"定位差异"类（无分片、无 segment optimizer、不压缩倒排）不在本清单内。

## 问题清单

### P0 正确性（可复现，必须先修）

| 编号 | 问题 | 位置 | 现象 | 影响 |
|---|---|---|---|---|
| P0-1 | 墓碑只存在内存，checkpoint 截断 WAL 后丢失 | `CollectionStore::checkpoint()`、`CollectionStore::open()`、`apply_delete_slot_locked()` | 删点 → `checkpoint()` → 重开，`get(被删 id)` 返回 `Ok(Some(_))`，`count()=2` 而 `scroll` 返回 3 条 | 已删除数据静默复活，计数与枚举结果自相矛盾 |
| P0-2 | payload 块分配跨页，写页不做越界检查 | `BitmaskInner::find_free_range()`、`Pages::write_value()` | 8 块/页场景下页文件被写到 1376 字节（`page_size=1024`），同进程读该点报 `payload read out of bounds`，重开后又"恢复正常" | 持久化损坏 + "块号↔页内偏移"不变式被破坏，后续分配与统计漂移 |
| P0-3 | `payloads.bin` 是已迁移完成却没下线的僵尸链路 | `storage/payloads.rs`、`open_with_storage()` 必需文件列表、`checkpoint()` / `ensure_capacity()`、`compaction_footprint()` | 文件按 12 B/slot 预分配、每次删除写一次，但 `open` 从不读它（墓碑位图从全零开始） | 三份真相（内存位图 / flag / tracker）无谓磁盘与搬迁成本，且是 P0-1 的诱因 |
| P0-4 | `meta.bin` 无校验和；损坏即整机起不来 | `Meta::save` / `Meta::load`、`LocalVectorEngine::open_with_storage()` | 唯一不可重建的文件是唯一没有 CRC 的文件；引擎 `open` 对集合错误直接 `?` 上抛 | 单字节翻转 = 该集合永久无法打开，且一个集合坏会导致整个引擎无法启动 |
| P0-5 | 容量增长崩溃窗口使 `vectors.bin` 与 meta 长度不符 | `ensure_capacity` → `Vectors::open()` | `grow_to()` 先 fsync 文件，`meta.save` 在之后；两次 fsync 之间崩溃即长度不等，`Vectors::open` 要求严格相等 | 与 P0-4 同一爆炸半径：集合永久打不开 |

### P1 效率（第二批，影响规模上限）

| 编号 | 问题 | 位置 | 量化 |
|---|---|---|---|
| P1-1 | 块分配每次都从 block 0 线性扫描整张位图 | `BitmaskInner::find_free_range()` | 每次 `put` 为 O(total_blocks)，批量导入退化为全表二次复杂度 |
| P1-2 | `open` 重复打开 payload store、重复读 payload | `open_with_storage()` 的重建循环；重建 filter/索引各读一遍 | 启动 O(数据量) 且常数翻倍 |
| P1-3 | `checkpoint` 全程持写锁（12 MB tracker + 位图 + 全页 fsync 在锁内） | `CollectionStore::checkpoint()` | 检查点期间写入完全冻结 |
| P1-4 | 向量段大小固定 8192 slot，不随维度缩放 | `meta::SEGMENT_SLOTS_DEFAULT`、`Vectors::grow_to` | 768 维时 25.2 MB/段；1 万点集合为 1 个点也要整段扩容 |
| P1-5 | `FilterBitmap` 对每个 payload 字段的每个取值自动建 `BitVec(capacity)` | `FilterBitmap::register_slot()` | 1M 容量下 1 万个不同取值 = 1.25 GB 常驻 |
| P1-6 | `reverse` 以 `PointId`（数字 id 已 `to_string`）为 key | `init_slot_locked()` | 1M 点约 50-70 MB 常驻，且每次 `open` 全量重建 |

### P2 演进（本轮不做，仅记录）

- **P2-1** 压实后立即丢弃已发布索引（`compaction.rs` 中 `self.index.store(Arc::new(None))`），任何压实都会让 ANN 退回精确扫描。
- **P2-2** `BlobDirectory` 的 `rec_size` 是编译期常量而非文件头字段，未来给记录加字段必须整目录重建。
- **P2-3** `kind`/`named` 的倒排没有落盘结构，重启靠全量快照 + WAL 回放。
- **P2-4** WAL 单文件不分段，且每条 upsert 内联完整向量与 payload。

## 修改方案

### P0-1：墓碑落盘为 `tombstones.bin`

- 在 `storage/tombstones.rs` 内新增持久化，不新建模块：
  ```text
  [0..4)  magic "VSTB"   [4..8) version u32=1   [8..12) crc32(bits)   [12..20) slot_capacity u64   [20..) bits
  ```
  `TombstoneBits::save(dir, capacity)` 走 tmp + fsync + rename + fsync(parent)，与 `meta.bin` 同构；`TombstoneBits::load(dir, capacity)` 校验 magic/version/CRC 并按 meta 容量 resize。
- `checkpoint()` 顺序改为：payload store → `keys`/`vectors` → **`tombstones.bin`** → `meta.bin` → WAL marker → `truncate`。墓碑必须早于 meta 与截断，保证"日志清零后墓碑仍在"。
- `open()` 从 `tombstones.bin` 读回位图，不再用 `bitvec![0; ...]`；`replay_wal` 在其之上幂等重放删除。
- 顺带修掉同一处的反查表缺陷：原 `open()` 把**所有** `keys.bin` 记录（含墓碑槽）都塞进 `reverse`，于是"删除 → 重启 → 重新写入同一 id"会命中旧墓碑槽，点写入后依旧不可见。现在墓碑槽不进入 `reverse`，重新写入会分配新槽。
- 缺失 + `meta.tombstone_count == 0` → 视为空位图（覆盖新建与空集合）；缺失但 meta 声称有墓碑，或 CRC 不匹配 → 返回 `CorruptData`，由引擎隔离该集合，**绝不静默复活**。
- 压实提交时（`commit_compaction`）在 `meta.save` 之前落一次墓碑，避免"墓碑归零 + WAL 截断"后重启读回旧位图。

### P0-2：payload 分配器页内约束

- `BitmaskInner::find_free_range` 改为**逐页扫描**：run 必须完全落在单页内，`count > blocks_per_page` 直接拒绝（单条 payload 大于一页无法表达，应报错而不是写坏文件）。
- `allocate` 增加循环：页内找不到连续 run 就扩一页再试，而不是只扩一页就假定成功（原实现 `count > blocks_per_page` 时会越界置位）。
- `Pages::write_value` 增加 `start + len <= page_size` 断言，并校验已存在页长度不小于 `page_size`；错误走 `CorruptData`。
- `MmapBitmask::open` 增加 `page_blocks == config.blocks_per_page()` 校验，防止配置漂移后按错页几何分配。
- 顺带解决 P1-1：位图内新增 `next_fit` 游标，分配从上次位置继续；`free_blocks` 把游标回退到被释放区间起点。

### P0-3：删除 `payloads.bin` 全链路

按"不做向后兼容"直接删除，涉及：

- 删除 `storage/payloads.rs`，`storage.rs` 去掉 `mod payloads` 与 `payloads: Payloads` 字段；
- `store_open.rs`：create 不再建文件，open 的必需文件列表去掉该项；
- `store_maintenance.rs`：`checkpoint` 去掉 `payloads.sync()`，`ensure_capacity` 去掉 `grow_to`，`memory_estimate` 的目录占用从 20 B/slot 改为 8 B/slot；
- `store_write.rs`：删除路径不再写 flag（墓碑已由 P0-1 的位图承担）；
- `compaction.rs`：`compaction_footprint` 去掉该文件；
- `directory.rs`：删除 `SLOT_REC_SIZE` / `FLAG_TOMBSTONE` / `set_flags`，文档改为只描述 `keys.bin`。

### P0-4：`meta.bin` 加 CRC 信封 + 引擎隔离

- `meta.rs` 改用信封：`magic "VSM2"` + `envelope_version u32` + `crc32(body)` + `postcard(Meta)`，与 `snapshot.rs` 同构；`load` 对 magic/CRC 不匹配返回 `CorruptData`，并保留 `Meta::validate` 的语义检查。
- `engine/core.rs` 的集合加载改为逐集合容错：任一 `open` 失败 → 把目录改名为 `<name>.suspect-<unix_ts>`（加载扫描本就跳过 `.suspect-`），记录 `warn` 日志，继续加载其余集合。只有目录枚举本身失败才让 `open` 失败。

### P0-5：`Vectors::open` 容忍长度偏差

- `file_len >= expected`：接受，只映射 `expected` 字节（多出的尾部视为分配残留并记一次 `warn`）。
- `file_len < expected`：`set_len(expected)` 补齐为零（meta 声明容量是权威，缺失槽位即未写入），记 `warn`。
- 这样"先扩文件后落 meta"与"先落 meta 后扩文件"两种崩溃顺序都能启动，`ensure_capacity` 的顺序不再需要调整。

### P1 本轮实施

- **P1-1**：随 P0-2 的 `next_fit` 游标一并完成。
- **P1-2**：`open` 只打开一次 `PayloadStore` 并复用于重建循环；filter 位图与 payload 索引在同一个 slot 循环里注册（原本已复用同一次读取，去掉重复 open 即达标）。

### P1/P2 未纳入本轮

以下五项已由 [`12-storage-efficiency-and-evolution.md`](12-storage-efficiency-and-evolution.md) 接手并实施，本文件不再重复：

- **P1-3** checkpoint 拆成"无锁落盘 + 短锁水位"两段（冻结面从读写全停缩到只停写）。
- **P1-4** 向量段大小按维度反算（目标 4 MiB/段）。
- **P1-5** `FilterBitmap` 分块增长 + 字段预算 + 超预算字段安全降级为后置过滤。
- **P1-6** `reverse` 拆成数值/字符串两张表。
- **P2-1** 压实后按槽位映射重编号并重新发布索引，不再丢弃。
- **P2-2** 目录记录长度写入文件头。

仍在 12 中仅作设计、未实施：**P2-3**（`kind`/`named` 倒排落盘）与 **P2-4**（WAL 分段 + 向量外置）。

## 回归测试

新增两个用例文件（此前为临时探针，现转为仓库内回归）：

- `tests/tombstone_checkpoint.rs`：3 点 → 删 1 点 → `checkpoint()` → 重开，断言 `get(被删 id) == None`、`count() == scroll 条数`、`meta.tombstone_count == 1`；另有"重开后继续删点 + 再 checkpoint + 再重开"的二次验证（墓碑文件可重复写），以及"重开后重新写入被删 id 可见"的反查表验证。
- `tests/payload_page_boundary.rs`：`page_size=1024`（8 块/页）下构造"页内已满、只剩跨页空隙"的分配压力，断言所有页文件长度恒等于 `page_size`、同进程内 `get()` 不报错、重开后 payload 长度不变。

单元测试补充：

- `bitmask.rs`：`allocate` 不跨页（跨页空隙被跳过）、`count > blocks_per_page` 报错、`next_fit` 复用刚释放的块、`total_blocks` 不足以容纳时按页增长。
- `pages.rs`：写入越界被拒绝且页文件长度不变。
- `meta.rs`：单字节翻转被 CRC 拒绝。

## 验收

- `cargo test --all-targets --features c_api` 通过（含新增用例）。
- 上述两个 P0 现象在新增测试中"修复后不再出现"，且测试在旧实现上会失败（已在评审阶段用探针验证过）。
- `payloads.bin` 不再被创建、读取或计入空间估算。
- 手工改写 `meta.bin` 任意字节：该集合被改名为 `.suspect-<ts>`，其余集合同样可打开。
- `vectors.bin` 比 meta 声明更长或更短时均可打开，并记录一次 `warn`。

## 实施状态（本轮已落地）

| 编号 | 状态 | 落地点 |
|---|---|---|
| P0-1 墓碑落盘 | 已实施 | `storage/tombstones.rs` 新增 `tombstones.bin`（`VSTB` + CRC + 容量）；`checkpoint()` 在 `meta.save` 与 WAL 截断之前落盘；`open()` 读回；压实提交前重写 |
| P0-2 分配不跨页 | 已实施 | `find_free_range()` 逐页扫描并带 next-fit 游标；`allocate()` 拒绝大于一页的请求；`write_value()` 增加页内断言；`MmapBitmask::open()` 校验 `page_blocks` |
| P0-2 补充：超大 payload 准入 | 已实施 | `validate_payload_size()` / `validate_payload_len()` 在 WAL 追加前拒绝超页 payload，避免写进日志后永远无法回放 |
| P0-3 删除僵尸文件 | 已实施 | 删除 `storage/payloads.rs`、`Payloads` 字段、`open` 必需文件项、`grow_to`/`sync`/footprint/flag 写入，以及 `directory.rs` 的 `SLOT_REC_SIZE`/`FLAG_TOMBSTONE`/`set_flags` |
| P0-4 meta 校验 | 已实施 | `meta.bin` 改为 `VSM2` + 信封版本 + CRC32；`LocalVectorEngine` 逐集合容错，失败目录改名 `.suspect-<ts>` 后跳过 |
| P0-5 容量崩溃窗口 | 已实施 | `Vectors::open()` 双向容忍：短则补零、长则忽略尾部，容量以 meta 为准 |
| P1-1 分配 O(n) 扫描 | 已实施 | 随 P0-2 的 next-fit 游标完成，`free_blocks` 回退游标以复用刚释放的区间 |
| P1-2 open 重复打开 | 已实施 | `open_with_storage()` 只打开一次 `PayloadStore`，重建循环复用之 |
| P1-3 / P1-4 / P1-5 / P1-6 | 已转 12 实施 | 见 `12-storage-efficiency-and-evolution.md` |
| P2-1 / P2-2 | 已转 12 实施 | 同上 |
| P2-3 / P2-4 | 未实施 | 12 中给出分阶段方案与触发条件 |

### 测试结果

- `cargo test --all-targets --features c_api --ignore-rust-version`：**全部通过**（307 个单元测试 + 13 个集成测试目标，含新增 3 个）。
- 新增用例：`tests/tombstone_checkpoint.rs`（4 条）、`tests/payload_page_boundary.rs`（2 条）、`tests/engine_quarantine.rs`（1 条）；单元测试补充在 `tombstones.rs`（4 条）、`meta.rs`（3 条）、`bitmask.rs`（3 条）、`pages.rs`（1 条）。
- 原先断言"`vectors.bin` 长度不符即拒绝"的用例改为断言"短则补齐、长则忽略"，与新契约一致。
