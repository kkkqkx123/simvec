# 存储效率与演进计划（12）

本文件承接 [`11-storage-correctness-fixes.md`](11-storage-correctness-fixes.md) 中明确留下的剩余项：P1-3 ~ P1-6 与 P2-1 ~ P2-4。11 的判据是"静默损坏数据"，所以那一轮只做正确性；这一轮的判据换成两条：

1. **资源上界可计算**——每一项都能用"容量 / 维度 / 基数量"写出内存与锁持有时间的公式，而不是"感觉更省"；
2. **改动不改变查询结果**——所有优化都在"派生结构"或"索引内部编号"上做，`search` / `get` / `count` 的语义与 11 之后完全一致。

仍然遵循 `docs/plan/README.md` 的实施约束：**在现有模块内原地重构、不新建 v2 替代模块、不做向后兼容与旧数据迁移**。因此本轮有两处落盘变化：`keys.bin` 头部升到 v2（32 字节、含 `rec_size`），新建集合的 `meta.segment_slots` 改为按维度计算；旧 `keys.bin` 会在版本校验处失败并被引擎隔离为 `.suspect-<ts>`，不写迁移代码。`hnsw.bin` / `index.bin` 的**格式**没变，只是压实后文件内容被重写为新编号（原来是被删除）。

## 问题与方案总览

| 编号 | 问题（11 中的量化） | 方案 | 状态 |
|---|---|---|---|
| P1-3 | `checkpoint` 全程持写锁：12 MB tracker + 位图 + 全页 fsync 在锁内 | 拆成"无锁落盘 + 短锁水位"两段，冻结面从"读写全停"缩到"只停写" | 已实施 |
| P1-4 | 段大小固定 8192 slot，768 维时 25.2 MB/段 | 段按维度反算，目标 4 MiB/段，夹在 [1024, 8192] | 已实施 |
| P1-5 | `FilterBitmap` 对每个取值建满宽 `BitVec(capacity)`；1M 容量下 1 万取值 = 1.25 GB | 分块增长 + 字段预算 + 超预算字段安全降级为后置过滤 | 已实施 |
| P1-6 | `reverse` 用 `HashMap<PointId, u32>`，1M 点约 50-70 MB | 拆成 `HashMap<u64, u32>` + `HashMap<String, u32>`，约 20 B/点 | 已实施 |
| P2-1 | 压实后立即丢弃已发布索引，ANN 退回精确扫描 | 压实把索引按槽位映射重编号并重新发布，不再重建 | 已实施 |
| P2-2 | `rec_size` 是编译期常量而非文件头字段 | `keys.bin` 头写入 `rec_size`，读取端以文件为准 | 已实施 |
| P2-3 | `kind`/`named` 的倒排不落盘，重启靠全量快照 + WAL | 分阶段方案（见文末），本轮不实施 | 设计完成 |
| P2-4 | WAL 单文件不分段，每条 upsert 内联完整向量与 payload | 分阶段方案（见文末），本轮不实施 | 设计完成 |

---

## P1-3：`checkpoint` 拆成两段

### 现状

`CollectionStore::checkpoint()` 的顺序是 `quiesce() → inner.write() → payload store checkpoint → keys/vectors sync → tombstones → meta → WAL marker → truncate`。问题不在顺序，而在于**整个 flush 在 `inner.write()` 里**：payload store 的 checkpoint 要 fsync 12 B/slot 的 tracker（1M 点 = 12 MB）、块位图和所有脏页（默认 32 MiB/页），`keys.bin` / `vectors.bin` 各一次整文件 fsync。这段时间里 `search` / `scroll` 连 `inner.read()` 都拿不到，**读路径和写路径一起停**。

### 方案

两段式，把"重活"移出写锁：

```text
阶段 0（写锁外）：compact_mutex（维护互斥）+ queue.quiesce()（关闭写入闸门）
阶段 1（无锁）  ：inner.read() 取 capacity/live → payload store checkpoint
                  → keys.sync → vectors.sync → tombstones.save
阶段 2（写锁内）：meta.save → WAL marker（自带 fsync）→ wal.truncate
```

关键是**冻结写入的不是写锁，而是队列闸门**：`quiesce()` 已经把 `in_flight` 与 `writers` 都等到 0，阶段 1 期间没有任何事务能落到 mmap/pwrite 上，所以"无锁读文件"是安全的——被冻结的是**应用路径**，不是**文件内容**。读路径不受影响，因为读者只拿 `inner.read()` 和 `ArcSwap` 快照。

两个必须一起改的细节：

- **`meta.bin` 必须在阶段 2 现读现写**，不能沿用阶段 1 的快照。`set_quantization_config` / `set_hnsw_config` 只拿 `inner.write()`、不走队列闸门，若用快照落盘会把它们刚写的配置覆盖回旧值。阶段 2 持有写锁，读到的就是最新 meta。
- **`compact_mutex` 必须持有整个 checkpoint**。压实会重写并 rename 掉 `vectors.bin` / `keys.bin` / `payloads_store`——正是阶段 1 要 fsync 的那些文件。原来两者靠 `inner.write()` 互相串行，现在写锁只在阶段 2 出现，必须以 `compact_mutex` 补上这层互斥（顺序 `compact_mutex → 闸门`，与 `compact()` 一致，无环）。

### 不变式

- 落盘顺序仍是"数据文件 → 墓碑 → meta → marker → truncate"，任何时刻崩溃都能靠未被截断的 WAL 前滚。
- 阶段 1 看到的 `slot_capacity` 就是阶段 2 写进 meta 的值：闸门内没有任何路径能改容量（`ensure_capacity` 只在 apply 路径里被调用）。

### 验收

`checkpoint_with_probe()` 在阶段 1 入口回调；测试在回调里先断言 `store.write_lock_free() == true`（同线程 `try_write()` 成功即证明写锁未被持有），再在回调里跑 `count()` / `scroll()`——若写锁被持有，这一步会直接死锁，所以"读路径没被冻结"是被正面验证的，不是靠时序概率。

---

## P1-4：段大小按维度反算

### 现状

`SEGMENT_SLOTS_DEFAULT = 8192` 是**槽位数**，于是段的字节数随维度线性膨胀：

| 维度 | 单段字节 |
|---|---|
| 128 | 4 MiB |
| 768 | 25.2 MiB |
| 1536 | 50.3 MiB |

段的用途是"读者映射有界范围、扩容时只重映射一段"，所以有意义的量是**字节数**而不是槽位数。副作用是"1 万点、768 维的集合只放 1 个点也要占 25 MB"。

### 方案

```rust
const SEGMENT_TARGET_BYTES: u64 = 4 * 1024 * 1024;
const SEGMENT_SLOTS_MIN: u32 = 1024;

fn default_segment_slots(dim: usize) -> u32 {
    (SEGMENT_TARGET_BYTES / (dim * 4) as u64) as u32
        .clamp(SEGMENT_SLOTS_MIN, SEGMENT_SLOTS_DEFAULT)
}
```

上界仍是 8192，所以**维度 ≤ 128 的集合布局完全不变**（dim 128 恰好 8192 slot / 4 MiB），只有宽向量会变小：768 维 → 1365 slot（4.2 MiB），1536 维 → 1024 slot（6.3 MiB，受下限约束）。

`segment_slots` 早已是 `meta.bin` 的字段，`Vectors::open` / 压缩 / 量化都从 meta 读，因此这只影响**新建集合**；老集合继续用自己文件里记录的段大小，读写路径零改动。

### 验收

- 新建 768 维集合：`meta.segment_slots == 1365`，`vectors.bin` 初始长度 ≈ 4 MiB（而非 25.2 MiB）。
- 新建 128 维集合：`meta.segment_slots == 8192`（与原布局一致，全部容量相关测试不变）。

---

## P1-5：`FilterBitmap` 的成本上界

### 现状与量化

每个 `(field, value)` 一个满宽 `BitVec(capacity)`。1M 容量时单个位图 125 KiB，1 万个不同取值 = 1.25 GB 常驻，而且这个结构是**对 payload 里所有字段自动建立**的（不需要用户声明索引）。

### 方案：三层约束

1. **分块增长**：位图从空开始，按 `GROWTH_CHUNK_BITS = 4096`（512 B）为步长增长到"当前写入槽位 + 1"。只出现在前 10 万槽的取值只花 12.5 KB，而不是 125 KB。读取端一律取 `min(位图长度, capacity)` 的前缀做 OR，长度不齐不再可能 panic。
2. **字段预算**：每字段的位图预算 = `max(capacity × 32 bit, 64 KiB)`，即"32 个满宽位图"的等效空间（1M 容量 = 4 MiB/字段）。记 `values`（取值数）与 `bits`（已分配位数）两个计数器，超预算即触发第 3 条。
3. **安全降级**：超预算的字段被整体丢弃并记入 `unindexed`；此后 `build_mask` 遇到涉及该字段的条件**直接返回 `None`**，整条查询退回后置过滤。

### 为什么第 3 条必须"返回 None"而不是"返回全 0"

`build_mask` 的语义是"候选集合"，`cond_mask` 由各取值的位图 OR 而成，初始为全 0。字段被丢弃后若仍参与计算，`cond_mask` 恒为全 0，与其它条件 AND 之后候选集为空——**查询会静默返回空结果**。这是这一项唯一真正的风险点，所以"字段是否被索引"必须是查询路径可见的状态，而不能只体现在 `map.get()` 的 `None` 上。

### 不变式

- `unindexed` 单调（实例生命周期内不恢复）：位图已删，恢复需要不存在的数据。`rebuild`（open / 压实）从零开始，清空 `unindexed`。
- 只要 `build_mask` 返回 `Some`，参与计算的字段就都在预算内且位图完整覆盖到 `capacity`，所以"返回的掩码不会排除任何匹配槽位"仍然成立。
- `unregister_slot` 对已丢弃字段是 no-op（`slot_entries` 在丢弃时已同步清理），不会把空条目重新插回 `map`。

### 验收

- 容量 1M、单字段写入 4 万个不同取值后：字段被丢弃、`allocated_bytes()` 归零（位图已释放），`unindexed_fields()` 含该字段。若没有预算，这一批数据本身就需要约 100 MB。
- 同一批数据上，`build_mask` 对涉及该字段的过滤返回 `None`；低基数字段仍返回完整掩码，且精确扫描结果与无掩码路径一致。

---

## P1-6：`reverse` 按 id 类型拆分

### 现状与量化

`HashMap<PointId, u32>`：`PointId` 是 `Num(u64) | Uuid(String)`，宽度取最宽的变体 = 32 B（24 B 的 `String` + tag + padding）。hashbrown 在 ~87% 装载率下每条约 48-56 B，1M 点 = 50-70 MB，且每次 `open` 全量重建。

### 方案

`IdMap { numeric: HashMap<u64, u32>, uuid: HashMap<String, u32> }`，位于 `storage.rs`（与 `StoreInner` 同文件，不新建模块）：

- 数值 id 的 key 是 `u64`，条目约 20 B（12 B 负载 + 控制字节 + 装载率），**1M 点约 20-24 MB**；
- `String` 只为真正的 uuid 分配；
- 查询签名改成 `get(&PointId) -> Option<u32>`：数值路径 hash 一个 `u64`（原来要 hash 32 B 含 padding 的枚举），uuid 路径直接借用调用方的 `&str`，不再 `to_string()` 分配；
- 负数收益：`reserved`（in-flight 预留表）同为 `IdMap`，两条表语义一致。

`kind` / `named` 的反查表是 `HashMap<String, u32>` 且调用点大量 `point_id.to_string()` 现分配，属于 P2-3 的落盘重设计范围（那张表会被倒排索引取代），本轮不动。

### 验收

`store.slot_of()` / `get()` / 删除 / 重开后的 `reverse.len()` 与原实现一致；`IdMap` 单测覆盖数值/字符串混合、覆盖写、删除后 `len`。

---

## P2-1：压实后重映射索引而不是丢弃

### 现状

`commit_compaction` 末尾 `self.index.store(Arc::new(None))` + `discard_index_files()`：任何一次压实都让集合退回精确扫描，直到维护线程重建完（HNSW 是 `O(nodes × ef_construct × log n)`，IVF 还要重跑 k-means）。而压实做的是**紧凑化重编号**：活槽位按原顺序映射到 `0..live`，拓扑完全没变。

### 方案

在**持久化表示**上做重编号，再用它重建内存索引：

```text
PersistedHnsw::remap(map)  —— 节点按 map 改 id、邻接表翻译并剪掉死 id；
                              entry 若被删除则退化为层级最深的存活节点
PersistedIvf::remap(map)   —— slot_list 按 map 重排、截到 live 长度；
                              lists 由 from_persisted 从 slot_list 重建
```

选择"持久化表示"而不是直接改内存结构，有三个原因：

1. `to_persisted` / `from_persisted` 已有完整的结构校验，重编号后仍走同一条验证路径；
2. 得到的新 `PersistedHnsw` / `PersistedIvf` 正好就是要写盘的字节，**顺带把 `hnsw.bin` / `index.bin` 更新成新编号**，重启不必回退精确扫描（旧实现是删掉文件）；
3. 发布仍然是一次 `ArcSwap::store`——搜索要么看到旧图（旧编号，配旧的 `vectors.bin` 快照），要么看到新图，**不会读到混合代**。原地改内存结构会破坏这一点。

代价：`O(nodes × degree)` 的序列化/反序列化，仍远小于重建。

### 两个必须显式处理的边角

- **节点表有洞**（某个活点从未进入图，卡在 `pending`）→ 不重映射。洞的邻接无法凭空生成，而 `from_persisted` 要求节点表从 0 起稠密。此时保持原行为：丢索引 + 置 `needs_rebuild`。
- **`pending` 里的槽位号也是旧的** → 用同一张 `map` 过滤并翻译，否则压实后维护线程会把点插到错误的槽位。

### 验收

压实时已发布 HNSW/IVF 索引的集合：压实后 `has_index()` 仍为 `true`、`built_at_live_count` 等于压实后 `live_count`、不再置 `needs_rebuild`（维护线程不会收到重建请求）；重开后 `hnsw.bin` / `index.bin` 直接命中（不打 fallback 日志），且 `search` 结果与精确扫描的 top-k 一致。

---

## P2-2：目录记录长度写入文件头

### 现状

`keys.bin` 的 `rec_size` 是编译期常量 `KEY_REC_SIZE = 8`，读写双方靠约定对齐；将来给记录加字段（例如 flag、版本号）必须整目录重建。

### 方案

头部从 24 B 扩到 32 B，`rec_size` 成为第 3 个字段：

```text
[0..4)  magic  [4..8) version=2  [8..12) rec_size u32  [12..16) reserved
[16..24) rec_capacity u64        [24..32) blob_len u64  [32..) rec array
```

`DirView` 携带 `rec_size`，`blob(slot)` 不再接收调用方传入的步长；`BlobDirectory::open` 从文件头读取并校验 `rec_size >= 8`（至少要放得下 `{off, len}` 前缀）；`replace_from` 校验新文件的步长与当前实例一致（防止混入异种文件）；`compaction::write_dir_file` 写入同一个字段。`KEY_REC_SIZE` 退化为"创建时用的默认值"。

`version` 从 1 升到 2，配合仓库既有的"不做向后兼容"策略：旧 `keys.bin` 在版本校验处失败并返回 `CorruptData`，由引擎隔离为 `.suspect-<ts>`（P0-4 的隔离路径），不写迁移代码。

### 验收

单测：`open` 后 `DirView::rec_size()` 等于创建时声明的步长（并据此正确读出 blob）；把头部第 8..12 字节改成 4 再 `open` 报 `CorruptData`；用一个步长不同的文件 `replace_from` 也报 `CorruptData`。

---

## P2-3（本轮只设计）：`kind` / `named` 的倒排落盘

### 现状

两种非 dense 集合的状态是"全量快照 + WAL 回放"：

- `kind_snapshot.bin` / `named_snapshot.bin`（`storage::snapshot` 信封，postcard 全量 `points`）；
- 内存里 `points: Vec<Option<Point>>` + `inverted: HashMap<u32, Vec<(u32, f32)>>`（稀疏倒排）+ `norms` + `reverse: HashMap<String, u32>`。

重启 = 反序列化全部点 + 逐点重建倒排。1 亿 posting 的集合，快照是几百 MB 的整块读写，倒排重建是 O(posting) 哈希插入；checkpoint 也是"整块重写"。

### 分阶段方案

1. **倒排改成持久化的 posting 段文件**：`postings/<dim_bucket>.bin`，每段 `[u32 term][u32 count][(slot u32, value f32) * count]`，按 term 排序追加；查询按 term 二分定位。追加式写入使 checkpoint 从"重写全部"变成"flush 脏段"。
2. **快照瘦身为"目录 + 元数据"**：`kind_snapshot.bin` 只保留 `slot → id` 与 `slot → 向量指针`（进入 `vectors.bin` 风格的分段 mmap），点向量本身搬进 mmap 文件，去掉 postcard 全量反序列化。
3. **`reverse` 与 `inverted` 的键改成 `PointId` / `u32`**，消掉 `point_id.to_string()` 的每次查询分配（与 P1-6 同一手法）。

不做第 2 步而只做第 1 步收益有限（快照仍是全量），只做第 2 步则倒排仍要重建，所以两步必须同批做——这也是本轮不拆分实施的原因：它是"换存储模型"，不是"改一个模块"。

### 触发条件

- `kind`/`named` 集合的单次重启耗时成为瓶颈（可先用 `metrics` 里的 open 耗时做观测）；
- 或单个集合的 posting 数超过 1 亿。

---

## P2-4（本轮只设计）：WAL 分段

### 现状

单文件 `wal.bin`，`truncate()` 是 `set_len(0)`；每条 `Upsert` 内联完整向量（`Vec<f32>`）与 JSON payload，所以日志体积 ≈ 数据体积 × 写入次数。1M 点、768 维的一次全量导入，仅向量就是 3 GB 日志。

### 分阶段方案

1. **分段**：`wal/` 目录 + `<first_lsn>.bin` 段文件，单段上限 `StorageConfig::wal_segment_bytes`（默认 64 MiB）；`truncate` 改为"删除全部段并新建空段"，是 O(段数) 的 `unlink`，不再是截断一个持续增长的文件。读路径按段名排序串联。
2. **水位截断（依赖第 1 步）**：checkpoint 记录"已落盘到的 lsn"，只删除 `last_lsn < 水位` 的整段，前缀截断从此是删文件而不是重写文件——这正是 P1-3 想进一步缩短"停写窗口"的前提。
3. **向量外置**：`vectors.bin` 的写入改为"先 pwrite 再 fsync，再写 WAL 记录（只含 slot 与长度）"。这会改变耐久性契约（WAL 不再是唯一权威），必须与 2 的水位语义一起设计，否则会出现"WAL 说没写、文件说有"的歧义。

第 1 步独立可用且低风险；第 2、3 步会动"WAL 是唯一权威"这条核心不变式，需要单独的等价性测试（崩溃点注入）。本轮不做。

### 触发条件

- 日志体积/数据体积比超过 10（`wal.appended_bytes()` 与 `memory_estimate()` 的比值，可在 metrics 中加一个 gauge）；
- 或单次 checkpoint 的停写时间仍超过 p99 写延迟的 2 倍。

---

## 实施顺序

| 批次 | 内容 | 理由 |
|---|---|---|
| 1 | P1-4、P2-2 | 纯格式/常量，先落地降低后续 diff 噪音 |
| 2 | P1-6 | 类型替换，编译期能发现全部调用点 |
| 3 | P1-5 | 独立模块，可用单测证明保守性 |
| 4 | P1-3 | 需要与 P1-4 之后的容量语义对齐 |
| 5 | P2-1 | 依赖前四项都稳定后的压实路径 |

## 实施状态（本轮已落地）

| 编号 | 落地点 |
|---|---|
| P1-3 | `store_maintenance.rs`：`checkpoint()` → `checkpoint_inner()` + `flush_data_files()`；`compact_mutex` 覆盖整个 checkpoint；`meta.save` 在阶段 2 现读现写 |
| P1-4 | `meta.rs`：`SEGMENT_TARGET_BYTES` / `SEGMENT_SLOTS_MIN` / `default_segment_slots()`；`store_open.rs` 的 `create*` 改为按维度取默认段大小 |
| P1-5 | `filter_bitmap.rs`：`GROWTH_CHUNK_BITS` 分块增长、`FieldCost` 预算记账、`drop_field()` + `unindexed`、`build_mask` 的 `None` 降级 |
| P1-6 | `storage.rs`：`IdMap`；`store_open.rs` / `store_read.rs` / `store_write.rs` / `store_payload.rs` / `compaction.rs` 调用点 |
| P2-1 | `index/persist.rs`：`PersistedHnsw::remap` / `PersistedIvf::remap`；`hnsw/maintain.rs`：`covers_all()`；`compaction.rs`：`remap_published_index()` + pending 重编号 |
| P2-2 | `directory.rs`：32 B 头部 + `rec_size` 字段 + `DirView::rec_size()`（仅测试用观察口）；`keys.rs` / `compaction.rs` 跟进 |
| P2-3 | 仅设计（见上） |
| P2-4 | 仅设计（见上） |

## 回归测试

新增（全部并入既有测试文件，不新建测试目标）：

| 位置 | 用例 | 断言 |
|---|---|---|
| `storage/tests.rs` | `test_checkpoint_flush_runs_without_the_write_lock` | 阶段 1 回调里 `write_lock_free()` 为真，且 `count` / `scroll` 在回调内可用（持写锁会死锁）；checkpoint 后重开数据完整 |
| `storage/tests.rs` | `test_wide_vectors_get_smaller_segments` | 768 维 → `segment_slots == 1365`、`vectors.bin` 首段 ≤ 8 MiB；8 维仍是 8192 |
| `storage/tests.rs` | `test_id_map_handles_both_id_kinds` | 数值/字符串混合、`"7"` 归一化为 `Num(7)`、覆盖写、删除、`len` |
| `storage/meta.rs` | `segment_slots_scale_with_the_dimension` | 4/128 → 8192，768 → 1365，1536/4096 → 1024（下限） |
| `storage/filter_bitmap.rs` | `test_high_cardinality_field_is_dropped_not_unbounded` | 4 万取值后字段被丢弃、位图内存归零、`build_mask` 返回 `None` |
| `storage/filter_bitmap.rs` | `test_low_cardinality_field_stays_indexed` | 1 万槽两取值 ≤ 4 KiB（分块增长），掩码长度 = capacity 且位正确 |
| `storage/filter_bitmap.rs` | `test_filter_mixing_indexed_and_dropped_fields_falls_back` | 混合条件整体回退，单条件仍可用 |
| `storage/directory.rs` | `rec_size_round_trips_through_the_header` | 步长从文件头读出，blob 读取正确 |
| `storage/directory.rs` | `unusable_record_size_is_rejected` / `replace_from_rejects_a_different_record_size` | 步长 < 8 与异种步长均报 `CorruptData` |
| `tests/hnsw_test.rs` | `compacted_index_survives_restart` | 压实后 `has_index` 为真、`built_at_live_count == points_count`、重开后从 `hnsw.bin` 命中且被删点不返回 |
| `tests/ivf_test.rs` | `compaction_remaps_the_published_index` | 压实后 IVF 仍在发布状态、`built_at_live_count == 38`、重开后命中 |
| `tests/filter_bitmap_test.rs` | `high_cardinality_field_still_filters_correctly` | 高基数字段降级后 `search` / `count_by_filter` 仍精确（400 点 400 个不同值） |

两条旧断言按新契约改写（它们编码的正是本轮要消除的行为）：

- `tests/hnsw_test.rs::deleted_points_are_never_returned`：原来断言"压实后 `!has_index`，等维护线程重建"，改为断言"压实后索引仍在，且被删点不再出现"。
- `tests/ivf_test.rs::compact_invalidates_index_and_rebuild_restores` → 重命名为 `compaction_remaps_the_published_index`，断言同上。

## 验收

- `cargo test --all-targets --features c_api`：**317 个单元测试 + 13 个集成目标（88 条）全部通过**。
- 1M 容量、单字段 4 万取值：位图预算生效，字段降级后常驻归零，查询仍精确。
- checkpoint 阶段 1 不持写锁（用例直接断言），读路径不再被 flush 阻塞。
- 压实一次后 `has_index()` 仍为 `true`，`built_at_live_count` 等于压实后 live 数，重开后 `hnsw.bin` / `index.bin` 直接加载。
- `keys.bin` 的 `rec_size` 被篡改或异种文件替换时报 `CorruptData`，不会出现"按 8 字节步长读 12 字节记录"的错位。
