# P2-3 / P2-4 细化设计与实施（13）

本文件是 [`12-storage-efficiency-and-evolution.md`](12-storage-efficiency-and-evolution.md) 中最后两项的落地设计：**P2-3（`kind` / `named` 的倒排落盘）** 与 **P2-4（WAL 分段）**。12 给的是方向（"换存储模型"、"分三段做"），这里给出可验收的细节：文件布局、状态机、崩溃窗口、资源上界公式、以及本轮实际落地的代码位置。

判断标准沿用 12 的两条：**资源上界可计算**、**改动不改变查询结果**。12 说明 P2-3 的三步"必须同批做"，本轮按这个要求实施了三步（倒排落盘、点向量 mmap、`reverse` 类型收窄），并把 12 明确划在"改变耐久性契约、需崩溃点注入"之外的 P2-4 第 3 步（向量外置）继续留到下一轮。

## 与 12 的差异（先说结论）

| 项 | 12 的原方案 | 本轮细化后的做法 | 原因 |
|---|---|---|---|
| 倒排文件组织 | `postings/<dim_bucket>.bin`，按 term 排序追加，查询二分定位 | 同样是桶文件 + 按 term 排序追加，但**定位用常驻字典而不是文件内二分**：`term >> 10` 选桶，桶内字典 `term -> [chain]` | 文件内二分要维护每文件的稀疏索引；字典每 term 一条 12 B 记录，打开时一次读完即可，查询变成两次哈希 |
| 快照瘦身 | `slot -> id` + 向量指针，向量进 mmap | 一致，落地为 `points.bin`（追加式 blob）+ 快照只存 `BlobRef{off,len}`；payload 随后也外置到 `payloads.bin`（见 [14](14-payload-externalization-and-wal-vectors.md)） | 本轮先解决“向量反序列化 + 倒排重建”；变长 JSON 用同一套追加式 blob 格式外置，不需要 dense 的 slot 寻址 |
| 倒排键类型 | `PointId` / `u32` | 一致：`reverse` 改成 `IdMap`（`HashMap<u64,u32>` + `HashMap<String,u32>`），term 本来就是 `u32` | — |
| WAL 段名 | `<first_lsn>.bin` | 一致，`{:020}` 补零，惰性创建（第一次真正要写时才建文件） | 避免"空段"这种既占 inode 又要在截断时特判的状态 |
| WAL 截断 | 只删除 `last_lsn < 水位` 的整段 | 一致，并把"尾段是否被覆盖"的判定做实（见下） | 尾段没有后继段名可比，需要单独记 `active_last_lsn` |
| P2-4 第 3 步 | 向量外置，改变"WAL 是唯一权威" | **不做**，仅保留设计 | 需要崩溃点注入的等价性测试，与 12 的"本轮不做"一致 |

---

# P2-4：WAL 分段 + 水位截断

## 现状与量化

单文件 `wal.bin`，记录是 `[u32 len][u32 crc32][postcard(WalTxn)]`，`truncate()` 是 `set_len(0)`。

- **体积**：一条 `Upsert` 内联完整向量与 JSON payload，日志体积 ≈ 数据体积 × 写入次数。1 M 点 × 768 维全量导入 ≈ 3 GB 单文件；`Compact` 之前这段时间里，**任何一个** `fsync` 都要落在这一个文件上。
- **截断语义**：`set_len(0)` 本身是 O(1)，但"截断后重新长大"意味着文件的块分配、目录项、以及崩溃后 replay 的起点全在一个对象上；`P1-3` 把 flush 移出写锁之后，**停写窗口**剩下的部分正是"marker append + truncate"这两次对同一文件的元数据操作。
- **可观测性**：无法回答"日志里最老的事务是哪一条"，因为没有段的边界。

## 方案

```text
wal/
  00000000000000000001.bin   ← 该段第一条事务的 lsn 命名，{:020} 补零
  00000000000000001234.bin
  ...
```

- **惰性创建**：`append_batch` 发现当前段长度已达 `StorageConfig::wal_segment_bytes`（默认 64 MiB）或尚无段时，用**本批次第一条事务的 lsn** 建新段。批不跨段：一条记录不能被切开，所以单批可能略微超出上限。
- **序号单调**：新段名取 `max(批首 lsn, 尾段名 + 1)`。事务号在同一集合内单调（dense 走提交队列 lsn，`kind`/`named` 走 `next_lsn`），正常情况下就是批首 lsn；取 max 是对"两个句柄打开同一目录"这类测试用法的兜底。
- **目录项先落盘**：建段后立刻 `fsync` 目录，再写记录 —— 记录在目录项持久之前不能算已提交。
- **失败回滚**：`write_all`/`sync_all` 失败时 `set_len(批起点)`，与单文件时代一致；否则半条记录会把后续 replay 全部截断。
- **replay 串段**：按段名升序逐段读，**只有最后一段**的"半条记录 / CRC 不符"被当作崩溃尾巴静默结束；**非最后一段**出现同样的损坏说明日志在后继段还有内容，必须报 `CorruptData`。这比单文件时代的"一律静默停止"更严格，也是分段带来的必然结果（单文件里"CRC 之后还有数据"是可疑，多段里"后面还有整段数据"是确凿）。
- **两种截断**：
  - `truncate()`：删掉所有段（压实用，此时全部派生文件已重写）；
  - `truncate_through(watermark)`：只删**整段覆盖**的段（checkpoint 用）。

## 水位截断的判定（本项唯一需要推敲的地方）

段名是该段第一条事务的 lsn，段内事务号单调，所以：

> 段 `i` 内的事务都落在 `[name(i), name(i+1))`，**全部 ≤ W** 等价于 `name(i+1) ≤ W + 1`。

尾段没有后继段名可比，因此每个句柄额外维护 `active_last_lsn`（本句柄追加/读到的最大 lsn）：

> 尾段被覆盖 ⟺ `active_last_lsn ≤ W`。

`active_last_lsn` 的三种来源，决定了这条判定的保守性：

| 时机 | 取值 | 效果 |
|---|---|---|
| 打开时无段 | `0` | 空日志，无可删 |
| 打开时有段 | `u64::MAX`（未知） | 尾段**不会**被误删；`replay`/`read_all` 走完一遍后回填真实最大值 |
| 每次 `append_batch` | `max(prev, 批最大 lsn)` | 写路径无需读文件即可精确判定 |

不变量：**只要 `truncate_through` 删除了某段，该段的全部事务都 ≤ 水位**，而水位来自"这些事务已经落到数据文件/快照里"的承诺（dense 是 `meta.last_applied_txn`，`kind`/`named` 是快照 lsn）。因此删除永远不丢需要一个事务。留下的"跨界段"保持原样 —— 前缀截断正是这套方案要避免的重写。

## 崩溃窗口

| 崩溃点 | 磁盘状态 | 恢复 |
|---|---|---|
| 建段后、写记录前 | 空段（0 字节） | replay 把它当最后一段读，读到 EOF 即结束；下次 append 直接续写 |
| 写记录中 | 尾段半条记录 | replay 在尾段静默停在半条记录之前 |
| 写记录中（非尾段） | 不可能：只有当前段在写 | — |
| 删除段中 | 部分段已删 | 剩下的段覆盖的是已落盘区间，replay 幂等跳过；重复字节无副作用 |
| 删除尾段后崩溃 | 段目录为空，`active=None` | 下批 append 新建段，段名取批首 lsn |

## 资源上界

- 段文件数 `N ≤ ⌈wal_bytes / wal_segment_bytes⌉ + 1`；
- 单文件字节 ≤ `wal_segment_bytes + 单批字节`；
- 一次 checkpoint 的删除成本 = `O(被删段数)` 次 `unlink`，最坏 `O(⌈W / wal_segment_bytes⌉)`；
- 一次 replay 的读放大 = 日志总字节（与单文件时代相同，但由于每段有上界，"重放多少"第一次可以被配置项约束）。

> 12 提到的"日志/数据体积比"指标不需要新 gauge：`wal().appended_bytes()` 与 `memory_estimate()`（或 `kind`/`named` 的 `storage_bytes()`）都在公开 API 上，比值可现算。

## 落地点

| 内容 | 位置 |
|---|---|
| 段目录、惰性建段、回滚、串段 replay、两种截断 | `src/storage/wal.rs` |
| `wal_segment_bytes`（默认 64 MiB）+ 校验 | `src/types/config/storage.rs` |
| 集合目录 `wal.bin` → `wal/`，三处 `create`/`open` | `src/storage/store_open.rs`、`src/kind_store/lifecycle.rs`、`src/named_store/lifecycle.rs` |
| dense checkpoint 走 `truncate_through(last_applied_txn)` | `src/storage/store_maintenance.rs` |
| `kind`/`named` checkpoint 走 `truncate_through(快照 lsn)` | 两侧 `lifecycle.rs::checkpoint_from_locked` |
| 压实仍用全量 `truncate()` | `src/storage/compaction.rs` |

## 验收（已并入既有测试目标）

| 位置 | 用例 | 断言 |
|---|---|---|
| `storage::wal::tests` | `test_segments_roll_over_and_replay_in_order` | 小段阈值下产生多段、段名升序、replay 顺序 = lsn 顺序、重开段名不变 |
| 同上 | `test_truncate_through_removes_only_covered_segments` | 水位落在第二段内部时只删首段，跨界段保留，剩余记录仍升序 |
| 同上 | `test_truncate_through_drops_the_tail_when_fully_covered` | 水位覆盖全日志时清空；下一次 append 以新 lsn 建段 |
| 同上 | `test_corrupt_middle_segment_is_reported` | 非尾段损坏报 `CorruptData`（尾段仍然静默） |
| `tests/storage_test.rs` | `wal_segments_roll_over_and_a_checkpoint_unlinks_them` | checkpoint 后 `wal/` 目录为空、`appended_bytes()==0`、再写入用新段并可重开 |

---

# P2-3：`kind` / `named` 的倒排与向量落盘

## 现状与量化

两种非 dense 集合的持久化是"全量快照 + WAL 回放"，内存里是 `points: Vec<Option<Point>>`（`VectorData` 内联）+ `inverted: HashMap<u32, Vec<(u32, f32)>>` + `norms` + `reverse: HashMap<String, u32>`。

以 1 M 点、每点 64 个稀疏项为例：

| 项 | 12 的估算 | 现在的字节 |
|---|---|---|
| 常驻倒排 | 64 M × 8 B = 512 MB + 每 term 的 `Vec` 头（24 B） | 同左，且**每次 open 重建**（64 M 次哈希插入） |
| 快照 | 全量 postcard：向量数据 512 MB + id + payload | 同左，且**每次 checkpoint 重写** |
| 点目录 | `KindPoint` = `PointId`(24) + `VectorData`(48) + payload(48) = 120 B/槽 | 120 B/槽 + 每条向量的两个堆块（`indices` 4n + `values` 4n） |

## 方案：三个文件，各管一件事

```text
points.bin                 追加式向量 blob（mmap，读零拷贝）
points.compact             压实时的暂存文件（rename 发布）
postings/<bucket>.bin      term >> 10 选桶，记录 [term u32][count u32][(slot u32, weight f32)*count]
postings/manifest.bin      字典：{lsn, 每桶长度, term -> [chain]}
kind_snapshot.bin          目录：{lsn, blobs_len, [slot, id, BlobRef, payload, norm]}
```

### 1) `points.bin`：向量搬出快照

- 编码（每个 blob 起始 4 字节对齐，因此 `&[f32]`/`&[u32]` 可直接从映射借用）：
  ```text
  [kind u8][pad 3][a u32][b u32] + 数组
  dense : a = 长度                 数组 = f32[a]
  sparse: a = 项数                 数组 = u32[a] 后接 f32[a]
  multi : a = 子向量维度, b = 个数  数组 = f32[a*b]
  ```
- `named` 的点把多个空间放进一个 blob：`[u32 n]` + `n × ([u16 空间号][pad 2][u32 长度][向量编码])`。空间号是集合声明顺序（`BTreeMap` 顺序），因此不需要在点上存空间名。
- 追加按 4 MiB 分块预分配 + 重映射：一次 `mmap` 换 4 MiB 写入，而不是每次 append 一次。
- **打开时的长度协商**：快照记录 `blobs_len`；实际文件更长 = 崩溃尾巴（追加了但没进快照），`set_len` 剪掉；更短 = 快照指向了不存在的字节，报 `CorruptData`。
- 覆盖写/删除**不回收**旧 blob，压实（`points.compact` → rename）时才重写。这是"追加式"的代价，与倒排的取舍一致。

### 2) `postings/`：倒排落盘

- 桶 = `term >> 10`（每桶最多 1024 个 term）。记录按 term 升序追加，所以一个桶文件内"新追加的记录段"是有序的；跨多次 flush 的同一 term 形成**链**：`term -> [(桶, 偏移, 条数)]`。
- 常驻的只有**字典**：每 term 一条 `Vec<Chain>`（1 条链 12 B），加上尚未 flush 的 **pending 窗口**（上限 8 MiB，超出即提前 flush）。
- 链太长（> 8 条）时，checkpoint 里把该桶整体重写为"每 term 一条记录"。因为桶内 term 数 ≤ 1024，重写成本被桶的字节量摊薄；跨桶重写不会级联（term 的桶由 `term >> 10` 唯一确定）。
- **删除不删倒排**：一条记录的 `count` 是它的边界，删掉其中一个 posting 就要重写整条记录。死槽的 posting 由读路径丢弃（扫描本来就要查 `points[slot].is_some()`），压实是唯一的回收时机。代价是 `posting_hits` 会把死 posting 算进去 —— 只会让"倒排累加 vs 全扫"的选择更早偏向全扫，不影响正确性。
- **patch 必须换槽**：`named` 的空间级更新会重写该点的 blob；如果沿用旧槽，旧的 `(slot, term)` posting 与新的混在一起，而"活跃槽上的陈旧 posting"在查询期无法与真 posting 区分（同槽同 term）。所以 patch 走"墓碑旧槽 + 分配新槽"，并在新槽上重建全部稀疏空间的 posting。代价与一次 upsert 相同。

### 3) `reverse` 收窄

`HashMap<String, u32>` → `IdMap { numeric: HashMap<u64,u32>, uuid: HashMap<String,u32> }`，与 P1-6 的 dense 反查表同一实现（提到 `pub(crate)`）：

- 查询路径 `inner.reverse.get(&point_id)` 不再 `to_string()`；
- WAL/快照带来的字符串 id 走 `get_str/insert_str`，数值 id 解析成 `u64`，不分配；
- 提交校验不再克隆全部 key（`validate_records` 从"复制整张表"改成"查表 + 批内增量"）。

## 一致性契约（本项的核心）

三份状态必须描述**同一个 lsn 的同一批点**：`points.bin` 的长度、`postings/manifest.bin` 的 `lsn`、`kind_snapshot.bin` 的 `lsn`。

```text
checkpoint:
  1. blobs.sync()                    // 向量字节 fsync
  2. postings.flush(Some(lsn))       // 追加桶 + fsync + 写 manifest（lsn）
  3. snapshot::write(...)            // 目录（含 blobs_len）
  4. wal.truncate_through(lsn)       // 只删被覆盖的整段
```

读取端的判定只有一条：

> `manifest.lsn == snapshot.lsn` 才使用磁盘上的倒排；否则**整份倒排丢弃**，从快照的点集重新反演。

理由是可重复性：replay 从快照 lsn 之后重放，会**重建出同样的槽号**；如果倒排里已经含有那些 `(slot, term)`，重放会再插一次，分数就翻倍。丢弃而不是"部分对齐"，因为没有办法知道哪部分对齐。

软 flush（pending 超 8 MiB，写路径上触发）**不写 manifest**：它只是把内存搬到文件，让内存有界；这些字节在下次 open 时被"裁剪回 manifest 记录的长度"丢弃，靠 replay 重建。这样"内存有界"和"磁盘状态只由 checkpoint 提交"两件事不冲突。

| 崩溃点 | 磁盘状态 | 恢复 |
|---|---|---|
| 追加了 blob、未 checkpoint | `points.bin` 比 `blobs_len` 长 | 打开时 `set_len(blobs_len)`，尾巴丢弃（数据在 WAL 里） |
| 追加了桶、未写 manifest | 桶比 manifest 记录的长度长 | 打开时裁剪回记录长度 |
| 写了 manifest、未写快照 | `manifest.lsn > snapshot.lsn` | 倒排整份丢弃，从快照 + replay 重建 |
| 写了快照、未截断 WAL | 快照 lsn ≥ 日志里的记录 | replay 跳过 `≤ lsn` 的事务（既有语义） |
| 压实中途 | 暂存文件 ≠ 正式文件 | 暂存文件是新建的 `points.compact`，正式文件只在 rename 那一刻换掉 |

## 资源上界

| 结构 | 公式 | 1 M 点 / 64 term 的取值 |
|---|---|---|
| 点目录 | `slot_capacity × size_of::<Option<KindPoint>>()` = 24(`PointId`) + 16(`BlobRef`) + 48(`payload` 句柄) ≈ 88 B | ≈ 88 MB |
| 范数缓存 | `8 B × 稀疏槽数`（`(u32, f32)` + 哈希开销） | ≈ 12 MB |
| 倒排字典 | `#term × (12 B 链 + 哈希开销) + Σ 额外链 × 12 B` | `#term` 级，与 posting 总量无关 |
| pending 窗口 | `≤ SOFT_FLUSH_BYTES = 8 MiB` | ≈ 8 MB |
| 磁盘倒排 | `#posting × 8 B + #记录 × 8 B` | ≈ 512 MB（在 page cache / mmap，不占堆） |
| 快照 | `#point × (id + 16 + payload)` | ≈ 30–40 MB（原来 ≥ 512 MB） |

对照原来：常驻 512 MB 倒排 + 每点两个向量堆块，变成"字典 + 8 MB 窗口 + mmap 页面"。打开成本从"反序列化全部点 + 全量哈希插入"变成"读一份字典 + 映射一个文件"。

## 落地点

| 内容 | 位置 |
|---|---|
| 追加式 mmap blob + 向量/多空间编码 + 长度协商 | `src/storage/point_blobs.rs` |
| 桶文件 + 字典 + pending + manifest + 桶重写 | `src/storage/postings.rs` |
| `VectorView`（借用视图）与 `to_data()` | `src/types/point.rs` |
| `IdMap` 提升到 `pub(crate)`，加 `get_str/insert_str/contains_str` | `src/storage.rs` |
| `kind`：`KindPoint{blob}`、快照瘦身、checkpoint、压实、patch | `src/kind_store.rs`、`src/kind_store/{lifecycle,read,write,state}.rs` |
| `named`：每空间 `Postings`、空间号、patch 换槽 | `src/named_store.rs`、`src/named_store/{lifecycle,read,write,state}.rs` |
| 观测口 `postings_restored()` / `storage_bytes()` | 两侧 `read.rs` |

## 验收（已并入既有测试目标）

| 位置 | 用例 | 断言 |
|---|---|---|
| `storage::point_blobs::tests` | `vectors_round_trip_through_the_blob_encoding` | dense/sparse/multi 编解码往返一致，`encoded_len` 与实际字节相符 |
| 同上 | `appends_are_addressable_and_aligned` | 奇数长度 blob 之后仍 4 字节对齐，视图可零拷贝解码 |
| 同上 | `a_long_file_is_truncated_to_the_snapshot_length` | 未提交尾巴不可达 |
| 同上 | `a_short_file_is_reported_as_corruption` / `growth_is_transparent_across_chunks` | 短文件报 `CorruptData`；跨 4 MiB 分块重映射后旧引用仍有效 |
| 同上 | `named_entries_round_trip_and_patch` | 多空间 blob 的逐空间视图与"只替换一个空间"的重写 |
| `storage::postings::tests` | `pending_postings_are_visible_before_a_flush` | 未 flush 的倒数可见（查询不依赖 flush） |
| 同上 | `a_checkpointed_index_reopens_without_rebuilding` | 同 lsn 打开直接命中字典，不再反演 |
| 同上 | `a_stale_manifest_discards_the_postings` | lsn 不匹配即丢弃，且只留 manifest |
| 同上 | `a_soft_flush_keeps_the_last_checkpoint_as_the_durable_point` | 软 flush 的字节不算提交 |
| 同上 | `terms_are_spread_over_buckets` / `repeated_flushes_collapse_into_one_record_per_term` | 桶划分正确；链数被重写约束在 `MAX_CHAINS` 内 |
| 同上 | `a_corrupt_manifest_makes_open_start_empty` / `reset_removes_every_bucket` | 损坏桶 → 空字典重建；压实重置清空目录 |
| `kind_store::tests` | `kind_restart_loads_the_posting_dictionary` | 重开后 `postings_restored()`、`storage_bytes()` 与写前一致；快照 < 向量字节的 1/4 |
| 同上 | `kind_stale_posting_manifest_is_rebuilt` | 删掉 manifest 后仍能查到全部点（重建路径） |
| 同上 | `kind_deleted_points_leave_no_phantom_hits` | 死槽的陈旧 posting 不产生命中（重启后同样） |
| 同上 | `kind_compaction_rewrites_blobs_and_postings` | 压实后 blob 字节减半、倒排仍可用、重开一致 |
| 同上 | `kind_payload_edits_survive_a_restart_without_touching_vectors` | payload 改动不影响向量、可跨重启 |
| `named_store::tests` | `named_restart_loads_the_posting_dictionaries` | 每空间字典恢复；dense 空间不受影响 |
| 同上 | `named_patch_reallocates_the_point_and_its_postings` | patch 后旧 term 不再命中（重启后同样），未改空间保留 |
| 同上 | `named_compaction_reclaims_blob_bytes` | 压实回收 blob 字节、倒排重建 |

## 未做与理由

1. **payload 外置已由 [14](14-payload-externalization-and-wal-vectors.md) 落地**：`kind_snapshot.bin` / `named_snapshot.bin` 只存 `payloads.bin` 里的 `BlobRef` 与本文件长度；14 未采用 dense 的 slot 寻址 `payloads_store`，而是复用追加式 blob 文件，避免把页几何移植到变长集合。
2. **P2-4 第 3 步（向量外置到 WAL 之外）**：会改变"WAL 是唯一权威"这条核心不变式，必须与"水位截断"的语义一起做，并配崩溃点注入的等价性测试。本轮不做。
3. **稠密空间的倒排**：`named` 的 dense 空间不建倒排（本来就不需要），因此 `postings/<idx>` 目录只在稀疏空间出现。
4. **旧的 `wal.bin` / 旧 `kind_snapshot.bin` 不做迁移**：与仓库既有策略一致（无向后兼容）。旧集合会在版本/结构校验处失败，由引擎隔离为 `.suspect-<ts>`；`keys.bin` v2 的同类处理见 12 的 P2-2。
