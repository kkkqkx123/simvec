# 载荷外置与 WAL 向量外置（14）

本文件承接 [`13-p2-3-p2-4-implementation.md`](13-p2-3-p2-4-implementation.md) 的“未做与理由”。13 把 `kind` / `named` 的向量搬进了 `points.bin`，但快照仍内联 JSON 载荷，WAL 也仍内联向量。本轮落地前者的外置，并给后者一份可直接执行的细化方案；后者改变“WAL 是唯一权威”的耐久性契约，需要独立的崩溃注入验证，不在本轮代码里一并改动。

| 项 | 13 的状态 | 本轮 |
|---|---|---|
| `kind` / `named` 快照的 JSON 载荷 | 内联在 `*_snapshot.bin` | **已外置**到 `payloads.bin`，快照只存 `BlobRef` |
| WAL 里的向量（dense / kind / named） | 内联在 `Upsert` / `UpsertVectors` | 未改：给出协议、崩溃窗口与验收设计 |
| dense 倒排、旧格式迁移 | 明确不做 | 仍不做（前者无检索收益，后者违反仓库策略） |

## 已实施：JSON 载荷外置

### 布局与不变量

每个 `kind` / `named` 集合目录新增一个 `payloads.bin`，复用 `PointBlobs` 的格式（4 字节对齐、12 字节头、4 MiB 分块增长）。快照字段从 `payload: Option<String>` 变成 `payload: Option<BlobRef>`，并在快照顶层记录 `payloads_len`。

点目录里的 `payload_blob: Option<BlobRef>` 只有两种含义：

- `Some(ref)`：当前内存载荷的 JSON 字节已经落在这个 ref 上，下次 checkpoint 直接复用；
- `None`：载荷自上次 checkpoint 后发生过 upsert、整点替换、删键或加字段，checkpoint 必须追加新 blob。

把“是否脏”编码进 ref 本身，而不是另设布尔位，是为了让 replay / restore / checkpoint 三条路径共用同一条判定：不需要跨路径同步第二个标志。

`payload = None` 的点不占 blob。压实只重编号向量槽，载荷 ref 保持原样——载荷文件是追加式的，旧字节只是不可达，并不因槽位变了而失效。

### checkpoint 顺序

在 13 的四步之间插入载荷提交：

```text
points.bin sync → postings.flush(lsn) → 追加脏载荷 + payloads.bin sync
    → snapshot::write({payloads_len, payload_blob, ...}) → wal.truncate_through(lsn)
```

载荷字节必须在快照之前 fsync，否则快照会指向尚未落盘的 JSON。反过来，先写字节后写快照的窗口是安全的：多出来的字节超过 `payloads_len`，打开时被裁掉。

`checkpoint_from_locked` 因此接收 `&mut Inner`：它要把新 blob ref 写回点目录。调用方本来就持有写锁（`checkpoint`、`compact`、写满阈值后的自动 checkpoint），没有新增锁范围。

### 打开时的协商

- 有快照：`payloads.bin` 被裁到 `payloads_len`（更长 = 未提交尾巴），更短 = `CorruptData`；快照里的每个 ref 都要能在映射中取到字节并解析成 `Payload`。
- 无快照（首次打开或快照损坏/丢失）：裁到 0，WAL 回放从事务里的 JSON 重建载荷，下次 checkpoint 再写 blob。

向量文件仍按 `blobs_len` 处理，语义不变。`storage::snapshot` 信封没有升版：旧快照的 postcard 字段与新结构不兼容，会在反序列化处失败并按既有策略隔离，不写迁移。

### 崩溃窗口

| 崩溃点 | 磁盘状态 | 恢复 |
|---|---|---|
| 追加载荷后、快照前 | 文件比 `payloads_len` 长 | 打开时裁掉尾巴；WAL 仍有 JSON |
| 快照后、WAL 截断前 | 快照 ref 指向已 fsync 的 JSON | 直接复用；WAL 记录被幂等跳过 |
| WAL 截断中途 | 部分段已删 | 快照已覆盖这些事务，剩余记录重放无害 |
| 载荷 ref 指向不存在字节 | 快照通过 CRC 但结构坏 | `CorruptData`，由引擎隔离，不用旧 JSON 兜底 |

资源上界：载荷文件字节 = `Σ JSON(payload)` + 4 KiB/点对齐 + 未被压实回收的旧版本；快照字节与载荷大小无关，只剩 `slot + id + 两个 BlobRef + norm`。观测口 `KindStore::payload_bytes()` / `NamedStore::payload_bytes()` 暴露已提交字节。

### 验收

`kind_store_tests.rs` 与 `named_store_tests.rs` 各加一条：

- 带大 JSON 载荷写入并 checkpoint 后，`payload_bytes()` 与载荷量相当，快照字节里找不到载荷标记；
- 连续两次无改动 checkpoint 不增大载荷文件；
- 改字段后再 checkpoint 会追加新 blob；重开后读到合并后的完整载荷与全部向量空间。

## 待实施：WAL 向量外置

这一步会让 `vectors.bin` / `points.bin` 从派生数据升级为 WAL 引用目标，必须先解决“字节先落盘、记录后落盘”的顺序，再谈收益。下面给出 dense 与非 dense 两条协议、崩溃表与测试方案；本轮不改代码。

### dense：`Upsert` 只记槽位，向量写进 `vectors.bin`

现状是 WAL 先 fsync、apply 时写 `vectors.bin`。外置后顺序反转：

1. `apply_txn` 的 reserve 段给新 id 定槽；把 `(slot, Vec<f32>)` 暂存进 `Ticket`。`WalPoint` 形状收窄为 `{id, slot, payload}`，不再有 `vector` 字段。
2. worker 取一个 batch：先把所有暂存向量 `pwrite` 到 `vectors.bin` 的目标槽（必要时先 `grow_to`，不持写锁），一次 `Vectors::sync()` fsync；失败则 settle 全部 ticket、不写 WAL。
3. 再照旧一次 `append_batch` + fsync；成功后 apply 只补 `keys.bin`、reverse、payload、位图、索引。`apply_upsert_locked` 与 replay 从该槽读回向量注册到索引，不再从记录里取。
4. `WalRecord::Upsert` 携带 `slot`，replay 直接用它定位，避免“reserve 数出来的号”和“文件里写过的号”靠隐式顺序对齐。

要点：

- 槽位在 apply 之前不可见：读路径靠 `keys.bin` / reverse 发现槽，向量提前落在 mmap 上不会被搜到。
- `Vectors::open` 已经容忍文件比 `meta.slot_capacity` 长或短，因此“文件先长、meta 后写”不需要额外文件；崩溃后 replay 会用同一个槽再写一遍。
- `count_ops`、`validate_point`、`validate_admission_locked` 的入口要改为从 staged 向量校验；`WalPoint::from_point` 不再能从点独自构造 WAL 记录，测试helper 改为显式传槽位。
- 量化镜像、filter bitmap 与 payload index 仍在 apply 段更新，不受外置影响。

### kind / named：`UpsertVectors` / `PatchNamedVectors` 携带 blob ref

写路径在 gate 内先把编码后的向量（named 的 patch 是合并后的 blob）append 到 `points.bin` 并 `sync()`，再把 `BlobRef` 放进 WAL 记录。载荷 ref 复用本轮的 `payloads.bin` 协议。回放只读 ref，不再 `append` 新 blob，也不再重新分配未知槽号——WAL 记录显式携带 `slot` 与 overwrite 的目标槽。

打开时的长度协商必须改：现在 `PointBlobs::open(..., snapshot.blobs_len)` 会把“未进快照的尾巴”直接裁掉；外置后这些尾巴正是未回放 WAL 引用 targets。新的高水位是：

```text
blobs_len = max(snapshot.blobs_len,
                max(ref.end for ref in 未截断 WAL 中 lsn > snapshot.lsn 的记录))
```

计算发生在构造 store 之前（一次只读 WAL 扫描），之后 replay 覆盖这些事务，下一次 checkpoint 把高水位写回快照，才能裁掉前缀。载荷文件同理，用 `payloads_len` 做同样的协商。若某个引用超出高水位，说明日志与 blob 文件不一致，报 `CorruptData`，不能“重建一个向量”兜底。

### 崩溃表（两条协议共用）

| 崩溃点 | 磁盘状态 | 恢复 |
|---|---|---|
| 向量字节 fsync 后、WAL append 前 | 孤立字节，无记录引用 | replay 看不到该事务；下次同槽写入覆盖 |
| WAL append 成功、apply 前 | 记录与字节都在 | replay 按槽/按 ref 重建，结果与崩溃前一致 |
| apply 后、快照前 | 内存已变、WAL 未截 | replay 幂等覆盖同一槽/ref |
| 快照后、水位截断前 | 快照 lsn ≥ 日志 lsn | replay 跳过，旧段留着无害 |
| blob sync 失败 | 没有 WAL 记录 | 全体 ticket 失败，集合仍可写 |

### 验收前置（必须先有这些注入点）

- `Vectors::fail_next_sync_for_test()` 与 `PointBlobs::fail_next_sync_for_test()`，语义与现有 `Wal::fail_next_append_for_test()` 一致：在 fsync 之前失败，不写任何字节。
- 每个注入点一条用例：失败后 `get` / `search` 看不到该点，紧接着的正常写入与重启后 `get` / `search` / `count` 与未注入时逐项一致。
- 在“WAL 已落、apply 未落”和“快照已落、WAL 未截”两个点各做一次强制重启比对；断言 `Upsert` 的槽号与向量字节和崩溃前完全相同。
- 回归断言 WAL 字节数不再随向量维度线性增长：同样 1000 次覆盖写，768 维集合的 `wal.appended_bytes()` 应接近 4 维集合。

### 为什么本轮不做

收益（日志体积）来自改变提交协议，而风险正好在提交协议的顺序与崩溃窗口。上表里的“孤立字节”和“高水位扫描”必须靠注入点正面验证，不能靠正常路径测试推断。把协议改完却缺少这些测试，会把当前“WAL 是唯一权威、apply 前不可能有磁盘数据”的不变式换成未经验证的双文件不变式，与“正确性优先”冲突。因此本轮只落载荷外置（不改变 WAL 权威），并把上面的实施顺序固定下来。
