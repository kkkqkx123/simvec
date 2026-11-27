# 与 ref/qdrant 的存储差距问题清单

本文对比当前 simvec 原生存储与 `ref/qdrant` 的实现，记录已确认的问题、能力差距和少量仍然合理的范围差异。问题清单是执行输入，分阶段修改方案见 `docs/plan/storage-gap-remediation.md`；格式版本与迁移的长期设计见 `docs/plan/long-term/versioning-and-migration.md`。

参照版本：simvec `0ce056a`，qdrant `7c59b2d`。范围只包括本地引擎的原生存储；wasm 内存引擎不在讨论范围内。

## 结论

simvec 当前采用“单集合单发布代 + 集合级 WAL/LSN”的模型，换取了单一提交点和较短的实现路径；qdrant 采用“shard 内多 segment + 每 segment 独立版本、id tracker、向量存储、payload 存储和索引”的模型，换取增量压实、并发 flush、按 segment 回收 WAL 和更强的扩展能力。

两者差异中，集群能力、单写者边界和开发阶段的格式策略可以保留；但 WAL mmap 化与预分配、payload 压缩、named/sparse 的一等公民存储与索引都不是永久性的范围差异，已列入改进项。

## 目录结构对照

| 项目 | simvec | qdrant |
| --- | --- | --- |
| 顶层 | 集合根保存 `generation.bin`、`.lock`、`wal/`、类型标记 | collection 保存 collection 配置、WAL 和多个 shard |
| 服务数据 | 全部位于 `generations/g<20位id>/` | shard 的 `segments/<uuid>/`，每个 segment 独立自治 |
| 元数据 | `meta.bin` + `segments.json` + `dense_snapshot.bin` | `segment.json`、`storage_version`、segment manifest |
| 向量 | 物理单文件 `vectors.bin`，内部按段 mmap；kind/named 用 `points.bin` blob | appendable chunked mmap、immutable mmap、多数据类型和多向量空间 |
| 点目录 | 全局 `IdMap`、`keys.bin`、逐点版本保存在 dense snapshot | 每 segment `id_tracker`，保存 external→internal、版本和删除版本 |
| Payload | `payloads_store/`：`tracker.bin`、`tracker.journal.bin`、`bitmask.bin`、`page_*.bin`，值带 CRC32 | blobstore gridstore/logstore，默认 LZ4，tracker journal 和打开时一致性校验 |
| WAL | `wal/<first-lsn>.bin`，postcard + CRC32，`acked-lsn` 绑定 generation | mmap 预分配段、magic/version/随机 CRC seed、CRC32C、8 字节对齐、`first-index` |
| 索引 | 按 immutable 逻辑段发布 HNSW/IVF/量化文件；字段索引内容打开时重建 | 每 immutable segment 建图，字段索引有 mutable/immutable/on-disk 变体 |

## 数据正确性和恢复风险

### checkpoint 标记复用 LSN 会破坏 WAL 段连续性

`src/storage/store_maintenance.rs` 在 generation lsn 推进后追加 `WalRecord::Compact`，其 `txn_id` 等于刚持久化的 watermark。`src/storage/wal.rs::replay_after` 要求一个已关闭段的最后 LSN 严格比下一段名小一。

如果旧 active 段已经包含该 watermark，而这条 checkpoint 标记又触发滚动，`next_segment_name` 会生成与 watermark 相同的段名；若在 `truncate_through` 之前崩溃，重开时连续性校验会把该集合判为 `CorruptData`。影响范围是所有在满段附近执行 checkpoint 的集合，且崩溃窗口正是维护路径的中段。

### WAL append 失败后的 reservation 残留可能造成槽位漂移

`src/storage/queue.rs::run_batch` 在 WAL append 失败时只把错误返回给等待者，没有撤销 `src/storage/store_write.rs::reserve_slots_locked` 已写入 `inner.reserved` 的槽位，也没有标记这些 LSN 为失败。`inner.next_lsn` 已前进，但失败事务不在日志中。

重放路径 `src/storage/store_open.rs::replay_wal` 按最终写入 WAL 的事务顺序重建 id→slot；而 live 状态仍保留失败事务的 reservation。若之后重试同一个 id 并发生 checkpoint，快照不包含失败窗口，重放可能为后续成功事务分配不同的槽位。

风险会被 WAL 中的 `SetPayload`、`DeletePayloadKeys`、`SetPayloadField` 放大：这些记录携带内部 slot 而不是外部 point id。一旦 live 与 replay 的槽位编号不一致，重放会把 payload 操作应用到错误的槽位。现有失败提交测试只检查点可见性，没有检查槽位和 payload 的重放一致性。

### 发布后失败依赖 poison 状态

generation manifest 一旦发布，任何在内存句柄切换、WAL ACK 或截断之前的失败都会让集合进入 `generation_poisoned`，必须重开。这是保守的正确性选择，但与 qdrant 的 proxy segment 吸收写入相比，运行期间故障会直接拒绝写入、checkpoint 和压实。该差异可以保留，但需要在运维层面暴露状态并支持自动重开或重建句柄。

## 并发与容量问题

### checkpoint 和压实期间全局停写

simvec 通过 `CommitQueue::quiesce()` 关闭写入 gate，等待队列清空后 fsync、推进水位并截断 WAL。实现简单且安全，但所有写入都要等待整个维护窗口；集合越大，停顿越明显。

qdrant 的 flush worker 可以与 update worker 并行：按 `applied_up_to` 上限刷新各 segment，取最小确认版本 ACK WAL，并用 `WalAckPins` 保护快照等仍需旧日志的消费者。simvec 缺少按 segment 的 durable version 和 ACK pin，因此无法做到同等粒度的并发。

### dense 数据物理上仍是整代单文件

`segments.json` 和 `hnsw_seg_*` / `ivf_seg_*` / `quant_seg_*` 只定义了逻辑段边界。`vectors.bin`、`keys.bin`、`tombstones.bin` 和 payload 文件仍属于整个 generation。压实会重写全集合而不是受影响段。

后果包括临时磁盘空间和 IO 与全集合大小成正比、单段高墓碑率触发全集合重写、快照无法增量复制，以及打开时一次性映射整代数据。qdrant 的 immutable segment 天然把压实、索引和快照限制在单个段内。

### 打开时重建成本高

字段索引只持久化 schema（`payload_indexes.json`），位图和 posting 内容在打开时按全部 live 槽位重建。若 `dense_snapshot.bin` 与 generation、几何或墓碑计数不匹配，还会扫描全部 slot、读 keys、重建过滤位图。qdrant 持久化 id tracker 和多种字段索引文件，并有 immutable/on-disk 变体。

### 快照是全量复制

`CollectionStore::snapshot_to` 在 maintenance lock、queue gate 和 store 读锁下复制整个 generation 和整个 WAL。恢复包自包含、易校验，但复制窗口长且会阻塞写入。qdrant 用 segment manifest 中的文件版本做增量快照，并用 WAL ACK pin 保护复制窗口。

## 明确列入改进项的能力差距

以下三项不属于合理的范围差异，必须进入改进计划。

### WAL mmap 化与预分配

当前 WAL 使用 `write_all + sync_all`，segment 懒创建、无预分配、无记录索引，回放把事务读进 `Vec`。qdrant 使用预分配 mmap segment、随机 CRC seed、CRC32C、8 字节对齐、后台预建段和 closed 段保留。

这不是编码格式偏好问题：当前实现存在每次 append 的系统调用开销、滚动时创建文件的尾延迟、无随机访问、无 WAL delta/远端复制所需的历史保留能力。改进方案见修改计划的 WAL 阶段。

### Payload 压缩与存储职责分离

当前页面值只保存 `[u32 CRC32][JSON]`，没有压缩；WAL upsert 内联完整向量和 JSON payload，`points.bin` / `payloads.bin` 是派生文件。qdrant 的 blobstore 默认 LZ4，并把压缩配置作为存储形式的一部分。

缺少压缩会放大 WAL、页面和快照体积；payload 全量内联会让重放重复解析和写入 blob。改进方案见修改计划的 payload 阶段。

### named/sparse 一等公民存储与索引

当前 kind/named 使用 blob + postings，全程 exact scan；ANN 与量化只覆盖默认 dense 空间。qdrant 在 segment 内为多向量空间和稀疏向量提供独立存储、索引和配置。

这不是“当前只支持 dense 所以无需优化”的边界：项目已经有 named/sparse API，但存储层把它们当作派生 blob，无法复用 segment 级索引、版本、字段索引和快照增量。改进方案见修改计划的 named/sparse 阶段。

### 其他索引与检索能力

simvec 缺少 bool/null/geo/facet/full-text/nested 索引、选择性估计、持久化 posting、HNSW graph healing、payload block、压缩邻接、多数据类型和 multivector。现有 `postings/` 是简单 bucket 格式，不承担 BM25、位置索引和 tokenizer。

这些能力差距影响查询性能和适用面，应在字段索引和 ANN 阶段中按依赖顺序补齐。

## 工程问题

`AGENTS.md` 描述单写者锁按“规范化路径”引用计数，但 `src/storage/generation.rs::CollectionLock` 实际按 `(dev, ino)` 引用计数，以支持 promote/restore 重命名后继续共享锁。代码注释与文件行为一致，项目说明需要更正。

## 仍然合理的差异

### 单写者与单一提交点

本地引擎对每个集合只允许一个进程写入。`generation.bin` 作为唯一提交点，比多 segment 版本协商更容易验证；发布后进入 poison 状态也比继续猜测磁盘状态更安全。该模型可以保留。

### 无集群能力

simvec 是本地库，不需要 shard 副本、Raft、clock 冲突解决、WAL delta 复制和远程对象存储。这些是 qdrant 的分布式需求，不是当前绑定矩阵的必要能力。

### 当前不提供格式向后兼容

项目在开发阶段把格式版本固定为 1，旧文件直接拒绝或丢弃派生数据。该策略当前合理；但长期方案已在 `docs/plan/long-term/versioning-and-migration.md` 单独设计，进入生产使用前必须按该方案启用版本演进能力。
