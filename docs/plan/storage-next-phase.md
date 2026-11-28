# 存储下一阶段任务（N1–N7）完整分析

本文是存储层改造的**下一阶段独立任务文档**：自包含、可单独评审、可单独排期，不依赖阅读此前的整改清单即可理解背景与决策。

- **事实源**：`AGENTS.md` 的存储章节。凡是本文与 `AGENTS.md` 冲突的，以 `AGENTS.md` 为准，并应把冲突当成文档缺陷修掉。
- **前序文档**：`docs/plan/storage-remediation-tasks.md`（A/B/C/D 缺陷与遗留清理，已完成）、`docs/plan/storage-remaining-tasks.md`（E1–E4 的阶段性记录）。
- **本文对前序结论的修订**：`storage-remaining-tasks.md` §E1 断言"段目录（catalog）是段级压实的前置条件"。重新核对布局后该结论**不成立**：段目录名已经是 slot base，只要段内紧凑、段地址不变，读写公式完全不用改，无需 catalog。本文 §2.3 给出推导与取舍，catalog 降级为"地址空间回收"的可选后续项。

> **实施状态（2026-11）**：N1a（点级追加写 + 派生 free list）、N7（`Turbo` create 期拒绝）、N4 的 Rust 侧（`maintenance_status` 的 C ABI 形状断言）、N6 的 I/O 计分骨架、N5 的六窗口注入骨架、N2 的「门内只剩发布」（tracker 自封顶 + `gate_wait_ms`）已落地；N1b/N1c/N1d/N3/N4b 仍待实施。代码级拆分、逐条验收用例、以及实施中对本文的 **8 处修订**（其中 4 处改变结论：free list 收窄到尾段、重放槽号不再保证一致、payload 段内不可变必须早于 N1b 的硬链接、`quant.bin` 必须与 N1b 同里程碑）见 [`storage-next-phase-impl.md`](./storage-next-phase-impl.md)，分阶段执行顺序与逐阶段状态见 [`storage-next-phase-stages.md`](./storage-next-phase-stages.md)（含实施后新增的 2 条修订 R9/R10）。

---

## 0. 现状基线（进入本阶段时的代码事实）

| 维度 | 现状 |
| --- | --- |
| 提交点 | `generations/g<id>/` + manifest 是唯一提交点；发布后失败进 poison，等待重开（`compaction.rs:693`） |
| 顺序权威 | WAL 是事务顺序权威；`wal/blobs.bin` 承载引用式记录的大块数据，只追加（`wal_blobs.rs`） |
| 派生数据 | 字段索引、ANN、量化码、payload 页均可丢弃重建 |
| 段布局 | `segments/<start>/{vectors.bin, keys.bin, payloads_store/, tombstones.bin, id_tracker.bin, ANN 文件}`，`<start>` 就是该段的 slot 基址（`segments.rs:103`） |
| slot 寻址 | `slot = index * segment_slots + local`，`index = slot / segment_slots`，无目录参与（`segments.rs:133`、`vectors.rs:262`） |
| 分配 | `meta.next_slot` 单调递增，删除不回收；无 free list（`store_write.rs:435`） |
| 压实 | 全局重排 slot 到 `0..live_count`，整代重写所有文件（`compaction.rs:120`、`:502`） |
| checkpoint | 批量刷盘不关门；关门后只复刷竞态段（`store_maintenance.rs:85-108`） |
| 快照 | 持 `compact_mutex` + 队列门，不持 store 读锁；整文件走 `FICLONE` 优先（`storage.rs:536-600`，`clone_file` 在 `:947`） |

本阶段要解决的四类问题：**压实成本与垃圾量无关**、**两个维护操作仍然全局停写**、**验证矩阵缺口**、**若干配置/元数据缺口**。

---

## 1. 任务总览

| 编号 | 任务 | 优先级 | 依赖 | 估时 | 一次性切换 |
| --- | --- | --- | --- | --- | --- |
| N1a | 点级追加写 + slot free list | P0 | — | 5–7 人日 | 是 |
| N1b | 段内紧凑压实（未选中段复用） | P0 | N1a | 6–8 人日 | 是 |
| N1c | blob 归属到段（`segments/<start>/blobs.bin`） | P1 | N1b | 3–4 人日 | 是 |
| N1d | 退役整代 `quant.bin` | P1 | N1b | 2–3 人日 | 是 |
| N2 | 摘掉 checkpoint 的队列门 | P1 | N1b（可选，见 §3） | 4–5 人日 | 否 |
| N3 | 摘掉快照的队列门 | P2 | N1a | 3–4 人日 | 否 |
| N4 | 绑定端到端用例（G1） | P2 | — | 2 人日 | 否 |
| N5 | 崩溃注入矩阵（G3） | P1 | N1b、N2 | 4–6 人日 | 否 |
| N6 | 性能基线（G4） | P3 | N1b | 1–2 人日 | 否 |
| N7 | `Turbo` 配置在 create 时拒绝（G5） | P3 | — | 0.5 人日 | 否 |

```text
N1a ──> N1b ──> N1c
          │  └──> N1d
          │
          ├──> N2 ──┐
          └──> N5 <─┘
N1a ──> N3
N7、N4、N6 独立可并行
```

**排期建议**：N1a+N1b 作为一个里程碑（唯一可交付的最小闭环，缺任一项都没有收益）；N7/N4 可插空做；N5 跟在 N1b+N2 之后一次性补齐；N1c/N1d 可以延到下一个里程碑，但要在 N1b 时把文件布局预留好。

---

## 2. N1 段级压实与 slot 回收

### 2.1 现象与代价

`compact()` 只要 `tombstone_count > 0` 就走完整流程（`compaction.rs:393`）：

1. `plan_slots` 把全部存活点压成 `0..live_count`（`compaction.rs:120`），`meta.next_slot = plan.live_count`（`:511`）。
2. 在 staging 目录里重写**每个段的每一个文件**：`vectors.bin`、`keys.bin`、`payloads_store/`、`tombstones.bin`、`id_tracker.bin`、每段 ANN 图，外加整代 `quant.bin`、`quant_meta.bin`、字段索引、`segments.json`、`meta.bin`（`compaction.rs:502-687`）。

后果：删 1 个点也重写全库；I/O 与**总数据量**成正比、与垃圾量无关。10 GB 集合回收 100 MB 垃圾要写 10 GB，并让所有派生结构（HNSW/IVF/字段索引/filter bitmap/reverse 映射）全部失效重建。

### 2.2 根因：三处耦合，缺一处都做不成段级压实

| 耦合 | 证据 | 影响 |
| --- | --- | --- |
| slot 地址是算术推导，不是查表 | `segments.rs:133` `segment_index`、`vectors.rs:262` `write_slot` 都用 `slot / segment_slots`；段目录名就是基址 `segments.rs:103` | 段不能搬家、不能合并、长度不能随意变，否则 `slot → (段, 偏移)` 失效 |
| 已发布段可被原地改写 | `store_write.rs:340` 对已存在 id 直接 `vectors.write_slot`；`payload_store.rs:347` `put_at` 释放旧块后重新分配（可能复用同页）；`store_write.rs:350` 对整代 `quant.bin` 原地写 | 一个段的字节不是其 lsn 的函数 → 段不能被两个 generation 共享（硬链接会互相污染），快照也无法无锁复制 |
| 压实语义绑定全局重排 | `compaction.rs:120` + `meta.next_slot = live_count` | 即使只重写一个段，也要把其余段的 slot 全部改写，等于全量重写 |

### 2.3 设计决策：不做 catalog，固定段地址

**结论**：段内紧凑 + 段地址不变，就足以把压实降为"只重写选中段"。

推导：设段 `i` 的基址 `base = i * segment_slots` 不变，段内把 live 点紧凑到 `local ∈ [0, live_i)`，则

- 读路径 `index = slot / segment_slots`、`local = slot % segment_slots` **不变**（`vectors.rs:262`、`segments::segment_index`）；
- `local < live_i ≤ segment_slots` 恒成立，段文件长度可以缩短，读取边界检查（`read_slot` 的 `end > seg.len()`）天然兜住；
- 未选中段完全不动，其文件可以被新 generation **硬链接**复用；
- 选中段的 ANN/keys/payloads/tracker 都是段内结构，重建范围就是该段。

代价（必须接受并记录）：

- **地址空间不回收**。`slot_capacity` 只增不减，`next_slot` 高位被永久占用。删除 90% 数据后磁盘能回收，但 filter bitmap、`reverse`、ANN 的 slot 上限仍按历史峰值。要回收地址空间必须重排，那时才需要 catalog（见 §2.9）。
- 段数只增不减，除尾部外每段至少一个目录 + 若干小文件；空段退化为"零长文件"而不是"消失"。

这两条是显式取舍，不是疏忽。相比"引入 catalog 让每个读写路径都查表"的代价（`vectors`/`keys`/`payload`/ANN/bitmap 全线改），固定地址方案的改动面小一个数量级，且崩溃一致性模型完全不变。

### 2.4 N1a 点级追加写 + slot free list

**问题**：现在 upsert 已有点 = 原地覆盖（`store_write.rs:340`），所以"干净段"不存在——任何段随时可能被改脏，段级压实就没有可复用的输入。

**方案**：

1. **upsert 已有点改为追加写**：分配新 slot（尾段或 free list），写新向量/新 payload 页，把**旧 slot 置墓碑**，`reverse[id] = new_slot`，tracker 的 `point_lsn` 推进。
2. **free list 派生，不持久化**。free offset 集合 = `{slot | 段内已墓碑且 slot < segment_next_slot}`，可从 tombstone 表 + 每段 tracker 重建；每个段的 tracker 已经持久化了 `offset`（`segments.rs:220`），足以在 open 时重算。**不新增权威状态**——这一点很关键，避免又引入一个"必须与 WAL 严格同步的第四份元数据"。
3. **分配顺序**：新 id 走尾段 `segment_next_slot`；replay 与实时写走**同一个** `allocate_slot_locked`，保证日志顺序与槽位一致（沿用 `resolve_upsert_slot_locked` 的位置，`store_write.rs:422`）。
4. **保留 `delete_lsn` 语义**：被墓碑的旧 slot 保留 `delete_lsn`，用于拒绝旧 WAL 记录复活（现有逻辑不变）。

**不变量**：

- 一个 id 在任一瞬间只有一个 live slot；`reverse` 指向它；其余 slot 全部墓碑。
- 一次 upsert 不修改任何**已发布段**里既有 live slot 的字节（只写入新位置 + 改墓碑位）。
- 追加写不改变 `last_applied_txn` 的单调性，也不改变 WAL 记录的幂等性（同一事务重放时 id 已指向新 slot，按"已应用"跳过）。

**代价与缓解**：

- **写放大转移**：更新不再覆盖，改为"写新 + 留旧墓碑"，垃圾增长更快 → N1b 必须在同一里程碑交付，否则磁盘只增不减。
- **读放大**：一个 id 的历史版本留在段里，但不被读取（只按 `reverse` 定位），不影响正确性。
- **必须重新审视的既有假设**：`space_ann.rs:129` 的 `write_slot` 写的是 `spaces/<space>/ann/` 下的**派生**向量副本，整份重建，不受影响（本文明确区分，避免下一轮误改）。

**验收**：

- `upsert` 已有点后，旧段文件的内容哈希与 mtime 不变（用例直接读文件字节对比）。
- 1000 次同 id 更新 → 1000 个墓碑、1 个 live slot、`count() == 1`，重开后一致。
- 重放路径一致性：同一批事务先实时应用、再"崩溃重开 + 重放"，两边 `id → slot` 与 `count()` 完全相同。

### 2.5 N1b 段内紧凑压实

**选择规则**：`segments::select_for_compaction(entries, threshold)`（`segments.rs:205`）已经存在并在 `compact()` 里作为 advisory 使用（`compaction.rs:387`，注释说明它"只是调度建议、不是闸门"）。N1b 把它从 advisory 变成**真正的选择**：

- 选中：`garbage_ratio ≥ threshold` 的段；
- 排除：可追加尾段（单段集合例外，沿用现有规则，见 `deferred-storage-completion.md` §2.3）；
- 显式 `compact()` 命令保持"全选"语义（用户要求回收就回收，不由阈值决定）；后台维护走阈值选择。

**重写范围**（只对选中段）：

| 文件 | 动作 |
| --- | --- |
| `vectors.bin` | 只写 live local，长度缩短到 `live_i * dim * 4` |
| `keys.bin` | 按新 local 重写记录数组 |
| `payloads_store/` | 只复制 live slot 的页（`PayloadStore::compact_to` 已有按 slot 复制能力，改为按段范围） |
| `tombstones.bin` | 清空该段（紧凑后无墓碑） |
| `id_tracker.bin` | 该段 live 点的 offset 改写为新 local；被丢弃 id 保留 free 条目 |
| ANN（HNSW/IVF） | 只重建该段的图 |
| 未选中段 | **逐文件硬链接**进新 generation（同 inode，零拷贝）；ANN 不重建 |

**`segments.json` 必须补两项**（原 T4 明确延后，理由正是"等段级压实"）：每段文件的 `len` 与 `crc32c`。有了它，快照/校验/诊断都不必重新扫全量；也才能断言"未选中段在新旧 generation 里字节相同"。

**`meta` 语义调整**：`slot_capacity` 不变（地址空间不缩），`next_slot` 只在尾段被压实时才可能下调；`live_count`/`tombstone_count` 随之更新。这与现状 `next_slot = live_count` 的差别要在 `AGENTS.md` 里写明，否则读者会以为漏改。

**发布与崩溃一致性**：完全沿用现有序列（staging → `DataFiles::open` 预校验 → manifest rename → 内存切换 → poison 标志），不变。硬链接的额外要求：**新 generation 与旧 generation 共享 inode，所以旧 generation 的 GC 不能"删目录就完"**，必须等读者 guard 归零（现有 `generation` 的 Arc 生命周期已经保证），且**任何写入必须先完成"当前 generation 的独占"**——N1a 的追加写已经消除了对共享 inode 的原地改写，这是 N1b 能安全硬链接的前提，也是 N1a 必须先做的唯一原因。

**验收**：

- 只选中一段时，未选中段在每个 generation 里的 **inode 不变**、mtime 不变、字节相同。
- 压实 I/O 上界：`Σ(选中段字节数) + 目录/manifest`，用 `MaintenanceStatus` 或测试钩子统计实际写入字节。
- 压实后 `count()`/`get()`/`search()`/`scroll()`/filter 查询逐点与压实前一致（随机集合 + 随机过滤条件对比）。
- 反复"写 → 删 → 压实 → 重开"100 轮，磁盘占用单调不增（除 WAL 波动）、`live_count` 正确。

### 2.6 N1c blob 归属到段

**问题**：`wal/blobs.bin` 是单文件、只追加，只在整体 truncate 时 `reset()`（`wal_blobs.rs:176`）。checkpoint 的部分截断不回收它（注释已说明这是有意的），因此**长时间运行的集合里 blob 只增不减**，`wal_blob_bytes` 正是为此新增的可观测项。

**方案**：`segments/<start>/blobs.bin`，`BlobRefSpec` 增加 `segment_start: u32`；压实只复制 live 点引用的 blob，被覆盖/删除的 blob 随段重写回收；`wal_blob_bytes` 改为按段求和。

**风险**：跨段引用（一个事务里 upsert 多个段）会让 blob 写两次或需要"归属判定"；建议归属规则取"事务里第一个点所在段"，压实按引用计数复制。这条要评审，因为它把"blob 是纯派生"变成"blob 有归属"，是一处语义升级。

### 2.7 N1d 退役整代 `quant.bin`

`quant.bin` 是整代可写数组（`store_write.rs:350` 原地写），与"段不可变"直接冲突：即使 N1a/N1b 完成，量化码仍是唯一挡住快照无锁复制（N3）与段级复用完整性的结构。`quant_seg_<start>.bin` 已经存在（AGENTS.md 记为 T9 的重复存储），把它提升为唯一副本、`quant_meta.bin` 只描述段副本，磁盘占用近半，且量化码获得与向量相同的段级生命周期。

### 2.8 N1 整体不变量（评审必须逐条确认）

1. 已发布段在一个 generation 内**字节不可变**（除追加区域）。
2. 段地址（目录名/slot 基址）在 generation 之间对同一段保持不变。
3. 未选中段在压实后与新 generation 共享 inode；选中段的旧 inode 只被旧 generation 引用。
4. 崩溃只可能留下两种可见状态：旧 manifest + 旧目录，或新 manifest + 新目录。
5. 目录/清单反例一律 `CorruptData`：条目重叠、`start` 不与 `segment_slots` 对齐、段文件长度超过 `live_i * dim * 4`、`segments.json` 的 `len/crc` 与实际不符。
6. 读侧只接受当前格式版本，不按版本分支。

### 2.9 可选后续：地址空间回收（catalog）

只有在"必须收缩 slot 地址空间"时才需要：新增 `segments/catalog.bin`（有序 `{slot_base, slot_len, physical_id}`），读写路径按 `slot_base` 二分定位，段目录名与 `physical_id` 解耦，于是可以合并/搬家/彻底删除空段。**本阶段不做**，理由是收益（降低 bitmap/reverse 的内存上限）远小于改动面（所有读路径 + ANN 定位 + 崩溃一致性重新验证）。若将来要做，它应该是独立里程碑，且与 N1 之间不能有中间态。

---

## 3. N2 摘掉 checkpoint 的队列门

### 3.1 现在门保护什么

`checkpoint_inner`（`store_maintenance.rs:85`）在批量刷盘之后 `queue.quiesce()`（`:106`），关门期间做四件事：

1. 复刷竞态段（`:108`，已降为增量）；
2. `publish_id_trackers`（`:115`）——把 `.staged` rename 成可见 tracker；
3. `meta.save` + `segments.json`（`:120`、`:137`）；
4. 写 checkpoint marker + `truncate_wal_through`（`:163` 起）。

门保护两件事：**WAL 不能把排队中的事务切出日志**；**可见的 tracker 必须指向已 fsync 的字节**。

### 3.2 方案：冻结前沿 + 派生截断点

1. **冻结前沿**：批量刷盘结束时取 `frozen = meta.last_applied_txn`，tracker 的 `lsn` 记为 `frozen`，`point_lsn` 按 `min(point_lsn, frozen)` 封顶（`read_id_trackers` 已有同样的上限裁剪，语义一致）。之后新写入落在 `frozen` 之上，不属于这份 tracker。
2. **截断点由持久化前沿决定**：`min_persisted = min(segment_persisted)`，再与 `wal_pins.lowest()` 取小（`storage.rs:698` 已实现 clamp）。排队中事务所在的段由 `truncate_through` 的"段首 LSN"规则保留，尾部段永不删除（`wal.rs:908`）。
3. **marker 顺序不变**：marker 在 `inner.write()` 内分配 LSN 并 append，与 worker 的分配同锁，天然有序。

去掉门之后剩下的窗口是"写 `meta.bin` + marker append + 段删除"，都在 `inner.write()` 内，微秒级。

### 3.3 必须先回答的语义问题

`tracker.lsn` 一旦允许落后于实时状态，open 侧"重放会重新分配 slot"的语义必须重新推演：重放可能给一个 id 分配新 slot（N1a 追加写之后更容易发生），而 tracker 里的 `point_lsn` 已被封顶到 `frozen`，重放会重新应用 `frozen` 之后的事务。要求：**重放对同一 (id, lsn) 幂等**，且 `reverse` 的重建顺序与实时应用一致。这是 N2 的主要风险，必须配专门的并发 + 崩溃用例；现有用例没有覆盖“gate 未关闭时 checkpoint 与写入并发”这条路径。

### 3.4 验收

- 持续写入（单线程 + 多线程）期间 checkpoint，断言**没有事务被拒绝或延迟到 gate 关闭窗口**：测试钩子记录 `quiesce` 窗口内提交的事务数（应为 0 等待，或断言 checkpoint 期间 upsert 的 p99 不超过基线的 N 倍）。
- 崩溃注入：tracker 发布后 / marker 后 / truncate 中三个窗口各注入一次，重开后 `count()`、`id → slot`、payload 全部一致。
- 不再出现"checkpoint 期间写请求排队等待"的观测（`MaintenanceStatus` 加 `gate_wait_ms` 计数器，可为空但要有）。

---

## 4. N3 摘掉快照的队列门

### 4.1 现状

`snapshot_capture_inner`（`storage.rs:536`）持 `compact_mutex` + `queue.quiesce()`，不持 store 读锁，整文件复制走 `FICLONE` 优先（`clone_file`，`storage.rs:947`）。门仍然覆盖整个复制窗口。

### 4.2 为什么现在摘不掉

复制期间如果某点被 upsert **原地覆盖**（当前实现的行为），快照会包含 `lsn` 之后的字节，而 WAL 副本里没有对应事务，重放补不回来 → 快照不自洽。这与 §2.2 第二条是同一个根因。

### 4.3 方案（N1a 完成后）

1. 取 `pinned = last_applied_txn`，注册 WAL ack pin（`pin_wal_ack`，`storage.rs:683`）。
2. 复制 generation **文件树**：未变化文件用 reflink/硬链接（`snapshot_to` 已有增量复用），尾段只复制 `≤ pinned` 的**前缀**（已实现 `copy_prefix`，`storage.rs:914`）。
3. 复制期间允许写入：新事务落在 WAL，快照清单一并复制 WAL 段（从 `pinned` 起的段），恢复时按 `pinned` 重放补齐。
4. 唯一的门缩短到"取 lsn + 记清单"这一步；不需要停写。

### 4.4 验收

- 快照窗口内并发 upsert 成功（断言无阻塞、无 gate 等待）。
- 快照 → 恢复 → 重开：`count()` 等于 `pinned` 时刻的 `count()`，且随后的 WAL 重放把快照补到导出时刻的最新状态（两种期望值都要在用例里写清，避免"恢复后多/少几条"的分歧）。
- `verify_snapshot` 对"前缀复制 + 全量 CRC"保持一致（`file_checksum_prefix` 已实现）。

---

## 5. N4–N7 验证与配置缺口

### 5.1 N4 绑定端到端（G1）

`maintenance_status` 与 4 个快照 API 已在 C ABI 暴露，python/js/go/java/r 均有实现（JS 的 `DEFINITIONS` 与 Python 的 `argtypes` 已补齐）。缺的是**端到端用例**：每种绑定各加一条 `create → upsert → maintenance_status → snapshot → verify → restore` 冒烟；`src/c_api/tests.rs` 补 `maintenance_status_c_abi`，断言 `segments` 数组的形状（而不只是 `healthy` 布尔）。

### 5.2 N5 崩溃注入矩阵（G3）

| 窗口 | 注入点 | 期望 |
| --- | --- | --- |
| blob 写后、WAL append 前 | 丢弃 WAL 记录 | 重开忽略孤儿 blob，`CorruptData` 不触发 |
| WAL fsync 后、apply 前 | 强杀 | 重放补上，`count()` 正确 |
| apply 后、ACK 前 | 强杀 | 重放幂等，不重复计数 |
| ACK 后、truncate 中 | 强杀 | 段与 tracker 自洽，截断可重入 |
| checkpoint 批量刷盘后 | 强杀 | 只有 `.staged` tracker，open 忽略 |
| tracker 发布后、manifest 前 | 强杀 | `point_lsn` 被 `tracker.lsn` 封顶，重放补齐 |
| 压实发布后、内存切换前 | 强杀 | poison → 重开按新 manifest 加载 |

现有 `tests/recovery_test.rs` 覆盖了其中一部分，N5 要把"逐窗口一一对应"补成矩阵，并用统一的注入钩子（环境变量 + 文件标记）而不是各自特判。

### 5.3 N6 性能基线（G4）

`benches/quick.rs` 是 `autobenches = false` 的共享助手模块，新档必须落在具体 bench 文件（`wal_bytes_vs_checkpoint`、`compaction_io_vs_garbage`）。验收口径：**压实写入字节数随垃圾段字节数线性增长，不随总数据量增长**——这是 N1b 的核心指标，必须在 bench 里可复现。

### 5.4 N7 `Turbo` 配置拒绝（G5）

`VectorDataType::Turbo`（`src/types/config/dtype.rs`）目前没有存储实现。create 时若配置为 Turbo，应直接 `InvalidConfig` 而不是等到第一次写入才报错或静默按 f32 存。改动点：`CollectionConfig` 校验入口 + 一条用例。

---

## 6. 实施顺序与里程碑

| 里程碑 | 内容 | 交付判据 |
| --- | --- | --- |
| M1（N1a+N1b） | 追加写 + free list + 段内紧凑 | 压实 I/O 与垃圾量成正比；未选中段 inode 不变；全量测试 + 崩溃注入通过 |
| M2（N5、N2） | 崩溃矩阵 + checkpoint 去门 | 持续写入下 checkpoint 不排队；7 个崩溃窗口全绿 |
| M3（N1c、N1d、N3） | blob/量化按段 + 快照去门 | 快照窗口内可写；blob 随段回收；磁盘占用减半 |
| M4（N4、N6、N7） | 绑定冒烟、bench、Turbo | 各绑定 CI 冒烟通过；bench 进基线 |

**M1 不可拆分**：只做追加写不做段级压实，磁盘只增不减；只做段级压实不做追加写，"干净段"永远不存在，压实退化为今天的全量重写。这是本阶段唯一一条硬约束。

---

## 7. 需要评审拍板的四个点

| # | 问题 | 选项 | 推荐 |
| --- | --- | --- | --- |
| 1 | 追加写 vs 保留原地写 + 段级"写时复制" | A 追加写（本文方案）；B 原地写 + 每次压实前 fork 段文件 | A。B 的复制成本等于今天的全量重写，且没有消除快照无锁复制的前置障碍 |
| 2 | free list 是否持久化 | A 派生（从 tombstone + tracker 重算）；B 写进 tracker | A。少一份权威状态，崩溃窗口少一个 |
| 3 | 尾段是否参与压实 | A 不参与（保持可追加）；B 参与并下调 `next_slot` | A。B 会让 `next_slot` 回退，与 WAL 重放分配交互复杂，收益仅"最后一段的垃圾" |
| 4 | blob 归属规则 | A 事务首点所在段；B 每段各存一份引用 | A + 引用计数。B 会让同一 blob 多份副本，回收更麻烦 |

---

## 8. 明确不做（本阶段）

- **地址空间回收 / catalog**：见 §2.9，收益不足。
- **`quant.bin` 之外的量化格式变更**：N1d 只做"按段"，不改编码。
- **wasm 目标下的存储**：内存引擎，不在此列。
- **CLI 新命令**：E4 已补齐 `maintenance_status` 与 4 个快照命令；`standalone-cli-shell.md` 里其余规划项按该文档节奏走。
- **格式版本升级**：本阶段不改动磁盘格式版本。

---

## 9. 验收总表

| 检查 | 命令 | 通过标准 |
| --- | --- | --- |
| 编译 | `cargo +1.97.0 check --all-targets`、`--features c_api` | 0 错误 0 警告 |
| 单测 | `cargo +1.97.0 test` | 全绿，且新增用例覆盖 §2.4/§2.5/§3.4/§4.4 |
| CLI | `cargo +1.97.0 test`（`cli/`） | 全绿 |
| 绑定 | 各绑定冒烟 | N4 的 5 条链路通过 |
| 崩溃 | `tests/recovery_test.rs` + 注入钩子 | 7 个窗口全绿 |
| 性能 | `benches/compaction_io_vs_garbage` | 压实字节数 ∝ 垃圾量 |
| 文档 | `AGENTS.md` | 段不可变、`next_slot` 语义、blob 归属、`segments.json` 新字段四处更新 |

工具链注意：仓库 `rust-version = "1.97"`，低于该版本的默认 `rustc` 会在 MSRV 检查处失败，命令需显式 `+1.97.0`。
