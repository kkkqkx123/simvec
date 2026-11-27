# 存储结构审计与加固方案

本文对比当前 `simvec` 原生存储与 `ref/qdrant`，给出完整问题清单、本轮已落地的修改和剩余改进方案。范围只包括本地引擎的原生存储；wasm 内存引擎不在讨论范围内。

## 现状对比

| 维度 | simvec | ref/qdrant |
| --- | --- | --- |
| 组织方式 | 集合根目录保存类型标记、`wal/`、`.lock` 与 `generation.bin`；全部 serving 文件在 `generations/g<id>/`，dense/kind/named 共用一套发布协议 | collection → shard → 多个 segment；appendable segment 接收写入，immutable segment 承载压实后的数据和索引 |
| 单写者 | 打开集合时持有 `.lock` 上的 advisory flock，进程退出自动释放 | WAL 目录持有 flock |
| 点标识 | 全局 IdMap + 全局 LSN 与逐点版本（含删除版本） | 每 segment 的 id tracker：external id → internal offset + version / delete version，另有 compressed / disk 变体 |
| 向量存储 | dense 分段 mmap 定长 f32、槽位原地覆盖；kind/named 的可变长向量进 `points.bin` blob | chunked appendable mmap + deleted bitvec、immutable mmap、graph-inline、多种数据类型与 multivector |
| Payload | `payloads_store/`：`config.json` + `tracker.bin` + `bitmask.bin` + `pages/page_*.bin`，内容可从 WAL 派生；kind/named 用 `payloads.bin` blob | blobstore gridstore / logstore，tracker journal、compacted tracker、gap 管理、打开时 pointer 一致性校验 |
| WAL | 集合级 `wal/<20位首LSN>.bin`，postcard + CRC32，批量 group commit，整段 unlink；torn tail 在追加前截回有效前缀 | shard 级 mmap 段，magic/version/随机 CRC seed、CRC32C、8 字节对齐、预建段队列、closed 段保留、first-index ACK |
| Checkpoint | flush 派生文件 → `meta.bin` → snapshot → `generation.bin.lsn` → `wal/acked-lsn` → `truncate_through`；任一 fsync 失败即中止 | segment 各自维护 version tracker 与 segment.json，flush 确认后才 ACK，WAL 才回收 |
| 压实 | 在 staging 构建完整新代、fsync、rename 目录、发布清单、切换句柄、最后截断 WAL；dense 竞争激烈时仍有写锁回退 | optimizer 构造完整新 segment，proxy 吸收期间写入，`swap` 一次切换 |
| Payload 索引 | schema 落盘、内容内存重建；Map/Numeric；主要加速 `must` | map/numeric/full-text/geo/facet/nested/null、mutable/immutable/on-disk、选择性估计 |
| ANN | 集合级 HNSW/IVF；新点进 `pending`，查询补扫；压实靠 remap | 每个 immutable segment 独立建图，appendable 段精确扫，graph healing、payload block |
| 具名/稀疏 | blob + postings + snapshot，全程 exact scan | segment 内一等公民，有独立存储与索引 |

## 完整问题清单

### 数据安全

1. **fsync 失败被吞掉**：`vectors.bin`、`keys.bin`、`points.bin`、`payloads.bin` 和 payload pages 的 `sync()` 曾用 `.ok()` 忽略 I/O 错误，并且先清除 dirty 标记。checkpoint 仍继续写 snapshot、推进 generation lsn 并截断 WAL，失败写到设备上的数据无法从 WAL 恢复。
2. **torn tail 成为后续日志的墙**：最后一段的坏记录被 replay 静默跳过，但新事务仍追加在物理文件末尾。下一次启动会再次停在同一坏记录，吞掉其后所有已提交事务。
3. **generation 发布后的错误路径**：清单替换成功后若 `DataFiles::open` 或其他步骤失败，进程继续持有旧 generation 的内存句柄；磁盘清单却已指向新代，后续 checkpoint 或 `stage()` 会写错目录，甚至删除已发布的新代。
4. **缺少跨进程单写者约束**：没有目录锁时两个进程可同时分配相同 lsn、同时改同一组文件；同进程的多个句柄（例如维护线程克隆的 handle）则需要共享同一把锁。
5. **derived 文件缺少重建式自愈**：payload tracker/page 在 generation lsn 尚未推进时仍只是派生状态，但 pointer 不一致会直接报 `CorruptData`，即使完整 WAL 还在。qdrant 通过 tracker journal 和打开时一致性修复处理此类窗口。

### 延迟与容量

6. **压实粒度是整集合**：dense 压实重写全部文件，4 次乐观重试后退回写锁下重写；kind/named 压实全程持有 `gate` 与 `inner` 大锁，查询也被阻塞。qdrant 用 appendable/immutable segment 与 proxy 吸收期间写入。
7. **dense snapshot 被 payload 写轻易作废**：快照曾用 payload tracker 版本做交叉校验；任何 payload-only 更新都会丢弃 id 目录，打开退化为全槽扫描并重建过滤位图与字段索引。
8. **启动重建成本高**：过滤位图和字段索引在打开时按全部 live 槽位重建；字段索引没有持久化 posting/bitmap。
9. **ANN 是集合级结构**：新点进入 `pending`，查询补扫成本随积压增长；压实靠 remap 整张图，覆盖不全时只能重建。
10. **WAL 编码与 IO 偏简单**：整批先编码进 `Vec`，id/payload 双重字符串化；segment 懒创建且 append 路径 fsync 目录；没有 mmap 预分配、随机 CRC seed、8 字节对齐、预建段和 closed 段保留。
11. **字段索引能力有限**：只有内存 Map/Numeric，`must_not` / `should` / `min_should` 退化为后置过滤；缺少 full-text、geo、facet、nested、null、on-disk immutable 索引与选择性估计。
12. **没有快照一致性 pin**：generation 目录可被复制，但没有 qdrant 式 WAL ACK pin 和文件清单，复制期间的写入可能只存在于被排除的 WAL 中。

### 工程约束

13. **文档漂移**：审计文档曾把已经修复的问题继续列在“问题清单”中；`AGENTS.md` 也需要同步新增的锁文件、fsync 契约和 poison 语义。
14. **重复生命周期代码**：`NamedStore::open` 曾连续两次调用 `Generation::open`，属于无意义的重复逻辑。
15. **空闲 worker 持有 store 强引用**：commit queue worker 曾在等待批次前 upgrade `Weak<CollectionStore>`，使 store 的 `Drop` 永远无法触发，从而无法关闭 worker、也无法释放目录锁。worker 现在先取批次再 upgrade。

## 本轮已落地的修改

### fsync 契约

`Vectors::sync`、`BlobDirectory::sync`、`PointBlobs::sync` 和 `Pages::sync` 全部返回 `Result`：fsync 成功后才清 dirty 标记，失败时保持 dirty 并把错误上抛。dense `checkpoint_inner`、`PayloadStore::checkpoint`、kind/named checkpoint 与 staging compaction 都改为 `?` 传播。因此 checkpoint 任一步失败都会停在 generation 清单之前，WAL 仍然完整。

### WAL torn tail

`WalState` 记录 active segment 的有效前缀长度。`replay_after` 返回最后一段的合法末尾；`append_batch` 在追加前把物理文件截回该前缀并 fsync。坏记录不再挡住之后写入的事务。`wal.rs` 增加“torn tail 后重开、写入、再重开”的回归测试。

### 单写者与发布 poison

生命周期模块提供 `CollectionLock`：Unix 上对集合根目录的 `.lock` 文件做 `LOCK_EX | LOCK_NB`，按规范化路径引用计数；同进程内第二个句柄共享同一把锁，其他进程打开同一目录会被拒绝。最后一个句柄释放时解锁。dense、kind、named 的 create/open 都持有该锁。

三类 store 增加 `generation_poisoned`：generation 清单发布成功即置位，只有完成内存切换、WAL 标记和截断后才清除。中间任何失败都让集合保持可读但拒绝写入/checkpoint/压实，必须重新打开。这避免了磁盘清单与内存句柄分叉后继续写错文件。

### dense snapshot 与 payload 解耦

`dense_snapshot.bin` 不再携带 payload tracker 版本。快照负责 id→slot 目录与逐点版本，payload 字节从当前 `PayloadStore` 读取；generation lsn 之后的 payload 记录由 WAL 回放重新应用。payload-only 写入不再让整个快照失效。`Tracker` 中的变更计数保留为文件内部状态，不再参与快照匹配。

### 清理

删除 `NamedStore::open` 中重复的 `Generation::open` 调用；commit worker 不再在空闲等待时持有 store 的强引用；测试用例在重开同一目录前显式释放旧句柄，promote 成功后立即释放旧 live 句柄。

## 剩余修改方案

### 派生文件自愈

当 generation lsn 尚未推进、WAL 仍完整时，payload tracker/pages 的一致性问题应当触发派生重建而不是直接隔离集合。方案是把 `PayloadStore::open` 的一致性校验分成“可重建窗口”和“已提交损坏”两类：前者清空 allocator 状态并从 generation lsn 回放；后者保持 `CorruptData`。payload value 增加 CRC，避免长度合法但内容被覆盖的静默错误。

### segment 化与有界压实

把 dense 的文件集合拆成 appendable 与 immutable 段，清单记录段描述；写入只进 appendable 段，optimizer 在 staging 构建 immutable 段，提交时一次切换 `SegmentSet` guard，用有界写入队列替代写锁下全量重写。kind/named 先把 blob/postings 的切换完全移出大锁，再逐步采用同样的段协议。

### 按段 ANN

图文件按 immutable 段发布，appendable 段继续精确扫描。`pending` 有界 drain，超限触发该段重建；查询先合并各段候选再统一打分和过滤。段冻结时修复图空洞，避免跨代 remap。

### WAL mmap 化

段文件改为预分配 mmap：头部包含 magic、format version、随机 CRC seed、first-lsn 和 closed 标记；记录 8 字节对齐，CRC 使用段内 seed。批量提交流式写入，不再整体编码进 `Vec`；后台补足空段队列，closed 段按 ACK 水位保留后回收。事务语义、逐点版本、generation 水位和 group commit 顺序保持不变。

### 字段索引能力

抽象 `FieldIndex` 的声明、注册、删除、查询、选择性和 checkpoint 接口；用位图集合运算实现 `must_not` / `should` / `min_should`；补 full-text、geo、facet、nested、null 索引；保存基数与直方图，按选择性选择位图或索引探测；immutable 段发布只读磁盘索引。

### 快照 pin

集合快照先固定 WAL ACK 水位，再复制 serving generation 文件，最后把水位写进快照清单；复制期间的写入必须留在被归档的 WAL 中。这样目录复制不再依赖“恰好没有并发写”。

## 验证重点

本轮回归覆盖：

- fsync 失败时 checkpoint 不推进 generation、不截断 WAL；
- WAL torn tail 后继续写入，重开仍能读到新事务；
- 同进程多个句柄共享目录锁，最后一个句柄释放后可重开；
- generation 发布失败路径把 store 置为 poisoned，拒绝后续写；
- payload-only 写入后 dense snapshot 仍可用于打开；
- 既有 checkpoint、WAL、压实、量化和恢复测试保持通过。

后续结构性修改需要补充：混合写入下的段压实重启一致性、字段索引选择性规划、按段 ANN 的召回对比，以及带并发写的快照复制恢复。目录锁的跨进程互斥需要在独立进程测试中验证；当前单元测试覆盖同进程共享与释放。
