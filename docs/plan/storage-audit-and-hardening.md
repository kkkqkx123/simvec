# 存储结构审计与加固方案

本文对比当前 `simvec` 原生存储与 `ref/qdrant`，给出完整问题清单、本轮已落地的修改和剩余改进方案。范围只包括本地引擎的原生存储；wasm 内存引擎不在讨论范围内。

## 现状对比

| 维度 | simvec | ref/qdrant |
| --- | --- | --- |
| 组织方式 | 集合根目录保存类型标记、`wal/`、`.lock` 与 `generation.bin`；全部 serving 文件在 `generations/g<id>/`，dense/kind/named 共用一套发布协议 | collection → shard → 多个 segment；appendable segment 接收写入，immutable segment 承载压实后的数据和索引 |
| 单写者 | 打开集合时持有 `.lock` 上的 advisory flock，进程退出自动释放 | WAL 目录持有 flock |
| 点标识 | 全局 IdMap + 全局 LSN 与逐点版本（含删除版本） | 每 segment 的 id tracker：external id → internal offset + version / delete version，另有 compressed / disk 变体 |
| 向量存储 | dense 分段 mmap 定长 f32、槽位原地覆盖；kind/named 的可变长向量进 `points.bin` blob | chunked appendable mmap + deleted bitvec、immutable mmap、graph-inline、多种数据类型与 multivector |
| Payload | `payloads_store/`：`config.json` + `tracker.bin` + `bitmask.bin` + `pages/page_*.bin`，页面值带 `[u32 CRC32][JSON]` 前缀，内容可从 WAL 派生；kind/named 用 `payloads.bin` blob | blobstore gridstore / logstore，默认 LZ4 压缩，tracker journal、compacted tracker、gap 管理、打开时 pointer 一致性校验 |
| WAL | 集合级 `wal/<20位首LSN>.bin`，postcard + CRC32，批量 group commit，整段 unlink；torn tail 在追加前截回有效前缀，滚动阈值只看有效长度，重开时校验段连续性 | shard 级 mmap 段，magic/version/随机 CRC seed、CRC32C、8 字节对齐、预建段队列、closed 段保留、first-index ACK |
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
6. **torn tail 可以借滚动变成中间段损坏**：旧 `append_batch` 先按物理长度滚动新段、再截断 torn tail。若坏记录把物理长度推过阈值，它就留在已关闭段里；下次重开时“仅最后一段容忍 torn tail”不再成立，整个集合报 `CorruptData`。
7. **WAL 段连续性没有校验**：`list_segments` 只排序去重，回放不检查上一段末尾 LSN 与下一段首 LSN 的关系。中间段丢失或名字重叠时，回放会静默跳过或重复事务；qdrant 打开时直接拒绝 overlapping / missing segment。
8. **payload 页面值没有内容校验**：pointer 长度合法时，被覆盖的 JSON 只能靠解析偶然发现，无法可靠区分位翻转与合法新值。qdrant blobstore 默认 LZ4 且对值有严格解码错误路径。

### 延迟与容量

9. **压实粒度是整集合**：dense 压实重写全部文件，4 次乐观重试后退回写锁下重写；kind/named 压实全程持有 `gate` 与 `inner` 大锁，查询也被阻塞。qdrant 用 appendable/immutable segment 与 proxy 吸收期间写入。
10. **dense snapshot 被 payload 写轻易作废**：快照曾用 payload tracker 版本做交叉校验；任何 payload-only 更新都会丢弃 id 目录，打开退化为全槽扫描并重建过滤位图与字段索引。
11. **启动重建成本高**：过滤位图和字段索引在打开时按全部 live 槽位重建；字段索引没有持久化 posting/bitmap。
12. **ANN 是集合级结构**：新点进入 `pending`，查询补扫成本随积压增长；压实靠 remap 整张图，覆盖不全时只能重建。
13. **WAL 编码与 IO 偏简单**：整批先编码进 `Vec`，id/payload 双重字符串化；segment 懒创建且 append 路径 fsync 目录；没有 mmap 预分配、随机 CRC seed、8 字节对齐、预建段和 closed 段保留。
14. **字段索引能力有限**：只有内存 Map/Numeric，`must_not` / `should` / `min_should` 退化为后置过滤；缺少 full-text、geo、facet、nested、null、on-disk immutable 索引与选择性估计。
15. **没有快照一致性 pin**：generation 目录可被复制，但没有 qdrant 式 WAL ACK pin 和文件清单，复制期间的写入可能只存在于被排除的 WAL 中。

### 工程约束

16. **文档漂移**：审计文档曾把已经修复的问题继续列在“问题清单”中；README 的集合布局描述也曾落后于 generation 布局。本轮已同步 `AGENTS.md` 的锁、fsync、poison、WAL 连续性与 payload CRC 契约，并更新 README 布局说明。
17. **重复生命周期代码**：`NamedStore::open` 曾连续两次调用 `Generation::open`，属于无意义的重复逻辑。
18. **空闲 worker 持有 store 强引用**：commit queue worker 曾在等待批次前 upgrade `Weak<CollectionStore>`，使 store 的 `Drop` 永远无法触发，从而无法关闭 worker、也无法释放目录锁。worker 现在先取批次再 upgrade。

## 本轮已落地的修改

### fsync 契约

`Vectors::sync`、`BlobDirectory::sync`、`PointBlobs::sync` 和 `Pages::sync` 全部返回 `Result`：fsync 成功后才清 dirty 标记，失败时保持 dirty 并把错误上抛。dense `checkpoint_inner`、`PayloadStore::checkpoint`、kind/named checkpoint 与 staging compaction 都改为 `?` 传播。因此 checkpoint 任一步失败都会停在 generation 清单之前，WAL 仍然完整。

### WAL torn tail 与段边界

`WalState` 记录 active segment 的有效前缀长度。`replay_after` 返回最后一段的合法末尾；`append_batch` 现在**先**把物理文件截回有效前缀并 fsync，**再**用有效长度判断是否滚动。坏记录不再挡住后续事务，也不会因为滚动变成不可恢复的中间段损坏。`replay_after` 另外校验每段末尾 LSN 必须正好衔接下一段名字，空段跳过该检查；有洞或重叠直接报 `CorruptData`。`wal.rs` 增加三条回归测试：torn tail 后重开写入、torn tail 触发滚动后再重开、中间段丢失。

### payload 值校验

payload 页面里的每个值改为 `[u32 CRC32][JSON bytes]`，`ValuePointer.length` 记录含 CRC 的总长度；`get` 先校验 CRC 再解析 JSON，打开时的 pointer 校验也拒绝短于 CRC 头的指针。单 payload 上限相应扣除 4 字节。`payload_store.rs` 增加翻转页面字节后读取失败的回归测试。

### 首个 checkpoint 前的 payload 自愈

`DataFiles::open` 现在接收 generation 的 replay floor。`PayloadStore::open_with_repair` 在 floor 为 0 时把无法校验的派生文件整体替换为空 store，随后由 WAL 回放重建指针和页面；floor 大于 0 时同样损坏仍然报 `CorruptData`。打开校验现在同时检查每个指针的 CRC，零长度页面会恢复成固定大小页面，避免 `ensure_page` 交出没有 mmap 的页。`tracker.journal.bin` 记录 checkpoint 之间的指针变更，checkpoint 后清空；打开时重放高版本完整记录并截断 torn tail。`storage/tests.rs` 增加“发布指针但未推进 generation、页面被毁、重开仍取回 payload”的回归测试，`tracker.rs` 增加 journal torn tail 测试。

### 字段索引集合运算

`PayloadIndexManager::plan_filter` 不再只处理 `must`：`must_not` 用条件掩码补集，`should` 用并集，`min_should` 用逐槽计数；任何组内出现无法索引的子句就放弃该组加速，保证 pre-mask 只会多给候选、不会漏掉匹配。完整过滤仍在候选上执行。`payload_index.rs` 增加四组 mask 语义测试。

### 集合快照 pin

`CollectionStore::snapshot_to` 在 maintenance 锁、提交队列 gate 和 store 读锁下复制 `generation.bin`、发布代全部文件和整个 `wal/`，写入带 CRC32 文件清单的 `snapshot.json`。`verify_snapshot` 校验清单中的长度与 CRC，并拒绝绝对路径和 `..` 穿越。原生 `restore_snapshot` 把验证后的文件复制到新目录，再通过 `CollectionStore::open` 校验恢复结果。`storage/tests.rs` 验证手动复制与 `restore_snapshot` 都能读到 checkpoint 后仅存在于 WAL 的点，位翻转和穿越路径都会被拒绝。

### bitmask 自愈

`MmapBitmask::open_or_rebuild` 以 tracker 的 live 指针为准：损坏的 `bitmask.bin` 直接重建，任何落在 live 指针上的空闲位触发精确重建；孤儿已分配块保留为可泄漏状态但不会别名。新增 stale-free 位图回归测试。

### 段边界清单与有界压实

generation 目录的 `segments.json` 记录段几何，`DataFiles` 持有 `SegmentSet` guard，写入跨段边界时更新，checkpoint 和 staging compaction 同步发布。压实争用回退改为队列 gate 下从只读快照 staging，只有最终提交持写锁，避免写锁下全量重写。

### 冻结段 ANN

HNSW/IVF 只覆盖 `SegmentSet::immutable_end` 之前的完整段；部分段留在 `pending` 中精确扫描。`register_slot`、build 发布和 drain 共用同一边界，pending 达到一个段大小即请求重建。小集合尚未填满首段时仍整体入图。新增部分段搜索回归测试。

### 单写者与发布 poison

生命周期模块提供 `CollectionLock`：Unix 上对集合根目录的 `.lock` 文件做 `LOCK_EX | LOCK_NB`，按规范化路径引用计数；同进程内第二个句柄共享同一把锁，其他进程打开同一目录会被拒绝。最后一个句柄释放时解锁。dense、kind、named 的 create/open 都持有该锁。

三类 store 增加 `generation_poisoned`：generation 清单发布成功即置位，只有完成内存切换、WAL 标记和截断后才清除。中间任何失败都让集合保持可读但拒绝写入/checkpoint/压实，必须重新打开。这避免了磁盘清单与内存句柄分叉后继续写错文件。

### dense snapshot 与 payload 解耦

`dense_snapshot.bin` 不再携带 payload tracker 版本。快照负责 id→slot 目录与逐点版本，payload 字节从当前 `PayloadStore` 读取；generation lsn 之后的 payload 记录由 WAL 回放重新应用。payload-only 写入不再让整个快照失效。`Tracker` 中的变更计数保留为文件内部状态，不再参与快照匹配。

### 清理

删除 `NamedStore::open` 中重复的 `Generation::open` 调用；commit worker 不再在空闲等待时持有 store 的强引用；测试用例在重开同一目录前显式释放旧句柄，promote 成功后立即释放旧 live 句柄。

## 剩余修改方案

本节按改动面从小到大列出细化方案。已落地：首个 checkpoint 前的 payload 自愈、tracker journal 与 bitmask 自愈、`must_not` / `should` / `min_should` 集合运算、集合快照 pin 与原生恢复、段边界清单与有界压实、冻结段 ANN 加追加段精确扫描。仍未落地的是物理文件按段拆分、每段独立图与 WAL mmap 化，它们同时改变文件协议、恢复水位和索引发布边界，必须分阶段验证。

### 派生文件自愈

**已完成部分。** `DataFiles::open` 把 generation 的 replay floor 传给 `PayloadStore::open_with_repair`；floor 为 0 时，任何无法通过指针/CRC 校验的派生状态都直接重建为空 store，后续 WAL 回放负责恢复。零长度页面在打开时恢复固定 extent；`open` 会校验每个持久化 blob 的 CRC。`tracker.journal.bin` 已落地：`put` 只追加带版本和 CRC 的变更记录，checkpoint 先发布紧凑 `tracker.bin` 再清空 journal；打开时只重放版本更高的完整记录，并截断 torn tail。bitmask 改为 `open_or_rebuild`：tracker 是 live 块的权威，损坏的 `bitmask.bin` 或任何落在 live 指针上的空闲位都会触发从指针表精确重建，孤儿已分配块允许保留。

**剩余步骤。**

1. 在 `PayloadStore::open_with_repair` 中把 floor 分成两段：`floor == 0` 时保持现在的整体重建；`floor > 0` 但存在未完成 journal 时，只回滚 journal 覆盖的槽位，再从 floor 之后的 WAL 重放。当前 journal 回放对所有 floor 生效，但不做槽位级回滚。
2. 验证：构造 tracker 半写、页面短文件两类崩溃注入；前者恢复，后者在 floor 大于 0 时保持 `CorruptData`。journal 半写与 stale-free 位图已分别由 `test_journal_replays_and_truncates_torn_tail` 和 `stale_free_bit_is_rebuilt_from_the_live_pointer` 覆盖。

### segment 化与有界压实

**已完成部分。** generation 目录新增 `segments.json`，记录 generation 的 `segment_slots` 与 `next_slot`；`DataFiles` 持有 `SegmentSet` guard，写入推进段边界时立即更新，checkpoint 与 staging compaction 发布对应清单。`compact()` 的争用回退改为先关闭队列 gate、再从只读快照 staging，只有最终提交持写锁，消除了写锁下全量重写的路径。

**剩余步骤。**

1. 把 dense 的物理文件按段拆分：每个 immutable 段拥有独立的 vector/key/tombstone/索引文件，appendable 段保持可增长；`SegmentSet` 从描述边界升级为持有每段文件句柄的 guard。
2. staging compaction 只重写冻结段，appendable 段用有界写缓冲吸收并发写入，提交时把缓冲刷成新段后一次性切换 guard。
3. kind/named 的 blob/postings 切换移出 `gate` 与 `inner`：只在最终发布后短暂持锁替换 `ArcSwap` 句柄，staging 阶段使用不可变快照。
4. 旧 generation 的删除仍以 reader guard 引用计数为准，不能提前 unlink。

**不变量。** 一个查询只使用一个 `SegmentSet` 快照；跨段 id 唯一性由段内 id tracker 加全局 LSN 版本共同保证；WAL floor 仍由 generation 清单决定；任何失败路径在 manifest 发布后都进入 poison。

**验证。** 混合写入下的压实-重启一致性、提交队列积压有界、reader 持有旧段时删除不提前发生、kind/named 压实期间查询不阻塞。

### 按段 ANN

**已完成部分。** ANN 只建在 `SegmentSet::immutable_end` 之前的完整段上；最后一个部分段不进图，写入留在 `pending` 中由搜索精确扫描。`register_slot`、build 发布和 `drain_pending_hnsw` 共用同一边界；pending 达到一个段的大小会置 `needs_rebuild`，drain 也不会把 appendable tail 塞进图。小集合在没有完整段之前仍整体入图，保持原有召回。

**剩余步骤。**

1. 图文件按 immutable 段发布，文件名包含段 id；每个段独立建图，而不是一张覆盖全部冻结段的图。
2. 段冻结时一次性建图并修复删除留下的图空洞；不再依赖跨代 remap 保持邻居关系。
3. 超限重建按段进行：只重建触发阈值的段，不重建整个集合。
4. 查询先按段取候选（图搜索或精确扫描），合并后统一做 payload 过滤、打分和 top-k；每段的 pre-mask 由同一份 `FilterPlan` 语义生成，保持保守。
5. 量化码随段发布，appendable 段继续用原始向量；段切换后旧量化文件引用计数归零才删除。

**不变量。** 任何时刻可见点都在“已发布图”或“段内精确扫描集合”之一中；召回率不得低于当前集合级基线的 1e-4 容差；图重建失败只影响该段。

**验证。** 持续写入下的召回对比、段冻结时图空洞修复、过滤查询与未过滤查询的候选一致性、段级索引文件损坏只隔离该段。

### WAL mmap 化

**目标文件。** `src/storage/wal.rs`。

**步骤。**

1. 段文件改为预分配 mmap，头部包含 magic、format version、随机 CRC seed、first-lsn 和 closed 标记；记录 8 字节对齐，CRC 使用段内 seed。
2. 批量提交改为流式写入 mmap，不再把整批编码进一个 `Vec`；group commit 的“一次可见、一次 fsync”语义保持不变。
3. 后台补足空段队列，append 不再等待创建文件；closed 段按 ACK 水位保留至少一个，回收按整段 unlink。
4. 读取使用段内索引随机访问，而不是顺序 `Read`；随机 CRC seed 让复用文件名时旧记录自动失效。
5. 沿用现有 generation 绑定和连续性校验；只恢复最后一段的 torn tail，其他段损坏仍报 `CorruptData`。

**不变量。** 日志顺序等于应用顺序；未 fsync 的记录不可见；ACK 先于 unlink；generation 切换后旧 ACK 不生效。

**验证。** 重写现有 WAL 测试为新的段格式；保留 torn tail、段丢失、checksum、truncate、group commit 与跨 generation ACK 六类回归。

### 字段索引能力

**已完成部分。** `must` / `must_not` / `should` / `min_should` 的位图集合运算，且无法索引的子句会禁用该组加速。

**剩余步骤。**

1. 把 `FieldIndex` 扩成显式接口：声明、注册、删除、查询、基数估计、直方图、checkpoint；Map/Numeric 先接入。
2. 增加 bool、null、geo、facet 索引；全文索引复用 `postings/` 的 bucket 格式，保持 tokenizer 与查询分词一致。
3. 保存基数和直方图，规划器按选择性选择“先位图过滤”或“先索引探测”，并记录 `explain` 中使用的索引。
4. immutable 段发布只读磁盘索引，appendable 段保留可变索引；段切换时索引文件随段一起发布。
5. nested 索引最后做：先支持 `must` / `must_not` 的嵌套成员匹配，再支持计数约束。

**验证。** 每种新索引的解析/查询单测；混合过滤器与全量后置过滤的结果逐点一致；选择性规划在高基数字段上退化为安全路径。

### 快照 pin

**已完成部分。** `CollectionStore::snapshot_to` 在锁和 gate 下复制 generation 与 WAL，写入带 CRC 清单的 `snapshot.json`；`verify_snapshot` 校验文件完整性。原生 `restore_snapshot(src, dest)` 会先验证清单，再把列出的文件复制到尚不存在的目标目录，并通过 `CollectionStore::open` 完成恢复校验；清单路径禁止绝对地址和 `..` 穿越。

**剩余步骤。**

1. 把快照接口接到 `LocalVectorEngine` 与 C ABI；头文件 regenerated 后同步 binding matrix 和各语言绑定。
2. kind/named 快照加入各自 root marker 与 blob/postings 文件清单，沿用同一 generation 与 WAL pin 协议。
3. 清单记录 ACK generation 与 lsn；恢复时若 ACK generation 与 generation 清单不符，按新集合的 generation 重新绑定，而不是沿用旧 ACK。

**验证。** 写入并发下的快照复制恢复到独立目录；清单位翻转被拒绝；kind/named 快照回放后 postings 与 payload blob 长度一致。

## 验证重点

本轮回归覆盖：

- fsync 失败时 checkpoint 不推进 generation、不截断 WAL；
- WAL torn tail 后继续写入，重开仍能读到新事务；
- torn tail 把物理长度推过滚动阈值后继续写入，重开不再把坏记录当成中间段；
- 删除 WAL 中间段后重开报 `CorruptData`，而不是静默回放剩余事务；
- payload 页面字节被翻转后打开报 checksum 错误；
- 首个 checkpoint 前持久化指针后毁掉页面，重开由 WAL 恢复 payload；
- tracker journal 重放高版本记录并截断 torn tail，checkpoint 后 journal 清空；
- live 指针上的空闲位在打开时 rebuild，已分配块不会被再次交给另一个 payload；
- 段边界清单在写入跨段和 checkpoint 后保持同步；
- 部分段保持在 `pending` 中精确扫描，drain 不把它塞进图；
- `must_not` / `should` / `min_should` 的索引掩码与后置过滤结果一致；
- 快照复制后可在独立目录恢复 checkpoint 后仅存在于 WAL 的点，位翻转和穿越路径被拒绝；
- 同进程多个句柄共享目录锁，最后一个句柄释放后可重开；
- generation 发布失败路径把 store 置为 poisoned，拒绝后续写；
- payload-only 写入后 dense snapshot 仍可用于打开；
- 既有 checkpoint、WAL、压实、量化和恢复测试保持通过。

后续结构性修改需要补充：物理分段文件的压实重启一致性、每段独立图的召回对比、字段索引选择性规划、新 WAL 段格式的 torn tail 与 ACK 回归，以及快照的 C ABI 暴露与 kind/named 快照。目录锁的跨进程互斥仍需在独立进程测试中验证；当前单元测试覆盖同进程共享与释放。
