# 延后存储重构统一实施方案

本文把当前明确延后的存储能力整理成一次可执行的重构方案。它只描述模块职责、数据布局、状态转换、不变量和验收边界，不展开完整代码。本文是临时执行清单，不作为项目事实源；落地后应把仍然成立的事实写回 `AGENTS.md`。

## 范围与原则

本轮重构覆盖以下能力：统一 CRC32C、HNSW 反向邻接与完整删除修复、dense 物理段与段级 id tracker、段级 flush 与 ACK 推进、引用式 WAL 与 blob 持久化、named/sparse 段级存储与 ANN、向量类型扩展与 multivector、按段快照清单与 C ABI 暴露。

统一原则如下：

- 不新增“替代版”模块，也不保留旧布局读取路径。格式变更一次性生效，不能识别的目录按 `CorruptData` 拒绝，由重建或重新导入恢复。
- generation manifest 仍是唯一提交点；manifest 发布后的任何失败继续进入 poison，等待重开。
- WAL 仍是事务顺序权威；blob、payload 页、字段索引、ANN 图都是派生数据，但引用式 WAL 会让 blob 在事务提交前成为恢复依据。
- 外部 id 是身份，slot 只是段内物理位置；日志与快照不依赖未提交的绝对 slot。
- 所有新距离 kernel 必须与 `naive` 在 1e-4 内一致；公开分数保持高分优先，内部距离保持低分优先。
- 性能比较只使用 quick 档，不与完整 bench 或跨机器数据比较。

## 当前基线

当前实现已经具备：v2 分段 WAL、mmap 顺序回放、group commit、失败批 LSN 回滚、WAL ACK pin、按外部 id 定位的 payload 记录、CRC 保护的 payload codec、可持久化字段索引、posting 词频、硬链接增量快照、删除局部 healing、Rust 侧快照 API。当前仍缺少物理段文件、段级 flush ACK、引用式 WAL、named/sparse ANN、新向量类型和快照 C ABI。

## 依赖顺序

| 顺序 | 工作 | 依赖 | 产物 |
| --- | --- | --- | --- |
| 一 | CRC32C 与 HNSW 反向邻接 | 无 | 统一校验原语；删除可完整清理所有入边 |
| 二 | dense 物理段与段级 tracker | 一 | 段目录、段内 id tracker、段级文件 guard |
| 三 | 段级 flush 与 ACK | 二 | 段水位、最小 ACK、维护状态 |
| 四 | 引用式 WAL 与 blob 持久化 | 二、三 | WAL 引用记录、段内 blob、可恢复版本链 |
| 五 | named/sparse 段级存储与 ANN | 二、四 | 每空间段目录、posting 规划、per-space ANN |
| 六 | 向量类型与 multivector | 二、四、五 | 类型标签、新 kernel、multi 存储与检索 |
| 七 | 快照清单与 C ABI | 二至六 | 段级文件版本清单、C 接口与绑定同步 |

顺序不能倒置：先在整代文件上实现段级 ACK，会遇到无法单独 fsync 的共享文件；先做引用式 WAL，会缺少稳定的 blob 所属段；先做 named ANN，会复制一套 dense 段机制。

## 统一校验替换为 CRC32C

目标是把 WAL 段、ACK 文件、generation 清单、快照信封、payload tracker/journal、bitmask、payload 页值、字段索引和 postings manifest 统一到 CRC32C，避免同一代数据混用两种多项式。

模块职责：在现有校验封装处替换算法，不新建通用校验模块；如果已有 `persist_crc` bench，只更新其调用与基准标签。`wal.rs` 保留链式 CRC 的覆盖范围：长度前缀、postcard 字节和零填充；seed 仍作为链首。`snapshot.rs` 保留 10 字节信封形状，但更换 magic，使旧文件直接失效。

文件格式：所有信封 magic 改为独立的 CRC32C 标记；格式版本仍保持开发阶段的 1。没有兼容读取，也没有“按旧 magic 自动重算”的回退。payload 页值仍为 `[crc32c][codec][data]`，CRC 覆盖 codec 与编码字节。字段索引、dense snapshot、kind/named snapshot、segment manifest、postings manifest 全部使用同一信封函数。

状态转换：打开时读到旧 magic 或 CRC 不匹配即为 `CorruptData`；派生文件可由 payload 与 WAL 重建时按现有丢弃规则处理，提交文件不重建。checkpoint 发布顺序不变，只是所有写入都使用新 CRC。

验收：WAL 七类回归、payload 位翻转、字段索引位翻转、快照位翻转、tracker/journal 位翻转全部保留；新增一项“旧 CRC32 文件被拒绝”的测试。

## HNSW 反向邻接与完整删除修复

目标是让删除后所有指向被删 slot 的入边都被移除，而不只处理被删节点自身邻接表 reachable 的邻居。

模块职责：在 `src/index/hnsw.rs` 的 `Node` 中增加每层的反向邻接集合；`build.rs`、`heal.rs`、`persist.rs` 共同维护它。持久化文件仍只写正向邻接，反向关系在加载图后由正向邻接一次扫描派生，不增加磁盘体积。

状态转换：

- 插入或溢出重选时，正向边的增加和删除必须同步更新对端反向集合；同一邻接变更完成后再递增版本计数器。
- 锁顺序固定为“正向节点锁 → 反向节点锁”，搜索只读正向邻接，不持有反向锁。
- 删除时先从反向集合取出所有指向被删 slot 的节点，再按层移除这些正向边，并用现有 diversity heuristic 重选邻居；重选结果反向更新。
- 若反向集合与正向邻接在加载后发现不一致，以正向邻接为准重建反向集合，并记录降级指标，不改变搜索结果。

不变量：搜索仍使用双读版本校验；墓碑过滤继续保留；healing 只改内存图，不改写已发布 `hnsw_seg_*.bin`；compaction 仍是最终拓扑重建点。修复后允许 `repair_hnsw` 不再因为单个墓碑就重建整段，仅在版本漂移超过阈值时重建。

验收：构造不对称邻接的单元测试，删除后断言所有层和所有节点的正向邻接都不含被删 slot；持续写入删除下探测结果不含墓碑；段图损坏仍只降级该段。

## dense 物理段与段级 id tracker

目标是把整代共享文件拆成可独立刷新、复制、损坏隔离和压实的段文件集合。

### 布局

generation 目录保留 `meta.bin`、`segments.json`、`generation.bin`，不再有整代 `vectors.bin`、`keys.bin`、`tombstones.bin`、`dense_snapshot.bin`。新增：

- `segments/<start>/vectors.bin`：该槽位区间内的向量行，按段内 offset 寻址。
- `segments/<start>/keys.bin`：段内 offset 到外部 id 的目录。
- `segments/<start>/tombstones.bin`：该段的删除位图，可随删除更新。
- `segments/<start>/id_tracker.bin`：外部 id 到段内 offset、point version、delete version、free 标记和 blob 引用的 CRC 信封。
- `segments/<start>/payloads_store/`：现有 payload 页、tracker、journal、bitmask，作用域缩到单段。
- `segments/<start>/field_indexes.bin`、`hnsw_seg.bin`、`ivf_seg.bin`、`quant_seg.bin`：该段派生索引。
- `segments/<start>/blobs.bin`：引用式 WAL 阶段启用的向量与 payload blob，初始阶段可为空。

`segments.json` 记录布局标记、每段起止槽位、容量、live/tombstone 数、applied/persisted lsn、文件版本/长度/CRC 和 generation。它与 `meta.bin` 不一致时按后者重建；损坏的派生段文件只禁用该段加速。

### 模块职责

`segments.rs` 从边界描述升级为段文件集合的发布点：`SegmentSet` 持有每段的 `SegmentFiles` guard，查询一次加载整个 guard。`vectors.rs`、`keys.rs`、`tombstones.rs` 改为传入段文件句柄，不再假定整代单文件。`payload_store.rs` 保持页式实现，路径改为段目录。`compaction.rs` 按段 staging。`store_open.rs` 按段打开并恢复 tracker。`search.rs` 与 `index_lifecycle.rs` 的所有搜索/构建路径只使用同一个段 guard。

### 段级 tracker 与槽位

全局 slot 仍可表示为 `segment.start + local_offset`，但不由单一 `next_slot` 分配。段 tracker 保存外部 id、local offset、point version、delete version 和 free 标记；free offset 以排序列表持久化，分配时取最小可用 offset，保证 live 与 replay 顺序一致。

新 id 默认进入 appendable 尾段；已删除 id 的重新 upsert 按其 delete version 在原段内复用 free offset。WAL upsert 不携带绝对 slot，重放只按 tracker 和日志顺序解析。删除保留 delete version，用于拒绝旧 WAL 记录复活已删除点。

### 写入与压实

写入按 slot 定位所属段：immutable 段的向量、键、索引不改写，但 tombstone、payload 和 blob 可以更新；appendable 段可增长。每段维护 dirty lsn 区间，供 flush 使用。

压实只选垃圾率超过阈值的冻结段。stage 目录内干净段以“先 flush 再按字节复制”的方式进入新 generation，不重排其 slot；选中段重排 live 点为最小 local offsets，重建 tracker、keys、payload、blob、field index 和 ANN。段 start 与 capacity 保持不变，后续段 slot 范围不移动；未使用的尾部 offset 进入 free list。staging 完成后一次发布 generation manifest，持有旧 guard 的读者退出后删除旧目录。

### 不变量

一个查询只使用一个段 guard；一个 id 在一个 generation 内只有一个权威段归属；generation manifest 仍是唯一提交点；旧段文件在读者 guard 归零前不得删除；单个派生段文件损坏只影响该段。

### 验收

持续写入下反复压实和重开，逐点比较 id、slot、payload、filter 和搜索结果；持有旧 guard 时确认旧段文件未提前删除；破坏单段 ANN/量化文件后其它段继续走 ANN，坏段回落 exact。

## 段级 flush 与 WAL ACK

目标是把 checkpoint 从“全局停写 fsync”改为“按段确认水位，短门发布提交点”。

模块职责：`StoreInner` 的 `segment_applied` / `segment_persisted` 升级为每段状态；`store_maintenance.rs` 实现 flush worker；`wal_ack.rs` 继续提供 pin；`types/info.rs` 暴露维护状态。

段 flush 过程：捕获当前 applied 前沿，按段并行刷新该段内 dirty 的向量、键、tombstone、payload、tracker、journal 和派生索引；刷新期间写入仍可进入其它段。对同一段的写入用短 gate 交换 dirty 集合，fsync 在 gate 外完成。成功后把该段 `persisted_lsn` 推进到前沿；任何失败保持水位不动并记录维护错误。

ACK 推进：取所有段 `persisted_lsn` 的最小值，再用 generation lsn 和最低 WAL pin 钳制。ACK 只能覆盖所有相关段都已 durable 的 lsn。截断仍只删除整段覆盖的 WAL 文件。checkpoint 的 manifest 发布与 ACK 发布仍需队列门，但门内只剩计数发布和截断，不再包含大文件 fsync。

恢复：open 从 `segments.json` 读取每段 persisted lsn；字段缺失或不一致时回落到 generation lsn 并扫描重建。段 flush 失败时 generation 与 ACK 都不推进，写入不进入 poison，但维护状态转为 unhealthy。

验收：checkpoint 期间持续写入无丢失；持 pin 后多次 flush 停在 pin 前，释放后恢复；注入单段 flush 失败，generation 与 WAL 均不推进；维护状态可见失败原因。

## 引用式 WAL 与 blob 持久化

目标是把 WAL 从“内联完整向量和 JSON”改为“引用已 fsync 的 blob + 版本”，同时让 blob 成为可验证的恢复依据。

### 记录与 blob

`wal.rs` 的 upsert 变体改为按段和空间引用 blob：`{segment_start, offset, length, crc32c, version, space}`。payload 写入改为 payload blob 引用，仍按外部 id 定位。旧的完整内联 upsert 变体删除，不做双写或兼容读取。

每段 `blobs.bin` 使用 8 字节对齐记录：`[crc32c][length u32][version u64][kind u8][payload bytes]`。kind 区分 dense vector、sparse vector、multi vector 和 JSON payload。blob 引用包含段起始和偏移，避免 compaction 后旧引用误指新内容；CRC 覆盖头部与数据，解压或解码失败即为 `CorruptData`。

### 写入顺序

先追加 blob 并 `flush_range` + fsync，再追加 WAL 引用并 group commit，最后应用到内存段文件。WAL append 失败会留下未引用 blob，视为可回收垃圾；blob fsync 失败不得写 WAL。payload 页仍服务于 live 读取，blob 服务于重放；两者都由同一逻辑 payload 编码产生。

compact 重写段时只复制 live blob，重建引用和段 tracker；被删除或被覆盖的旧 blob 随段重写回收。不能原地删除 blob，以免破坏仍在 WAL 中的引用。

### checkpoint 顺序

段 blob sync → 段 tracker/journal → 段向量/键/tombstone → 字段索引 → dense snapshot/段 manifest → generation 发布 → ACK → WAL 截断。每一步传播 I/O 错误；ACK 不得超过 blob 已 durable 的水位。

### 恢复

replay 从 generation lsn 之后读取 WAL 引用，按段找到 blob，校验 CRC、版本和空间类型，再应用逻辑点。版本低于 tracker 中 point version 或 delete version 的记录跳过；引用缺失、CRC 失败或类型不匹配为 `CorruptData`，不得静默回退到旧值。

验收：blob fsync 前崩溃、blob 后 WAL 前崩溃、WAL 后 apply 前崩溃、ACK 后截断中崩溃四类注入；每类都逐点比较 id、slot、payload、filter 与搜索结果。WAL 体积与 checkpoint 时间使用 quick bench 做相对比较。

## named/sparse 段级存储与 ANN

目标是把 kind/named 从“根 blob + 全量 exact”升级为复用段基础设施的一等空间。

布局：generation 内新增 `spaces/<space>/segments/<start>/`，每个段拥有 vectors/blob、id tracker、postings、payload 引用和可选 ANN/量化文件；`spaces.json` 记录每空间 kind、distance、dim、multi config、量化配置和 generation/lsn。根 `kind.bin`、`named.bin` 只保留类型标记。

WAL：`UpsertVectors`、`PatchNamedVectors` 改为按空间引用 blob，使用与 dense 相同的 point version、delete version 和外部 id 语义。commit 路径按空间路由到段 tracker，live 与 replay 共用解析函数。

named dense：每个声明该空间且支持 ANN 的段构建独立 HNSW 或 IVF 文件，量化码按段发布；查询对每个目标空间独立取候选，再沿用现有 fuse 与分数转换。未声明、损坏或未训练的段回落到该空间 exact scan。配置校验明确列出组合：dense 支持 HNSW/IVF/量化，sparse 只用 posting + exact，multi 只用 MaxSim。

sparse：段内保存 indices/values 的紧凑 CSR，posting 文件保存 term 到 `(local_offset, weight)`，manifest 保存 term 文档频率。规划器按 IDF 与最高词频选择 posting 累积或全扫描；两种路径共享同一过滤器、距离函数和分数转换。文档频率是派生统计，损坏时重建，不改变候选资格。

验收：named/sparse 与 exact baseline 在 quick 数据上逐点一致；混合过滤结果与全量后置过滤一致；每空间独立 checkpoint 与重开；posting 损坏只失去加速。

## 向量类型与 multivector

目标是在段 blob 和 WAL 引用中支持 f16、uint8、binary、turbo 与多向量，并保持 `naive` 为唯一正确性基线。

类型标签进入段向量文件头、blob 头和 WAL 引用；读取时先按标签解码，再进入现有 kernel 分发。`f32` 保持默认。f16 用半精度转换；uint8 带每段 scale/zero-point；binary 保存打包位；turbo 仅在距离语义可明确映射时开放，否则配置拒绝。multivector 复用 dense blob 的连续子向量布局，段 tracker 记录子向量数量和 MaxSim 比较规则。

距离模块职责：`distance.rs` 增加类型分发；`naive.rs` 负责所有类型的基准实现；AVX2/AVX-512/NEON 按类型新增 kernel，不满足 1e-4 一致性时关闭该 kernel 并回落 naive。索引构建读取解码后的向量，不直接消费压缩字节。

配置校验：按空间声明允许的类型与距离组合；不支持的组合在创建集合时拒绝，而不是运行时降级。公开分数方向不变。

验收：每种类型与每个 metric 的 naive 对照；SIMD 与 naive 1e-4 一致性；multivector 的 MaxSim 与 exact baseline 一致；类型标签损坏报 `CorruptData`。

## 快照清单与 C ABI

目标是把快照清单对齐物理段，并把快照能力暴露给 C ABI 与绑定。

快照：`SnapshotInfo` 聚合 generation、lsn、每段 manifest 的文件版本/长度/CRC 和 WAL 水位；增量快照比较四元组相同的条目并硬链接，跨文件系统退化为复制。恢复仍复制成独立目录并通过正常 `CollectionStore::open` 验证。段级文件版本来自 `segments.json` 和各段 manifest，不再用 generation id 统一代替 WAL 文件版本。WAL 段条目记录其绑定 generation 与首 lsn。

C ABI：在现有 `src/c_api` 模块中增加快照创建、增量快照、校验和恢复函数；不新建 C API 替代模块。路径参数按 UTF-8 校验，错误经现有 error 通道返回；快照信息以 JSON 字符串返回并用现有字符串释放函数释放。重新生成 `include/simvec.h`，同步 `docs/binding-matrix.md` 和 python/js/go/java/r 绑定；wasm 不暴露文件快照。

验收：位翻转、路径穿越、缺文件、版本不匹配、目标已存在全部拒绝；第二次快照只硬链接未变段；C ABI 快照可从 Rust 创建的集合恢复并搜索。

## 崩溃点与恢复矩阵

每次涉及文件格式或 WAL 语义的改动都要覆盖以下窗口：

- generation 发布前：旧 generation 继续服务，staging 被丢弃。
- 发布后但未切换句柄：poison，重开加载新 manifest。
- ACK 前：WAL 完整，重放补齐所有未确认事务。
- ACK 后但未 truncate：重放跳过 ACK 前缀，段连续性仍校验。
- truncate 中：ACK 已 durable，存活段必须仍是连续后缀。
- blob fsync 前：WAL 不得引用该 blob。
- blob fsync 后、WAL 前：孤立 blob 可忽略。
- WAL 后、apply 前：重放按 id/version 重建，槽位与 live 一致。

所有恢复测试都要比较 id→slot、point/delete version、payload、filter bitmap、字段索引、搜索和 scroll 结果，不能只比较点数。

## 明确不做

本轮不引入跨进程多写者、集群复制、WAL delta 远端传输、对象存储、分片路由、格式版本迁移框架，也不把开发期旧文件重新变成可读。所有“兼容旧格式”的需求都应转化为重建、重新导入或显式迁移任务，而不是在读取路径叠加分支。
