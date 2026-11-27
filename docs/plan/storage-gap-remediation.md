# 存储差距修改方案

本文按依赖顺序给出 `docs/issue/storage-comparison-with-qdrant.md` 中问题的修改阶段。每个阶段包含目标、主要改动、不变量和验收方式。设计文档不展开完整代码，只描述模块职责、状态转换和验证边界。

总原则是“先正确性，再并发与容量，再能力扩展”。WAL 和重放语义没有稳定之前，不应开始物理分段或索引扩展；版本与迁移能力属于未来任务，见 `docs/plan/long-term/versioning-and-migration.md`。

## 总体依赖顺序

1. 修正 WAL 段边界与 checkpoint 标记语义。
2. 修正失败提交后的 reservation 和 slot 身份。
3. 完成 WAL mmap 化与预分配。
4. 拆分 dense 物理段并引入段级 id tracker。
5. 建立段级 flush 水位、WAL ACK pin 和并发写入路径。
6. 改造 payload 存储：压缩、blob 外置、槽位身份解耦。
7. 补齐字段索引持久化与 named/sparse 一等公民支持。
8. 改造成增量快照并暴露必要接口。
9. 补齐 ANN 高级能力与文档同步。

前面的阶段可以独立验证；后面的阶段依赖前面的文件格式和并发语义稳定。

## 阶段一：WAL 段边界与 checkpoint 标记

目标：消除 checkpoint 标记复用 LSN 时在滚动边界造成 `CorruptData` 的窗口。

主要改动：

- checkpoint 标记不再使用 generation 的 watermark 作为 `txn_id`，改为从同一个单调 LSN 序列分配新值。generation 水位保持不变，仍表示数据文件包含到 watermark；标记本身只是 WAL 中的 no-op，重放后自然推进下一轮的起始 LSN。
- 保留现有段名必须等于“上一段最后 LSN + 1”的连续性校验；新增测试明确覆盖 checkpoint 标记触发滚动的场景。
- 明确区分三种水位：generation 的数据文件水位、WAL ACK 水位、队列已分配 LSN。三者不得混用。

不变量：

- 任何已发布 generation 的 WAL 重放下界不超过该 generation 的 lsn。
- 已关闭段的最后 LSN 与下一段名严格衔接。
- checkpoint 失败时不得推进 generation 或 ACK。

验收：

- 构造一个填满 active segment 的事务，随后执行 checkpoint 并重开；不得出现连续性错误。
- 在 checkpoint 标记写入后、WAL 截断前注入崩溃，重开后数据和 LSN 均正确。
- 段丢失、段重叠、中间段 torn record 仍按现有契约报 `CorruptData`。

## 阶段二：提交失败后的 reservation 与 slot 身份

目标：让 live 状态和 WAL 重放在任何 WAL 写入失败后仍得到相同的 id→slot 映射。

主要改动：

- reserve 阶段记录每个 ticket 涉及的 id 集合。WAL append 失败时，在 `inner.write()` 中删除这些 id 对应的 reservation；已经分配的 LSN 不回退，LSN 空洞允许存在。
- 同一个 id 在后续成功事务中重新 reservation 时，槽位从当前 `next_slot` 与剩余 reservation 计算；重放只看到成功事务，因此会按相同顺序分配相同槽位。
- 把 payload 类 WAL 记录从内部 slot 定位改为外部 point id 定位。应用时先通过 id tracker 解析槽位，live 与 replay 共用同一解析路径。
- 删除或改写现有“失败提交后可继续写入”的测试，使其同时断言槽位、payload、filter bitmap 和字段索引在重开后一致。

不变量：

- 只有成功写入 WAL 的 transaction 可以改变持久状态。
- reservation 是纯内存状态，失败后必须完全消失。
- WAL 中不出现依赖未提交槽位的记录。

验收：

- 失败事务 → 新事务 → checkpoint → 重开，逐点比较 id→slot、payload 和索引结果。
- 失败事务重试同一 id 后，slot 与重放结果一致。
- 失败发生于 group commit 中间某条事务时，其它成功事务不受影响。

## 阶段三：WAL mmap 化与预分配

目标：降低 append 尾延迟、支持随机访问，并为未来的日志复制保留 closed segment。

主要改动：

- 段文件改为预分配 mmap。段头保存 magic、格式版本、随机 CRC seed、first-lsn 和 closed 标记；记录按 8 字节对齐，CRC 改为 CRC32C 并覆盖段内 seed。
- 后台维护预建空段队列，append 不再在关键路径上创建文件；closed 段按 ACK 水位和保留策略回收，至少保留一个 closed 段。
- append 直接流式写入 mmap，不再先把整批编码成一个大 `Vec`；一次 group commit 对应一次明确的可见性发布和一次 flush。
- 回放使用段内索引按记录随机访问，最后一段的 torn tail 仍可容忍，其它段损坏仍报 `CorruptData`。
- 保留 generation 绑定、ACK generation 校验和段连续性校验。

不变量：

- 未 flush 的记录不可见；一个 batch 要么全部可见，要么全部不可见。
- ACK 必须先于整段 unlink；旧 generation 的 ACK 不生效。
- 段名与段内 first-lsn 的关系在预建、关闭和重开后保持一致。

验收：

- 保留 torn tail、段丢失、段重叠、checksum、truncate、group commit、跨 generation ACK 七类回归。
- 与阶段一、阶段二组合后重跑失败提交和 checkpoint 滚动测试。
- 使用现有 bench 的 quick 档做提交延迟 smoke，不与完整 bench 数值比较。

## 阶段四：dense 物理段拆分与段级 id tracker

目标：把逻辑段升级为物理段，使压实、索引和快照只影响被改动的段。

主要改动：

- 在 generation 目录下为每个 immutable 段建立独立文件集合，包含向量、id→slot 目录、删除位图、payload 引用和字段索引产物；appendable 段保持可增长。
- `SegmentSet` 从边界描述升级为持有各段文件句柄的 guard，查询只使用一个 guard 快照。
- 全局 `IdMap` 拆成段级 id tracker：保存外部 id、段内 offset、point version 和 delete version；全局 LSN 仅作为跨段事务水位，不再承担段内寻址。
- staging 压实只重写冻结段；appendable 段通过有界写缓冲吸收期间写入，提交时把缓冲刷成新段后一次切换 guard。
- 旧 generation 的删除以 reader guard 引用计数为准，不得在仍有读者时 unlink。

不变量：

- 一个查询只使用一个段快照，不混合两代文件。
- 每个 id 在一个生成内只有一个权威段归属。
- generation manifest 仍是唯一提交点；发布后任何失败继续进入 poison。

验收：

- 持续写入下压实并反复重开，比较搜索结果、scroll 结果和逐段计数。
- 只读事务持有旧 guard 时确认旧段文件未提前删除。
- 单个段文件损坏只影响该段，其它段继续服务。

## 阶段五：段级 flush 水位、ACK pin 与并发写入

目标：移除 checkpoint 对全局写入 gate 的长期依赖。

主要改动：

- 每个段维护 version tracker 与 persisted version；flush worker 按 `applied_up_to` 上限刷新段，并取所有段确认版本的最小值推进 WAL ACK。
- 引入 WAL ACK pin：快照、压实和远程复制先注册 pin，完成文件捕获后再释放；ACK 不得超过最低 pin 之前的 LSN。
- checkpoint 从“全局停写”改为“冻结需改写的段 + 等待相关队列 drain”；未涉及的 appendable 段继续接收写入。
- compaction 的 proxy 层吸收期间写入，提交时只短暂持有全局写锁完成 guard 切换。
- 保留 generation lsn 作为回放下界，ACK 仍绑定 generation。

不变量：

- 被 ACK 的每个 LSN 在所有相关段文件中都已 durable。
- pin 存在时 ACK 只前进到 pin 之前。
- 任何段 flush 失败都阻止 ACK，并报告到维护状态而不是静默继续。

验收：

- checkpoint 期间持续写入，确认没有事务丢失、ACK 不越过未 flush 版本。
- 持有 pin 后触发多次 flush，确认 ACK 被正确卡住，释放后恢复。
- 注入段 flush 失败，确认 generation 与 WAL 都不推进。

## 阶段六：payload 压缩、blob 外置与身份解耦

目标：减小 WAL、页面和快照体积，并让 payload 存储与事务日志职责分离。

主要改动：

- payload 页面值增加压缩方式标识，支持无压缩与 LZ4；CRC 覆盖范围包含压缩头和压缩后字节，避免解压前后的歧义。
- WAL upsert 逐步从“内联完整 JSON”过渡为“引用 payload blob + 版本”；在过渡完成前保留内联记录，但读取路径统一先解析为逻辑 payload。
- `points.bin` 与 `payloads.bin` 从纯派生文件升级为段内 payload/vector blob，带长度、校验和和版本；WAL 只记录引用和必要元数据。
- payload 操作统一按外部 id 定位，与阶段二的 WAL 改动合并验收。
- 页面分配器保留现有 CRC 校验和 journal 语义，压缩只改变值编码，不改变指针语义。

不变量：

- blob 未 durable 前，WAL 中必须有足够信息恢复逻辑值。
- 压缩后的值仍通过 CRC 校验后才解析，解码失败不得静默返回旧值。
- checkpoint 顺序仍满足 blob flush → tracker/journal → snapshot → ACK → truncate。

验收：

- 同一 payload 在两种压缩配置下往返一致；位翻转被 CRC 拒绝。
- 大 payload 写入后崩溃重开，重放结果与崩溃前一致。
- WAL 体积和 checkpoint 时间使用 quick bench 做前后对比。

## 阶段七：字段索引持久化与 named/sparse 一等公民

目标：补齐查询能力，并让 named/sparse 复用 segment 级基础设施。

主要改动：

- 把 `FieldIndex` 扩成明确接口：声明、删除、更新、查询、基数估计、checkpoint、发布 immutable 文件。Map/Numeric 先接入，新增 bool/null/geo/facet/full-text/nested。
- 持久化 posting 和直方图，规划器按选择性决定先走位图还是先走索引探测；`must`、`must_not`、`should`、`min_should` 继续沿用现有集合语义。
- full-text 复用并扩展 `postings/`：增加 tokenizer、term 字典、位置、文档长度和评分输入；与查询分词保持同一实现。
- named 向量按向量空间建立段内存储和索引配置；各空间独立记录维度、距离和量化状态。
- sparse 向量改为段内一等存储，保存 indices/values 与 posting 索引，支持 IDF 和 top-k 候选；不再只走 blob + 全量 exact scan。
- 配置校验明确列出支持组合；支持后逐步放开当前 ANN 仅限默认 dense 的限制。

不变量：

- pre-mask 只允许多给候选，不得漏掉任何满足过滤条件的点。
- 每个索引文件都绑定到段、generation 和版本；损坏时只丢弃该段加速能力。
- named/sparse 的 WAL 记录与 dense 共用同一 id/version 语义。

验收：

- 每种索引的解析、持久化和重开测试；混合过滤器与全量后置过滤结果逐点一致。
- 高基数字段上的规划退化为安全路径。
- named/sparse 查询与 exact baseline 在 quick bench 数据上结果一致。

## 阶段八：增量快照与接口暴露

目标：把全量复制改为按文件版本复制，并保护复制窗口内的 WAL。

主要改动：

- 每个段发布 manifest，记录段版本、文件相对路径、文件版本、长度和 CRC；集合快照清单聚合所有段 manifest 与 generation/WAL 水位。
- 复制前注册 WAL ACK pin；只复制相对上一次快照发生版本变化的文件；完成后释放 pin。
- 恢复时先校验清单位翻转、路径穿越和文件版本，再按目标目录的新 generation 重建 WAL 绑定。
- 快照接口接入公开 Rust API。若接入 C ABI，必须重新生成 `include/simvec.h`，并同步 `docs/binding-matrix.md` 与相关语言绑定。

不变量：

- 快照内部的所有文件属于同一个 generation 视图。
- WAL 在 pin 期间不得被 ACK 越过快照需要的水位。
- 恢复后的目录必须通过正常 `CollectionStore::open` 校验。

验收：

- 写入并发下做快照并恢复到独立目录，逐点比较搜索和 scroll。
- 位翻转、路径穿越、缺文件和版本不匹配都被拒绝。
- 第二次快照只复制变化段文件。

## 阶段九：ANN 高级能力与文档同步

目标：在物理段和索引文件稳定后，补齐 qdrant 已有的检索能力。

主要改动：

- 增加 HNSW graph healing，只修补受删除影响的邻接，而不是整段重建。
- 按 payload 分布建立 payload block 索引，减少过滤查询的无效图遍历。
- 邻接表支持压缩存储和分层读取；量化码与段图一起发布。
- 评估并逐步引入 f16/uint/binary/turbo 向量类型和 multivector；所有新 kernel 仍以 `naive` 为正确性基线，误差不超过 1e-4。
- 同步 `AGENTS.md` 中锁键、文件布局和能力边界的描述，避免文档落后于实现。

不变量：

- 候选覆盖不低于当前基线；任何派生索引损坏只降级该段，不影响正确性。
- 公开分数仍然是高分优先，内部距离仍然是低分优先。

验收：

- 持续写入与删除下的召回对比；段图损坏隔离测试。
- 过滤查询与未过滤查询的候选一致性。
- 所有新距离 kernel 与 `naive` 的一致性测试。

## 跨阶段验证重点

- 每个改变文件格式或 WAL 语义的阶段都必须包含崩溃注入：generation 发布前、发布后但未切换句柄、ACK 前、ACK 后但未截断、截断中。
- 所有恢复测试都要检查 id→slot、payload、过滤位图、字段索引和搜索结果，而不只是点数量。
- 性能测试统一使用 quick 档做相对比较；不同机器和完整 bench 数值不可比。
- 版本与迁移相关改动不在本计划内实施，按 `docs/plan/long-term/versioning-and-migration.md` 在未来启用。
