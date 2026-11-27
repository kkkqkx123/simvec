# 存储改造实施审计与修订方案

本文记录对 `docs/plan/storage-gap-remediation.md` 及其细化方案的实施审计结果：哪些目标已经落地，哪些目标只完成了收缩后的第一步，哪些实现存在正确性或工程问题，以及后续修改顺序。本文是执行清单，不展开完整代码。

## 审计结论

总体方向无需推翻：单一 generation 提交点、commit queue 统一 LSN、payload WAL 按外部 id 定位、WAL 绑定 generation、派生索引可丢弃重放，这些都保持了现有正确性模型。阶段一、阶段二、阶段七、阶段八与细化方案基本一致；阶段三存在明确实现缺口；阶段四、五、六、九的主方案目标高于本轮实际落地范围；另有一处 payload 大小校验的正确性问题和一批遗留代码。

本轮修订的原则是：只修正确性与明确的说明偏差，不借机引入新一代模块，不为开发期旧格式保留兼容分支，不把未实现的物理段拆分或段级 ACK 写成已完成。

## 完整问题清单

### 正确性

- **压缩后的 payload 大小未在写日志前校验。** 准入路径用原始 JSON 长度比较 `max_payload_bytes`，但 LZ4 值实际写入页的是带 codec 和压缩块头的编码字节。`put_at` 在应用阶段按编码长度拒绝时，WAL 已经追加，提交队列会进入失败状态；重开重放也会在同一条记录上失败。多个 `SetPayloadField` 在同一批次内累积增长同样可能越过页面上限，逐条预校验无法覆盖。
- **HNSW 删除修复只遍历被删节点自身的邻接表。** 若邻接因溢出重选而不对称，仍可能存在其他节点指向被删 slot。墓碑过滤保证结果正确，但“邻接表不再指向被删 slot”的强验收口径不成立；完全修复发生在 compaction 重建。
- **WAL rollover 中段写失败可能留下同名段文件。** 段创建、目录同步、段头写入发生在状态登记之前；段头写入失败时目录里可能留下目标段名，后续同名创建会冲突。该路径依赖 `.spare` 覆盖或人工清理，未被作为显式状态处理。
- **快照清单的文件版本字段只用于增量复用。** 恢复校验依赖 CRC 与路径检查，没有单独验证每个文件的 generation 绑定与 `SnapshotInfo.generation` 的一致性。对位翻转威胁模型足够，但对清单被整体改写的情况不是强绑定。

### 完整性

- **WAL replay 没有使用只读 mmap。** 活动段 append 已经 mmap 化并有预分配 `.spare`，但 `replay_after` 与 open 期的前缀扫描仍把整段读进 `Vec`。`read_segment_bytes` 构造了记录偏移索引，却只用于 `debug_assert`，回放本身仍是顺序解码。
- **`.spare` 补建发生在 WAL 状态锁内。** 细化方案要求滚动后在锁外补建；当前首次滚动会在持锁状态下创建并 fsync 预分配文件。
- **append 仍缓存整批编码结果。** 当前是按事务分别编码进 `Vec`，不是一整块大 `Vec`，但与“只流式写映射区、不在内存缓存整批”的表述仍有差距。
- **dense 物理段未拆分。** 文件仍是整代 `vectors.bin`、`keys.bin`、`tombstones.bin` 和 payload 目录；`segments.json` 只发布边界与统计。压实仍重写整个 generation，按段选择只用于调度提示。
- **段级 flush 水位未驱动 ACK。** `segment_applied` 在应用时更新但没有后续消费者；`segment_persisted` 只在 checkpoint 末尾统一推进。checkpoint 仍需要队列门完成水位移与截断，只是把最重的 fsync 移出了门。
- **`points.bin` / `payloads.bin` 仍是派生 blob。** WAL upsert 继续内联完整向量和 JSON，细化方案明确把引用式 WAL 列为后续项；主方案中的“WAL 只记引用”没有实现。
- **named/sparse 仍不是物理段一等公民。** 每个空间有独立 postings、manifest 与词频规划，但向量仍在 blob 文件中并走 exact scan；per-space ANN 未接入。
- **CRC 多项式仍是 CRC32。** 链式结构和覆盖字节与细化方案一致；主方案中的 CRC32C 未替换，细化方案已明确暂缓。
- **新距离类型与 multivector 未实现。** 阶段九把它们列为评估项而非本轮交付，当前状态与细化方案一致。

### 集成与可观测

- **按段压实选择没有参与调度。** `segments_needing_compaction` 只在 compaction 中打印 debug 日志，引擎仍用全局墓碑率判断是否调度。
- **`durable_wal_floor` 没有接入公开状态。** 目前主要是测试辅助；阶段五提到的维护状态暴露没有完成。
- **维护失败只记录日志。** checkpoint、compaction 与索引构建失败会写 tracing 日志，但没有独立的健康状态字段供上层查询。
- **快照接口只在 Rust 侧暴露。** `LocalVectorEngine` 提供两个 dense 便捷方法，C ABI 和语言绑定按计划未改。

### 遗留代码

- payload 解码保留 `[u32 CRC][JSON]` 旧格式分支，并有对应测试与项目说明；这与项目“不做向后兼容、不留旧格式代码”的约定冲突。
- `WalRecord::DropCollection` 只作为 match 占位存在，从未构造。
- `Wal::read_all`、`read_all_acked` 与测试辅助 `read_segment_mmap` 只服务单测，生产路径没有使用。
- WAL 注释仍描述“段内随机访问索引”，与实际顺序扫描不符。
- `payload_store` 模块文档仍写旧的 `[crc][JSON]` 布局；解码函数中存在 `let _ = C::None` 空操作。
- `meta.last_applied_txn` 注释仍写“reserved for the WAL work item”，`storage.rs` 有空的分节注释，`queue.rs` 的 `Ticket` 文档重复一行。
- checkpoint 直接使用 `wal_pins.clamp`，compaction 走 `truncate_wal_through`，两处 pin 钳制逻辑重复。

## 修订方案

### 阶段一：对齐范围与文档

保留现有主方案作为长期目标，但在本文中明确本轮交付边界：物理段拆分、段级 flush、引用式 WAL、named/sparse ANN、CRC32C 与多向量类型属于后续任务。更新项目说明中的 payload 旧格式描述和 WAL 回放描述，避免文档宣称超过实现。

### 阶段二：payload 准入校验

在 payload store 增加按实际 codec 计算编码长度的校验入口。所有写日志前的 payload 准入都改用它，包括 upsert、`SetPayload` 和字段合并。提交队列在分配 LSN 后、WAL append 前，对整批 payload 变更做一次顺序模拟校验，覆盖同一批次内多个字段写入的累积增长；批次校验失败时回滚 LSN 与 reservation 并结算等待者。保留 `put_at` 的最终长度检查作为防御性断言，但正常路径不应再触发。

删除 legacy payload 解码分支、对应测试与文档描述，旧页值统一按 `CorruptData` 处理。

### 阶段三：WAL 回放与滚动清理

把 replay 与前缀扫描改为只读 mmap；零长度文件按空段处理。删除未使用的记录偏移索引及对应注释。将 `.spare` 补建移出 WAL 状态锁，仅在本次 append 确实发生滚动后执行；失败仍为 best-effort。删除未使用的 `DropCollection` 变体和测试专用读取助手，测试改用 `read_all_from`。

### 阶段四：调度与段语义对齐

让引擎的 compaction 调度使用按段选择结果，而不是全局墓碑率；appendable 尾段不被单独触发压实。压实仍重写整个 generation，并在本文中明确其为当前实现的诚实边界。保留 `segments.json` 的逐段统计作为后续物理拆分的基础。

### 阶段五：HNSW 与文档边界

局部 healing 继续只处理被删节点可达的邻接，避免每次删除都全图扫描；结果正确性继续由墓碑过滤保证。把“完全移除所有入边”的验收口径改为“探测结果不含墓碑、完全拓扑修复由 compaction 完成”，并在代码中补充该边界说明。

### 延后项

以下项目不在本轮修改：dense 物理文件拆分、段级 flush worker、引用式 WAL、named/sparse 物理段与 ANN、CRC32C、压缩邻接分层加载、f16/uint/binary/turbo 与 multivector、快照 C ABI。它们互相依赖格式稳定，应在新的格式版本设计内统一处理，而不是在原模块里叠加兼容层。

## 实施结果

本轮已经完成：payload 编码后长度准入与批次合并校验、旧 payload 格式分支清理、WAL replay 与前缀扫描的只读 mmap 化、未使用记录索引清理、`.spare` 锁外补建与滚动失败清理、未使用 WAL 变体与读取助手清理、按段压实调度、快照 generation 绑定校验，以及项目说明中与这些事实不一致的描述。

明确延后且已在本文记录理由：dense 物理文件拆分、段级 flush worker、引用式 WAL、named/sparse ANN、CRC32C、多向量类型、快照 C ABI，以及删除时对全部反向入边的穷尽清理。下一阶段的完整细化方案见 `docs/plan/deferred-storage-refactor.md`。

## 验收方式

- payload：接近页面上限的 LZ4 写入在进入 WAL 前被拒绝；同批次多次字段合并的最终大小被校验；拒绝后重开无残留记录。
- WAL：现有 torn tail、段丢失/重叠、checksum、truncate、group commit、跨 generation ACK、checkpoint 滚动回归覆盖 mmap 回放。
- compaction：只有冻结段达到墓碑阈值时才调度；尾段单独高墓碑不会触发压实。
- 清理：旧 payload 格式读取报 `CorruptData`；未使用的 WAL 变体与测试助手从代码中移除。
- 性能类改动只使用 quick bench 做相对比较，不与完整 bench 或跨机器数据比较。
