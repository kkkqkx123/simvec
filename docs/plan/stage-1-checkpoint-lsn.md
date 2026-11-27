# 阶段一细化方案：WAL 段边界与 checkpoint 标记语义

对照 `docs/plan/storage-gap-remediation.md` 阶段一与 `docs/issue/storage-comparison-with-qdrant.md`
中“checkpoint 标记复用 LSN 会破坏 WAL 段连续性”一节。本方案只描述模块职责、
状态转换与验证边界，不展开完整代码。

## 对照 qdrant

qdrant 的 WAL entry index 是单调递增且永不复用的：每个 segment 有 `first-index`，
新 segment 命名/续接都从“上一段最后 index + 1”推导，entry 之间只有“空洞”（未提交
的序号），没有“复用”。`Wal::truncate_through` / ACK 只推进水位，不产生新 entry；
凡是写入日志的新记录（含各类 no-op 标记）都分配新的递增序号。段连续性校验
（closed 段 `[start, start+len)` 与下一段 `start` 严格衔接、重叠/缺失即错）之所以
成立，正是因为序号永不复用。

simvec 现状相反：`CollectionStore::checkpoint_inner`（`src/storage/store_maintenance.rs`）、
`commit_generation`（`src/storage/compaction.rs`）以及 `set_quantization_config` 的
`Quantize` 标记都复用“已持久水位”（`meta.last_applied_txn`）作为 marker 的 `txn_id`。
当 marker 恰好触发段滚动时，`next_segment_name` 会算出与水位相同的段名：
旧 active 段已包含该水位，新段又以该水位命名；若在 `truncate_through` 之前崩溃，
重开时连续性校验把“上一段尾 LSN + 1 != 下一段名”判为 `CorruptData`。窗口虽窄，
但正好落在维护路径中段。

## 三种水位的定义（本阶段起强制区分）

- 数据文件水位：`Generation::lsn`，表示 serving generation 的文件已包含到该 LSN。
  仅在 checkpoint / compaction 发布时推进。
- WAL ACK 水位：`wal/acked-lsn`，表示该 LSN 及之前可被截断。仅由
  `truncate_through` / `truncate` 在删除覆盖段之前发布，且绑定到发布它的 generation。
- 分配水位：`StoreInner::next_lsn`，下一个可分配的 LSN。只增不减，失败事务留下
  的空洞允许存在（见阶段二），永不回退、不复用。

不变量：`ACK <= generation.lsn < next_lsn`（无事务时取等号左侧；marker 推进后
`generation.lsn < next_lsn` 严格成立）。任何已发布 generation 的 WAL 重放下界
不超过该 generation 的 lsn；已关闭段的最后 LSN 与下一段名严格衔接。

## 修改内容

1. marker 分配新的 LSN。`checkpoint_inner` 在持有 store 写锁、队列门已关闭的
   阶段二临界区内，取 `marker_lsn = inner.next_lsn` 并 `next_lsn += 1`，再以该值
   写入 `WalRecord::Compact`。`generation.set_lsn(watermark)` 保持为原 watermark，
   不变；`truncate_through(watermark)` 也保持以 watermark 截断，因此 marker 所在
   段必然存活（其首 LSN 为 `marker_lsn > watermark`）。重放时 marker 按普通事务
   被跳过下界（`txn_id > floor` 成立），执行为 no-op，并自然把 `last_applied` /
   `next_lsn` 推到 marker 之后。
2. `set_quantization_config` 的 `Quantize` 改走提交队列（按队列顺序分配全新
   LSN），不再复用 `last_applied_txn`（旧写法存在与并发写入重复分配的风险）；
   best-effort 语义不变，失败不影响已持久的 `meta.bin`。`commit_generation`
   （压实）则不写 marker：提交时队列门已排空，不存在新于 watermark 的事务，
   `truncate_through(watermark)` 自然清空日志；存活的 generation lsn 与新
   发布的 ACK（必然不超过 generation lsn）共同固定重放下界，下一个分配的
   LSN 不可能碰撞。
3. 保留段名连续性校验与“末段 torn tail 可容忍、中间段损坏即 `CorruptData`”
   的全部语义；marker 存活后，连续性对 marker 段同样生效，不再有复用名。
4. `checkpoint` 失败路径不变：任何 fsync 失败都中止于 manifest 发布之前，
   generation 与 ACK 均不推进，WAL 保持完整。

## 崩溃窗口

- manifest 发布前崩溃：旧 generation 服务，marker 段（若已写）因 ACK 未推进而
  被重放为 no-op，无影响。
- manifest 已发布、marker 未写即崩溃：新 generation 的 lsn 为 watermark，
  marker 缺失；`next_lsn` 在重放后从 `max(last_seen, floor) + 1` 恢复，
  不与已截断前缀冲突。
- marker 已写、truncate 前崩溃：重放看到 marker（no-op）与完整旧段，连续性
  校验通过（段名均为全新 LSN），随后可再次 checkpoint 截断。
- truncate 中崩溃：ACK 已先行发布，存活段重放跳过已覆盖前缀；marker 段首 LSN
  大于 ACK，保留。

## 验收

- 填满 active 段的事务后执行 checkpoint 并重开：不得出现连续性错误，且
  `segment_names` 严格递增、marker 段首 LSN 大于 watermark。
- 在 marker 写入后、`truncate_through` 前注入崩溃：重开后数据与 LSN 正确，
  再次 checkpoint 可完成截断。
- 原有七类 WAL 回归（torn tail、段丢失/重叠、checksum、truncate、group commit、
  跨 generation ACK）全部保留并通过。
