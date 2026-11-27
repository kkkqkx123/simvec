# 阶段二细化方案：提交失败后的 reservation 与 slot 身份

对照 `docs/plan/storage-gap-remediation.md` 阶段二与问题清单中“WAL append 失败后
的 reservation 残留可能造成槽位漂移”一节。只描述职责、状态转换与验证边界。

## 对照 qdrant

qdrant 的 segment 内 `IdTracker` 按 external id 维护 `internal_id`、point version
与 delete version；payload 与向量写入都以 external id + version 生效，version
旧的写入被丢弃。slot（internal id）只是 segment 内的物理位置，日志与快照中
永不直接引用；live 与重放走同一套“external → internal”解析，因此物理位置
即使重排，语义仍一致。

simvec 现状有两处偏离：其一，`reserve_slots_locked` 写入 `inner.reserved` 后，
若 WAL batch 失败，`queue::run_batch` 只向等待者返回错误，不清除 reservation，
`next_lsn` 却已前进——live 保留了失败事务的槽位，重放看不到它们，后续成功
事务的槽位分配从此分叉；其二，`SetPayload` / `DeletePayloadKeys` /
`SetPayloadField` 携带内部 `slot`，一旦 live 与重放的槽位编号不一致，payload
会被应用到错误的槽位。现有“失败提交后可继续写入”测试只检查点可见性，
未比较槽位与 payload 的重放一致性。

## 修改内容

1. LSN 改由 worker 在 flush 时统一分配，失败回滚，不留空洞。reserve 阶段只做
   槽位预留（存在性集合，不分配具体槽位号）与入队；worker 取出整批 backlog 后，
   在一次 `inner.write()` 下按队列顺序连续分配 LSN，随后做单次 WAL append。
   append 失败时整批回滚：回退 `next_lsn` 到本批起始值（worker 是唯一的批量
   分配者，失败批处理期间 checkpoint/压实 marker 不可能并发分配，见阶段一
   门控，故回退安全），并删除本批 upsert id 对应的 `reserved` 条目，再结算错误。
   之所以不采用“LSN 不回退、空洞允许存在”：空洞会让段名连续性校验
   （`next == prev_last + 1`）在跨越空洞的滚动边界上误报 `CorruptData`，而放宽
   校验又会使真正的段丢失无法检出；单分配者 + 原子整批回滚从根本上消灭空洞，
   是更强的保证。清理只删除仍存在的条目；被后继事务复用期望的 id 在应用时
   按“reserved → reverse → 新槽”解析，与重放一致（第 3 点）。
2. payload 记录改按外部 point id 定位。`WalRecord::SetPayload` /
   `DeletePayloadKeys` / `SetPayloadField` 的 `slot: u32` 改为
   `point_id: String`。构造侧（`store_payload` 的三个公开方法、kind/named 的
   `commit_for_point`）在门内按 id 构造记录，不再先解析 slot；应用侧
   （dense `apply_records_locked`、kind/named `commit_records_locked` 与各自
   lifecycle 重放）统一先经 id tracker（dense 为 `reverse`，kind/named 为各自
   `reverse`）解析槽位：id 缺失或已墓碑则跳过（no-op），同 batch 内先 upsert
   后改 payload 的顺序应用自然可见。
3. live 与重放共用解析函数。dense 槽位号一律在应用时按日志顺序分配
   （`resolve_upsert_slot_locked`：`reserved` 命中则消费并取 `next_slot` 递增，
   `reverse` 命中则复用，否则按新槽分配——覆盖重放与清理后的 skippers）；
   reserve 阶段与重放都不再产生绝对槽位号，从根本上消除编号复用与碰撞。
   payload 目标槽同样按 id 实时解析：dense 新增按 id 解析的小函数，`apply_txn`
   的预校验（`SetPayloadField` 合并后大小检查）与 `apply_records_locked` 的应用
   都调用它；kind/named 同理。WAL 中不再出现依赖未提交槽位的记录。
4. `validate_admission_locked` 语义不变（仍按 `reverse + reserved` 去重计数），
   清理后 `reserved` 为空，后续准入计数与重放一致。

## 状态转换

- reserve（`inner.write`，分配 LSN + 槽位 + 入队）→ flush（worker 单次 fsync）
  → 成功则按 LSN 顺序应用并移除 reservation、发布 `reverse`；失败则整批回滚、
  清除本批 reservation、结算错误但不清毒队列（后继 batch 仍可提交）。
- 应用失败（非 WAL 原因）仍按现有 poison 语义停写：live 可能已与重放分叉，
  必须重开而非继续猜测。

## 不变量

- 只有成功写入 WAL 的事务改变持久状态；reservation 是纯内存状态，失败后
  必须完全消失。
- 同一 LSN 顺序下，live 分配的槽位序列与重放重建的槽位序列相同。
- payload 操作的效果只取决于 id 与日志顺序，不取决于槽位编号。

## 验收

- 失败事务 → 新事务 → checkpoint → 重开，逐点比较 id→slot、payload、
  filter bitmap 与字段索引：live 与重开后完全一致。
- 失败事务重试同一 id：新槽位与重放结果一致。
- group commit 中间 batch 失败：同 batch 等待者均收到错误，后继 batch 仍可
  成功提交且不受污染。
- 更新“失败提交后可继续写入”测试，使其同时断言槽位、payload 与索引的重开
  一致性。
