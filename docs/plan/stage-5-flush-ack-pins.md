# 阶段五细化方案：段级 flush 水位、WAL ACK pin 与并发写入

对照 `docs/plan/storage-gap-remediation.md` 阶段五与问题清单中“checkpoint 和
压实期间全局停写”一节。只描述职责、状态转换与验证边界。

## 对照 qdrant

qdrant 的 flush worker 与 update worker 并行：按 `applied_up_to` 上限刷新各
segment，取所有 segment 确认版本的最小值推进 WAL ACK，并用 `WalAckPins` 保护
快照与远端复制所需的旧日志。checkpoint 不需要全局停写，未涉及的 segment 继续
接收写入；compaction 的 proxy 层吸收期间写入，提交时只短暂持有写锁切换。

simvec 现状是 checkpoint 与压实经 `CommitQueue::quiesce` 关闭写入门，等待队列
排空后刷新并截断。实现简单但集合越大停顿越明显，且缺少按段 durable 版本与
ACK pin，无法做到同等粒度的并发。

## 目标

移除 checkpoint 对全局写入门的长期依赖：刷新按段确认版本推进 ACK，pin 保护
复制窗口内的旧日志，未涉及的 appendable 段在刷新期间继续接收写入。compaction
沿用现有代理思路，提交时只短暂持有写锁切换守卫。

## 职责划分

- 段版本：每段维护已应用 LSN 与已持久 LSN。写入推进前者，刷新推进后者。
  ACK 取所有段已持久版本的最小值，保证被确认的每个 LSN 在相关段文件中都已
  落盘。
- ACK pin：集合持有 pin 注册表，快照与复制先注册 pin 再捕获文件，完成后释放。
  存在 pin 时 ACK 不得越过最低 pin 之前；截断按有效水位执行，pin 之后的日志
  保留。
- 并发检查点：刷新分两遍。第一遍不持写入门，与新写入并发执行，只覆盖刷新起点
  之前已应用的事务；第二遍在门内补刷尾部增量，随后发布水位与截断。发布水位不
  超过第一遍覆盖的前沿，门内时间只剩补刷与发布。
- 压实代理：维持现有“门外重写、门内提交”结构，提交时只在写锁下切换 generation
  守卫与重建内存目录。期间写入经由 WAL 与内存状态自然吸收，不需要额外代理层。
- 水位绑定：generation lsn 仍是回放下界，ACK 仍绑定 generation；跨 generation
  的 ACK 读作零，不生效。

## 状态转换

- 写入推进段已应用版本，刷新推进段已持久版本，ACK 取最小已持久版本并受 pin
  钳制。
- pin 注册后触发多次刷新，ACK 被卡住；释放后恢复推进。
- 段刷新失败阻止 ACK 并上报维护状态，不静默推进。

## 不变量

- 被 ACK 的每个 LSN 在所有相关段文件中都已 durable。
- pin 存在时 ACK 只前进到 pin 之前。
- 任何段 flush 失败都不推进 generation 与 WAL。

## 验收

- 检查点期间持续写入，确认没有事务丢失，ACK 不越过未刷新版本。
- 持有 pin 后触发多次刷新，确认 ACK 被正确卡住，释放后恢复。
- 注入段刷新失败，确认 generation 与 WAL 都不推进。
- 快照复制窗口内的 WAL 段在 pin 释放前不被截断。
