# 阶段四细化方案：dense 物理段拆分与段级 id tracker

对照 `docs/plan/storage-gap-remediation.md` 阶段四与
`docs/issue/storage-comparison-with-qdrant.md` 中“dense 数据物理上仍是整代单文件”
一节。只描述模块职责、状态转换与验证边界。

## 对照 qdrant

qdrant 的 shard 由多个自治 segment 组成，每个 segment 拥有独立的 id tracker
（external→internal、point version、delete version）、向量存储、payload 存储与
字段索引。压实、索引构建与快照都以 segment 为单位，打开时按 segment 逐个映射，
单个 segment 文件损坏只降级该 segment。

simvec 现状是整代单文件：`vectors.bin`、`keys.bin`、`tombstones.bin` 与 payload
文件属于整个 generation，`segments.json` 与 `hnsw_seg_*` 只描述逻辑边界。压实
重写全集合，快照全量复制，打开时一次性映射整代数据。

## 目标

把逻辑段升级为物理段意识：查询只使用一个段快照，压实与快照只触及被改动的段，
单个段文件损坏只影响该段。完整物理拆分需要格式迁移，本阶段分两步走：先落实
段级 tracker、段快照与选择性压实规划（不改文件布局），再把向量与目录文件拆为
按段独立文件（新 generation 启用，旧 generation 经压实迁移）。

## 职责划分

- 段清单：`segments.json` 在现有水位字段之外记录每段的起止槽位、存活数、
  墓碑数与已持久 LSN。清单仍是派生文件，与 `meta.bin` 不一致时按后者重建。
- 段级 tracker：全局 `IdMap` 保留 external→slot 权威映射，新增按段统计
  （每段存活数、墓碑数、已应用 LSN、已持久 LSN）。槽位号仍全局连续，段归属由
  槽位区间推导，不引入第二套寻址。
- 段快照：查询沿用现有整代 `DataFiles` 守卫语义，明确为一次查询只持有一个
 守卫，不混用两代文件。守卫引用计数决定旧 generation 文件何时可删。
- 选择性压实：压实规划先按段计算垃圾率，只重写超过阈值的冻结段；干净段字节
  直接复制，不重排槽位。槽位重编号仍在新 generation 内一次完成，保持外部 id
  稳定。
- 损坏隔离：每段 ANN 与量化文件已按段发布，损坏只丢弃该段加速能力，其槽位
  回落到精确扫描。向量与目录文件拆分完成后，同样的隔离扩展到向量本体。

## 状态转换

- 写入：按槽位区间更新所属段统计与已应用 LSN，不触及已持久 LSN。
- 检查点：刷新数据文件后把所覆盖段的已持久 LSN 推到水位，清单随 `meta.bin`
  一起发布。
- 压实：冻结段按垃圾率入选，appendable 段不入选；新 generation 发布后切换
  守卫，旧文件在引用计数归零后删除。
- 打开：清单缺失或与 `meta.bin` 不一致时扫描重建；损坏的段 ANN 文件仅禁用
  该段加速，不阻断打开。

## 不变量

- 一个查询只使用一个段快照，不混合两代文件。
- 每个 id 在一个生成内只有一个权威段归属。
- generation manifest 仍是唯一提交点；发布后失败进入 poison，拒绝写入直到重开。

## 验收

- 持续写入下压实并反复重开，搜索、scroll 与逐段计数一致。
- 只读事务持有旧守卫时确认旧段文件未提前删除。
- 单个段 ANN 文件损坏只影响该段，其它段继续服务；向量文件损坏的隔离随物理
  拆分完成后补齐。
