# 阶段七细化方案：字段索引持久化与 named/sparse 一等公民

对照 `docs/plan/storage-gap-remediation.md` 阶段七与问题清单中“打开时重建成本高”、
“named/sparse 一等公民存储与索引”、“其他索引与检索能力”三节。本文只描述职责、
状态转换和验证边界，不展开完整实现代码。

## 对照 qdrant

qdrant 为每个 immutable segment 发布独立的 id tracker、向量存储、payload 存储和多类
字段索引；字段索引存在 mutable / immutable / on-disk 变体，全文索引带 tokenizer、
term 字典、位置与文档长度，稀疏向量通过 segment 内 posting 索引参与检索。simvec
在阶段四到阶段六之后已经有段边界、段级索引文件、压缩 payload 和按外部 id 定位的
payload 记录，但字段索引只持久化 schema，内容在每次 open 时全量重建；稀疏与 named
空间虽然有独立空间和 posting bucket，规划器只看 posting 条数。

本轮实现把字段索引升级为可检查点发布的派生文件，并补齐索引谓词与稀疏规划统计。
facet、nested 的专用磁盘索引、BM25 打分、multivector 和 f16/uint/binary/turbo
kernel 不在本轮落地范围内；它们的接口边界在阶段九方案中列出。

## 职责划分

- 索引种类：`PayloadIndexManager` 为每个字段持有一个 `FieldIndex` 和一个
  `ShapeIndex`。schema 决定主索引：Integer/Float 走排序数值索引，Geo 走坐标索引，
  Text 走全文 token 索引，Keyword/Bool/Datetime 走归一化键值索引。Bool 与数值字符串
  共用键值归一化，因此 `Match` 谓词不需要第二种编码。
- 值形态：`ShapeIndex` 与字段类型正交，记录字段是否存在、是否为 null、是否为空和
  元素数量，因此 `IsNull`、`IsEmpty` 与 `ValuesCount` 对所有已声明字段都可加速。
  `IsNull` 对缺失字段为真，与后置过滤器一致。
- 全文：`TextIndex` 保存 term → `(slot, positions)` 的 posting、文档长度和可参与
  子串/前缀判断的原始字符串。注册与查询共用同一个 lowercase/alphanumeric tokenizer。
  `Match` 允许 token 命中带来的假阳性，`Contains` 与 `StartsWith` 在位图上即可精确
  判断；所有假阳性都仍经过完整后置过滤。
- 规划器：`must`、`must_not`、`should`、`min_should` 继续沿用集合语义；任一子句无
  可用索引时该组不参与收窄。高基数字段由既有 `FilterBitmap` 预算退化为纯后置过滤。
- 持久化：`field_indexes.bin` 用统一 CRC 信封发布，携带 generation、lsn、slot 容量
  和每个字段的索引内容。声明变更、checkpoint 与压实重写都会重发该文件；open 时只有
  generation、lsn 和字段定义全部一致才采用，任何损坏或版本不匹配都丢弃并重建。
  它是纯加速数据，不参与提交决策。
- 压实：staging 阶段按新 slot 编号重建字段 posting，并把结果写入 staged generation；
  新 generation 发布后索引文件天然属于新代，不再跨代复用旧位图。
- named/sparse：每个空间继续拥有独立的 `postings/` 与生成绑定的 manifest。manifest
  增加 term 文档频率；规划器用词频最高的查询 term 判断走 posting 累积还是全扫描。
  该判断只影响执行路径，不改变候选资格、打分或公开分数方向。

## 状态转换

- 声明索引 → 按当前 payload 填充 → 写 `payload_indexes.json` 和 `field_indexes.bin`。
- 写入 payload → 同一路径更新 typed posting、shape 位图和自动 bitmap。
- checkpoint → payload 文件同步 → 写 `dense_snapshot.bin` → 写 `field_indexes.bin` →
  发布 generation lsn。文件领先于 manifest 时在 open 被丢弃。
- open → 读取定义 → 尝试采用索引文件 → 失败时按 payload 与 WAL 重放重建。
- sparse 写入 → pending/flush → checkpoint 写 manifest 与文档频率 → open 按 snapshot
  lsn 决定是否恢复字典与统计。

## 不变量

- pre-mask 只允许多给候选，绝不漏掉满足完整过滤条件的点。
- 索引文件绑定 generation 与 lsn；损坏只失去该代加速，不阻塞 open 或改变结果。
- 字段索引与 sparse 统计都是派生数据；WAL 与 generation 仍是权威恢复来源。
- named/sparse 与 dense 共用外部 id 和点版本语义；posting 规划不改变分数契约。

## 验收

- Keyword/Bool/Numeric/Text/Geo/IsNull/IsEmpty/ValuesCount 的解析、持久化与重开测试。
- checkpoint 后破坏索引文件，重开仍能由 payload 与 WAL 恢复相同查询结果。
- 混合过滤器与全量后置过滤结果逐点一致；未声明字段仍走安全路径。
- 稀疏 posting 的文档频率跨 checkpoint 恢复；空查询与高频查询的规划路径稳定。
- 性能只使用 quick bench 做相对比较，不跨机器、不与完整 bench 数值比较。
