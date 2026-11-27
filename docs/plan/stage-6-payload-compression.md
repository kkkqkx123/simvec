# 阶段六细化方案：payload 压缩、blob 外置与身份解耦

对照 `docs/plan/storage-gap-remediation.md` 阶段六与问题清单中“payload 压缩与
存储职责分离”一节。只描述职责、状态转换与验证边界。

## 对照 qdrant

qdrant 的 blobstore（gridstore/logstore）默认 LZ4 压缩，并把压缩配置作为存储
形式的一部分；WAL 只记录逻辑操作，blob 文件为派生数据，重放按 external id
重建。simvec 现状是页面值只存 `[u32 CRC32][JSON]`，无压缩；WAL upsert 内联完整
向量与 JSON，`points.bin` 与 `payloads.bin` 为派生文件。缺少压缩会放大 WAL、
页面与快照体积，全量内联则让重放重复解析与写入 blob。

## 目标

减小 WAL、页面与快照体积，并让 payload 存储与事务日志职责分离：日志保留逻辑
值权威，blob 文件为可重建派生。过渡期间保留内联记录保证可恢复，读取路径统一
为逻辑 payload。

## 职责划分

- 值编码：页面值扩展为 `[u32 CRC][u8 codec][data]`，codec 区分无压缩与 LZ4。
  CRC 覆盖 codec 与压缩后字节，解压前先校验，解码失败不返回旧值。旧格式
  `[u32 CRC][JSON]` 在打开与读取时仍可识别，按需重写为新格式，不要求一次性
  迁移。
- 压缩配置：归属 `StoreConfig` 几何的一部分，随 `payloads_store/config.json`
  持久化。新写入按配置编码，读取按值内 codec 解码，与配置无关，保证配置变更
  后旧值仍可读。
- WAL 语义：dense upsert 与 payload 记录继续内联逻辑值，WAL 仍是权威日志；
  kind 与 named 的 `points.bin` 与 `payloads.bin` 仍为派生文件，重放按 external
  id 重建 blob。引用式 WAL（只记 blob 引用与版本）为后续项，本阶段不改变 WAL
  记录形状，只通过压缩减小体积。
- 身份解耦：payload 操作统一按外部 id 定位，应用时经 id tracker 解析槽位，
  live 与重放共用同一解析路径。槽位号只在应用时按日志顺序分配，不出现在日志
  中。
- 检查点顺序：`blobs.sync → postings.flush → checkpoint_payloads → snapshot →
  WAL 截断` 保持不变，每一步传播 IO 错误，manifest 仍是提交点。

## 状态转换

- 写入按配置压缩后落页，WAL 内联逻辑值；读取先校验 CRC 再按 codec 解码。
- 配置变更只影响新写入，旧值按自带 codec 解码，不重写存量。
- kind 与 named 重放按 id 重建 blob 与索引，字典 lsn 与快照 lsn 不一致时丢弃
  字典并重建。

## 不变量

- blob 未 durable 前，WAL 中有足够信息恢复逻辑值。
- 压缩值经 CRC 校验后才解析，位翻转被拒绝。
- 页面分配器指针语义不变，压缩只改变值编码。

## 验收

- 同一 payload 在两种压缩配置下往返一致；位翻转被 CRC 拒绝。
- 大 payload 写入后崩溃重开，重放结果与崩溃前一致。
- WAL 体积与检查点时间用 quick 档做前后对比，不与完整 bench 比较，不跨机器
  比较。
- kind 与 named 检查点顺序测试锁定顺序，错序写入被拒绝或重建。
