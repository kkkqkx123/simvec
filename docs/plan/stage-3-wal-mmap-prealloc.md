# 阶段三细化方案：WAL mmap 化与预分配

对照 `docs/plan/storage-gap-remediation.md` 阶段三与问题清单中“WAL mmap 化与
预分配”一节。只描述职责、状态转换与验证边界。

## 对照 qdrant

qdrant `lib/wal` 的 segment 是预分配 mmap 文件：段头含 magic（`wal`）、格式
版本、随机 CRC seed、`first-index`；entry 按 8 字节对齐，CRC32C 覆盖“长度 +
数据 + 填充”并以前一条 CRC 链式累加（seed 作为链首），因此复用旧文件名的
段必然 CRC 失配；后台 `SegmentCreator` 预建空段队列，append 从不在关键路径
上创建文件；closed 段按 ACK 水位与保留策略回收，至少保留一个；回放用段内
索引按记录随机访问（`Entry` 为 mmap 视图），只有最后一段的 torn tail 可容忍，
中间段任何损坏都是 `InvalidData`。

simvec 现状：`write_all + sync_all` 追加、段懒创建、无预分配、无记录索引、
回放把事务读进 `Vec`；`segment_bytes` 阈值用物理长度判断，torn 尾会污染滚动
决策（已有 valid-prefix 修复）；CRC 为无 seed 的单记录 CRC32。

## 段格式（版本 1）

- 段头 32 字节：magic `SVWC`、格式版本 `1`、随机 seed `u32`、首 LSN `u64`、
  保留位。段名仍为首 LSN 的 20 位零填充十进制；头内首 LSN 必须与段名一致，
  否则按 `CorruptData` 处理。
- 记录：`[u32 len][u32 crc][postcard bytes][0~7 零填充]`，总长 8 字节对齐。
  CRC 以 seed 为链首、以上一条记录 CRC 为链式初值，覆盖范围为
  “len 字节 + 数据 + 填充”，多项式沿用站内 CRC32 实现并保留向 CRC32C
  迁移的余地（链式结构与校验覆盖范围与 qdrant 一致，替换多项式不改格式）。
- 文件创建即预分配到 `segment_bytes`（至少容纳段头加一条记录，向下 8 对齐）；
  有效前缀 `valid_len` 与最高 LSN 由内存状态维护，`appended_bytes` 按有效
  字节计量，不再用稀疏文件的物理长度。

## 修改内容

1. 段句柄 mmap 化。active 段持有 `MmapMut`，append 按事务逐个编码后顺序
   写入映射区（不再拼装整批大 `Vec`），一次 group commit 对应一次可见性发布
   与一次 `flush_range + sync`；失败时不发布 `valid_len`（映射内脏页不影响
   已发布前缀）。
   回放使用只读 `Mmap` 按记录随机访问并建立段内索引。
2. 预建空段队列。wal 目录维持一个 `.spare` 预分配 spare 文件（与段名空间不
   相交，目录扫描静默跳过）；滚动时若 spare 容量匹配则直接重命名为目标段名
   并写入新段头，随后同步补建下一个 spare。补建在本次 `append_batch` 的锁外
   尾部同步完成：首次滚动仍付费，其后滚动不再在关键路径上创建文件；
   后台线程化补建作为后续优化，接口上保留队列长度可配。
3. 滚动与校验语义保留。滚动前先把 torn tail 截断再决策（只用 `valid_len`）；
   新段名仍为 `max(首事务 LSN, 尾段名 + 1)`——阶段一保证 marker 一律取新 LSN，
   故复用名不可能出现；回放仍要求 closed 段 `last + 1 == next`，torn tail 仅
   容忍于最后一段；`acked-lsn` 的 generation 绑定与段连续性校验保留。
4. `appended_bytes` 改为有效字节：open 时扫描各段有效前缀求和；append 按
   记录有效长度累加；截断按移除段的有效长度扣减。checkpoint 的
   size-based 调度语义不变。
5. 保留策略：本地 checkpoint 仍允许截空（ACK 即 generation 水位时段可清空，
   marker 段因首 LSN 大于水位而存活）；“至少保留一个 closed 段”作为面向
   远端复制（WAL delta）时的保留配置，本地默认不改变截断语义，避免 checkpoint
   后必留一段的无谓重放。

## 状态转换

- `open_or_create`：建目录 → 列段 → 扫描各段头与有效前缀（坏头/坏 CRC 按
  回放同等规则定性：末段截断为有效前缀，中间段直接错）→ 绑定时不读 ACK
  以外的任何水位。
- `append_batch`：失败注入检查 → 修 torn 尾 → 决策滚动（spare 命中则改名，
  否则创建预分配段并目录 fsync）→ 流式编码写入映射 → `flush_range` →
  发布 `valid_len` / `last_lsn` / `appended_bytes`。
- `truncate_through`：先发布 ACK（绑定 generation）并目录 fsync → 按水位
  unlink 全覆盖段（尾段仅当 `active_last_lsn <= watermark`）→ 目录 fsync。
- `replay_after`：按段顺序 mmap 解析，跳过 `<= floor` 的事务，末段 torn 止，
  中间段坏即错，并做段名连续性校验。

## 验收

- 保留 torn tail、段丢失、段重叠、checksum、truncate、group commit、跨
  generation ACK 七类回归（按新格式重写其注入手段：位翻转必须落在有效前缀内）。
- 与阶段一、阶段二组合后重跑失败提交与 checkpoint 滚动测试。
- quick 档提交延迟 smoke：只做相对比较，不与完整 bench 数值比较，不跨机器比较。
