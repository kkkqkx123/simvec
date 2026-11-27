# 存储结构审计与加固计划

本文对比 `simvec` 当前存储实现与 `ref/qdrant`，记录完整问题清单、已落地的修改和后续方向。只讨论本地引擎的原生存储；wasm 内存引擎不在范围内。

## 现状对比

| 维度 | simvec | ref/qdrant |
| --- | --- | --- |
| 组织方式 | 集合根目录保存类型标记、`wal/` 和 `generation.bin`； serving 数据全部位于 `generations/g<id>/`，密集、kind、named 共用同一套发布协议 | collection → shard → 多个 segment；appendable segment 接收写入，不可变 segment 承载压实后的数据和索引 |
| 点标识 | `IdMap<PointId, slot>` + 全局 LSN，没有逐点版本 | id tracker 同时维护 external id → internal offset 和逐点 version / delete version |
| 向量存储 | 分段 mmap 的定长 f32，槽位原地覆盖；`kind`/`named` 的可变长向量走 `points.bin` blob | chunked mmap（可追加）+ deleted bitvec、不可变 mmap、graph-inline、多种数据类型与 multivector |
| Payload | `payloads_store/`：`config.json` + `tracker.bin` + `bitmask.bin` + `pages/page_*.bin`，内容由 WAL 派生 | blobstore gridstore/logstore，带 tracker journal、compacted tracker、gap 管理、打开时 pointer 一致性校验、raw payload 直通 |
| WAL | 集合级 `wal/<20位首LSN>.bin`，postcard + CRC32，group commit；checkpoint 按整段 unlink | shard 级 mmap 段，magic/version/随机 CRC seed，CRC32C、8 字节对齐、预建段、retain closed、`first-index` ACK 水位 |
| Checkpoint | flush 派生文件 → `meta.bin` → marker → `truncate_through`；kind/named 另有 snapshot + posting manifest LSN | segment 有 `segment.json` / version file，per-component `VersionTracker`，WAL 只在 flush 确认后截断，snapshot 带文件清单 |
| 压实 | 重写整个集合的所有文件，逐个 rename；竞争严重时退化为全局写锁 | optimizer 在临时目录构建完整新 segment，proxy 吸收期间写入，`swap_new` 一次切换 |
| Payload 索引 | schema 持久化到 `payload_indexes.json`，内容在内存重建；Map/Numeric 两类；只加速 `must` | 多种 field index（map/numeric/full-text/geo/facet/nested/null），mutable/immutable/on-disk，带选择性估计 |
| ANN | 集合级 HNSW/IVF；新点进 `pending`，查询时暴力补扫；压实靠 remap 保留图 | 每个不可变 segment 独立建图，appendable segment 精确扫，图链接 mmap、graph healing、payload blocks |
| 具名/稀疏 | `points.bin` + `payloads.bin` + `postings/` + snapshot，全程 exact scan | sparse/named/multivector 是 segment 内一等公民，有独立存储和索引 |

## 问题清单

### 严重：多文件提交不具备原子性

1. dense 压实按顺序替换 `vectors.bin` → `quant.bin` → `keys.bin` → `payloads_store/`，随后才写 tombstones、meta、索引和截断 WAL。任一中间点崩溃都会留下不同代际的文件组合；WAL 只能覆盖上次 checkpoint 之后的增量，无法修复旧数据。
2. `kind`/`named` 压实先用 `PointBlobs::replace_from` 覆盖 `points.bin`，之后才写新 snapshot。旧 snapshot 记录的 `blobs_len` 与截断后的新文件不兼容时，`PointBlobs::open` 直接报 `CorruptData`；此前 checkpoint 已截断 WAL 前缀，无法从头重建。
3. 压实失败重试后会在写锁下整体重写，写入停顿与集合大小成正比；qdrant 的 optimizer 不这样做。

### 严重：缺少逐点版本和可靠的已应用水位

4. dense 打开从 WAL 头部全量重放，`meta.last_applied_txn` 只推进、不跳过；没有 per-point version。若已删除的 WAL 段在崩溃后重新出现，旧 upsert 会覆盖较新状态，甚至把已删除 id 以新槽位重新插入。
5. `Wal::remove_segment` 原本只 unlink，不 fsync WAL 目录；段删除本身缺少持久化保证。
6. kind/named 依赖 snapshot 的 LSN 跳过 WAL；snapshot 损坏时回退全量重放，但 WAL 前缀可能已被截断，这是设计上的耦合点。

### 中等：派生文件缺乏自愈与校验

7. `bitmask.bin` 原为原地覆盖写，半写文件会让 `MmapBitmask::open` 失败并导致整个集合被隔离；qdrant 用 journal 修复 torn tracker，并对 bitmask/gap 做打开时自检。
8. `tracker.bin` 虽走 tmp+rename，但没有 fsync 父目录；指针也没有像 qdrant `Logstore::validate_consistency` 那样与 page 文件长度交叉校验。
9. dense 压缩/写入失败时，量化码会静默保持 `ready=true`。rescore 只能修正候选顺序，不能补回被陈旧 quant code 漏掉的近邻。

### 中等：启动成本、WAL 编码与功能差距

10. dense 没有带 LSN 的 snapshot：open 时扫描全部槽位重建 id map、filter bitmap、payload index，再重放全部 WAL。
11. WAL 的 id 与 payload 以字符串/JSON 形式存进 postcard，双重序列化；单批整体编码进内存；没有 mmap 段、随机 CRC seed、预建段队列和 closed 段保留。
12. payload index 只有 Map/Numeric 且只在内存，未加速 `must_not` / `should` / `min_should`，没有 full-text、geo、facet、nested、on-disk immutable 索引和选择性估计。
13. HNSW 新点只进 `pending`，查询补扫成本随积压增长；索引是集合级全局文件，压实靠 remap，图有空洞时只能丢弃重建。
14. `AGENTS.md` 的 dense 布局描述遗漏 `tombstones.bin`、`payloads_store/`、`payload_indexes.json`、量化与索引产物，文档与实现已漂移。

## 本轮已落地的修改

1. **WAL ACK 水位**：`wal/acked-lsn` 以带 CRC 的文件持久化；`truncate_through` 和 `truncate` 在 unlink 前先写入并 fsync 水位，段删除后 fsync WAL 目录。dense 恢复改用 `read_all_acked()`，只重放水位之后的记录；compaction 也改走 `truncate_through`，不再使用无水位保护的 `truncate()`。
2. **payload bitmask 原子落盘**：`bitmask.bin` 改为 tmp → fsync → rename → fsync 父目录；`tracker.bin` 的 rename 补上父目录 fsync；打开时校验每个 tracker 指针落在对应 page 内，缺失页面或越界指针直接报损坏，不再静默返回空 payload。
3. **量化失败失效**：quant code 写入失败时持久化地把 `quant_meta.bin` 的 `ready` 置为 false，重新打开仍走 exact/重建路径，避免陈旧 code 造成 recall 下降。
4. **generation 原子提交**：新增统一 `generation.bin` 清单与 staging/publish 协议；dense 以 `DataFiles` 整体切换文件句柄，kind/named 在写锁内切换 blob/postings；压实先写完整新代再发布清单，WAL 只在清单之后截断。
5. **WAL 回放绑定清单**：dense 以 `min(ack, generation.lsn)` 回放并在 ack 超前时报损坏；kind/named 直接以 `generation.lsn` 回放，snapshot 与清单不一致报 `CorruptData`。
6. **文档**：新增本文；`AGENTS.md` 更新实际目录布局、generation 规则与 WAL 水位规则。

## 细化的落地方案

### 已实现：generation 原子提交

#### 目录与清单

```
<collection>/
  generation.bin                    # postcard + CRC，记录 generation id / lsn
  wal/                              # 集合级日志，跨 generation 共享
  meta.bin | kind.bin | named.bin   # 类型标记（meta.bin 随 serving generation 移动）
  generations/
    g00000000000000000001/          # 当前发布代
      ...                            # 该代的全部数据文件
    .g00000000000000000002-staging/ # 未发布构建目录，open 时忽略
```

`generation.bin` 用已有快照信封写入（magic + version + CRC + postcard），字段为 `version: u32 = 1`、`generation: u64`、`lsn: u64`。它是唯一的提交点：只有它的 rename 完成，新目录才对 open 可见。目录内容整体 fsync 后先 rename 到最终 `g<id>`，再原子替换清单；崩溃时只会出现“旧清单 + 未发布临时目录”或“新清单 + 完整新目录”，不会出现半代文件。

各代文件清单：

- dense：`meta.bin`、`vectors.bin`、`keys.bin`、`tombstones.bin`、`payloads_store/`、`payload_indexes.json`、可选 `quant.bin`/`quant_meta.bin` 与 `index.bin`/`hnsw.bin`；
- kind：`kind_snapshot.bin`、`points.bin`、`payloads.bin`、`postings/`；
- named：`named_snapshot.bin`、`points.bin`、`payloads.bin`、`postings/<space>/`。

#### 读路径与句柄切换

- dense 把所有文件句柄收进 `DataFiles`，通过一层 `ArcSwap<DataFiles>` 整体发布。查询在开始时取一次 guard，保证 vectors/keys/tombstones/payload/index 来自同一代；已有 mmap 由 guard 持有，切换后仍有效。
- kind/named 的查询本来就持有 `inner` 锁，因此 blob/postings 句柄的 `ArcSwap` 切换与该锁配合即可：写锁下先发布清单，再换句柄，最后截断 WAL。
- 被替代的 dense generation 由后台线程等待最后一个 `Arc<DataFiles>` 引用释放后删除；kind/named 在 `inner` 锁内完成切换后直接删除旧目录。

#### 操作顺序（失败即停止，不补偿）

1. 创建：建临时代 → 写全部文件并 fsync → rename 目录 → 发布 `generation.bin`（lsn=0）。
2. checkpoint（原地写当前代）：数据文件 → snapshot/meta → `generation.bin.lsn=lsn` → `wal/acked-lsn` → `truncate_through(lsn)`。清单先于 WAL 水位，恢复永远不会越过清单声称已落地的数据。
3. 压实：`stage()` → 在 staging 目录写完整一代（含索引重映射、snapshot、payload 重写）并 fsync → 预打开校验 → rename 目录 → 发布 `generation.bin(new_id, lsn)` → 切换进程内句柄 → `truncate_through(lsn)` → 清理旧代。
4. 恢复：dense 以 `min(ack, generation.lsn)` 为回放下界（ack 大于清单视为损坏）；kind/named 直接以 `generation.lsn` 为下界，snapshot 缺失或与清单 lsn 不一致报 `CorruptData`。这样 checkpoint 后重新出现的旧 WAL 段不会污染新代。

#### 失败矩阵

| 崩溃点 | 磁盘状态 | open 结果 |
| --- | --- | --- |
| staging 写入中 | 旧 `generation.bin` + 残留 `.g*-staging` | 忽略临时目录，继续旧代 |
| 代目录 rename 后、清单替换前 | 旧清单 + 完整但未发布的 `g<N>` | 继续旧代，临时新一代被忽略 |
| 清单替换后、WAL 截断前 | 新清单 + 完整新代 + 旧 WAL 段 | 按清单 lsn 回放尾部，旧段被跳过 |
| 清单替换后、进程内句柄未切换 | 新磁盘代 + 本进程旧句柄 | 本进程继续服务一致的老文件；重启使用新代 |
| meta/snapshot 与清单 lsn 不一致 | 数据自相矛盾 | `CorruptData`，不猜测、不降级回放 |

### 后续修改方案（细化）

#### 版本化恢复

1. dense 当前仍以 `meta.bin` + WAL 重建 id 目录和过滤结构；下一步把 id→slot 目录、tombstone 位图、payload tracker 版本和 `next_slot`/`slot_capacity` 写进带 LSN 的 dense snapshot，与 `generation.bin` 同代发布。open 只回放 snapshot.lsn 之后的记录。
2. 在 id tracker 中增加 `last_applied_lsn`，apply 时丢弃 `lsn <= last_applied_lsn` 的记录；这是 generation 水位之外的第二道防线，主要防手工放回旧段的场景。
3. 把 dense 的 WAL ACK 文件也绑定 generation id（当前仅绑定 lsn）；kind/named 的恢复已完全以清单为准，不再读 ACK 文件。

#### 段化与索引能力

1. 引入 appendable/immutable segment：appendable 只追加，immutable 由 optimizer 在新 generation 内构建；写入经 proxy 吸收切换窗口，避免整体写锁重写。
2. ANN 按段建图：查询 fan-out 后归并 top-k；用有界 drain 替代无限增长的 `pending`；图链接持久化并带 generation 版本。
3. field index 抽象 mutable/immutable/on-disk 三类，并补 full-text/BM25、geo、facet、nested、prefix、null；增加基数/直方图统计供过滤规划器选择路径。
4. WAL 迁移到 mmap 段、随机 CRC seed、raw payload bytes 与流式编码，保留 first-index ACK 语义。

## 验证重点

已实现部分的自动化覆盖：`test_unpublished_generation_is_ignored`、`test_generation_advances_with_checkpoint_and_compaction`、`kind_compaction_publishes_a_generation`，以及既有 checkpoint/WAL/量化/压实测试。

- checkpoint 后杀进程：重放只覆盖 ACK 之后的记录；人工放回旧 WAL 段不会污染数据。
- checkpoint 中途杀进程：bitmask/tracker 文件要么完整、要么保留上一版，集合可以打开。
- tracker 指向不存在或过短的 page 时 open 报 `CorruptData` 而不是吞掉 payload。
- quant code 写入失败后，`is_ready()` 为 false，搜索走 exact，rebuild 后恢复 quantized。
- 压实中途杀进程：generation manifest 仍指向旧一代，未发布临时目录被忽略，WAL 未被提前截断。
