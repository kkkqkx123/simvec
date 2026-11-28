# 存储整改任务清单与修改方案

本文是 `docs/plan/storage-gap-remediation.md` 的收尾清单：方案主线已经落地，剩下的是**已确认缺陷的修复**、**遗留代码清理**和**明确延后项**。与 `docs/plan/storage-implementation-audit.md` 的“当前基线”描述冲突时，以 `AGENTS.md` 的存储章节为准（三份审计类文档的描述已落后于代码，见任务 D1）。

## 版本策略（本轮的硬约束）

- 所有磁盘格式版本号**统一锁定为 1**，本轮不引入版本 2，也不写任何“版本 N 到版本 M”的迁移路径。
- 版本字段仍然存在，仍然在读取时校验：读到非 1 的版本必须报 `CorruptData`，这是**前向拒绝**，不是向后兼容。
- 当前不存在需要兼容的旧数据，因此禁止出现“按版本分支读取”的代码；任何 `if version == 1 { .. } else if version == 2 { .. }` 形态的写法都视为缺陷。
- `payload_store/bitmask.rs` 的自相矛盾用例是本策略的直接反例：常量是 1，用例却断言“版本 1 必须被拒”，见任务 A1。

## 任务总览

| 编号 | 任务 | 类型 | 优先级 | 状态 |
| --- | --- | --- | --- | --- |
| A1 | bitmask 版本用例与常量自相矛盾 | 缺陷 | P0 | 本轮完成 |
| A2 | `Wal::truncate_through` 错误路径打坏内存状态 | 缺陷 | P0 | 本轮完成 |
| A3 | 零长度 / 短段文件导致 open 失败 | 缺陷 | P0 | 本轮完成 |
| A4 | JS 缺失 `simvec_maintenance_status` 的 FFI 声明，Python 缺失 `argtypes` | 集成缺陷 | P0 | 本轮完成 |
| A5 | 磁盘格式版本常量不统一（WAL 段与目录文件为 2） | 一致性缺陷 | P1 | 本轮完成 |
| B1 | `repair_hnsw()` 挂在每次写操作上 | 性能缺陷 | P1 | 本轮完成 |
| B2 | 快照连同 WAL 预分配空洞与 `.spare` 一起复制 | 容量缺陷 | P1 | 本轮完成 |
| T1 | 为 A2 / A3 / B1 / B2 补回归用例 | 测试 | P1 | 本轮完成 |
| C1 | `truncate_wal_through` 重复实现 `WalAckPins::clamp` | 遗留代码 | P2 | 本轮完成 |
| C2 | `store_maintenance` 中 `let _ = quant;` 死代码 | 遗留代码 | P2 | 本轮完成 |
| C3 | 单文件 `TombstoneBits::load` / `save` 是否遗留（核查后否定） | 文档缺陷 | P2 | 本轮完成（结论：不是遗留） |
| C4 | WAL 注释与常量不一致（`SVWL` / `SVWC`、已删除的 `read_all`） | 文档缺陷 | P2 | 本轮完成 |
| D1 | 三份审计文档“当前基线”过期 | 文档缺陷 | P2 | 本轮完成 |
| E1 | 压实仍整代重写、slot 全局重排（阶段四 T9） | 能力缺口 | 延后 | 设计完成（`storage-remaining-tasks.md`），代码未动 |
| E2 | 段级 flush 仍需关门复刷 | 能力缺口 | P1 | 部分完成：关门时只刷竞态段；摘 gate 见 E2b |
| E3 | 快照全程持队列门（quiesce） | 能力缺口 | P1 | 部分完成：移除读锁 + reflink 快路径；摘 gate 依赖 E1b |
| E4 | CLI 缺 `maintenance_status` 与四个快照命令 | 能力落差 | P1 | 本轮完成 |

本轮只做 A、B、C、D 四组，且不做任何磁盘格式变更：A1 修的是用例而不是格式，B2 只影响快照内文件长度，读回路径本来就允许短段和缺失 `.spare`。

---

## A1 bitmask 版本用例与常量自相矛盾

文件：`src/storage/payload_store/bitmask.rs`。

根因：`BITMASK_VERSION = 1`（:35），`read_file` 对 `version != BITMASK_VERSION` 报错（:347-352）；而用例 `a_version_one_file_is_rejected`（:530-552）把版本字节写成 `1` 后断言 `open` 必须失败，并 `panic!("must reject a version-1 file")`。写入等于空操作，用例在任何实现下都必然失败。这是“完成文档要求版本 2、代码把版本锁在 1”的残留。

修改：

- 用例改名为 `an_unsupported_version_is_rejected`，改为写入 `BITMASK_VERSION + 1`，断言错误文本包含 `version`，注释只说明该版本不被本构建接受。
- `BITMASK_VERSION` 保持 1，注释只说明它是写入文件头的布局版本。
- 不改 `BITMASK_HEADER_LEN`、不改 CRC 覆盖范围，格式不变。

不变量：版本 1 的文件可以读；任何非 1 版本的文件在 CRC 校验之前就被拒。

## A2 `Wal::truncate_through` 错误路径打坏内存状态

文件：`src/storage/wal.rs:889-945`。

根因：函数先 `mem::take(&mut state.segments)` / `mem::take(&mut state.sealed_valid)` 把注册表搬空，随后在循环里 `remove_segment(...)?`。任一次 unlink 失败就提前返回，此时 `segments` 已被清空，而 `state.active` 可能仍指向一个已被删除或带 slack 的旧段：

- 下一次 append 走到 `state.segments.last().expect("active segment is listed")`（:641）直接 panic；
- 或者旧段带着预分配 slack 留在盘上，被后续重放当作中间段解析，报 `CorruptData`。

修改：不再搬空注册表，改为“先规划、后应用、失败保留”：

- 用 `state.segments.clone()` / `state.sealed_valid.clone()` 作为只读输入（段数量级很小，克隆成本可忽略）。
- 对每个被覆盖的段尝试 `remove_segment`；成功则计入 `removed_bytes` 并（若是尾段）标记 `removed_tail`；失败则把该段原样保留在 `kept` 里，并记下第一个错误。
- 只有 `removed_tail` 为真时才退役 `state.active` / `active_last_lsn` / `active_valid_len`；否则尾段句柄必须继续可用。
- `sealed_valid` 必须始终与 `segments[..len-1]` 一一对应：`kept_valid` 按“保留顺序”累积，随后在“尾段仍然打开且没被删除”时弹出最后一项（打开中的尾段长度由 `active_valid_len` 承担，不进密封表）。尾段长度读取也改为优先 `active_valid_len`，避免把密封表里残留的项当成活跃段长度。
- `roll_over_locked` 的密封写入加一条长度判断（`sealed_valid.len() < segments.len()`）：截断失败会留下“已列出但无句柄”的段，此时条目已存在，不能再追加一条，否则并行数组会错位。
- 状态写回后 `sync_dir`，再扣减 `appended_bytes`，最后返回记录的第一个错误。

不变量：函数返回后 `state.segments` 与盘上实际存在的段一致（可能多保留几个本该删除的段，但绝不会少）；`state.active` 存在时 `segments.last()` 必然等于它所对应的段；`sealed_valid.len()` 恒等于 `segments.len()` 减去“是否存在打开尾段”（0 或 1）。

## A3 零长度 / 短段文件导致 open 失败

文件：`src/storage/wal.rs:384-396`。

根因：open 对所有段无条件 `MmapMut::map_mut(&file)`。`mmap` 要求长度大于 0，而“创建文件 → 写头”之间崩溃会留下 0 字节段（`create_segment_file` 先 `set_len(capacity)`，但 `take_spare` + rename 之后、`write_segment_header` 之前仍可能被截断，人工修复场景也可能留下空段）。回放路径（`walk_tail_prefix`）把空段当空段容忍，open 却在更早的地方直接报错，两者契约不一致。

修改：取得文件句柄后先取 `metadata().len()`：

- 长度小于配置容量时统一 `set_len(capacity)` 补回预分配长度再映射：0 长度是崩溃残留，中等长度是快照里的有效前缀；补到满容量后快路径不再 remap。
- 长度小于 `SEG_HEADER_LEN` 时同时把 `tail_valid` 重置为 0、`tail_crc` 清空，让 append 走“重新写头”分支（`append_batch` 已处理 `append_start < SEG_HEADER_LEN`）。
- 长度正常的段行为不变，仍然映射整文件（含 slack）。

不变量：空段/短段可以被 open 并继续追加；`active_valid_len` 不会大于文件实际长度。

## A5 磁盘格式版本常量不统一

文件：`src/storage/wal.rs:97`、`src/storage/directory.rs:38`、`src/storage/compaction.rs:46`。

根因：`payload_store/bitmask.rs` 的版本是 1，WAL 段头与目录文件头却写着 2。版本号在没有任何已发布布局需要迁移的前提下分成两套，只会让读者以为存在“v2 相对 v1 的格式差异”，并诱发“要不要写 v1 读取分支”的错误讨论（A1 的用例就是这种讨论的产物）。

修改：三个常量统一为 1，注释只说明它是写入文件头的布局版本。`directory::FILE_VERSION` 与 `compaction::FILE_VERSION` 必须同时改：compaction 写的是 directory 打开的同一套文件头，两者不一致会让压实后的文件在打开时被判版本不支持。段头 magic 的文档同时从 `SVWL` 订正为常量实际的 `SVWC`。

不变量：全仓库的磁盘格式版本只出现 1；读侧一律 `!= 常量` 即 `CorruptData`，没有分支。

## A4 绑定层 FFI 声明缺失

文件：`bindings/js/src/native.js`、`bindings/python/simvec/_native.py`。

根因：`src/c_api/maintenance.rs:289` 导出了 `simvec_maintenance_status`，`include/simvec.h:407` 也声明了，但：

- JS 的 `DEFINITIONS` 表里没有该符号，而 `bindings/js/src/engine.js:499` 直接调用 `functions.simvec_maintenance_status(...)`，运行期必然抛“function not found”；
- Python 的 `_native.py` 完全没有设置该函数的 `argtypes` / `restype`，`engine.py:474` 的 `maintenance_status` 依赖隐式整型转换，`char**` 出参会被按 32 位截断。

修改：JS 按 `{ arguments: ["pointer", "string", "pointer"], return: "int32" }` 补一条声明，位置与 `simvec_collection_metrics` 相邻；Python 按 `[_engine_p, _c_str, _c_out_str]` + `c_int` 补齐。四个快照函数的声明已存在，不动。

不变量：`native.js` 的 `DEFINITIONS` 与 `include/simvec.h` 的导出集合一致（至少覆盖 engine.js 实际调用的全部符号）。

## B1 `repair_hnsw()` 挂在每次写操作上

文件：`src/engine/maintenance.rs:181-195`、`src/storage/store_maintenance.rs:340-405`、`src/storage/store_write.rs:468-502`。

根因：`after_mutation` 在**每次** upsert / delete 之后调用 `store.repair_hnsw()`。该函数会抢 `build_mutex`、对每个段统计墓碑密度、必要时整段重建。HNSW 发布后，一次单点 upsert 的尾部成本变成 O(全部 slot) 的墓碑扫描，与阶段九“只修补受影响邻接”的设计相反。

同时，删除路径已经在 `heal_hnsw_after_delete` 里做了单段局部修补，`repair_hnsw` 对删除是重复劳动，对 upsert 完全没必要。

修改：给 `CollectionStore` 增加一个“自上次全量修补以来新增的墓碑数”计数器 `hnsw_tombstones_pending: AtomicU64`：

- `update_tombstone_bit(slot, true)` 时 `fetch_add(1)`；
- `after_mutation` 只在 `hnsw_tombstones_pending >= HNSW_REPAIR_TOMBSTONE_BATCH`（取 256）时才调用 `repair_hnsw()`；
- `repair_hnsw()` 成功返回（含“没有已发布图，早退”）后把计数清零；返回错误时保留计数，下一批写操作会重试。

不变量：段内墓碑比例超过 25% 时仍会被修补（只是延后到累积到批量阈值），局部 `heal_deleted` 的即时性不受影响；upsert 路径不再触碰 HNSW 图。

## B2 快照复制 WAL 预分配空洞与 `.spare`

文件：`src/storage.rs:566-579`、`src/storage.rs:773-850`。

根因：快照对整个 `wal/` 目录做递归 `fs::copy`，把所有文件原样搬走：

- `.spare` 是一个按 `segment_bytes` 预分配的完整空洞文件（默认数十 MB），纯属滚动优化，缺失时 `take_spare` 返回 `false`、`roll_over_locked` 直接 `create_segment_file`，因此它对快照毫无意义；
- 尾部活跃段带着预分配 slack（未被记录覆盖的零字节），`snapshot::file_checksum` 还要把这些零字节整体读一遍算 CRC。

修改：

- 复制时跳过 `wal/.spare` 与 `wal/.spare.tmp`。
- 活跃段只复制有效前缀（`active_valid_len`），由 `Wal` 暴露一个 `pub(crate) fn tail_valid_len(&self) -> Option<(u64, u64)>`（段名 + 有效长度）供快照使用；清单里的 `len` / `crc32` 由新增的 `snapshot::file_checksum_prefix` 对**源文件的前缀**计算，与实际写出的字节数一致（`file_checksum` 变成它的 `None` 包装）。
- 增量复用的比较键仍是 `path + version + len + crc32`：由于两侧都按“有效前缀”复制，未变化时键仍然相等，硬链接复用不受影响。

不变量：快照恢复后 open 可以正常打开（短段走 A3 的补长路径，缺失 `.spare` 走直接创建路径）；重放读到的记录集合与截断前一致。

## T1 回归用例

新增四个用例，覆盖本轮每一处行为变化，都不依赖外部注入钩子：

| 用例 | 位置 | 覆盖 |
| --- | --- | --- |
| `test_truncate_through_keeps_a_segment_that_will_not_unlink` | `src/storage/wal.rs` | A2：把已覆盖段的文件替换成同名目录让 unlink 失败，断言返回错误、注册表仍列出该段、盘上不存在“列出但缺失”的段、后续 append 不 panic |
| `test_open_tolerates_an_empty_tail_segment` | `src/storage/wal.rs` | A3：把尾段截成 0 字节后重开，断言 open 成功、回放为空、追加可用且重放只剩新事务 |
| `test_snapshot_copies_only_the_wal_valid_prefix` | `src/storage/tests.rs` | B2：断言快照里的尾段长度等于有效前缀、清单 `len` 一致、`.spare` 不进快照，且恢复后的库能继续写 |
| `test_hnsw_repair_is_batched_by_tombstones` | `src/storage/tests.rs` | B1：300 次 upsert 不触发全量修补，第 256 次删除触发，修补完成后计数归零 |

## C1 `truncate_wal_through` 重复实现 `WalAckPins::clamp`

文件：`src/storage.rs:667-680`。

`truncate_wal_through` 里手写了一遍 `lowest <= watermark => lowest - 1`，与 `wal_ack.rs:51-56` 的 `clamp` 完全同构。改为直接调用 `self.wal_pins.clamp(watermark)`，保留 `effective == 0` 的短路（含义是“全部被 pin 住，什么都不做”）。

## C2 `store_maintenance` 死代码

文件：`src/storage/store_maintenance.rs:510-516`。

`let quant = data.quant.read()...meta_snapshot(); let _ = quant;` 是重构残留：紧接着 520 行又拿了一次 `data.quant.read()` 并调用 `qs.rebuild(...)`。整段删除（含 `meta_snapshot()` 调用），不改变加锁顺序。

## C3 单文件 `TombstoneBits::load` / `save` 是否遗留（核查结论：不是）

文件：`src/storage/tombstones.rs:126`、`:246`、`:179`、`:218`。

早先的审计把这两个函数记为“仅测试使用”，据此给它们加 `#[cfg(test)]` 会直接编译失败：`save_segment`（:179）内部调用单文件 `save`，`load_segments`（:218）内部调用单文件 `load`，它们是每段落盘/读取的实际实现，只是调用点在本文件内、用 `Self::load` / `self.save` 的写法，绕过了“按模块路径 grep 调用者”的检查。

结论：**不加门控，也不删除**。修的是文档而不是代码——`docs/plan/deferred-storage-completion.md` 里“`load` 已 `#[cfg(test)]`”的说法不成立，一并订正。这条留在清单里是为了记录核查结果，避免下次再被当成死代码处理。

## C4 WAL 注释与常量不一致

文件：`src/storage/wal.rs`。

- 模块文档写 `[magic 4 = "SVWL"]`，而 `SEG_MAGIC = *b"SVWC"`（:99），`:98` 的注释也写 `SVWL`。三处统一为 `SVWC`。
- `:407` 的注释仍写“`replay`/`read_all` refines the bound”，`read_all` 已被删除，改为 `replay_after`。

## D1 三份审计文档的“当前基线”过期

文件：`docs/plan/storage-implementation-audit.md`、`docs/plan/deferred-storage-refactor.md`、`docs/issue/storage-comparison-with-qdrant.md`。

三份文档仍在描述改造前状态（“WAL 用 `write_all` + `sync_all`”“dense 数据仍是整代单文件”“CRC 仍是 CRC32”“payload 无压缩”），与 `AGENTS.md` 和代码冲突，会直接误导后续读者。修改方式：在各自开头补一段“本文记录的是改造前基线，现状见 `AGENTS.md` 存储章节与 `docs/plan/storage-remediation-tasks.md`”的声明，而不是逐条改写历史记录（历史审计价值需要保留）。

---

## 剩余能力缺口（E1–E4）

设计与落地情况见 `docs/plan/storage-remaining-tasks.md`。摘要：

| 编号 | 内容 | 状态 | 剩余部分 |
| --- | --- | --- | --- |
| E1 | 压实仍整代重写、slot 全局重排 | 只出设计 | 前置条件是“已发布段不可变”（点级追加写 + 段目录），分 E1a–E1d 四阶段 |
| E2 | 关门后全量 re-flush | 已实施增量刷盘 | E2b：摘掉 gate 本身，需要 tracker 冻结前沿 |
| E3 | 快照全程持队列门 | 已实施（去读锁 + reflink） | E3b：摘掉 gate，依赖 E1b |
| E4 | CLI 缺 `maintenance_status` 与快照命令 | 已实施 5 个命令 | 无 |

## 验收方式与结果

工具链注意：仓库 `rust-version = "1.97"`，本机默认 `rustc` 是 1.93，直接 `cargo check` 会在 MSRV 检查处失败（不是代码问题）。下面的命令都显式指定 `+1.97.0`。

| 命令 | 结果 |
| --- | --- |
| `cargo +1.97.0 check --all-targets` | 通过，无警告 |
| `cargo +1.97.0 check --features c_api --all-targets` | 通过（C ABI 路径未受影响） |
| `cargo +1.97.0 test --lib storage::` | 182 通过（含 4 个新用例） |
| `cargo +1.97.0 test` | 424 lib + 12 个集成测试文件全部通过 |
| `cargo +1.97.0 test`（`cli/`） | 3 通过 |
| `node --check bindings/js/src/native.js` | 通过，`DEFINITIONS` 已含 `simvec_maintenance_status` |
| `python3 -m py_compile bindings/python/simvec/_native.py` | 通过 |

未执行（受环境限制）：JS 的 `require('node:ffi')` 冒烟——本机 Node 22 没有该内建模块；Python 的 `import simvec` 需要先构建 `libsimvec` 共享库。两者都只做了静态校验。

## 落地清单（按文件）

| 文件 | 改动 |
| --- | --- |
| `src/storage/payload_store/bitmask.rs` | A1：用例改为拒绝“当前版本 + 1” |
| `src/storage/wal.rs` | A2（截断错误路径）、A3（空段补长 + 快照短段补长）、A5（版本 1、magic 注释）、B2（`tail_valid_len`）、C4（注释）、`sealed_valid` 并行关系、2 个新用例 |
| `src/storage.rs` | B2（`WalCopy`/`is_wal_scratch`/`copy_prefix`）、C1（复用 `clamp`）、`hnsw_tombstones_pending` 字段、`lowest_wal_pin` 收为测试用 |
| `src/storage/snapshot.rs` | B2：`file_checksum_prefix`（`file_checksum` 变为其包装） |
| `src/storage/store_maintenance.rs` | B1（`HNSW_REPAIR_TOMBSTONE_BATCH`、`hnsw_repair_due`、修补后清零）、C2（删死代码） |
| `src/storage/store_write.rs` | B1：删除时累加待修补计数 |
| `src/storage/store_open.rs` | B1：两处构造函数补字段 |
| `src/engine/maintenance.rs` | B1：`after_mutation` 按批触发修补 |
| `src/storage/directory.rs`、`src/storage/compaction.rs` | A5：目录文件头版本 1（两处必须同步） |
| `src/storage/tombstones.rs` | C3：订正文档措辞（不加门控） |
| `src/storage/tests.rs` | T1：快照前缀用例、HNSW 批量修补用例 |
| `bindings/js/src/native.js`、`bindings/python/simvec/_native.py` | A4：补 `simvec_maintenance_status` 声明 |
| `AGENTS.md` | 段格式版本、快照复制范围、HNSW 修补时机三处事实更新 |
| `docs/plan/stage-3-wal-mmap-prealloc.md`、`docs/plan/deferred-storage-completion.md` | 版本描述统一为 1；删掉“`load` 已 `cfg(test)`”的错误结论 |
| `docs/plan/storage-implementation-audit.md`、`docs/plan/deferred-storage-refactor.md`、`docs/issue/storage-comparison-with-qdrant.md` | D1：加基线提示 |

## 后续（不在本轮）

E1（压实整代重写 + slot 全局重排）的完整分析与分阶段方案见 `docs/plan/storage-next-phase.md`（N1a–N1d），它要求把点级原地写改成追加写、引入 slot free list，属格式级切换，必须独立评审。E2b（摘掉 checkpoint 的 gate，N2）不依赖 N1；E3b（摘掉快照的 gate，N3）依赖 N1a 的“已发布段不可变”。
