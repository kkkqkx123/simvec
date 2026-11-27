# 阶段八细化方案：增量快照与接口暴露

对照 `docs/plan/storage-gap-remediation.md` 阶段八与问题清单中“快照是全量复制”一节。
本文只描述职责、状态转换和验证边界，不展开完整实现代码。

## 对照 qdrant

qdrant 的 segment manifest 记录文件版本与长度，快照按版本差异复制，并用 WAL ACK pin
保护复制窗口。simvec 阶段五已有 `WalAckPins`，`snapshot_to` 在全量复制期间持 maintenance
锁、queue gate、store 读锁并 pin WAL；但每次都复制整个 generation 和 WAL，复制窗口与
集合大小成正比。

本轮把“按文件版本判断变化”加入现有快照包格式，而不是引入外部快照链：新快照仍是
一个完整、可独立校验和恢复的目录，只是未变化文件通过硬链接复用已验证的基础快照。

## 职责划分

- 文件版本：`SnapshotFile` 记录相对路径、generation 版本、长度和 CRC。四者全部一致
  才认为文件未变化；同长度的内容改写因 CRC 不同会被重新复制。
- 清单：`SnapshotInfo` 记录 generation、lsn、`reused_files` 和全部文件条目。
  `verify_snapshot` 仍逐文件校验长度、CRC 和路径不逃逸；复用文件与复制文件在校验上
  没有区别。
- 增量捕获：`CollectionStore::snapshot_to_incremental(base, dest)` 先验证基础快照，
  再按 `snapshot_to` 相同的锁与 pin 顺序捕获当前视图。命中基础条目的文件先硬链接，
  跨文件系统链接失败时退化为普通复制。所有文件都写入新清单，恢复不依赖基础快照。
- 复制顺序：读取每个源文件一次算长度和 CRC，决定复用或复制；generation 目录、WAL
  和根清单仍在同一 pin 窗口内捕获，generation 与 lsn 保持一致。
- 恢复：`restore_snapshot` 校验后再复制到新目录并以 `CollectionStore::open` 验证。
  硬链接复用不改变恢复语义；目标仍必须不存在。

## 状态转换

- 全量快照 → 清单中 `reused_files = 0`。
- 增量快照 → 对比基础清单 → 相同版本/长度/CRC 的文件硬链接，其余复制 → 写新清单。
- 基础清单损坏、路径逃逸或目标已存在 → 在复制前失败，不产生半份快照。
- 恢复 → 校验清单 → 复制全部条目 → open 验证 generation/WAL 绑定。

## 不变量

- 快照内所有文件属于同一个 generation 视图；WAL pin 覆盖整个捕获窗口。
- 清单中的每条路径都通过 `checked_path`，长度与 CRC 都通过校验。
- 复用只发生在内容完全一致时；快照文件视为不可变，恢复写出独立副本。
- 恢复后的目录必须通过正常 `CollectionStore::open` 校验。

## 公开接口

- `CollectionStore::snapshot_to` 与 `CollectionStore::snapshot_to_incremental` 保持
  Rust 公开 API；`LocalVectorEngine` 增加两个同名语义的便捷方法，仅服务 dense 集合。
- `restore_snapshot` 从 crate 根重新导出；`SnapshotInfo` 与 `verify_snapshot` 继续由
  `simvec::storage` 暴露。
- C ABI 与 `include/simvec.h` 不变；快照暂不进入语言绑定矩阵。

## 验收

- 无写入的第二次快照复用全部文件；写入后的增量快照只复制变化文件。
- 增量快照可独立恢复到新目录，搜索与 scroll 结果与写入前一致。
- 位翻转、路径穿越、缺文件、版本不匹配和已存在的目标都被拒绝。
- 快照期间的 WAL 段在 pin 释放前不被截断。
