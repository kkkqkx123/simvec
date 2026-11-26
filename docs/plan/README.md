# simvec 具名 / 稀疏 / 多向量改进计划

本目录记录问题清单与分阶段实施方案。方案以 `ref/qdrant` 的“点级具名向量 + 每空间类型与索引”模型为参照，但保留 simvec 的单进程、WAL + 快照、精确扫描定位。

## 问题清单

### 正确性与一致性

1. `LocalVectorEngine::apply_txn` 对 kind / named 集合调用 `apply_records`，只修改内存不写 WAL；重启后协调事务丢失。
2. `KindStore` / `NamedStore` 的 `checkpoint` 在释放内部锁后才 `wal.truncate()`，并发提交可以把已 fsync 的事务写进被截断的区间。
3. payload 修改按 slot 寻址，而覆盖写会 tombstone 旧 slot；`require_slot` 与提交分处两个临界区，payload 更新会静默丢失。
4. kind / named 集合没有压实；`Compact` 记录被忽略，slot 表和全量快照持续增长；`compact_collection` 只覆盖 dense。
5. `max_points` 只由 dense / memory 路径执行；kind / named 路径不检查。
6. wasm / memory 后端只处理 dense `vector`，对稀疏、多向量、具名载荷静默忽略，但能力声明宣称全部支持。

### 领域模型

7. 集合级 `VectorKind` 与 `VectorsConfig::Single/Named` 把“向量类型”“是否具名”“索引层级”耦合为互斥集合种类；Qdrant 中它们是正交的空间属性。
8. `VectorPoint` 同时有 `vector` / `data` / `named` 三字段，`SearchQuery` 同时有 `vector` / `data` / `target` / `named_vector`，WAL 也有三种 upsert 变体；非法组合只能靠运行时检查。
9. 具名集合不能包含多向量空间；即使所有空间都是 dense，也被强制走精确扫描，无法复用现有 HNSW/IVF。
10. 具名写入必须一次给齐全部空间；不支持按空间增量更新或删除。
11. 查询只能指定一个目标空间，没有多目标检索与融合；结果无法回传完整具名向量 map。
12. `KindStore` 与 `NamedStore` 的 lifecycle / read / write / state 基本同构，重复实现。

### 稀疏

13. `vector_size` 被当作词表上界；Qdrant 稀疏维度是开放的 `u32` 空间。
14. 输入要求索引严格递增且权重非负；Qdrant 只要求唯一，并在预处理时排序。
15. 候选集先收集全部 posting 再排序去重，然后逐点 merge 打分；缺少 posting 权重累加、TopK、过滤下推与全扫阈值选择。

### 多向量

16. `Vec<Vec<f32>>` 带来嵌套分配；Qdrant 使用 `flattened + dim + offsets`。
17. MaxSim 写死，没有 comparator 抽象；每对子向量单独调用距离内核，没有批量化或 metric 预处理。

### 接口与文档

18. C API 的 JSON 入口仍以 dense 便捷字段为主；语言绑定没有稀疏 / 具名 / 多向量 helper。
19. README 与能力矩阵仍以 dense 用法为主。

## 阶段划分

1. **持久化与一致性修复**：统一 kind/named 的提交门闩，协调事务落 WAL，payload 改写原子化，补压实与容量准入，memory 后端拒绝不支持的配置。
2. **领域模型统一**：点 / 查询 / 配置 / WAL 统一到“空间名 -> 向量”的单一 map；移除三字段歧义；具名空间允许 dense / sparse / multi；保留 FFI JSON 的便捷字段作为边界适配。
3. **稀疏检索改进**：稀疏索引归一化，接受乱序唯一索引；候选打分改为 posting 权重累加 + 每空间范数缓存，避免逐点 merge。
4. **多向量与具名查询补齐**：多向量扁平化为 `flat + dim`；引入 comparator 配置；具名空间允许 multi；多目标查询 RRF 融合；结果回传具名向量 map；按空间增量写入。
5. **接口、文档与回归对齐**：C API DTO 保持既有 JSON 便捷字段并新增统一 `vectors` 字段；README 与测试同步；集中编译与测试。

## 本轮范围说明

- dense 具名空间在本轮仍由精确扫描的 `NamedStore` 承载，ANN / 量化不跨空间启用；`validate_kind` 会明确拒绝这种配置，不会静默降级。
- 稀疏 `Idf` modifier 和 posting 压缩未纳入本轮；输入权重仍要求非负，以保证倒排候选集不会漏掉负分结果。
- MultiComparator 当前只有 `MaxSim`，配置位已就位但暂无第二种算法。

## 实施状态

五个阶段的代码改动均已完成。本地原生路径、C API JSON DTO、memory/wasm 能力拒绝、配置 / WAL / 点 / 查询类型与测试、README 均已同步；`cargo test --all-targets --features c_api` 通过。

## 实施约束

- 按项目约束在现有模块内原地重构，不新建 “v2” 替代模块。
- 不做向后兼容与旧数据迁移；配置、WAL、类型形状可直接调整。
- 编译与测试集中执行，避免在开发过程中反复全量构建。

## 后续设计评审

- `10-qdrant-design-review.md`：对照 `ref/qdrant` 的稀疏、多向量与具名实现，列出剩余问题清单与分优先级的修改方案。
