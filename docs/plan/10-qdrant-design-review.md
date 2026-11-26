# 对照 Qdrant 的设计评审与修改方案

本文对照 `ref/qdrant` 中稀疏、多向量与具名向量的实现，评审当前设计，给出问题清单与原地修改方案。

当前方向是正确的：点、查询与 WAL 已统一到“空间名 -> 向量”的单一 map，稀疏维度开放，多向量扁平存储，多目标查询支持融合。但实现仍是三套并行存储加集合级类型互斥，与 Qdrant 的“一个集合/segment + 每空间独立参数与索引”模型存在结构性差距。

## 对照结论

| 方面 | 当前实现 | Qdrant 参考实现 | 判断 |
|---|---|---|---|
| 配置模型 | `VectorKind` + 必填 `vector_size` + 集合级 `kind`，与权威 `vectors` map 并存 | `VectorNameConfig` 区分 dense 与 sparse；multi 是 dense 空间的一个配置位 | 类型、是否具名、索引层级仍耦合在集合上；稀疏用伪维度占位 |
| 存储架构 | `CollectionStore` / `KindStore` / `NamedStore` 三路并行，具名强制 Flat 并拒绝量化 | 同一 segment 内每空间独立 vector storage 与 index，具名 dense 可用 HNSW/量化 | 具名 dense 空间无法复用现有 HNSW/IVF，是最大的能力缺口 |
| 稀疏检索 | posting 累加后全量候选排序；过滤先全扫构造允许槽集合 | `SearchContext` 在线 TopK、批量推进、尾部 posting 快路径、惰性过滤、低基数走全扫 | 正确但未优化的参考级实现 |
| 多向量 | `MultiVector { flat, dim }`，MaxSim 写死，比较器配置位不生效 | `TypedMultiDenseVector` + 按 metric 特化的 `score_max_similarity` + `MultiVectorConfig` | 扁平化做对了，构造与配置不完整 |
| 具名语义 | 写入必须全量空间，删除只能整点 | 支持部分向量、按名删除、`HasVector` 条件 | 不变式更安全，但无法增量加空间或按空间清理 |
| 融合 | `NamedStore` 内部硬编码 RRF k=60，无权重 | `rrf_scoring` 默认 k=2、支持每路权重，另有 DBSF 与显式 Fusion 查询 | 融合应属于查询层，而非具名 store 的内部逻辑 |

## 问题清单

### 正确性与一致性

1. **校验发生在 WAL 落盘之后**。`src/kind_store/write.rs` 与 `src/named_store/write.rs` 的 `commit_records_locked` 先追加事务，再由 `apply_upsert` / `apply_patch` 校验维度、空间名、容量与 payload JSON。非法请求会给调用方返回错误，但事务已持久化；重启 replay 再次触发同一错误，集合无法打开。
2. **分数契约不一致**。`src/distance.rs` 的 `to_score` 对 Euclid / Manhattan 使用 `sqrt` 后的距离，而 `src/kind_store/score.rs` 的 `dense_similarity` 与 `max_sim` 使用未开方的内部距离。dense 集合、具名 dense 空间与 multi 空间在同一度量下的分数不可比较，`score_threshold` 语义随 store 变化。
3. **稀疏空查询与零分查询语义偏离**。`VectorData::validate` 拒绝空 indices，与本目录稀疏收尾文档的验收口径冲突；权重全为零的 Dot 查询会把共享词项的点以零分返回，因为累加不跳过零贡献候选。Qdrant 在空向量时直接返回空结果，并跳过零分候选。

### 领域模型

4. **集合级类型与三 store 并行**。具名集合被强制为 Flat 并拒绝量化，稀疏与多向量被限定在 `KindStore`，不能与 dense 空间共存，也无法复用 ANN 路径。Qdrant 中空间类型与索引层级是正交属性。
5. **配置双源与伪维度**。`vector_size` 对稀疏空间没有意义，却用 `SPARSE_DIM_BOUND` 占位；`with_sparse` 忽略传入的词表上界；集合级便捷字段与 `vectors` map 同时存在，`collection_config` 回填时二者可能不一致。
6. **具名不变式过强且融合不可配置**。全量 map 使新增空间与按空间删除不可行，检索侧依赖 `expect` 保证每个空间都存在。融合逻辑内嵌于 `NamedStore::search_fused`，RRF 常数固定为 60，无权重与替代算法。

### 检索

7. **过滤下推是伪下推**。`KindStore::search` 与 `NamedStore::search` 先全扫所有槽构造允许集合，再累加 posting；同时精确扫描集合被禁止建立 payload index，没有基数估计，过滤查询成本恒为全表规模。
8. **TopK 物化全部候选，无代价阈值**。`sparse_candidate_scores` 返回所有命中槽后统一排序，`truncate_topk` 只延迟排序而不减少累加量。Qdrant 用在线 TopK、阈值剪枝与基数选择决定全扫还是倒排；当前文档称“无全局统计支撑阈值”，但 posting 长度本身就是可用的词项频率统计。
9. **多向量构造与配置不完整**。`MultiVector::new` 不校验子向量等长与维度为零，`vectors()` 用 `dim.max(1)` 掩盖非法状态；`MultiComparator` 配置位不参与评分；MaxSim 每对子向量单独走一次动态内核分派。

## 修改方案

### 第一优先级：持久化正确性与分数统一

- 把形状校验、容量准入和 payload 编码前移到 WAL 追加之前，追加之后只做纯状态应用；replay 不再重复校验，非法记录不可能进入日志。
- 精确扫描 store 统一调用 `distance::to_score`，删除 `src/kind_store/score.rs` 内的局部距离映射；multi 的内层相似度也走同一映射。为跨 store 的同度量分数一致性补回归。
- 稀疏空查询返回空结果，零贡献候选不进入结果；评估放开负权重——当前没有剪枝逻辑，非负限制不再必要。同步修正文档与测试。

### 第二优先级：配置与空间模型收敛

- 在现有 `src/types/config/vectors.rs` 内把 `VectorSpaceConfig` 改为 dense 与 sparse 两种枚举形态：dense 携带维度、距离与可选 multi 配置；sparse 只携带 modifier 与权重类型，不再声明维度。取消集合级 `kind` 与单空间镜像字段，便捷字段只作为 FFI JSON 边界适配。
- 统一以 `vectors` map 为唯一权威，修正 `collection_config` 与 info 的回填，明确 `with_sparse` 的参数语义。
- 若保持具名全量映射不变式，提供显式的空间演进入口（重写全部点）并定义检索侧缺失空间策略；若要支持部分向量，则同步放开按空间删除并移除 `expect` 依赖。

### 第三优先级：稀疏检索路径

- 过滤改为对 posting 命中惰性求值，按槽缓存布尔结果，避免预先全扫。让精确扫描 store 复用 `storage/payload_index`，为路径选择提供基数估计。
- 用 posting 总长度与存活点数做粗代价比较，决定倒排累加还是预筛全扫；累加阶段使用可复用的按槽分数缓冲和固定容量堆，结合 `score_threshold` 在线剪枝。
- 后续按需要引入 IDF 与 posting 压缩，保持与零分、负权重新语义一致。

### 第四优先级：多向量与具名查询

- 为 `MultiVector` 增加可失败的构造入口，在入口拒绝 ragged 输入与零维度，移除 `dim.max(1)` 的掩盖行为。
- 把 `MultiComparator` 真正接入评分分派；若短期没有第二种算法，则移除该配置位，避免无效旋钮。MaxSim 内层按 metric 特化并批量化。
- 融合逻辑上移到查询层，配置化为 RRF（含权重）与 DBSF；`with_vector` 支持按空间选择；明确每路目标的分页、阈值与最终分页之间的关系。

### 第五优先级：引擎分派收敛

- 在现有模块内实现统一的空间索引 trait 与句柄枚举，用一个集合名到句柄的注册表替代三个 map，使读写入口只分派一次。不新建替代模块，三个 store 在各自原模块内实现统一接口。

## 明确不做（本轮取舍）

- 不照搬 Qdrant 的 segment optimizer、mmap / 压缩倒排与跨分片 WAL 顺序模型；继续保持单进程 WAL + 快照 + 精确扫描的定位。
- 后台维护仍只服务 dense 集合；稀疏与具名维持手动压实；memory 与 wasm 后端继续拒绝非 dense 配置与查询。
- 不引入按空间删除，除非第二优先级决定放宽全量映射不变式。

## 验收口径

- 非法 upsert 与 patch 不写入 WAL，重启后集合可正常打开。
- 同一度量下 dense、具名 dense 与 multi 的分数可比较；空稀疏与零权查询返回空结果。
- 带过滤的稀疏检索与“不过滤检索后再过滤”结果一致，且不再恒付全表扫描代价。
- 非法 multi 在构造期被拒绝；配置项均实际生效；融合参数可配置。
- 引擎读写入口不再按 store 类型重复分支。
