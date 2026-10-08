# simvec

[English](README.md)

`simvec` 是一个 Rust 本地向量搜索引擎库：提供精确扫描（exact scan）存储与
检索、可选的 ANN 索引、WAL 持久化以及基于 mmap 的向量存储——全部运行在
单进程内，不依赖任何外部服务。

## 特性

- **默认精确扫描，数据量大时自动升级 ANN。** 集合初始为暴力扫描，当存活
  点数超过 `full_scan_threshold`（默认 10 000，与 Qdrant 默认值一致）时
  自动提升为已发布索引。
- **HNSW 与 IVF-Flat 索引。** HNSW 支持并发构建
  （`max_indexing_threads`）、迭代式过滤扫描扩展、基于陈旧度的重建以及
  pending 槽位排空。IVF 支持自动 `lists` 选择、逐查询 `nprobe`、基于漂移
  的重建以及从精确扫描的提升。
- **WAL 持久化。** 每次变更先追加写 WAL 并 fsync，然后才应用到内存；重放
  是幂等的，崩溃恢复可以重复应用事务而不会重复写入数据。协调式事务由
  `LocalVectorEngine::apply_txn` 处理。
- **两阶段发布模拟。** 索引构建写入 `*.rebuild-*` 临时目录，原子发布，并
  保留 `.old-*` 备份用于反向恢复（`restore_promote_backup`）。
- **Payload 存储与过滤。** 完整 JSON payload、可选的 payload 字段索引，
  以及丰富的过滤语言：match、range、values-count、地理（半径 / 矩形）、
  嵌套条件，以及 `must` / `should` / `must_not` 组合器。
- **SIMD 距离内核。** x86-64 上支持 AVX2+FMA 和 AVX-512，aarch64 上支持
  NEON，另有一个始终作为正确性基线的朴素实现。支持 Cosine、Euclid、Dot、
  Manhattan 四种度量。
- **量化。** 标量量化和乘积量化（PQ，k-means 训练码本），仅作为检索加速
  使用（精确分数总是重新计算）。
- **墓碑与压实（compaction）。** 删除是逻辑删除；`compact_collection`
  物理回收槽位。后台维护线程每 30 秒执行一次索引构建、压实、漂移检查和
  HNSW 提升。
- **指标。** 每个 `MetricsSnapshot` 包含检索路径计数器（扫描 / IVF /
  HNSW、重试、回退）。可选的 `lock-metrics` 特性用于锁竞争排查的等待时间
  埋点。

## 用法

```rust
use simvec::{
    CollectionConfig, DistanceMetric, LocalVectorEngine, SearchQuery, VectorPoint,
};

// 打开（或创建）以某目录为根的引擎。每个集合是一个子目录，
// 打开时会加载已存在的集合。
let engine = LocalVectorEngine::open("./data")?;

// 创建集合：384 维 cosine，默认 HNSW 层级。
let config = CollectionConfig::new(384, DistanceMetric::Cosine);
engine.create_collection("docs", &config)?;

// 写入数据（WAL 支持：先 fsync 再应用到内存）。
let point = VectorPoint::new("doc-1", vec![0.1; 384])
    .with_payload_kv("title", "hello".into());
engine.upsert("docs", point)?;

// 检索。分数遵循 crate 级契约：越大越相似。
let query = SearchQuery::new(vec![0.1; 384], 10);
let results = engine.search("docs", &query)?;
for r in &results {
    println!("{} -> {}", r.id, r.score);
}
```

过滤、kNN 模式、范围查询、分页和 payload 投影均通过 `SearchQuery` 的
构建器完成：

```rust
use simvec::{FilterCondition, SearchQuery, VectorFilter};

let filter = VectorFilter::new().must(FilterCondition::match_value("color", "red"));
let query = SearchQuery::new(vec![0.1; 384], 10)
    .with_knn(10, Some(64))                       // kNN，显式指定 ef_search
    .with_filter(filter)
    .with_score_threshold(0.5)
    .with_payload_include(vec!["title".into()]);
```

## 架构

| 模块 | 职责 |
|------|------|
| `engine` | `LocalVectorEngine`：集合注册表、WAL 支持的变更、后台维护线程、发布 / 提升 / 恢复 |
| `storage` | `CollectionStore`：单个集合目录——向量（mmap）、payload、WAL、墓碑、元数据、payload 索引、量化、检索 |
| `index` | HNSW 图、IVF（k-means）以及索引持久化 |
| `distance` | 距离内核（AVX2 / AVX-512 / NEON / naive）与分数契约 |
| `filter` / `filter_cond` | 过滤求值与条件类型 |
| `metrics` | `MetricsSnapshot` 与检索路径埋点 |
| `types` | 共享类型层（远程 Qdrant 后端同样使用） |

### 存储布局

每个集合是引擎根目录下的一个子目录，包含 `vectors.bin`（mmap 的原始 f32
行）、WAL 段文件、payload 存储、元数据和索引产物。删除以墓碑表示；压实
会重写整个目录。重建临时目录（`*.rebuild-*`）、发布备份（`*.old-*`）和
隔离目录（`.suspect-*`）永远不会被加载进服务注册表——它们仅用于崩溃恢复
和排查。

### 分数契约

引擎内部以距离排序（越小越近），输出时转换为相似度分数。分数是公开契约：
在所有度量和后端上都是越大越好，`score_threshold` 是其下界。

| 度量 | 内部距离 | 输出分数 |
|------|----------|----------|
| Euclid | `Σ(a-b)²`（平方，不开方） | `1/(1+sqrt(d²))` |
| Dot | `-Σ(a·b)` | `Σ(a·b)` |
| Cosine | `1 - similarity`，截断到 `[-1, 1]` | `similarity` |
| Manhattan | `Σ\|a-b\|` | `1/(1+sqrt(d))` |

Cosine 在计算时实时求范数，而不是写入时归一化：`vectors.bin` 是数据的
唯一副本，`get()` / `with_vector` 必须返回原始字节。

## Feature 开关

| Feature | 默认 | 说明 |
|---------|------|------|
| `lock-metrics` | 关闭 | 编译进邻接表 / 链表写锁等待埋点；指标经 `MetricsSnapshot` 输出 |
| `simd_portable` | 关闭 | 可移植 `std::simd` 内核（stable Rust 尚不可用）；目前作为占位实现委托给 `naive`，待 `std::simd` 稳定 |

## 测试与基准

```sh
cargo test -p simvec          # 单元测试 + 集成测试
cargo bench -p simvec         # criterion 基准
```

基准覆盖向量扫描、IVF、HNSW 构建与写入、并发检索、分配统计以及持久化
CRC 开销。

## 许可证

Apache-2.0
