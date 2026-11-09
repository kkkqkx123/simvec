# simvec

[English](README.md)

`simvec` 是一个 Rust 本地向量搜索引擎库：以 HNSW 为默认 ANN 索引、精确扫描为兜底，WAL 持久化以及基于 mmap 的向量存储——全部运行在单进程内，不依赖任何外部服务。

## 特性

- **默认 HNSW，晋升前走精确扫描。** `CollectionConfig::new` 创建的集合默认使用 HNSW 层级；在存活点数超过 `full_scan_threshold`（默认 10 000，与 Qdrant 默认值一致）之前走暴力精确扫描，之后自动提升为已发布的 HNSW 图。`FLAT` 集合则永久停留在精确扫描。
- **HNSW 索引，另有可选的 IVF-Flat。** HNSW 支持并发构建（`max_indexing_threads`）、迭代式过滤扫描扩展、基于陈旧度的重建以及 pending 槽位排空。IVF-Flat 是面向大批量集合的显式可选方案（默认关闭 `auto_promotion`）：自动 `lists` 选择、逐查询 `nprobe`、基于漂移的重建。
- **WAL 持久化。** 每次变更先追加写 WAL 并 fsync，然后才应用到内存；重放是幂等的，崩溃恢复可以重复应用事务而不会重复写入数据。协调式事务由 `LocalVectorEngine::apply_txn` 处理。
- **两阶段发布模拟。** 索引构建写入 `*.rebuild-*` 临时目录，原子发布，并保留 `.old-*` 备份用于反向恢复（`restore_promote_backup`）。
- **Payload 存储与过滤。** 完整 JSON payload、可选的 payload 字段索引，以及丰富的过滤语言：match、range、values-count、地理（半径 / 矩形）、嵌套条件，以及 `must` / `should` / `must_not` 组合器。
- **SIMD 距离内核。** x86-64 上支持 AVX2+FMA 和 AVX-512，aarch64 上支持 NEON，另有一个始终作为正确性基线的朴素实现。支持 Cosine、Euclid、Dot、Manhattan 四种度量。
- **量化。** 标量量化和乘积量化（PQ，k-means 训练码本），仅作为检索加速使用（精确分数总是重新计算）。
- **墓碑与压实（compaction）。** 删除是逻辑删除；`compact_collection` 物理回收槽位。后台维护线程每 30 秒执行一次索引构建、压实、漂移检查和 HNSW 提升。
- **指标。** 每个 `MetricsSnapshot` 包含检索路径计数器（扫描 / IVF / HNSW、重试、回退）。可选的 `lock-metrics` 特性用于锁竞争排查的等待时间埋点。

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

过滤、kNN 模式、范围查询、分页和 payload 投影均通过 `SearchQuery` 的构建器完成：

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

每个集合是引擎根目录下的一个子目录，包含 `vectors.bin`（mmap 的原始 f32 行）、WAL 段文件、payload 存储、元数据和索引产物。删除以墓碑表示；压实会重写整个目录。重建临时目录（`*.rebuild-*`）、发布备份（`*.old-*`）和隔离目录（`.suspect-*`）永远不会被加载进服务注册表——它们仅用于崩溃恢复和排查。

### 分数契约

引擎内部以距离排序（越小越近），输出时转换为相似度分数。分数是公开契约：在所有度量和后端上都是越大越好，`score_threshold` 是其下界。

| 度量 | 内部距离 | 输出分数 |
|------|----------|----------|
| Euclid | `Σ(a-b)²`（平方，不开方） | `1/(1+sqrt(d²))` |
| Dot | `-Σ(a·b)` | `Σ(a·b)` |
| Cosine | `1 - similarity`，截断到 `[-1, 1]` | `similarity` |
| Manhattan | `Σ\|a-b\|` | `1/(1+sqrt(d))` |

Cosine 在计算时实时求范数，而不是写入时归一化：`vectors.bin` 是数据的唯一副本，`get()` / `with_vector` 必须返回原始字节。

## Feature 开关

| Feature | 默认 | 说明 |
|---------|------|------|
| `lock-metrics` | 关闭 | 编译进邻接表 / 链表写锁等待埋点；指标经 `MetricsSnapshot` 输出 |
| `simd_portable` | 关闭 | 可移植 `std::simd` 内核（stable Rust 尚不可用）；目前作为占位实现委托给 `naive`，待 `std::simd` 稳定 |
| `wasm` | 关闭 | 浏览器内存引擎（`bindings/wasm`，经 wasm-bindgen）；原生路径不涉及存储、索引或线程变更 |

## C API 及语言绑定

`c_api` cargo 特性在同一引擎之上暴露稳定的 C ABI：所有入口返回整型状态码，入参一律借用，分配的出参统一用 `simvec_free_string` 释放；配置、查询、过滤器、payload 与检索结果以 JSON 穿越 FFI，直接复用 Rust 侧的 serde 结构。手写的 `include/simvec.h` 与 cbindgen 输出保持对齐（`build.rs` 在 cbindgen 可用时重生成）；用 `cargo build --features c_api` 构建，默认构建仍是纯 Rust rlib。

`bindings/python` 是该 C API 上的纯 Python ctypes 封装，不含任何原生代码：先构建 Rust 动态库，再通过 `SIMVEC_LIB` 指向它（否则回退到 checkout 旁的 `target/debug` / `target/release`）。配置、过滤器、payload 与查询都是普通 dict，向量接受任意浮点序列并以连续 float32 数组传递，`Engine` 把非零状态码映射为类型化异常。在 `bindings/python` 下用 `SIMVEC_LIB=/path/to/libsimvec.so python -m pytest tests -q` 运行烟囱测试。

`bindings/js` 是同构的零依赖 JavaScript 封装：经 Node.js 内建 `node:ffi` 模块调用 C API（Node 22+，需 `--experimental-ffi`），同样经 `SIMVEC_LIB` 解析 `libsimvec` 并回退到 `target/debug` / `target/release`，对外提供驼峰命名的 `Engine` 类与类型化错误。在 `bindings/js` 下用 `SIMVEC_LIB=/path/to/libsimvec.so node --experimental-ffi --test tests/` 运行烟囱测试。

`bindings/go` 是同一 C API 上的薄 cgo 封装，不含原生代码：先构建 Rust 动态库，再针对它构建或测试 Go 模块（需要启用 cgo 的 C 工具链）。配置、过滤器、payload 与查询为可序列化的普通 map 或值，编码为 Rust 侧已理解的 JSON 形状；向量以 `[]float32` 切片作为连续数组穿越边界，`[]float64` 调用者可用 `Float32s` 辅助转换（窄化转换）；点标识为字符串。方法名按 Go 导出的驼峰命名并一对一映射到 C API；七个状态码返回包装 `SimvecError` 的类型化错误，其余返回保留数字码的基类。`Engine` 实例可并发使用，但 `Close` 不得与进行中的调用并发执行。构建与测试：

```sh
cargo build --features c_api
cd bindings/go && go test ./...
```

针对自定义库位置，改用 `CGO_CFLAGS`、`CGO_LDFLAGS` 或 `LD_LIBRARY_PATH`，而非依赖默认搜索路径。

`bindings/java` 是同一 C API 上的 JNI 封装，不含原生代码：先构建 Rust 动态库与 JNI 桥接，再针对它们构建或测试 Java 包（桥接需要 C 工具链与 JDK 头文件）。加载器按顺序检查 `simvec.jni.lib` 系统属性、`SIMVEC_JNI_LIB` 环境变量、工作区根下的 `target/debug` 与 `target/release` 输出，最后是 `java.library.path`；当 `SIMVEC_LIB` 指定显式 Rust 库路径时，该文件会被预加载，使桥接针对它解析。配置、过滤器、payload 与查询为普通 map 或 list，编码为 Rust 侧已理解的 JSON 形状；向量以 `float[]` 切片作为连续数组穿越边界，另有 `double[]` 重载以拷贝方式窄化转换；点标识为字符串。方法名按 Java 驼峰命名并一对一映射到 C API；七个状态码抛出 `SimvecError` 的类型化子类，其余抛出基类并保留数字码（可通过 `getCode()` 访问）。`Engine` 实例可并发使用（所有方法均同步）。构建与测试：

```sh
cargo build --features c_api
sh bindings/java/scripts/build-native.sh
cd bindings/java && mvn test
```

`bindings/r` 是同一 C API 上的 R 包，不含原生代码：先构建 Rust 动态库，再针对它安装 R 包（需要 C 工具链）。`configure` 优先检查 `SIMVEC_LIB` 环境变量（显式命名库文件），其次检查工作区根下的 `target/debug` 与 `target/release` 输出。配置、过滤器、payload 与查询为普通 list，经 jsonlite 序列化为 Rust 侧已理解的 JSON 形状；向量为数值向量，由桥接层拷贝转换为连续单精度数组（窄化转换）；点标识为字符串；计数以双精度穿越。函数名为蛇形 `simvec_*`，一对一映射到 C API；七个状态码发出携带 `simvec_error` 基类的类型化条件，其余发出基类并保留数字码（可通过 `code` 字段访问）。引擎为持有单个外部指针的环境对象。构建、安装与烟囱测试：

```sh
cargo build --features c_api
R CMD INSTALL bindings/r
Rscript bindings/r/tests/smoke.R
```

`bindings/wasm` 是同一 crate 在 `wasm` 特性下构建的浏览器内存引擎：纯 WebAssembly，无原生动态库与文件系统，因此集合仅存活于页面会话，除非调用者持久化快照。检索为单线程线性精确扫描；索引构建、量化、压实与维护为语义兼容的空操作，始终报告未生效；发布、备份与引擎默认配置分组不在此暴露。在 `bindings/wasm` 下运行 `npm run build` 构建（Rust 侧需要安装 `wasm32-unknown-unknown` 目标）；npm 包仅发布纯 JS，`pkg/` 由调用者构建。持久化通过 `exportSnapshot` / `importSnapshot` 显式完成，可选的 `persist.js` 辅助覆盖 OPFS 写入（支持时经临时文件加读回验证与移动，否则回退到经验证的直接覆盖，并保留上一份正常快照为 `.bak`）、存在性检查、清理、配额、持久许可与带页面隐藏刷盘的防抖自动保存。构建后用 `node tests/smoke.mjs` 运行烟囱流程。基本用法：

```js
const engine = await Engine.create();
engine.createCollection("docs", { vector_size: 2, distance: "Dot" });
engine.upsert("docs", "a", [1, 0], { color: "red" });
const hits = engine.searchVector("docs", [1, 0], 10);
engine.close();
```

## 测试与基准

```sh
cargo test -p simvec          # 单元测试 + 集成测试
cargo bench -p simvec         # criterion 基准
```

基准覆盖向量扫描、IVF、HNSW 构建与写入、并发检索、分配统计以及持久化
CRC 开销。

## 许可证

Apache-2.0
