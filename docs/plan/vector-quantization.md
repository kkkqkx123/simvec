# 向量量化设计方案（对照 Qdrant）

本文对照 `ref/qdrant` 的量化实现，梳理 simvec 当前的量化实现，并给出后续演进设计。

- Qdrant 参考：`ref/qdrant/lib/segment/src/vector_storage/quantized/`、`ref/qdrant/lib/quantization/`、`ref/qdrant/lib/segment/src/types.rs`
- 本项目现状：`src/storage/quant.rs`、`src/storage/quant/codec.rs`、`src/storage/quant/rebuild.rs`、`src/types/config.rs`、`src/storage/search/quantized.rs`

## 1. Qdrant 的量化实现

### 1.1 总体架构

Qdrant 的量化是**按 segment 构建的、不可变的派生结构**，与 HNSW 同属 segment 优化阶段产物：

```
QuantizationConfig (types.rs: L1162)
  ├─ Scalar  (ScalarQuantizationConfig, 仅 Int8)
  ├─ Product (ProductQuantizationConfig, CompressionRatio X4..X64)
  ├─ Binary  (BinaryQuantizationConfig, OneBit/TwoBits/OneAndHalfBits + query_encoding)
  └─ Turbo   (TurboQuantQuantizationConfig, 1/1.5/2/4/8 bits，本 checkout 新增)
        │
        ▼
QuantizedVectors (vector_storage/quantized/quantized_vectors/)
  codec (lib/quantization/src/encoded_vectors_{pq,u8,binary,tq}.rs)
  × code backend {Ram, Mmap, ChunkedMmap}
```

关键文件：

| 关注点 | 文件 | 要点 |
|---|---|---|
| 存储/后端决策 | `quantized_vectors/config.rs` | `QuantizedStorageKind`：`method × {Ram,Mmap,ChunkedMmap}`，标量/PQ 禁止 Mutable（appendable），Binary/Turbo 支持 chunked 追加 |
| 构建入口 | `quantized_vectors/create.rs` | `SegmentBuilder::build` 时对非删除向量（`calibration_mask` 排除 deleted 占位向量）统一训练+编码；多向量走 offsets 表 |
| PQ codec | `lib/quantization/src/encoded_vectors_pq.rs` | 每子空间 256 质心、1 byte/子段；采样 `KMEANS_SAMPLE_SIZE=10k`、`MAX_ITERATIONS=100`、accuracy 1e-5；查询走 LUT（ADC） |
| Scalar codec | `lib/quantization/src/encoded_vectors_u8.rs` | u8 编码（实际 0..127，7bit）；P² 流式分位数估计（`p_square.rs`）裁剪离群值；每个向量前置 4 字节修正常量，把整数 SIMD 打分经仿射 `multiplier*score + offsets` 还原到原度量；AVX512-VNNI/AVX/NEON 多套内核 |
| Binary codec | `lib/quantization/src/encoded_vectors_binary.rs` | 1bit=符号；2bit=按维度 z-score(σ=2/3) 三档；1.5bit=2bit 变体（维度对共享位）；查询可选非对称 4/8bit 标量量化并做**位转置**，用 AND-popcount 批量打分 |
| 打分器 | `quantized_query_scorer.rs`、`quantized_scorer_builder.rs` | 查询预处量（cosine 归一化等）→ `encode_query` 一次 → 逐码打分；Recommend/Discover/Context 走 `QuantizedCustomQueryScorer` |
| HNSW 集成 | `index/hnsw_index/`（read_view、point_scorer.rs）+ `index/vector_index_search_common.rs` | 图遍历全程用量化码打分；点对点内部打分优先量化、失败回退原向量；`get_oversampled_top`（oversampling × top）→ `postprocess_search_result` 用原向量 rescore |
| 生命周期 | `segment_constructor/segment_builder.rs` L913、`old_index.rs` | 随 segment 优化构建；量化配置任何变更（`mismatch_requires_rebuild`）触发重建并阻止 HNSW 增量复用；构建路径可 `stopped` 中止 |
| 搜索参数 | `types.rs` `QuantizationSearchParams` | `ignore` / `rescore: Option<bool>` / `oversampling: Option<f64>`；rescore 默认值按方法取：**Binary=true，Scalar/PQ=false** |
| 图内联 | `HnswConfig.inline_storage` | 可把原始+量化向量副本内联进 HNSW 图文件，减少随机寻道 |

### 1.2 设计要点

1. **不可变派生产物**：Scalar/PQ 量化文件在 segment 重建（optimization）时一次性生成，之后只读；更新通过 segment 替换而非原地修改。仅 Binary/Turbo 经 feature flag 支持 appendable chunked 存储（`upsert_vector` + live reload）。
2. **训练只用活数据**：deleted 向量不参与分位数/码本/统计量拟合（cosine 场景下占位向量非单位长度会污染拟合）。
3. **打分即还原**：标量码携带修正常量、PQ 查询预计算 LUT、二值用 XOR/AND-popcount——都用整数/位运算 SIMD 内核，把量化距离仿射回原度量。
4. **召回可调**：oversampling 扩大粗筛候选，rescore 用原向量精排；两者是搜索期参数，与量化类型解耦。
5. **成本计量**：codec 层记 `HwMetric`（冷存储 IO、CPU），支撑基于代价的查询计划。

## 2. simvec 当前实现

### 2.1 架构

simvec 的量化是**collection 级、可重建的派生结构**（`quant.bin` + `quant_meta.bin`，见 `src/storage/quant.rs` 模块注释）：

- 原始向量始终是权威数据（`vectors.bin` f32），`quant.bin` 随时可从其重建——与 ANN 索引同地位。
- `QuantStore`：分段 mmap（`segment_slots` 段、`ArcSwap<Vec<Arc<Mmap>>>` 快照读）、`grow_to` 就地扩容、`write_slot` 在 commit 应用时逐槽编码写入。meta 走 postcard+CRC+原子 rename（`save_meta`/`read_tagged`），损坏或与 collection 元数据不一致时自动丢弃、回退精确检索。
- 三种类型（`src/types/config.rs` L232）：
  - **Scalar**：全局 min/max + 分位数裁剪（`quantile`，默认 0.99，`rebuild.rs` L24 排序取双侧分位），u8 线性映射。
  - **Binary**：符号位打包（`codec.rs` L28），无训练。
  - **Product**：按子空间用 `crate::index::kmeans` 训练 256 质心（max_iter=20），codebook 平铺存于 meta；未训练（`ready=false`）时 upsert 跳过、搜索回退精确路径。
- 编码/打分（`codec.rs`）：`encode` 逐类型编码；`distance_quantized` 逐槽调用——Scalar 反量化后朴素循环算距离，Binary 每槽先编码查询位再 popcount（查询位已预计算传入），PQ 逐子空间找质心算 L2/Dot/Cosine/L1（未做 LUT）。
- 搜索集成（`src/storage/search/quantized.rs`）：量化就绪时全量粗筛 `k × oversample`（Scalar 2x/Binary 4x/PQ 3x 硬编码），命中后用原向量重排。与 HNSW 路径（`search/ann.rs`）**相互独立**——量化只作用于精确扫描路径。
- 生命周期：`store_open.rs` 建/开、`store_maintenance.rs` 的 `build_quantization` 训练重建、compaction 时 `write_compacted_file`/`replace_from` 按 slot 映射搬移量化码。

### 2.2 与 Qdrant 的差距

| 维度 | Qdrant | simvec 现状 | 差距评估 |
|---|---|---|---|
| 打分性能 | 整数 SIMD / LUT / popcount 内核，O(1) 查表 | `distance_quantized` 逐槽朴素循环，每次 `meta.read()` | **主要瓶颈**，最大收益点 |
| PQ 查询 | 预计算 `chunks×256` LUT，每槽 m 次查表 | 每槽每子空间直接访问 codebook 计算距离 | 应改 LUT（含 Dot 的 `||q||²−2·q·c+c·c` 化简） |
| 查询编码 | `encode_query` 一次，之后逐槽 O(码长) | Scalar/PQ 逐槽重算；Binary 查询位已预计算 | 统一 `encode_query` 预计算 |
| 训练采样 | 固定 10k 采样，100 iter，可中止 | 全量参与训练，20 iter，不可中止 | 大 collection 训练成本高 |
| 分位数估计 | P² 流式单遍 | 全量收集+排序 O(n log n) 内存/时间 | 改流式或 reservoir 采样 |
| rescore/oversampling 配置 | 搜索期参数 `QuantizationSearchParams`，rescore 默认按类型 | oversample 硬编码，rescore 无条件执行 | 增加配置面 |
| Appendable 量化 | Binary/Turbo chunked 追加 + live reload | 不支持：所有类型靠 `write_slot` 逐槽写，PQ 未训练时静默缺码 | 现有逐槽写模型已天然"追加"，缺 PQ 就绪前的正确性处理需复核 |
| HNSW 集成 | 图遍历用量化码打分 + 点对点内部打分回退 | 量化只加速精确扫描，HNSW 遍历仍打原向量 | 中期方向 |
| 内联图存储 | `inline_storage` 原始+量化副本进图文件 | 无 | 长期方向 |
| 多向量/sparse | 多向量 offsets、sparse 独立路径 | 无 | 按需 |
| 成本计量 | HwMetric | 无 | 低优先级 |

## 3. 设计方案

按收益/成本排序分四阶段。核心原则与现有架构一致：`quant.bin` 是可重建的派生数据，原始 `vectors.bin` 始终权威；量化损坏一律回退精确路径。

### 阶段 1：打分内核重构（性能，最高优先级）

目标：消除逐槽的元数据读锁与重复计算，对齐 Qdrant 的查询预处理模型。

1. **查询预处理下沉**：在 `search/quantized.rs` 开头对查询做一次 `encode_query`，产出按类型特化的查询对象：
   - Scalar：`(query_f32, min, scale)` 已捕获，无需每槽读 meta；
   - Binary：现有 `query_bits` 机制推广为唯一路径；
   - PQ：`QueryPq { lut: Vec<f32> /* m×256 */ }`。
2. **PQ 改 LUT（ADC）**：构建时对每个子空间预计算查询子向量到 256 质心的距离（Euclid 直接累加；Dot 用 `-dot`；L1 用绝对差），逐槽打分退化为 `m` 次查表累加。Cosine 用 `1 − dot/(‖q‖·‖c_code‖)`：LUT 存 `-dot` 与质心范数平方两个表，或直接存预除 `‖q‖` 的点积表。
3. **meta 读锁出热路径**：打分循环开始前取一次 `meta_snapshot()`（QuantMeta 克隆成本 = codebook Arc 化，建议把 codebook 从 `Vec<f32>` 改为 `Arc<Vec<f32>>`），循环内零锁。
4. **SIMD 内核**：沿用 `src/distance/` 现有 kernel 分层（kernel.rs 分发 + avx2/avx512/neon/portable/naive），新增：
   - Scalar：i16 提升的 u8 点积/绝对差内核（参照 qdrant `_mm256_madd_epi16` 思路）；
   - Binary：u8/u128 XOR-popcount 批量内核；
   - PQ LUT 查表本就 O(m)，naive 即可。

验收：`benches/vector_scan_bench.rs` 增加量化路径基准；Scalar/PQ 逐槽打分应有数量级提升（从 O(dim) 降为 O(1)~O(m) 查表）。

### 阶段 2：训练质量与成本

1. **采样训练**：`rebuild.rs` 对 live_slots 超 `TRAIN_SAMPLE_SIZE`（默认 10_000，对齐 qdrant）时做确定性采样（沿用现有 kmeans 的固定种子策略），PQ 每子空间共享同一采样集；迭代数提到 100、增加质心增量 < 1e-5 早停（`index/kmeans.rs` 增加.accuracy 参数）。
2. **分位数流式估计**：Scalar 分位数裁剪改为 P² 单遍估计（或 reservoir 采样后排序），去掉全量 `values` 收集（当前 10M 向量 × 1536 维会占 ~60GB）。
3. **可中止**：`rebuild` 接受 `&AtomicBool`，在逐 slot 编码循环与 kmeans 迭代间检查，中止时保持 `ready=false`（派生数据，重来即可）。
4. **空簇/小样本**：PQ 子空间样本 < 256 时保留现有"训练出的质心数取 min"行为，但 pad 零质心改为复制已有质心，避免零质心吸入所有编码。

### 阶段 3：搜索参数配置面

1. **`QuantizationSearchParams`**（`src/types/query.rs` 或 config）：`ignore: bool`（默认 false）、`rescore: Option<bool>`、`oversampling: Option<f64>`；`SearchQuery` 透传。默认 rescore：Binary=true、Scalar=false、PQ=false（对齐 qdrant `default_rescoring`）——PQ 粗排后用原向量 rescore 的开销由 oversampling 预算控制。
2. **rescore 可选化**：`search/quantized.rs` 当 `rescore=false` 时跳过原向量重排，直接返回量化距离 top-k（打分函数需与精确度量单调一致；Binary 的 Hamming 与 Cosine 不严格单调，文档须注明 recall 特性）。
3. **`exact` 语义**：查询带 `exact=true` 时忽略量化走精确路径（现有 `has_quantization` 分支加条件）。

### 阶段 4：HNSW 深度集成（中期）

1. **量化图遍历**：`search/ann.rs` 图遍历打分改为优先用量化码（阶段 1 的查询对象 + `read_slot`），oversample 遍历候选、退出时用原向量 rescore top 结果。需要处理 `ready=false`（PQ 未训练）时回退原向量打分。
2. **点对点打分**：HNSW 启发式需要"存储点 vs 存储点"距离，优先对第二个点编码后走量化路径（Scalar/Binary 可行；PQ 非对称时回退原向量，对齐 qdrant `new_internal` 的回退逻辑）。
3. **正确性护栏**：量化图遍历结果与精确路径做 A/B 测试（`tests/` 增加 recall 断言：量化路径 recall@10 ≥ 阈值）。

### 不做的事

- 不引入 TurboQuant、多向量量化、sparse 量化（当前 collection 模型为单稠密向量）。
- 不做 appendable chunked 量化存储：现有 `write_slot` 逐槽模型（commit 应用时编码写入）已覆盖追加语义；PQ 未训练期间搜索走精确路径已保证正确性（需在阶段 2 复核该窗口内 `write_slot` 报错路径是否会阻塞写入——当前 `store_write.rs` 应容错跳过）。
- 不做 `inline_storage` 图内联（依赖阶段 4 完成后再评估）。

## 5. 实现状态（已落地）

以上阶段 1–4 已实现，验证见 `tests/quantization_recall_test.rs`（Scalar/Binary/Product 召回护栏 + ignore/exact 回退断言）。与设计的偏差：

- **SIMD 内核未做**：阶段 1 的第 4 项（avx2/avx512/neon u8 点积与 popcount 内核）暂缓，当前为优化过的 portable 循环；查询预处理 + PQ LUT 已消除主要瓶颈，SIMD 可作为后续独立 PR 接入 `src/distance/` kernel 分层。
- **kmeans 早停精度参数未加**：沿用现有 `CONVERGENCE_TOL=1e-3`，迭代数提升到 100（大样本时）。
- **小样本 PQ 训练保护**：样本 < 8×256 时迭代数降为 20，且 kmeans 的 k 压到样本数的 1/4——空簇重播种的 O(n·k) 成本在未优化构建下会主导训练时间（详见 rebuild.rs 注释）；不足 256 的质心由复制填充，码长不变。
- **P² 流式分位数改为 stride 采样**（等效有界内存），未引入 P² 估计器。
- **可中止**：`QuantStore::rebuild` 接受 `&AtomicBool` 协作检查；`build_quantization` 暂以永远为 false 的标志接入（尚无外部取消源）。

## 6. 风险与兼容

- **格式兼容**：阶段 1/2 不改 `quant.bin` 布局（码长不变），仅 meta 内部结构变化；`QUANT_VERSION` 提升到 2，`read_tagged` 版本不符即丢弃重建（派生数据，安全）。
- **recall 回归**：PQ 改 LUT、采样训练都会改变数值结果；以 recall@k 对比测试守护，训练采样可能略降 recall，可接受（换取训练时间可控）。
- **并发**：打分热路径从 `meta.read()` 改为 snapshot 后，`rebuild` 写 meta 时读者持旧快照继续跑，天然安全（ArcSwap 模式与 segments 一致）。
