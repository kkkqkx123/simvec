# vec-bench — simvec 端到端性能测试

与 `benches/`（criterion，Rust 微基准）分开的**端到端基准**：真实数据集
（parquet + ground truth）上对比 simvec、Qdrant（本地模式）与 pgvector 的
插入吞吐、查询延迟与召回率。数据集下载逻辑提炼自 `ref/VectorDBBench`
（仅保留 Aliyun OSS 国内镜像与幂等校验，不引入该框架）。

## 目录

```
vec_bench/
├── datasets.py          # 数据集注册表 + OSS 镜像下载（国内源）
├── runner.py            # insert -> optimize -> search 评测循环
├── metrics.py           # recall@k 与延迟分位数
├── backends/
│   ├── simvec_backend.py    # 本项目，经 bindings/python FFI 驱动 cdylib
│   ├── qdrant_backend.py    # qdrant-client 本地模式
│   └── pgvector_backend.py  # psycopg + pgvector 扩展
└── __main__.py          # CLI: python -m vec_bench {download,run}
```

## 准备

```bash
# 1. simvec 动态库 + Python 绑定（simvec 后端必需）
cargo build --features c_api
export SIMVEC_LIB=$PWD/target/debug/libsimvec.so   # 或加入 fallback 路径
pip install -e bindings/python                     # 或将 bindings/python 加入 PYTHONPATH

# 2. 基准依赖
pip install -r vec-bench/requirements.txt
```

## 使用

```bash
cd vec-bench

# 只下载数据集（国内镜像 assets.zilliz.com.cn，匿名访问，按文件大小幂等续传）
python -m vec_bench download --dataset cohere_small_100k

# 跑单个后端；结果 JSON 落在 results/
python -m vec_bench run --backend simvec --dataset cohere_small_100k -k 100
python -m vec_bench run --backend qdrant  --dataset cohere_small_100k -k 100
python -m vec_bench run --backend pgvector --dataset cohere_small_100k -k 100 \
    --dsn postgresql://user:pass@localhost/bench

# 先截取查询数做冒烟
python -m vec_bench run --backend simvec --dataset cohere_small_100k --max-queries 100
```

## 评测口径

- **插入**：固定 batch（默认 1000）逐批写入，报告总吞吐 vec/s。
  Qdrant 期间关闭索引（`indexing_threshold=0`），插完重建——与
  VectorDBBench 的 qdrant_local 客户端一致。
- **optimize**：阻塞到索引可查（simvec `build_index`；Qdrant 轮询
  collection GREEN；pgvector `CREATE INDEX ... USING hnsw`）。
- **查询**：test.parquet 全量（可用 `--max-queries` 截断），先 50 条预热
  不计时，之后逐条计时报 p50/p95/p99。
- **召回率**：结果 id 与 ground truth（neighbors.parquet）前 k 名的
  交集占比。所有引擎统一使用各自原生 cosine/l2 语义（simvec 在线归一化，
  无需预归一化数据）。

## 数据集

| 名称 | 维度 | 距离 | 规模 | 分片 |
|---|---|---|---|---|
| cohere_small_100k | 768 | cosine | 10 万 | 1 |
| cohere_medium_1m | 768 | cosine | 100 万 | 1 |
| cohere_large_10m | 768 | cosine | 1000 万 | 10 |
| gist_small_100k | 960 | l2 | 10 万 | 1 |
| gist_medium_1m | 960 | l2 | 100 万 | 1 |

缓存根目录默认 `/tmp/vec-bench/dataset`（`--data-root` 可改），目录名与
VectorDBBench 一致（如 `cohere_small_100k`），两边可共用同一份缓存。
devs 服务器上数据集同步与离线 Python 环境见
`docs/dev/devs-sync-and-remote-build.md`。

## 结果对比建议

同一数据集、同一 k 下，对每个后端跑 2–3 档索引参数（如 ef_search 128/256/512）
画 recall-latency 曲线；结果 JSON 中记录了全部参数，可直接并排比较。
