# devs 服务器同步与远程编译说明（simvec）

本地全量编译耗时高；devs（`ssh devs`）持有暖构建缓存，且是 vec-bench
端到端基准的执行机。代码同步到 devs 编译，本说明约定同步方式。devs
无外网：数据集与 Python 依赖一律在本地备包后离线传输，不使用校验和
（按项目政策，以包内文件数与解压前后体积抽查代替）。

## 目标位置

- 同步包存放：`/data/kkkqkx/simvec/`（历史 `simvec-src-*.tar.*` 同目录）
- 代码解压位置：`/data/kkkqkx/simvec/repo/`（直接覆盖，非 git 仓库）
- 数据集存放：`/data/kkkqkx/simvec/vec-bench/data/`（单独打包，见下文）
- 构建日志：`/data/kkkqkx/simvec/` 下按 `build-<主题>.log` 命名

## 传输方式选择

| 变更规模 | 内容 | 方式 |
|---|---|---|
| 小规模（数个文件） | 改动的源文件 | `scp` 直接覆盖远端对应路径 |
| 大规模（多文件/重构/首次） | 整个工作树 | `tar.zst` 打包 + `scp` + 远端覆盖解压 |
| 基准准备 | `vec-bench/data/`（数据集缓存） | 单独压缩包，仅在需要跑基准时传输，日常代码同步排除 |
| 任何情况 | `target/`、`node_modules`、构建产物 | 永不传输 |

构建产物永不传输：devs 靠自身暖缓存增量编译；`target/` 体积大且与本地
工具链绑定，传过去也无法复用。

## 打包排除清单

以仓库根为基准，打包时排除：`target/`、各级 `.git`（含 submodule 的
gitlink）、`vec-bench/data/`（数据集包另传）、`vec-bench/results/`、
各级 `node_modules/`、`__pycache__/`、`.mypy_cache/`、本地运行遗留。
`Cargo.lock` 必须进包：devs 无外网，锁文件是离线可复现构建的前提。
`ref/VectorDBBench` 仅本地分析用，不进同步包。

## 同步步骤

本地打包（`sha` 取本地 HEAD 短哈希）：

```bash
cd /home/kkkqkx/code
tar --zstd -cf ~/sync-pkgs/simvec-src-<sha>.tar.zst \
  --exclude='*/target' --exclude='*/.git' \
  --exclude='simvec/vec-bench/data' --exclude='simvec/vec-bench/results' \
  --exclude='simvec/ref' --exclude='*/node_modules' \
  --exclude='*/__pycache__' --exclude='*/.mypy_cache' \
  --exclude='*.bak' simvec
```

记录构建信息（HEAD 哈希、日期、本地工具链、未提交变更），随包传输：

```bash
{ echo "sha=<sha>"; echo "date=$(date -u +%FT%TZ)"; rustc --version | sed 's/^/local_rustc=/';
  git -C ~/code/simvec status --short; } > ~/sync-pkgs/simvec-src-<sha>.BUILD_INFO.txt
scp ~/sync-pkgs/simvec-src-<sha>.tar.zst ~/sync-pkgs/simvec-src-<sha>.BUILD_INFO.txt \
  devs:/data/kkkqkx/simvec/
```

远端覆盖解压并抽查（文件数对得上即视为完整）：

```bash
ssh devs "cd /data/kkkqkx/simvec && tar --zstd -xf simvec-src-<sha>.tar.zst -C repo --strip-components=1 \
  && cp simvec-src-<sha>.BUILD_INFO.txt repo/BUILD_INFO.txt \
  && tar --zstd -tf simvec-src-<sha>.tar.zst | wc -l && du -sh repo"
```

小规模变更直接传文件，路径与仓库根一一对应：

```bash
scp src/storage/wal.rs devs:/data/kkkqkx/simvec/repo/src/storage/wal.rs
```

## 远端编译

devs 上 cargo 不在默认 PATH，先导出。编译放后台跑，日志落盘：

```bash
ssh devs "export PATH=\$HOME/.cargo/bin:\$PATH && cd /data/kkkqkx/simvec/repo \
  && nohup cargo build --features c_api > ../build-c_api.log 2>&1 &"
```

常用验证命令：

```bash
cargo check -p simvec --features c_api   # C ABI 必须可编译（cdylib）
cargo test -p simvec                     # 单元 + 集成测试
```

注意：`include/simvec.h` 已提交；若改动 `src/c_api`，需在装有 cbindgen
的环境重新生成并随包同步，同时更新 `docs/binding-matrix.md`。

## 数据集同步（vec-bench）

数据集与代码分开打包，仅在需要跑基准时传输。本地先下载好：

```bash
cd ~/code/simvec/vec-bench
python -m vec_bench download --dataset cohere_small_100k   # 缓存到 /tmp/vec-bench/dataset
tar --zstd -cf ~/sync-pkgs/simvec-data-<数据集>.tar.zst -C /tmp/vec-bench/dataset .
scp ~/sync-pkgs/simvec-data-<数据集>.tar.zst devs:/data/kkkqkx/simvec/
ssh devs "mkdir -p /data/kkkqkx/simvec/vec-bench/data \
  && tar --zstd -xf /data/kkkqkx/simvec/simvec-data-<数据集>.tar.zst -C /data/kkkqkx/simvec/vec-bench/data"
```

devs 无外网，`vec_bench download` 在远端**不可用**；一律本地下载后打包。
数据集目录名与 VectorDBBench 一致（如 `cohere_small_100k`），同步时用
`--data-root /data/kkkqkx/simvec/vec-bench/data` 指向该缓存。

## devs 基准 Python 环境

devs 已有离线 Python 工具链（uv + CPython 3.11，见 `/data/kkkqkx/tools/`，
系统 Python 3.6.9 不可用）。vec-bench 用独立 venv，依赖离线安装：

```bash
# 本地（有外网）：备 cp311 wheelhouse
pip download -d wheelhouse311 -p cp311 --only-binary=:all: \
  pyarrow numpy qdrant-client psycopg pgvector
tar --zstd -cf ~/sync-pkgs/simvec-wheelhouse311.tar.zst wheelhouse311
scp ~/sync-pkgs/simvec-wheelhouse311.tar.zst devs:/data/kkkqkx/simvec/packages/

# devs：建 venv 并离线安装
ssh devs "cd /data/kkkqkx/simvec \
  && /data/kkkqkx/tools/uv/uv venv --python /data/kkkqkx/tools/python3.11/bin/python3.11 .venv-vecbench \
  && /data/kkkqkx/tools/uv/uv pip install --python .venv-vecbench/bin/python \
       --offline --no-index --find-links packages/wheelhouse311 \
       pyarrow numpy qdrant-client psycopg pgvector"
```

选 wheel 时注意 devs glibc 2.27（Ubuntu 18.04），装完先 `import pyarrow` 冒烟。

## 跑基准

```bash
ssh devs
export PATH=/data/kkkqkx/simvec/repo/target/release:\$PATH
source /data/kkkqkx/simvec/.venv-vecbench/bin/activate
export SIMVEC_LIB=/data/kkkqkx/simvec/repo/target/release/libsimvec.so
cd /data/kkkqkx/simvec/repo/vec-bench
python -m vec_bench run --backend simvec --dataset cohere_small_100k \
  --data-root /data/kkkqkx/simvec/vec-bench/data --out results/
```

## devs 上的 pgvector

系统 PG 版本过老，不改动；pgvector 以 Docker 方式运行（镜像自包含，
与宿主 glibc 无关）。前提：账号在 `docker` 组内（`docker info` 可用）。

```bash
# 本地（有外网）
docker pull pgvector/pgvector:pg16
docker save pgvector/pgvector:pg16 | zstd > ~/sync-pkgs/pgvector-pg16.image.tar.zst
scp ~/sync-pkgs/pgvector-pg16.image.tar.zst devs:/data/kkkqkx/simvec/packages/

# devs：导入并起库
ssh devs "docker load -i /data/kkkqkx/simvec/packages/pgvector-pg16.image.tar.zst \
  && docker run -d --name pgvector-bench -p 5433:5432 \
     -e POSTGRES_PASSWORD=bench -e POSTGRES_DB=bench pgvector/pgvector:pg16"
```

基准命令（`--dsn` 指向该库）：

```bash
python -m vec_bench run --backend pgvector --dataset cohere_small_100k \
  --dsn postgresql://postgres:bench@localhost:5433/bench
```

## 反向取回

结果 JSON 体积小，直接取回：

```bash
scp -r devs:/data/kkkqkx/simvec/repo/vec-bench/results ./vec-bench/results-devs/
```

## 注意事项

- 覆盖解压不删除远端多余文件：改名或删除源文件后，需手动删除远端对应
  文件，否则旧文件仍会被编译。基准性结论以前建议用空目录全新解压复核。
- submodule（若有）以包内 checkout 为准：devs 无外网无法执行 submodule
  更新，升级 submodule 后必须走压缩包全量同步。
- `vec-bench/data/` 与 `results/` 日常不同步；跑基准前另打数据集包。
- 非高峰时段传大包；单包超过约 2GB 改用 `rsync --partial --append-verify`
  断点续传。
- 基准前确认远端 `target/` 已按 release 构建过 `--features c_api`，
  否则 simvec 后端会因缺少 cdylib 报绑定不可用。
