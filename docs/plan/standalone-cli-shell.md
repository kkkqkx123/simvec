# 独立命令行工具（simvec shell）设计方案

本文给出独立命令行工具的设计。这里的“类似 SQLite CLI”指交互形态和分层方式，而不是实现语言：SQLite shell 是 C 程序，是因为 SQLite 本体是 C 库；simvec 本体是 Rust，等价选择是独立的 Rust shell。它作为单独的 Cargo 包编译成单独的可执行文件，通过 `simvec` 的公开 Rust API 使用引擎，不进入库 crate，也不依赖 C ABI。

## 目标与边界

- CLI 是独立可执行目标：单独的 Cargo 包、单独的二进制、单独的依赖集合，使用 `cargo build --manifest-path cli/Cargo.toml` 构建。
- 根 `simvec` crate 不增加 `[[bin]]`、不增加 feature、不改变 `crate-type`；默认 `cargo build` 仍只构建库。
- CLI 只使用 `simvec` 的公开 Rust API（`LocalVectorEngine` 及 `types` 中的公开类型），不引用 `engine`、`storage`、`index` 的私有模块。
- 首发不通过 C ABI 调用引擎。C ABI 是给非 Rust 绑定使用的稳定外部边界；Rust CLI 走它只会引入手写 FFI 声明、字符串出参管理和状态码转换，得不到额外收益。
- CLI 提供交互式、stdin 批处理和单条命令三种入口，对应 SQLite shell 的使用方式。
- 不实现 SQL 解析、查询计划器或 SQL DDL/DML。命令语言是公开引擎 API 的薄命令表，复杂参数用 JSON 表达。
- 不改动 `include/simvec.h`、`src/c_api`、`docs/binding-matrix.md`。CLI 不是语言绑定，不进入绑定矩阵。

## 现状依据

- `LocalVectorEngine` 已公开完整能力：集合生命周期、点写入/删除、检索、payload 操作、事务、索引与量化维护、默认配置、临时集合提升与备份恢复。
- `types` 中的 `CollectionConfig`、`SearchQuery`、`VectorFilter`、`IndexConfig`、量化配置等都可 serde 反序列化，CLI 可以直接复用同一套 JSON 形状。
- `EngineResult<T>` 使用 Rust 错误类型，CLI 可以直接打印 `Display` 文本，不需要再映射成 C 状态码。
- 当前 C ABI 已经由 Python、JavaScript、Go、Java、R 绑定覆盖，其一致性由绑定测试保证；CLI 没有必要重复验证这条边界。
- 仓库中只有 `vec-bench` 的 Python 基准入口，没有通用数据库 shell。

## 方案选择

| 方案 | 做法 | 结论 |
| --- | --- | --- |
| 根 crate 加 `[[bin]]` | 在 `src/bin` 写 Rust 程序 | 不采用。虽然编译目标独立，但 CLI 依赖、发布和构建都被根包绑定，不是独立包。 |
| 独立 Cargo 包直接调用 Rust API | `cli/` 作为单独包，路径依赖 `simvec`，调用 `LocalVectorEngine` | 采用。保持语言一致、类型安全，且编译产物是独立可执行文件。 |
| 独立 Rust 包经 C ABI 调用 | 手写 FFI 声明或 bindgen，链接 `libsimvec` | 不采用。第一方 Rust 消费者绕过类型系统，需要重复声明头文件、释放字符串、转换错误码；C ABI 已由绑定测试覆盖。 |
| 独立 C shell | 仿照 `shell.c` 包含 `simvec.h` | 不采用。语言不匹配，且需要额外 C 工具链和 FFI 样板；SQLite 这样做是因为它本身是 C。 |

“独立编译对象”在 Rust 项目里对应独立 package 和独立 binary target，不对应 C 源文件。CLI 与库的关系仍然是 SQLite 式的：库提供稳定 API，shell 只做用户输入、命令分派和结果展示。

## 总体结构

新增文件：

- `cli/Cargo.toml`：包名 `simvec-cli`，bin 名 `simvec`；路径依赖 `simvec = { path = ".." }`；依赖 `serde_json` 与 `serde` 的 derive 能力。
- `cli/Cargo.lock`：独立包自己的锁文件。
- `cli/src/main.rs`：进程入口，把参数交给库入口。
- `cli/src/lib.rs`：参数解析、模式选择、进程退出码。
- `cli/src/shell.rs`：行读取、提示符、续行、点命令、shell 状态。
- `cli/src/commands.rs`：普通命令表与引擎调用。
- `cli/src/dto.rs`：CLI 侧 JSON 输入 DTO，转换成公开 Rust 类型。
- `cli/src/output.rs`：JSON/标量输出、输出重定向。
- `cli/src/error.rs`：CLI 错误文本与 serde / 引擎错误转换。
- `cli/tests/lexer.rs`、`cli/tests/smoke.rs`：单元测试与端到端测试。
- `scripts/build-cli.sh`：统一构建入口。
- `cli/README.md`：命令清单与示例。

不新增 Rust 核心模块，不修改根 `Cargo.toml`，不修改 C API 或头文件。

数据流：

```
命令行参数 / stdin / 交互式输入
  -> CLI 行读取与引号感知切词
  -> 点命令本地处理，或普通命令解析 JSON 后调用公开 Rust API
  -> LocalVectorEngine
  -> serde JSON 输出、标量输出、Rust 错误信息
```

## 命令模型

沿用 SQLite shell 的两层命令：点命令由 shell 自己解释，普通命令映射到引擎 API。点命令以 `.` 开头；普通命令使用短名。复杂参数仍是单个 JSON token，建议用单引号包裹。

命令的 JSON 形状与现有绑定保持一致，但不是直接转发给 C API：CLI 在 `dto.rs` 中定义小的输入 DTO，再转换成 `CollectionConfig`、`SearchQuery`、`VectorPoint`、`TxnOp` 等公开类型。例如点输入接受 `vector` 快捷字段或完整 `vectors` map，查询输入接受 `vector` / `vectors` / `named_vector` 等快捷字段。这样命令对使用者友好，同时底层仍然只有一套引擎语义。

### 点命令

| 命令 | 行为 |
| --- | --- |
| `.open <path>` | 释放当前引擎后打开新的引擎根目录。 |
| `.close` | 关闭当前引擎；之后的库命令报告未打开。 |
| `.quit` / `.exit` | 关闭引擎并退出。 |
| `.help [command]` | 从命令表生成帮助。 |
| `.tables` | 列出集合名。 |
| `.schema <collection>` | 输出集合配置。 |
| `.info <collection>` | 输出 `CollectionInfo`。 |
| `.output <path>` / `.stdout` | 重定向结果输出或恢复标准输出；错误始终走 stderr。 |
| `.read <path>` | 逐行执行文件中的命令。 |
| `.echo on|off` | 执行前是否回显命令。 |
| `.bail on|off` | 遇错是否停止当前脚本；交互模式默认 off，批处理默认 on。 |
| `.timer on|off` | 在 stderr 输出每条引擎命令耗时。 |
| `.mode json|pretty` | 紧凑或缩进 JSON 输出；默认 `json`。 |
| `.version` | 打印 CLI 版本与依赖的库版本。 |

### 普通命令与 Rust API 映射

| 命令 | 使用的公开 API |
| --- | --- |
| `create <collection> [config_json]` | `LocalVectorEngine::create_collection`；省略 JSON 时使用 `CollectionConfig::default()`。 |
| `drop <collection>` | `delete_collection` |
| `exists <collection>` | `collection_exists` |
| `list` | `collection_names` |
| `info <collection>` | `collection_info` |
| `config <collection>` | `collection_config` |
| `upsert <collection> <points_json>` | `upsert_batch`；接受点数组，支持 dense、sparse、multi 与 `vectors`。 |
| `delete <collection> <ids_json>` | `delete_batch` |
| `delete_filter <collection> <filter_json>` | `delete_by_filter` |
| `set_payload <collection> <id> <payload_json>` | `set_payload` |
| `merge_payload <collection> <id> <fields_json>` | `set_payload_fields` |
| `set_field <collection> <id> <key> <value_json>` | `set_payload_field` |
| `delete_keys <collection> <id> <keys_json>` | `delete_payload` |
| `update_vectors <collection> <id> <vectors_json>` | `update_named_vectors` |
| `index_create <collection> <field> <schema>` | `create_payload_index`；schema 名称转换为 `PayloadSchemaType`。 |
| `index_drop <collection> <field>` | `delete_payload_index` |
| `indexes <collection>` | `list_payload_indexes` |
| `search <collection> <query_json>` | `search` |
| `get <collection> <id>` | `get` |
| `count <collection>` | `count` |
| `count_filter <collection> <filter_json>` | `count_by_filter` |
| `scroll <collection> <limit> [offset] [with_payload] [with_vector]` | `scroll` |
| `scroll_filter <collection> <filter_json> <limit> [offset] [with_payload] [with_vector]` | `scroll_by_filter` |
| `txn <ops_json>` | `apply_txn`；CLI DTO 转成 `TxnOp` 列表。 |
| `build_index <collection>` | `build_index` |
| `build_quant <collection>` | `build_quantization` |
| `has_index <collection>` | `has_index` |
| `has_quant <collection>` | `has_quantization` |
| `drop_index <collection>` | `drop_index` |
| `drop_published <collection>` | `drop_published_index` |
| `compact <collection>` | `compact_collection` |
| `drain_pending <collection>` | `drain_pending` |
| `metrics <collection>` | `collection_metrics` |
| `maintenance` | `run_maintenance_sweep` |
| `promote <live> <temp> <suffix>` | `promote_temp_collection` |
| `restore <live>` | `restore_promote_backup` |
| `temps` | `list_temp_collections` |
| `backups <live>` | `list_old_backups_for` |
| `set_default_hnsw <json>` / `get_default_hnsw` | `set_default_hnsw_config` / `default_hnsw_config` |
| `set_default_ivf <json>` / `get_default_ivf` | `set_default_ivf_config` / `default_ivf_config` |
| `set_default_quant <json>` / `get_default_quant` | `set_default_quantization_config` / `default_quantization_config` |

典型交互：

```text
$ simvec ./data
simvec> create docs '{"vector_size":3,"distance":"Cosine"}'
simvec> upsert docs '[{"id":"a","vector":[1,0,0],"payload":{"title":"hello"}},{"id":"b","vector":[0,1,0]}]'
simvec> search docs '{"vector":[1,0,0],"limit":5,"with_payload":true}'
simvec> .tables
simvec> .quit
```

单条命令与脚本：

```text
simvec ./data "count docs"
simvec --batch ./data < commands.txt
```

入口规则参考 SQLite：给出命令参数时执行单条命令后退出；stdin 不是终端时走批处理；否则进入交互提示符。`--batch` 和 `--interactive` 用于显式覆盖终端判断。

## 输入处理

- 行读取使用可增长字符串，不限制 JSON payload 长度。
- 切词器识别单引号、双引号和反斜杠转义；引号内空格不分词，转义引号不提前结束 token。
- 一行末尾仍有未闭合引号时进入续行状态，提示符切换到 `...>`；EOF 时未闭合则报语法错误。
- 语句以换行结束，不引入分号终止符，避免与 JSON 字符串中的分号冲突；`.read` 保持一行一条命令。
- 以 `#` 开头的行是注释；引号内或 token 中间的 `#` 不触发注释。
- 参数个数不符合命令表时打印该命令用法，不调用引擎。
- JSON 解析失败报告为命令错误，错误文本包含命令名和 serde 的解析位置；不静默跳过该命令。

## 状态、错误与输出

- shell 状态保存 `Option<LocalVectorEngine>` 和输出/回显/中断策略。`.open` 先替换句柄，旧引擎在 `Drop` 时关闭维护线程；`.close` 后指针置为 `None`。
- 所有库命令先检查引擎是否已打开；未打开时报告 shell 本地错误。
- 引擎错误直接打印 `VectorSearchError` 的 `Display` 文本；JSON 解析错误单独标注。CLI 不伪造 C 状态码，也不维护第二套错误码表。
- 实现 `VectorEngineError` 到 CLI 错误类别的轻量映射仅用于帮助文本分组，不改变错误原文。
- `.bail` 打开时脚本在首个错误停止；关闭时记录错误并继续下一条命令，进程退出码仍为非零。
- 实现 `Serialize` 的结果使用 `serde_json::to_string` 或 `to_string_pretty` 输出；计数、存在性、布尔状态输出十进制整数。`.output` 只重定向结果，错误、帮助和计时信息走 stderr。
- 表格输出不是首发目标；若后续增加，只在 `output.rs` 中扩展渲染，不影响命令分派和引擎调用。

## 构建与分发

- `scripts/build-cli.sh` 执行 `cargo build --release --manifest-path cli/Cargo.toml --target-dir target`，产物为 `target/release/simvec`。
- CLI 构建只需要 Rust 工具链，不需要 C 编译器，也不需要 `c_api` feature。
- 独立包使用自己的 `Cargo.lock`，不把 CLI 依赖写进根 `Cargo.lock`。
- 首发不引入 clap、rustyline、crossterm 等外部 CLI 框架；参数解析、行读取和引号切词由标准库实现。外部依赖只有 `serde`、`serde_json` 和 dev 依赖 `tempfile`。
- 由于没有新增 C API，`include/simvec.h` 不需要重生成，绑定矩阵不变。

## 验证计划

- `cli/tests/lexer.rs` 覆盖：空行、连续空白、单引号 JSON、双引号路径、转义引号、未闭合引号、注释、含 `#` 的字符串。
- `cli/tests/smoke.rs` 使用 `CARGO_BIN_EXE_simvec` 和临时目录，覆盖：
  - `.open`、创建集合、列表、配置、信息、`.close`、重新打开；
  - 批量写入、搜索、取点、计数、过滤计数、分页扫描；
  - payload 写入/合并/删键、payload 索引创建与列举；
  - 具名向量 patch、跨集合事务；
  - 索引构建/量化构建、指标、压缩、维护扫描；
  - 默认配置读写、临时集合与备份列表；
  - 未知集合、非法 JSON、未知命令、参数不足、未打开引擎时的错误路径；
  - `.bail` 开/关、`.read`、`.output`、`.echo`、`.timer`、`.mode pretty`。
- 命令级测试直接调用 `commands.rs` 的 handler，避免把每个用例都写成进程级测试；进程级测试只覆盖入口模式和退出码。
- 核心库测试不变；CLI 不重复 C ABI 绑定测试。

## 落地顺序

1. 建立 `cli/` 独立包与构建脚本，先让空白 shell 能打开、关闭和退出引擎。
2. 实现行读取、引号切词、点命令、错误与输出；补齐切词器测试。
3. 实现 DTO 转换层，确保 JSON 形状与现有绑定一致。
4. 实现集合管理与读写检索命令。
5. 实现 payload、事务、维护、默认配置和高级集合命令。
6. 实现 `.read`、`.output`、`.bail`、`.timer`、入口模式选择和 `.mode pretty`。
7. 增加冒烟测试并接入现有验证流程。
8. 实现完成后同步更新 `README.md` 的命令行用法和 `AGENTS.md` 的目录、构建命令与非显然规则；只替换成当前事实，不追加历史方案。

## 非目标与后续演进

- 不实现 SQL 方言、`.dump`、`.restore` 或数据库文件导入导出；这些能力需要先在公开 Rust API 中成立。
- 不引入行编辑库或复杂表格渲染库；如需增强，仍限定在 CLI 包内。
- 不把 CLI 配置写进集合目录，也不改变引擎根目录布局。
- 若未来确实需要“只链接已安装动态库”的发行形态，再单独评估 Rust FFI 包装；它不改变本方案中命令层和 shell 结构。
