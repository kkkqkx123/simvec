# simvec shell

`simvec` 的独立命令行工具。它是单独的 Rust Cargo 包，路径依赖核心库并直接调用公开的 `LocalVectorEngine` API；不经过 C ABI，不需要 C 编译器，也不改变核心库构建。

## 构建

```sh
sh scripts/build-cli.sh
# 产物：target/release/simvec
```

也可以直接执行：

```sh
cargo build --release --manifest-path cli/Cargo.toml --target-dir target
```

## 使用

```sh
target/release/simvec ./data
target/release/simvec ./data "count docs"
target/release/simvec --batch ./data < commands.txt
```

第一个位置参数是引擎根目录；集合位于该目录下。给出命令参数时执行一条命令后退出；stdin 不是终端时自动进入批处理；`--interactive` 和 `--batch` 可以显式选择模式。

## 命令约定

以 `.` 开头的命令由 shell 自己处理：

```text
.open <path>          .close                .quit
.help [command]       .tables               .schema <collection>
.info <collection>    .output <path>        .stdout
.read <path>          .echo on|off          .bail on|off
.timer on|off         .mode json|pretty     .version
```

其余命令映射到引擎 API。复杂参数用 JSON，建议用单引号包裹：

```text
simvec> create docs '{"vector_size":3,"distance":"Cosine"}'
simvec> upsert docs '[{"id":"a","vector":[1,0,0],"payload":{"title":"hello"}}]'
simvec> search docs '{"vector":[1,0,0],"limit":5,"with_payload":true}'
simvec> .tables
simvec> .quit
```

运行 `.help` 查看完整命令表，`.help <command>` 查看单条命令用法。

## 输入与输出

- 一行一条命令；JSON 中的引号和空格由引号感知切词器保护。
- 未闭合引号会进入续行输入，提示符为 `...>`。
- 以 `#` 开头的行是注释。
- 结果默认输出紧凑 JSON；`.mode pretty` 切换为缩进 JSON。计数和布尔状态输出整数。
- 错误、帮助和计时走 stderr，不跟随 `.output` 重定向。

## 测试

```sh
cargo test --manifest-path cli/Cargo.toml --target-dir target
```
