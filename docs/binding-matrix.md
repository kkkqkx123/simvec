# 绑定功能矩阵：C API / Python / JavaScript / Go / Java / R / WASM

统计口径：C 侧以 `include/simvec.h` 与 `src/c_api` 对照为准，两者一致，共 50 个导出函数。
Python 侧看 `bindings/python/simvec/engine.py` 的 `Engine` 方法与 `_native.py` 的签名声明。
JavaScript 侧看 `bindings/js/src/engine.js` 的 `Engine` 方法与 `native.js` 的签名声明。
Go 侧看 `bindings/go/engine.go` 的 `Engine` 方法。
Java 侧看 `bindings/java/src/main/java/io/simvec/Engine.java` 的 `Engine` 方法与 `NativeLib` 的签名声明。
R 侧看 `bindings/r/R/simvec.R` 的 `simvec_*` 函数与 `src/simvecr.c` 的桥接声明。
WASM 侧看 `bindings/wasm/src/index.js` 的 `Engine` 方法与 `src/wasm_api.rs` 的边界声明，内存形态。
`Rust` 原生能力以 `src/engine.rs` 的 `LocalVectorEngine` 公开方法为准，内存形态以 `src/memory_engine.rs` 的 `MemoryEngine` 为准。

符号说明：表格中用“有”表示该层已封装可用，用“缺”表示该层未封装，
用“内部”表示仅在绑定实现内部使用、不对调用者暴露。

## 生命周期与错误原语

| 能力 | C API | Python Engine | JS Engine | Go Engine | Java Engine | R | 备注 |
|---|---|---|---|---|---|---|---|
| 打开引擎 | `simvec_open` | `Engine(path)` 构造器 | `new Engine(dirPath)` | `NewEngine(dir)` | `new Engine(dirPath)` | `simvec_open(path)` 构造环境对象 | 五端齐全 |
| 关闭引擎 | `simvec_close` | `close`，上下文管理器与析构兜底 | `close`，`closed` 属性与显式资源管理 | `Close`，`Closed` 查询与终结器兜底 | `close`，`isClosed` 查询、资源块写法与回收器兜底 | `simvec_close`，可重复调用与终结器兜底，`simvec_closed` 查询 | 五端齐全 |
| 释放返回字符串 | `simvec_free_string` | 内部调用 | 内部调用 | 内部调用 | 内部调用 | 内部调用 | 调用者不可见是正确设计 |
| 查询最近错误码 | `simvec_errcode` | `last_code` 只读属性 | `lastCode` 只读属性 | `LastCode` 只读访问 | `lastCode` 只读访问 | `simvec_last_code` | 异常之外可主动查询 |
| 查询最近错误文本 | `simvec_last_error` | `last_message` 只读属性 | `lastMessage` 只读属性 | `LastMessage` 只读访问 | `lastMessage` 只读访问 | `simvec_last_message` | 同上 |
| 无句柄错误文本 | `simvec_last_error_text` | 打开失败时内部读取 | 打开失败时内部读取 | 打开失败时内部读取 | 打开失败时内部读取 | 打开失败时内部读取 | 同上 |

## 集合管理

| 能力 | C API | Python Engine | JS Engine | Go Engine | Java Engine | R |
|---|---|---|---|---|---|---|
| 创建集合 | `simvec_create_collection` | `create_collection` | `createCollection` | `CreateCollection` | `createCollection` | `simvec_create_collection` |
| 删除集合 | `simvec_delete_collection` | `delete_collection` | `deleteCollection` | `DeleteCollection` | `deleteCollection` | `simvec_delete_collection` |
| 集合是否存在 | `simvec_collection_exists` | `collection_exists` | `collectionExists` | `CollectionExists` | `collectionExists` | `simvec_collection_exists` |
| 列出集合 | `simvec_list_collections` | `list_collections` | `listCollections` | `ListCollections` | `listCollections` | `simvec_list_collections` |
| 集合信息 | `simvec_collection_info` | `collection_info` | `collectionInfo` | `CollectionInfo` | `collectionInfo` | `simvec_collection_info` |
| 集合配置 | `simvec_collection_config` | `collection_config` | `collectionConfig` | `CollectionConfig` | `collectionConfig` | `simvec_collection_config` |
| 提升临时集合 | `simvec_promote_temp_collection` | `promote_temp_collection` | `promoteTempCollection` | `PromoteTempCollection` | `promoteTempCollection` | `simvec_promote_temp_collection` |
| 恢复提升备份 | `simvec_restore_promote_backup` | `restore_promote_backup`，返回备份名 | `restorePromoteBackup`，返回备份名 | `RestorePromoteBackup`，返回备份名 | `restorePromoteBackup`，返回备份名 | `simvec_restore_promote_backup`，返回备份名 |
| 列出临时集合 | `simvec_list_temp_collections` | `list_temp_collections` | `listTempCollections` | `ListTempCollections` | `listTempCollections` | `simvec_list_temp_collections` |
| 列出备份代际 | `simvec_list_old_backups` | `list_old_backups` | `listOldBackups` | `ListOldBackups` | `listOldBackups` | `simvec_list_old_backups` |

该分组五端完全对齐，无缺口。

## 写入与载荷

| 能力 | C API | Python Engine | JS Engine | Go Engine | Java Engine | R |
|---|---|---|---|---|---|---|
| 单点写入 | `simvec_upsert` | `upsert`，支持任意浮点序列与 numpy 连续数组 | `upsert`，`Float32Array` 零拷贝，其余序列拷贝转换 | `Upsert`，`[]float32` 主形态另有 `Float32s` 辅助转换 | `upsert`，`float[]` 主形态另有 `double[]` 重载拷贝转换 | `simvec_upsert`，数值向量拷贝为连续单精度数组 |
| 批量写入 | `simvec_upsert_batch` | `upsert_batch` | `upsertBatch` | `UpsertBatch` | `upsertBatch` | `simvec_upsert_batch` |
| 按标识删除 | `simvec_delete` | `delete` | `delete` | `Delete` | `delete` | `simvec_delete` |
| 批量删除 | `simvec_delete_batch` | `delete_batch` | `deleteBatch` | `DeleteBatch` | `deleteBatch` | `simvec_delete_batch` |
| 按过滤器删除 | `simvec_delete_by_filter` | `delete_by_filter`，返回删除计数 | `deleteByFilter`，返回删除计数 | `DeleteByFilter`，返回删除计数 | `deleteByFilter`，返回删除计数 | `simvec_delete_by_filter`，返回删除计数 |
| 整体替换载荷 | `simvec_set_payload` | `set_payload` | `setPayload` | `SetPayload` | `setPayload` | `simvec_set_payload` |
| 合并载荷字段 | `simvec_merge_payload` | `merge_payload` | `mergePayload` | `MergePayload` | `mergePayload` | `simvec_merge_payload` |
| 单字段载荷直写 | `simvec_set_payload_field` | `set_payload_field`，值接受任意可序列化对象 | `setPayloadField`，值接受任意可序列化对象 | `SetPayloadField`，值接受任意可序列化值 | `setPayloadField`，值接受任意可序列化对象 | `simvec_set_payload_field`，值接受任意可序列化值 |
| 删除载荷字段 | `simvec_delete_payload_keys` | `delete_payload_keys` | `deletePayloadKeys` | `DeletePayloadKeys` | `deletePayloadKeys` | `simvec_delete_payload_keys` |
| 创建载荷索引 | `simvec_create_payload_index` | `create_payload_index` | `createPayloadIndex` | `CreatePayloadIndex` | `createPayloadIndex` | `simvec_create_payload_index` |
| 删除载荷索引 | `simvec_delete_payload_index` | `delete_payload_index`，返回是否命中 | `deletePayloadIndex`，返回是否命中 | `DeletePayloadIndex`，返回是否命中 | `deletePayloadIndex`，返回是否命中 | `simvec_delete_payload_index`，返回是否命中 |
| 列出载荷索引 | `simvec_list_payload_indexes` | `list_payload_indexes` | `listPayloadIndexes` | `ListPayloadIndexes`，具名 `PayloadIndex` 结构 | `listPayloadIndexes`，具名 `PayloadIndex` 记录列表 | `simvec_list_payload_indexes`，列表 |

该分组五端完全对齐，无缺口。向量缓冲仅在调用期间持有，
不再随引擎实例长期累积。

## 事务

| 能力 | C API | Python Engine | JS Engine | Go Engine | Java Engine | R |
|---|---|---|---|---|---|---|
| 跨集合提交 | `simvec_apply_txn`，操作数组走 JSON，单元素形状为带 `op` 区分的上报与删除描述 | `apply_txn`，操作字典列表透传 | `applyTxn`，操作对象列表透传 | `ApplyTxn`，操作列表透传 | `applyTxn`，操作列表透传 | `simvec_apply_txn`，操作列表透传 |

五端齐全。Rust 侧按集合分组后逐集合提交的语义保持不变。

## 读取与检索

| 能力 | C API | Python Engine | JS Engine | Go Engine | Java Engine | R |
|---|---|---|---|---|---|---|
| 通用查询 | `simvec_search` | `search`，查询字典透传为 JSON | `search`，查询对象透传为 JSON | `Search`，查询映射透传为 JSON | `search`，查询映射透传为 JSON | `simvec_search`，查询列表透传为 JSON |
| 向量快捷查询 | `simvec_search_vector` | `search_vector`，含过滤器、分数阈值、载荷与向量开关 | `searchVector`，参数语义相同 | `SearchVector`，参数语义相同 | `searchVector`，参数语义相同 | `simvec_search_vector`，参数语义相同 |
| 单点读取 | `simvec_get` | `get` | `get` | `Get` | `get` | `simvec_get` |
| 计数 | `simvec_count` | `count` | `count` | `Count` | `count` | `simvec_count`，以双精度返回 |
| 分页扫描 | `simvec_scroll` | `scroll`，偏移量可选 | `scroll`，偏移量可选 | `Scroll`，具名 `ScrollPage` 结构，偏移量可选 | `scroll`，具名 `ScrollPage` 记录，偏移量可选 | `simvec_scroll`，具名列表，偏移量可选 |

该分组五端完全对齐，无缺口。

## 维护与指标

| 能力 | C API | Python Engine | JS Engine | Go Engine | Java Engine | R |
|---|---|---|---|---|---|---|
| 构建索引 | `simvec_build_index` | `build_index` | `buildIndex` | `BuildIndex`，返回是否生效 | `buildIndex`，返回是否生效 | `simvec_build_index`，返回是否生效 |
| 构建量化 | `simvec_build_quantization` | `build_quantization` | `buildQuantization` | `BuildQuantization`，返回是否生效 | `buildQuantization`，返回是否生效 | `simvec_build_quantization`，返回是否生效 |
| 索引是否存在 | `simvec_has_index` | `has_index` | `hasIndex` | `HasIndex` | `hasIndex` | `simvec_has_index` |
| 量化是否存在 | `simvec_has_quantization` | `has_quantization` | `hasQuantization` | `HasQuantization` | `hasQuantization` | `simvec_has_quantization` |
| 丢弃索引 | `simvec_drop_index` | `drop_index` | `dropIndex` | `DropIndex` | `dropIndex` | `simvec_drop_index` |
| 丢弃已发布索引 | `simvec_drop_published_index` | `drop_published_index` | `dropPublishedIndex` | `DropPublishedIndex` | `dropPublishedIndex` | `simvec_drop_published_index` |
| 压缩回收 | `simvec_compact` | `compact`，返回回收后存活数 | `compact`，返回回收后存活数 | `Compact`，返回回收后存活数 | `compact`，返回回收后存活数 | `simvec_compact`，返回回收后存活数 |
| 排空待处理队列 | `simvec_drain_pending` | `drain_pending` | `drainPending` | `DrainPending` | `drainPending` | `simvec_drain_pending` |
| 集合指标快照 | `simvec_collection_metrics` | `collection_metrics` | `collectionMetrics` | `CollectionMetrics` | `collectionMetrics` | `simvec_collection_metrics` |
| 全局维护扫描 | `simvec_run_maintenance_sweep` | `run_maintenance_sweep` | `runMaintenanceSweep` | `RunMaintenanceSweep` | `runMaintenanceSweep` | `simvec_run_maintenance_sweep` |

该分组五端完全对齐，无缺口。

## 引擎级默认配置

| 能力 | C API | Python Engine | JS Engine | Go Engine | Java Engine | R |
|---|---|---|---|---|---|---|
| 设置默认 HNSW 配置 | `simvec_set_default_hnsw_config` | `set_default_hnsw_config` | `setDefaultHnswConfig` | `SetDefaultHnswConfig` | `setDefaultHnswConfig` | `simvec_set_default_hnsw_config` |
| 读取默认 HNSW 配置 | `simvec_get_default_hnsw_config`，未设置时返回 JSON 空值 | `get_default_hnsw_config`，未设置时返回空 | `getDefaultHnswConfig`，未设置时返回空 | `GetDefaultHnswConfig`，未设置时返回空 | `getDefaultHnswConfig`，未设置时返回空 | `simvec_get_default_hnsw_config`，未设置时返回空 |
| 设置默认 IVF 配置 | `simvec_set_default_ivf_config` | `set_default_ivf_config` | `setDefaultIvfConfig` | `SetDefaultIvfConfig` | `setDefaultIvfConfig` | `simvec_set_default_ivf_config` |
| 读取默认 IVF 配置 | `simvec_get_default_ivf_config`，未设置时返回 JSON 空值 | `get_default_ivf_config`，未设置时返回空 | `getDefaultIvfConfig`，未设置时返回空 | `GetDefaultIvfConfig`，未设置时返回空 | `getDefaultIvfConfig`，未设置时返回空 | `simvec_get_default_ivf_config`，未设置时返回空 |
| 设置默认量化配置 | `simvec_set_default_quantization_config` | `set_default_quantization_config` | `setDefaultQuantizationConfig` | `SetDefaultQuantizationConfig` | `setDefaultQuantizationConfig` | `simvec_set_default_quantization_config` |
| 读取默认量化配置 | `simvec_get_default_quantization_config`，未设置时返回 JSON 空值 | `get_default_quantization_config`，未设置时返回空 | `getDefaultQuantizationConfig`，未设置时返回空 | `GetDefaultQuantizationConfig`，未设置时返回空 | `getDefaultQuantizationConfig`，未设置时返回空 | `simvec_get_default_quantization_config`，未设置时返回空 |

五端齐全。创建集合时未指定的配置段继续沿用引擎默认值。

## 错误码映射

C 侧共 14 个状态码。五个绑定的类型化异常只覆盖其中 7 个，
其余统一降级为基类异常，原始数字码保留在消息与属性中。
划分已写入各份自述，不再需要阅读源码确认。

已具名映射的状态码：未找到、已存在、非法配置、非法维度、误用、
容量超限、过滤器错误。基类兜底的状态码：通用错误、内部错误、
内存不足、输入输出错误、数据损坏、非有限数值。

五个绑定在该行为上完全一致，基类名称分别为
`SimvecError`（Python、JavaScript、Go、Java）与 `simvec_error` 条件
（R），子类集合一一对应。Java 另提供 `codeOf` 与 `isNotFound`
系列辅助判断，Go 另提供 `CodeOf` 与 `IsNotFound` 系列辅助判断，
R 另提供 `simvec_code()` 与 `is_not_found()` 系列辅助判断。

## Rust 原生能力对照

`LocalVectorEngine` 公开方法均已进入 C API，
五个绑定同步可用。早期缺口中的跨集合事务、单字段载荷直写、
已发布索引独立丢弃、临时集合提升与备份恢复及相关列表、
引擎级默认配置读写均已补齐。`collection_names` 对应集合列表查询，
`root_dir` 由绑定构造参数保存路径覆盖，无需单独暴露。

## 绑定基础设施对比

- 动态库定位：Python 与 JavaScript 一致，优先读环境变量中的显式路径，
  其次按包相对路径查找工作区构建产物，找不到时抛基类异常并列出尝试过的路径。
  Go 走编译期搜索路径与运行期搜索路径，默认指向工作区构建产物。
  Java 由桥接加载器按系统属性、环境变量、工作区构建产物、系统库路径的顺序定位桥接库，
  并预加载环境变量指定的 Rust 动态库，找不到时列出尝试过的路径。
  R 由 `configure` 在安装期解析显式路径或工作区构建产物并写入构建变量，
  运行期走系统库搜索路径与内嵌运行路径。
- 内存归属：五端一致，入参借用、出参由库分配并由绑定层统一释放，
  调用者不直接接触释放函数。
- 数据穿越：配置、过滤器、载荷、查询走 JSON 文本，向量走连续单精度数组。
  Python 额外接受 numpy 数组，JavaScript 额外接受 `Float32Array`，
  Java 主形态为 `float[]` 另有 `double[]` 重载拷贝转换，
  Go 主形态为 `[]float32` 另有 `Float32s` 辅助转换，
  R 主形态为数值向量由桥接层拷贝转换，计数以双精度返回。
- 资源管理：Python 用上下文管理器加可重复调用的关闭，
  JavaScript 用可重复调用的关闭加显式资源管理符号与是否关闭属性，
  Go 用可重复调用的关闭加是否关闭查询与终结器兜底，
  Java 用可重复调用的关闭加是否关闭查询、资源块写法与回收器兜底，
  R 用可重复调用的关闭加是否关闭查询与终结器兜底。
- 测试覆盖：五绑定烟囱测试场景对等，均覆盖集合、写入、读取、载荷索引、
  索引与量化维护、事务、批量与过滤删除、压缩、提升与默认配置全链路。
  另有 Rust 侧 C API 回归测试覆盖空指针、非法 JSON、正常往返与新增高级路径。
- 打包分发：五绑定仅发布纯代码包，动态库由调用者先构建并指定路径，
  自述已统一推荐命令与路径写法。轮子、npm 包、发行产物内嵌动态库仍是后续工作。
- 类型描述：Python 随包提供类型存根与 `py.typed` 标记，
  JavaScript 随包提供类型声明，Java 随包提供具名 `ScrollPage` 与 `PayloadIndex` 记录，
  Go 随包提供具名 `ScrollPage` 与 `PayloadIndex` 结构，
  R 以具名列表返回分页与载荷索引；五者均与现有方法签名保持一致。

## WASM 绑定对照（内存形态）

WASM 为浏览器内存形态，与五端本机形态并存。`Rust` 侧是 `MemoryEngine` 加 `WasmEngine` 薄边界，`JS` 侧是 `bindings/wasm` 纯包。复杂值走 JSON 文本复用已有序列化形状，向量走单精度数组。符号含义与五端一致，另用“空操作”表示语义兼容但不生效，用“缺席”表示按设计不提供。

### 生命周期与错误

| 能力 | WASM | 备注 |
|---|---|---|
| 创建引擎 | 有，`new` 无路径 | 会话级内存 |
| 关闭引擎 | 有，`close` 可重复调用加 `closed` 查询 | 与 JS 一致 |
| 错误查询 | 缺席 | 调用直接抛类型化异常 |
| 类型化异常 | 有，与本机五端同码 | 七个具名加基类兜底 |

### 集合管理

| 能力 | WASM | 备注 |
|---|---|---|
| 创建集合 | 有 | 同形 |
| 删除集合 | 有 | 同形 |
| 集合是否存在 | 有 | 布尔 |
| 列出集合 | 有 | 数组 |
| 集合信息 | 有 | 解析后结构 |
| 集合配置 | 有 | 解析后结构 |
| 提升临时集合 | 缺席 | 无文件语义 |
| 恢复提升备份 | 缺席 | 无文件语义 |
| 列出临时集合 | 缺席 | 无文件语义 |
| 列出备份代际 | 缺席 | 无文件语义 |

### 写入与载荷

| 能力 | WASM | 备注 |
|---|---|---|
| 单点写入 | 有 | 同形 |
| 批量写入 | 有 | 同形 |
| 按标识删除 | 有 | 同形 |
| 批量删除 | 有 | 同形 |
| 按过滤器删除 | 有，返回计数 | 同形 |
| 整体替换载荷 | 有 | 同形 |
| 合并载荷字段 | 有 | 同形 |
| 单字段载荷直写 | 有 | 同形 |
| 删除载荷字段 | 有 | 同形 |
| 创建载荷索引 | 有，仅声明 | 检索仍精确扫描 |
| 删除载荷索引 | 有，返回是否命中 | 同形 |
| 列出载荷索引 | 有 | 同形 |

### 事务

| 能力 | WASM | 备注 |
|---|---|---|
| 跨集合提交 | 有 | 按集合排序逐集合应用，单集合失败即停 |

### 读取与检索

| 能力 | WASM | 备注 |
|---|---|---|
| 通用查询 | 有 | 同形 |
| 向量快捷查询 | 有，`JS` 侧组装通用查询 | 单一检索入口 |
| 单点读取 | 有，缺失抛未找到 | 与本机一致 |
| 计数 | 有，宿主数字 | 同形 |
| 分页扫描 | 有，回点位数组加下一页标记 | 同形 |

### 维护与指标

| 能力 | WASM | 备注 |
|---|---|---|
| 构建索引 | 空操作，返回未生效 | 内存精确扫描 |
| 构建量化 | 空操作，返回未生效 | 同上 |
| 索引是否存在 | 有，恒为假 | 同上 |
| 量化是否存在 | 有，恒为假 | 同上 |
| 丢弃索引 | 空操作，缺失集合抛未找到 | 同上 |
| 丢弃已发布索引 | 空操作，缺失集合抛未找到 | 同上 |
| 压缩回收 | 有，返回存活数 | 点位即时删除 |
| 排空待处理队列 | 空操作 | 无后台队列 |
| 集合指标快照 | 有，空快照 | 同形 |
| 全局维护扫描 | 空操作 | 无后台线程 |

### 引擎级默认配置

| 能力 | WASM | 备注 |
|---|---|---|
| 默认 HNSW 读写 | 缺席 | 按设计不提供 |
| 默认 IVF 读写 | 缺席 | 按设计不提供 |
| 默认量化读写 | 缺席 | 按设计不提供 |

### 持久化

| 能力 | WASM | 备注 |
|---|---|---|
| 导出快照 | 有，版本一 JSON | 调用者自理存储 |
| 导入快照 | 有，先校验后替换 | 损坏拒绝 |
| 修订计数 | 有，单调递增 | 每次实际变更推进，只读不推进 |
| 分集合导出导入 | 有 | 不影响其他集合 |
| 单文件 OPFS 适配 | 有，可选 `persist.js` | 临时文件加读回验证，保留备份，缺失接口时回落报错 |
| 分片目录 OPFS 适配 | 有，可选 `persist.js` | 每集合一文件加清单 |
| 存在查询与删除 | 有 | 单文件管理 |
| 配额与持久许可 | 有 | 缺失接口时显式报错 |
| 自动保存 | 有，可选 `persist.js` | 防抖加页面隐藏刷盘，后台失败走回调 |

## Go 绑定对照

Go 绑定（`bindings/go`，cgo 薄封装）与 C API 全量对齐，50 个导出函数逐一映射，
方法命名按 Go 首字母大写驼峰改写，语义与 Python 和 JavaScript 两绑定一致：
存在性查询返回布尔值，删除载荷索引返回是否命中，构建索引与构建量化返回是否生效，
过滤删除与压缩返回计数，查询与读取返回解析后的通用结构，默认配置未设置时返回空。

差异仅来自语言习惯：点标识只接受字符串，整数标识由调用者先转字符串；
向量主形态为 `[]float32`，另提供 `Float32s` 辅助转换双精度入参；
配置、过滤器、载荷、查询为可序列化值，统一序列化为 JSON 后透传；
输出的 JSON 文本解析为通用结构返回，其中分页扫描为具名 `ScrollPage` 结构，
载荷索引列表为具名 `PayloadIndex` 结构。

错误映射与现有绑定完全一致：七个状态码返回具名包装错误，
其余统一降级为 `SimvecError` 基类并保留数字码，另提供 `IsNotFound` 系列与 `CodeOf` 辅助判断。
资源管理为可重复调用的关闭加是否关闭查询与终结器兜底，另有只读的最近状态码与错误文本访问。
动态库定位走编译期搜索路径与运行期搜索路径，默认指向工作区构建产物，
调用者保持绑定与动态库来自同一检出；启用 cgo 与 C 工具链为前置要求。
冒烟测试与现有绑定场景对等，另加错误映射、重复关闭、小规模并发读写用例。

## Java 绑定对照

Java 绑定（`bindings/java`，JNI 薄封装）与 C API 全量对齐，50 个导出函数逐一映射，
方法命名按 Java 首字母小写驼峰改写，语义与 Python、JavaScript、Go 绑定一致：
存在性查询返回布尔值，删除载荷索引返回是否命中，构建索引与构建量化返回是否生效，
过滤删除与压缩返回计数，查询与读取返回解析后的通用结构，默认配置未设置时返回空。

差异仅来自语言习惯：点标识只接受字符串，整数标识由调用者先转字符串；
向量主形态为 `float[]`，另提供 `double[]` 重载拷贝转换并在自述中注明窄化语义；
配置、过滤器、载荷、查询为通用映射或列表，统一序列化为 JSON 后透传；
输出的 JSON 文本解析为通用结构返回，其中分页扫描为具名 `ScrollPage` 记录，
载荷索引列表为具名 `PayloadIndex` 记录。文本穿越统一走标准 UTF-8 字节形态，
桥接层按字节长度构造结果，不经过改写版 UTF-8。

错误映射与现有绑定完全一致：七个状态码抛出具名异常，
其余统一降级为 `SimvecError` 基类并保留数字码与 `getCode()` 访问，
另提供 `isNotFound` 系列与 `codeOf` 辅助判断。
资源管理为互斥执行下的可重复调用关闭加是否关闭查询、资源块写法与回收器兜底，
另有只读的最近状态码与错误文本访问。动态库定位由桥接加载器按系统属性、
环境变量、工作区构建产物、系统库路径的顺序完成，调用者保持绑定、
桥接库、动态库三者来自同一检出；启用 C 工具链与 JDK 头文件为前置要求。
冒烟测试与现有绑定场景对等，另加错误映射、重复关闭、小规模并发读写用例。

## R 绑定对照

R 绑定（`bindings/r`，基础 R 外部指针加 C 桥接）与 C API 全量对齐，
50 个导出函数逐一映射，函数命名按 R 小写下划线风格改写为 `simvec_*`，
语义与 Python、JavaScript、Go、Java 绑定一致：存在性查询返回逻辑值，
删除载荷索引返回是否命中，构建索引与构建量化返回是否生效，
过滤删除与压缩返回计数，查询与读取返回解析后的列表结构，
默认配置未设置时返回空。

差异仅来自语言习惯：点标识只接受字符向量，整数标识由调用者先转字符；
向量主形态为数值向量，由桥接层拷贝转换为连续单精度数组并在自述中注明窄化语义，
计数以双精度返回；配置、过滤器、载荷、查询为 R 列表，
统一经 jsonlite 序列化为 JSON 后透传；输出的 JSON 文本解析为列表返回，
其中分页扫描为含点位与下一页标记的具名列表，载荷索引列表为列表。

错误映射与现有绑定完全一致：七个状态码发出具名条件，
其余统一降级为 `simvec_error` 基类条件并保留数字码，另提供
`simvec_code()` 与 `is_not_found()` 系列辅助判断。
资源管理为可重复调用的关闭加是否关闭查询与终结器兜底，
另有只读的最近状态码与错误文本访问。动态库定位由 `configure`
在安装期解析显式路径或工作区构建产物并写入构建变量，
调用者保持绑定与动态库来自同一检出；启用 C 工具链为前置要求，
同一引擎对象不支持并发调用。冒烟测试与现有绑定场景对等。
