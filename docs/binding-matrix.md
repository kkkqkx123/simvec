# 绑定功能矩阵：C API / Python / JavaScript

统计口径：C 侧以 `include/simvec.h` 与 `src/c_api` 对照为准，两者一致，共 50 个导出函数。
Python 侧看 `bindings/python/simvec/engine.py` 的 `Engine` 方法与 `_native.py` 的签名声明。
JavaScript 侧看 `bindings/js/src/engine.js` 的 `Engine` 方法与 `native.js` 的签名声明。
`Rust` 原生能力以 `src/engine.rs` 的 `LocalVectorEngine` 公开方法为准。

符号说明：表格中用“有”表示该层已封装可用，用“缺”表示该层未封装，
用“内部”表示仅在绑定实现内部使用、不对调用者暴露。

## 生命周期与错误原语

| 能力 | C API | Python Engine | JS Engine | 备注 |
|---|---|---|---|---|
| 打开引擎 | `simvec_open` | `Engine(path)` 构造器 | `new Engine(dirPath)` | 三端齐全 |
| 关闭引擎 | `simvec_close` | `close`，上下文管理器与析构兜底 | `close`，`closed` 属性与显式资源管理 | 三端齐全 |
| 释放返回字符串 | `simvec_free_string` | 内部调用 | 内部调用 | 调用者不可见是正确设计 |
| 查询最近错误码 | `simvec_errcode` | `last_code` 只读属性 | `lastCode` 只读属性 | 异常之外可主动查询 |
| 查询最近错误文本 | `simvec_last_error` | `last_message` 只读属性 | `lastMessage` 只读属性 | 同上 |
| 无句柄错误文本 | `simvec_last_error_text` | 打开失败时内部读取 | 打开失败时内部读取 | 同上 |

## 集合管理

| 能力 | C API | Python Engine | JS Engine |
|---|---|---|---|
| 创建集合 | `simvec_create_collection` | `create_collection` | `createCollection` |
| 删除集合 | `simvec_delete_collection` | `delete_collection` | `deleteCollection` |
| 集合是否存在 | `simvec_collection_exists` | `collection_exists` | `collectionExists` |
| 列出集合 | `simvec_list_collections` | `list_collections` | `listCollections` |
| 集合信息 | `simvec_collection_info` | `collection_info` | `collectionInfo` |
| 集合配置 | `simvec_collection_config` | `collection_config` | `collectionConfig` |
| 提升临时集合 | `simvec_promote_temp_collection` | `promote_temp_collection` | `promoteTempCollection` |
| 恢复提升备份 | `simvec_restore_promote_backup` | `restore_promote_backup`，返回备份名 | `restorePromoteBackup`，返回备份名 |
| 列出临时集合 | `simvec_list_temp_collections` | `list_temp_collections` | `listTempCollections` |
| 列出备份代际 | `simvec_list_old_backups` | `list_old_backups` | `listOldBackups` |

该分组三端完全对齐，无缺口。

## 写入与载荷

| 能力 | C API | Python Engine | JS Engine |
|---|---|---|---|
| 单点写入 | `simvec_upsert` | `upsert`，支持任意浮点序列与 numpy 连续数组 | `upsert`，`Float32Array` 零拷贝，其余序列拷贝转换 |
| 批量写入 | `simvec_upsert_batch` | `upsert_batch` | `upsertBatch` |
| 按标识删除 | `simvec_delete` | `delete` | `delete` |
| 批量删除 | `simvec_delete_batch` | `delete_batch` | `deleteBatch` |
| 按过滤器删除 | `simvec_delete_by_filter` | `delete_by_filter`，返回删除计数 | `deleteByFilter`，返回删除计数 |
| 整体替换载荷 | `simvec_set_payload` | `set_payload` | `setPayload` |
| 合并载荷字段 | `simvec_merge_payload` | `merge_payload` | `mergePayload` |
| 单字段载荷直写 | `simvec_set_payload_field` | `set_payload_field`，值接受任意可序列化对象 | `setPayloadField`，值接受任意可序列化对象 |
| 删除载荷字段 | `simvec_delete_payload_keys` | `delete_payload_keys` | `deletePayloadKeys` |
| 创建载荷索引 | `simvec_create_payload_index` | `create_payload_index` | `createPayloadIndex` |
| 删除载荷索引 | `simvec_delete_payload_index` | `delete_payload_index`，返回是否命中 | `deletePayloadIndex`，返回是否命中 |
| 列出载荷索引 | `simvec_list_payload_indexes` | `list_payload_indexes` | `listPayloadIndexes` |

该分组三端完全对齐，无缺口。向量缓冲仅在调用期间持有，
不再随引擎实例长期累积。

## 事务

| 能力 | C API | Python Engine | JS Engine |
|---|---|---|---|
| 跨集合提交 | `simvec_apply_txn`，操作数组走 JSON，单元素形状为带 `op` 区分的上报与删除描述 | `apply_txn`，操作字典列表透传 | `applyTxn`，操作对象列表透传 |

三端齐全。Rust 侧按集合分组后逐集合提交的语义保持不变。

## 读取与检索

| 能力 | C API | Python Engine | JS Engine |
|---|---|---|---|
| 通用查询 | `simvec_search` | `search`，查询字典透传为 JSON | `search`，查询对象透传为 JSON |
| 向量快捷查询 | `simvec_search_vector` | `search_vector`，含过滤器、分数阈值、载荷与向量开关 | `searchVector`，参数语义相同 |
| 单点读取 | `simvec_get` | `get` | `get` |
| 计数 | `simvec_count` | `count` | `count` |
| 分页扫描 | `simvec_scroll` | `scroll`，偏移量可选 | `scroll`，偏移量可选 |

该分组三端完全对齐，无缺口。

## 维护与指标

| 能力 | C API | Python Engine | JS Engine |
|---|---|---|---|
| 构建索引 | `simvec_build_index` | `build_index` | `buildIndex` |
| 构建量化 | `simvec_build_quantization` | `build_quantization` | `buildQuantization` |
| 索引是否存在 | `simvec_has_index` | `has_index` | `hasIndex` |
| 量化是否存在 | `simvec_has_quantization` | `has_quantization` | `hasQuantization` |
| 丢弃索引 | `simvec_drop_index` | `drop_index` | `dropIndex` |
| 丢弃已发布索引 | `simvec_drop_published_index` | `drop_published_index` | `dropPublishedIndex` |
| 压缩回收 | `simvec_compact` | `compact`，返回回收后存活数 | `compact`，返回回收后存活数 |
| 排空待处理队列 | `simvec_drain_pending` | `drain_pending` | `drainPending` |
| 集合指标快照 | `simvec_collection_metrics` | `collection_metrics` | `collectionMetrics` |
| 全局维护扫描 | `simvec_run_maintenance_sweep` | `run_maintenance_sweep` | `runMaintenanceSweep` |

该分组三端完全对齐，无缺口。

## 引擎级默认配置

| 能力 | C API | Python Engine | JS Engine |
|---|---|---|---|
| 设置默认 HNSW 配置 | `simvec_set_default_hnsw_config` | `set_default_hnsw_config` | `setDefaultHnswConfig` |
| 读取默认 HNSW 配置 | `simvec_get_default_hnsw_config`，未设置时返回 JSON 空值 | `get_default_hnsw_config`，未设置时返回空 | `getDefaultHnswConfig`，未设置时返回空 |
| 设置默认 IVF 配置 | `simvec_set_default_ivf_config` | `set_default_ivf_config` | `setDefaultIvfConfig` |
| 读取默认 IVF 配置 | `simvec_get_default_ivf_config`，未设置时返回 JSON 空值 | `get_default_ivf_config`，未设置时返回空 | `getDefaultIvfConfig`，未设置时返回空 |
| 设置默认量化配置 | `simvec_set_default_quantization_config` | `set_default_quantization_config` | `setDefaultQuantizationConfig` |
| 读取默认量化配置 | `simvec_get_default_quantization_config`，未设置时返回 JSON 空值 | `get_default_quantization_config`，未设置时返回空 | `getDefaultQuantizationConfig`，未设置时返回空 |

三端齐全。创建集合时未指定的配置段继续沿用引擎默认值。

## 错误码映射

C 侧共 14 个状态码。两个绑定的类型化异常只覆盖其中 7 个，
其余统一降级为基类异常，原始数字码保留在消息与属性中。
划分已写入两份自述，不再需要阅读源码确认。

已具名映射的状态码：未找到、已存在、非法配置、非法维度、误用、
容量超限、过滤器错误。基类兜底的状态码：通用错误、内部错误、
内存不足、输入输出错误、数据损坏、非有限数值。

Python 与 JavaScript 在该行为上完全一致，基类名称分别为
`SimvecError`，子类集合一一对应。

## Rust 原生能力对照

`LocalVectorEngine` 公开方法均已进入 C API，
两绑定同步可用。早期缺口中的跨集合事务、单字段载荷直写、
已发布索引独立丢弃、临时集合提升与备份恢复及相关列表、
引擎级默认配置读写均已补齐。`collection_names` 对应集合列表查询，
`root_dir` 由绑定构造参数保存路径覆盖，无需单独暴露。

## 绑定基础设施对比

- 动态库定位：两绑定一致，优先读环境变量中的显式路径，
  其次按包相对路径查找工作区构建产物，找不到时抛基类异常并列出尝试过的路径。
- 内存归属：两绑定一致，入参借用、出参由库分配并由绑定层统一释放，
  调用者不直接接触释放函数。
- 数据穿越：配置、过滤器、载荷、查询走 JSON 文本，向量走连续单精度数组。
  Python 额外接受 numpy 数组，JavaScript 额外接受 `Float32Array`。
- 资源管理：Python 用上下文管理器加可重复调用的关闭，
  JavaScript 用可重复调用的关闭加显式资源管理符号与是否关闭属性。
- 测试覆盖：两绑定烟囱测试场景对等，均覆盖集合、写入、读取、载荷索引、
  索引与量化维护、事务、批量与过滤删除、压缩、提升与默认配置全链路。
  另有 Rust 侧 C API 回归测试覆盖空指针、非法 JSON、正常往返与新增高级路径。
- 打包分发：两绑定仅发布纯代码包，动态库由调用者先构建并指定路径，
  自述已统一推荐命令与环境变量写法。轮子与 npm 包内嵌动态库仍是后续工作。
- 类型描述：Python 随包提供类型存根与 `py.typed` 标记，
  JavaScript 随包提供类型声明，两者与现有方法签名保持一致。
