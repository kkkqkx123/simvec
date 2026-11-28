use std::path::Path;

use serde_json::json;
use simvec::storage::verify_snapshot;
use simvec::{restore_snapshot, LocalVectorEngine, PointId};

use crate::dto;
use crate::error::{CliError, Result};
use crate::output::Output;

pub const COMMANDS: &[(&str, &str)] = &[
    ("create <collection> [config_json]", "create a collection"),
    ("drop <collection>", "delete a collection and its data"),
    ("exists <collection>", "print 1 when the collection is loaded"),
    ("list", "list loaded collections"),
    ("info <collection>", "print collection information"),
    ("config <collection>", "print collection configuration"),
    ("upsert <collection> <points_json>", "upsert one point or an array of points"),
    ("delete <collection> <ids_json>", "delete one id or an array of ids"),
    ("delete_filter <collection> <filter_json>", "delete points matching a filter"),
    ("set_payload <collection> <id> <payload_json>", "replace a point payload"),
    ("merge_payload <collection> <id> <fields_json>", "merge payload fields"),
    ("set_field <collection> <id> <key> <value_json>", "set one payload field"),
    ("delete_keys <collection> <id> <keys_json>", "remove payload keys"),
    ("update_vectors <collection> <id> <vectors_json>", "patch named vector spaces"),
    ("index_create <collection> <field> <schema>", "create a payload field index"),
    ("index_drop <collection> <field>", "drop a payload field index"),
    ("indexes <collection>", "list payload field indexes"),
    ("search <collection> <query_json>", "run a vector query"),
    ("get <collection> <id>", "fetch one point"),
    ("count <collection>", "count live points"),
    ("count_filter <collection> <filter_json>", "count points matching a filter"),
    ("scroll <collection> <limit> [offset] [with_payload] [with_vector]", "scan a page of points"),
    (
        "scroll_filter <collection> <filter_json> <limit> [offset] [with_payload] [with_vector]",
        "scan a filtered page of points",
    ),
    ("txn <ops_json>", "commit a coordinated upsert/delete transaction"),
    ("build_index <collection>", "build and publish an index"),
    ("build_quant <collection>", "build quantized storage"),
    ("has_index <collection>", "print whether an index is published"),
    ("has_quant <collection>", "print whether quantization is ready"),
    ("drop_index <collection>", "discard the published index"),
    ("drop_published <collection>", "discard published index files"),
    ("compact <collection>", "reclaim tombstoned slots"),
    ("drain_pending <collection>", "drain pending HNSW slots"),
    ("metrics <collection>", "print the collection metrics snapshot"),
    (
        "maintenance_status <collection>",
        "print per-segment durability and the last maintenance failure",
    ),
    (
        "snapshot <collection> <dest>",
        "copy a collection's serving generation and WAL into a new directory",
    ),
    (
        "snapshot_incremental <collection> <base> <dest>",
        "snapshot that hard-links unchanged files from a verified base",
    ),
    ("verify_snapshot <src>", "validate a snapshot manifest and every file it names"),
    ("restore_snapshot <src> <dest>", "restore a verified snapshot into a new directory"),
    ("maintenance", "run one maintenance sweep"),
    ("promote <live> <temp> <suffix>", "promote a temporary collection"),
    ("restore <live>", "restore the newest promote backup"),
    ("temps", "list temporary collections"),
    ("backups <live>", "list promote backups"),
    ("set_default_hnsw <json>", "set the engine-wide HNSW defaults"),
    ("get_default_hnsw", "print the engine-wide HNSW defaults"),
    ("set_default_ivf <json>", "set the engine-wide IVF defaults"),
    ("get_default_ivf", "print the engine-wide IVF defaults"),
    ("set_default_quant <json>", "set the engine-wide quantization defaults"),
    ("get_default_quant", "print the engine-wide quantization defaults"),
];

pub fn print_help(command: Option<&str>) {
    match command {
        Some(name) => {
            if let Some((usage, summary)) = COMMANDS.iter().find(|(usage, _)| {
                usage.split_whitespace().next() == Some(name)
            }) {
                eprintln!("{usage}\n  {summary}");
            } else {
                eprintln!("unknown command: {name}");
            }
        }
        None => {
            eprintln!("engine commands:");
            for (usage, summary) in COMMANDS {
                eprintln!("  {usage}\n      {summary}");
            }
        }
    }
}

fn require<'a>(argv: &'a [String], min: usize, max: usize, usage: &str) -> Result<&'a [String]> {
    if argv.len() < min || argv.len() > max {
        Err(CliError::new(format!("usage: {usage}")))
    } else {
        Ok(argv)
    }
}

fn number_arg(value: &str, command: &str) -> Result<u64> {
    value
        .parse::<u64>()
        .map_err(|err| CliError::new(format!("{command}: invalid number {value:?}: {err}")))
}

fn flag_arg(value: &str, command: &str) -> Result<bool> {
    match value.to_ascii_lowercase().as_str() {
        "1" | "on" | "true" => Ok(true),
        "0" | "off" | "false" => Ok(false),
        _ => Err(CliError::new(format!(
            "{command}: expected on/off or 1/0, got {value:?}"
        ))),
    }
}

fn point_id(id: &str) -> PointId {
    PointId::from(id)
}

fn collection_config(
    engine: &LocalVectorEngine,
    output: &mut Output,
    name: &str,
) -> Result<()> {
    let config = engine
        .collection_config(name)?
        .ok_or_else(|| CliError::new(format!("collection not found: {name}")))?;
    output.print_json(&config)
}

pub fn execute(engine: &LocalVectorEngine, output: &mut Output, argv: &[String]) -> Result<()> {
    let command = argv[0].as_str();
    let args = &argv[1..];
    match command {
        "create" => {
            require(args, 1, 2, "create <collection> [config_json]")?;
            let config = dto::config(command, args.get(1).map(String::as_str))?;
            engine.create_collection(&args[0], &config)?;
            Ok(())
        }
        "drop" => {
            require(args, 1, 1, "drop <collection>")?;
            engine.delete_collection(&args[0])?;
            Ok(())
        }
        "exists" => {
            require(args, 1, 1, "exists <collection>")?;
            output.print_bool(engine.collection_exists(&args[0]))
        }
        "list" => {
            require(args, 0, 0, "list")?;
            output.print_json(&engine.collection_names())
        }
        "info" => {
            require(args, 1, 1, "info <collection>")?;
            output.print_json(&engine.collection_info(&args[0])?)
        }
        "config" => {
            require(args, 1, 1, "config <collection>")?;
            collection_config(engine, output, &args[0])
        }
        "upsert" => {
            require(args, 2, 2, "upsert <collection> <points_json>")?;
            let points = dto::points(command, &args[1])?;
            engine.upsert_batch(&args[0], &points)?;
            Ok(())
        }
        "delete" => {
            require(args, 2, 2, "delete <collection> <ids_json>")?;
            let ids = dto::ids(command, &args[1])?;
            engine.delete_batch(&args[0], &ids)?;
            Ok(())
        }
        "delete_filter" => {
            require(args, 2, 2, "delete_filter <collection> <filter_json>")?;
            let filter = dto::filter(command, &args[1])?;
            let deleted = engine.delete_by_filter(&args[0], &filter)?;
            output.print_u64(deleted)
        }
        "set_payload" => {
            require(args, 3, 3, "set_payload <collection> <id> <payload_json>")?;
            let payload = dto::payload(command, &args[2])?;
            engine.set_payload(&args[0], &point_id(&args[1]), payload)?;
            Ok(())
        }
        "merge_payload" => {
            require(args, 3, 3, "merge_payload <collection> <id> <fields_json>")?;
            let fields = dto::payload(command, &args[2])?;
            engine.set_payload_fields(&args[0], &point_id(&args[1]), fields)?;
            Ok(())
        }
        "set_field" => {
            require(args, 4, 4, "set_field <collection> <id> <key> <value_json>")?;
            let value = dto::value(command, &args[3])?;
            engine.set_payload_field(&args[0], &point_id(&args[1]), args[2].clone(), value)?;
            Ok(())
        }
        "delete_keys" => {
            require(args, 3, 3, "delete_keys <collection> <id> <keys_json>")?;
            let keys = dto::strings(command, &args[2])?;
            engine.delete_payload(&args[0], &point_id(&args[1]), keys)?;
            Ok(())
        }
        "update_vectors" => {
            require(args, 3, 3, "update_vectors <collection> <id> <vectors_json>")?;
            let vectors = dto::vectors(command, &args[2])?;
            engine.update_named_vectors(&args[0], &point_id(&args[1]), vectors)?;
            Ok(())
        }
        "index_create" => {
            require(args, 3, 3, "index_create <collection> <field> <schema>")?;
            let schema = dto::schema(command, &args[2])?;
            engine.create_payload_index(&args[0], &args[1], schema)?;
            Ok(())
        }
        "index_drop" => {
            require(args, 2, 2, "index_drop <collection> <field>")?;
            let existed = engine.delete_payload_index(&args[0], &args[1])?;
            output.print_bool(existed)
        }
        "indexes" => {
            require(args, 1, 1, "indexes <collection>")?;
            let indexes = engine
                .list_payload_indexes(&args[0])?
                .into_iter()
                .map(|(field, schema)| json!({ "field": field, "schema": schema.as_str() }))
                .collect::<Vec<_>>();
            output.print_json(&indexes)
        }
        "search" => {
            require(args, 2, 2, "search <collection> <query_json>")?;
            let query = dto::query(command, &args[1])?;
            output.print_json(&engine.search(&args[0], &query)?)
        }
        "get" => {
            require(args, 2, 2, "get <collection> <id>")?;
            let point = engine
                .get(&args[0], &point_id(&args[1]))?
                .ok_or_else(|| CliError::new(format!("point not found: {}", args[1])))?;
            output.print_json(&point)
        }
        "count" => {
            require(args, 1, 1, "count <collection>")?;
            output.print_u64(engine.count(&args[0])?)
        }
        "count_filter" => {
            require(args, 2, 2, "count_filter <collection> <filter_json>")?;
            let filter = dto::filter(command, &args[1])?;
            output.print_u64(engine.count_by_filter(&args[0], &filter)?)
        }
        "scroll" => {
            require(args, 2, 5, "scroll <collection> <limit> [offset] [with_payload] [with_vector]")?;
            let limit = number_arg(&args[1], command)? as usize;
            let offset = args.get(2).map(|value| point_id(value));
            let with_payload = match args.get(3) {
                Some(value) => Some(flag_arg(value, command)?),
                None => None,
            };
            let with_vector = match args.get(4) {
                Some(value) => Some(flag_arg(value, command)?),
                None => None,
            };
            let (points, next) = engine.scroll(
                &args[0],
                limit,
                offset.as_ref(),
                with_payload,
                with_vector,
            )?;
            output.print_json(&json!({
                "points": points,
                "next_offset": next.map(|id| id.to_string()),
            }))
        }
        "scroll_filter" => {
            require(
                args,
                3,
                6,
                "scroll_filter <collection> <filter_json> <limit> [offset] [with_payload] [with_vector]",
            )?;
            let filter = dto::filter(command, &args[1])?;
            let limit = number_arg(&args[2], command)? as usize;
            let offset = args.get(3).map(|value| point_id(value));
            let with_payload = match args.get(4) {
                Some(value) => Some(flag_arg(value, command)?),
                None => None,
            };
            let with_vector = match args.get(5) {
                Some(value) => Some(flag_arg(value, command)?),
                None => None,
            };
            let (points, next) = engine.scroll_by_filter(
                &args[0],
                &filter,
                limit,
                offset.as_ref(),
                with_payload,
                with_vector,
            )?;
            output.print_json(&json!({
                "points": points,
                "next_offset": next.map(|id| id.to_string()),
            }))
        }
        "txn" => {
            require(args, 1, 1, "txn <ops_json>")?;
            let ops = dto::txn_ops(command, &args[0])?;
            engine.apply_txn(ops)?;
            Ok(())
        }
        "build_index" => {
            require(args, 1, 1, "build_index <collection>")?;
            let published = engine.build_index(&args[0])?;
            output.print_bool(published)
        }
        "build_quant" => {
            require(args, 1, 1, "build_quant <collection>")?;
            let ready = engine.build_quantization(&args[0])?;
            output.print_bool(ready)
        }
        "has_index" => {
            require(args, 1, 1, "has_index <collection>")?;
            output.print_bool(engine.has_index(&args[0]))
        }
        "has_quant" => {
            require(args, 1, 1, "has_quant <collection>")?;
            output.print_bool(engine.has_quantization(&args[0]))
        }
        "drop_index" => {
            require(args, 1, 1, "drop_index <collection>")?;
            engine.drop_index(&args[0])?;
            Ok(())
        }
        "drop_published" => {
            require(args, 1, 1, "drop_published <collection>")?;
            engine.drop_published_index(&args[0])?;
            Ok(())
        }
        "compact" => {
            require(args, 1, 1, "compact <collection>")?;
            output.print_u64(engine.compact_collection(&args[0])?)
        }
        "drain_pending" => {
            require(args, 1, 1, "drain_pending <collection>")?;
            engine.drain_pending(&args[0])?;
            Ok(())
        }
        "metrics" => {
            require(args, 1, 1, "metrics <collection>")?;
            output.print_json(&engine.collection_metrics(&args[0])?)
        }
        "maintenance_status" => {
            require(args, 1, 1, "maintenance_status <collection>")?;
            output.print_json(&engine.maintenance_status(&args[0])?)
        }
        "snapshot" => {
            require(args, 2, 2, "snapshot <collection> <dest>")?;
            output.print_json(&engine.snapshot_collection(&args[0], &args[1])?)
        }
        "snapshot_incremental" => {
            require(
                args,
                3,
                3,
                "snapshot_incremental <collection> <base> <dest>",
            )?;
            output.print_json(&engine.snapshot_collection_incremental(
                &args[0], &args[1], &args[2],
            )?)
        }
        "verify_snapshot" => {
            require(args, 1, 1, "verify_snapshot <src>")?;
            output.print_json(&verify_snapshot(Path::new(&args[0]))?)
        }
        "restore_snapshot" => {
            require(args, 2, 2, "restore_snapshot <src> <dest>")?;
            restore_snapshot(&args[0], &args[1])?;
            Ok(())
        }
        "maintenance" => {
            require(args, 0, 0, "maintenance")?;
            engine.run_maintenance_sweep();
            Ok(())
        }
        "promote" => {
            require(args, 3, 3, "promote <live> <temp> <suffix>")?;
            engine.promote_temp_collection(&args[0], &args[1], &args[2])?;
            Ok(())
        }
        "restore" => {
            require(args, 1, 1, "restore <live>")?;
            output.print_json(&engine.restore_promote_backup(&args[0])?)
        }
        "temps" => {
            require(args, 0, 0, "temps")?;
            output.print_json(&engine.list_temp_collections())
        }
        "backups" => {
            require(args, 1, 1, "backups <live>")?;
            output.print_json(&engine.list_old_backups_for(&args[0]))
        }
        "set_default_hnsw" => {
            require(args, 1, 1, "set_default_hnsw <json>")?;
            let config = dto::value(command, &args[0])?;
            let config = serde_json::from_value(config)
                .map_err(|err| CliError::json(command, err))?;
            engine.set_default_hnsw_config(config);
            Ok(())
        }
        "get_default_hnsw" => {
            require(args, 0, 0, "get_default_hnsw")?;
            output.print_json(&engine.default_hnsw_config())
        }
        "set_default_ivf" => {
            require(args, 1, 1, "set_default_ivf <json>")?;
            let config = dto::value(command, &args[0])?;
            let config = serde_json::from_value(config)
                .map_err(|err| CliError::json(command, err))?;
            engine.set_default_ivf_config(config);
            Ok(())
        }
        "get_default_ivf" => {
            require(args, 0, 0, "get_default_ivf")?;
            output.print_json(&engine.default_ivf_config())
        }
        "set_default_quant" => {
            require(args, 1, 1, "set_default_quant <json>")?;
            let config = dto::value(command, &args[0])?;
            let config = serde_json::from_value(config)
                .map_err(|err| CliError::json(command, err))?;
            engine.set_default_quantization_config(config);
            Ok(())
        }
        "get_default_quant" => {
            require(args, 0, 0, "get_default_quant")?;
            output.print_json(&engine.default_quantization_config())
        }
        _ => Err(CliError::new(format!(
            "unknown command: {command}; run .help for a command list"
        ))),
    }
}
