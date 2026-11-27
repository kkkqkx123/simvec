use std::collections::BTreeMap;

use serde::de::DeserializeOwned;
use serde::Deserialize;
use simvec::{
    CollectionConfig, FusionConfig, NamedVectorQuery, Payload, PayloadSchemaType, PayloadSelector,
    PointId, QuantizationSearchParams, SearchMode, SearchQuery, TxnOp, VectorData, VectorFilter,
    VectorPoint, DEFAULT_VECTOR_NAME,
};

use crate::error::{CliError, Result};

/// Command-line point shape. It accepts the same `vector` shortcut and full
/// `vectors` map as the C boundary DTOs, then converts to the public point
/// type used by the engine.
#[derive(Debug, Deserialize)]
struct PointDto {
    id: String,
    #[serde(default)]
    vector: Vec<f32>,
    #[serde(default)]
    vectors: BTreeMap<String, VectorData>,
    #[serde(default)]
    payload: Option<Payload>,
}

impl PointDto {
    fn into_point(self) -> VectorPoint {
        let vectors = if self.vectors.is_empty() {
            BTreeMap::from([(
                DEFAULT_VECTOR_NAME.to_string(),
                VectorData::Dense(self.vector),
            )])
        } else {
            self.vectors
        };
        let mut point = VectorPoint {
            id: PointId::from(self.id.as_str()),
            vectors,
            payload: self.payload,
        };
        point.normalize();
        point
    }
}

/// Command-line query shape mirroring the JSON accepted by the bindings.
#[derive(Debug, Deserialize)]
struct QueryDto {
    #[serde(default)]
    vector: Vec<f32>,
    #[serde(default)]
    data: Option<VectorData>,
    #[serde(default)]
    target: Option<String>,
    #[serde(default)]
    named_vector: Option<VectorData>,
    #[serde(default)]
    vectors: Vec<NamedVectorQuery>,
    limit: usize,
    #[serde(default)]
    offset: Option<usize>,
    #[serde(default)]
    score_threshold: Option<f32>,
    #[serde(default)]
    filter: Option<VectorFilter>,
    #[serde(default)]
    with_payload: Option<bool>,
    #[serde(default)]
    with_vector: Option<bool>,
    #[serde(default)]
    nprobe: Option<usize>,
    #[serde(default)]
    search_mode: Option<SearchMode>,
    #[serde(default)]
    payload_selector: Option<PayloadSelector>,
    #[serde(default)]
    quantization: Option<QuantizationSearchParams>,
    #[serde(default)]
    fusion: Option<FusionConfig>,
    #[serde(default)]
    vector_spaces: Option<Vec<String>>,
}

impl QueryDto {
    fn into_query(self) -> SearchQuery {
        let vectors = if !self.vectors.is_empty() {
            self.vectors
        } else if let (Some(space), Some(data)) = (self.target, self.named_vector) {
            vec![NamedVectorQuery { space, data }]
        } else if let Some(data) = self.data {
            vec![NamedVectorQuery {
                space: DEFAULT_VECTOR_NAME.to_string(),
                data,
            }]
        } else {
            vec![NamedVectorQuery {
                space: DEFAULT_VECTOR_NAME.to_string(),
                data: VectorData::Dense(self.vector),
            }]
        };
        SearchQuery {
            vectors,
            limit: self.limit,
            offset: self.offset,
            score_threshold: self.score_threshold,
            filter: self.filter,
            with_payload: self.with_payload,
            with_vector: self.with_vector,
            nprobe: self.nprobe,
            search_mode: self.search_mode,
            payload_selector: self.payload_selector,
            quantization: self.quantization,
            fusion: self.fusion,
            vector_spaces: self.vector_spaces,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum TxnOpDto {
    Upsert {
        collection: String,
        point: PointDto,
    },
    Delete {
        collection: String,
        id: String,
    },
}

fn decode<T: DeserializeOwned>(command: &str, text: &str) -> Result<T> {
    serde_json::from_str(text).map_err(|err| CliError::json(command, err))
}

fn as_value(command: &str, text: &str) -> Result<serde_json::Value> {
    decode(command, text)
}

pub fn config(command: &str, text: Option<&str>) -> Result<CollectionConfig> {
    match text {
        Some(raw) => decode(command, raw),
        None => Ok(CollectionConfig::default()),
    }
}

pub fn points(command: &str, text: &str) -> Result<Vec<VectorPoint>> {
    let value = as_value(command, text)?;
    let values = match value {
        serde_json::Value::Array(items) => items,
        item @ serde_json::Value::Object(_) => vec![item],
        _ => {
            return Err(CliError::new(format!(
                "{command}: expected a point object or an array of points"
            )))
        }
    };
    let items = values
        .into_iter()
        .map(|value| serde_json::from_value::<PointDto>(value))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|err| CliError::json(command, err))?;
    Ok(items.into_iter().map(PointDto::into_point).collect())
}

pub fn ids(command: &str, text: &str) -> Result<Vec<PointId>> {
    let value = as_value(command, text)?;
    let raw = match value {
        serde_json::Value::Array(items) => items
            .into_iter()
            .map(|item| {
                item.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| CliError::new(format!("{command}: ids must be strings")))
            })
            .collect::<Result<Vec<_>>>()?,
        serde_json::Value::String(id) => vec![id],
        _ => {
            return Err(CliError::new(format!(
                "{command}: expected an id string or an array of id strings"
            )))
        }
    };
    Ok(raw.iter().map(|id| PointId::from(id.as_str())).collect())
}

pub fn query(command: &str, text: &str) -> Result<SearchQuery> {
    decode::<QueryDto>(command, text).map(QueryDto::into_query)
}

pub fn filter(command: &str, text: &str) -> Result<VectorFilter> {
    decode(command, text)
}

pub fn payload(command: &str, text: &str) -> Result<Payload> {
    decode(command, text)
}

pub fn value(command: &str, text: &str) -> Result<serde_json::Value> {
    as_value(command, text)
}

pub fn strings(command: &str, text: &str) -> Result<Vec<String>> {
    decode(command, text)
}

pub fn vectors(command: &str, text: &str) -> Result<BTreeMap<String, VectorData>> {
    decode(command, text)
}

pub fn txn_ops(command: &str, text: &str) -> Result<Vec<TxnOp>> {
    let items: Vec<TxnOpDto> = decode(command, text)?;
    Ok(items
        .into_iter()
        .map(|item| match item {
            TxnOpDto::Upsert { collection, point } => TxnOp::Upsert {
                collection,
                point: point.into_point(),
            },
            TxnOpDto::Delete { collection, id } => TxnOp::Delete {
                collection,
                point_id: PointId::from(id.as_str()),
            },
        })
        .collect())
}

pub fn schema(command: &str, text: &str) -> Result<PayloadSchemaType> {
    match text.trim().to_ascii_lowercase().as_str() {
        "keyword" => Ok(PayloadSchemaType::Keyword),
        "integer" => Ok(PayloadSchemaType::Integer),
        "float" => Ok(PayloadSchemaType::Float),
        "text" => Ok(PayloadSchemaType::Text),
        "bool" => Ok(PayloadSchemaType::Bool),
        "geo" => Ok(PayloadSchemaType::Geo),
        "datetime" => Ok(PayloadSchemaType::Datetime),
        _ => Err(CliError::new(format!("{command}: unknown schema: {text}"))),
    }
}
