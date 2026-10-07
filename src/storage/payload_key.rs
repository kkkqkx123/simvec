//! Payload value key normalization shared by equality pre-filters.
//!
//! Keys are normalized per the filter semantics:
//! - string payload values key on the string itself;
//! - numbers key on their numeric value (`42` and `42.0` share a key) so
//!   `MatchAny` typed comparisons agree; the `Match` string comparison is
//!   served by also keying the query's numeric string form (`"42"` →
//!   number key) while non-integral floats like `42.5` keep a distinct key;
//! - booleans key on the boolean;
//! - null, objects and nested objects are not indexed (they never satisfy
//!   an equality condition).

/// A normalized payload/filter value used as an index key.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum Key {
    Str(String),
    Num(u64),
    Bool(bool),
}

impl Key {
    /// Lookup keys for a `Match` condition value string.
    pub(crate) fn for_match(value: &str) -> Vec<Key> {
        let mut keys = vec![Key::Str(value.to_string())];
        if let Ok(n) = value.parse::<f64>() {
            keys.push(Key::Num(n.to_bits()));
        }
        match value {
            "true" => keys.push(Key::Bool(true)),
            "false" => keys.push(Key::Bool(false)),
            _ => {}
        }
        keys
    }

    /// Keys a payload value indexes under (empty for non-scalar values).
    pub(crate) fn for_value(value: &serde_json::Value) -> Vec<Key> {
        match value {
            serde_json::Value::String(s) => vec![Key::Str(s.clone())],
            serde_json::Value::Number(n) => n
                .as_f64()
                .map(|f| vec![Key::Num(f.to_bits())])
                .unwrap_or_default(),
            serde_json::Value::Bool(b) => vec![Key::Bool(*b)],
            _ => Vec::new(),
        }
    }
}

/// Extract the `(field, key)` pairs a payload registers under. Array values
/// register each element individually so `MatchAny`-style containment works.
pub(crate) fn collect_keys(field: &str, value: &serde_json::Value, out: &mut Vec<(String, Key)>) {
    if let serde_json::Value::Array(items) = value {
        for item in items {
            for key in Key::for_value(item) {
                out.push((field.to_string(), key));
            }
        }
    } else {
        for key in Key::for_value(value) {
            out.push((field.to_string(), key));
        }
    }
}
