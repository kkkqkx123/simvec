//! Full-text payload index for one field.
//!
//! A term dictionary maps normalized tokens and whole values to slot bitmaps
//! with positions and document lengths; substring and prefix predicates use
//! the stored scalar strings directly. Query-side normalization lives in
//! [`tokenize`] so registration and lookup cannot diverge.
//!
//! Masks may contain false positives (a token hit that the full filter
//! rejects), but never omit a slot the full filter would accept.

use std::collections::HashMap;

use bitvec::prelude::*;
use serde::{Deserialize, Serialize};

use crate::types::ConditionType;

/// Strings of one indexed document, split by the role the full filter gives
/// them: a scalar string supports substring matching, array elements only
/// equality and prefix matching.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct TextDocument {
    pub scalar: Vec<String>,
    pub elements: Vec<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct TextIndex {
    /// term -> (slot, positions)
    postings: HashMap<String, Vec<(u32, Vec<u32>)>>,
    lengths: HashMap<u32, u32>,
    documents: HashMap<u32, TextDocument>,
    capacity: usize,
}

impl TextIndex {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            postings: HashMap::new(),
            lengths: HashMap::new(),
            documents: HashMap::new(),
            capacity: capacity.max(1),
        }
    }

    pub(crate) fn resize(&mut self, capacity: usize) {
        self.capacity = capacity.max(1);
    }

    pub(crate) fn register_slot(&mut self, slot: u32, value: Option<&serde_json::Value>) {
        self.unregister_slot(slot);
        let Some(value) = value else {
            return;
        };
        let document = match value {
            serde_json::Value::String(text) => TextDocument {
                scalar: vec![text.clone()],
                elements: Vec::new(),
            },
            serde_json::Value::Array(items) => TextDocument {
                scalar: Vec::new(),
                elements: items.iter().filter_map(stringify).collect(),
            },
            // Scalar numbers and booleans only participate through their
            // string form. `Contains` / `StartsWith` may then give a false
            // positive for them; the full filter removes it.
            other => TextDocument {
                scalar: Vec::new(),
                elements: stringify(other).into_iter().collect(),
            },
        };
        if document.scalar.is_empty() && document.elements.is_empty() {
            return;
        }

        let mut keys: Vec<String> = Vec::new();
        for text in document.scalar.iter().chain(document.elements.iter()) {
            keys.push(text.to_lowercase());
            keys.extend(tokenize(text));
        }
        keys.sort_unstable();
        keys.dedup();
        let mut length = 0u32;
        for key in keys {
            let entry = self.postings.entry(key).or_default();
            length += 1;
            entry.push((slot, vec![length - 1]));
        }
        self.lengths.insert(slot, length);
        self.documents.insert(slot, document);
    }

    fn unregister_slot(&mut self, slot: u32) {
        if self.documents.remove(&slot).is_none() {
            return;
        }
        self.lengths.remove(&slot);
        self.postings.retain(|_, entries| {
            entries.retain(|(owner, _)| *owner != slot);
            !entries.is_empty()
        });
    }

    pub(crate) fn bits(&self, condition: &ConditionType) -> Option<BitVec> {
        let mut mask = BitVec::repeat(false, self.capacity);
        match condition {
            ConditionType::Match { value } => self.keys_for_match(value).iter().for_each(|key| {
                self.or_postings(&mut mask, key);
            }),
            ConditionType::MatchAny { values } => {
                for value in values {
                    let keys = match value {
                        serde_json::Value::String(text) => self.keys_for_match(text),
                        other => stringify(other).into_iter().collect(),
                    };
                    for key in keys {
                        self.or_postings(&mut mask, &key);
                    }
                }
            }
            ConditionType::Contains { value } => {
                for (&slot, doc) in &self.documents {
                    let hit = doc.scalar.iter().any(|text| text.contains(value.as_str()))
                        || doc.elements.iter().any(|text| text == value);
                    if hit && (slot as usize) < mask.len() {
                        mask.set(slot as usize, true);
                    }
                }
            }
            ConditionType::StartsWith { prefix } => {
                for (&slot, doc) in &self.documents {
                    let hit = doc
                        .scalar
                        .iter()
                        .chain(doc.elements.iter())
                        .any(|text| text.starts_with(prefix.as_str()));
                    if hit && (slot as usize) < mask.len() {
                        mask.set(slot as usize, true);
                    }
                }
            }
            _ => return None,
        }
        Some(mask)
    }

    fn keys_for_match(&self, value: &str) -> Vec<String> {
        let mut keys = vec![value.to_lowercase()];
        keys.extend(tokenize(value));
        keys
    }

    fn or_postings(&self, mask: &mut BitVec, key: &str) {
        if let Some(entries) = self.postings.get(key) {
            for &(slot, _) in entries {
                if (slot as usize) < mask.len() {
                    mask.set(slot as usize, true);
                }
            }
        }
    }

    /// Term dictionary with positions, for checkpoint serialization.
    pub(crate) fn export(&self) -> TextExport {
        let mut postings: Vec<_> = self
            .postings
            .iter()
            .map(|(term, entries)| (term.clone(), entries.clone()))
            .collect();
        postings.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        let mut lengths: Vec<_> = self.lengths.iter().map(|(&s, &l)| (s, l)).collect();
        lengths.sort_unstable_by_key(|(slot, _)| *slot);
        let mut documents: Vec<_> = self.documents.iter().map(|(&s, d)| (s, d.clone())).collect();
        documents.sort_unstable_by_key(|(slot, _)| *slot);
        TextExport {
            postings,
            lengths,
            documents,
        }
    }

    pub(crate) fn restore(&mut self, export: TextExport) {
        self.postings = export.postings.into_iter().collect();
        self.lengths = export.lengths.into_iter().collect();
        self.documents = export.documents.into_iter().collect();
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct TextExport {
    pub postings: Vec<(String, Vec<(u32, Vec<u32>)>)>,
    pub lengths: Vec<(u32, u32)>,
    pub documents: Vec<(u32, TextDocument)>,
}

/// Lowercase alphanumeric tokens. Both registration and lookup go through
/// this one implementation, which is the same rule used by query tokenization
/// for text fields.
pub(crate) fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|part| !part.is_empty())
        .map(|part| part.to_lowercase())
        .collect()
}

fn stringify(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(text) => Some(text.clone()),
        serde_json::Value::Number(number) => Some(number.to_string()),
        serde_json::Value::Bool(flag) => Some(flag.to_string()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn token_postings_cover_exact_and_case_variants() {
        let mut index = TextIndex::new(4);
        index.register_slot(0, Some(&json!("Hello, World")));
        index.register_slot(1, Some(&json!(["rust", "vector"])));
        let exact = index
            .bits(&ConditionType::Match {
                value: "Hello, World".to_string(),
            })
            .unwrap();
        assert!(exact[0] && !exact[1]);
        let token = index
            .bits(&ConditionType::Match {
                value: "hello".to_string(),
            })
            .unwrap();
        assert!(token[0] && !token[1]);
        let array = index
            .bits(&ConditionType::Match {
                value: "vector".to_string(),
            })
            .unwrap();
        assert!(array[1] && !array[0]);
        let world = index
            .bits(&ConditionType::Match {
                value: "world".to_string(),
            })
            .unwrap();
        assert!(world[0] && !world[1]);
    }

    #[test]
    fn substring_and_prefix_masks_are_exact() {
        let mut index = TextIndex::new(3);
        index.register_slot(0, Some(&json!("hello world")));
        index.register_slot(1, Some(&json!(["hello", "rust"])));
        let contains = index
            .bits(&ConditionType::Contains {
                value: "lo wo".to_string(),
            })
            .unwrap();
        assert!(contains[0] && !contains[1]);
        let prefix = index
            .bits(&ConditionType::StartsWith {
                prefix: "hell".to_string(),
            })
            .unwrap();
        assert!(prefix[0] && prefix[1]);
    }
}
