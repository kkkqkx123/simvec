use std::collections::HashMap;

use crate::error::Result;

use super::{MemoryEngine, MemoryTxnOp};

impl MemoryEngine {
    /// Apply a batch of cross-collection operations in sorted collection order.
    pub fn apply_txn(&self, ops: Vec<MemoryTxnOp>) -> Result<()> {
        let mut grouped: HashMap<String, Vec<MemoryTxnOp>> = HashMap::new();
        for op in ops {
            match &op {
                MemoryTxnOp::Upsert { collection, .. } | MemoryTxnOp::Delete { collection, .. } => {
                    grouped.entry(collection.clone()).or_default().push(op);
                }
            }
        }
        let mut names: Vec<String> = grouped.keys().cloned().collect();
        names.sort_unstable();
        for name in names {
            let group = grouped.remove(&name).expect("group present");
            for op in group {
                match op {
                    MemoryTxnOp::Upsert { point, .. } => {
                        self.upsert(&name, point)?;
                    }
                    MemoryTxnOp::Delete { point_id, .. } => {
                        self.delete(&name, &point_id)?;
                    }
                }
            }
        }
        Ok(())
    }
}
