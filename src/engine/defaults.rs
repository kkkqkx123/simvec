use super::LocalVectorEngine;
use crate::types::{HnswConfig, IvfConfig};

impl LocalVectorEngine {
    /// Default IVF configuration for IVF collections created without one.
    pub fn set_default_ivf_config(&self, config: IvfConfig) {
        *self.default_ivf.write() = Some(config.clone());
        let stores = self.stores.read();
        for store in stores.values().filter_map(|h| h.as_dense()) {
            let _ = store.set_ivf_config(config.clone());
        }
    }

    /// Default HNSW configuration for HNSW collections created without one.
    pub fn set_default_hnsw_config(&self, config: HnswConfig) {
        *self.default_hnsw.write() = Some(config.clone());
        let stores = self.stores.read();
        for store in stores.values().filter_map(|h| h.as_dense()) {
            let _ = store.set_hnsw_config(config.clone());
        }
    }

    /// Default quantization configuration for collections created without one.
    pub fn set_default_quantization_config(&self, config: crate::types::QuantizationConfig) {
        *self.default_quantization.write() = Some(config.clone());
        let stores = self.stores.read();
        for store in stores.values().filter_map(|h| h.as_dense()) {
            let _ = store.set_quantization_config(config.clone());
        }
    }

    /// Current default HNSW configuration injected into HNSW collections
    /// created without one. `None` means the store-level `HnswConfig::default`
    /// applies.
    pub fn default_hnsw_config(&self) -> Option<crate::types::HnswConfig> {
        self.default_hnsw.read().clone()
    }

    /// Current default IVF configuration injected into IVF collections
    /// created without one.
    pub fn default_ivf_config(&self) -> Option<crate::types::IvfConfig> {
        self.default_ivf.read().clone()
    }

    /// Current default quantization configuration injected into collections
    /// created without one.
    pub fn default_quantization_config(&self) -> Option<crate::types::QuantizationConfig> {
        self.default_quantization.read().clone()
    }
}
