use crate::api::IndexConfig;

use super::xndescent::XNDescentConfig;

#[derive(Debug, Clone)]
pub struct RhtsdgConfig {
    pub alpha: f32,
    pub occ_threshold: u32,
    pub knn_k: usize,
    pub nndescent: XNDescentConfig,
}

impl RhtsdgConfig {
    pub fn from_index_config(config: &IndexConfig) -> Self {
        let num_points_hint = config.params.rhtsdg_knn_k.unwrap_or(16);
        Self {
            alpha: config.params.rhtsdg_alpha.unwrap_or(1.2),
            occ_threshold: config.params.rhtsdg_occ_threshold.unwrap_or(4),
            knn_k: num_points_hint.max(1),
            nndescent: XNDescentConfig {
                k: num_points_hint.max(1),
                sample_count: config
                    .params
                    .rhtsdg_sample_count
                    .unwrap_or(num_points_hint.min(8).max(1)),
                iter_count: config.params.rhtsdg_iter_count.unwrap_or(10),
                reverse_count: config.params.rhtsdg_reverse_count.unwrap_or(8),
                use_shortcut: config.params.rhtsdg_use_shortcut.unwrap_or(false),
            },
        }
    }
}
