use super::fastscan::{fastscan_topk, FsCandidate, UsqFastScanState};
use super::layout::UsqLayout;
use super::quantizer::{UsqQueryState, UsqQuantizer};

/// Two-stage search: 1-bit fastscan coarse filter → B-bit rerank.
///
/// Returns up to `top_k` results as `(id, L2_distance)` sorted ascending.
///
/// # Arguments
/// - `quantizer`       — trained USQ quantizer (has config + rotator).
/// - `layout`          — SoA store holding all encoded vectors.
/// - `fs_state`        — precomputed fastscan LUT for this query.
/// - `query_state`     — precomputed query state (rotated query, centroid score, quantized).
/// - `q_norm_sq`       — ‖q_rot‖², used to convert IP score to L2 distance.
/// - `top_k`           — number of nearest neighbours to return.
pub fn scan_and_rerank(
    quantizer: &UsqQuantizer,
    layout: &UsqLayout,
    fs_state: &UsqFastScanState,
    query_state: &UsqQueryState,
    q_norm_sq: f32,
    top_k: usize,
) -> Vec<(i64, f32)> {
    if top_k == 0 || layout.is_empty() {
        return Vec::new();
    }

    let n = layout.len();

    // Tier-aware candidate count.
    let n_candidates = match quantizer.config().nbits {
        1 => (top_k * 20).max(200),
        2..=4 => (top_k * 15).max(150),
        _ => (top_k * 30).max(300),
    }
    .min(n);

    // If the entire layout fits within the candidate budget, skip fastscan.
    if n <= n_candidates {
        return brute_force_rerank(quantizer, layout, query_state, q_norm_sq, top_k);
    }

    // Stage 1: fastscan — produce coarse candidates.
    let candidates: Vec<FsCandidate> = fastscan_topk(layout, fs_state, n_candidates);

    // Stage 2: rerank with B-bit quantizer.
    let mut results: Vec<(i64, f32)> = candidates
        .iter()
        .map(|c| {
            let score = quantizer.score_with_meta(
                query_state,
                layout.norm_at(c.idx),
                layout.vmax_at(c.idx),
                layout.quant_quality_at(c.idx),
                layout.packed_bits_at(c.idx),
            );
            let dist = (q_norm_sq + layout.norm_sq_at(c.idx) - 2.0 * score).max(0.0);
            (layout.id_at(c.idx), dist)
        })
        .collect();

    results.sort_by(|a, b| a.1.total_cmp(&b.1));
    results.truncate(top_k);
    results
}

// ─── Helpers ───────────────────────────────────────────────────────────────────

/// Full brute-force rerank when the layout is smaller than the fastscan budget.
fn brute_force_rerank(
    quantizer: &UsqQuantizer,
    layout: &UsqLayout,
    query_state: &UsqQueryState,
    q_norm_sq: f32,
    top_k: usize,
) -> Vec<(i64, f32)> {
    let mut results: Vec<(i64, f32)> = (0..layout.len())
        .map(|i| {
            let score = quantizer.score_with_meta(
                query_state,
                layout.norm_at(i),
                layout.vmax_at(i),
                layout.quant_quality_at(i),
                layout.packed_bits_at(i),
            );
            let dist = (q_norm_sq + layout.norm_sq_at(i) - 2.0 * score).max(0.0);
            (layout.id_at(i), dist)
        })
        .collect();

    results.sort_by(|a, b| a.1.total_cmp(&b.1));
    results.truncate(top_k);
    results
}
