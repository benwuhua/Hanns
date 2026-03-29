use super::{
    scalar_scan_layout, ExRaBitQFastScanState, ExRaBitQLayout, ExRaBitQQuantizer, ScoredCandidate,
};

pub fn rerank_candidates(
    quantizer: &ExRaBitQQuantizer,
    layout: &ExRaBitQLayout,
    state: &ExRaBitQFastScanState,
    candidates: &[ScoredCandidate],
    top_k: usize,
) -> Vec<(i64, f32)> {
    let mut reranked = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let encoded = super::EncodedVector {
            short_code: layout.short_code_at(candidate.idx).to_vec(),
            long_code: layout.long_code_at(candidate.idx).to_vec(),
            factor: layout.factor_at(candidate.idx),
            short_factors: super::ExShortFactors {
                ip: layout.short_ip_at(candidate.idx),
                sum_xb: layout.short_sum_xb_at(candidate.idx),
                err: layout.short_err_at(candidate.idx),
            },
            x_norm: layout.x_norm_at(candidate.idx),
            x2: layout.x2_at(candidate.idx),
        };
        let distance = if state.use_high_accuracy {
            quantizer.rerank_distance_high_accuracy(
                &state.unit_query,
                state.sumq,
                state.y,
                state.y2,
                candidate.rabitq_ip,
                &encoded,
            )
        } else {
            quantizer.rerank_distance(
                &state.residual,
                state.half_sum_residual,
                state.y2,
                candidate.rabitq_ip,
                &encoded,
            )
        };
        reranked.push((layout.id_at(candidate.idx), distance));
    }
    reranked.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    reranked.truncate(top_k);
    reranked
}

pub fn scan_and_rerank(
    quantizer: &ExRaBitQQuantizer,
    layout: &ExRaBitQLayout,
    state: &ExRaBitQFastScanState,
    shortlist: usize,
    top_k: usize,
) -> Vec<(i64, f32)> {
    let candidates = scalar_scan_layout(layout, state, shortlist.max(top_k));
    rerank_candidates(quantizer, layout, state, &candidates, top_k)
}
