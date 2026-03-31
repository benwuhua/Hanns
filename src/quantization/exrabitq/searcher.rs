use super::fastscan::scan_layout_bitmask;
use super::{
    scan_layout, ExRaBitQFastScanState, ExRaBitQLayout, ExRaBitQQuantizer, ScoredCandidate,
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
        let distance = if state.use_high_accuracy {
            quantizer.rerank_distance_high_accuracy_from_parts(
                &state.unit_query,
                state.sumq,
                state.y,
                state.y2,
                candidate.rabitq_ip,
                layout.factor_at(candidate.idx).xipnorm,
                layout.x2_at(candidate.idx),
                layout.long_code_at(candidate.idx),
            )
        } else {
            quantizer.rerank_distance_from_parts(
                &state.residual,
                state.half_sum_residual,
                state.y2,
                candidate.rabitq_ip,
                layout.factor_at(candidate.idx).xipnorm,
                layout.x2_at(candidate.idx),
                layout.long_code_at(candidate.idx),
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
    if state.use_high_accuracy {
        let _ = shortlist;
        return scan_layout_bitmask(layout, quantizer, state, top_k);
    }
    let candidates = scan_layout(layout, state, shortlist.max(top_k));
    rerank_candidates(quantizer, layout, state, &candidates, top_k)
}
