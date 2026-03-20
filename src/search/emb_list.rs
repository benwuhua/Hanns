use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

#[derive(Clone, Copy, Debug)]
struct ScoreDoc {
    score: f32,
    doc_id: usize,
}

impl PartialEq for ScoreDoc {
    fn eq(&self, other: &Self) -> bool {
        self.score.total_cmp(&other.score) == Ordering::Equal && self.doc_id == other.doc_id
    }
}

impl Eq for ScoreDoc {}

impl PartialOrd for ScoreDoc {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoreDoc {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| self.doc_id.cmp(&other.doc_id))
    }
}

/// Multi-vector document collection (ColBERT-style late interaction).
pub struct EmbeddingList {
    pub dim: usize,
    pub metric: String,
    pub doc_offsets: Vec<usize>,
    pub vectors: Vec<f32>,
}

impl EmbeddingList {
    /// Construct an empty embedding list.
    pub fn new(dim: usize, metric: impl Into<String>) -> Self {
        Self {
            dim,
            metric: metric.into(),
            doc_offsets: vec![0],
            vectors: Vec::new(),
        }
    }

    /// Append one document's vectors (flat, n_vecs * dim).
    pub fn add_doc(&mut self, vecs: &[f32]) {
        assert!(self.dim > 0, "dim must be > 0");
        assert!(
            vecs.len().is_multiple_of(self.dim),
            "vecs length {} must be multiple of dim {}",
            vecs.len(),
            self.dim
        );
        let n_vecs = vecs.len() / self.dim;
        self.vectors.extend_from_slice(vecs);
        let next = self.doc_offsets[self.doc_offsets.len() - 1] + n_vecs;
        self.doc_offsets.push(next);
    }

    /// Number of documents.
    pub fn num_docs(&self) -> usize {
        self.doc_offsets.len().saturating_sub(1)
    }

    /// Search top-k docs by MAX_SIM score (descending).
    pub fn search(&self, query_vecs: &[f32], top_k: usize) -> Vec<(usize, f32)> {
        if self.dim == 0 || top_k == 0 || self.num_docs() == 0 || query_vecs.is_empty() {
            return Vec::new();
        }
        if !query_vecs.len().is_multiple_of(self.dim) {
            return Vec::new();
        }

        let scores = self.score_all(query_vecs);
        if scores.is_empty() {
            return Vec::new();
        }

        let k = top_k.min(scores.len());
        if k == scores.len() {
            let mut all: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
            all.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            return all;
        }

        let mut heap: BinaryHeap<Reverse<ScoreDoc>> = BinaryHeap::with_capacity(k + 1);
        for (doc_id, score) in scores.into_iter().enumerate() {
            let cur = Reverse(ScoreDoc { score, doc_id });
            if heap.len() < k {
                heap.push(cur);
                continue;
            }
            if let Some(worst) = heap.peek() {
                if score > worst.0.score {
                    let _ = heap.pop();
                    heap.push(cur);
                }
            }
        }

        let mut out = Vec::with_capacity(heap.len());
        while let Some(Reverse(item)) = heap.pop() {
            out.push((item.doc_id, item.score));
        }
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        out
    }

    /// Calculate MAX_SIM scores for selected doc ids (same order as input ids).
    pub fn calc_dist_by_ids(&self, query_vecs: &[f32], doc_ids: &[usize]) -> Vec<f32> {
        if self.dim == 0 || query_vecs.is_empty() || !query_vecs.len().is_multiple_of(self.dim) {
            return vec![f32::NEG_INFINITY; doc_ids.len()];
        }
        doc_ids
            .iter()
            .map(|&doc_id| {
                if doc_id >= self.num_docs() {
                    return f32::NEG_INFINITY;
                }
                let start = self.doc_offsets[doc_id];
                let end = self.doc_offsets[doc_id + 1];
                if end <= start {
                    return f32::NEG_INFINITY;
                }
                let s = start * self.dim;
                let e = end * self.dim;
                crate::search::max_sim::max_sim(&self.metric, query_vecs, &self.vectors[s..e], self.dim)
            })
            .collect()
    }

    fn score_all(&self, query_vecs: &[f32]) -> Vec<f32> {
        let n_docs = self.num_docs();
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            (0..n_docs)
                .into_par_iter()
                .map(|doc_id| {
                    let start = self.doc_offsets[doc_id];
                    let end = self.doc_offsets[doc_id + 1];
                    if end <= start {
                        return f32::NEG_INFINITY;
                    }
                    let s = start * self.dim;
                    let e = end * self.dim;
                    crate::search::max_sim::max_sim(
                        &self.metric,
                        query_vecs,
                        &self.vectors[s..e],
                        self.dim,
                    )
                })
                .collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            (0..n_docs)
                .map(|doc_id| {
                    let start = self.doc_offsets[doc_id];
                    let end = self.doc_offsets[doc_id + 1];
                    if end <= start {
                        return f32::NEG_INFINITY;
                    }
                    let s = start * self.dim;
                    let e = end * self.dim;
                    crate::search::max_sim::max_sim(
                        &self.metric,
                        query_vecs,
                        &self.vectors[s..e],
                        self.dim,
                    )
                })
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_emb_list() -> EmbeddingList {
        let mut e = EmbeddingList::new(2, "ip");
        // doc0: moderately aligned with query [1, 0]
        e.add_doc(&[0.6, 0.0, 0.5, 0.1]);
        // doc1: best aligned with query [1, 0]
        e.add_doc(&[1.0, 0.0, 0.9, 0.1]);
        // doc2: mostly orthogonal
        e.add_doc(&[0.0, 1.0, 0.1, 0.9]);
        e
    }

    #[test]
    fn test_emb_list_basic() {
        let e = sample_emb_list();
        let query = vec![1.0, 0.0];
        let res = e.search(&query, 3);
        assert_eq!(res[0].0, 1);
        assert!(res[0].1 >= res[1].1);
    }

    #[test]
    fn test_emb_list_calc_dist_by_ids() {
        let e = sample_emb_list();
        let query = vec![1.0, 0.0];
        let scores = e.calc_dist_by_ids(&query, &[0, 1, 2]);
        assert_eq!(scores.len(), 3);
        assert!(scores[1] > scores[0]);
        assert!(scores[0] > scores[2]);
    }

    #[test]
    fn test_emb_list_top_k() {
        let mut e = EmbeddingList::new(2, "ip");
        e.add_doc(&[0.2, 0.0, 0.2, 0.0]); // doc0
        e.add_doc(&[0.4, 0.0, 0.4, 0.0]); // doc1
        e.add_doc(&[0.6, 0.0, 0.6, 0.0]); // doc2
        e.add_doc(&[0.8, 0.0, 0.8, 0.0]); // doc3
        e.add_doc(&[1.0, 0.0, 1.0, 0.0]); // doc4

        let query = vec![1.0, 0.0];
        let top2 = e.search(&query, 2);
        assert_eq!(top2.len(), 2);
        assert!(top2[0].1 >= top2[1].1);
        assert_eq!(top2[0].0, 4);
        assert_eq!(top2[1].0, 3);
    }
}
