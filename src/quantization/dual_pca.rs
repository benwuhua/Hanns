use crate::api::{KnowhereError, Result};
use crate::quantization::pca::PcaTransform;

#[derive(Debug, Clone)]
pub struct DualPca {
    pub query_pca: PcaTransform,
    pub doc_pca: PcaTransform,
    pub query_dim: usize,
    pub doc_dim: usize,
}

impl DualPca {
    /// Train two PCA transforms independently for query/doc domains.
    pub fn train(
        query_samples: &[f32],
        query_n: usize,
        query_dim: usize,
        query_out_dim: usize,
        doc_samples: &[f32],
        doc_n: usize,
        doc_dim: usize,
        doc_out_dim: usize,
    ) -> Result<Self> {
        if query_dim == 0 || doc_dim == 0 {
            return Err(KnowhereError::InvalidArg(
                "DualPca requires non-zero input dims".to_string(),
            ));
        }
        if query_out_dim == 0 || query_out_dim > query_dim {
            return Err(KnowhereError::InvalidArg(format!(
                "invalid query_out_dim {} for query_dim {}",
                query_out_dim, query_dim
            )));
        }
        if doc_out_dim == 0 || doc_out_dim > doc_dim {
            return Err(KnowhereError::InvalidArg(format!(
                "invalid doc_out_dim {} for doc_dim {}",
                doc_out_dim, doc_dim
            )));
        }
        if query_samples.len() != query_n * query_dim {
            return Err(KnowhereError::InvalidArg(
                "query_samples length mismatch".to_string(),
            ));
        }
        if doc_samples.len() != doc_n * doc_dim {
            return Err(KnowhereError::InvalidArg(
                "doc_samples length mismatch".to_string(),
            ));
        }

        let query_pca = PcaTransform::train(query_samples, query_n, query_dim, query_out_dim);
        let doc_pca = PcaTransform::train(doc_samples, doc_n, doc_dim, doc_out_dim);

        Ok(Self {
            query_pca,
            doc_pca,
            query_dim: query_out_dim,
            doc_dim: doc_out_dim,
        })
    }

    #[inline]
    pub fn project_query(&self, v: &[f32]) -> Vec<f32> {
        self.query_pca.apply_one(v)
    }

    #[inline]
    pub fn project_doc(&self, v: &[f32]) -> Vec<f32> {
        self.doc_pca.apply_one(v)
    }

    pub fn project_doc_batch(&self, vectors: &[f32], n: usize) -> Vec<f32> {
        assert_eq!(
            vectors.len(),
            n * self.doc_pca.d_in,
            "doc batch length mismatch"
        );

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;

            let projected: Vec<Vec<f32>> = (0..n)
                .into_par_iter()
                .map(|i| {
                    let off = i * self.doc_pca.d_in;
                    self.project_doc(&vectors[off..off + self.doc_pca.d_in])
                })
                .collect();
            let mut out = Vec::with_capacity(n * self.doc_dim);
            for row in projected {
                out.extend_from_slice(&row);
            }
            return out;
        }

        #[cfg(not(feature = "parallel"))]
        {
            let mut out = Vec::with_capacity(n * self.doc_dim);
            for i in 0..n {
                let off = i * self.doc_pca.d_in;
                out.extend_from_slice(&self.project_doc(&vectors[off..off + self.doc_pca.d_in]));
            }
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DualPca;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn l2(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let d = x - y;
                d * d
            })
            .sum()
    }

    fn top_k_ids(data: &[f32], n: usize, dim: usize, query: &[f32], k: usize) -> Vec<usize> {
        let mut scored: Vec<(f32, usize)> = (0..n)
            .map(|i| {
                let off = i * dim;
                (l2(query, &data[off..off + dim]), i)
            })
            .collect();
        scored.sort_by(|a, b| a.0.total_cmp(&b.0));
        scored.into_iter().take(k).map(|(_, i)| i).collect()
    }

    #[test]
    fn test_dual_pca_shapes() {
        let mut rng = StdRng::seed_from_u64(42);
        let n = 128usize;
        let dim = 64usize;
        let out_dim = 32usize;
        let data: Vec<f32> = (0..n * dim).map(|_| rng.gen::<f32>()).collect();

        let dual = DualPca::train(&data, n, dim, out_dim, &data, n, dim, out_dim).unwrap();
        let q = dual.project_query(&data[..dim]);
        let docs = dual.project_doc_batch(&data, n);

        assert_eq!(dual.query_dim, out_dim);
        assert_eq!(dual.doc_dim, out_dim);
        assert_eq!(q.len(), out_dim);
        assert_eq!(docs.len(), n * out_dim);
    }

    #[test]
    fn test_dual_pca_recall_at_10() {
        let mut rng = StdRng::seed_from_u64(7);
        let n_docs = 512usize;
        let n_queries = 64usize;
        let latent_dim = 12usize;
        let dim = 64usize;
        let out_dim = 32usize;
        let top_k = 10usize;

        let basis: Vec<f32> = (0..latent_dim * dim)
            .map(|_| rng.gen_range(-1.0f32..1.0f32))
            .collect();

        let make_vec = |rng: &mut StdRng| -> Vec<f32> {
            let latent: Vec<f32> = (0..latent_dim)
                .map(|_| rng.gen_range(-1.0f32..1.0f32))
                .collect();
            let mut out = vec![0.0f32; dim];
            for d in 0..dim {
                let mut sum = 0.0f32;
                for z in 0..latent_dim {
                    sum += latent[z] * basis[z * dim + d];
                }
                out[d] = sum + rng.gen_range(-0.01f32..0.01f32);
            }
            out
        };

        let mut docs = Vec::with_capacity(n_docs * dim);
        for _ in 0..n_docs {
            docs.extend_from_slice(&make_vec(&mut rng));
        }
        let mut queries = Vec::with_capacity(n_queries * dim);
        for _ in 0..n_queries {
            queries.extend_from_slice(&make_vec(&mut rng));
        }

        let dual =
            DualPca::train(&docs, n_docs, dim, out_dim, &docs, n_docs, dim, out_dim).unwrap();
        let doc_proj = dual.project_doc_batch(&docs, n_docs);

        let mut hits = 0usize;
        for qi in 0..n_queries {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let q_proj = dual.project_query(q);
            let gt = top_k_ids(&docs, n_docs, dim, q, top_k);
            let approx = top_k_ids(&doc_proj, n_docs, out_dim, &q_proj, top_k);
            for id in gt {
                if approx.contains(&id) {
                    hits += 1;
                }
            }
        }

        let recall = hits as f32 / (n_queries * top_k) as f32;
        assert!(recall >= 0.7, "dual-pca recall@10 too low: {:.3}", recall);
    }
}
