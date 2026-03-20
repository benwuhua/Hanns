//! MAX_SIM distance for multi-vector (late interaction) retrieval.
//! Used in ColBERT-style models where each document/query has multiple token embeddings.

/// Compute MAX_SIM score between a query embedding set and a document embedding set.
///
/// score = Σ_i max_j sim(q_i, d_j)
///
/// Returns a similarity score (higher = more similar).
pub fn max_sim_ip(q_vecs: &[f32], d_vecs: &[f32], dim: usize) -> f32 {
    if dim == 0 || q_vecs.is_empty() || d_vecs.is_empty() {
        return 0.0;
    }
    let nq = q_vecs.len() / dim;
    let nd = d_vecs.len() / dim;
    if nq == 0 || nd == 0 {
        return 0.0;
    }

    let mut total = 0.0f32;
    for qi in 0..nq {
        let q = &q_vecs[qi * dim..(qi + 1) * dim];
        let mut best = f32::NEG_INFINITY;
        for di in 0..nd {
            let d = &d_vecs[di * dim..(di + 1) * dim];
            let score: f32 = q.iter().zip(d.iter()).map(|(a, b)| a * b).sum();
            if score > best {
                best = score;
            }
        }
        total += best;
    }
    total
}

/// MAX_SIM with cosine similarity (normalized IP)
pub fn max_sim_cosine(q_vecs: &[f32], d_vecs: &[f32], dim: usize) -> f32 {
    if dim == 0 || q_vecs.is_empty() || d_vecs.is_empty() {
        return 0.0;
    }
    let nq = q_vecs.len() / dim;
    let nd = d_vecs.len() / dim;
    if nq == 0 || nd == 0 {
        return 0.0;
    }

    let mut total = 0.0f32;
    for qi in 0..nq {
        let q = &q_vecs[qi * dim..(qi + 1) * dim];
        let q_norm = q.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        let mut best = f32::NEG_INFINITY;
        for di in 0..nd {
            let d = &d_vecs[di * dim..(di + 1) * dim];
            let d_norm = d.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
            let dot: f32 = q.iter().zip(d.iter()).map(|(a, b)| a * b).sum();
            let score = dot / (q_norm * d_norm);
            if score > best {
                best = score;
            }
        }
        total += best;
    }
    total
}

/// MAX_SIM with negative L2 (higher = closer)
pub fn max_sim_l2(q_vecs: &[f32], d_vecs: &[f32], dim: usize) -> f32 {
    if dim == 0 || q_vecs.is_empty() || d_vecs.is_empty() {
        return 0.0;
    }
    let nq = q_vecs.len() / dim;
    let nd = d_vecs.len() / dim;
    if nq == 0 || nd == 0 {
        return 0.0;
    }

    let mut total = 0.0f32;
    for qi in 0..nq {
        let q = &q_vecs[qi * dim..(qi + 1) * dim];
        let mut best = f32::NEG_INFINITY;
        for di in 0..nd {
            let d = &d_vecs[di * dim..(di + 1) * dim];
            let sq_dist: f32 = q
                .iter()
                .zip(d.iter())
                .map(|(a, b)| {
                    let diff = a - b;
                    diff * diff
                })
                .sum();
            let score = -sq_dist;
            if score > best {
                best = score;
            }
        }
        total += best;
    }
    total
}

/// MAX_SIM with Hamming distance on binary vectors (represented as f32, ±1 or 0/1)
/// score = max(-hamming(q_i, d_j)) = max(matches - mismatches)
pub fn max_sim_hamming(q_vecs: &[f32], d_vecs: &[f32], dim: usize) -> f32 {
    if dim == 0 || q_vecs.is_empty() || d_vecs.is_empty() {
        return 0.0;
    }
    let nq = q_vecs.len() / dim;
    let nd = d_vecs.len() / dim;
    if nq == 0 || nd == 0 {
        return 0.0;
    }

    let mut total = 0.0f32;
    for qi in 0..nq {
        let q = &q_vecs[qi * dim..(qi + 1) * dim];
        let mut best = f32::NEG_INFINITY;
        for di in 0..nd {
            let d = &d_vecs[di * dim..(di + 1) * dim];
            let matches = q
                .iter()
                .zip(d.iter())
                .filter(|(a, b)| (**a >= 0.0) == (**b >= 0.0))
                .count() as f32;
            let score = matches - (dim as f32 - matches);
            if score > best {
                best = score;
            }
        }
        total += best;
    }
    total
}

/// MAX_SIM with Jaccard similarity (binary vectors as f32)
pub fn max_sim_jaccard(q_vecs: &[f32], d_vecs: &[f32], dim: usize) -> f32 {
    if dim == 0 || q_vecs.is_empty() || d_vecs.is_empty() {
        return 0.0;
    }
    let nq = q_vecs.len() / dim;
    let nd = d_vecs.len() / dim;
    if nq == 0 || nd == 0 {
        return 0.0;
    }

    let mut total = 0.0f32;
    for qi in 0..nq {
        let q = &q_vecs[qi * dim..(qi + 1) * dim];
        let mut best = f32::NEG_INFINITY;
        for di in 0..nd {
            let d = &d_vecs[di * dim..(di + 1) * dim];
            let intersection = q
                .iter()
                .zip(d.iter())
                .filter(|(a, b)| **a >= 0.0 && **b >= 0.0)
                .count() as f32;
            let union = q
                .iter()
                .zip(d.iter())
                .filter(|(a, b)| **a >= 0.0 || **b >= 0.0)
                .count() as f32;
            let score = if union > 0.0 {
                intersection / union
            } else {
                0.0
            };
            if score > best {
                best = score;
            }
        }
        total += best;
    }
    total
}

/// Dispatch MAX_SIM by metric name string
pub fn max_sim(metric: &str, q_vecs: &[f32], d_vecs: &[f32], dim: usize) -> f32 {
    match metric.to_ascii_lowercase().as_str() {
        "ip" | "inner_product" => max_sim_ip(q_vecs, d_vecs, dim),
        "cosine" => max_sim_cosine(q_vecs, d_vecs, dim),
        "l2" => max_sim_l2(q_vecs, d_vecs, dim),
        "hamming" => max_sim_hamming(q_vecs, d_vecs, dim),
        "jaccard" => max_sim_jaccard(q_vecs, d_vecs, dim),
        _ => max_sim_ip(q_vecs, d_vecs, dim),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_sim_ip_identical() {
        let dim = 4usize;
        let vecs = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let score = max_sim_ip(&vecs, &vecs, dim);
        assert!((score - 2.0).abs() < 1e-5, "score={}", score);
        println!("max_sim_ip identical: {}", score);
    }

    #[test]
    fn test_max_sim_ip_cross() {
        let dim = 4usize;
        let q = vec![1.0f32, 0.0, 0.0, 0.0];
        let d = vec![0.0f32, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let score = max_sim_ip(&q, &d, dim);
        assert!((score - 1.0).abs() < 1e-5, "score={}", score);
        println!("max_sim_ip cross: {}", score);
    }

    #[test]
    fn test_max_sim_dispatch() {
        let dim = 2usize;
        let q = vec![1.0f32, 0.0];
        let d = vec![1.0f32, 0.0];
        let score_ip = max_sim("ip", &q, &d, dim);
        let score_cos = max_sim("cosine", &q, &d, dim);
        assert!(score_ip > 0.0);
        assert!((score_cos - 1.0).abs() < 1e-5);
        println!("max_sim dispatch: ip={} cosine={}", score_ip, score_cos);
    }
}
