use crate::{api::MetricType, simd};
use std::collections::HashSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub enum DistanceBackend {
    L2FastPath,
    Fallback,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TsdgTrace {
    pub stage1_pairs_checked: usize,
    pub stage2_reverse_candidates: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct DistanceMatrix<'a> {
    dim: usize,
    point_count: usize,
    points: &'a [f32],
    node_ids: Option<&'a [u32]>,
    metric: MetricType,
    l2_distance_sq_ptr_kernel: Option<simd::L2DistanceSqPtrKernel>,
}

impl<'a> DistanceMatrix<'a> {
    pub fn from_points(dim: usize, points: &'a [f32]) -> Self {
        Self::from_points_with_metric(dim, points, MetricType::L2)
    }

    pub fn from_points_with_metric(dim: usize, points: &'a [f32], metric: MetricType) -> Self {
        Self::from_points_for_nodes(dim, points, metric, None)
    }

    pub fn from_points_for_nodes(
        dim: usize,
        points: &'a [f32],
        metric: MetricType,
        node_ids: Option<&'a [u32]>,
    ) -> Self {
        assert!(dim > 0, "dimension must be positive");
        assert_eq!(
            points.len() % dim,
            0,
            "point buffer length must be divisible by dim"
        );
        let point_count = points.len() / dim;
        if let Some(node_ids) = node_ids {
            assert!(
                node_ids
                    .iter()
                    .all(|&node_id| (node_id as usize) < point_count),
                "node_ids must reference valid point indices"
            );
        }
        Self {
            dim,
            point_count,
            points,
            node_ids,
            metric,
            l2_distance_sq_ptr_kernel: (metric == MetricType::L2)
                .then(simd::l2_distance_sq_ptr_kernel),
        }
    }

    #[doc(hidden)]
    pub fn distance_backend(&self) -> DistanceBackend {
        if self.l2_distance_sq_ptr_kernel.is_some() {
            DistanceBackend::L2FastPath
        } else {
            DistanceBackend::Fallback
        }
    }

    #[inline]
    pub fn distance(&self, lhs: usize, rhs: usize) -> f32 {
        let lhs = self.resolve_idx(lhs);
        let rhs = self.resolve_idx(rhs);
        assert!(
            lhs < self.point_count && rhs < self.point_count,
            "distance indices must reference valid point rows"
        );
        let lhs_start = lhs * self.dim;
        let rhs_start = rhs * self.dim;
        if let Some(kernel) = self.l2_distance_sq_ptr_kernel {
            unsafe {
                return kernel(
                    self.points.as_ptr().add(lhs_start),
                    self.points.as_ptr().add(rhs_start),
                    self.dim,
                );
            }
        }

        let lhs_slice = &self.points[lhs_start..lhs_start + self.dim];
        let rhs_slice = &self.points[rhs_start..rhs_start + self.dim];
        metric_distance(self.metric, lhs_slice, rhs_slice)
    }

    fn alpha_factor(&self, alpha: f32) -> f32 {
        match self.metric {
            MetricType::L2 => alpha * alpha,
            _ => alpha,
        }
    }

    fn resolve_idx(&self, idx: usize) -> usize {
        self.node_ids
            .map(|node_ids| node_ids[idx] as usize)
            .unwrap_or(idx)
    }
}

pub fn sort_neighbors_by_center_distance(
    center: usize,
    neighbors: &[u32],
    distance: &DistanceMatrix<'_>,
) -> Vec<(u32, f32)> {
    let mut ordered: Vec<(u32, f32)> = neighbors
        .iter()
        .map(|&id| (id, distance.distance(center, id as usize)))
        .collect();
    ordered.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    ordered
}

pub fn stage1_prune_neighbors(
    center: usize,
    neighbors: &[u32],
    distance: &DistanceMatrix<'_>,
    alpha: f32,
) -> (Vec<u32>, Vec<u32>) {
    stage1_prune_neighbors_with_trace(center, neighbors, distance, alpha, None)
}

pub fn stage1_prune_neighbors_with_trace(
    center: usize,
    neighbors: &[u32],
    distance: &DistanceMatrix<'_>,
    alpha: f32,
    mut trace: Option<&mut TsdgTrace>,
) -> (Vec<u32>, Vec<u32>) {
    let ordered = sort_neighbors_by_center_distance(center, neighbors, distance);
    let mut deleted = vec![false; ordered.len()];
    let mut occ_counts = vec![0u32; ordered.len()];
    let alpha_factor = distance.alpha_factor(alpha);

    for i in 0..ordered.len() {
        if deleted[i] {
            continue;
        }
        for j in (i + 1)..ordered.len() {
            if deleted[j] {
                continue;
            }
            if let Some(trace) = trace.as_deref_mut() {
                trace.stage1_pairs_checked += 1;
            }
            let dist_between = distance.distance(ordered[i].0 as usize, ordered[j].0 as usize);
            let dist_to_center = ordered[j].1;
            if alpha_factor * dist_between < dist_to_center {
                deleted[j] = true;
            } else if dist_between < dist_to_center {
                occ_counts[j] += 1;
            }
        }
    }

    let mut alive = Vec::new();
    let mut occs = Vec::new();
    for (idx, (neighbor, _)) in ordered.iter().enumerate() {
        if !deleted[idx] {
            alive.push(*neighbor);
            occs.push(occ_counts[idx]);
        }
    }
    (alive, occs)
}

pub fn stage2_filter_neighbors(
    center: usize,
    alive: &[u32],
    occs: &[u32],
    reverse: &[u32],
    distance: &DistanceMatrix<'_>,
    occ_threshold: u32,
    max_k: usize,
) -> Vec<u32> {
    stage2_filter_neighbors_with_trace(
        center,
        alive,
        occs,
        reverse,
        distance,
        occ_threshold,
        max_k,
        None,
    )
}

pub fn stage2_filter_neighbors_with_trace(
    center: usize,
    alive: &[u32],
    occs: &[u32],
    reverse: &[u32],
    distance: &DistanceMatrix<'_>,
    occ_threshold: u32,
    max_k: usize,
    mut trace: Option<&mut TsdgTrace>,
) -> Vec<u32> {
    let mut combined: Vec<(u32, u32)> = alive.iter().copied().zip(occs.iter().copied()).collect();
    let mut seen: HashSet<u32> = alive.iter().copied().collect();

    for &rev in reverse {
        if seen.contains(&rev) {
            continue;
        }
        if let Some(trace) = trace.as_deref_mut() {
            trace.stage2_reverse_candidates += 1;
        }

        let occ = alive
            .iter()
            .copied()
            .filter(|&existing| {
                let dist_between = distance.distance(rev as usize, existing as usize);
                let dist_to_center = distance.distance(center, existing as usize);
                dist_between < dist_to_center
            })
            .count() as u32;

        if occ <= occ_threshold {
            combined.push((rev, occ));
            seen.insert(rev);
        }
    }

    combined.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    combined.truncate(max_k);
    combined.into_iter().map(|(id, _)| id).collect()
}

fn metric_distance(metric: MetricType, lhs: &[f32], rhs: &[f32]) -> f32 {
    match metric {
        MetricType::L2 => lhs
            .iter()
            .zip(rhs.iter())
            .map(|(a, b)| {
                let delta = a - b;
                delta * delta
            })
            .sum::<f32>(),
        MetricType::Ip => -lhs.iter().zip(rhs.iter()).map(|(a, b)| a * b).sum::<f32>(),
        MetricType::Cosine => {
            let dot = lhs.iter().zip(rhs.iter()).map(|(a, b)| a * b).sum::<f32>();
            let lhs_norm = lhs.iter().map(|v| v * v).sum::<f32>().sqrt();
            let rhs_norm = rhs.iter().map(|v| v * v).sum::<f32>().sqrt();
            if lhs_norm == 0.0 || rhs_norm == 0.0 {
                1.0
            } else {
                1.0 - dot / (lhs_norm * rhs_norm)
            }
        }
        MetricType::Hamming => unreachable!("rhtsdg does not support hamming"),
    }
}
