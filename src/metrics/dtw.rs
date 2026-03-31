//! Dynamic Time Warping (DTW) distance variants.

#[inline]
fn parse_lengths(a: &[f32], b: &[f32], d: usize) -> Option<(usize, usize)> {
    if d == 0 || !a.len().is_multiple_of(d) || !b.len().is_multiple_of(d) {
        return None;
    }
    Some((a.len() / d, b.len() / d))
}

#[inline]
fn l2_step(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
        .sqrt()
}

#[inline]
fn cosine_step(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let na = na.sqrt();
    let nb = nb.sqrt();
    if na <= f32::EPSILON || nb <= f32::EPSILON {
        if na <= f32::EPSILON && nb <= f32::EPSILON {
            0.0
        } else {
            1.0
        }
    } else {
        1.0 - (dot / (na * nb))
    }
}

#[inline]
fn ip_step(a: &[f32], b: &[f32]) -> f32 {
    -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
}

fn dtw_core<F, C>(a: &[f32], b: &[f32], d: usize, step_dist: F, allow: C) -> f32
where
    F: Fn(&[f32], &[f32]) -> f32,
    C: Fn(usize, usize) -> bool,
{
    let Some((n, m)) = parse_lengths(a, b, d) else {
        return f32::INFINITY;
    };
    if n == 0 || m == 0 {
        return f32::INFINITY;
    }

    let mut prev = vec![f32::INFINITY; m + 1];
    let mut curr = vec![f32::INFINITY; m + 1];
    prev[0] = 0.0;

    for i in 1..=n {
        curr[0] = f32::INFINITY;
        let ai = &a[(i - 1) * d..i * d];
        for j in 1..=m {
            if !allow(i - 1, j - 1) {
                curr[j] = f32::INFINITY;
                continue;
            }
            let bj = &b[(j - 1) * d..j * d];
            let cost = step_dist(ai, bj);
            let best_prev = prev[j].min(curr[j - 1]).min(prev[j - 1]);
            curr[j] = cost + best_prev;
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[m]
}

/// 标准 DTW（无约束，步进距离为 L2）
pub fn dtw(a: &[f32], b: &[f32], d: usize) -> f32 {
    dtw_core(a, b, d, l2_step, |_i, _j| true)
}

/// 带 Sakoe-Chiba 带约束的 DTW（|i-j| <= window）
pub fn dtw_sakoe_chiba(a: &[f32], b: &[f32], d: usize, window: usize) -> f32 {
    dtw_core(a, b, d, l2_step, move |i, j| i.abs_diff(j) <= window)
}

/// 带 Itakura 约束的 DTW（i/j 与 j/i 均不超过 slope）
pub fn dtw_itakura(a: &[f32], b: &[f32], d: usize, slope: f32) -> f32 {
    let slope = slope.max(1.0);
    dtw_core(a, b, d, l2_step, move |i, j| {
        let i1 = (i + 1) as f32;
        let j1 = (j + 1) as f32;
        i1 / j1 <= slope && j1 / i1 <= slope
    })
}

/// 显式 L2 版本，等价于 dtw()
pub fn dtw_l2(a: &[f32], b: &[f32], d: usize) -> f32 {
    dtw(a, b, d)
}

/// DTW + cosine 距离（每步用 1 - cosine）
pub fn dtw_cosine(a: &[f32], b: &[f32], d: usize) -> f32 {
    dtw_core(a, b, d, cosine_step, |_i, _j| true)
}

/// DTW + IP 距离（每步用 -dot）
pub fn dtw_ip(a: &[f32], b: &[f32], d: usize) -> f32 {
    dtw_core(a, b, d, ip_step, |_i, _j| true)
}

/// 通用 dispatch（metric: \"l2\" | \"cosine\" | \"ip\" | \"sakoe_chiba\" | \"itakura\"）
pub fn dtw_dispatch(metric: &str, a: &[f32], b: &[f32], d: usize) -> f32 {
    match metric.to_ascii_lowercase().as_str() {
        "l2" => dtw_l2(a, b, d),
        "cosine" => dtw_cosine(a, b, d),
        "ip" => dtw_ip(a, b, d),
        "sakoe_chiba" => {
            let Some((n, m)) = parse_lengths(a, b, d) else {
                return f32::INFINITY;
            };
            let w = n.abs_diff(m) + n.max(m) / 10;
            dtw_sakoe_chiba(a, b, d, w)
        }
        "itakura" => dtw_itakura(a, b, d, 2.0),
        _ => dtw_l2(a, b, d),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtw_identical() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = a.clone();
        let v = dtw(&a, &b, 1);
        assert!(v.abs() < 1e-6, "expected 0, got {}", v);
    }

    #[test]
    fn test_dtw_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 2.0, 2.0];
        // For 1D with L2 step (=|x-y|), optimal total cost is 2.
        let v = dtw(&a, &b, 1);
        assert!((v - 2.0).abs() < 1e-6, "expected 2, got {}", v);
    }

    #[test]
    fn test_dtw_sakoe_chiba() {
        let a = vec![0.0, 1.0, 2.0, 3.0];
        let b = vec![0.0, 1.0, 2.0, 4.0];
        // window=0 forces diagonal alignment
        let v = dtw_sakoe_chiba(&a, &b, 1, 0);
        assert!((v - 1.0).abs() < 1e-6, "expected 1, got {}", v);
    }

    #[test]
    fn test_dtw_cosine() {
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let v = dtw_cosine(&a, &b, 2);
        assert!(v.abs() < 1e-6, "expected 0, got {}", v);
    }

    #[test]
    fn test_dtw_dispatch() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 4.0];
        let d1 = dtw_dispatch("l2", &a, &b, 1);
        let d2 = dtw(&a, &b, 1);
        assert!((d1 - d2).abs() < 1e-6);
    }
}
