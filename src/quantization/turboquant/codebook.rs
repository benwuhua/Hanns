const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;
const SQRT_2: f64 = std::f64::consts::SQRT_2;

#[inline]
fn normal_pdf(x: f64) -> f64 {
    if x.is_infinite() {
        0.0
    } else {
        (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
    }
}

#[inline]
fn erf_approx(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly = (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736)
        * t
        + 0.254_829_592)
        * t;
    sign * (1.0 - poly * (-x * x).exp())
}

#[inline]
fn normal_cdf(x: f64) -> f64 {
    if x.is_infinite() {
        if x.is_sign_negative() {
            0.0
        } else {
            1.0
        }
    } else {
        0.5 * (1.0 + erf_approx(x / SQRT_2))
    }
}

fn truncated_normal_mean(left: f64, right: f64) -> Option<f64> {
    let mass = normal_cdf(right) - normal_cdf(left);
    if mass <= 1e-12 {
        return None;
    }
    Some((normal_pdf(left) - normal_pdf(right)) / mass)
}

/// Return `2^bits` ascending Lloyd-Max centroids for a standard Gaussian.
/// The caller applies the final `1/sqrt(dim)` scale for unit-norm vectors.
pub fn gaussian_lloyd_max_centroids(bits: u8, _dim: usize) -> Vec<f64> {
    assert!((1..=8).contains(&bits), "bits must be in 1..=8");

    let levels = 1usize << bits;
    if levels == 2 {
        return vec![-SQRT_2_OVER_PI, SQRT_2_OVER_PI];
    }

    let mut centroids: Vec<f64> = (0..levels)
        .map(|i| {
            let t = (i as f64 + 0.5) / levels as f64;
            -3.5 + 7.0 * t
        })
        .collect();

    for _ in 0..100 {
        let mut bounds = Vec::with_capacity(levels + 1);
        bounds.push(f64::NEG_INFINITY);
        for i in 0..levels - 1 {
            bounds.push(0.5 * (centroids[i] + centroids[i + 1]));
        }
        bounds.push(f64::INFINITY);

        let mut next = centroids.clone();
        let mut max_delta = 0.0f64;
        for i in 0..levels {
            if let Some(updated) = truncated_normal_mean(bounds[i], bounds[i + 1]) {
                max_delta = max_delta.max((updated - centroids[i]).abs());
                next[i] = updated;
            }
        }

        centroids = next;
        if max_delta < 1e-9 {
            break;
        }
    }

    centroids
}
