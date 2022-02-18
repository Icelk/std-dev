use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::{hash, ops};

#[cfg(feature = "regression")]
#[path = "regression.rs"]
pub mod regression;

pub mod percentile;

pub use percentile::{median, percentile, percentile_rand, Fraction};
pub use regression::{best_fit_ols as regression_best_fit, Determination, Predictive};

use self::percentile::cluster;

pub type Cluster = (f64, usize);
#[derive(Debug)]
pub struct OwnedClusterList {
    list: Vec<Cluster>,
    len: usize,
}
impl OwnedClusterList {
    /// The float is the value. The integer is the count.
    pub fn new(list: Vec<Cluster>) -> Self {
        let len = ClusterList::size(&list);
        Self { list, len }
    }
    pub fn borrow(&self) -> ClusterList {
        ClusterList {
            list: &self.list,
            len: self.len,
        }
    }
}
impl Deref for OwnedClusterList {
    type Target = [Cluster];
    fn deref(&self) -> &Self::Target {
        &self.list
    }
}
impl DerefMut for OwnedClusterList {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.list
    }
}

/// F64 wrapper that implements [`Ord`] and [`Hash`].
///
/// You should probably not be using this unless you know what you're doing.
#[derive(Debug, Copy, Clone)]
pub struct F64OrdHash(pub f64);
impl F64OrdHash {
    fn key(&self) -> u64 {
        self.0.to_bits()
    }
}
impl hash::Hash for F64OrdHash {
    fn hash<H>(&self, state: &mut H)
    where
        H: hash::Hasher,
    {
        self.key().hash(state)
    }
}
impl PartialEq for F64OrdHash {
    fn eq(&self, other: &F64OrdHash) -> bool {
        self.key() == other.key()
    }
}
impl Eq for F64OrdHash {}
impl PartialOrd for F64OrdHash {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for F64OrdHash {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap()
    }
}

/// A list of clusters.
///
/// A cluster is a value and the count.
///
/// `m` in `O(m)` means the count of clusters.
#[derive(Debug)]
pub struct ClusterList<'a> {
    list: &'a [Cluster],
    len: usize,
}
impl<'a> ClusterList<'a> {
    /// The float is the value. The integer is the count.
    pub fn new(list: &'a [Cluster]) -> Self {
        let len = Self::size(list);
        Self { list, len }
    }

    fn size(list: &[Cluster]) -> usize {
        list.iter().map(|(_, count)| *count).sum()
    }

    /// O(1)
    pub fn len(&self) -> usize {
        self.len
    }
    /// O(1)
    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }
    /// O(m)
    pub fn sum(&self) -> f64 {
        let mut sum = 0.0;
        for (v, count) in self.list.iter() {
            sum += v * *count as f64;
        }
        sum
    }
    fn sum_squared_diff(&self, base: f64) -> f64 {
        let mut sum = 0.0;
        for (v, count) in self.list.iter() {
            sum += (v - base).powi(2) * *count as f64;
        }
        sum
    }
    /// Can be used in [`Self::new`].
    pub fn split_start(&self, len: usize) -> OwnedClusterList {
        let mut sum = 0;
        let mut list = Vec::new();
        for (v, count) in self.list {
            sum += count;
            if sum >= len {
                list.push((*v, *count - (sum - len)));
                break;
            } else {
                list.push((*v, *count));
            }
        }
        debug_assert_eq!(len, Self::size(&list));
        OwnedClusterList { list, len }
    }
    /// Can be used in [`Self::new`].
    pub fn split_end(&self, len: usize) -> OwnedClusterList {
        let mut sum = 0;
        let mut list = Vec::new();
        for (v, count) in self.list.iter().rev() {
            sum += count;
            if sum >= len {
                list.insert(0, (*v, *count - (len - sum)));
                break;
            } else {
                list.insert(0, (*v, *count))
            }
        }
        debug_assert_eq!(len, Self::size(&list));
        OwnedClusterList { list, len }
    }

    /// Groups [`Cluster`]s with the same value together, by adding their count.
    ///
    /// This speeds up calculations enormously.
    ///
    /// O(n)
    pub fn optimize_values(self) -> OwnedClusterList {
        let mut collected = HashMap::with_capacity(16);
        for (v, count) in self.list {
            let c = collected.entry(F64OrdHash(*v)).or_insert(0);
            *c += count;
        }
        let list = collected.into_iter().map(|(f, c)| (f.0, c)).collect();
        OwnedClusterList {
            list,
            len: self.len,
        }
    }
}

/// Returned from [`standard_deviation`] and similar functions.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct StandardDeviationOutput<T> {
    pub standard_deviation: T,
    pub mean: T,
}
/// Returned from [`percentiles_cluster`] and similar functions.
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct PercentilesOutput {
    pub median: f64,
    pub lower_quadrille: Option<f64>,
    pub higher_quadrille: Option<f64>,
}

/// Mean of clustered `values`.
pub fn mean_cluster(values: &ClusterList) -> f64 {
    values.sum() / values.len() as f64
}
/// Mean of `values`.
pub fn mean<'a, T: std::iter::Sum<&'a T> + ops::Div + num_traits::FromPrimitive>(
    values: &'a [T],
) -> <T as std::ops::Div>::Output {
    values.iter().sum::<T>()
        / T::from_usize(values.len())
            .expect("Value can not be converted from usize. Check your type in the call to standard_deviation.")
}
/// Get the standard deviation of `values`.
/// The mean is also returned from this, because it's required to compute the standard deviation.
///
/// O(m), where m is the number of [`Cluster`]s.
pub fn standard_deviation_cluster(values: &ClusterList) -> StandardDeviationOutput<f64> {
    std();
    let m = mean_cluster(values);
    let squared_deviations = values.sum_squared_diff(m);
    let variance: f64 = squared_deviations / (values.len() - 1) as f64;
    StandardDeviationOutput {
        standard_deviation: variance.sqrt(),
        mean: m,
    }
}
fn std() {
    let test: &[f64] = &[2343.3, 324.2, 324.523, 0.234234];
    let std_dev = standard_deviation(test);
    println!("{std_dev:?}");
}
pub fn standard_deviation<
    'a,
    T: std::iter::Sum<&'a T>
        + std::iter::Sum
        + ops::Div<Output = T>
        + ops::Sub<Output = T>
        + ops::Mul<Output = T>
        + num_traits::identities::One
        + num_traits::real::Real
        + num_traits::FromPrimitive
        + Copy
        + 'a,
>(
    values: &'a [T],
) -> StandardDeviationOutput<T> {
    let m = mean(values);
    let squared_deviations: T = values
        .iter()
        .map(|t| {
            let diff = *t - m;

            diff * diff
        })
        .sum();
    let variance: T = squared_deviations / (T::from_usize( values.len()).expect("Value can not be converted from usize. Check your type in the call to standard_deviation.") - T::one());
    let std_dev = variance.sqrt();

    StandardDeviationOutput {
        standard_deviation: std_dev,
        mean: m,
    }
}

/// Get a collection of percentiles from `values`.
pub fn percentiles_cluster(values: &mut OwnedClusterList) -> PercentilesOutput {
    let lower = if values.borrow().len() >= 4 {
        Some(cluster::percentile_rand(values, Fraction::new(1, 4)).resolve())
    } else {
        None
    };
    let higher = if values.borrow().len() >= 4 {
        Some(cluster::percentile_rand(values, Fraction::new(3, 4)).resolve())
    } else {
        None
    };
    PercentilesOutput {
        median: cluster::median(values).resolve(),
        lower_quadrille: lower,
        higher_quadrille: higher,
    }
}
