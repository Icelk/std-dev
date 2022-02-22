use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::{hash, ops};

#[cfg(feature = "regression")]
#[path = "regression.rs"]
pub mod regression;

pub mod percentile;

#[cfg(feature = "percentile-rand")]
pub use percentile::percentile_rand;
pub use percentile::{median, percentile, Fraction};
#[cfg(feature = "regression")]
pub use regression::{best_fit_ols as regression_best_fit, Determination, Predictive};

use self::percentile::cluster;

/// > As all algorithms are executed in linear time now, this is not as useful, but nevertheless an interesting feature.
/// > If you already have clustered data, this feature is great.
///
/// When using this, calculations are done per _unique_ value.
/// Say you have a dataset of infant height, in centimeters.
/// That's probably only going to be some 40 different values, but potentially millions of entries.
/// Using clusters, all that data is only processed as `O(40)`, not `O(millions)`. (I know that notation isn't right, but you get my point).
pub type Cluster = (f64, usize);

/// Owned variant of [`ClusterList`].
/// Use [`Self::borrow`] to get a [`ClusterList`].
/// The inner slice is accessible through the [`Deref`] and [`DerefMut`], which means you can use
/// this as a mutable slice.
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
/// When [`PartialOrd`] returns [`None`], we return [`std::cmp::Ordering::Equal`].
///
/// You should probably not be using this unless you know what you're doing.
#[derive(Debug, Copy, Clone)]
#[repr(transparent)]
pub struct F64OrdHash(pub f64);
impl F64OrdHash {
    fn key(&self) -> u64 {
        self.0.to_bits()
    }
    pub fn to_mut_f64_slice(me: &mut [Self]) -> &mut [f64] {
        // SAFETY: Since we have the same layout (repr(transparent)), this is fine.
        unsafe { std::mem::transmute(me) }
    }
    pub fn from_mut_f64_slice(slice: &mut [f64]) -> &mut [Self] {
        // SAFETY: Since we have the same layout (repr(transparent)), this is fine.
        unsafe { std::mem::transmute(slice) }
    }
    pub fn to_f64_slice(me: &[Self]) -> &[f64] {
        // SAFETY: Since we have the same layout (repr(transparent)), this is fine.
        unsafe { std::mem::transmute(me) }
    }
    pub fn from_f64_slice(slice: &[f64]) -> &[Self] {
        // SAFETY: Since we have the same layout (repr(transparent)), this is fine.
        unsafe { std::mem::transmute(slice) }
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
        self.0
            .partial_cmp(&other.0)
            .unwrap_or_else(|| match (self.0.is_nan(), other.0.is_nan()) {
                (true, true) | (false, false) => std::cmp::Ordering::Equal,
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
            })
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
    /// Returns the value at `idx`. This iterates the clusters to get the value.
    ///
    /// # Panics
    ///
    /// Panics if [`Self::is_empty`] or if `idx >= self.len()`.
    #[inline]
    #[allow(clippy::should_implement_trait)] // `TODO`
    pub fn index(&self, mut idx: usize) -> &f64 {
        for (v, c) in self.list {
            let c = *c;
            if idx < c {
                return v;
            }
            idx -= c;
        }
        &self.list.last().unwrap().0
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

/// Helper-trait for types used by [`mean`].
///
/// This is implemented generically when the feature `generic-impl` is enabled.
pub trait Mean<'a, D>: std::iter::Sum<&'a Self> + ops::Div<Output = D>
where
    Self: 'a,
{
    fn from_usize(n: usize) -> Self;
}
#[cfg(feature = "generic-impls")]
impl<'a, T: std::iter::Sum<&'a Self> + ops::Div + num_traits::FromPrimitive> Mean<'a, T::Output>
    for T
where
    T: 'a,
{
    fn from_usize(n: usize) -> Self {
        Self::from_usize(n).expect("Value can not be converted from usize. Check your type in the call to standard_deviation/mean.")
    }
}
#[cfg(not(feature = "generic-impls"))]
macro_rules! impl_mean {
    ($($t:ty, )+) => {
        $(
        impl<'a> Mean<'a, <$t as ops::Div>::Output> for $t {
            fn from_usize(n: usize) -> Self {
                n as _
            }
        }
        )+
    };
}
#[cfg(not(feature = "generic-impls"))]
impl_mean!(f32, f64, u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize,);

/// Helper-trait for types used by [`standard_deviation`].
///
/// This is implemented generically when the feature `generic-impl` is enabled.
pub trait StandardDeviation<'a>:
    Copy
    + Mean<'a, Self>
    + std::iter::Sum<&'a Self>
    + std::iter::Sum
    + ops::Div<Output = Self>
    + ops::Sub<Output = Self>
    + ops::Mul<Output = Self>
where
    Self: 'a,
{
    fn one() -> Self;
    fn sqrt(self) -> Self;
}
#[cfg(feature = "generic-impls")]
impl<
        'a,
        T: Copy
            + Mean<'a, Self>
            + std::iter::Sum<&'a Self>
            + std::iter::Sum
            + ops::Div<Output = Self>
            + ops::Sub<Output = Self>
            + ops::Mul<Output = Self>
            + num_traits::identities::One
            + num_traits::real::Real,
    > StandardDeviation<'a> for T
where
    T: 'a,
{
    fn one() -> Self {
        Self::one()
    }
    fn sqrt(self) -> Self {
        self.sqrt()
    }
}
#[cfg(not(feature = "generic-impls"))]
macro_rules! impl_std_dev {
    ($($t:ty, )+) => {
        $(
        impl<'a> StandardDeviation<'a> for $t {
            fn one() -> Self {
                1.1
            }
            fn sqrt(self) -> Self {
                <$t>::sqrt(self)
            }
        }
        )+
    };
}
#[cfg(not(feature = "generic-impls"))]
impl_std_dev!(f32, f64,);

/// Mean of clustered `values`.
pub fn mean_cluster(values: &ClusterList) -> f64 {
    values.sum() / values.len() as f64
}
/// Mean of `values`.
pub fn mean<'a, D, T: Mean<'a, D>>(values: &'a [T]) -> D {
    values.iter().sum::<T>() / T::from_usize(values.len())
}

/// Get the standard deviation of `values`.
/// The mean is also returned from this, because it's required to compute the standard deviation.
///
/// O(m), where m is the number of [`Cluster`]s.
pub fn standard_deviation_cluster(values: &ClusterList) -> StandardDeviationOutput<f64> {
    let m = mean_cluster(values);
    let squared_deviations = values.sum_squared_diff(m);
    let variance: f64 = squared_deviations / (values.len() - 1) as f64;
    StandardDeviationOutput {
        standard_deviation: variance.sqrt(),
        mean: m,
    }
}
/// Get the standard deviation of `values`.
/// The mean is also returned from this, because it's required to compute the standard deviation.
///
/// O(n)
// `TODO`: Remove dependency of `num_traits`, create our own trait which implements the methods, then cfg
// if not num_traits, implement for f64,f32. Else, derive from the current traits.
pub fn standard_deviation<'a, T: StandardDeviation<'a>>(
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
    let variance: T = squared_deviations / (T::from_usize(values.len()) - T::one());
    let std_dev = variance.sqrt();

    StandardDeviationOutput {
        standard_deviation: std_dev,
        mean: m,
    }
}

/// Get a collection of percentiles from `values`.
pub fn percentiles_cluster(values: &mut OwnedClusterList) -> PercentilesOutput {
    fn percentile(
        values: &mut OwnedClusterList,
        target: impl percentile::OrderedListIndex,
    ) -> percentile::MeanValue<f64> {
        #[cfg(feature = "percentile-rand")]
        {
            cluster::percentile_rand(values, target)
        }
        #[cfg(not(feature = "percentile-rand"))]
        {
            cluster::percentile(values, target, &mut cluster::pivot_fn::middle())
        }
    }
    let lower = if values.borrow().len() >= 4 {
        Some(percentile(values, Fraction::new(1, 4)).resolve())
    } else {
        None
    };
    let higher = if values.borrow().len() >= 4 {
        Some(percentile(values, Fraction::new(3, 4)).resolve())
    } else {
        None
    };
    PercentilesOutput {
        median: cluster::median(values).resolve(),
        lower_quadrille: lower,
        higher_quadrille: higher,
    }
}
