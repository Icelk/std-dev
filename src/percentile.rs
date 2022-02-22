//! Percentile / median calculations.
//!
//! - `O(n log n)` [`naive_percentile`] (simple to understand)
//! - probabilistic `O(n)` [`percentile`] (recommended, fastest, and also quite simple to understand)
//! - deterministic `O(n)` [`median_of_medians`] (harder to understand, probably slower than the
//!     probabilistic version. However guarantees linear time, so useful in critical applications.)
//!
//! You should probably use [`percentile_rand`].
//!
//! The linear time algoritms are implementations following [this blogpost](https://rcoh.me/posts/linear-time-median-finding/).
#[cfg(feature = "percentile-rand")]
use rand::Rng;
use std::borrow::Cow;
use std::ops;

// `TODO`: Add `_by` functions (e.g. `percentile_by`) to implement comparator functions without
// wrappers.

/// The result of a percentile (e.g. median) lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeanValue<T> {
    /// A single value was found.
    Single(T),
    /// The percentile lies between two values. Take the mean of these to get the percentile.
    Mean(T, T),
}
impl MeanValue<crate::F64OrdHash> {
    pub fn resolve(self) -> f64 {
        self.map(|v| v.0).resolve()
    }
}
impl<T: PercentileResolve> MeanValue<T> {
    pub fn resolve(self) -> T {
        PercentileResolve::compute(self)
    }
}
impl<T> MeanValue<T> {
    pub fn into_single(self) -> Option<T> {
        if let Self::Single(t) = self {
            Some(t)
        } else {
            None
        }
    }
    pub fn map<O>(self, mut f: impl FnMut(T) -> O) -> MeanValue<O> {
        match self {
            Self::Single(v) => MeanValue::Single(f(v)),
            Self::Mean(a, b) => MeanValue::Mean(f(a), f(b)),
        }
    }
}
impl<T: Clone> MeanValue<&T> {
    pub fn clone_inner(&self) -> MeanValue<T> {
        match self {
            Self::Single(t) => MeanValue::Single((*t).clone()),
            Self::Mean(a, b) => MeanValue::Mean((*a).clone(), (*b).clone()),
        }
    }
}
/// Resolves the mean function to return a concrete value.
/// Accessible through [`MeanValue::resolve`].
pub trait PercentileResolve
where
    Self: Sized,
{
    fn mean(a: Self, b: Self) -> Self;
    fn compute(percentile: MeanValue<Self>) -> Self {
        match percentile {
            MeanValue::Single(me) => me,
            MeanValue::Mean(a, b) => PercentileResolve::mean(a, b),
        }
    }
}

impl<T: num_traits::identities::One + ops::Add<Output = T> + ops::Div<Output = T>> PercentileResolve
    for T
{
    fn mean(a: Self, b: Self) -> Self {
        (a + b) / (T::one() + T::one())
    }
}

/// Trait to get the index of a sorted list. Implemented by [`Fraction`], [`KthSmallest`], and
/// [`KthLargest`].
///
/// The target list does not need to be a list, but indexable, and does not need to be sorted, as
/// long as the k-th smallest element is accessible (which it is for all lists, see [`percentile_rand`]).
pub trait OrderedListIndex {
    /// Returns the index this object is targeting.
    /// Could be either a single value or the mean of two.
    fn index(&self, len: usize) -> MeanValue<usize>;
}

/// A fraction.
/// This is used to get a percentile from any function in this module.
///
/// We use two methods of getting the index in a list.
/// The first one can give a mean of two values it the [`Self::denominator`]
/// [`usize::is_power_of_two`]. This enables `3/8` to give a correct result.
/// If the above isn't applicable, the fallback `idx = ceil(len * fraction) - 1`.
///
/// Please contribute if you need a more solid system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fraction {
    pub numerator: usize,
    pub denominator: usize,
}
impl Fraction {
    pub const HALF: Self = Self::new(1, 2);
    pub const ONE_QUARTER: Self = Self::new(1, 4);
    pub const THREE_QUARTERS: Self = Self::new(3, 4);
    /// This MUST be the shortest form.
    pub const fn new(numerator: usize, denominator: usize) -> Self {
        Self {
            numerator,
            denominator,
        }
    }
}
impl OrderedListIndex for Fraction {
    fn index(&self, len: usize) -> MeanValue<usize> {
        assert!(self.numerator <= self.denominator);
        fn power_of_two(me: Fraction, len: usize) -> MeanValue<usize> {
            if me.denominator == 2 {
                if len % 2 == 0 {
                    MeanValue::Mean(len / 2 - 1, len / 2)
                } else {
                    MeanValue::Single(len / 2)
                }
            } else {
                // if say `me == 4/8`, we'd get `0/4`. That's no good! But we don't have to worry
                // about that, as we require (for now) fractions to be in their most simplified
                // form.
                let new = Fraction::new(me.numerator % (me.denominator / 2), me.denominator / 2);
                let mut value = power_of_two(new, len / 2);
                if me.numerator > me.denominator / 2 {
                    // len % 2 because if the value is odd, we add one (the middle term is removed).
                    value = value.map(|v| v + len / 2 + len % 2);
                }

                value
            }
        }
        // `TODO`: implement https://en.wikipedia.org/wiki/Percentile#The_linear_interpolation_between_closest_ranks_method

        // exception for when self.denominator.is_power_of_two(), as we want quartiles and median
        // to be the mean of two values sometimes.
        if self.denominator.is_power_of_two() {
            power_of_two(*self, len)
        } else {
            // ceil(len * percentile) - 1
            let m = len * self.numerator;
            let rem = m % self.denominator;
            let rem = usize::from(rem > 1);
            MeanValue::Single((m / self.denominator + rem - 1).min(len))
        }
    }
}
// `TODO` implement Eq and Ord for `Fraction`
// requires prime factoring (or just converting to floats...) (https://crates.io/crates/primal-sieve)

/// Get the k-th smallest value.
/// Implements [`OrderedListIndex`].
pub struct KthSmallest {
    pub k: usize,
}
impl KthSmallest {
    pub fn new(k: usize) -> Self {
        Self { k }
    }
}
impl OrderedListIndex for KthSmallest {
    fn index(&self, len: usize) -> MeanValue<usize> {
        assert!(self.k < len);
        MeanValue::Single(self.k)
    }
}
/// Get the k-th largest value.
/// Implements [`OrderedListIndex`].
pub struct KthLargest {
    pub k: usize,
}
impl KthLargest {
    pub fn new(k: usize) -> Self {
        Self { k }
    }
}
impl OrderedListIndex for KthLargest {
    fn index(&self, len: usize) -> MeanValue<usize> {
        assert!(self.k < len);
        // `-1` because if len == 2 and k==0, we want 1, as that's the second index.
        MeanValue::Single(len - 1 - self.k)
    }
}

/// Percentile by sorting.
///
/// # Performance & scalability
///
/// This will be very quick for small sets.
/// O(n) performance when `values.len() < 5`, else O(n log n).
pub fn naive_percentile<T: Ord>(values: &mut [T], target: impl OrderedListIndex) -> MeanValue<&T> {
    debug_assert!(!values.is_empty(), "we must have more than 0 values!");
    values.sort();
    target.index(values.len()).map(|idx| &values[idx])
}
/// quickselect algorithm
///
/// Consider using [`percentile_rand`] or [`median`].
///
/// `pivot_fn` must return a value from the supplied slice.
pub fn percentile<T: Ord + Clone>(
    values: &mut [T],
    target: impl OrderedListIndex,
    pivot_fn: &mut impl FnMut(&mut [T]) -> Cow<'_, T>,
) -> MeanValue<T> {
    target
        .index(values.len())
        .map(|v| quickselect(values, v, pivot_fn).clone())
}
/// Convenience function for [`percentile`] with a random `pivot_fn`.
#[cfg(feature = "percentile-rand")]
pub fn percentile_rand<T: Ord + Clone>(
    values: &mut [T],
    target: impl OrderedListIndex,
) -> MeanValue<T> {
    percentile(values, target, &mut pivot_fn::rand())
}
/// Convenience function for [`percentile`] with the 50% mark as the target and [`pivot_fn::rand`]
/// (if the `percentile-rand` feature is enabled, else [`pivot_fn::middle`]).
pub fn median<T: Ord + Clone>(values: &mut [T]) -> MeanValue<T> {
    #[cfg(feature = "percentile-rand")]
    {
        percentile_rand(values, Fraction::HALF)
    }
    #[cfg(not(feature = "percentile-rand"))]
    {
        percentile(values, Fraction::HALF, &mut pivot_fn::middle())
    }
}
/// Low level function used by this module.
fn quickselect<'a, T: Ord + Clone>(
    values: &'a mut [T],
    k: usize,
    pivot_fn: &mut impl FnMut(&mut [T]) -> Cow<'_, T>,
) -> &'a T {
    if values.len() == 1 {
        assert_eq!(k, 0);
        return &values[0];
    }
    if values.len() <= 5 {
        values.sort_unstable();
        return &values[k];
    }

    let pivot = pivot_fn(values);
    let pivot = pivot.into_owned();

    let (lows, highs_inclusive) = split_include(values, |v| *v < pivot);
    let (highs, pivots) = split_include(highs_inclusive, |v| *v > pivot);

    if k < lows.len() {
        quickselect(lows, k, pivot_fn)
    } else if k < lows.len() + pivots.len() {
        &pivots[0]
    } else {
        quickselect(highs, k - lows.len() - pivots.len(), pivot_fn)
    }
}
/// Moves items in the slice and splits it so the first returned slice contains all elements where
/// `predicate` is true. The second contains all other.
pub fn split_include<T>(
    slice: &mut [T],
    mut predicate: impl FnMut(&T) -> bool,
) -> (&mut [T], &mut [T]) {
    let mut add_index = 0;
    let mut index = 0;
    let len = slice.len();
    while index < len {
        let value = &mut slice[index];
        if predicate(value) {
            slice.swap(index, add_index);
            add_index += 1;
        }
        index += 1;
    }

    slice.split_at_mut(add_index)
}
/// Same result as [`percentile_rand`] but in deterministic linear time.
/// But probabilistically way slower.
pub fn median_of_medians<T: Ord + Clone + PercentileResolve>(
    values: &mut [T],
    target: impl OrderedListIndex,
) -> MeanValue<T> {
    percentile(values, target, &mut pivot_fn::median_of_medians())
}

pub mod pivot_fn {
    use super::*;

    pub trait SliceSubset<T> {
        fn len(&self) -> usize;
        fn is_empty(&self) -> bool {
            self.len() == 0
        }
        /// Returns [`None`] if `idx >= Self::len`.
        fn get(&self, idx: usize) -> Option<&T>;
    }
    impl<T> SliceSubset<T> for [T] {
        #[inline]
        fn len(&self) -> usize {
            <[T]>::len(self)
        }
        #[inline]
        fn get(&self, idx: usize) -> Option<&T> {
            <[T]>::get(self, idx)
        }
    }
    impl<T> SliceSubset<T> for &[T] {
        #[inline]
        fn len(&self) -> usize {
            <[T]>::len(self)
        }
        #[inline]
        fn get(&self, idx: usize) -> Option<&T> {
            <[T]>::get(self, idx)
        }
    }
    impl<T> SliceSubset<T> for &mut [T] {
        #[inline]
        fn len(&self) -> usize {
            <[T]>::len(self)
        }
        #[inline]
        fn get(&self, idx: usize) -> Option<&T> {
            <[T]>::get(self, idx)
        }
    }
    //// See todo note under `clusters`.
    //
    // impl<'a> SliceSubset<f64> for ClusterList<'a> {
    // #[inline]
    // fn len(&self) -> usize {
    // self.len()
    // }
    // #[inline]
    // fn get(&self, idx: usize) -> Option<&f64> {
    // if idx < self.len() {
    // Some(self.index(idx))
    // } else {
    // None
    // }
    // }
    // }

    #[cfg(feature = "percentile-rand")]
    pub fn rand<T: Clone, S: SliceSubset<T> + ?Sized>() -> impl FnMut(&mut S) -> Cow<'_, T> {
        let mut rng = rand::thread_rng();
        move |slice| {
            let idx = rng.sample(rand::distributions::Uniform::new(0_usize, slice.len()));
            // UNWRAP: it's less than `slice.len`.
            // We assume `!slice.is_empty()`.
            Cow::Borrowed(slice.get(idx).unwrap())
        }
    }
    pub fn middle<T: Clone, S: SliceSubset<T> + ?Sized>() -> impl FnMut(&mut S) -> Cow<'_, T> {
        // UNWRAP: it's less than `slice.len`.
        // We assume `!slice.is_empty()`.
        move |slice| Cow::Borrowed(slice.get(slice.len() / 2).unwrap())
    }
    /// Slice the list using the median of medians method.
    /// It's not recommended to use this.
    /// See the [module-level documentation](super) for more info.
    ///
    /// Picks a good pivot within l, a list of numbers.
    /// This algorithm runs in O(n) time.
    pub fn median_of_medians<T: Ord + Clone + PercentileResolve>(
    ) -> impl FnMut(&mut [T]) -> Cow<'_, T> {
        move |l| {
            let len = l.len();
            assert!(len > 0);

            // / 5 * 5 truncates to the lower 5-multiple.
            // We only want full chunks.
            let chunks = l[..(len / 5) * 5].chunks_mut(5);

            // Next, we sort each chunk. Each group is a fixed length, so each sort
            // takes constant time. Since we have n/5 chunks, this operation
            // is also O(n)
            let sorted_chunks = chunks.map(|c| {
                c.sort_unstable();
                c
            });

            let medians = sorted_chunks.map(|chunk| chunk[2].clone());
            let mut medians: Vec<_> = medians.collect();
            let median_of_medians =
                percentile(&mut medians, Fraction::new(1, 2), &mut median_of_medians());
            Cow::Owned(median_of_medians.resolve())
        }
    }
}

/// Operations on [`crate::Cluster`]s.
///
/// This attempts to implement all functionality of this module but using clusters.
/// This turned out to be quite difficult.
pub mod cluster {
    use super::*;
    use crate::{Cluster, ClusterList, OwnedClusterList};
    use std::ops::{Deref, DerefMut};

    // `TODO`: use `super::pivot_fn` instead. That doesn't however seem to work, due to idiotic
    // lifetime requirements.
    pub mod pivot_fn {
        use super::*;
        #[cfg(feature = "percentile-rand")]
        pub fn rand() -> impl FnMut(&ClusterList) -> f64 {
            let mut rng = rand::thread_rng();
            move |slice| {
                let idx = rng.sample(rand::distributions::Uniform::new(0_usize, slice.len()));
                // Panic (index call): it's less than `slice.len`.
                // We assume `!slice.is_empty()`.
                *slice.index(idx)
            }
        }
        pub fn middle() -> impl FnMut(&ClusterList) -> f64 {
            // Panic (index call): it's less than `slice.len`.
            // We assume `!slice.is_empty()`.
            move |slice| *slice.index(slice.len() / 2)
        }
    }

    /// Percentile by sorting.
    ///
    /// # Performance & scalability
    ///
    /// This will be very quick for small sets.
    /// O(n) performance when `values.len() < 5`, else O(n log n).
    pub fn naive_percentile(
        values: &mut OwnedClusterList,
        target: impl OrderedListIndex,
    ) -> MeanValue<f64> {
        values
            .list
            .sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let values = values.borrow();
        let len = values.len();
        target.index(len).map(|idx| *values.index(idx))
    }
    /// quickselect algorithm
    ///
    /// Consider using [`percentile_rand`] or [`median`].
    ///
    /// `pivot_fn` must return a value from the supplied slice.
    pub fn percentile(
        values: &mut OwnedClusterList,
        target: impl OrderedListIndex,
        pivot_fn: &mut impl FnMut(&ClusterList) -> f64,
    ) -> MeanValue<f64> {
        target
            .index(values.borrow().len())
            .map(|idx| quickselect(&mut values.into(), idx, pivot_fn))
    }
    /// Convenience function for [`percentile`] with [`pivot_fn::rand`].
    #[cfg(feature = "percentile-rand")]
    pub fn percentile_rand(
        values: &mut OwnedClusterList,
        target: impl OrderedListIndex,
    ) -> MeanValue<f64> {
        percentile(values, target, &mut pivot_fn::rand())
    }
    /// Convenience function for [`percentile`] with the 50% mark as the target and [`pivot_fn::rand`]
    /// (if the `percentile-rand` feature is enabled, else [`pivot_fn::middle`]).
    pub fn median(values: &mut OwnedClusterList) -> MeanValue<f64> {
        #[cfg(feature = "percentile-rand")]
        {
            percentile_rand(values, Fraction::HALF)
        }
        #[cfg(not(feature = "percentile-rand"))]
        {
            percentile(values, Fraction::HALF, &mut pivot_fn::middle())
        }
    }
    struct ClusterMut<'a> {
        list: &'a mut [Cluster],
        len: usize,
    }
    impl<'a> Deref for ClusterMut<'a> {
        type Target = [Cluster];
        fn deref(&self) -> &Self::Target {
            self.list
        }
    }
    impl<'a> DerefMut for ClusterMut<'a> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            self.list
        }
    }
    impl<'a> From<&'a ClusterMut<'a>> for ClusterList<'a> {
        fn from(c: &'a ClusterMut<'a>) -> Self {
            ClusterList {
                list: c.list,
                len: c.len,
            }
        }
    }
    impl<'a> From<&'a mut OwnedClusterList> for ClusterMut<'a> {
        fn from(l: &'a mut OwnedClusterList) -> Self {
            Self {
                list: &mut l.list,
                len: l.len,
            }
        }
    }
    impl<'a> ClusterMut<'a> {
        fn list(&self) -> ClusterList {
            ClusterList::from(self)
        }
    }
    fn quickselect<'a>(
        values: &'a mut ClusterMut<'a>,
        k: usize,
        pivot_fn: &mut impl FnMut(&ClusterList) -> f64,
    ) -> f64 {
        if values.len() == 1 {
            debug_assert!(k < values.list().len());
            return values[0].0;
        }

        let pivot = pivot_fn(&values.list());

        let (mut lows, mut highs_inclusive) = include(values, |v| v < pivot);
        let (mut highs, pivots) = include(&mut highs_inclusive, |v| v > pivot);

        if k < lows.list().len() {
            quickselect(&mut lows, k, pivot_fn)
        } else if k < lows.list().len() + pivots.list().len() {
            pivots[0].0
        } else if highs.is_empty() {
            quickselect(&mut lows, k, pivot_fn)
        } else {
            quickselect(
                &mut highs,
                k - lows.list().len() - pivots.list().len(),
                pivot_fn,
            )
        }
    }
    fn include<'a>(
        slice: &'a mut ClusterMut<'a>,
        mut predicate: impl FnMut(f64) -> bool,
    ) -> (ClusterMut<'a>, ClusterMut<'a>) {
        let mut add_index = 0;
        let mut index = 0;
        let len = slice.len();
        let mut total_len = 0;
        let cluser_len = slice.list().len();
        while index < len {
            let value = &mut slice[index];
            if predicate(value.0) {
                total_len += value.1;
                slice.swap(index, add_index);
                add_index += 1;
            }
            index += 1;
        }

        let (a, b) = slice.split_at_mut(add_index);
        (
            ClusterMut {
                list: a,
                len: total_len,
            },
            ClusterMut {
                list: b,
                len: cluser_len - total_len,
            },
        )
    }
}
