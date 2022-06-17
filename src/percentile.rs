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
use std::cmp;

/// The result of a percentile (e.g. median) lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeanValue<T> {
    /// A single value was found.
    Single(T),
    /// The percentile lies between two values. Take the mean of these to get the percentile.
    Mean(T, T),
}
impl MeanValue<crate::F64OrdHash> {
    #[inline]
    pub fn resolve(self) -> f64 {
        self.map(|v| v.0).resolve()
    }
}
impl<T: PercentileResolve> MeanValue<T> {
    #[inline]
    pub fn resolve(self) -> T {
        PercentileResolve::compute(self)
    }
}
impl<T> MeanValue<T> {
    #[inline]
    pub fn into_single(self) -> Option<T> {
        if let Self::Single(t) = self {
            Some(t)
        } else {
            None
        }
    }
    #[inline]
    pub fn map<O>(self, mut f: impl FnMut(T) -> O) -> MeanValue<O> {
        match self {
            Self::Single(v) => MeanValue::Single(f(v)),
            Self::Mean(a, b) => MeanValue::Mean(f(a), f(b)),
        }
    }
}
impl<T: Clone> MeanValue<&T> {
    #[inline]
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
    #[inline]
    fn compute(percentile: MeanValue<Self>) -> Self {
        match percentile {
            MeanValue::Single(me) => me,
            MeanValue::Mean(a, b) => PercentileResolve::mean(a, b),
        }
    }
}

#[cfg(feature = "generic-impls")]
impl<T: num_traits::identities::One + std::ops::Add<Output = T> + std::ops::Div<Output = T>>
    PercentileResolve for T
{
    #[inline]
    fn mean(a: Self, b: Self) -> Self {
        (a + b) / (T::one() + T::one())
    }
}
#[cfg(not(feature = "generic-impls"))]
macro_rules! impl_resolve {
    ($($t:ty:$two:expr, )+) => {
        $(
        impl PercentileResolve for $t {
            #[inline]
            fn mean(a: Self, b: Self) -> Self {
                (a + b) / $two
            }
        }
        )+
    };
}
#[cfg(not(feature = "generic-impls"))]
macro_rules! impl_resolve_integer {
    ($($t:ty, )+) => {
        impl_resolve!($($t:2, )+);
    };
}
#[cfg(not(feature = "generic-impls"))]
macro_rules! impl_resolve_float {
    ($($t:ty, )+) => {
        impl_resolve!($($t:2.0, )+);
    };
}
#[cfg(not(feature = "generic-impls"))]
impl_resolve_integer!(u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize,);
#[cfg(not(feature = "generic-impls"))]
impl_resolve_float!(f32, f64,);

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
///
/// # Eq & Ord
///
/// You have to enable the `simplify-fraction` to get [`Eq`], [`PartialEq`], [`Ord`], and
/// [`PartialOrd`] implementations.
/// This is due to not knowing if the input to [`Self::new`] is fully simplified. Since we have to
/// use prime factorization to find out, this is an optional feature.
#[derive(Debug, Clone, Copy)]
pub struct Fraction {
    pub numerator: usize,
    pub denominator: usize,
}
impl Fraction {
    pub const HALF: Self = Self {
        numerator: 1,
        denominator: 2,
    };
    pub const ONE_QUARTER: Self = Self {
        numerator: 1,
        denominator: 4,
    };
    pub const THREE_QUARTERS: Self = Self {
        numerator: 3,
        denominator: 4,
    };
    /// If the feature `simplify-fraction` isn't enabled (it is by default), this MUST be the shortest form.
    ///
    /// If the feature `simplify-fraction` is enabled, this [simplifies](Self::simplify) the
    /// fraction.
    /// This has to compute the primes up to `max(numerator, denominator).sqrt()`, so don't run it
    /// in a loop.
    ///
    /// # Panics
    ///
    /// Panics if `numerator > denominator`.
    #[inline]
    pub fn new(numerator: usize, denominator: usize) -> Self {
        assert!(numerator <= denominator);
        #[cfg(feature = "simplify-fraction")]
        {
            Self {
                numerator,
                denominator,
            }
            .simplify()
        }
        #[cfg(not(feature = "simplify-fraction"))]
        {
            Self {
                numerator,
                denominator,
            }
        }
    }

    /// This has to compute the primes up to `max(numerator, denominator).sqrt()`, so don't run it
    /// in a loop.
    #[cfg(feature = "simplify-fraction")]
    pub fn simplify(self) -> Self {
        fn cluster_to_iter<T: Copy>(slice: &[(T, usize)]) -> impl Iterator<Item = T> + '_ {
            slice
                .iter()
                .flat_map(|(num, count)| std::iter::repeat(*num).take(*count))
        }

        if self.numerator == 0 {
            return Self {
                numerator: 0,
                denominator: 1,
            };
        }
        if self.denominator == 0 {
            panic!("denominator is 0");
        }
        // +1 for truncation and precision loss.
        let limit = (self.numerator.max(self.denominator) as f64).sqrt() as usize + 1;
        let sieve = primal_sieve::Sieve::new(limit);
        // UNWRAP: We have enough values.
        let mut num_facs = sieve.factor(self.numerator).unwrap();
        num_facs.sort_unstable_by_key(|(num, _count)| *num);
        // UNWRAP: We have enough values.
        let mut den_facs = sieve.factor(self.denominator).unwrap();
        den_facs.sort_unstable_by_key(|(num, _count)| *num);

        let num_iter = cluster_to_iter(&num_facs);
        let den_iter = cluster_to_iter(&den_facs);

        let both_divisible_by = iter_set::intersection(num_iter, den_iter).product::<usize>();
        debug_assert_eq!(self.numerator % both_divisible_by, 0);
        debug_assert_eq!(self.denominator % both_divisible_by, 0);
        Self {
            numerator: self.numerator / both_divisible_by,
            denominator: self.denominator / both_divisible_by,
        }
    }
}
impl OrderedListIndex for Fraction {
    fn index(&self, len: usize) -> MeanValue<usize> {
        assert!(self.numerator <= self.denominator);
        fn assert_not_zero(denominator: usize) {
            assert_ne!(denominator, 0);
        }
        assert_not_zero(self.denominator);
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
        if self.denominator == 1 {
            MeanValue::Single(self.numerator)
        } else if self.denominator.is_power_of_two() {
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

#[cfg(feature = "simplify-fraction")]
impl PartialEq for Fraction {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        // we don't need to simplify, as [`Self::new`] always does it, there's no way to not get a
        // simplified `Fraction`.
        self.numerator == other.numerator && self.denominator == other.denominator
    }
}
#[cfg(feature = "simplify-fraction")]
impl Eq for Fraction {}
#[cfg(feature = "simplify-fraction")]
impl Ord for Fraction {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // we don't need to simplify, as [`Self::new`] always does it, there's no way to not get a
        // simplified `Fraction`.

        // If we multiply `me` with `other.denominator`, out denominators are the same.
        // We don't need `me.denominators`, so we don't calculate it.
        let my_numerator_with_same_denominator = self.numerator * other.denominator;
        // Same reasoning as above.
        let other_numerator_with_same_denominator = other.numerator * self.denominator;
        my_numerator_with_same_denominator.cmp(&other_numerator_with_same_denominator)
    }
}
#[cfg(feature = "simplify-fraction")]
impl PartialOrd for Fraction {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

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
    #[inline]
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
    #[inline]
    fn index(&self, len: usize) -> MeanValue<usize> {
        assert!(self.k < len);
        // `-1` because if len == 2 and k==0, we want 1, as that's the second index.
        MeanValue::Single(len - 1 - self.k)
    }
}

#[inline(always)]
fn a_cmp_b<T: Ord>(a: &T, b: &T) -> cmp::Ordering {
    a.cmp(b)
}

/// Percentile by sorting.
///
/// See [`naive_percentile_by`] for support for a custom comparator function.
///
/// # Performance & scalability
///
/// This will be very quick for small sets.
/// O(n) performance when `values.len() < 5`, else O(n log n).
#[inline]
pub fn naive_percentile<T: Ord>(values: &mut [T], target: impl OrderedListIndex) -> MeanValue<&T> {
    naive_percentile_by(values, target, &mut a_cmp_b)
}
/// Same as [`naive_percentile`] but with a custom comparator function.
#[inline]
pub fn naive_percentile_by<'a, T>(
    values: &'a mut [T],
    target: impl OrderedListIndex,
    compare: &mut impl FnMut(&T, &T) -> cmp::Ordering,
) -> MeanValue<&'a T> {
    debug_assert!(!values.is_empty(), "we must have more than 0 values!");
    values.sort_by(compare);
    target.index(values.len()).map(|idx| &values[idx])
}
/// quickselect algorithm
///
/// Consider using [`percentile_rand`] or [`median`].
/// See [`percentile_by`] for support for a custom comparator function.
///
/// `pivot_fn` must return a value from the supplied slice.
#[inline]
pub fn percentile<T: Clone + Ord>(
    values: &mut [T],
    target: impl OrderedListIndex,
    pivot_fn: &mut impl FnMut(&mut [T]) -> Cow<'_, T>,
) -> MeanValue<T> {
    percentile_by(values, target, pivot_fn, &mut a_cmp_b)
}
/// Same as [`percentile`] but with a custom comparator function.
#[inline]
pub fn percentile_by<T: Clone>(
    values: &mut [T],
    target: impl OrderedListIndex,
    mut pivot_fn: &mut impl FnMut(&mut [T]) -> Cow<'_, T>,
    mut compare: &mut impl FnMut(&T, &T) -> cmp::Ordering,
) -> MeanValue<T> {
    target
        .index(values.len())
        .map(|v| quickselect(values, v, &mut pivot_fn, &mut compare).clone())
}
/// Convenience function for [`percentile`] with [`pivot_fn::rand`].
#[cfg(feature = "percentile-rand")]
#[inline]
pub fn percentile_rand<T: Ord + Clone>(
    values: &mut [T],
    target: impl OrderedListIndex,
) -> MeanValue<T> {
    percentile(values, target, &mut pivot_fn::rand())
}
/// Get the value at `target` in `values`.
/// Uses the best method available ([`percentile_rand`] if feature `percentile-rand` is enabled,
/// else [`pivot_fn::middle`])
#[inline]
pub fn percentile_default_pivot<T: Ord + Clone>(
    values: &mut [T],
    target: impl OrderedListIndex,
) -> MeanValue<T> {
    percentile_default_pivot_by(values, target, &mut a_cmp_b)
}
/// Same as [`percentile_default_pivot`] but with a custom comparator function.
#[inline]
pub fn percentile_default_pivot_by<T: Clone>(
    values: &mut [T],
    target: impl OrderedListIndex,
    compare: &mut impl FnMut(&T, &T) -> cmp::Ordering,
) -> MeanValue<T> {
    #[cfg(feature = "percentile-rand")]
    {
        percentile_by(values, target, &mut pivot_fn::rand(), compare)
    }
    #[cfg(not(feature = "percentile-rand"))]
    {
        percentile_by(values, target, &mut pivot_fn::middle(), compare)
    }
}

/// Convenience function for [`percentile`] with the 50% mark as the target and [`pivot_fn::rand`]
/// (if the `percentile-rand` feature is enabled, else [`pivot_fn::middle`]).
///
/// See [`percentile_default_pivot_by`] for supplying a custom comparator function.
/// This is critical for types which does not implement [`Ord`] (e.g. f64).
#[inline]
pub fn median<T: Ord + Clone>(values: &mut [T]) -> MeanValue<T> {
    percentile_default_pivot(values, Fraction::HALF)
}
/// Low level function used by this module.
fn quickselect<T: Clone>(
    values: &mut [T],
    mut k: usize,
    mut pivot_fn: impl FnMut(&mut [T]) -> Cow<'_, T>,
    mut compare: impl FnMut(&T, &T) -> cmp::Ordering,
) -> &T {
    if k >= values.len() {
        k = values.len() - 1;
    }
    if values.len() == 1 {
        assert_eq!(k, 0);
        return &values[0];
    }
    if values.len() <= 5 {
        values.sort_unstable_by(compare);
        return &values[k];
    }

    let pivot = pivot_fn(values);
    let pivot = pivot.into_owned();

    let (lows, highs_inclusive) =
        split_include(values, |v| compare(v, &pivot) == cmp::Ordering::Less);
    let (highs, pivots) = split_include(highs_inclusive, |v| {
        compare(v, &pivot) == cmp::Ordering::Greater
    });

    if k < lows.len() {
        quickselect(lows, k, pivot_fn, compare)
    } else if k < lows.len() + pivots.len() {
        &pivots[0]
    } else {
        quickselect(highs, k - lows.len() - pivots.len(), pivot_fn, compare)
    }
}
/// Moves items in the slice and splits it so the first returned slice contains all elements where
/// `predicate` is true. The second contains all other.
#[inline]
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
    median_of_medians_by(values, target, &|a, b| a.cmp(b))
}
/// Same as [`median_of_medians`] but with a custom comparator function.
pub fn median_of_medians_by<T: Clone + PercentileResolve>(
    values: &mut [T],
    target: impl OrderedListIndex,
    mut compare: &impl Fn(&T, &T) -> cmp::Ordering,
) -> MeanValue<T> {
    percentile_by(
        values,
        target,
        &mut pivot_fn::median_of_medians(compare),
        &mut compare,
    )
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
    #[inline]
    pub fn rand<T: Clone, S: SliceSubset<T> + ?Sized>() -> impl FnMut(&mut S) -> Cow<'_, T> {
        let mut rng = rand::thread_rng();
        move |slice| {
            let idx = rng.sample(rand::distributions::Uniform::new(0_usize, slice.len()));
            // UNWRAP: it's less than `slice.len`.
            // We assume `!slice.is_empty()`.
            Cow::Borrowed(slice.get(idx).unwrap())
        }
    }
    #[inline]
    pub fn middle<T: Clone, S: SliceSubset<T> + ?Sized>() -> impl FnMut(&mut S) -> Cow<'_, T> {
        #[inline(always)]
        fn inner<T: Clone, S: SliceSubset<T> + ?Sized>(slice: &mut S) -> Cow<'_, T> {
            // UNWRAP: it's less than `slice.len`.
            // We assume `!slice.is_empty()`.
            Cow::Borrowed(slice.get(slice.len() / 2).unwrap())
        }
        inner
    }
    /// Slice the list using the median of medians method.
    /// It's not recommended to use this.
    /// See the [module-level documentation](super) for more info.
    ///
    /// Picks a good pivot within l, a list of numbers.
    /// This algorithm runs in O(n) time.
    pub fn median_of_medians<T: Clone + PercentileResolve>(
        mut compare: &impl Fn(&T, &T) -> cmp::Ordering,
    ) -> impl FnMut(&mut [T]) -> Cow<'_, T> + '_ {
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
                c.sort_unstable_by(compare);
                c
            });

            let medians = sorted_chunks.map(|chunk| chunk[2].clone());
            let mut medians: Vec<_> = medians.collect();
            let median_of_medians = percentile_by(
                &mut medians,
                Fraction::new(1, 2),
                &mut median_of_medians(compare),
                &mut compare,
            );
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
        #[inline]
        pub fn rand() -> impl FnMut(&ClusterList) -> f64 {
            let mut rng = rand::thread_rng();
            move |slice| {
                let idx = rng.sample(rand::distributions::Uniform::new(0_usize, slice.len()));
                // Panic (index call): it's less than `slice.len`.
                // We assume `!slice.is_empty()`.
                *slice.index(idx)
            }
        }
        #[inline]
        pub fn middle() -> impl FnMut(&ClusterList) -> f64 {
            #[inline(always)]
            fn inner(slice: &ClusterList) -> f64 {
                // Panic (index call): it's less than `slice.len`.
                // We assume `!slice.is_empty()`.
                *slice.index(slice.len() / 2)
            }
            inner
        }
    }

    /// Percentile by sorting.
    ///
    /// See [`naive_percentile_by`] for support for a custom comparator function.
    ///
    /// # Performance & scalability
    ///
    /// This will be very quick for small sets.
    /// O(n) performance when `values.len() < 5`, else O(n log n).
    #[inline]
    pub fn naive_percentile(
        values: &mut OwnedClusterList,
        target: impl OrderedListIndex,
    ) -> MeanValue<f64> {
        naive_percentile_by(values, target, &mut crate::F64OrdHash::f64_cmp)
    }
    /// Same as [`naive_percentile`] but with a custom comparator function.
    #[inline]
    pub fn naive_percentile_by(
        values: &mut OwnedClusterList,
        target: impl OrderedListIndex,
        compare: &mut impl FnMut(f64, f64) -> cmp::Ordering,
    ) -> MeanValue<f64> {
        values.list.sort_unstable_by(|a, b| compare(a.0, b.0));
        let values = values.borrow();
        let len = values.len();
        target.index(len).map(|idx| *values.index(idx))
    }
    /// quickselect algorithm
    ///
    /// Consider using [`percentile_rand`] or [`median`].
    /// See [`percentile_by`] for support for a custom comparator function.
    ///
    /// `pivot_fn` must return a value from the supplied slice.
    #[inline]
    pub fn percentile(
        values: &mut OwnedClusterList,
        target: impl OrderedListIndex,
        pivot_fn: &mut impl FnMut(&ClusterList) -> f64,
    ) -> MeanValue<f64> {
        percentile_by(values, target, pivot_fn, &mut crate::F64OrdHash::f64_cmp)
    }
    /// Same as [`percentile`] but with a custom comparator function.
    #[inline]
    pub fn percentile_by(
        values: &mut OwnedClusterList,
        target: impl OrderedListIndex,
        mut pivot_fn: &mut impl FnMut(&ClusterList) -> f64,
        mut compare: &mut impl FnMut(f64, f64) -> cmp::Ordering,
    ) -> MeanValue<f64> {
        target
            .index(values.borrow().len())
            .map(|idx| quickselect(&mut values.into(), idx, &mut pivot_fn, &mut compare))
    }
    /// Convenience function for [`percentile`] with [`pivot_fn::rand`].
    #[cfg(feature = "percentile-rand")]
    #[inline]
    pub fn percentile_rand(
        values: &mut OwnedClusterList,
        target: impl OrderedListIndex,
    ) -> MeanValue<f64> {
        percentile(values, target, &mut pivot_fn::rand())
    }
    /// Get the value at `target` in `values`.
    /// Uses the best method available ([`percentile_rand`] if feature `percentile-rand` is enabled,
    /// else [`pivot_fn::middle`])
    #[inline]
    pub fn percentile_default_pivot(
        values: &mut OwnedClusterList,
        target: impl OrderedListIndex,
    ) -> MeanValue<f64> {
        percentile_default_pivot_by(values, target, &mut crate::F64OrdHash::f64_cmp)
    }
    /// Same as [`percentile_default_pivot`] but with a custom comparator function.
    #[inline]
    pub fn percentile_default_pivot_by(
        values: &mut OwnedClusterList,
        target: impl OrderedListIndex,
        compare: &mut impl FnMut(f64, f64) -> cmp::Ordering,
    ) -> MeanValue<f64> {
        #[cfg(feature = "percentile-rand")]
        {
            percentile_by(values, target, &mut pivot_fn::rand(), compare)
        }
        #[cfg(not(feature = "percentile-rand"))]
        {
            percentile_by(values, target, &mut pivot_fn::middle(), compare)
        }
    }

    /// Convenience function for [`percentile`] with the 50% mark as the target and [`pivot_fn::rand`]
    /// (if the `percentile-rand` feature is enabled, else [`pivot_fn::middle`]).
    ///
    /// See [`percentile_default_pivot_by`] for supplying a custom comparator function.
    /// This is critical for types which does not implement [`Ord`] (e.g. f64).
    #[inline]
    pub fn median(values: &mut OwnedClusterList) -> MeanValue<f64> {
        percentile_default_pivot(values, Fraction::HALF)
    }

    struct ClusterMut<'a> {
        list: &'a mut [Cluster],
        len: usize,
    }
    impl<'a> Deref for ClusterMut<'a> {
        type Target = [Cluster];
        #[inline]
        fn deref(&self) -> &Self::Target {
            self.list
        }
    }
    impl<'a> DerefMut for ClusterMut<'a> {
        #[inline]
        fn deref_mut(&mut self) -> &mut Self::Target {
            self.list
        }
    }
    impl<'a> From<&'a ClusterMut<'a>> for ClusterList<'a> {
        #[inline]
        fn from(c: &'a ClusterMut<'a>) -> Self {
            ClusterList {
                list: c.list,
                len: c.len,
            }
        }
    }
    impl<'a> From<&'a mut OwnedClusterList> for ClusterMut<'a> {
        #[inline]
        fn from(l: &'a mut OwnedClusterList) -> Self {
            Self {
                list: &mut l.list,
                len: l.len,
            }
        }
    }
    impl<'a> ClusterMut<'a> {
        #[inline]
        fn list(&self) -> ClusterList {
            ClusterList::from(self)
        }
    }
    fn quickselect<'a>(
        values: &'a mut ClusterMut<'a>,
        k: usize,
        mut pivot_fn: impl FnMut(&ClusterList) -> f64,
        mut compare: impl FnMut(f64, f64) -> cmp::Ordering,
    ) -> f64 {
        if values.len() == 1 {
            debug_assert!(k < values.list().len());
            return values[0].0;
        }

        let pivot = pivot_fn(&values.list());

        let (mut lows, mut highs_inclusive) =
            split_include(values, |v| compare(v, pivot) == cmp::Ordering::Less);
        let (mut highs, pivots) = split_include(&mut highs_inclusive, |v| {
            compare(v, pivot) == cmp::Ordering::Greater
        });

        if k < lows.list().len() {
            quickselect(&mut lows, k, pivot_fn, compare)
        } else if k < lows.list().len() + pivots.list().len() {
            pivots[0].0
        } else if highs.is_empty() {
            quickselect(&mut lows, k, pivot_fn, compare)
        } else {
            quickselect(
                &mut highs,
                k - lows.list().len() - pivots.list().len(),
                pivot_fn,
                compare,
            )
        }
    }
    #[inline]
    fn split_include<'a>(
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
