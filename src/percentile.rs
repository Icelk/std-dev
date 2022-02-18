//! Percentile / median calculations.
//!
//! - `O(n log n)` [`naive_percentile`] (simple to understand)
//! - probabilistic `O(n)` [`percentile`] (recommended, fastest, and also quite simple to understand)
//! - deterministic `O(n)` [`median_of_medians`] (harder to understand, probably slower than the
//!     probabilistic version.)
//!
//! You should probably use [`percentile_rand`].
//!
//! The linear time algoritms are implementations following [this blogpost](https://rcoh.me/posts/linear-time-median-finding/).

use rand::Rng;
use std::borrow::Cow;
use std::ops;

/// The result of a percentile (e.g. mean) lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Percentile<T> {
    /// A single value was found.
    Single(T),
    /// The percentile lies between two values. Take the mean of these to get the percentile.
    Mean(T, T),
}
impl<T: PercentileResolve> Percentile<T> {
    pub fn resolve(self) -> T {
        PercentileResolve::compute(self)
    }
}
impl<T> Percentile<T> {
    pub fn into_single(self) -> Option<T> {
        if let Self::Single(t) = self {
            Some(t)
        } else {
            None
        }
    }
    pub fn map<O>(self, mut f: impl FnMut(T) -> O) -> Percentile<O> {
        match self {
            Self::Single(v) => Percentile::Single(f(v)),
            Self::Mean(a, b) => Percentile::Mean(f(a), f(b)),
        }
    }
}
impl<T: Clone> Percentile<&T> {
    pub fn clone_inner(&self) -> Percentile<T> {
        match self {
            Self::Single(t) => Percentile::Single((*t).clone()),
            Self::Mean(a, b) => Percentile::Mean((*a).clone(), (*b).clone()),
        }
    }
}
/// Resolves the mean function to return a concrete value.
/// Accessible through [`Percentile::resolve`].
pub trait PercentileResolve
where
    Self: Sized,
{
    fn mean(a: Self, b: Self) -> Self;
    fn compute(percentile: Percentile<Self>) -> Self {
        match percentile {
            Percentile::Single(me) => me,
            Percentile::Mean(a, b) => PercentileResolve::mean(a, b),
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

/// Matching of this with the percentile is done on a best-effort basis.
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
// `TODO` implement Eq and Ord for `Fraction`
// requires prime factoring (or just converting to floats...)

/// Percentile by sorting.
///
/// # Performance & scalability
///
/// This will be very quick for small sets.
/// O(n) performance when `values.len() < 5`, else O(n log n).
pub fn naive_percentile<T: Ord>(values: &mut [T]) -> Percentile<&T> {
    assert!(!values.is_empty());
    values.sort();
    if values.len() % 2 == 0 {
        // even
        let a = &values[values.len() / 2 - 1];
        let b = &values[values.len() / 2];
        Percentile::Mean(a, b)
    } else {
        // odd
        Percentile::Single(&values[values.len() / 2])
    }
}
/// quickselect algorithm
///
/// Consider using [`percentile_rand`] or [`median`].
///
/// `pivot_fn` must return an integer if range [0..values.len()).
pub fn percentile<T: Ord + Clone>(
    values: &mut [T],
    target_percentile: Fraction,
    pivot_fn: &mut impl FnMut(&mut [T]) -> Cow<'_, T>,
) -> Percentile<T> {
    assert!(
        values.len() >= target_percentile.denominator,
        "percentile calculation got too few values"
    );
    let len = values.len();
    let target = percentile_index(len, target_percentile);
    match target {
        Percentile::Single(v) => quickselect(values, v, pivot_fn).clone_inner(),
        Percentile::Mean(a, b) => Percentile::Mean(
            quickselect(values, a, pivot_fn)
                .into_single()
                .unwrap()
                .clone(),
            quickselect(values, b, pivot_fn)
                .into_single()
                .unwrap()
                .clone(),
        ),
    }
}
/// Convenience function for [`percentile`] with a random `pivot_fn`.
pub fn percentile_rand<T: Ord + Clone>(
    values: &mut [T],
    target_percentile: Fraction,
) -> Percentile<T> {
    let mut rng = rand::thread_rng();
    percentile(values, target_percentile, &mut |slice| {
        let idx = rng.sample(rand::distributions::Uniform::new(0_usize, slice.len()));
        Cow::Borrowed(&slice[idx])
    })
}
/// Convenience function for [`percentile`] with the 50% mark as the target and a random
/// `pivot_fn`.
pub fn median<T: Ord + Clone>(values: &mut [T]) -> Percentile<T> {
    percentile_rand(values, Fraction::new(1, 2))
}
/// Low level function used by this module.
fn quickselect<'a, T: Ord + Clone>(
    values: &'a mut [T],
    k: usize,
    pivot_fn: &mut impl FnMut(&mut [T]) -> Cow<'_, T>,
) -> Percentile<&'a T> {
    if values.len() == 1 {
        assert_eq!(k, 0);
        return Percentile::Single(&values[0]);
    }
    if values.len() <= 5 {
        let naive = naive_percentile(values);
        return naive;
    }

    let pivot = pivot_fn(values);
    let pivot = pivot.into_owned();

    let (lows, highs_inclusive) = include(values, |v| *v < pivot);
    let (highs, pivots) = include(highs_inclusive, |v| *v > pivot);

    if k < lows.len() {
        quickselect(lows, k, pivot_fn)
    } else if k < lows.len() + pivots.len() {
        Percentile::Single(&pivots[0])
    } else {
        quickselect(highs, k - lows.len() - pivots.len(), pivot_fn)
    }
}
/// Moves items in the slice and splits it so the first returned slice contains all elements where
/// `predicate` is true. The second contains all other.
pub fn include<T>(slice: &mut [T], mut predicate: impl FnMut(&T) -> bool) -> (&mut [T], &mut [T]) {
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
    target_percentile: Fraction,
) -> Percentile<T> {
    percentile(values, target_percentile, &mut median_of_medians_pivot_fn)
}
/// Pick a good pivot within l, a list of numbers
/// This algorithm runs in O(n) time.
fn median_of_medians_pivot_fn<T: Ord + Clone + PercentileResolve>(l: &mut [T]) -> Cow<'_, T> {
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
    let median_of_medians = percentile(
        &mut medians,
        Fraction::new(1, 2),
        &mut median_of_medians_pivot_fn,
    );
    Cow::Owned(median_of_medians.resolve())
}

fn percentile_index(len: usize, percentile: Fraction) -> Percentile<usize> {
    // median
    if percentile == Fraction::HALF {
        if len % 2 == 0 {
            Percentile::Mean(len / 2, len / 2 - 1)
        } else {
            Percentile::Single(len / 2)
        }
    } else if percentile == Fraction::ONE_QUARTER || percentile == Fraction::THREE_QUARTERS {
        percentile_index(len / 2, Fraction::HALF).map(|v| {
            if percentile == Fraction::THREE_QUARTERS {
                v + len / 2 + len % 2
            } else {
                v
            }
        })
    } else {
        // ceil(len * percentile) - 1
        let m = len * percentile.numerator;
        let rem = m % percentile.denominator;
        let rem = usize::from(rem > 1);
        Percentile::Single(m / percentile.denominator + rem - 1)
    }
}

/// Operations on [`crate::Cluster`]s.
///
/// This attempts to implement all functionality of this module but using clusters.
/// This turned out to be quite difficult.
pub mod cluster {
    use std::ops::{Deref, DerefMut};

    use crate::{Cluster, ClusterList, OwnedClusterList};

    use super::*;

    /// Percentile by sorting.
    ///
    /// # Performance & scalability
    ///
    /// This will be very quick for small sets.
    /// O(n) performance when `values.len() < 5`, else O(n log n).
    pub fn naive_percentile(
        values: &mut OwnedClusterList,
        target_percentile: Fraction,
    ) -> Percentile<f64> {
        values
            .list
            .sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let values = values.borrow();
        let len = values.len();
        // `TODO`: check if this works with percentile
        let even = len * target_percentile.numerator % target_percentile.denominator == 0;
        let target = len * target_percentile.numerator / target_percentile.denominator;
        let mut len = len;

        for (pos, (v, count)) in values.list.iter().enumerate() {
            len -= *count;
            if len <= target && even {
                let overstep = target - len;
                return Percentile::Mean(
                    *v,
                    values.list[pos + if overstep == 0 { 1 } else { 0 }].0,
                );
            }
            if len <= target && !even {
                return Percentile::Single(*v);
            }
        }
        Percentile::Single(0.0)
    }
    /// quickselect algorithm
    ///
    /// Consider using [`percentile_rand`] or [`median`].
    ///
    /// `pivot_fn` must return an integer if range [0..values.len()).
    pub fn percentile(
        values: &mut OwnedClusterList,
        target_percentile: Fraction,
        pivot_fn: &mut impl FnMut(&ClusterList) -> f64,
    ) -> Percentile<f64> {
        assert!(
            values.borrow().len() >= target_percentile.denominator,
            "percentile calculation got too few values"
        );
        let len = values.borrow().len();
        let target = percentile_index(len, target_percentile);
        match target {
            Percentile::Single(k) => quickselect(&mut values.into(), k, pivot_fn),
            Percentile::Mean(a, b) => Percentile::Mean(
                quickselect(&mut values.into(), a, pivot_fn)
                    .into_single()
                    .unwrap(),
                quickselect(&mut values.into(), b, pivot_fn)
                    .into_single()
                    .unwrap(),
            ),
        }
    }
    /// Convenience function for [`percentile`] with a random `pivot_fn`.
    pub fn percentile_rand(
        values: &mut OwnedClusterList,
        target_percentile: Fraction,
    ) -> Percentile<f64> {
        let mut rng = rand::thread_rng();
        percentile(values, target_percentile, &mut |slice| {
            let mut idx = rng.sample(rand::distributions::Uniform::new(0_usize, slice.len()));
            for item in slice.list {
                if idx < item.1 {
                    return item.0;
                }
                idx -= item.1;
            }
            slice.list.last().unwrap().0
        })
    }
    /// Convenience function for [`percentile`] with the 50% mark as the target and a random
    /// `pivot_fn`.
    pub fn median(values: &mut OwnedClusterList) -> Percentile<f64> {
        percentile_rand(values, Fraction::new(1, 2))
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
    ) -> Percentile<f64> {
        if values.len() == 1 {
            // assert_eq!(k, 0);
            return Percentile::Single(values[0].0);
        }

        let pivot = pivot_fn(&values.list());

        let (mut lows, mut highs_inclusive) = include(values, |v| v < pivot);
        let (mut highs, pivots) = include(&mut highs_inclusive, |v| v > pivot);

        if k < lows.list().len() {
            quickselect(&mut lows, k, pivot_fn)
        } else if k < lows.list().len() + pivots.list().len() {
            Percentile::Single(pivots[0].0)
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
