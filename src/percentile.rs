use std::borrow::Cow;

/// Percentile / median calculations.
///
/// - `O(n log n)` [`naive_percentile`] (simple to understand)
/// - probabilistic `O(n)` [`percentile`] (recommended, fastest, and also quite simple to understand)
/// - deterministic `O(n)` [`median_of_medians`] (harder to understand, probably slower than the
///     probabilistic version.)
///
/// You should probably use [`percentile_rand`].
///
/// The linear time algoritms are implementations following [this blogpost](https://rcoh.me/posts/linear-time-median-finding/).
use rand::Rng;

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
}
impl<T: Clone> Percentile<&T> {
    pub fn clone_inner(&self) -> Percentile<T> {
        match self {
            Self::Single(t) => Percentile::Single((*t).clone()),
            Self::Mean(a, b) => Percentile::Mean((*a).clone(), (*b).clone()),
        }
    }
}
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

macro_rules! impl_percentile_resolv_float {
    ($($t: ty, )+) => {
        $(
        impl PercentileResolve for $t {
            fn mean(a: Self, b: Self) -> Self {
                (a + b) / 2.0
            }
        }
        )+
    };
}
macro_rules! impl_percentile_resolv_int {
    ($($t: ty, )+) => {
        $(
        /// This implementation may not be exact if the mean of the values does not give an
        /// integer.
        /// In that case, the value is rounded down.
        impl PercentileResolve for $t {
            fn mean(a: Self, b: Self) -> Self {
                (a + b) / 2
            }
        }
        )+
    };
}
impl_percentile_resolv_float!(f32, f64,);
impl_percentile_resolv_int!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize,);
// impl <T: Deref<Target = Resolv>, Resolv: PercentileResolve> PercentileResolve for T {
// fn mean(a: Self, b: Self) -> Self {
// a.deref().mean(b.deref())
// }
// }

pub struct Fraction {
    pub numerator: usize,
    pub denominator: usize,
}
impl Fraction {
    pub fn new(numerator: usize, denominator: usize) -> Self {
        Self {
            numerator,
            denominator,
        }
    }
}

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
    let target_len =
        (len * target_percentile.numerator / target_percentile.denominator).clamp(0, len);
    if (len * target_percentile.numerator) % target_percentile.denominator == 1 {
        quickselect(values, target_len, pivot_fn).clone_inner()
    } else {
        Percentile::Mean(
            quickselect(values, target_len - 1, pivot_fn)
                .into_single()
                .unwrap()
                .clone(),
            quickselect(values, target_len, pivot_fn)
                .into_single()
                .unwrap()
                .clone(),
        )
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
    let (pivots, highs) = include(highs_inclusive, |v| *v > pivot);

    if k < lows.len() {
        quickselect(lows, k, pivot_fn)
    } else if k < lows.len() + pivots.len() {
        Percentile::Single(&pivots[0])
    } else {
        quickselect(highs, k - lows.len() - pivots.len(), pivot_fn)
    }
}
fn include<T>(slice: &mut [T], mut predicate: impl FnMut(&T) -> bool) -> (&mut [T], &mut [T]) {
    let add_index = 0;
    let mut index = 0;
    let len = slice.len();
    while index < len {
        let value = &mut slice[index];
        if predicate(value) {
            slice.swap(index, add_index);
        }
        index += 1;
    }

    slice.split_at_mut(add_index)
}
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
