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

impl PercentileResolve for f64 {
    fn mean(a: Self, b: Self) -> Self {
        (a + b) / 2.0
    }
}
impl PercentileResolve for f32 {
    fn mean(a: Self, b: Self) -> Self {
        (a + b) / 2.0
    }
}

// pub trait Slice<'a, T> {
// // type Iter: Iterator<Item = &'a T>;

// fn len(&self) -> usize;
// fn is_empty(&self) -> bool {
// self.len() == 0
// }
// /// `len` is the number of elements from the start - how long the new slice should be.
// fn split_to(&'a mut self, len: usize) -> Self;
// /// `len` is the number of elements from the end - how long the new slice should be.
// fn split_from(&'a mut self, len: usize) -> Self;
// fn value(&self, idx: usize) -> &T;
// fn sort(&mut self);
// // fn iter(&'a self) -> Self::Iter;
// // fn retain(&'a mut self, predicate: &fn(&T) -> bool);
// /// First returned value contains all values where `predicate` is true.
// /// The second contains the rest.
// fn include(&'a mut self, predicate: impl FnMut(&T) -> bool) -> (Vec<T>, Vec<T>)
// where
// Self: Sized;
// }
// impl<'a, T: Ord + Clone> Slice<'a, T> for &'a mut [T] {
// // type Iter = std::slice::Iter<'a, T>;
// fn len(&self) -> usize {
// (**self).len()
// }
// fn split_to(&'a mut self, len: usize) -> Self {
// &mut self[..len]
// }
// fn split_from(&'a mut self, len: usize) -> Self {
// let idx = self.len() - len;
// &mut self[idx..]
// }
// fn value(&self, idx: usize) -> &T {
// &self[idx]
// }
// fn sort(&mut self) {
// self.sort_unstable()
// }
// // fn iter(&'a self) -> Self::Iter {
// // (**self).iter()
// // }
// // fn retain(&'a mut self, predicate: &fn(&T) -> bool) {
// // let mut add_index = 0;
// // let mut index = 0;
// // let len = self.len();
// // while index < len {
// // let value = &mut self[index];
// // if predicate(value) {
// // self.swap(index, add_index);
// // }
// // index += 1;
// // }
// // let me = self.split_to(add_index);
// // std::mem::replace(self, me);
// // }
// fn include(&'a mut self, mut predicate: impl FnMut(&T) -> bool) -> (Vec<T>, Vec<T>)
// where
// Self: Sized,
// {
// let add_index = 0;
// let mut index = 0;
// let len = self.len();
// while index < len {
// let value = &mut self[index];
// if predicate(value) {
// self.swap(index, add_index);
// }
// index += 1;
// }

// // `FIXME`: don't convert these to Vecs. Use traits!
// // The issue is inherit to lifetimes: we create a new object which need to be valid for 'a
// // lifetime, because it's `Self`. That's however not the case. But Rust won't let me.
// let (a, b) = self.split_at_mut(add_index);
// (a.to_vec(), b.to_vec())
// }
// }
// // impl<'a, T: Ord> Slice<'a, T> for [T] {
// // fn len(&self) -> usize {
// // (*self).len()
// // }
// // fn split_to(&'a mut self, len: usize) -> &'a mut Self {
// // &mut self[..len]
// // }
// // fn split_from(&'a mut self, len: usize) -> &'a mut Self {
// // let idx = self.len() - len;
// // &mut self[idx..]
// // }
// // fn value(&'a self, idx: usize) -> &'a T {
// // &self[idx]
// // }
// // fn sort(&mut self) {
// // self.sort_unstable()
// // }
// // }

// /// Percentile by sorting.
// ///
// /// # Performance & scalability
// ///
// /// This will be very quick for small sets.
// /// O(n) performance when `values.len() < 5`, else O(n log n).
// pub fn naive_percentile<'a, T: Ord, S: Slice<'a, T>>(values: &mut S) -> Percentile<&T> {
// assert!(!values.is_empty());
// values.sort();
// if values.len() % 2 == 0 {
// // even
// let a = values.value(values.len() / 2 - 1);
// let b = values.value(values.len() / 2);
// Percentile::Mean(a, b)
// } else {
// // odd
// Percentile::Single(values.value(values.len() / 2))
// }
// }
// /// quickselect algorithm
// ///
// /// `pivot_fn` must return an integer if range [0..values.len()).
// pub fn percentile<'a, T: Ord + Clone + 'a, S: Slice<'a, T>>(
// values: &'a mut S,
// pivot_fn: &mut impl FnMut(&S) -> usize,
// ) -> Percentile<T> {
// let len = values.len();
// if len % 2 == 1 {
// quickselect(values, len / 2, pivot_fn)
// } else {
// Percentile::Mean(
// quickselect(values, len / 2 - 1, pivot_fn)
// .into_single()
// .unwrap(),
// quickselect(values, len / 2, pivot_fn)
// .into_single()
// .unwrap(),
// )
// }
// }
// fn quickselect<'a, T: Ord + Clone, S: Slice<'a, T>>(
// values: &mut S,
// k: usize,
// pivot_fn: &mut impl FnMut(&S) -> usize,
// ) -> Percentile<T> {
// if values.len() == 1 {
// assert_eq!(k, 0);
// return Percentile::Single(values.value(0).clone());
// }
// if values.len() <= 5 {
// let naive = naive_percentile(values);
// return naive.clone_inner();
// }

// let pivot = pivot_fn(values);

// let pivot_value = values.value(pivot).clone();
// let (mut lows, mut highs_inclusive) = values.include(|v| *v < pivot_value);
// let (pivots, mut highs) = highs_inclusive.as_mut_slice().include(|v| *v > pivot_value);

// if k < lows.len() {
// quickselect(&mut lows.as_mut_slice(), k, pivot_fn)
// } else if k < lows.len() + pivots.len() {
// Percentile::Single(pivots.as_mut_slice().value(0).clone())
// } else {
// quickselect(&mut highs, k - lows.len() - pivots.len(), pivot_fn)
// }
// }

use rand::Rng;

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
    pivot_fn: &mut impl FnMut(&[T]) -> usize,
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
        rng.sample(rand::distributions::Uniform::new(0_usize, slice.len()))
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
    pivot_fn: &mut impl FnMut(&[T]) -> usize,
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

    let pivot_value = values[pivot].clone();
    let (lows, highs_inclusive) = include(values, |v| *v < pivot_value);
    let (pivots, highs) = include(highs_inclusive, |v| *v > pivot_value);

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
