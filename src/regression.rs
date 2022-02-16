//! Vocabulary:
//!
//! - Predictors - the independent values (usually denoted `x`) from which we want a equation to get the:
//! - outcomes - the dependant variables. Usually `y` or `f(x)`.
//! - model - create an equation which optimally (can optimize for different priorities) fits the data.
//!
//! The `*Coefficients` structs implement [`Predictive`] which calculates the [predicted outcomes](Predictive::predict_outcome)
//! using the model and their [error](Predictive::error); and [`Display`] which can be used to
//! show the equations.
//!
//! Linear regressions are often used by other regression methods. All linear regressions therefore
//! implement the [`LinearEstimator`] trait. You can use the `*Linear` structs to choose which method to
//! use.
//!
//! # Info on implementation
//!
//! Details and comments on implementation can be found as docs under each item.
//!
//! ## Power & exponent
//!
//! I reverse the exponentiation to get a linear model. Then, I solve it using the method linked
//! above. Then, I transform the returned variables to fit the target model.
//!
//! Many programs (including LibreOffice Calc) simply discards negative & zero values. I chose to
//! go the explicit route and add additional terms to satisfy requirements.
//! This is naturally a fallback, and should be a warning sign your data is bad.
//!
//! Under these methods the calculations are inserted, and how to handle the data.

use std::fmt::{self, Display};
use std::ops::Deref;

pub use derived::{
    exponential, exponential_ols, power, power_ols, ExponentialCoefficients, PowerCoefficients,
};
pub use ols::LinearOls;

trait Model: Predictive + Display {}
impl<T: Predictive + Display> Model for T {}

pub struct DynModel {
    model: Box<dyn Model>,
}
impl DynModel {
    pub fn new(model: impl Predictive + Display + 'static) -> Self {
        Self {
            model: Box::new(model),
        }
    }
}
impl Predictive for DynModel {
    fn predict_outcome(&self, predictor: f64) -> f64 {
        self.model.predict_outcome(predictor)
    }
}
impl Display for DynModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.model.fmt(f)
    }
}

pub trait Predictive {
    /// Calculates the predicted outcome of `predictor`.
    fn predict_outcome(&self, predictor: f64) -> f64;
}
/// Helper trait to make the [`Determination::error`] method take a generic iterator.
///
/// This enables [`Predicative`] to be `dyn`.
pub trait Determination: Predictive {
    /// Calculates the R² (coefficient of determination), the proportion of variation in predicted
    /// model.
    ///
    /// `predictors` are the x values (input to the function).
    /// `outcomes` are the observed dependant variable.
    /// `len` is the count of data points.
    ///
    /// If `predictors` and `outcomes` have different lengths, the result might be unexpected.
    ///
    /// O(n)
    // For implementation, see https://en.wikipedia.org/wiki/Coefficient_of_determination#Definitions
    fn error(
        &self,
        predictors: impl Iterator<Item = f64>,
        outcomes: impl Iterator<Item = f64> + Clone,
        len: usize,
    ) -> f64 {
        let outcomes_mean = outcomes.clone().sum::<f64>() / len as f64;
        let residuals = predictors
            .zip(outcomes.clone())
            .map(|(pred, out)| out - self.predict_outcome(pred));
        // Sum of the square of the residuals
        let res: f64 = residuals.map(|residual| residual * residual).sum();
        let tot: f64 = outcomes
            .map(|out| {
                let diff = out - outcomes_mean;
                diff * diff
            })
            .sum();

        1.0 - (res / tot)
    }
}
impl<T: Predictive> Determination for T {}

#[derive(Debug, Clone, PartialEq)]
pub struct LinearCoefficients {
    /// slope, x coefficient
    pub k: f64,
    /// y intersect, additive
    pub m: f64,
}
impl Predictive for LinearCoefficients {
    fn predict_outcome(&self, predictor: f64) -> f64 {
        self.k * predictor + self.m
    }
}
impl Display for LinearCoefficients {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let p = f.precision().unwrap_or(5);
        write!(f, "{:.2$}x + {:.2$}", self.k, self.m, p)
    }
}

pub trait LinearEstimator {
    fn model(&self, predictors: &[f64], outcomes: &[f64]) -> LinearCoefficients;
}

/// Estimators derived from others, usual [`LinearEstimator`].
///
/// See the docs on the items for more info about how they're created.
pub mod derived {
    use super::*;
    fn min(slice: &[f64]) -> Option<f64> {
        slice
            .iter()
            .copied()
            .map(crate::F64OrdHash)
            .min()
            .map(|f| f.0)
    }

    #[derive(Debug, Clone, PartialEq)]
    pub struct PowerCoefficients {
        /// Constant
        pub k: f64,
        /// exponent
        pub e: f64,
        /// If the predictors needs to have an offset applied to remove values under 1.
        pub predictor_additive: Option<f64>,
        /// If the outcomes needs to have an offset applied to remove values under 1.
        pub outcome_additive: Option<f64>,
    }
    impl Predictive for PowerCoefficients {
        fn predict_outcome(&self, predictor: f64) -> f64 {
            self.k * (predictor + self.predictor_additive.unwrap_or(0.0)).powf(self.e)
                - self.outcome_additive.unwrap_or(0.0)
        }
    }
    impl Display for PowerCoefficients {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let p = f.precision().unwrap_or(5);
            write!(
                f,
                "{:.3$} * {x}^{:.3$}{}",
                self.k,
                self.e,
                if let Some(out) = self.outcome_additive {
                    format!(" - {:.1$}", out, p)
                } else {
                    String::new()
                },
                p,
                x = if let Some(pred) = self.predictor_additive {
                    format!("(x + {:.1$})", pred, p)
                } else {
                    "x".to_string()
                },
            )
        }
    }

    /// Convenience-method for [`power`] using [`LinearOls`].
    pub fn power_ols(predictors: &mut [f64], outcomes: &mut [f64]) -> PowerCoefficients {
        power(predictors, outcomes, &LinearOls)
    }
    /// Fits a curve with the equation `y = a * x^b` (optionally with an additional subtractive term if
    /// any outcome is negative and an additive to the `x` if any predictor is negative).
    ///
    /// Also sometimes called "growth".
    ///
    /// # Panics
    ///
    /// Panics if either `x` or `y` don't have the length `len`.
    /// `len` must be greater than 2.
    ///
    /// # Derivation
    ///
    /// y=b * x^a
    ///
    /// lg(y) = lg(b * x^a)
    /// lg(y) = lg(b) + a(lg x)
    ///
    /// Transform: y => lg (y), x => lg(x)
    ///
    /// When values found, take 10^b to get b and a is a
    pub fn power<E: LinearEstimator>(
        predictors: &mut [f64],
        outcomes: &mut [f64],
        estimator: &E,
    ) -> PowerCoefficients {
        assert!(predictors.len() > 2);
        assert!(outcomes.len() > 2);
        let predictor_min = min(predictors).unwrap();
        let outcome_min = min(outcomes).unwrap();
        power_given_min(predictors, outcomes, predictor_min, outcome_min, estimator)
    }
    /// Same as [`power`] without the [`Clone`] requirement for the iterators, but takes a min
    /// value.
    ///
    /// # Panics
    ///
    /// See [`power`].
    pub fn power_given_min<E: LinearEstimator>(
        predictors: &mut [f64],
        outcomes: &mut [f64],
        predictor_min: f64,
        outcome_min: f64,
        estimator: &E,
    ) -> PowerCoefficients {
        assert_eq!(predictors.len(), outcomes.len());
        assert!(predictors.len() > 2);

        // If less than 1, exception. Read more about this in the `power` function docs.
        let predictor_additive = if predictor_min < 1.0 {
            Some(1.0 - predictor_min)
        } else {
            None
        };
        let outcome_additive = if outcome_min < 1.0 {
            Some(1.0 - outcome_min)
        } else {
            None
        };

        predictors
            .iter_mut()
            .for_each(|pred| *pred = (*pred + predictor_additive.unwrap_or(0.0)).log2());
        outcomes
            .iter_mut()
            .for_each(|y| *y = (*y + outcome_additive.unwrap_or(0.0)).log2());

        let coefficients = estimator.model(predictors, outcomes);
        let k = 2.0_f64.powf(coefficients.m);
        let e = coefficients.k;
        PowerCoefficients {
            k,
            e,
            predictor_additive,
            outcome_additive,
        }
    }

    #[derive(Debug)]
    pub struct ExponentialCoefficients {
        /// Constant
        pub k: f64,
        /// base
        pub b: f64,
        /// If the predictors needs to have an offset applied to remove values under 1.
        pub predictor_additive: Option<f64>,
        /// If the outcomes needs to have an offset applied to remove values under 1.
        pub outcome_additive: Option<f64>,
    }
    impl Predictive for ExponentialCoefficients {
        fn predict_outcome(&self, predictor: f64) -> f64 {
            self.k
                * self
                    .b
                    .powf(predictor + self.predictor_additive.unwrap_or(0.0))
                - self.outcome_additive.unwrap_or(0.0)
        }
    }
    impl Display for ExponentialCoefficients {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let p = f.precision().unwrap_or(5);
            write!(
                f,
                "{:.3$} * {:.3$}^{x}{}",
                self.k,
                self.b,
                if let Some(out) = self.outcome_additive {
                    format!(" - {:.1$}", out, p)
                } else {
                    String::new()
                },
                p,
                x = if let Some(pred) = self.predictor_additive {
                    format!("(x + {:.1$})", pred, p)
                } else {
                    "x".to_string()
                },
            )
        }
    }

    /// Convenience-method for [`exponential`] using [`LinearOls`].
    pub fn exponential_ols(
        predictors: &mut [f64],
        outcomes: &mut [f64],
    ) -> ExponentialCoefficients {
        exponential(predictors, outcomes, &LinearOls)
    }
    /// Fits a curve with the equation `y = a * b^x` (optionally with an additional subtractive term if
    /// any outcome is negative and an additive to the `x` if any predictor is negative).
    ///
    /// Also sometimes called "growth".
    ///
    /// # Panics
    ///
    /// Panics if either `x` or `y` don't have the length `len`.
    /// `len` must be greater than 2.
    ///
    /// # Derivation
    ///
    /// y=b * a^x
    ///
    /// lg(y) = lg(b * a^x)
    /// lg(y) = lg(b) + x(lg a)
    ///
    /// Transform: y => lg (y), x => x
    ///
    /// When values found, take 10^b to get b and 10^a to get a
    pub fn exponential<E: LinearEstimator>(
        predictors: &mut [f64],
        outcomes: &mut [f64],
        estimator: &E,
    ) -> ExponentialCoefficients {
        assert!(predictors.len() > 2);
        assert!(outcomes.len() > 2);
        let predictor_min = min(predictors).unwrap();
        let outcome_min = min(outcomes).unwrap();
        exponential_given_min(predictors, outcomes, predictor_min, outcome_min, estimator)
    }
    /// Same as [`exponential`] without the [`Clone`] requirement for the iterators, but takes a min
    /// value.
    ///
    /// # Panics
    ///
    /// See [`exponential`].
    pub fn exponential_given_min<E: LinearEstimator>(
        predictors: &mut [f64],
        outcomes: &mut [f64],
        predictor_min: f64,
        outcome_min: f64,
        estimator: &E,
    ) -> ExponentialCoefficients {
        assert_eq!(predictors.len(), outcomes.len());
        assert!(predictors.len() > 2);

        // If less than 1, exception. Read more about this in the `exponential` function docs.
        let predictor_additive = if predictor_min < 1.0 {
            Some(1.0 - predictor_min)
        } else {
            None
        };
        let outcome_additive = if outcome_min < 1.0 {
            Some(1.0 - outcome_min)
        } else {
            None
        };

        if let Some(predictor_additive) = predictor_additive {
            predictors
                .iter_mut()
                .for_each(|pred| *pred += predictor_additive);
        }
        outcomes
            .iter_mut()
            .for_each(|y| *y = (*y + outcome_additive.unwrap_or(0.0)).log2());

        let coefficients = estimator.model(predictors, outcomes);
        let k = 2.0_f64.powf(coefficients.m);
        let b = 2.0_f64.powf(coefficients.k);
        ExponentialCoefficients {
            k,
            b,
            predictor_additive,
            outcome_additive,
        }
    }
}

/// This module enables the use of [`rug::Float`] inside of [`nalgebra`].
///
/// Many functions are not implemented. PRs are welcome.
#[cfg(feature = "arbitrary-precision")]
pub mod arbitrary_linear_algebra {
    use std::fmt::{self, Display};
    use std::ops::{
        Add, AddAssign, Deref, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
    };

    use nalgebra::{ComplexField, RealField};
    use rug::Assign;

    pub const HARDCODED_PRECISION: u32 = 256;
    #[derive(Debug, Clone, PartialEq, PartialOrd)]
    pub struct FloatWrapper(pub rug::Float);
    impl From<rug::Float> for FloatWrapper {
        fn from(f: rug::Float) -> Self {
            Self(f)
        }
    }

    impl simba::scalar::SupersetOf<f64> for FloatWrapper {
        fn is_in_subset(&self) -> bool {
            self.0.prec() <= 53
        }
        fn to_subset(&self) -> Option<f64> {
            if simba::scalar::SupersetOf::<f64>::is_in_subset(self) {
                Some(self.0.to_f64())
            } else {
                None
            }
        }
        fn to_subset_unchecked(&self) -> f64 {
            self.0.to_f64()
        }
        fn from_subset(element: &f64) -> Self {
            rug::Float::with_val(HARDCODED_PRECISION, element).into()
        }
    }
    impl simba::scalar::SubsetOf<Self> for FloatWrapper {
        fn to_superset(&self) -> Self {
            self.clone()
        }

        fn from_superset_unchecked(element: &Self) -> Self {
            element.clone()
        }

        fn is_in_subset(_element: &Self) -> bool {
            true
        }
    }
    impl num_traits::cast::FromPrimitive for FloatWrapper {
        fn from_i64(n: i64) -> Option<Self> {
            Some(rug::Float::with_val(HARDCODED_PRECISION, n).into())
        }
        fn from_u64(n: u64) -> Option<Self> {
            Some(rug::Float::with_val(HARDCODED_PRECISION, n).into())
        }
    }
    impl Display for FloatWrapper {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            self.0.fmt(f)
        }
    }
    impl simba::simd::SimdValue for FloatWrapper {
        type Element = FloatWrapper;
        type SimdBool = bool;

        #[inline(always)]
        fn lanes() -> usize {
            1
        }

        #[inline(always)]
        fn splat(val: Self::Element) -> Self {
            val
        }

        #[inline(always)]
        fn extract(&self, _: usize) -> Self::Element {
            self.clone()
        }

        #[inline(always)]
        unsafe fn extract_unchecked(&self, _: usize) -> Self::Element {
            self.clone()
        }

        #[inline(always)]
        fn replace(&mut self, _: usize, val: Self::Element) {
            *self = val
        }

        #[inline(always)]
        unsafe fn replace_unchecked(&mut self, _: usize, val: Self::Element) {
            *self = val
        }

        #[inline(always)]
        fn select(self, cond: Self::SimdBool, other: Self) -> Self {
            if cond {
                self
            } else {
                other
            }
        }
    }
    impl Neg for FloatWrapper {
        type Output = Self;
        fn neg(self) -> Self::Output {
            Self(-self.0)
        }
    }
    impl Add for FloatWrapper {
        type Output = Self;
        fn add(mut self, rhs: Self) -> Self::Output {
            self.0 += rhs.0;
            self
        }
    }
    impl Sub for FloatWrapper {
        type Output = Self;
        fn sub(mut self, rhs: Self) -> Self::Output {
            self.0 -= rhs.0;
            self
        }
    }
    impl Mul for FloatWrapper {
        type Output = Self;
        fn mul(mut self, rhs: Self) -> Self::Output {
            self.0 *= rhs.0;
            self
        }
    }
    impl Div for FloatWrapper {
        type Output = Self;
        fn div(mut self, rhs: Self) -> Self::Output {
            self.0 /= rhs.0;
            self
        }
    }
    impl Rem for FloatWrapper {
        type Output = Self;
        fn rem(mut self, rhs: Self) -> Self::Output {
            self.0 %= rhs.0;
            self
        }
    }
    impl AddAssign for FloatWrapper {
        fn add_assign(&mut self, rhs: Self) {
            self.0 += rhs.0;
        }
    }
    impl SubAssign for FloatWrapper {
        fn sub_assign(&mut self, rhs: Self) {
            self.0 -= rhs.0;
        }
    }
    impl MulAssign for FloatWrapper {
        fn mul_assign(&mut self, rhs: Self) {
            self.0 *= rhs.0;
        }
    }
    impl DivAssign for FloatWrapper {
        fn div_assign(&mut self, rhs: Self) {
            self.0 /= rhs.0;
        }
    }
    impl RemAssign for FloatWrapper {
        fn rem_assign(&mut self, rhs: Self) {
            self.0 %= rhs.0;
        }
    }
    impl num_traits::Zero for FloatWrapper {
        fn zero() -> Self {
            Self(rug::Float::with_val(HARDCODED_PRECISION, 0.0))
        }
        fn is_zero(&self) -> bool {
            self.0 == 0.0
        }
    }
    impl num_traits::One for FloatWrapper {
        fn one() -> Self {
            Self(rug::Float::with_val(HARDCODED_PRECISION, 1.0))
        }
    }
    impl num_traits::Num for FloatWrapper {
        type FromStrRadixErr = rug::float::ParseFloatError;
        fn from_str_radix(s: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
            rug::Float::parse_radix(s, radix as i32).map(|f| Self(rug::Float::with_val(HARDCODED_PRECISION, f)))
        }
    }
    impl num_traits::Signed for FloatWrapper {
        fn abs(&self) -> Self {
            self.0.as_abs().to_owned().into()
        }
        fn abs_sub(&self, other: &Self) -> Self {
            if self.0 <= other.0 {
                rug::Float::with_val(self.prec(), 0.0f64).into()
            } else {
                Self(self.0.clone() - &other.0)
            }
        }
        fn signum(&self) -> Self {
            self.0.clone().signum().into()
        }
        fn is_positive(&self) -> bool {
            self.0.is_sign_positive()
        }
        fn is_negative(&self) -> bool {
            self.0.is_sign_negative()
        }
    }
    impl approx::AbsDiffEq for FloatWrapper {
        type Epsilon = Self;
        fn default_epsilon() -> Self::Epsilon {
            rug::Float::with_val(HARDCODED_PRECISION, f64::EPSILON).into()
        }
        fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
            if self.0 == other.0 {
                return true;
            }
            if self.0.is_infinite() || other.0.is_infinite() {
                return false;
            }
            let mut buffer = self.clone();
            buffer.0.assign(&self.0 - &other.0);
            buffer.0.abs_mut();
            let abs_diff = buffer;
            abs_diff.0 <= epsilon.0
        }
    }
    impl approx::RelativeEq for FloatWrapper {
        fn default_max_relative() -> Self::Epsilon {
            rug::Float::with_val(HARDCODED_PRECISION, f64::EPSILON).into()
        }
        fn relative_eq(
            &self,
            other: &Self,
            epsilon: Self::Epsilon,
            max_relative: Self::Epsilon,
        ) -> bool {
            if self.0 == other.0 {
                return true;
            }
            if self.0.is_infinite() || other.0.is_infinite() {
                return false;
            }
            let mut buffer = self.clone();
            buffer.0.assign(&self.0 - &other.0);
            buffer.0.abs_mut();
            let abs_diff = buffer;
            if abs_diff.0 <= epsilon.0 {
                return true;
            }

            let abs_self = self.0.as_abs();
            let abs_other = other.0.as_abs();

            let largest = if *abs_other > *abs_self {
                &*abs_other
            } else {
                &*abs_self
            };

            abs_diff.0 <= largest * max_relative.0
        }
    }
    impl approx::UlpsEq for FloatWrapper {
        fn default_max_ulps() -> u32 {
            // Should not be used, see comment below.
            4
        }
        fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, _max_ulps: u32) -> bool {
            // taking the difference of the bits makes no sense when using arbitrary floats.
            approx::AbsDiffEq::abs_diff_eq(&self, &other, epsilon)
        }
    }
    impl nalgebra::Field for FloatWrapper {}
    impl RealField for FloatWrapper {
        fn is_sign_positive(&self) -> bool {
            todo!()
        }

        fn is_sign_negative(&self) -> bool {
            todo!()
        }

        fn copysign(self, _sign: Self) -> Self {
            todo!()
        }

        fn max(self, _other: Self) -> Self {
            todo!()
        }

        fn min(self, _other: Self) -> Self {
            todo!()
        }

        fn clamp(self, _min: Self, _max: Self) -> Self {
            todo!()
        }

        fn atan2(self, _other: Self) -> Self {
            todo!()
        }

        fn min_value() -> Option<Self> {
            todo!()
        }

        fn max_value() -> Option<Self> {
            todo!()
        }

        fn pi() -> Self {
            todo!()
        }

        fn two_pi() -> Self {
            todo!()
        }

        fn frac_pi_2() -> Self {
            todo!()
        }

        fn frac_pi_3() -> Self {
            todo!()
        }

        fn frac_pi_4() -> Self {
            todo!()
        }

        fn frac_pi_6() -> Self {
            todo!()
        }

        fn frac_pi_8() -> Self {
            todo!()
        }

        fn frac_1_pi() -> Self {
            todo!()
        }

        fn frac_2_pi() -> Self {
            todo!()
        }

        fn frac_2_sqrt_pi() -> Self {
            todo!()
        }

        fn e() -> Self {
            todo!()
        }

        fn log2_e() -> Self {
            todo!()
        }

        fn log10_e() -> Self {
            todo!()
        }

        fn ln_2() -> Self {
            todo!()
        }

        fn ln_10() -> Self {
            todo!()
        }
    }
    impl ComplexField for FloatWrapper {
        type RealField = Self;

        fn from_real(re: Self::RealField) -> Self {
            re
        }
        fn real(self) -> Self::RealField {
            self
        }
        fn imaginary(mut self) -> Self::RealField {
            self.0.assign(0.0);
            self
        }
        fn modulus(self) -> Self::RealField {
            self.abs()
        }
        fn modulus_squared(self) -> Self::RealField {
            self.0.square().into()
        }
        fn argument(mut self) -> Self::RealField {
            if self.0.is_sign_positive() || self.0.is_zero() {
                self.0.assign(0.0);
                self
            } else {
                Self::pi()
            }
        }
        fn norm1(self) -> Self::RealField {
            self.abs()
        }
        fn scale(self, factor: Self::RealField) -> Self {
            self.0.mul(factor.0).into()
        }
        fn unscale(self, factor: Self::RealField) -> Self {
            self.0.div(factor.0).into()
        }
        fn floor(self) -> Self {
            todo!()
        }
        fn ceil(self) -> Self {
            todo!()
        }
        fn round(self) -> Self {
            todo!()
        }
        fn trunc(self) -> Self {
            todo!()
        }
        fn fract(self) -> Self {
            todo!()
        }
        fn mul_add(self, _a: Self, _b: Self) -> Self {
            todo!()
        }
        fn abs(self) -> Self::RealField {
            self.0.abs().into()
        }
        fn hypot(self, other: Self) -> Self::RealField {
            self.0.hypot(&other.0).into()
        }
        fn recip(self) -> Self {
            todo!()
        }
        fn conjugate(self) -> Self {
            self
        }
        fn sin(self) -> Self {
            todo!()
        }
        fn cos(self) -> Self {
            todo!()
        }
        fn sin_cos(self) -> (Self, Self) {
            todo!()
        }
        fn tan(self) -> Self {
            todo!()
        }
        fn asin(self) -> Self {
            todo!()
        }
        fn acos(self) -> Self {
            todo!()
        }
        fn atan(self) -> Self {
            todo!()
        }
        fn sinh(self) -> Self {
            todo!()
        }
        fn cosh(self) -> Self {
            todo!()
        }
        fn tanh(self) -> Self {
            todo!()
        }
        fn asinh(self) -> Self {
            todo!()
        }
        fn acosh(self) -> Self {
            todo!()
        }
        fn atanh(self) -> Self {
            todo!()
        }
        fn log(self, _base: Self::RealField) -> Self {
            todo!()
        }
        fn log2(self) -> Self {
            todo!()
        }
        fn log10(self) -> Self {
            todo!()
        }
        fn ln(self) -> Self {
            todo!()
        }
        fn ln_1p(self) -> Self {
            todo!()
        }
        fn sqrt(self) -> Self {
            self.0.sqrt().into()
        }
        fn exp(self) -> Self {
            todo!()
        }
        fn exp2(self) -> Self {
            todo!()
        }
        fn exp_m1(self) -> Self {
            todo!()
        }
        fn powi(self, _n: i32) -> Self {
            todo!()
        }
        fn powf(self, _n: Self::RealField) -> Self {
            todo!()
        }
        fn powc(self, _n: Self) -> Self {
            todo!()
        }
        fn cbrt(self) -> Self {
            todo!()
        }
        fn try_sqrt(self) -> Option<Self> {
            todo!()
        }
        fn is_finite(&self) -> bool {
            self.0.is_finite()
        }
    }
    impl Deref for FloatWrapper {
        type Target = rug::Float;
        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }
}

/// [Ordinary least squares](https://en.wikipedia.org/wiki/Ordinary_least_squares) implementation.
///
/// # Implementation details
///
/// This implementation uses linear algebra (namely matrix multiplication, transposed matrices &
/// the inverse).
/// For now, I'm not educated enough to understand how to derive it.
/// I've linked great resources below.
///
/// The implementation in code should be relatively simple to follow.
///
/// [Linear regression](https://towardsdatascience.com/implementing-linear-and-polynomial-regression-from-scratch-f1e3d422e6b4)
/// [How the linear algebra works](https://medium.com/@andrew.chamberlain/the-linear-algebra-view-of-least-squares-regression-f67044b7f39b)
pub mod ols {
    use super::*;

    pub struct LinearOls;
    impl LinearEstimator for LinearOls {
        fn model(&self, predictors: &[f64], outcomes: &[f64]) -> LinearCoefficients {
            let coefficients = polynomial(
                predictors.iter().copied(),
                outcomes.iter().copied(),
                predictors.len(),
                1,
            );
            LinearCoefficients {
                k: coefficients[1],
                m: coefficients[0],
            }
        }
    }

    /// The length of the inner vector is `order + 1`.
    ///
    /// The inner list is in order of smallest exponent to largest: `[0, 2, 1]` means `y = 1x² + 2x + 0`.
    #[derive(Debug)]
    pub struct PolynomialCoefficients {
        coefficients: Vec<f64>,
    }
    impl Deref for PolynomialCoefficients {
        type Target = [f64];
        fn deref(&self) -> &Self::Target {
            &self.coefficients
        }
    }
    impl Display for PolynomialCoefficients {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let mut first = true;
            for (order, mut coefficient) in self.coefficients.iter().copied().enumerate().rev() {
                if !first {
                    if coefficient.is_sign_positive() {
                        write!(f, " + ")?;
                    } else {
                        write!(f, " - ")?;
                        coefficient = -coefficient;
                    }
                }

                let p = f.precision().unwrap_or(5);

                match order {
                    0 => write!(f, "{coefficient:.*}", p)?,
                    1 => write!(f, "{coefficient:.*}x", p)?,
                    _ => write!(f, "{coefficient:.0$}x^{order:.0$}", p)?,
                }

                first = false;
            }
            Ok(())
        }
    }

    impl Predictive for PolynomialCoefficients {
        #[cfg(feature = "arbitrary-precision")]
        fn predict_outcome(&self, predictor: f64) -> f64 {
            use rug::ops::PowAssign;
            use rug::Assign;
            use std::ops::MulAssign;

            let precision = (64 + self.len() * 2) as u32;
            // let precision = arbitrary_linear_algebra::HARDCODED_PRECISION;
            let mut out = rug::Float::with_val(precision, 0.0f64);
            let original_predictor = predictor;
            let mut predictor = rug::Float::with_val(precision, predictor);
            for (order, coefficient) in self.coefficients.iter().copied().enumerate() {
                // assign to never create a new value.
                predictor.pow_assign(order as u32);
                predictor.mul_assign(coefficient);
                out += &predictor;
                predictor.assign(original_predictor)
            }
            out.to_f64()
        }
        #[cfg(not(feature = "arbitrary-precision"))]
        fn predict_outcome(&self, predictor: f64) -> f64 {
            let mut out = 0.0;
            for (order, coefficient) in self.coefficients.iter().copied().enumerate() {
                out += predictor.powi(order as i32) * coefficient;
            }
            out
        }
    }

    /// # Panics
    ///
    /// Panics if either `x` or `y` don't have the length `len`.
    ///
    /// Also panics if `order + 1 > len`.
    pub fn polynomial(
        x: impl Iterator<Item = f64> + Clone,
        y: impl Iterator<Item = f64>,
        len: usize,
        order: usize,
    ) -> PolynomialCoefficients {
        fn polynomial_simple(
            x: impl Iterator<Item = f64> + Clone,
            y: impl Iterator<Item = f64>,
            len: usize,
            order: usize,
        ) -> PolynomialCoefficients {
            let x_original = x.clone();
            let mut x_iter = x;

            let design = nalgebra::DMatrix::from_fn(len, order + 1, |row: usize, column: usize| {
                if column == 0 {
                    1.0
                } else if column == 1 {
                    x_iter.next().unwrap()
                } else {
                    if row == 0 {
                        x_iter = x_original.clone();
                    }
                    x_iter.next().unwrap().powi(column as _)
                }
            });

            let t = design.transpose();
            let y = nalgebra::DMatrix::from_iterator(len, 1, y);
            let result = ((&t * &design).try_inverse().unwrap() * &t) * y;

            PolynomialCoefficients {
                coefficients: result.iter().copied().collect(),
            }
        }
        #[cfg(feature = "arbitrary-precision")]
        fn polynomial_arbitrary(
            x: impl Iterator<Item = f64> + Clone,
            y: impl Iterator<Item = f64>,
            len: usize,
            order: usize,
        ) -> PolynomialCoefficients {
            use rug::ops::PowAssign;
            let precision = (64 + order * 2) as u32;
            // let precision = arbitrary_linear_algebra::HARDCODED_PRECISION;
            // let zero_limit = rug::Float::with_val(arbitrary_linear_algebra::HARDCODED_PRECISION, 1e-17f64).into();
            println!("Precision {:?}", precision);
            let x = x.map(|x| {
                arbitrary_linear_algebra::FloatWrapper::from(rug::Float::with_val(precision, x))
            });
            let y = y.map(|y| {
                arbitrary_linear_algebra::FloatWrapper::from(rug::Float::with_val(precision, y))
            });

            let x_original = x.clone();
            let mut x_iter = x;

            let design = nalgebra::DMatrix::from_fn(len, order + 1, |row: usize, column: usize| {
                if column == 0 {
                    rug::Float::with_val(precision, 1.0_f64).into()
                } else if column == 1 {
                    x_iter.next().unwrap()
                } else {
                    if row == 0 {
                        x_iter = x_original.clone();
                    }
                    let mut f = x_iter.next().unwrap();
                    f.0.pow_assign(column as u32);
                    f
                }
            });

            let t = design.transpose();
            let y = nalgebra::DMatrix::from_iterator(len, 1, y);
            let result = ((&t * &design).try_inverse().unwrap() * &t) * y;

            PolynomialCoefficients {
                coefficients: result.iter().map(|f| f.0.to_f64()).collect(),
            }
        }

        debug_assert!(order < len, "order + 1 must be less than or equal to len");

        #[cfg(feature = "arbitrary-precision")]
        if order < 10 {
            polynomial_simple(x, y, len, order)
        } else {
            polynomial_arbitrary(x, y, len, order)
        }
        #[cfg(not(feature = "arbitrary-precision"))]
        polynomial_simple(x, y, len, order)
    }
}
