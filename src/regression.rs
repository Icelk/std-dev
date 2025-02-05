//! Various regression models to fit the best line to your data.
//! All written to be understandable.
//!
//! Vocabulary:
//!
//! - Predictors - the independent values (usually denoted `x`) from which we want a equation to get the:
//! - outcomes - the dependant variables. Usually `y` or `f(x)`.
//! - model - create an equation which optimally (can optimize for different priorities) fits the data.
//!
//! The `*Coefficients` structs implement [`Predictive`] which calculates the [predicted outcomes](Predictive::predict_outcome)
//! using the model and their [determination](Determination::determination); and [`Display`] which can be used to
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
//! See [`derived`] for the implementations.
//!
//! I reverse the exponentiation to get a linear model. Then, I solve it using the method linked
//! above. Then, I transform the returned variables to fit the target model.
//!
//! This is not very good, as the errors of large values are reduced compared to small values when
//! taking the logarithm. I have plans to address this bias in the future.
//! The current behaviour is however still probably the desired behaviour, as small values are
//! often relatively important to larger.
//!
//! Many programs (including LibreOffice Calc) simply discards negative & zero values. I chose to
//! go the explicit route and add additional terms to satisfy requirements.
//! This is naturally a fallback, and should be a warning sign your data is bad.
//!
//! Under these methods the calculations are inserted, and how to handle the data.
#![deny(missing_docs)]

use std::fmt::{self, Display};
use std::ops::Deref;

#[doc(inline)]
pub use models::*;

pub use binary_search::Options as BinarySearchOptions;
#[cfg(feature = "ols")]
pub use derived::{exponential_ols, power_ols};
pub use gradient_descent::{
    ParallelOptions as GradientDescentParallelOptions,
    SimultaneousOptions as GradientDescentSimultaneousOptions,
};
#[cfg(feature = "ols")]
pub use ols::OlsEstimator;
pub use spiral::{SpiralLinear, SpiralLogisticWithCeiling};
pub use theil_sen::{LinearTheilSen, PolynomialTheilSen};

trait Model: Predictive + Display {}
impl<T: Predictive + Display> Model for T {}

/// Generic model. This enables easily handling results from several models.
pub struct DynModel {
    model: Box<dyn Model>,
}
impl DynModel {
    /// Wrap `model` in a [`Box`].
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

/// Something that can predict the outcome from a predictor.
pub trait Predictive {
    /// Calculates the predicted outcome of `predictor`.
    fn predict_outcome(&self, predictor: f64) -> f64;
    /// Put this predicative model in a box.
    /// This is useful for conditionally choosing different models.
    fn boxed(self) -> DynModel
    where
        Self: Sized + Display + 'static,
    {
        DynModel::new(self)
    }
}
impl<T: Predictive + ?Sized> Predictive for &T {
    fn predict_outcome(&self, predictor: f64) -> f64 {
        (**self).predict_outcome(predictor)
    }
}
/// Helper trait to make the [R²](Determination::determination) method take a generic iterator.
///
/// This enables [`Predictive`] to be `dyn`.
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
    fn determination(
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

        let mut diff = res / tot;

        if diff.is_nan() {
            diff = 0.
        };

        1.0 - diff
    }
    /// Convenience method for [`Determination::determination`] when using slices.
    fn determination_slice(&self, predictors: &[f64], outcomes: &[f64]) -> f64 {
        assert_eq!(
            predictors.len(),
            outcomes.len(),
            "predictors and outcomes must have the same number of items"
        );
        Determination::determination(
            self,
            predictors.iter().cloned(),
            outcomes.iter().cloned(),
            predictors.len(),
        )
    }
}
impl<T: Predictive> Determination for T {}

/// The models (functions) we can use regression to optimize for.
///
/// You can naturally implement these yourself.
pub mod models {
    use super::*;
    use std::f64::consts::E;

    pub use trig::*;

    macro_rules! estimator {
        ($(
            $(#[$docs:meta])*
            $name:ident -> $item:ty,
            $($(#[$more_docs:meta])* ($($arg:ident: $ty:ty),*),)?
            $model:ident, $box:ident
        )+) => {
            $(
            $(#[$docs])*
            pub trait $name {
                // #[doc = stringify!("Model the [`", $item, "`] from `predictors` and `outcomes`."]
                #[doc = "Model the [`"]
                #[doc = stringify!($item)]
                #[doc = "`] from `predictors` and `outcomes`."]
                $($(#[$more_docs])*)?
                ///
                /// # Panics
                ///
                /// The two slices must have the same length.
                fn $model(&self, predictors: &[f64], outcomes: &[f64], $($($arg: $ty),*)?) -> $item;
                /// Put this estimator in a box.
                /// This is useful for conditionally choosing different estimators.
                fn $box(self) -> Box<dyn $name>
                where
                    Self: Sized + 'static,
                {
                    Box::new(self)
                }
            }
            impl<T: $name + ?Sized> $name for &T {
                fn $model(&self, predictors: &[f64], outcomes: &[f64], $($($arg:$ty),*)?) -> $item {
                    (**self).$model(predictors, outcomes, $($($arg),*)?)
                }
            }
            )+
        };
    }

    /// The coefficients of a line.
    #[derive(Debug, Clone, Copy, PartialEq)]
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

    /// The length of the inner vector is `degree + 1`.
    ///
    /// The inner list is in order of smallest exponent to largest: `[0, 2, 1]` means `y = 1x² + 2x + 0`.
    #[derive(Clone, Debug)]
    pub struct PolynomialCoefficients {
        pub(crate) coefficients: Vec<f64>,
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
            for (degree, mut coefficient) in self.coefficients.iter().copied().enumerate().rev() {
                if coefficient.abs() < 1e-100 {
                    continue;
                }
                if !first {
                    if coefficient.is_sign_positive() {
                        write!(f, " + ")?;
                    } else {
                        write!(f, " - ")?;
                        coefficient = -coefficient;
                    }
                }

                let p = f.precision().unwrap_or(5);

                match degree {
                    0 => write!(f, "{coefficient:.*}", p)?,
                    1 => write!(f, "{coefficient:.*}x", p)?,
                    2..=9 => write!(f, "{coefficient:.0$}x^{degree:.0$}", p)?,
                    _ => write!(f, "{coefficient:.0$}x^{{{degree:.0$}}}", p)?,
                }

                first = false;
            }
            Ok(())
        }
    }
    impl PolynomialCoefficients {
        #[inline(always)]
        fn naive_predict(&self, predictor: f64) -> f64 {
            match self.coefficients.len() {
                0 => 0.,
                1 => self.coefficients[0],
                2 => self.coefficients[1] * predictor + self.coefficients[0],
                3 => {
                    self.coefficients[2] * predictor * predictor
                        + self.coefficients[1] * predictor
                        + self.coefficients[0]
                }
                4 => {
                    let p2 = predictor * predictor;
                    self.coefficients[3] * predictor * p2
                        + self.coefficients[2] * p2
                        + self.coefficients[1] * predictor
                        + self.coefficients[0]
                }
                _ => {
                    let mut out = 0.0;
                    let mut pred = 1.;
                    for coefficient in self.coefficients.iter().copied() {
                        out += pred * coefficient;
                        pred *= predictor;
                    }
                    out
                }
            }
        }

        /// Returns the coefficients for the derivative of these coefficients.
        pub fn derivative(&self) -> Self {
            let mut coeffs = Vec::with_capacity(self.len().saturating_sub(1));
            for (idx, coeff) in self.coefficients.iter().enumerate().skip(1) {
                coeffs.push(*coeff * (idx) as f64);
            }
            Self {
                coefficients: coeffs,
            }
        }
        /// Returns the coefficients for the integral (primitive function) of these coefficients.
        pub fn integral(&self) -> Self {
            let mut coeffs = Vec::with_capacity(self.len() + 1);
            coeffs.push(0.);
            for (idx, coeff) in self.coefficients.iter().enumerate() {
                coeffs.push(*coeff / (idx + 1) as f64);
            }
            Self {
                coefficients: coeffs,
            }
        }
    }
    impl Predictive for PolynomialCoefficients {
        #[cfg(feature = "arbitrary-precision")]
        fn predict_outcome(&self, predictor: f64) -> f64 {
            if self.coefficients.len() < 10 {
                self.naive_predict(predictor)
            } else {
                use rug::ops::PowAssign;
                use rug::Assign;
                use std::ops::MulAssign;

                let precision = (64 + self.len() * 2) as u32;
                // let precision = arbitrary_linear_algebra::HARDCODED_PRECISION;
                let mut out = rug::Float::with_val(precision, 0.0f64);
                let original_predictor = predictor;
                let mut predictor = rug::Float::with_val(precision, predictor);
                for (degree, coefficient) in self.coefficients.iter().copied().enumerate() {
                    // assign to never create a new value.
                    predictor.pow_assign(degree as u32);
                    predictor.mul_assign(coefficient);
                    out += &predictor;
                    predictor.assign(original_predictor)
                }
                out.to_f64()
            }
        }
        #[cfg(not(feature = "arbitrary-precision"))]
        #[inline(always)]
        fn predict_outcome(&self, predictor: f64) -> f64 {
            self.naive_predict(predictor)
        }
    }
    /// The coefficients of a power (also called growth) function (`kx^e`).
    #[derive(Debug, Clone, PartialEq)]
    pub struct PowerCoefficients {
        /// Constant
        pub k: f64,
        /// exponent
        pub e: f64,
        /// If the predictors needs to have an offset applied to remove values under 1.
        ///
        /// Defaults to 0.
        pub predictor_additive: f64,
        /// If the outcomes needs to have an offset applied to remove values under 1.
        ///
        /// Defaults to 0.
        pub outcome_additive: f64,
    }
    impl Predictive for PowerCoefficients {
        fn predict_outcome(&self, predictor: f64) -> f64 {
            self.k * (predictor + self.predictor_additive).powf(self.e) - self.outcome_additive
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
                if self.outcome_additive != 0. {
                    format!(" - {:.1$}", self.outcome_additive, p)
                } else {
                    String::new()
                },
                p,
                x = if self.predictor_additive != 0. {
                    format!("(x + {:.1$})", self.predictor_additive, p)
                } else {
                    "x".to_string()
                },
            )
        }
    }
    impl From<LinearCoefficients> for PolynomialCoefficients {
        fn from(coefficients: LinearCoefficients) -> Self {
            Self {
                coefficients: vec![coefficients.m, coefficients.k],
            }
        }
    }
    impl<T: Into<Vec<f64>>> From<T> for PolynomialCoefficients {
        fn from(t: T) -> Self {
            Self {
                coefficients: t.into(),
            }
        }
    }

    /// The coefficients of a exponential function (`kb^x`).
    #[derive(Debug, Clone, PartialEq)]
    pub struct ExponentialCoefficients {
        /// Constant
        pub k: f64,
        /// base
        pub b: f64,
        /// If the predictors needs to have an offset applied to remove values under 1.
        ///
        /// Defaults to 0.
        pub predictor_additive: f64,
        /// If the outcomes needs to have an offset applied to remove values under 1.
        ///
        /// Defaults to 0.
        pub outcome_additive: f64,
    }
    impl Predictive for ExponentialCoefficients {
        fn predict_outcome(&self, predictor: f64) -> f64 {
            self.k * self.b.powf(predictor + self.predictor_additive) - self.outcome_additive
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
                if self.outcome_additive != 0. {
                    format!(" - {:.1$}", self.outcome_additive, p)
                } else {
                    String::new()
                },
                p,
                x = if self.predictor_additive != 0. {
                    format!("(x + {:.1$})", self.predictor_additive, p)
                } else {
                    "x".to_string()
                },
            )
        }
    }

    /// The coefficients of a [logistic function](https://en.wikipedia.org/wiki/Logistic_function).
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct LogisticCoefficients {
        /// The x value of the curve's midpoint
        pub x0: f64,
        /// The curve's maximum value
        pub l: f64,
        /// The logistic growth rate or steepness of the curve
        pub k: f64,
    }
    impl Predictive for LogisticCoefficients {
        #[inline(always)]
        fn predict_outcome(&self, predictor: f64) -> f64 {
            self.l / (1. + E.powf(-self.k * (predictor - self.x0)))
        }
    }
    impl Display for LogisticCoefficients {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let p = f.precision().unwrap_or(5);
            write!(
                f,
                "{:.3$} / (1 + e^({}(x {})))",
                self.l,
                if self.k.is_sign_negative() {
                    format!("{:.1$}", -self.k, p)
                } else {
                    format!("-{:.1$}", self.k, p)
                },
                if self.x0.is_sign_negative() {
                    format!("+ {:.1$}", -self.x0, p)
                } else {
                    format!("- {:.1$}", self.x0, p)
                },
                p
            )
        }
    }

    estimator!(
        /// Implemented by all estimators yielding a linear 2 variable regression (a line).
        LinearEstimator -> LinearCoefficients, model_linear, boxed_linear

        /// Implemented by all estimators yielding a polynomial regression.
        PolynomialEstimator -> PolynomialCoefficients,
        /// Also takes a `degree` of the target polynomial. Some estimators may panic when `degree`
        /// is out of their range.
        (degree: usize), model_polynomial, boxed_polynomial

        /// Implemented by all estimators yielding a power regression.
        PowerEstimator -> PowerCoefficients, model_power, boxed_power

        /// Implemented by all estimators yielding an exponential regression.
        ExponentialEstimator -> ExponentialCoefficients, model_exponential, boxed_exponential

        /// Implemented by all estimators yielding an logistic regression.
        LogisticEstimator -> LogisticCoefficients, model_logistic, boxed_logistic
    );

    /// Traits and coefficients of trigonometric functions.
    pub mod trig {
        use super::*;

        macro_rules! simple_coefficients {
            ($(
                $(#[$docs:meta])+
                $name:ident, f64::$fn:ident
            )+) => {
                simple_coefficients!($($(#[$docs])* $name, v f64::$fn(v), stringify!($fn))+);
            };
            ($(
                $(#[$docs:meta])+
                $name:ident, 1 / f64::$fn:ident, $disp:expr
            )+) => {
                simple_coefficients!($($(#[$docs])* $name, v 1.0/f64::$fn(v), $disp)+);
            };
            ($(
                $(#[$docs:meta])+
                $name:ident, $v:ident $fn:expr, $disp:expr
            )+) => {
                $(
                $(#[$docs])+
                #[derive(PartialEq, Clone, Debug)]
                pub struct $name {
                    /// The amplitude of this function.
                    pub amplitude: f64,
                    /// The frequency of this function.
                    pub frequency: f64,
                    /// The phase of this function (x offset).
                    pub phase: f64,
                }
                impl Predictive for $name {
                    fn predict_outcome(&self, predictor: f64) -> f64 {
                        let $v = predictor * self.frequency + self.phase;
                        self.amplitude * $fn
                    }
                }
                impl Display for $name {
                    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                        let p = f.precision().unwrap_or(5);
                        write!(
                            f,
                            "{:.4$}{}({:.4$}x+{:.4$})",
                            self.amplitude,
                            $disp,
                            self.frequency,
                            self.phase,
                            p,
                        )
                    }
                }
                impl $name {
                    #[inline(always)]
                    pub(crate) fn wrap(array: [f64; 3]) -> Self {
                        Self {
                            amplitude: array[0],
                            frequency: array[1],
                            phase: array[2] % (std::f64::consts::PI * 2.),
                        }
                    }
                }
                )+
            };
        }
        simple_coefficients!(
            /// The coefficients of a sine wave.
            SineCoefficients, f64::sin
            /// The coefficients of a cosine wave.
            CosineCoefficients, f64::cos
            /// The coefficients of a tangent function.
            TangentCoefficients, f64::tan
        );
        simple_coefficients!(
            /// The coefficients of a secant function.
            SecantCoefficients,
            1 / f64::sin, "sec"
            /// The coefficients of a cosecant function.
            CosecantCoefficients,
            1 / f64::cos, "csc"
            /// The coefficients of a cotangent function.
            CotangentCoefficients,
            1 / f64::tan, "cot"
        );

        estimator!(
            /// Implemented by all estimators yielding a sine wave.
            SineEstimator -> SineCoefficients, (max_frequency: f64), model_sine, boxed_sine
            /// Implemented by all estimators yielding a cosine wave.
            CosineEstimator -> CosineCoefficients, (max_frequency: f64), model_cosine, boxed_cosine
            /// Implemented by all estimators yielding a tangent function.
            TangentEstimator -> TangentCoefficients, (max_frequency: f64), model_tangent, boxed_tangent

            /// Implemented by all estimators yielding a secant function.
            SecantEstimator -> SecantCoefficients, (max_frequency: f64), model_secant, boxed_sesecant
            /// Implemented by all estimators yielding a cosecant function.
            CosecantEstimator -> CosecantCoefficients, (max_frequency: f64), model_cosecant, boxed_cosecant
            /// Implemented by all estimators yielding a cotangent function.
            CotangentEstimator -> CotangentCoefficients, (max_frequency: f64), model_cotangent, boxed_cotangent
        );
    }
}

/// Finds the model best fit to the input data.
/// This is done using heuristics and testing of methods.
///
/// # Panics
///
/// Panics if the model has less than two parameters or if the two slices have different lengths.
///
/// # Heuristics
///
/// These seemed good to me. Any ideas on improving them are welcome.
///
/// - Power and exponentials only if no data is < 1.
///   This is due to the sub-optimal behaviour of logarithm with values close to and under 0.
///   This restriction might be lifted to just < 1e-9 in the future.
/// - Power is heavily favoured if `let distance_from_integer = -(0.5 - exponent % 1).abs() + 0.5;
///   distance_from_integer < 0.15 && -2.5 <= exponent <= 3.5`
/// - Power is also heavily favoured if the same as above occurs but with the reciprocal of the
///   exponent. Then, the range 0.5 < exponent.recip() <= 3.5 is considered.
/// - Exponential favoured if R² > 0.8, which seldom happens with exponential regression.
/// - Bump the rating of linear, as that's probably what you want.
/// - 2'nd degree polynomial is only considered if `n > 15`, where `n` is `predictors.len()`.
/// - 3'nd degree polynomial is only considered if `n > 50`
pub fn best_fit(
    predictors: &[f64],
    outcomes: &[f64],
    linear_estimator: &impl LinearEstimator,
) -> DynModel {
    // These values are chosen from heuristics in my brain
    /// Additive
    const LINEAR_BUMP: f64 = 0.0;
    /// Multiplicative
    const POWER_BUMP: f64 = 1.5;
    /// Multiplicative
    const EXPONENTIAL_BUMP: f64 = 1.3;
    /// Used to partially mitigate [overfitting](https://en.wikipedia.org/wiki/Overfitting).
    ///
    /// Multiplicative
    // `TODO`: remove when we use generic polynomial provider
    #[allow(unused)]
    const SECOND_DEGREE_DISADVANTAGE: f64 = 0.94;
    /// Used to partially mitigate [overfitting](https://en.wikipedia.org/wiki/Overfitting).
    ///
    /// Multiplicative
    // `TODO`: remove when we use generic polynomial provider
    #[allow(unused)]
    const THIRD_DEGREE_DISADVANTAGE: f64 = 0.9;

    let mut best: Option<(DynModel, f64)> = None;
    macro_rules! update_best {
        ($new: expr, $e: ident, $modificator: expr, $err: expr) => {
            let $e = $err;
            let weighted = $modificator;
            if let Some((_, error)) = &best {
                if weighted > *error {
                    best = Some((DynModel::new($new), weighted))
                }
            } else {
                best = Some((DynModel::new($new), weighted))
            }
        };
        ($new: expr, $e: ident, $modificator: expr) => {
            update_best!(
                $new,
                $e,
                $modificator,
                $new.determination_slice(predictors, outcomes)
            )
        };
        ($new: expr) => {
            update_best!($new, e, e)
        };
    }

    let predictor_min = derived::min(predictors).unwrap();
    let outcomes_min = derived::min(outcomes).unwrap();

    if predictor_min >= 1.0 && outcomes_min >= 1.0 {
        let mut mod_predictors = predictors.to_vec();
        let mut mod_outcomes = outcomes.to_vec();
        let power = derived::power_given_min(
            &mut mod_predictors,
            &mut mod_outcomes,
            predictor_min,
            outcomes_min,
            linear_estimator,
        );

        let distance_from_integer = -(0.5 - power.e % 1.0).abs() + 0.5;
        let mut power_bump = 1.0;
        if distance_from_integer < 0.15 && power.e <= 3.5 && power.e >= -2.5 {
            power_bump *= POWER_BUMP;
        }
        let distance_from_fraction = -(0.5 - power.e.recip() % 1.0).abs() + 0.5;
        if distance_from_fraction < 0.1 && power.e.recip() <= 3.5 && power.e.recip() > 0.5 {
            power_bump *= POWER_BUMP;
        }
        let certainty = power.determination_slice(predictors, outcomes);
        if certainty > 0.8 {
            power_bump *= EXPONENTIAL_BUMP;
        }
        if certainty > 0.92 {
            power_bump *= EXPONENTIAL_BUMP;
        }

        update_best!(power, e, e * power_bump, certainty);

        mod_predictors[..].copy_from_slice(predictors);
        mod_outcomes[..].copy_from_slice(outcomes);

        let exponential = derived::exponential_given_min(
            &mut mod_predictors,
            &mut mod_outcomes,
            predictor_min,
            outcomes_min,
            linear_estimator,
        );
        let certainty = exponential.determination_slice(predictors, outcomes);

        let mut exponential_bump = if certainty > 0.8 {
            EXPONENTIAL_BUMP
        } else {
            1.0
        };
        if certainty > 0.92 {
            exponential_bump *= EXPONENTIAL_BUMP;
        }

        update_best!(exponential, e, e * exponential_bump, certainty);
    }
    // `TODO`: use generic polynomial provider.
    #[cfg(feature = "ols")]
    if predictors.len() > 15 {
        let degree_2 = ols::polynomial(
            predictors.iter().copied(),
            outcomes.iter().copied(),
            predictors.len(),
            2,
        );

        update_best!(degree_2, e, e * SECOND_DEGREE_DISADVANTAGE);
    }
    #[cfg(feature = "ols")]
    if predictors.len() > 50 {
        let degree_3 = ols::polynomial(
            predictors.iter().copied(),
            outcomes.iter().copied(),
            predictors.len(),
            3,
        );

        update_best!(degree_3, e, e * THIRD_DEGREE_DISADVANTAGE);
    }

    let linear = linear_estimator.model_linear(predictors, outcomes);
    update_best!(linear, e, e + LINEAR_BUMP);
    // UNWRAP: We just set it, at least there's a linear.
    best.unwrap().0
}
/// Convenience function for [`best_fit`] using [`OlsEstimator`].
#[cfg(feature = "ols")]
pub fn best_fit_ols(predictors: &[f64], outcomes: &[f64]) -> DynModel {
    best_fit(predictors, outcomes, &OlsEstimator)
}

/// Estimators derived from others, usual [`LinearEstimator`].
///
/// These do not (for now) implement [`PowerEstimator`] nor [`ExponentialEstimator`]
/// because of the requirement of mutable slices instead of immutable ones.
///
/// See the docs on the items for more info about how they're created.
pub mod derived {
    use super::*;
    pub(super) fn min(slice: &[f64]) -> Option<f64> {
        slice
            .iter()
            .copied()
            .map(crate::F64OrdHash)
            .min()
            .map(|f| f.0)
    }

    /// Convenience-method for [`power`] using [`OlsEstimator`].
    #[cfg(feature = "ols")]
    pub fn power_ols(predictors: &mut [f64], outcomes: &mut [f64]) -> PowerCoefficients {
        power(predictors, outcomes, &OlsEstimator)
    }
    /// Fits a curve with the equation `y = a * x^b` (optionally with an additional subtractive term if
    /// any outcome is < 1 and an additive to the `x` if any predictor is < 1).
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

        let coefficients = estimator.model_linear(predictors, outcomes);
        let k = 2.0_f64.powf(coefficients.m);
        let e = coefficients.k;
        PowerCoefficients {
            k,
            e,
            predictor_additive: predictor_additive.unwrap_or(0.),
            outcome_additive: outcome_additive.unwrap_or(0.),
        }
    }

    /// Convenience-method for [`exponential`] using [`OlsEstimator`].
    #[cfg(feature = "ols")]
    pub fn exponential_ols(
        predictors: &mut [f64],
        outcomes: &mut [f64],
    ) -> ExponentialCoefficients {
        exponential(predictors, outcomes, &OlsEstimator)
    }
    /// Fits a curve with the equation `y = a * b^x` (optionally with an additional subtractive term if
    /// any outcome is < 1 and an additive to the `x` if any predictor is < 1).
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

        let coefficients = estimator.model_linear(predictors, outcomes);
        let k = 2.0_f64.powf(coefficients.m);
        let b = 2.0_f64.powf(coefficients.k);
        ExponentialCoefficients {
            k,
            b,
            predictor_additive: predictor_additive.unwrap_or(0.),
            outcome_additive: outcome_additive.unwrap_or(0.),
        }
    }
}

/// This module enables the use of [`rug::Float`] inside of [`nalgebra`].
///
/// Many functions are not implemented. PRs are welcome.
#[cfg(feature = "arbitrary-precision")]
pub mod arbitrary_linear_algebra {
    use std::cell::RefCell;
    use std::fmt::{self, Display};
    use std::ops::{
        Add, AddAssign, Deref, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
    };

    use nalgebra::{ComplexField, RealField};
    use rug::Assign;

    thread_local! {
        /// The default precision.
        ///
        /// This is thread-local.
        pub static DEFAULT_PRECISION: RefCell<u32> = const { RefCell::new(256) };
    }
    /// Set the default precision **for this thread**.
    pub fn set_default_precision(new: u32) {
        DEFAULT_PRECISION.with(|v| *v.borrow_mut() = new);
    }
    /// Get the default precision.
    /// Can be set using [`set_default_precision`].
    pub fn default_precision() -> u32 {
        DEFAULT_PRECISION.with(|v| *v.borrow())
    }
    /// A wrapper around [`rug::Float`] to implement traits for.
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
            rug::Float::with_val(default_precision(), element).into()
        }
    }
    impl simba::scalar::SupersetOf<f32> for FloatWrapper {
        fn is_in_subset(&self) -> bool {
            self.0.prec() <= 24
        }
        fn to_subset(&self) -> Option<f32> {
            if simba::scalar::SupersetOf::<f32>::is_in_subset(self) {
                Some(self.0.to_f32())
            } else {
                None
            }
        }
        fn to_subset_unchecked(&self) -> f32 {
            self.0.to_f32()
        }
        fn from_subset(element: &f32) -> Self {
            rug::Float::with_val(default_precision(), element).into()
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
            Some(rug::Float::with_val(default_precision(), n).into())
        }
        fn from_u64(n: u64) -> Option<Self> {
            Some(rug::Float::with_val(default_precision(), n).into())
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

        const LANES: usize = 1;

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
            Self(rug::Float::with_val(default_precision(), 0.0))
        }
        fn is_zero(&self) -> bool {
            self.0 == 0.0
        }
    }
    impl num_traits::One for FloatWrapper {
        fn one() -> Self {
            Self(rug::Float::with_val(default_precision(), 1.0))
        }
    }
    impl num_traits::Num for FloatWrapper {
        type FromStrRadixErr = rug::float::ParseFloatError;
        fn from_str_radix(s: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
            rug::Float::parse_radix(s, radix as i32)
                .map(|f| Self(rug::Float::with_val(default_precision(), f)))
        }
    }
    impl num_traits::Signed for FloatWrapper {
        fn abs(&self) -> Self {
            (*self.0.as_abs()).clone().into()
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
            rug::Float::with_val(default_precision(), f64::EPSILON).into()
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
            rug::Float::with_val(default_precision(), f64::EPSILON).into()
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
#[cfg(feature = "ols")]
pub mod ols {
    use std::cell::RefCell;

    use nalgebra::DMatrix;

    use super::*;

    #[must_use]
    struct RuntimeMatrices {
        design: DMatrix<f64>,
        transposed: DMatrix<f64>,
        outcomes: DMatrix<f64>,
        intermediary1: DMatrix<f64>,
        intermediary2: DMatrix<f64>,
        result: DMatrix<f64>,

        len: usize,
        degree: usize,
    }
    impl RuntimeMatrices {
        fn new() -> Self {
            Self {
                design: DMatrix::zeros(0, 0),
                transposed: DMatrix::zeros(0, 0),
                outcomes: DMatrix::zeros(0, 0),
                intermediary1: DMatrix::zeros(0, 0),
                intermediary2: DMatrix::zeros(0, 0),
                result: DMatrix::zeros(0, 0),

                len: 0,
                degree: 0,
            }
        }
        /// No guarantees are made to the content of the matrix.
        #[inline]
        fn resize(&mut self, len: usize, degree: usize) {
            if self.len == len && self.degree == degree {
                return;
            }
            let rows = len;
            let columns = degree + 1;
            self.design.resize_mut(rows, columns, 0.);
            self.transposed.resize_mut(columns, rows, 0.);
            self.outcomes.resize_mut(rows, 1, 0.);
            self.intermediary1.resize_mut(columns, columns, 0.);
            self.intermediary2.resize_mut(rows, columns, 0.);
            self.result.resize_mut(columns, 1, 0.);
            self.len = len;
            self.degree = degree;
        }
    }
    thread_local! {static RUNTIME: RefCell<RuntimeMatrices> = RefCell::new(RuntimeMatrices::new());}

    /// Linear: `O(n)`
    /// Polynomial: `O(n*degree)`, which when using a set `degree` becomes `O(n)`
    pub struct OlsEstimator;
    impl LinearEstimator for OlsEstimator {
        fn model_linear(&self, predictors: &[f64], outcomes: &[f64]) -> LinearCoefficients {
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
    impl PolynomialEstimator for OlsEstimator {
        fn model_polynomial(
            &self,
            predictors: &[f64],
            outcomes: &[f64],
            degree: usize,
        ) -> PolynomialCoefficients {
            assert_eq!(predictors.len(), outcomes.len());
            polynomial(
                predictors.iter().copied(),
                outcomes.iter().copied(),
                predictors.len(),
                degree,
            )
        }
    }

    /// # Panics
    ///
    /// Panics if either `x` or `y` don't have the length `len`.
    ///
    /// Also panics if `degree + 1 > len`.
    #[inline(always)]
    pub fn polynomial(
        predictors: impl Iterator<Item = f64> + Clone,
        outcomes: impl Iterator<Item = f64>,
        len: usize,
        degree: usize,
    ) -> PolynomialCoefficients {
        // this is the same as [`polynomial_simple_preallocated`], but clearer for the reader
        #[allow(unused)]
        fn polynomial_simple(
            predictors: impl Iterator<Item = f64> + Clone,
            outcomes: impl Iterator<Item = f64>,
            len: usize,
            degree: usize,
        ) -> PolynomialCoefficients {
            let predictor_original = predictors.clone();
            let mut predictor_iter = predictors;

            let design =
                nalgebra::DMatrix::from_fn(len, degree + 1, |row: usize, column: usize| {
                    if column == 0 {
                        1.0
                    } else if column == 1 {
                        predictor_iter.next().unwrap()
                    } else {
                        if row == 0 {
                            predictor_iter = predictor_original.clone();
                        }
                        predictor_iter.next().unwrap().powi(column as _)
                    }
                });

            let t = design.transpose();
            let outcomes = nalgebra::DMatrix::from_iterator(len, 1, outcomes);
            let result = ((&t * &design)
                .try_inverse()
                .unwrap_or_else(|| (&t * &design).pseudo_inverse(0e-6).unwrap())
                * &t)
                * outcomes;

            PolynomialCoefficients {
                coefficients: result.iter().copied().collect(),
            }
        }
        // like [`polynomial_simple`], but with persistent allocations.
        fn polynomial_simple_preallocated(
            predictors: impl Iterator<Item = f64> + Clone,
            outcomes: impl Iterator<Item = f64>,
            len: usize,
            degree: usize,
        ) -> PolynomialCoefficients {
            RUNTIME.with(move |runtime| {
                let mut runtime = runtime.borrow_mut();
                // cheap clone call, it's an iterator
                let predictor_original = predictors.clone();
                let mut predictor_iter = predictors;

                runtime.resize(len, degree);

                let RuntimeMatrices {
                    design,
                    transposed,
                    outcomes: outcomes_matrix,
                    intermediary1,
                    intermediary2,
                    result,
                    ..
                } = &mut *runtime;

                {
                    let (rows, columns) = design.shape();
                    for column in 0..columns {
                        for row in 0..rows {
                            let v = unsafe { design.get_unchecked_mut((row, column)) };
                            *v = if column == 0 {
                                1.0
                            } else if column == 1 {
                                predictor_iter.next().unwrap()
                            } else {
                                if row == 0 {
                                    predictor_iter = predictor_original.clone();
                                }
                                predictor_iter.next().unwrap().powi(column as _)
                            };
                        }
                    }
                }
                design.transpose_to(transposed);

                {
                    let rows = outcomes_matrix.nrows();
                    for (row, outcome) in (0..rows).zip(outcomes) {
                        let v = unsafe { outcomes_matrix.get_unchecked_mut((row, 0)) };
                        *v = outcome;
                    }
                }

                transposed.mul_to(design, intermediary1);

                if !intermediary1.try_inverse_mut() {
                    let im = std::mem::replace(intermediary1, DMatrix::zeros(0, 0));
                    let pseudo_inverse = im.pseudo_inverse(1e-8).unwrap();
                    *intermediary1 = pseudo_inverse;
                }
                *intermediary2 = &*intermediary1 * &*transposed;
                intermediary2.mul_to(outcomes_matrix, result);

                PolynomialCoefficients {
                    coefficients: runtime.result.iter().copied().collect(),
                }
            })
        }
        #[cfg(feature = "arbitrary-precision")]
        fn polynomial_arbitrary(
            predictors: impl Iterator<Item = f64> + Clone,
            outcomes: impl Iterator<Item = f64>,
            len: usize,
            degree: usize,
        ) -> PolynomialCoefficients {
            use rug::ops::PowAssign;
            let precision = (64 + degree * 2) as u32;
            let old = arbitrary_linear_algebra::default_precision();
            arbitrary_linear_algebra::set_default_precision(precision);

            let predictors = predictors.map(|x| {
                arbitrary_linear_algebra::FloatWrapper::from(rug::Float::with_val(precision, x))
            });
            let outcomes = outcomes.map(|y| {
                arbitrary_linear_algebra::FloatWrapper::from(rug::Float::with_val(precision, y))
            });

            let predictor_original = predictors.clone();
            let mut predictor_iter = predictors;

            let design =
                nalgebra::DMatrix::from_fn(len, degree + 1, |row: usize, column: usize| {
                    if column == 0 {
                        rug::Float::with_val(precision, 1.0_f64).into()
                    } else if column == 1 {
                        predictor_iter.next().unwrap()
                    } else {
                        if row == 0 {
                            predictor_iter = predictor_original.clone();
                        }
                        let mut f = predictor_iter.next().unwrap();
                        f.0.pow_assign(column as u32);
                        f
                    }
                });

            let t = design.transpose();
            let outcomes = nalgebra::DMatrix::from_iterator(len, 1, outcomes);
            let result = ((&t * &design).try_inverse().unwrap() * &t) * outcomes;

            arbitrary_linear_algebra::set_default_precision(old);

            PolynomialCoefficients {
                coefficients: result.iter().map(|f| f.0.to_f64()).collect(),
            }
        }

        debug_assert!(degree < len, "degree + 1 must be less than or equal to len");

        #[cfg(feature = "arbitrary-precision")]
        if degree < 10 {
            polynomial_simple_preallocated(predictors, outcomes, len, degree)
        } else {
            polynomial_arbitrary(predictors, outcomes, len, degree)
        }
        #[cfg(not(feature = "arbitrary-precision"))]
        polynomial_simple_preallocated(predictors, outcomes, len, degree)
    }
}

/// [Theil-Sen estimator](https://en.wikipedia.org/wiki/Theil%E2%80%93Sen_estimator), a robust
/// linear (also implemented as polynomial) estimator.
/// Up to ~27% of values can be *outliers* - erroneous data far from the otherwise good data -
/// without large effects on the result.
///
/// [`LinearTheilSen`] implements [`LinearEstimator`].
pub mod theil_sen {
    use super::*;
    use crate::{percentile, F64OrdHash};
    use std::fmt::Debug;

    /// A buffer returned by [`PermutationIter`] to avoid allocations.
    pub struct PermutationIterBuffer<T> {
        buf: Vec<(T, T)>,
    }
    impl<T> Deref for PermutationIterBuffer<T> {
        type Target = [(T, T)];
        fn deref(&self) -> &Self::Target {
            &self.buf
        }
    }
    /// An iterator over the permutations.
    #[derive(Debug)]
    pub struct PermutationIter<'a, T> {
        s1: &'a [T],
        s2: &'a [T],
        iters: Vec<usize>,
        values: Option<Vec<(T, T)>>,
        values_backup: Vec<(T, T)>,
        pairs: usize,
    }
    impl<'a, T: Copy + Debug> PermutationIter<'a, T> {
        fn new(s1: &'a [T], s2: &'a [T], pairs: usize) -> Self {
            assert!(
                pairs > 1,
                "each coordinate pair must be associated with at least one."
            );
            assert_eq!(s1.len(), s2.len());
            assert!(pairs <= s1.len());
            let iters = Vec::with_capacity(pairs);
            let values_backup = Vec::with_capacity(pairs);
            let mut me = Self {
                s1,
                s2,
                iters,
                values: None,
                values_backup,
                pairs,
            };
            for i in 0..pairs {
                // `+ (not last) as usize` since the last iterator doesn't first read vec.
                me.iters.push(i + usize::from(i + 1 < pairs) - 1);
            }
            #[allow(clippy::needless_range_loop)] // clarity
            for i in 0..pairs - 1 {
                me.values_backup.push((me.s1[i], me.s2[i]));
            }
            me.values_backup.push(me.values_backup[0]);
            me.values = Some(me.values_backup.clone());
            me
        }
        /// Please hand the buffer back after each iteration. This greatly reduces allocations.
        #[inline(always)]
        pub fn give_buffer(&mut self, buf: PermutationIterBuffer<T>) {
            debug_assert!(self.values.is_none());
            self.values = Some(buf.buf)
        }
        /// Collects the permutations in `pairs` number of [`Vec`]s, with `returned[i]` containing
        /// the i-th pair.
        pub fn collect_by_index(mut self) -> Vec<Vec<(T, T)>> {
            let mut vecs = Vec::with_capacity(self.pairs);
            for _ in 0..self.pairs {
                vecs.push(Vec::new());
            }
            while let Some(buf) = self.next() {
                for (pos, pair) in buf.iter().enumerate() {
                    vecs[pos].push(*pair)
                }

                self.give_buffer(buf);
            }
            vecs
        }
        /// Collects `LEN` pairs from this iterator in a [`Vec`].
        ///
        /// # Panics
        ///
        /// `LEN` must be the same length as `pairs` supplied to [`permutations_generic`].
        pub fn collect_len<const LEN: usize>(mut self) -> Vec<[(T, T); LEN]> {
            let mut vec = Vec::new();
            while let Some(buf) = self.next() {
                let array = <[(T, T); LEN]>::try_from(&*buf).expect(
                    "tried to collect with set len, but permutations returned different len",
                );
                vec.push(array);
                self.give_buffer(buf);
            }
            vec
        }
    }
    impl<T: Copy + Debug> Iterator for PermutationIter<'_, T> {
        type Item = PermutationIterBuffer<T>;
        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            for (num, iter) in self.iters.iter_mut().enumerate().rev() {
                *iter += 1;

                if let Some(value) = self.s1.get(*iter) {
                    // SAFETY: they are the same length, so getting from one guarantees we can get
                    // the same index from the other one.
                    let next = (*value, *unsafe { self.s2.get_unchecked(*iter) });

                    let values = &mut self.values_backup;
                    if let Some(v) = self.values.as_mut() {
                        // SAFETY: The length of `self.values`, `self.values_backup`, and
                        // `self.iters` are equal. Therefore, we can get the index of `self.values`
                        // returned by the enumeration of `self.iters`
                        *unsafe { v.get_unchecked_mut(num) } = next;
                    }
                    // SAFETY: See above.
                    *unsafe { values.get_unchecked_mut(num) } = next;
                    if num + 1 == self.pairs {
                        let values = match self.values.take() {
                            Some(x) => x,
                            None => self.values_backup.clone(),
                        };
                        return Some(PermutationIterBuffer { buf: values });
                    } else {
                        // optimization - if items left is less than what is required to fill the "tower"
                        // of succeeding indices, we return
                        if self.s1.len() - *iter <= self.pairs - 1 - num {
                            continue;
                        }

                        // Not pushing unsafe/inline as hard here, as this isn't as hot of a path.

                        #[allow(clippy::needless_range_loop)] // clarity
                        for i in num + 1..self.pairs {
                            // start is 1+ the previous?
                            let new = self.iters[i - 1] + usize::from(i + 1 < self.pairs);
                            self.iters[i] = new;
                            // fix values for lower iterators than the top one
                            if i + 1 < self.pairs {
                                if new >= self.s1.len() {
                                    continue;
                                }
                                values[i] = (self.s1[new], self.s2[new]);
                                if let Some(v) = self.values.as_mut() {
                                    v[i] = (self.s1[new], self.s2[new]);
                                }
                            }
                        }
                        return self.next();
                    }
                }
            }
            None
        }
    }
    /// The returned iterator is a bit funky.
    /// It returns a buffer, which at all costs should be reused.
    /// This could either be done using a while loop
    /// (e.g. `while let Some(buf) = iter.next() { iter.give_buffer(buf) }`)
    /// or any of the [built-in methods](PermutationIter).
    /// If you know the length at compile time, use [`PermutationIter::collect_len`].
    pub fn permutations_generic<'a, T: Copy + Debug>(
        s1: &'a [T],
        s2: &'a [T],
        pairs: usize,
    ) -> PermutationIter<'a, T> {
        PermutationIter::new(s1, s2, pairs)
    }
    /// Lower-bound estimate (up to `pairs > 20`), within 100x (which is quite good for factorials).
    ///
    /// The original equation is `elements!/(pairs! (elements - pairs)!)`
    #[inline]
    pub fn estimate_permutation_count(elements: usize, pairs: usize) -> f64 {
        let e = elements as f64;
        let p = pairs as f64;
        e.powf(p) / (p.powf(p - 0.8))
    }
    /// An exact count of permutations.
    /// Returns [`None`] if the arithmetic can't fit.
    #[inline]
    pub fn permutation_count(elements: usize, pairs: usize) -> Option<usize> {
        fn factorial(num: u128) -> Option<u128> {
            match num {
                0 | 1 => Some(1),
                _ => factorial(num - 1)?.checked_mul(num),
            }
        }

        Some(
            (factorial(elements as _)?
                / (factorial(pairs as _)?.checked_mul(factorial((elements - pairs) as _)?))?)
                as usize,
        )
    }

    /// Unique permutations of two elements - an iterator of all the pairs of associated values in the slices.
    ///
    /// This function will behave unexpectedly if `s1` and `s2` have different lengths.
    ///
    /// Returns an iterator which yields `O(n²)` items.
    // `TODO`: Make these return indices.
    pub fn permutations<'a, T: Copy>(
        s1: &'a [T],
        s2: &'a [T],
    ) -> impl Iterator<Item = ((T, T), (T, T))> + 'a {
        s1.iter()
            .zip(s2.iter())
            .enumerate()
            .flat_map(|(pos, (t11, t21))| {
                // +1 because we don't want our selves.
                let left = &s1[pos + 1..];
                let left_other = &s2[pos + 1..];
                left.iter()
                    .zip(left_other.iter())
                    .map(|(t12, t22)| ((*t11, *t21), (*t12, *t22)))
            })
    }

    /// Linear estimation using the Theil-Sen estimatior. This is robust against outliers.
    /// `O(n²)`
    pub struct LinearTheilSen;
    impl LinearEstimator for LinearTheilSen {
        #[inline]
        fn model_linear(&self, predictors: &[f64], outcomes: &[f64]) -> LinearCoefficients {
            slow_linear(predictors, outcomes)
        }
    }
    /// Polynomial estimation using the Theil-Sen estimatior. Very slow and should probably not be
    /// used.
    /// `O(n^degree)`
    pub struct PolynomialTheilSen;
    impl PolynomialEstimator for PolynomialTheilSen {
        #[inline]
        fn model_polynomial(
            &self,
            predictors: &[f64],
            outcomes: &[f64],
            degree: usize,
        ) -> PolynomialCoefficients {
            slow_polynomial(predictors, outcomes, degree)
        }
    }

    /// Naive Theil-Sen implementation, which checks each line.
    ///
    /// Time & space: O(n²)
    ///
    /// # Panics
    ///
    /// Panics if `predictors.len() != outcomes.len()`.
    pub fn slow_linear(predictors: &[f64], outcomes: &[f64]) -> LinearCoefficients {
        assert_eq!(predictors.len(), outcomes.len());
        // I've isolated the `Vec`s into blocks so we only have one at a time.
        // This reduces memory usage.
        let median_slope = {
            let slopes = permutations(predictors, outcomes).map(|((x1, y1), (x2, y2))| {
                // Δy/Δx
                (y1 - y2) / (x1 - x2)
            });
            let mut slopes: Vec<_> = slopes.map(F64OrdHash).collect();

            percentile::median(&mut slopes).resolve()
        };

        //// Old intersect code. Gets the median of all x components, and the median of all y
        //// components. We then use that as a point to extrapolate the intersection.
        //
        // let predictor_median = {
        // let mut predictors = predictors.to_vec();
        // let predictors = F64OrdHash::from_mut_f64_slice(&mut predictors);
        // percentile::median(predictors).resolve()
        // };
        // let outcome_median = {
        // let mut outcomes = outcomes.to_vec();
        // let outcomes = F64OrdHash::from_mut_f64_slice(&mut outcomes);
        // percentile::median(outcomes).resolve()
        // };
        // y=slope * x + intersect
        // y - slope * x = intersect
        // let intersect = outcome_median - median_slope * predictor_median;

        // New intersect. This works by getting the median point by it's y value. Then, we
        // extrapolate from that.
        //
        // This produces much better results, but isn't what's commonly used.
        //
        // See https://stats.stackexchange.com/a/96166
        // for reference.
        let median = {
            let mut values: Vec<_> = predictors.iter().zip(outcomes.iter()).collect();
            match percentile::percentile_default_pivot_by(
                &mut values,
                crate::Fraction::HALF,
                &mut |a, b| F64OrdHash::f64_cmp(*a.1, *b.1),
            ) {
                percentile::MeanValue::Single(v) => (*v.0, *v.1),
                percentile::MeanValue::Mean(v1, v2) => ((v1.0 + v2.0) / 2.0, (v1.1 + v2.1) / 2.0),
            }
        };
        let intersect = median.1 - median.0 * median_slope;

        LinearCoefficients {
            k: median_slope,
            m: intersect,
        }
    }

    /// Naive Theil-Sen implementation, which checks each polynomial.
    ///
    /// Time & space: O(n^m) where m is `degree + 1`.
    ///
    /// # Panics
    ///
    /// Panics if `predictors.len() != outcomes.len()`.
    pub fn slow_polynomial(
        predictors: &[f64],
        outcomes: &[f64],
        degree: usize,
    ) -> PolynomialCoefficients {
        assert_eq!(predictors.len(), outcomes.len());

        // if degree == 0, get median.
        if degree == 0 {
            let mut outcomes = outcomes.to_vec();
            let constant = crate::percentile::percentile_default_pivot_by(
                &mut outcomes,
                crate::Fraction::HALF,
                &mut |a, b| crate::F64OrdHash::f64_cmp(*a, *b),
            )
            .resolve();
            return PolynomialCoefficients {
                coefficients: vec![constant],
            };
        }

        // init
        let mut iter = permutations_generic(predictors, outcomes, degree + 1);
        let mut coefficients = Vec::with_capacity(degree + 1);
        let permutations_count = permutation_count(predictors.len(), degree + 1)
            .unwrap_or_else(|| estimate_permutation_count(predictors.len(), degree + 1) as usize);
        for _ in 0..degree + 1 {
            // now that's a lot of allocations.
            coefficients.push(Vec::with_capacity(permutations_count))
        }

        // Hard-code some of these to increase performance. Else, we'd have to do linear algebra to
        // get the equation of a straight line from two points.
        match degree {
            0 => unreachable!("we handled this above"),
            1 => {
                while let Some(buf) = iter.next() {
                    debug_assert_eq!(buf.len(), 2);

                    // SAFETY: I know the buf is going to be exactly 2 in length, as I wrote the
                    // code.
                    let p1 = unsafe { buf.get_unchecked(0) };
                    let x1 = p1.0;
                    let y1 = p1.1;
                    let p2 = unsafe { buf.get_unchecked(1) };
                    let x2 = p2.0;
                    let y2 = p2.1;

                    let slope = (y1 - y2) / (x1 - x2);
                    // y=slope * x + intersect
                    // intersect = y - slope * x
                    // we could've chosen p2, it doesn't matter.
                    let intersect = y1 - x1 * slope;

                    // SAFETY: we pushed these vecs to `coefficients` above.
                    unsafe {
                        coefficients.get_unchecked_mut(1).push(slope);
                        coefficients.get_unchecked_mut(0).push(intersect);
                    }

                    iter.give_buffer(buf);
                }
            }
            // 10x performance increase with this hand-crafted technique.
            2 => {
                while let Some(buf) = iter.next() {
                    debug_assert_eq!(buf.len(), 3);

                    // SAFETY: I know the buf is going to be exactly 2 in length, as I wrote the
                    // code.
                    let p1 = unsafe { buf.get_unchecked(0) };
                    let x1 = p1.0;
                    let y1 = p1.1;
                    let p2 = unsafe { buf.get_unchecked(1) };
                    let x2 = p2.0;
                    let y2 = p2.1;
                    let p3 = unsafe { buf.get_unchecked(2) };
                    let x3 = p3.0;
                    let y3 = p3.1;

                    // Derived from the systems of equation this makes.
                    // See https://math.stackexchange.com/a/680695

                    let a = (x1 * (y3 - y2) + x2 * (y1 - y3) + x3 * (y2 - y1))
                        / ((x1 - x2) * (x1 - x3) * (x2 - x3));
                    let b = (y2 - y1) / (x2 - x1) - a * (x1 + x2);
                    let c = y1 - a * x1 * x1 - b * x1;

                    // SAFETY: we pushed these vecs to `coefficients` above.
                    unsafe {
                        coefficients.get_unchecked_mut(2).push(a);
                        coefficients.get_unchecked_mut(1).push(b);
                        coefficients.get_unchecked_mut(0).push(c);
                    }

                    iter.give_buffer(buf);
                }
            }
            #[cfg(not(feature = "ols"))]
            _ => {
                panic!("unsupported degree for polynomial Theil-Sen. Supports 1,2 without the OLS cargo feature.");
            }
            #[cfg(feature = "ols")]
            _ => {
                while let Some(buf) = iter.next() {
                    #[inline(always)]
                    fn tuple_first(t: &(f64, f64)) -> f64 {
                        t.0
                    }
                    #[inline(always)]
                    fn tuple_second(t: &(f64, f64)) -> f64 {
                        t.1
                    }

                    debug_assert_eq!(buf.len(), degree + 1);

                    let predictors = buf.iter().map(tuple_first);
                    let outcomes = buf.iter().map(tuple_second);

                    let polynomial = ols::polynomial(predictors, outcomes, degree + 1, degree);
                    for (pos, coefficient) in polynomial.iter().enumerate() {
                        // SAFETY: we pushed these vecs to `coefficients` above.
                        // pos is less than the size of `coefficients`.
                        unsafe { coefficients.get_unchecked_mut(pos).push(*coefficient) };
                    }

                    iter.give_buffer(buf);
                }
            }
        }

        #[inline(always)]
        fn f64_cmp(a: &f64, b: &f64) -> std::cmp::Ordering {
            crate::F64OrdHash::f64_cmp(*a, *b)
        }

        let mut result = Vec::with_capacity(degree + 1);
        for mut coefficients in coefficients {
            // `TODO`: Choose coefficients for a single point (the median of the coefficient with the
            // highest exponential) instead of then median of the single values.

            // 5x boost in performance here when using `O(n)` median instead of sorting. (when
            // using args `-t -d5` with a detaset of 40 values).
            let median = crate::percentile::percentile_default_pivot_by(
                &mut coefficients,
                crate::Fraction::HALF,
                &mut f64_cmp,
            )
            .resolve();
            result.push(median);
        }
        PolynomialCoefficients {
            coefficients: result,
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn permutations_eq_1() {
            let s1 = [1., 2., 3., 4., 5.];
            let s2 = [2., 4., 6., 8., 10.];

            let permutations1 = permutations(&s1, &s2)
                .map(|(x, y)| [x, y])
                .collect::<Vec<_>>();
            let permutations2 = permutations_generic(&s1, &s2, 2).collect_len();

            assert_eq!(permutations1, permutations2);
        }
        #[test]
        #[cfg(feature = "rand")]
        fn permutations_eq_2() {
            use rand::Rng;

            let mut s1 = [0.0; 20];
            let mut s2 = [0.0; 20];

            let mut rng = rand::rng();
            rng.fill(&mut s1);
            rng.fill(&mut s2);

            let permutations1 = permutations(&s1, &s2)
                .map(|(x, y)| [x, y])
                .collect::<Vec<_>>();
            let permutations2 = permutations_generic(&s1, &s2, 2).collect_len();

            assert_eq!(permutations1, permutations2);
        }
        #[test]
        fn permutations_len_3() {
            let s1 = [1., 2., 3., 4., 5.];
            let s2 = [2., 4., 6., 8., 10.];

            let permutations = permutations_generic(&s1, &s2, 3).collect_len::<3>();

            let expected: &[[(f64, f64); 3]] = &[
                [(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)],
                [(1.0, 2.0), (2.0, 4.0), (4.0, 8.0)],
                [(1.0, 2.0), (2.0, 4.0), (5.0, 10.0)],
                [(1.0, 2.0), (3.0, 6.0), (4.0, 8.0)],
                [(1.0, 2.0), (3.0, 6.0), (5.0, 10.0)],
                [(1.0, 2.0), (4.0, 8.0), (5.0, 10.0)],
                [(2.0, 4.0), (3.0, 6.0), (4.0, 8.0)],
                [(2.0, 4.0), (3.0, 6.0), (5.0, 10.0)],
                [(2.0, 4.0), (4.0, 8.0), (5.0, 10.0)],
                [(3.0, 6.0), (4.0, 8.0), (5.0, 10.0)],
            ];

            assert_eq!(expected.len(), permutation_count(5, 3).unwrap());

            assert_eq!(permutations, expected,);
        }
        #[test]
        fn permutations_len_4_1() {
            let s1 = [1., 2., 3., 4., 5.];
            let s2 = [2., 4., 6., 8., 10.];

            let permutations = permutations_generic(&s1, &s2, 4).collect_len();

            let expected: &[[(f64, f64); 4]] = &[
                [(1.0, 2.0), (2.0, 4.0), (3.0, 6.0), (4.0, 8.0)],
                [(1.0, 2.0), (2.0, 4.0), (3.0, 6.0), (5.0, 10.0)],
                [(1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (5.0, 10.0)],
                [(1.0, 2.0), (3.0, 6.0), (4.0, 8.0), (5.0, 10.0)],
                [(2.0, 4.0), (3.0, 6.0), (4.0, 8.0), (5.0, 10.0)],
            ];

            assert_eq!(expected.len(), permutation_count(5, 4).unwrap());

            assert_eq!(permutations, expected,);
        }
        #[test]
        fn permutations_len_4_2() {
            let s1 = [1., 2., 3., 4., 5., 6.];
            let s2 = [2., 4., 6., 8., 10., 12.];

            let permutations = permutations_generic(&s1, &s2, 4).collect_len();

            let expected: &[[(f64, f64); 4]] = &[
                [(1.0, 2.0), (2.0, 4.0), (3.0, 6.0), (4.0, 8.0)],
                [(1.0, 2.0), (2.0, 4.0), (3.0, 6.0), (5.0, 10.0)],
                [(1.0, 2.0), (2.0, 4.0), (3.0, 6.0), (6.0, 12.0)],
                [(1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (5.0, 10.0)],
                [(1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (6.0, 12.0)],
                [(1.0, 2.0), (2.0, 4.0), (5.0, 10.0), (6.0, 12.0)],
                [(1.0, 2.0), (3.0, 6.0), (4.0, 8.0), (5.0, 10.0)],
                [(1.0, 2.0), (3.0, 6.0), (4.0, 8.0), (6.0, 12.0)],
                [(1.0, 2.0), (3.0, 6.0), (5.0, 10.0), (6.0, 12.0)],
                [(1.0, 2.0), (4.0, 8.0), (5.0, 10.0), (6.0, 12.0)],
                [(2.0, 4.0), (3.0, 6.0), (4.0, 8.0), (5.0, 10.0)],
                [(2.0, 4.0), (3.0, 6.0), (4.0, 8.0), (6.0, 12.0)],
                [(2.0, 4.0), (3.0, 6.0), (5.0, 10.0), (6.0, 12.0)],
                [(2.0, 4.0), (4.0, 8.0), (5.0, 10.0), (6.0, 12.0)],
                [(3.0, 6.0), (4.0, 8.0), (5.0, 10.0), (6.0, 12.0)],
            ];

            assert_eq!(expected.len(), permutation_count(6, 4).unwrap());

            assert_eq!(permutations, expected,);
        }
    }
}

/// Spiral estimator, a robust sampling estimator.
/// This should be more robust than [`theil_sen`].
///
/// > This is a brainchild of this library's lead developer [Icelk](mailto:Icelk<main@icelk.dev>).
///
/// The [`spiral::Options`] implement most of the [estimators](models).
///
/// # Advantages
///
/// You supply a `fitness_function` to all functions which tells the algorithm which lines are
/// good. The magnitude is irrelevant, only order is considered. The algorithm tries to *minimize*
/// the returned value. **This allows you to choose the desired properties of resulting
/// line/polynomial, without checking all possible values.**
///
/// # Caveats
///
/// The polynomial regression implementation only allows for degrees 1 & 2.
/// See [details](#details) for more info on this.
///
/// The sampling technique means this might miss the right point to close in on. Therefore, I
/// highly recommend using the [higher quality options](spiral::Options::new).
///
/// ## Robustness
///
/// Since this uses a fitness function, the robustness is determined by that. Using the "default"
/// `manhattan_distance` gives good results (think least squares, but without the squared
/// importance of errors). This is what the implementations for [`spiral::Options`] does.
///
/// Since this tests a wide range of possibilities before deciding on one, it's very likely we
/// don't get trapped in a local maxima.
///
/// # Performance
///
/// The functions are `O(fitness function)` where `O(fitness function)` is the time
/// complexity of your `fitness_function`. That's often `O(n)` as you'd probably in some way
/// sum up the points relative to the model.
///
/// This puts the algorithm similar to [`ols`], but with much worse (read: 4x-100x) performance.
/// This may be justified by the [advantages](#advantages).
/// It scales much better than [`theil_sen`] and is more robust, but when the count of points is
/// small, `theil_sen` is faster.
///
/// # Details
///
/// The idea is to make a [phase space](https://en.wikipedia.org/wiki/Phase_space)
/// of the parameters to a line (`y=(slope)*x + (y-intersect)`). We then traverse the phase space
/// with a [logarithmic spiral](https://en.wikipedia.org/wiki/Logarithmic_spiral)
/// and sample points (we start at the angle θ e.g. `-12π` and go to a max value, e.g. `12π`)
/// on an interval. When the range of the spiral has been sampled, we choose the best point and
/// create a spiral there. Depending on how far out the new point, scale the whole spiral's size
/// (by `distance.sqrt().sqrt()`). Then repeat.
///
/// Parameters are chosen for an optimal spiral. The logarithmic spiral was chosen due to the
/// distribution of unknown numbers (which the coefficients of the line are). There's generally
/// more numbers in the range 0..100 than in 100..200. Therefore, we search more numbers in 0..100.
///
/// We can do this in 3 dimensions by creating a 3d spiral. For this, I used a
/// [spherical spiral](https://en.wikipedia.org/wiki/Spiral#Spherical_spirals)
/// where the radius grows with the angle of the spiral, calculated by `e^(θk)` where `k` is a
/// parameter, similar to how a logarithmic spiral is created.
///
/// > On implementing third degree polynomials,
/// > can we get a spiral on a hypersphere? Or do we just need a even distribution?
///
/// See [`spiral::Options`] for more info on the parameters.
pub mod spiral {
    use super::*;
    use std::f64::consts::{E, TAU};
    use std::ops::Range;
    use utils::*;

    /// Samples points on a logarithmic spiral in the phase space of all possible straight lines.
    ///
    /// See [`Options`].
    ///
    /// Can be used for models other than linear, as this just optimizes two floats according to
    /// `fitness_function`. The returned values are the best match.
    pub fn two_variable_optimization(
        fitness_function: impl Fn([f64; 2]) -> f64,
        options: Options,
    ) -> [f64; 2] {
        let Options {
            exponent_coefficient,
            angle_coefficient,
            num_lockon,
            samples_per_rotation,
            range,
            turns: _,
        } = options;
        let advance = TAU / samples_per_rotation;
        let mut best = ((f64::MIN, [1.; 2], 1.), [0.; 2]);
        let mut last_best = f64::MIN;

        let mut exponent_coefficients = [exponent_coefficient; 2];

        for i in 0..num_lockon {
            let mut theta = range.start;
            while theta < range.end {
                let r = E.powf(theta * angle_coefficient);
                let a0 = r * theta.cos();
                let b0 = r * theta.sin();
                let a = a0 * exponent_coefficients[0] + best.1[0];
                let b = b0 * exponent_coefficients[1] + best.1[1];

                let coeffs = [a, b];

                let fitness = fitness_function(coeffs);
                if fitness > best.0 .0 {
                    best = ((fitness, [a0, b0], r), coeffs);
                }

                theta += advance;
            }
            // If the best didn't change, we aren't going to find better results.
            if last_best == best.0 .0 && i != 0 {
                return best.1;
            }
            // Update "zoom" of spiral
            // don't go full out, "ease" this into several approaching steps with .sqrt to avoid
            // overcorrection.
            let best_size = best.0;
            exponent_coefficients[0] *= (best_size.1[0].abs() + best_size.2 / 32.).sqrt();
            exponent_coefficients[1] *= (best_size.1[1].abs() + best_size.2 / 32.).sqrt();

            last_best = best.0 .0;

            // uncomment line below to see how the coefficient changes and the current best.
            // reveals how it shrinks the spiral, and sometimes enlarges it to later zoom in (if
            // enough iterations are allowed)
            //
            // println!("Iteration complete. exponent_coefficients: {exponent_coefficients:.3?} best: {best:.3?}");
        }
        best.1
    }
    /// Samples points on a spherical spiral in the phase space of all second degree polynomials.
    /// As θ (the angle) increases, the imaginary sphere's size is increased.
    /// This gives a good distribution of sample points in 3d space.
    ///
    /// See [`Options`].
    ///
    /// This function just optimizes three floats according to `fitness_function`.
    /// The returned value is the best match.
    pub fn three_variable_optimization(
        fitness_function: impl Fn([f64; 3]) -> f64,
        options: Options,
    ) -> [f64; 3] {
        // See the function above for more documentation.
        // This is the same, but with three dimensions instead.
        let Options {
            exponent_coefficient,
            angle_coefficient,
            num_lockon,
            samples_per_rotation,
            range,
            turns,
        } = options;
        let advance = TAU / samples_per_rotation;

        let mut best = ((f64::MIN, [1.; 3], 1.), [0.; 3]);
        let mut last_best = f64::MIN;

        let mut exponent_coefficients = [exponent_coefficient; 3];

        for i in 0..num_lockon {
            let mut theta = range.start;
            while theta < range.end {
                let r = E.powf(theta * angle_coefficient);
                let a0 = r * theta.sin() * (turns * theta).cos();
                let b0 = r * theta.sin() * (turns * theta).sin();
                let c0 = r * theta.cos();
                let a = a0 * exponent_coefficients[0] + best.1[0];
                let b = b0 * exponent_coefficients[1] + best.1[1];
                let c = c0 * exponent_coefficients[2] + best.1[2];

                let coeffs = [a, b, c];

                let fitness = fitness_function(coeffs);
                if fitness > best.0 .0 {
                    best = ((fitness, [a0, b0, c0], r), coeffs);
                }

                theta += advance;
            }
            if last_best == best.0 .0 && i != 0 {
                return best.1;
            }

            let best_size = best.0;
            exponent_coefficients[0] *= (best_size.1[0].abs() + best_size.2 / 32.).sqrt();
            exponent_coefficients[1] *= (best_size.1[1].abs() + best_size.2 / 32.).sqrt();
            exponent_coefficients[2] *= (best_size.1[2].abs() + best_size.2 / 32.).sqrt();

            last_best = best.0 .0;
        }
        best.1
    }

    /// Options for the spiral.
    ///
    /// This also implements most [estimator](models) traits.
    /// These all use the manhattan distance as their fitness function.
    /// The estimators have `O(n)` runtime performance and `O(1)` size performance.
    ///
    /// > Polynomial estimator only supports degrees 1 & 2.
    ///
    /// See [module-level documentation](self) for more info about concepts.
    ///
    /// > The [`Self::range`] limit samples on the spiral. This causes a upper limit for
    /// > coefficients and a lower "precision" cutoff. You can increase
    /// > [`Self::exponent_coefficient`] if you know your data will have large numbers.
    /// > The algorithm will increase the size of the spiral of a line outside the spiral is found.
    ///
    /// Use a graphing tool (e.g. Desmos) and plot `r=ae^(kθ)`.
    /// a is [`Self::exponent_coefficient`].
    /// k is [`Self::angle_coefficient`].
    ///
    /// To keep the same "size" of the spiral, you have to multiply both ends of [`Self::range`]
    /// with the factor of [`Self::angle_coefficient`].
    ///
    /// # Performance
    ///
    /// The methods are `O(1)*O(fitness_function)` where `O(fitness_function)` is the time
    /// complexity of your `fitness_function`. That's often `O(n)` as you'd probably in some way
    /// sum up the points relative to the model.
    ///
    /// The following options affect the performance as follows (roughly, no coefficients)
    /// `O(num_lockon * samples_per_rotation * range.length)`.
    ///
    /// Keep in mind you should probably not lower [`Self::angle_coefficient`] bellow `0.15`
    /// if you don't increase the [`Self::range`].
    #[must_use]
    #[derive(Debug, Clone, PartialEq)]
    pub struct Options {
        /// The initial scale of the spiral.
        ///
        /// This gets adjusted when locking on.
        pub exponent_coefficient: f64,
        /// The "density" of the spiral.
        pub angle_coefficient: f64,
        /// How many lockons we are allowed to do. This is a max value.
        pub num_lockon: usize,
        /// The count of samples per each rotation in the spiral.
        pub samples_per_rotation: f64,
        /// The range of angles to test.
        pub range: Range<f64>,
        /// The turns of the 3d spiral when using [`three_variable_optimization`].
        /// Frequency of turns per sphere. Is unnecessary to turn up when
        /// [`Self::samples_per_rotation`] is low.
        pub turns: f64,
    }
    impl Options {
        /// Create a new set of options with good defaults.
        ///
        /// `level` is allowed to be in the range [1..=9].
        /// Higher values are more "precise" - they take longer but are also way
        /// (especially `level>4`) more likely to return good results.
        ///
        /// Expect a 2-4x increase in runtime per increment of `level`.
        ///
        /// # Panics
        ///
        /// Panics if `!(1..=9).contains(level)`.
        pub fn new(level: u8) -> Self {
            if !(1..=9).contains(&level) {
                panic!("level of spiral::Options is out of bounds. Accepts 1..=9");
            }
            let level = level as usize - 1;
            // have settings for all levels (1 through 9)
            //
            // These values are based on my intuition of the algorithm.
            let num_lockon = [8, 8, 10, 12, 16, 16, 24, 32, 64];
            let angle_coefficient = [0.23, 0.23, 0.15, 0.13, 0.11, 0.09, 0.07, 0.05, 0.03];
            // these are odd values to avoid repeating the same angle on multiple turns
            let samples_per_rotation = [15., 19., 29., 34., 38., 49., 71., 115., 217.];
            let turns = [10., 12., 12., 14., 16., 16., 16., 16., 24.];
            let range = [
                -2.0..2.,
                -2.0..4.,
                -4.0..4.,
                -4.0..6.,
                -6.0..6.,
                -6.0..6.,
                -6.0..6.,
                -6.0..8.,
                -8.0..12.,
            ];
            let num_lockon = num_lockon[level];
            let angle_coefficient = angle_coefficient[level];
            let samples_per_rotation = samples_per_rotation[level];
            let turns = turns[level];
            let range = range[level].clone();
            let range = (range.start * TAU)..(range.end * TAU);
            Self {
                exponent_coefficient: 10.,
                angle_coefficient,
                num_lockon,
                samples_per_rotation,
                range,
                turns,
            }
        }
    }
    impl Default for Options {
        fn default() -> Self {
            Self::new(5)
        }
    }

    pub(super) struct SecondDegreePolynomial(pub(super) [f64; 3]);
    impl Predictive for SecondDegreePolynomial {
        fn predict_outcome(&self, predictor: f64) -> f64 {
            self.0[0] + self.0[1] * predictor + self.0[2] * predictor * predictor
        }
    }

    /// [`LinearEstimator`] for the spiral estimator using a fitness function and [`Options`]
    /// provided by you.
    /// `O(fitness function)`
    pub struct SpiralLinear<F: Fn(&LinearCoefficients, &[f64], &[f64]) -> f64>(pub F, pub Options);
    impl<F: Fn(&LinearCoefficients, &[f64], &[f64]) -> f64> LinearEstimator for SpiralLinear<F> {
        fn model_linear(&self, predictors: &[f64], outcomes: &[f64]) -> LinearCoefficients {
            wrap_linear(two_variable_optimization(
                #[inline(always)]
                |model| self.0(&wrap_linear(model), predictors, outcomes),
                self.1.clone(),
            ))
        }
    }

    macro_rules! impl_estimator {
        ($(
            $name:ident, $method:ident, $fn:ident, $ret:ident, $wrap:expr,
        )+) => {
            $(
                impl $name for Options {
                    fn $method(&self, predictors: &[f64], outcomes: &[f64]) -> $ret {
                        $wrap($fn(
                            #[inline(always)]
                            |model| manhattan_distance(&$wrap(model), predictors, outcomes),
                            self.clone(),
                        ))
                    }
                }
            )+
        };
    }
    macro_rules! impl_estimator_trig {
        ($(
            $name:ident, $method:ident, $fn:ident, $ret:ident, $wrap:expr,
        )+) => {
            $(
                impl $name for Options {
                    fn $method(&self, predictors: &[f64], outcomes: &[f64], max_frequency: f64) -> $ret {
                        $wrap($fn(
                            #[inline(always)]
                            |model| trig_adjusted_manhattan_distance(&$wrap(model), model, predictors, outcomes, max_frequency),
                            self.clone(),
                        ))
                    }
                }
            )+
        };
    }
    impl_estimator!(
        LinearEstimator,
        model_linear,
        two_variable_optimization,
        LinearCoefficients,
        wrap_linear,
        //
        PowerEstimator,
        model_power,
        two_variable_optimization,
        PowerCoefficients,
        wrap_power,
        //
        ExponentialEstimator,
        model_exponential,
        two_variable_optimization,
        ExponentialCoefficients,
        wrap_exponential,
        //
        LogisticEstimator,
        model_logistic,
        three_variable_optimization,
        LogisticCoefficients,
        wrap_logistic,
    );
    impl_estimator_trig!(
        SineEstimator,
        model_sine,
        three_variable_optimization,
        SineCoefficients,
        SineCoefficients::wrap,
        //
        CosineEstimator,
        model_cosine,
        three_variable_optimization,
        CosineCoefficients,
        CosineCoefficients::wrap,
        //
        TangentEstimator,
        model_tangent,
        three_variable_optimization,
        TangentCoefficients,
        TangentCoefficients::wrap,
        //
        SecantEstimator,
        model_secant,
        three_variable_optimization,
        SecantCoefficients,
        SecantCoefficients::wrap,
        //
        CosecantEstimator,
        model_cosecant,
        three_variable_optimization,
        CosecantCoefficients,
        CosecantCoefficients::wrap,
        //
        CotangentEstimator,
        model_cotangent,
        three_variable_optimization,
        CotangentCoefficients,
        CotangentCoefficients::wrap,
    );
    impl PolynomialEstimator for Options {
        fn model_polynomial(
            &self,
            predictors: &[f64],
            outcomes: &[f64],
            degree: usize,
        ) -> PolynomialCoefficients {
            match degree {
                1 => wrap_linear(two_variable_optimization(
                    #[inline(always)]
                    |model| manhattan_distance(&wrap_linear(model), predictors, outcomes),
                    self.clone(),
                ))
                .into(),
                2 => three_variable_optimization(
                    #[inline(always)]
                    |model| {
                        manhattan_distance(&SecondDegreePolynomial(model), predictors, outcomes)
                    },
                    self.clone(),
                )
                .into(),
                _ => panic!("unsupported degree for polynomial spiral. Supports 1,2."),
            }
        }
    }

    /// Implements [`LogisticEstimator`] with a known ceiling for the input values.
    /// Uses manhattan distance as the fitness function.
    ///
    /// This can be used to model logistic growth with a known max.
    #[derive(Debug, Clone, PartialEq)]
    pub struct SpiralLogisticWithCeiling {
        /// The options of the spiral regression.
        pub opts: Options,
        /// The max value of the input values.
        /// This becomes [`LogisticCoefficients::l`].
        pub ceiling: f64,
    }

    impl SpiralLogisticWithCeiling {
        /// Create a new estimator with `ceiling` and `opts`.
        pub fn new(opts: Options, ceiling: f64) -> Self {
            Self { opts, ceiling }
        }
    }
    impl LogisticEstimator for SpiralLogisticWithCeiling {
        fn model_logistic(&self, predictors: &[f64], outcomes: &[f64]) -> LogisticCoefficients {
            fn wrap(a: [f64; 2], max: f64) -> LogisticCoefficients {
                LogisticCoefficients {
                    x0: a[0],
                    l: max,
                    k: a[1],
                }
            }
            wrap(
                two_variable_optimization(
                    #[inline]
                    |model| manhattan_distance(&wrap(model, self.ceiling), predictors, outcomes),
                    self.opts.clone(),
                ),
                self.ceiling,
            )
        }
    }
}

/// Assumes the fitness function has a minimal slope when the value is optimal (i.e. e.g.
/// `(x-4.).abs()` will not work, since it's slope is constant and then changes sign)
#[allow(missing_docs)]
pub mod gradient_descent {
    use super::*;
    use utils::BorrowedPolynomial;

    pub struct ParallelOptions {
        pub learn_rate: f64,
        pub factor_decrease: f64,
        pub rough_max_sign_changes: usize,
        pub rough_slope_reduction_goal: f64,
        pub rough_iterations_base: usize,
        pub rough_iterations_per_degree: usize,
        pub fine_iterations_base: usize,
        pub fine_iterations_per_degree: usize,
    }
    impl Default for ParallelOptions {
        fn default() -> Self {
            Self {
                learn_rate: 50.,
                factor_decrease: 1.2,
                rough_max_sign_changes: 100,
                rough_slope_reduction_goal: 4.,
                rough_iterations_base: 64,
                rough_iterations_per_degree: 13,
                fine_iterations_base: 4,
                fine_iterations_per_degree: 3,
            }
        }
    }
    pub struct SimultaneousOptions {
        pub learn_rate: f64,
        pub factor_decrease: f64,
        pub factor_increase: f64,
        pub target_accuracy: f64,
    }
    impl SimultaneousOptions {
        pub fn new(target_accuracy: f64) -> Self {
            Self {
                learn_rate: 0.0001,
                factor_decrease: 1.5,
                factor_increase: 1.8,
                target_accuracy,
            }
        }
    }
    impl ParallelOptions {
        #[inline(always)]
        fn adjusted_slope(n: f64) -> f64 {
            let n = n / 8.;
            let ln = match n.partial_cmp(&0.) {
                Some(std::cmp::Ordering::Less) => -((-n + 1.).ln()),
                Some(std::cmp::Ordering::Greater) => (n + 1.).ln(),
                _ => 0.,
            };
            ln * 8.
        }
        pub fn polynomial_optimization(
            &self,
            n: usize,
            target_accuracy: f64,
            fitness_function: impl Fn(&[f64]) -> f64,
        ) -> Vec<f64> {
            let mut values: Vec<f64> = std::iter::repeat(0.).take(n).collect();
            let mut factors: Vec<f64> = std::iter::repeat(1.).take(n).collect();
            let dx = (target_accuracy / 2.).max(1e-11);

            let get_slope = |dx: f64, i: usize, values: &mut [f64]| {
                let y1 = fitness_function(values);
                values[i] += dx;
                let y2 = fitness_function(values);
                values[i] -= dx;
                (y1 - y2) / dx
            };

            let rough_iterations =
                self.rough_iterations_base + self.rough_iterations_per_degree * n;
            let fine_iterations = self.fine_iterations_base + self.fine_iterations_per_degree * n;

            for _ in 0..rough_iterations {
                for i in 0..n {
                    let first_slope = get_slope(dx, i, &mut values);
                    let mut slope_positive = first_slope.is_sign_positive();
                    let mut factor = factors[i];
                    let mut sign_changes = 0;
                    loop {
                        let slope = get_slope(dx, i, &mut values);
                        if slope.is_sign_positive() != slope_positive {
                            slope_positive = !slope_positive;
                            factor /= self.factor_decrease;
                            sign_changes += 1;
                        }
                        values[i] += Self::adjusted_slope(slope) * self.learn_rate * factor;
                        // println!(
                        // "Slope {slope} add {} log {} factor {factor}",
                        // adjusted_slope(slope) * self.learn_rate * factor,
                        // adjusted_slope(slope),
                        // );
                        if slope.abs() < first_slope.abs() / self.rough_slope_reduction_goal
                            || sign_changes > self.rough_max_sign_changes
                            || slope.abs() < target_accuracy * 2.
                        {
                            break;
                        }
                        // `TODO`: if all factors are under target, return values
                    }
                    factors[i] = factor;
                }
            }

            for _ in 0..fine_iterations {
                for i in 0..n {
                    let mut factor = factors[i];
                    let mut slope_positive = get_slope(dx, i, &mut values).is_sign_positive();
                    loop {
                        let slope = get_slope(dx, i, &mut values);
                        if slope.is_sign_positive() != slope_positive {
                            slope_positive = !slope_positive;
                            factor /= self.factor_decrease;
                        }
                        values[i] += Self::adjusted_slope(slope) * self.learn_rate * factor;
                        // println!(
                        // "Slope {slope} add {} log {} {factor}",
                        // adjusted_slope(slope) * self.learn_rate * factor,
                        // adjusted_slope(slope),
                        // );
                        if slope.abs() < target_accuracy {
                            break;
                        }
                        // `TODO`: if all factors are under target, return values
                    }
                    factors[i] = factor;
                }
            }

            values
        }
    }
    impl SimultaneousOptions {
        #[inline(always)]
        fn adjusted_slope(n: f64) -> f64 {
            let n = n / 0.1;
            let ln = match n.partial_cmp(&0.) {
                Some(std::cmp::Ordering::Less) => -((-n + 1.).ln()),
                Some(std::cmp::Ordering::Greater) => (n + 1.).ln(),
                _ => 0.,
            };
            ln * 0.1
        }
        pub fn polynomial_optimization(
            &self,
            n: usize,
            fitness_function: impl Fn(&[f64]) -> f64,
        ) -> Vec<f64> {
            let mut values: Vec<f64> = std::iter::repeat(0.).take(n).collect();
            let mut factors: Vec<f64> = std::iter::repeat(1.).take(n).collect();
            let mut slopes: Vec<f64> = std::iter::repeat(0.).take(n).collect();
            let dx = 1e-11;

            let get_slope = |dx: f64, i: usize, values: &mut [f64]| {
                let y1 = fitness_function(values);
                values[i] += dx;
                let y2 = fitness_function(values);
                values[i] -= dx;
                (y1 - y2) / dx
            };

            let mut i = 0_usize;
            loop {
                i += 1;
                if i > 1_000_000 {
                    break;
                }
                for i in 0..n {
                    let slope = get_slope(dx, i, &mut values);
                    if slope.is_sign_positive() != slopes[i].is_sign_positive() {
                        if slope.abs() > slopes[i].abs() * 4.
                            && self.factor_decrease * self.factor_decrease
                                < (slope.abs() / slopes[i].abs())
                        {
                            factors[i] /= self
                                .factor_decrease
                                .max((slope.abs() / slopes[i].abs()).sqrt());
                        } else {
                            factors[i] /= self.factor_decrease;
                        }
                        factors[i] = factors[i].max(1e-6);
                    } else {
                        factors[i] *= self.factor_increase;
                    }
                    slopes[i] = slope;
                }

                for i in 0..n {
                    let factor = factors[i];
                    let slope = slopes[i];
                    values[i] += Self::adjusted_slope(slope) * self.learn_rate * factor;
                    // println!(
                    // "{i} slope {slope} add {} {factor}",
                    // adjusted_log(slope) * self.learn_rate * factor,
                    // );
                }
                if i % 20 == 0 {
                    let mut v = 0.;
                    let mut factor = 1.;
                    for slope in &slopes {
                        v += slope.abs() * factor;
                        factor /= 5.0;
                    }
                    // println!("{v} {slopes:?}");
                    if v < self.target_accuracy {
                        break;
                    }
                }
            }
            values
        }
    }

    impl PolynomialEstimator for ParallelOptions {
        fn model_polynomial(
            &self,
            predictors: &[f64],
            outcomes: &[f64],
            degree: usize,
        ) -> PolynomialCoefficients {
            PolynomialCoefficients::from(self.polynomial_optimization(degree + 1, 1e-6, |v| {
                -BorrowedPolynomial(v).determination_slice(predictors, outcomes)
            }))
        }
    }
    impl LinearEstimator for ParallelOptions {
        fn model_linear(&self, predictors: &[f64], outcomes: &[f64]) -> LinearCoefficients {
            let v = self.polynomial_optimization(2, 1e-8, |v| {
                -LinearCoefficients { k: v[0], m: v[1] }.determination_slice(predictors, outcomes)
            });
            LinearCoefficients { k: v[0], m: v[1] }
        }
    }
    impl PolynomialEstimator for SimultaneousOptions {
        fn model_polynomial(
            &self,
            predictors: &[f64],
            outcomes: &[f64],
            degree: usize,
        ) -> PolynomialCoefficients {
            PolynomialCoefficients::from(self.polynomial_optimization(degree + 1, |v| {
                -BorrowedPolynomial(v).determination_slice(predictors, outcomes)
            }))
        }
    }
    impl LinearEstimator for SimultaneousOptions {
        fn model_linear(&self, predictors: &[f64], outcomes: &[f64]) -> LinearCoefficients {
            let v = self.polynomial_optimization(2, |v| {
                -LinearCoefficients { k: v[0], m: v[1] }.determination_slice(predictors, outcomes)
            });
            LinearCoefficients { k: v[0], m: v[1] }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn one_variable() {
            let now = std::time::Instant::now();
            let v = ParallelOptions::default()
                .polynomial_optimization(1, 1e-12, |values| (values[0] - 42.4242).powi(2));
            println!("{v:?} {:?}", now.elapsed());
        }
        #[test]
        fn two_variable_regression() {
            let now = std::time::Instant::now();
            let x = [1.3, 4.7, 9.4];
            let y = [4., 5.3, 6.7];
            let v = ParallelOptions::default().polynomial_optimization(2, 1e-6, |values| {
                -LinearCoefficients {
                    k: values[0],
                    m: values[1],
                }
                .determination_slice(&x, &y)
            });
            let coeffs = LinearCoefficients { k: v[0], m: v[1] };
            println!(
                "{coeffs} R² {} {:?}",
                coeffs.determination_slice(&x, &y),
                now.elapsed()
            );
        }
    }
}

/// A random binary searching n-variable optimizer.
///
/// Independently binary searches the variables over the entire range of `f64`s.
/// Supports any number of variables to be optimized together.
///
/// This has performance in the ballpark of OLS, but enables you to give your own function,
/// which means you can optimize for things other than least squares along straight lines.
/// This opens up the opportunity to fit other functions (any you want) and
/// to use functions less prone to outliers (least squares is very prone).
pub mod binary_search {
    use super::*;
    #[cfg(feature = "binary_search_rng")]
    use rand::Rng;
    use std::borrow::Cow;

    /// A trait which allows storage of n-variable optimization, either on the stack through arrays
    /// (`[f64; VARIABLE_COUNT]`) or allocated on the heap through `Vec`.
    #[allow(clippy::len_without_is_empty)] // just no
    pub trait NVariableStorage:
        std::ops::IndexMut<usize, Output = f64> + AsRef<[f64]> + AsMut<[f64]> + Clone
    {
        /// Associated data for use in construction of this type.
        /// The number of arguments in case of using a `Vec`,
        /// nothing when using arrays, as we know their length in
        /// compile time.
        type Data;
        /// The structure that's given to the fitness function.
        /// Needs to have the same length as the `.as_ref()` implementation of this struct.
        type Given<'a>: AsRef<[f64]>
        where
            Self: 'a;
        /// Creates a new storage filled with `num`.
        fn new_filled(data: &Self::Data, num: f64) -> Self;
        /// Borrow the current variables.
        fn borrow(&self) -> Self::Given<'_>;
    }
    impl<const LENGTH: usize> NVariableStorage for [f64; LENGTH] {
        type Data = ();
        type Given<'a> = [f64; LENGTH];
        fn new_filled(_data: &(), num: f64) -> Self {
            [num; LENGTH]
        }
        fn borrow(&self) -> Self::Given<'_> {
            *self
        }
    }

    /// Dynamically sized storage, for use when the number of variables isn't known at compile
    /// time.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct VariableLengthStorage(pub usize);
    impl NVariableStorage for Vec<f64> {
        type Data = VariableLengthStorage;
        type Given<'a> = &'a [f64];
        fn new_filled(data: &Self::Data, num: f64) -> Self {
            vec![num; data.0]
        }
        fn borrow(&self) -> Self::Given<'_> {
            self
        }
    }
    impl From<usize> for VariableLengthStorage {
        fn from(n: usize) -> Self {
            Self(n)
        }
    }

    /// Generated using:
    /// ```
    /// (0..61)
    ///     .into_iter()
    ///     .map(|i| {
    ///         f64::MAX.powf(1. / (2.0f64.powi(i+2)))
    ///     })
    ///     .collect::<Vec<_>>();
    /// ```
    // braces for folding
    static SQRTS_FROM_F64_MAX: [f64; 61] = {
        [
            1.157920892373162e77,
            3.402823669209385e38,
            1.8446744073709552e19,
            4294967296.0,
            65536.0,
            256.0,
            16.0,
            4.0,
            2.0,
            core::f64::consts::SQRT_2,
            1.189207115002721,
            1.0905077326652577,
            1.0442737824274138,
            1.0218971486541166,
            1.0108892860517005,
            1.0054299011128027,
            1.0027112750502025,
            1.0013547198921082,
            1.0006771306930664,
            1.0003385080526823,
            1.0001692397053021,
            1.0000846162726944,
            1.0000423072413958,
            1.0000211533969647,
            1.0000105766425498,
            1.0000052883072919,
            1.0000026441501502,
            1.0000013220742012,
            1.0000006610368821,
            1.0000003305183864,
            1.0000001652591795,
            1.0000000826295863,
            1.0000000413147923,
            1.000000020657396,
            1.000000010328698,
            1.000000005164349,
            1.0000000025821745,
            1.0000000012910872,
            1.0000000006455436,
            1.0000000003227718,
            1.0000000001613858,
            1.000000000080693,
            1.0000000000403464,
            1.0000000000201732,
            1.0000000000100866,
            1.0000000000050433,
            1.0000000000025218,
            1.0000000000012608,
            1.0000000000006304,
            1.0000000000003153,
            1.0000000000001577,
            1.0000000000000788,
            1.0000000000000393,
            1.0000000000000198,
            1.0000000000000098,
            1.0000000000000049,
            1.0000000000000024,
            1.0000000000000013,
            1.0000000000000007,
            1.0000000000000002,
            1.0000000000000002,
        ]
    };

    /// Options for the binary search optimization.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct Options {
        /// Number of iterations to search for the optimal value
        pub iterations: usize,
        /// How fine values you can get. 59 covers the whole range of `f64`
        /// 30 seems to get you ~7 significant digits
        pub precision: usize,
        /// The assumed max value. Use `f64::MAX` to cover the whole range of `f64`.
        pub max: f64,
        /// The factor for the randomness introduced when binary searching.
        /// Higher values result in finding more optimal values, but can also make it hard for the
        /// algorithm to find a good value.
        #[cfg(feature = "binary_search_rng")]
        pub randomness_factor: f64,
        /// Config for using [`random_subset_regression`].
        #[cfg(feature = "random_subset_regression")]
        pub random_subset_regression: Option<random_subset_regression::Config>,
    }
    impl Default for Options {
        fn default() -> Self {
            Self {
                iterations: 30,
                precision: 30,
                max: f64::MAX,
                #[cfg(feature = "binary_search_rng")]
                randomness_factor: 1.,
                #[cfg(feature = "random_subset_regression")]
                random_subset_regression: Some(Default::default()),
            }
        }
    }
    impl Options {
        /// Get max precision of every variable.
        pub fn max_precision(&self) -> Self {
            let mut me = *self;
            me.precision = 61;
            me
        }

        /// Like [`Options::n_variable_optimization`] but without random variation. More easily
        /// falls into local maxima (a variables thought to be the best). Useful for independent
        /// variables.
        ///
        /// Faster than [`Options::n_variable_optimization`].
        pub fn n_variable_optimization_no_rng<NV: NVariableStorage>(
            &self,
            fitness_function: impl Fn(NV::Given<'_>) -> f64,
            data: NV::Data,
        ) -> NV {
            let initial_center = self.max.sqrt();

            let mut values = NV::new_filled(&data, 0.);
            let factors = if self.max == f64::MAX {
                Cow::Borrowed(
                    &SQRTS_FROM_F64_MAX[0..(self.precision.min(SQRTS_FROM_F64_MAX.len()))],
                )
            } else {
                let mut f = initial_center;
                Cow::Owned(
                    (0..self.precision.min(61))
                        .map(|_| {
                            f = f.sqrt();
                            f
                        })
                        .collect::<Vec<_>>(),
                )
            };
            let n = values.as_ref().len();

            for _ in 0..self.iterations {
                for i in 0..n {
                    let mut center = initial_center;
                    // for each precision level
                    for factor in factors.as_ref() {
                        // -1 to get 0 (the repeated division by sqrt approaches 1)
                        let center_over = center * factor;
                        let center_under = center / factor;
                        let value_over = center_over - 1.0;
                        let value_under = center_under - 1.0;

                        let value_negative = -value_under;
                        values[i] = value_negative;
                        let fitness_negative = fitness_function(values.borrow());

                        values[i] = value_over;
                        let fitness_over = fitness_function(values.borrow());
                        values[i] = value_under;
                        let fitness_under = fitness_function(values.borrow());
                        let best_current_sign = fitness_over.min(fitness_under);

                        // if negative is optimal
                        if fitness_negative < best_current_sign {
                            values[i] = -value_over;
                            let fitness_negative_over = fitness_function(values.borrow());
                            values[i] = -value_under;
                            let fitness_negative_under = fitness_function(values.borrow());

                            let best_opposite_sign =
                                fitness_negative_over.min(fitness_negative_under);
                            if best_opposite_sign < best_current_sign {
                                if fitness_negative_under < fitness_negative_over {
                                    center = -center_under;
                                // values[i] = -value_under already set
                                } else {
                                    center = -center_over;
                                    values[i] = -value_over;
                                }
                                continue;
                            }
                        }

                        if !fitness_over.is_finite() || fitness_under < fitness_over {
                            center = center_under;
                            // values[i] = value_under already set
                        } else {
                            center = center_over;
                            values[i] = value_over;
                        }
                    }
                }
            }
            values
        }
        /// Optimize `n` variables to `fitness_function`.
        /// Will return a set of values which (hopefully) minimize `fitness_function`.
        #[cfg(feature = "binary_search_rng")]
        pub fn n_variable_optimization<NV: NVariableStorage>(
            &self,
            fitness_function: impl Fn(NV::Given<'_>) -> f64,
            data: NV::Data,
            rng: &mut impl Rng,
        ) -> NV {
            let initial_center = self.max.sqrt();

            let mut values = NV::new_filled(&data, 0.);
            // pregenerate all successive sqrts of `initial_center`
            // we could instead do `factor = factor.sqrt()` at the end of each
            // `for _ in 0..self.precision`, but then, we'd do this `self.iterations` times.
            //
            // if `self.max` is f64::MAX, then use pregenerated list and don't allocate!
            let factors = if self.max == f64::MAX {
                Cow::Borrowed(
                    &SQRTS_FROM_F64_MAX[0..(self.precision.min(SQRTS_FROM_F64_MAX.len()))],
                )
            } else {
                let mut f = initial_center;
                Cow::Owned(
                    (0..self.precision.min(61))
                        .map(|_| {
                            f = f.sqrt();
                            f
                        })
                        .collect::<Vec<_>>(),
                )
            };

            // track best values
            let mut best_fitness = f64::MAX;
            let mut best_values = values.clone();

            let n = values.as_ref().len();

            for iter in 0..self.iterations {
                // decrease randomness at the end
                let progress = 1.0 - iter as f64 / self.iterations as f64;
                // gen f32 since that takes less bytes
                let rng_factor = 1.
                    + (2.0 * rng.random::<f32>() as f64 - 1.) * self.randomness_factor * progress;

                // for each variable to optimize
                for i in 0..n {
                    let mut center = initial_center;
                    // for each precision level (`for _ in 0..self.precision`, see note at definition
                    // of `factors`)
                    for factor in factors.as_ref() {
                        let factor = factor * rng_factor;
                        // -1 to get 0 (the repeated division by sqrt approaches 1)
                        let center_over = center * factor;
                        let center_under = center / factor;
                        let value_over = center_over - 1.0;
                        let value_under = center_under - 1.0;

                        let value_negative = -value_under;
                        values[i] = value_negative;
                        let fitness_negative = fitness_function(values.borrow());

                        values[i] = value_over;
                        let fitness_over = fitness_function(values.borrow());
                        // micro-optimization: we check the value_under last, so we don't have to
                        // set it again. This is optimal, because value_under is most likely
                        values[i] = value_under;
                        let fitness_under = fitness_function(values.borrow());
                        let best_current_sign = fitness_under.min(fitness_over);

                        // if negative is optimal
                        if fitness_negative < best_current_sign {
                            values[i] = -value_over;
                            let fitness_negative_over = fitness_function(values.borrow());
                            values[i] = -value_under;
                            let fitness_negative_under = fitness_function(values.borrow());

                            let best_opposite_sign =
                                fitness_negative_over.min(fitness_negative_under);
                            if best_opposite_sign < best_current_sign {
                                if fitness_negative_under < fitness_negative_over {
                                    center = -center_under;
                                // values[i] = -value_under already set
                                } else {
                                    center = -center_over;
                                    values[i] = -value_over;
                                }
                                continue;
                            }
                        }
                        if !fitness_over.is_finite() || fitness_under < fitness_over {
                            center = center_under;
                            // values[i] = value_under already set
                        } else {
                            center = center_over;
                            values[i] = value_over;
                        }
                    }
                }

                // update best value
                let fitness = fitness_function(values.borrow());
                if fitness < best_fitness {
                    best_values.as_mut().copy_from_slice(values.as_ref());
                    best_fitness = fitness;
                }
            }
            best_values
        }
    }

    #[cfg(feature = "binary_search_rng")]
    macro_rules! impl_estimator {
        ($(
            $name:ident, $method:ident, $ret:ident, $wrap:expr,
        )+) => {
            $(
                impl $name for Options {
                    fn $method(&self, predictors: &[f64], outcomes: &[f64]) -> $ret {
                        use rand::SeedableRng;
                        let mut rng = rand_xorshift::XorShiftRng::from_rng(&mut rand::rng());

                        #[cfg(feature = "random_subset_regression")]
                        if let Some(random_config) = &self.random_subset_regression {
                            let subsets =
                                random_subset_regression::Subsets::new(
                                    predictors,
                                    outcomes,
                                    random_config,
                                    &mut rng
                                );
                            if let Some(subsets) = subsets {
                                return $wrap(self.n_variable_optimization(
                                    |model| {
                                        let (predictors, outcomes) = subsets.next_subset();
                                        -utils::manhattan_distance(
                                            &$wrap(model),
                                            predictors,
                                            outcomes,
                                        )
                                    },
                                    (),
                                    &mut rng,
                                ));
                            }
                        }
                        $wrap(self.n_variable_optimization(
                            #[inline(always)]
                            |model| -utils::manhattan_distance(&$wrap(model), predictors, outcomes),
                            (),
                            &mut rng,
                        ))
                    }
                }
            )+
        };
    }
    #[cfg(feature = "binary_search_rng")]
    macro_rules! impl_estimator_trig {
        ($(
            $name:ident, $method:ident, $ret:ident, $wrap:expr,
        )+) => {
            $(
                impl $name for Options {
                    fn $method(&self, predictors: &[f64], outcomes: &[f64], max_frequency: f64) -> $ret {
                        use rand::SeedableRng;
                        let mut rng = rand_xorshift::XorShiftRng::from_rng(&mut rand::rng());

                        #[cfg(feature = "random_subset_regression")]
                        if let Some(random_config) = &self.random_subset_regression {
                            let subsets =
                                random_subset_regression::Subsets::new(
                                    predictors,
                                    outcomes,
                                    random_config,
                                    &mut rng
                                );
                            if let Some(subsets) = subsets {
                                return $wrap(self.n_variable_optimization(
                                    |model| {
                                        let (predictors, outcomes) = subsets.next_subset();
                                        -utils::trig_adjusted_manhattan_distance(
                                            &$wrap(model),
                                            model,
                                            predictors,
                                            outcomes,
                                            max_frequency,
                                        )
                                    },
                                    (),
                                    &mut rng,
                                ));
                            }
                        }
                        $wrap(self.n_variable_optimization(
                            #[inline(always)]
                            |model| {
                                -utils::trig_adjusted_manhattan_distance(
                                    &$wrap(model),
                                    model,
                                    predictors,
                                    outcomes,
                                    max_frequency
                                )
                            },
                            (),
                            &mut rng,
                        ))
                    }
                }
            )+
        };
    }

    #[cfg(feature = "binary_search_rng")]
    impl_estimator!(
        LinearEstimator,
        model_linear,
        LinearCoefficients,
        utils::wrap_linear,
        //
        PowerEstimator,
        model_power,
        PowerCoefficients,
        utils::wrap_power,
        //
        ExponentialEstimator,
        model_exponential,
        ExponentialCoefficients,
        utils::wrap_exponential,
        //
        LogisticEstimator,
        model_logistic,
        LogisticCoefficients,
        utils::wrap_logistic,
    );
    #[cfg(feature = "binary_search_rng")]
    impl_estimator_trig!(
        SineEstimator,
        model_sine,
        SineCoefficients,
        SineCoefficients::wrap,
        //
        CosineEstimator,
        model_cosine,
        CosineCoefficients,
        CosineCoefficients::wrap,
        //
        TangentEstimator,
        model_tangent,
        TangentCoefficients,
        TangentCoefficients::wrap,
        //
        SecantEstimator,
        model_secant,
        SecantCoefficients,
        SecantCoefficients::wrap,
        //
        CosecantEstimator,
        model_cosecant,
        CosecantCoefficients,
        CosecantCoefficients::wrap,
        //
        CotangentEstimator,
        model_cotangent,
        CotangentCoefficients,
        CotangentCoefficients::wrap,
    );
    #[cfg(feature = "binary_search_rng")]
    impl PolynomialEstimator for Options {
        fn model_polynomial(
            &self,
            predictors: &[f64],
            outcomes: &[f64],
            degree: usize,
        ) -> PolynomialCoefficients {
            use rand::SeedableRng;
            let mut rng = rand_xorshift::XorShiftRng::from_rng(&mut rand::rng());

            #[cfg(feature = "random_subset_regression")]
            if let Some(random_config) = &self.random_subset_regression {
                let subsets = random_subset_regression::Subsets::new(
                    predictors,
                    outcomes,
                    random_config,
                    &mut rng,
                );
                if let Some(subsets) = subsets {
                    return PolynomialCoefficients {
                        coefficients: (self.n_variable_optimization(
                            |model| {
                                let (predictors, outcomes) = subsets.next_subset();
                                -utils::manhattan_distance(
                                    &utils::BorrowedPolynomial(model),
                                    predictors,
                                    outcomes,
                                )
                            },
                            (degree + 1).into(),
                            &mut rng,
                        )),
                    };
                }
            }
            PolynomialCoefficients {
                coefficients: (self.n_variable_optimization(
                    #[inline(always)]
                    |model| {
                        -utils::manhattan_distance(
                            &utils::BorrowedPolynomial(model),
                            predictors,
                            outcomes,
                        )
                    },
                    (degree + 1).into(),
                    &mut rng,
                )),
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::super::*;
        use super::*;

        #[test]
        fn one_variable_regression() {
            let now = std::time::Instant::now();
            let values = super::Options::default()
                .max_precision()
                .n_variable_optimization_no_rng::<[f64; 1]>(
                    |s| (s[0] - 42.42424242424242).abs(),
                    (),
                );
            println!("{values:?} {:?}", now.elapsed());
        }
        #[test]
        #[cfg(feature = "binary_search_rng")]
        fn two_variable_regression() {
            let mut rng = rand::rng();
            let now = std::time::Instant::now();
            let x = [1.3, 4.7, 9.4];
            let y = [4., 5.3, 6.7];
            let v = Options::default().n_variable_optimization::<[f64; 2]>(
                |values| {
                    -utils::manhattan_distance(
                        &LinearCoefficients {
                            k: values[0],
                            m: values[1],
                        },
                        &x,
                        &y,
                    )
                },
                (),
                &mut rng,
            );
            let coeffs = LinearCoefficients { k: v[0], m: v[1] };
            println!(
                "{coeffs} R² {} {:?}",
                coeffs.determination_slice(&x, &y),
                now.elapsed()
            );
        }
        #[test]
        #[cfg(feature = "binary_search_rng")]
        fn second_degree_regression() {
            // init thread rng
            let _rng = rand::rng();
            let now = std::time::Instant::now();
            let x = [1.3, 4.7, 9.4];
            let y = [4., 5.3, 6.7];
            let coeffs = Options::default().model_polynomial(&x, &y, 2);
            println!(
                "{coeffs} R² {} {:?}",
                coeffs.determination_slice(&x, &y),
                now.elapsed()
            );
        }
        #[test]
        #[cfg(feature = "binary_search_rng")]
        fn two_variable_optimization() {
            use rand::SeedableRng;
            // init thread rng

            let mut rng = rand_xorshift::XorShiftRng::from_rng(&mut rand::rng());
            let now = std::time::Instant::now();
            let coeffs = Options::default()
                .max_precision()
                .n_variable_optimization::<[f64; 2]>(
                    |[v1, v2]| (v1 - 5.959).abs() + (v2 - (-234.234)).abs(),
                    (),
                    &mut rng,
                );
            println!("{coeffs:?} {:?}", now.elapsed());
        }
    }
}

/// Improves speed of regression by only taking a few points into account.
///
/// Randomly selects several sets of points which are checked. Works with [`binary_search`]
/// and can easily be expanded to [`spiral`] and [`gradient_descent`].
#[cfg(feature = "random_subset_regression")]
#[allow(dead_code)] // the user interacts with this through `binary_search`, so when that's
                    // disabled, this becomes dead code.
pub mod random_subset_regression {
    use rand::prelude::Distribution;
    use rand::Rng;
    /// Config for generation of subsets of points.
    /// See [`super::binary_search::Options::random_subset_regression`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Config {
        /// How many points each subset should contain.
        pub subset_length: usize,
        /// If [`Config::subset_length`] * this < the count of points, don't discard any points.
        /// If this is set to 1 (a panic will happen) the implementation would most likely pick
        /// several duplicate points.
        pub minimum_factor_of_length: usize,
        /// How many subsets to vary between
        pub subsets_count: usize,
    }
    impl Default for Config {
        fn default() -> Self {
            Self {
                subset_length: 100,
                minimum_factor_of_length: 4,
                subsets_count: 8,
            }
        }
    }
    pub(crate) struct Subsets {
        subsets: Vec<(Vec<f64>, Vec<f64>)>,
        i: std::rc::Rc<std::cell::RefCell<usize>>,
    }
    impl Subsets {
        pub(crate) fn new(
            x: &[f64],
            y: &[f64],
            config: &Config,
            rng: &mut impl Rng,
        ) -> Option<Self> {
            if x.len() != y.len() {
                return None;
            }
            if x.len() < config.subset_length * config.minimum_factor_of_length {
                return None;
            }
            if config.minimum_factor_of_length < 2 {
                eprintln!("random_subset_regression failed because configured `minimum_factor_of_length` is less than 2");
                return None;
            }
            if config.subsets_count < 2 {
                eprintln!(
                    "random_subset_regression failed because configured `subsets_count` is less than 2"
                );
                return None;
            }
            let distribution = rand::distr::Uniform::new(0, x.len()).unwrap();
            let subsets = (0..config.subsets_count)
                .map(|_| {
                    let mut new_x = Vec::with_capacity(config.subset_length);
                    let mut new_y = Vec::with_capacity(config.subset_length);
                    for _ in 0..config.subset_length {
                        let idx = distribution.sample(rng);
                        new_x.push(x[idx]);
                        new_y.push(y[idx]);
                    }
                    (new_x, new_y)
                })
                .collect();
            Some(Self {
                subsets,
                i: std::rc::Rc::new(std::cell::RefCell::new(0)),
            })
        }

        pub(crate) fn next_subset(&self) -> (&[f64], &[f64]) {
            let index = *self.i.borrow();
            let (predictors, outcomes) = &self.subsets[index];
            *self.i.borrow_mut() += 1;
            if index + 1 == self.subsets.len() {
                *self.i.borrow_mut() = 0;
            }
            (predictors, outcomes)
        }
    }
}

mod utils {
    use super::*;

    /// Like [`Determination::determination_slice`] but faster and more robust to outliers - values
    /// aren't squared (which increases the magnitude of outliers).
    ///
    /// `O(n)`
    #[inline(always)]
    pub(crate) fn manhattan_distance(
        model: &impl Predictive,
        predictors: &[f64],
        outcomes: &[f64],
    ) -> f64 {
        let mut error = 0.;
        for (predictor, outcome) in predictors.iter().copied().zip(outcomes.iter().copied()) {
            let predicted = model.predict_outcome(predictor);
            let length = (predicted - outcome).abs();
            error += length;
        }

        -error
    }

    pub(super) fn trig_adjusted_manhattan_distance(
        model: &impl Predictive,
        params: [f64; 3],
        predictors: &[f64],
        outcomes: &[f64],
        max_frequency: f64,
    ) -> f64 {
        let mut base = manhattan_distance(model, predictors, outcomes);
        if params[0].is_sign_negative()
            || params[1].is_sign_negative()
            || params[2].is_sign_negative()
        {
            base *= 10.;
        }
        if params[1] > max_frequency {
            base *= 10.;
        }
        base
    }

    #[inline(always)]
    pub(super) fn wrap_linear(a: [f64; 2]) -> LinearCoefficients {
        LinearCoefficients { k: a[1], m: a[0] }
    }
    #[inline(always)]
    pub(super) fn wrap_power(a: [f64; 2]) -> PowerCoefficients {
        PowerCoefficients {
            e: a[1],
            k: a[0],
            predictor_additive: 0.,
            outcome_additive: 0.,
        }
    }
    #[inline(always)]
    pub(super) fn wrap_exponential(a: [f64; 2]) -> ExponentialCoefficients {
        ExponentialCoefficients {
            b: a[1],
            k: a[0],
            predictor_additive: 0.,
            outcome_additive: 0.,
        }
    }
    #[inline(always)]
    pub(super) fn wrap_logistic(a: [f64; 3]) -> LogisticCoefficients {
        LogisticCoefficients {
            x0: a[0],
            l: a[1],
            k: a[2],
        }
    }
    pub(super) struct BorrowedPolynomial<'a>(pub(super) &'a [f64]);
    impl Predictive for BorrowedPolynomial<'_> {
        #[inline(always)]
        fn predict_outcome(&self, predictor: f64) -> f64 {
            match self.0.len() {
                0 => 0.,
                1 => self.0[0],
                2 => self.0[1] * predictor + self.0[0],
                3 => self.0[2] * predictor * predictor + self.0[1] * predictor + self.0[0],
                4 => {
                    let p2 = predictor * predictor;
                    self.0[3] * p2 * predictor + self.0[2] * p2 + self.0[1] * predictor + self.0[0]
                }
                _ => {
                    let mut out = 0.0;
                    let mut pred = 1.;
                    for coefficient in self.0.iter().copied() {
                        out += pred * coefficient;
                        pred *= predictor;
                    }
                    out
                }
            }
        }
    }
}
