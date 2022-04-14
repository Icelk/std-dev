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

use std::fmt::{self, Display};
use std::ops::Deref;

#[doc(inline)]
pub use models::*;

pub use derived::{exponential, power, ExponentialCoefficients, PowerCoefficients};
#[cfg(feature = "ols")]
pub use derived::{exponential_ols, power_ols};
#[cfg(feature = "ols")]
pub use ols::{LinearOls, PolynomialOls};
pub use spiral::estimators::*;
pub use theil_sen::{LinearTheilSen, PolynomialTheilSen};

trait Model: Predictive + Display {}
impl<T: Predictive + Display> Model for T {}

/// Generic model. This enables easily handling results from several models.
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
        (&**self).predict_outcome(predictor)
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

        1.0 - (res / tot)
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

pub mod models {
    use std::f64::consts::E;

    use super::*;

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
                    _ => write!(f, "{coefficient:.0$}x^{degree:.0$}", p)?,
                }

                first = false;
            }
            Ok(())
        }
    }
    impl PolynomialCoefficients {
        #[inline(always)]
        fn naive_predict(&self, predictor: f64) -> f64 {
            let mut out = 0.0;
            for (degree, coefficient) in self.coefficients.iter().copied().enumerate() {
                out += predictor.powi(degree as i32) * coefficient;
            }
            out
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
                "{:.3$} / (1 + e^(-{:.3$}(x - {:.3$})))",
                self.l, self.k, self.x0, p
            )
        }
    }

    /// Implemented by all methods yielding a linear 2 variable regression (a line).
    pub trait LinearEstimator {
        /// Model the [`LinearCoefficients`] from `predictors` and `outcomes`.
        ///
        /// # Panics
        ///
        /// The two slices must have the same length.
        fn model(&self, predictors: &[f64], outcomes: &[f64]) -> LinearCoefficients;
        /// Put this estimator in a box.
        /// This is useful for conditionally choosing different estimators.
        fn boxed(self) -> Box<dyn LinearEstimator>
        where
            Self: Sized + 'static,
        {
            Box::new(self)
        }
    }
    impl<T: LinearEstimator + ?Sized> LinearEstimator for &T {
        fn model(&self, predictors: &[f64], outcomes: &[f64]) -> LinearCoefficients {
            (**self).model(predictors, outcomes)
        }
    }
    /// Implemented by all methods yielding a polynomial regression.
    pub trait PolynomialEstimator {
        /// Model the [`PolynomialCoefficients`] from `predictors` and `outcomes` with `degree`.
        ///
        /// # Panics
        ///
        /// The two slices must have the same length.
        fn model(
            &self,
            predictors: &[f64],
            outcomes: &[f64],
            degree: usize,
        ) -> PolynomialCoefficients;
        /// Put this estimator in a box.
        /// This is useful for conditionally choosing different estimators.
        fn boxed(self) -> Box<dyn PolynomialEstimator>
        where
            Self: Sized + 'static,
        {
            Box::new(self)
        }
    }
    /// Implemented by all methods yielding a logistic regression.
    pub trait LogisticEstimator {
        /// Model the [`LogisticCoefficients`] from `predictors` and `outcomes`.
        ///
        /// # Panics
        ///
        /// The two slices must have the same length.
        fn model(&self, predictors: &[f64], outcomes: &[f64]) -> LogisticCoefficients;
        /// Put this estimator in a box.
        /// This is useful for conditionally choosing different estimators.
        fn boxed(self) -> Box<dyn LogisticEstimator>
        where
            Self: Sized + 'static,
        {
            Box::new(self)
        }
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
/// distance_from_integer < 0.15 && -2.5 <= exponent <= 3.5`
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

    let linear = linear_estimator.model(predictors, outcomes);
    update_best!(linear, e, e + LINEAR_BUMP);
    // UNWRAP: We just set it, at least there's a linear.
    best.unwrap().0
}
/// Convenience function for [`best_fit`] using [`LinearOls`].
#[cfg(feature = "ols")]
pub fn best_fit_ols(predictors: &[f64], outcomes: &[f64]) -> DynModel {
    best_fit(predictors, outcomes, &LinearOls)
}

/// Estimators derived from others, usual [`LinearEstimator`].
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
    #[cfg(feature = "ols")]
    pub fn power_ols(predictors: &mut [f64], outcomes: &mut [f64]) -> PowerCoefficients {
        power(predictors, outcomes, &LinearOls)
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
    #[cfg(feature = "ols")]
    pub fn exponential_ols(
        predictors: &mut [f64],
        outcomes: &mut [f64],
    ) -> ExponentialCoefficients {
        exponential(predictors, outcomes, &LinearOls)
    }
    /// Fits a curve with the equation `y = a * b^x` (optionally with an additional subtractive term if
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
            rug::Float::parse_radix(s, radix as i32)
                .map(|f| Self(rug::Float::with_val(HARDCODED_PRECISION, f)))
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
        intermediary: DMatrix<f64>,
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
                intermediary: DMatrix::zeros(0, 0),
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
            self.intermediary.resize_mut(columns, columns, 0.);
            self.result.resize_mut(columns, 1, 0.);
            self.len = len;
            self.degree = degree;
        }
    }
    thread_local! {static RUNTIME: RefCell<RuntimeMatrices> = RefCell::new(RuntimeMatrices::new());}

    /// `O(n)`
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
    /// `O(n*degree)`, which when using a set `degree` becomes `O(n)`
    pub struct PolynomialOls;
    impl PolynomialEstimator for PolynomialOls {
        fn model(
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
                    intermediary,
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

                transposed.mul_to(design, intermediary);
                if !intermediary.try_inverse_mut() {
                    let im = std::mem::replace(intermediary, DMatrix::zeros(0, 0));
                    let pseudo_inverse = im.pseudo_inverse(1e-8).unwrap();
                    *intermediary = pseudo_inverse;
                }
                *intermediary *= &*transposed;
                intermediary.mul_to(outcomes_matrix, result);

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
            // let precision = arbitrary_linear_algebra::HARDCODED_PRECISION;
            // let zero_limit = rug::Float::with_val(arbitrary_linear_algebra::HARDCODED_PRECISION, 1e-17f64).into();
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

    pub struct PermutationIterBuffer<T> {
        buf: Vec<(T, T)>,
    }
    impl<T> Deref for PermutationIterBuffer<T> {
        type Target = [(T, T)];
        fn deref(&self) -> &Self::Target {
            &self.buf
        }
    }
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
    impl<'a, T: Copy + Debug> Iterator for PermutationIter<'a, T> {
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
        fn model(&self, predictors: &[f64], outcomes: &[f64]) -> LinearCoefficients {
            slow_linear(predictors, outcomes)
        }
    }
    /// Polynomial estimation using the Theil-Sen estimatior. Very slow and should probably not be
    /// used.
    /// `O(n^degree)`
    pub struct PolynomialTheilSen;
    impl PolynomialEstimator for PolynomialTheilSen {
        #[inline]
        fn model(
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
            #[derive(Debug, Clone, Copy)]
            struct CmpFirst<T, V>(T, V);
            impl<T: PartialEq, V> PartialEq for CmpFirst<T, V> {
                #[inline]
                fn eq(&self, other: &Self) -> bool {
                    self.0.eq(&other.0)
                }
            }
            impl<T: PartialEq + Eq, V> Eq for CmpFirst<T, V> {}
            impl<T: PartialOrd, V> PartialOrd for CmpFirst<T, V> {
                #[inline]
                fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                    self.0.partial_cmp(&other.0)
                }
            }
            impl<T: Ord, V> Ord for CmpFirst<T, V> {
                #[inline]
                fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                    self.0.cmp(&other.0)
                }
            }

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
        fn permutations_eq_2() {
            use rand::Rng;

            let mut s1 = [0.0; 20];
            let mut s2 = [0.0; 20];

            let mut rng = rand::thread_rng();
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
/// # Advantages
///
/// You supply a `fitness_function` to all functions which tells the algorithm which lines are
/// good. The magnitude is irrelevant, only order is considered. The algorithm tries to *minimize*
/// the returned value. **This allows you to choose the desired properties of resulting
/// line/polynomial, without checking all possible values.**
///
/// # Caveats
///
/// When using [`PolynomialSpiralManhattanDistance`] it only supports degree 1&2.
/// See [details](#details) for more info on this.
///
/// The sampling technique means this might miss the right point to close in on. Therefore, I
/// highly recommend using the [higher quality options](spiral::Options::new).
///
/// ## Robustness
///
/// Since this uses a fitness function, the robustness is determined by that. Using the "default"
/// [`spiral::manhattan_distance`] (e.g. [`LinearSpiralManhattanDistance`],
/// [`PolynomialSpiralManhattanDistance`]) gives good results.
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

    /// Like [`Determination::determination_slice`] but faster and more robust to outliers - values
    /// aren't squared, which increases the magnitude of outliers.
    ///
    /// `O(n)`
    pub fn manhattan_distance(
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

        0.0 - error
    }

    pub mod estimators {
        use super::*;

        struct SecondDegreePolynomial([f64; 3]);
        impl Predictive for SecondDegreePolynomial {
            fn predict_outcome(&self, predictor: f64) -> f64 {
                self.0[0] + self.0[1] * predictor + self.0[2] * predictor * predictor
            }
        }
        #[inline(always)]
        fn wrap_linear(a: [f64; 2]) -> LinearCoefficients {
            LinearCoefficients { k: a[1], m: a[0] }
        }

        /// [`LinearEstimator`] for the spiral estimator using the fast and robust
        /// [`manhattan_distance`] fitness function.
        /// `O(n)`
        pub struct LinearSpiralManhattanDistance(pub Options);
        impl LinearEstimator for LinearSpiralManhattanDistance {
            fn model(&self, predictors: &[f64], outcomes: &[f64]) -> LinearCoefficients {
                wrap_linear(two_variable_optimization(
                    #[inline(always)]
                    |model| manhattan_distance(&wrap_linear(model), predictors, outcomes),
                    self.0.clone(),
                ))
            }
        }
        /// [`LinearEstimator`] for the spiral estimator using a fitness function and [`Options`]
        /// provided by you.
        /// `O(fitness function)`
        pub struct LinearSpiral<F: Fn(&LinearCoefficients, &[f64], &[f64]) -> f64>(
            pub F,
            pub Options,
        );
        impl<F: Fn(&LinearCoefficients, &[f64], &[f64]) -> f64> LinearEstimator for LinearSpiral<F> {
            fn model(&self, predictors: &[f64], outcomes: &[f64]) -> LinearCoefficients {
                wrap_linear(two_variable_optimization(
                    #[inline(always)]
                    |model| self.0(&wrap_linear(model), predictors, outcomes),
                    self.1.clone(),
                ))
            }
        }
        /// [`PolynomialEstimator`] for the spiral estimator using the fast and robust
        /// [`manhattan_distance`] fitness function.
        /// `O(n)`
        ///
        /// **IMPORTANT**: only supports degrees of 1&2.
        pub struct PolynomialSpiralManhattanDistance(pub Options);
        impl PolynomialEstimator for PolynomialSpiralManhattanDistance {
            fn model(
                &self,
                predictors: &[f64],
                outcomes: &[f64],
                degree: usize,
            ) -> PolynomialCoefficients {
                match degree {
                    1 => wrap_linear(two_variable_optimization(
                        #[inline(always)]
                        |model| manhattan_distance(&wrap_linear(model), predictors, outcomes),
                        self.0.clone(),
                    ))
                    .into(),
                    2 => three_variable_optimization(
                        #[inline(always)]
                        |model| {
                            manhattan_distance(&SecondDegreePolynomial(model), predictors, outcomes)
                        },
                        self.0.clone(),
                    )
                    .into(),
                    _ => panic!("unsupported degree for polynomial spiral. Supports 1,2."),
                }
            }
        }

        pub struct LogisticSpiralManhattanDistance(pub Options);
        impl LogisticEstimator for LogisticSpiralManhattanDistance {
            fn model(&self, predictors: &[f64], outcomes: &[f64]) -> LogisticCoefficients {
                #[inline(always)]
                fn wrap(a: [f64; 3]) -> LogisticCoefficients {
                    LogisticCoefficients {
                        x0: a[0],
                        l: a[1],
                        k: a[2],
                    }
                }
                wrap(three_variable_optimization(
                    #[inline(always)]
                    |model| manhattan_distance(&wrap(model), predictors, outcomes),
                    self.0.clone(),
                ))
            }
        }
    }

    /// Options for the spiral.
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
        pub exponent_coefficient: f64,
        pub angle_coefficient: f64,
        pub num_lockon: usize,
        pub samples_per_rotation: f64,
        pub range: Range<f64>,
        /// The turns of the 3d spiral when using [`second_degree_polynomial`].
        /// Frequency of turns per sphere. Is unnecessary to turn up when
        /// [`Self::samples_per_rotation`] is low.
        pub turns: f64,
    }
    impl Options {
        /// Default, with a good trade-off between speed and precision.
        pub fn new() -> Self {
            Self {
                exponent_coefficient: 10.,
                angle_coefficient: 0.07,
                num_lockon: 16,
                // these are odd values to avoid repeating the same angle on multiple turns
                samples_per_rotation: 47.,
                range: (-6. * TAU)..(6. * TAU),
                turns: 16.,
            }
        }
        /// About 4x faster than [`Self::new`].
        pub fn fast() -> Self {
            Self {
                exponent_coefficient: 10.,
                angle_coefficient: 0.13,
                num_lockon: 8,
                samples_per_rotation: 37.,
                range: (-4. * TAU)..(4. * TAU),
                turns: 16.,
            }
        }
        /// About 4x faster than [`Self::fast`].
        /// About 3x slower than analytical OLS solution (which provides way less flexibility).
        pub fn faster() -> Self {
            Self {
                exponent_coefficient: 10.,
                angle_coefficient: 0.23,
                num_lockon: 8,
                samples_per_rotation: 19.,
                range: (-2. * TAU)..(4. * TAU),
                turns: 12.,
            }
        }
        /// About 4x slower than [`Self::new`].
        pub fn precise() -> Self {
            Self {
                exponent_coefficient: 10.,
                angle_coefficient: 0.05,
                num_lockon: 32,
                samples_per_rotation: 115.,
                range: (-6. * TAU)..(6. * TAU),
                turns: 16.,
            }
        }
    }
    impl Default for Options {
        fn default() -> Self {
            Self::new()
        }
    }

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
            mut exponent_coefficient,
            angle_coefficient,
            num_lockon,
            samples_per_rotation,
            range,
            turns: _,
        } = options;
        let advance = TAU / samples_per_rotation;
        let mut best = ((f64::MIN, 1.), [0.; 2]);
        let mut last_best = f64::MIN;

        for i in 0..num_lockon {
            let mut theta = range.start;
            while theta < range.end {
                let a = exponent_coefficient * E.powf(angle_coefficient * theta) * theta.cos()
                    + best.1[0];
                let b = exponent_coefficient * E.powf(angle_coefficient * theta) * theta.sin()
                    + best.1[1];

                let fitness = fitness_function([a, b]);
                if fitness > best.0 .0 {
                    best = ((fitness, E.powf(angle_coefficient * theta)), [a, b]);
                }

                theta += advance;
            }
            // If the best didn't change, we aren't going to find better results.
            if last_best == best.0 .0 && i != 0 {
                return best.1;
            }
            last_best = best.0 .0;
            // don't go full out, "ease" this into several approaching steps with .sqrt to avoid
            // overcorrection.
            exponent_coefficient *= best.0 .1.sqrt();

            // uncomment line below to see how the coefficient changes and the current best.
            // reveals how it shrinks the spiral, and sometimes enlarges it to later zoom in (if
            // enough iterations are allowed)
            //
            // println!("Iteration complete. exponent_coefficient: {exponent_coefficient} best: {best:?}");
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
            mut exponent_coefficient,
            angle_coefficient,
            num_lockon,
            samples_per_rotation,
            range,
            turns,
        } = options;
        let advance = TAU / samples_per_rotation;

        let polynomial_identity = [0.; 3];
        let mut best = ((f64::MIN, 1.), polynomial_identity);
        let mut last_best = f64::MIN;

        let mut current_coefficients = polynomial_identity;

        for i in 0..num_lockon {
            let mut theta = range.start;
            while theta < range.end {
                let r = E.powf(theta * angle_coefficient) * exponent_coefficient;
                let a = r * theta.sin() * (turns * theta).cos() + best.1[0];
                let b = r * theta.sin() * (turns * theta).sin() + best.1[1];
                let c = r * theta.cos() + best.1[2];

                current_coefficients[0] = a;
                current_coefficients[1] = b;
                current_coefficients[2] = c;

                let fitness = fitness_function(current_coefficients);
                if fitness > best.0 .0 {
                    best = ((fitness, r / exponent_coefficient), current_coefficients);
                }

                theta += advance;
            }
            if last_best == best.0 .0 && i != 0 {
                return best.1;
            }
            last_best = best.0 .0;
            exponent_coefficient *= best.0 .1.sqrt();
        }
        best.1
    }
}
