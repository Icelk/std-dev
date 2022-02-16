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
        x: impl Iterator<Item = f64>,
        y: impl Iterator<Item = f64>,
        len: usize,
        order: usize,
    ) -> PolynomialCoefficients {
        debug_assert!(order < len, "order + 1 must be less than or equal to len");
        // `TODO`: Save a copy of the iterator, then iterate over it in the from_fn call.
        // When a new column is began, start again.
        let x: Vec<_> = x.collect();
        let design = nalgebra::DMatrix::from_fn(len, order + 1, |row: usize, column: usize| {
            if column == 0 {
                1.0
            } else if column == 1 {
                x[row]
            } else {
                x[row].powi(column as _)
            }
        });

        let t = design.transpose();
        let y = nalgebra::DMatrix::from_iterator(len, 1, y);
        let result = ((&t * &design).try_inverse().unwrap() * &t) * y;

        PolynomialCoefficients {
            coefficients: result.iter().copied().collect(),
        }
    }
}
