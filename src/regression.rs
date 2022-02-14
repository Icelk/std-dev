//! The `*Coefficients` structs all implement [`Predicative`] and [`Display`], which can be used to
//! view the equations.
//!
//! # Info on implementation
//!
//! [Linear regression](https://towardsdatascience.com/implementing-linear-and-polynomial-regression-from-scratch-f1e3d422e6b4)
//! [How the linear algebra works](https://medium.com/@andrew.chamberlain/the-linear-algebra-view-of-least-squares-regression-f67044b7f39b)
//!
//! ## Power & exponent
//!
//! I reverse the exponentiation to get a linear model. Then, I solve it using the method linked
//! above. Then, I transform the returned variables to fit the target model.
//!
//! Below, I've inserted the calculations for resolving what to do with the data.
//!
//! ### Linear (solved starting point)
//!
//! y=ax+b
//!
//! ### Power
//!
//! y=b * x^a
//!
//! lg(y) = lg(b * x^a)
//! lg(y) = lg(b) + a(lg x)
//!
//! Transform: y => lg (y), x => lg(x)
//!
//! When values found, take 10^b to get b and a is a
//!
//! ### Growth/exponential
//!
//! y=b * a^x
//!
//! lg(y) = lg(b * a^x)
//! lg(y) = lg(b) + x(lg a)
//!
//! Transform: y => lg (y), x => x
//!
//! When values found, take 10^b to get b and 10^a to get a

use std::fmt::{self, Display};
use std::ops::Deref;

pub trait Predictive {
    /// Calculates the predicted outcome of `predictor`.
    fn predict_outcome(&self, predictor: f64) -> f64;

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

            match order {
                0 => write!(f, "{coefficient}")?,
                1 => write!(f, "{coefficient}x")?,
                _ => write!(f, "{coefficient}x^({order})")?,
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
pub fn linear(
    x: impl Iterator<Item = f64>,
    y: impl Iterator<Item = f64>,
    len: usize,
    order: usize,
) -> PolynomialCoefficients {
    debug_assert!(
        order < len,
        "order + 1 must be less than or equal to len"
    );
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

#[derive(Debug)]
pub struct PowerCoefficients {
    /// Constant
    k: f64,
    /// exponent
    e: f64,
    predictor_additive: Option<f64>,
    outcome_additive: Option<f64>,
}
impl Predictive for PowerCoefficients {
    fn predict_outcome(&self, predictor: f64) -> f64 {
        self.k * (predictor + self.predictor_additive.unwrap_or(0.0)).powf(self.e)
            - self.outcome_additive.unwrap_or(0.0)
    }
}
impl Display for PowerCoefficients {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} * {x}^{}{}",
            self.k,
            self.e,
            if let Some(out) = self.outcome_additive {
                format!(" - {}", out)
            } else {
                String::new()
            },
            x = if let Some(pred) = self.predictor_additive {
                format!("(x + {})", pred)
            } else {
                "x".to_string()
            },
        )
    }
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
pub fn power(
    predictors: impl Iterator<Item = f64> + Clone,
    outcomes: impl Iterator<Item = f64> + Clone,
    len: usize,
) -> PowerCoefficients {
    assert!(len > 2);
    let outcome_min = outcomes.clone().map(crate::F64OrdHash).min().unwrap().0;
    let predictor_min = predictors.clone().map(crate::F64OrdHash).min().unwrap().0;
    power_given_min(predictors, outcomes, len, predictor_min, outcome_min)
}
/// Same as [`power`] without the [`Clone`] requirement for the iterators, but takes a min
/// value.
///
/// # Panics
///
/// See [`power`].
pub fn power_given_min(
    predictors: impl Iterator<Item = f64>,
    outcomes: impl Iterator<Item = f64>,
    len: usize,
    predictor_min: f64,
    outcome_min: f64,
) -> PowerCoefficients {
    assert!(len > 2);

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

    let predictors = predictors.map(|pred| (pred + predictor_additive.unwrap_or(0.0)).log2());
    let outcomes = outcomes.map(|y| (y + outcome_additive.unwrap_or(0.0)).log2());

    let coefficients = linear(predictors, outcomes, len, 1);
    let k = 2.0_f64.powf(coefficients[0]);
    let e = coefficients[1];
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
    k: f64,
    /// base
    b: f64,
    predictor_additive: Option<f64>,
    outcome_additive: Option<f64>,
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
        write!(
            f,
            "{} * {}^{x}{}",
            self.k,
            self.b,
            if let Some(out) = self.outcome_additive {
                format!(" - {}", out)
            } else {
                String::new()
            },
            x = if let Some(pred) = self.predictor_additive {
                format!("(x + {})", pred)
            } else {
                "x".to_string()
            },
        )
    }
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
// `TODO`: If we want cheesy, get the point of the "expected" mean point on the line, and offset it
// by that much. Is this even possible? Do we have to do iterators to get the best match? Seems
// like a hacky solution.
//
// Exclude the minimum value?
//
// Or, if this is just unfixible; we're loosing data when moving it randomly down. This just
// enables an approximation when the data is under 1.0.
pub fn exponential(
    predictors: impl Iterator<Item = f64> + Clone,
    outcomes: impl Iterator<Item = f64> + Clone,
    len: usize,
) -> ExponentialCoefficients {
    assert!(len > 2);
    let predictor_min = predictors.clone().map(crate::F64OrdHash).min().unwrap().0;
    let outcome_min = outcomes.clone().map(crate::F64OrdHash).min().unwrap().0;
    exponential_given_min(predictors, outcomes, len, predictor_min, outcome_min)
}
/// Same as [`exponential`] without the [`Clone`] requirement for the iterators, but takes a min
/// value.
///
/// # Panics
///
/// See [`exponential`].
pub fn exponential_given_min(
    predictors: impl Iterator<Item = f64>,
    outcomes: impl Iterator<Item = f64>,
    len: usize,
    predictor_min: f64,
    outcome_min: f64,
) -> ExponentialCoefficients {
    assert!(len > 2);

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

    let outcomes = outcomes.map(|y| (y + outcome_additive.unwrap_or(0.0)).log2());
    let predictors = predictors.map(|pred| pred + predictor_additive.unwrap_or(0.0));

    let coefficients = linear(predictors, outcomes, len, 1);
    let k = 2.0_f64.powf(coefficients[0]);
    let b = 2.0_f64.powf(coefficients[1]);
    ExponentialCoefficients {
        k,
        b,
        predictor_additive,
        outcome_additive,
    }
}
