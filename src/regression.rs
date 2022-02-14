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
}
impl Predictive for PowerCoefficients {
    fn predict_outcome(&self, predictor: f64) -> f64 {
        self.k * predictor.powf(self.e)
    }
}
impl Display for PowerCoefficients {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} * x^{}", self.k, self.e)
    }
}

/// Fits a curve with the equation `y = a * x^b` (optionally with an additional subtractive term if
/// any input is negative).
///
/// # Panics
///
/// Panics if either `x` or `y` don't have the length `len`.
pub fn power(
    x: impl Iterator<Item = f64>,
    y: impl Iterator<Item = f64> + Clone,
    len: usize,
) -> PowerCoefficients {
    let x = x.map(|x| x.log2());
    let y = y.map(|y| y.log2());

    let coefficients = linear(x, y, len, 1);
    let k = 2.0_f64.powf(coefficients[0]);
    let e = coefficients[1];
    PowerCoefficients { k, e }
}

#[derive(Debug)]
pub struct ExponentialCoefficients {
    /// Constant
    k: f64,
    /// base
    b: f64,
}
impl Predictive for ExponentialCoefficients {
    fn predict_outcome(&self, predictor: f64) -> f64 {
        self.k * self.b.powf(predictor)
    }
}
impl Display for ExponentialCoefficients {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} * {}^x", self.k, self.b)
    }
}

/// Fits a curve with the equation `y = a * b^x` (optionally with an additional subtractive term if
/// any input is negative).
///
/// Also sometimes called "growth".
///
/// # Panics
///
/// Panics if either `x` or `y` don't have the length `len`.
/// `len` must be greater than 2.
pub fn exponential(
    x: impl Iterator<Item = f64>,
    y: impl Iterator<Item = f64> + Clone,
    len: usize,
) -> ExponentialCoefficients {
    assert!(len > 2);
    // let min = y.clone().map(crate::F64OrdHash).min().unwrap().0;
    // let add = (-min).max(0.0);
    let y = y.map(|y| y.log2());

    let coefficients = linear(x, y, len, 1);
    let k = 2.0_f64.powf(coefficients[0]);
    let b = 2.0_f64.powf(coefficients[1]);
    ExponentialCoefficients { k, b }
}
