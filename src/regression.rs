use std::fmt::{self, Display};

/// The length of the inner vector is `order + 1`.
///
/// The inner list is in order of smallest exponent to largest: `[0, 2, 1]` means `y = 1x² + 2x + 0`.
pub struct PolynomialCoefficients {
    coefficients: Vec<f64>,
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
                _ => write!(f, "{coefficient}x^{order}")?,
            }

            first = false;
        }
        Ok(())
    }
}

/// # Panics
///
/// Panics if either `x` or `y` don't have the length `len`.
///
/// Also panics if `order + 1 > len`.
pub fn linear_regression(
    x: impl Iterator<Item = f64>,
    y: impl Iterator<Item = f64>,
    len: usize,
    order: usize,
) -> PolynomialCoefficients {
    let x: Vec<_> = x.collect();
    let design = nalgebra::DMatrix::from_fn(len, order + 1, |row: usize, column: usize| {
        if column == 0 {
            1.0
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
