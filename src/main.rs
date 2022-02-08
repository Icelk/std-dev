use std::io::{stdin, stdout, BufRead, Write};
use std::process::exit;
use std::str::FromStr;

fn f() {
    stdout().lock().flush().unwrap();
}

struct Output {
    s: f64,
    m: f64,
}
struct MedianOutput {
    median: f64,
    lower_quadrille: f64,
    higher_quadrille: f64,
}

fn std_dev(values: &[f64]) -> Output {
    let m = values.iter().copied().sum::<f64>() / values.len() as f64;
    let deviations = values.iter().map(|v| v - m);
    let squared_deviations = deviations.map(|v: f64| v.powi(2));
    let sum: f64 = squared_deviations.sum();
    let variance: f64 = sum / (values.len() - 1) as f64;
    Output {
        s: variance.sqrt(),
        m,
    }
}
fn median(values: &mut [f64]) -> MedianOutput {
    fn median(sorted_values: &[f64]) -> f64 {
        // even
        if sorted_values.len() % 2 == 0 {
            let b = sorted_values.len() / 2;
            let a = b - 1;
            (sorted_values[a] + sorted_values[b]) / 2.0
        } else {
            sorted_values[sorted_values.len() / 2]
        }
    }
    values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let even = values.len() % 2 == 0;
    MedianOutput {
        median: median(values),
        lower_quadrille: median(&values[..values.len() / 2]),
        higher_quadrille: median(&values[values.len() / 2 + if even { 0 } else { 1 }..]),
    }
}

fn parse<T: FromStr>(s: &str) -> Option<T> {
    if let Ok(v) = s.parse() {
        Some(v)
    } else {
        eprintln!("Failed to parse value {s:?}");
        None
    }
}

fn main() {
    loop {
        print!("> ");
        f();
        let mut s = String::new();
        stdin().lock().read_line(&mut s).unwrap();

        let mut iter: Vec<_> = s
            .split(',')
            .map(|s| s.split_whitespace())
            .flatten()
            .filter_map(|s| {
                Some(if let Some((v, count)) = s.split_once('x') {
                    let count = parse(count)?;
                    vec![v; count].into_iter().filter_map(parse)
                } else {
                    vec![s; 1].into_iter().filter_map(parse)
                })
            })
            .flatten()
            .collect();

        if iter.is_empty() {
            eprintln!("\nNo input. Exiting.");
            exit(0);
        }

        let out = std_dev(&iter);

        let median = median(&mut iter);

        println!(
            "Standard deviation: {}, mean: {}, median: {}, lower quadrille: {}, higher quadrille: {}",
            out.s, out.m, median.median, median.lower_quadrille, median.higher_quadrille,
        );
    }
}
