use std::env;
use std::io::{stdin, BufRead};
use std::process::exit;
use std::str::FromStr;
use std::time::Instant;

pub mod lib;

fn parse<T: FromStr>(s: &str) -> Option<T> {
    if let Ok(v) = s.parse() {
        Some(v)
    } else {
        eprintln!("Failed to parse value {s:?}");
        None
    }
}

fn main() {
    let performance_print = env::var("DEBUG_PERFORMANCE")
        .ok()
        .map_or(false, |s| !s.trim().is_empty());

    #[cfg(feature = "prettier")]
    let tty = atty::is(atty::Stream::Stdin);

    loop {
        #[cfg(feature = "prettier")]
        if tty {
            use std::io::{stdout, Write};

            print!("> ");
            stdout().lock().flush().unwrap();
        }
        let mut s = String::new();

        stdin().lock().read_line(&mut s).unwrap();

        if s.trim().is_empty() {
            exit(0);
        }

        let now = Instant::now();

        let values: Vec<_> = s
            .split(',')
            .map(|s| s.split_whitespace())
            .flatten()
            .filter_map(|s| {
                Some(if let Some((v, count)) = s.split_once('x') {
                    let count = parse(count)?;
                    (parse(v)?, count)
                } else {
                    (parse(s)?, 1)
                })
            })
            .collect();

        if values.is_empty() {
            eprintln!("Only invalid input. Try again.");
            continue;
        }

        let values = lib::ClusterList::new(&values);

        if performance_print {
            println!("Parsing took {}µs", now.elapsed().as_micros());
        }
        let now = Instant::now();

        let mut values = values.optimize_values();

        if performance_print {
            println!("Optimizing input took {}µs", now.elapsed().as_micros());
        }
        let now = Instant::now();

        let mean = lib::std_dev(values.borrow());

        if performance_print {
            println!(
                "Standard deviation & mean took {}µs",
                now.elapsed().as_micros()
            );
        }
        let now = Instant::now();

        // Sort of clusters required.
        values.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let median = lib::median(lib::ClusterList::new(&values));

        if performance_print {
            println!("Median & quadrilles took {}µs", now.elapsed().as_micros());
        }

        println!(
            "Standard deviation: {}, mean: {}, median: {}{}{}",
            mean.standard_deviation,
            mean.mean,
            median.median,
            median
                .lower_quadrille
                .as_ref()
                .map_or("".into(), |quadrille| {
                    format!(", lower quadrille: {}", *quadrille)
                }),
            median
                .higher_quadrille
                .as_ref()
                .map_or("".into(), |quadrille| {
                    format!(", upper quadrille: {}", *quadrille)
                }),
        );
    }
}
