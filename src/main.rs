use std::collections::HashMap;
use std::io::{stdin, stdout, BufRead, Write};
use std::process::exit;
use std::str::FromStr;
use std::time::Instant;
use std::{env, hash};

pub mod lib;

fn f() {
    stdout().lock().flush().unwrap();
}

fn parse<T: FromStr>(s: &str) -> Option<T> {
    if let Ok(v) = s.parse() {
        Some(v)
    } else {
        eprintln!("Failed to parse value {s:?}");
        None
    }
}

#[derive(Debug, Copy, Clone)]
struct F64Hash(f64);

impl F64Hash {
    fn key(&self) -> u64 {
        self.0.to_bits()
    }
}

impl hash::Hash for F64Hash {
    fn hash<H>(&self, state: &mut H)
    where
        H: hash::Hasher,
    {
        self.key().hash(state)
    }
}

impl PartialEq for F64Hash {
    fn eq(&self, other: &F64Hash) -> bool {
        self.key() == other.key()
    }
}

impl Eq for F64Hash {}

fn main() {
    let performance_print = env::var("DEBUG_PERFORMANCE")
        .ok()
        .map_or(false, |s| !s.trim().is_empty());

    let tty = atty::is(atty::Stream::Stdin);

    loop {
        if tty {
            print!("> ");
            f();
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

        if performance_print {
            println!("Parsing took {}µs", now.elapsed().as_micros());
        }
        let now = Instant::now();

        let mut collected = HashMap::with_capacity(16);
        for (v, count) in &values {
            let c = collected.entry(F64Hash(*v)).or_insert(0);
            *c += count;
        }
        let mut values: Vec<_> = collected.into_iter().map(|(f, c)| (f.0, c)).collect();

        if performance_print {
            println!("Optimizing input took {}µs", now.elapsed().as_micros());
        }
        let now = Instant::now();

        let mean = lib::std_dev(lib::ClusterList::new(&values));

        if performance_print {
            println!(
                "Standard deviation & mean took {}µs",
                now.elapsed().as_micros()
            );
        }
        let now = Instant::now();

        values.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let median = lib::median(lib::ClusterList::new(&values));

        if performance_print {
            println!("Median & quadrilles took {}µs", now.elapsed().as_micros());
        }

        println!(
            "Standard deviation: {}, mean: {}, median: {}, lower quadrille: {:?}, higher quadrille: {:?}",
            mean.standard_deviation, mean.mean, median.median, median.lower_quadrille, median.higher_quadrille,
        );
    }
}
