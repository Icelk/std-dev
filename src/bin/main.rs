use std::env;
use std::io::{stdin, BufRead};
use std::process::exit;
use std::str::FromStr;
use std::time::Instant;

use clap::Arg;

#[cfg(feature = "regression")]
use std::fmt::Display;
#[cfg(feature = "regression")]
use std::io::Write;
pub use std_dev;
#[cfg(feature = "regression")]
use std_dev::regression::{
    Determination, DynModel, LinearEstimator, PolynomialEstimator, Predictive,
};

fn parse<T: FromStr>(s: &str) -> Option<T> {
    if let Ok(v) = s.parse() {
        Some(v)
    } else {
        eprintln!("Failed to parse value {s:?}");
        None
    }
}
#[derive(Debug)]
enum InputValue {
    Count(Vec<std_dev::Cluster>),
    List(Vec<Vec<f64>>),
}
impl InputValue {
    fn is_empty(&self) -> bool {
        match self {
            Self::Count(count) => count.is_empty(),
            Self::List(l) => l.is_empty(),
        }
    }
}

fn input(
    _is_tty: bool,
    debug_performance: bool,
    multiline: bool,
    _last_prompt: &mut Instant,
) -> Option<InputValue> {
    #[cfg(feature = "pretty")]
    {
        if _is_tty {
            use std::io::stdout;

            if multiline {
                print!("multiline > ");
            } else {
                print!("> ")
            }
            stdout().lock().flush().unwrap();
        }
        *_last_prompt = Instant::now();
    }
    let mut s = String::new();

    let mut now = Instant::now();

    let values = if multiline {
        let mut values = Vec::with_capacity(8);
        let stdin = stdin();
        let stdin = stdin.lock().lines();
        let mut lines = 0_usize;
        for line in stdin {
            if lines == 0 {
                now = Instant::now();
            }
            lines += 1;
            let line = line.unwrap();
            if line.trim().is_empty() {
                break;
            }
            let mut current = Vec::with_capacity(2);
            for segment in line
                .split(',')
                .map(|s| s.trim().split_whitespace())
                .flatten()
            {
                let f = parse(segment.trim());
                if let Some(f) = f {
                    current.push(f)
                }
            }
            values.push(current);
            #[cfg(feature = "pretty")]
            {
                if _is_tty && _last_prompt.elapsed().as_millis() > 10 {
                    use std::io::stdout;

                    let next = values.len() + 1;
                    print!("{next} > ");
                    stdout().lock().flush().unwrap();
                }
                *_last_prompt = Instant::now();
            }
        }
        if lines <= 1 {
            exit(0);
        }
        InputValue::List(values)
    } else {
        stdin().lock().read_line(&mut s).unwrap();
        now = Instant::now();

        if s.trim().is_empty() {
            exit(0);
        }

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
        InputValue::Count(values)
    };

    if values.is_empty() {
        eprintln!("Only invalid input. Try again.");
        return None;
    }

    if debug_performance {
        println!("Parsing/reading input took {}µs", now.elapsed().as_micros());
    }
    Some(values)
}

#[cfg(feature = "regression")]
fn print_regression(
    regression: &(impl std_dev::regression::Predictive + Display),
    x: impl Iterator<Item = f64>,
    y: impl Iterator<Item = f64> + Clone,
    len: usize,
    precision: Option<usize>,
) {
    if let Some(precision) = precision {
        println!(
            "Determination: {:.1$}, Predicted equation: {regression:.1$}",
            regression.determination(x, y, len),
            precision,
        );
    } else {
        println!(
            "Determination: {:.4}, Predicted equation: {regression}",
            regression.determination(x, y, len),
        );
    }
}

fn main() {
    let mut app = clap::command!();

    app = app
        .arg(Arg::new("debug-performance").short('p').long("debug-performance"))
        .arg(Arg::new("multiline")
            .short('m')
            .long("multiline")
            .help("Accept multiple lines as one input. Two consecutive newlines is treated as the series separator. When not doing regression analysis the second 'column' is the count of the first. Acts more like CSV.")
        )
        .arg(Arg::new("precision")
            .short('n')
            .long("precision")
            .help("Sets the precision of the output. When this isn't set, Rust decides how many digits to print. \
                  The determination will be 4 decimal places long. When this is set, all numbers are rounded.")
            .takes_value(true)
            .validator(|v| v.parse::<usize>().map_err(|_| "precision needs to be a positive integer".to_owned()))
        );

    #[cfg(feature = "regression")]
    {
        app = app.subcommand(clap::Command::new("regression")
            .about("Find a equation which describes the input data. Tries to automatically determine the model if no arguments specifying it are provided. \
            Predictors are the independent values (usually denoted `x`) from which we want a equation to get the \
            outcomes - the dependant variables, usually `y` or `f(x)`.")
            .group(clap::ArgGroup::new("model")
                   .arg("degree")
                   .arg("linear")
                   .arg("power")
                   .arg("exponential")
            )
            .arg(Arg::new("degree")
                .short('d')
                .long("degree")
                .help("Degree of polynomial.")
                .takes_value(true)
                .validator(|o| o.parse::<usize>().map_err(|_| "Degree must be an integer".to_owned()))
            )
            .arg(Arg::new("linear")
                 .short('l')
                 .long("linear")
                 .help("Tries to fit a line to the provided data.")
            )
            .arg(Arg::new("power")
                .short('p')
                .long("power")
                .help("Tries to fit a curve defined by the equation `a * x^b` to the data.\
                If any of the predictors are below 1, x becomes (x+c), where c is an offset to the predictors. This is due to the arithmetic issue of taking the log of negative numbers and 0.\
                A negative addition term will be appended if any of the outcomes are below 1.")
            )
            .arg(Arg::new("exponential")
                .short('e')
                .visible_alias("growth")
                .long("exponential")
                .help("Tries to fit a curve defined by the equation `a * b^x` to the data. \
                If any of the predictors are below 1, x becomes (x+c), where c is an offset to the predictors. This is due to the arithmetic issue of taking the log of negative numbers and 0. \
                A negative addition term will be appended if any of the outcomes are below 1.")
            )
            .arg(Arg::new("theil_sen")
                .long("theil-sen")
                .short('t')
                .help("Use the Theil-Sen estimator instead of OLS for linear and polynomial (slow). Not applied when -l or -d aren't supplied.")
            )
            .arg(Arg::new("plot")
                .long("plot")
                .help("Plots the regression and input variables in a SVG.")
            )
            .arg(Arg::new("plot_filename")
                .long("plot-out")
                .help("File name (without extension) for SVG plot.")
                .takes_value(true)
                .requires("plot")
            )
            .arg(Arg::new("plot_samples")
                .long("plot-samples")
                .help("Count of sample points when drawing the curve. Always set to 2 for linear regressions.")
                .takes_value(true)
                .requires("plot")
            )
            .arg(Arg::new("plot_title")
                .long("plot-title")
                .help("Title of plot.")
                .takes_value(true)
                .requires("plot")
            )
            .arg(Arg::new("plot_x_axis")
                .long("plot-axis-x")
                .help("Name of x axis of plot (the first column of data).")
                .takes_value(true)
                .requires("plot")
            )
            .arg(Arg::new("plot_y_axis")
                .long("plot-axis-y")
                .help("Name of y axis of plot (the second column of data).")
                .takes_value(true)
                .requires("plot")
            )
        );
    }

    let matches = app.get_matches();

    let debug_performance = env::var("DEBUG_PERFORMANCE").ok().map_or_else(
        || matches.is_present("debug-performance"),
        |s| !s.trim().is_empty(),
    );

    #[cfg(feature = "pretty")]
    let tty = atty::is(atty::Stream::Stdin);
    #[cfg(not(feature = "pretty"))]
    let tty = false;

    let mut last_prompt = Instant::now();

    'main: loop {
        let multiline = {
            matches.is_present("multiline") || matches.subcommand_matches("regression").is_some()
        };
        let input = if let Some(i) = input(tty, debug_performance, multiline, &mut last_prompt) {
            i
        } else {
            continue;
        };

        match matches.subcommand() {
            #[cfg(feature = "regression")]
            Some(("regression", config)) => {
                let values = {
                    match input {
                        InputValue::Count(_) => {
                            eprintln!("You cannot use `<value>x<count>` notation for point entry");
                            continue 'main;
                        }
                        InputValue::List(list) => {
                            // Higher dimensional analysis?:
                            // let dimension = list.first().unwrap().len();
                            let dimension = 2;

                            for item in &list {
                                if item.len() != dimension {
                                    eprintln!("Expected {dimension} values per line.");
                                    continue 'main;
                                }
                            }
                            list
                        }
                    }
                };

                let len = values.len();
                let x_iter = values.iter().map(|d| d[0]);
                let y_iter = values.iter().map(|d| d[1]);
                let mut x: Vec<f64> = x_iter.clone().collect();
                let mut y: Vec<f64> = y_iter.clone().collect();

                let now = Instant::now();

                let model: DynModel =
                    if config.is_present("power") || config.is_present("exponential") {
                        if config.is_present("power") {
                            let coefficients = std_dev::regression::power_ols(&mut x, &mut y);
                            DynModel::new(coefficients)
                        } else {
                            assert!(config.is_present("exponential"));

                            let coefficients = std_dev::regression::exponential_ols(&mut x, &mut y);
                            DynModel::new(coefficients)
                        }
                    } else if config.is_present("linear") || config.is_present("degree") {
                        let degree = {
                            if let Ok(degree) = config.value_of_t("degree") {
                                degree
                            } else {
                                1
                            }
                        };
                        if degree + 1 > len {
                            eprintln!("Degree of polynomial is too large; add more datapoints.");
                            continue 'main;
                        }

                        if degree == 1 {
                            let estimator = {
                                if config.is_present("theil_sen") {
                                    std_dev::regression::LinearTheilSen.boxed()
                                } else {
                                    std_dev::regression::LinearOls.boxed()
                                }
                            };

                            estimator.model(&x, &y).boxed()
                        } else {
                            let estimator = {
                                if config.is_present("theil_sen") {
                                    std_dev::regression::PolynomialTheilSen.boxed()
                                } else {
                                    std_dev::regression::PolynomialOls.boxed()
                                }
                            };

                            estimator.model(&x, &y, degree).boxed()
                        }
                    } else {
                        let mut predictors: Vec<f64> = x_iter.clone().collect();
                        let mut outcomes: Vec<f64> = y_iter.clone().collect();
                        std_dev::regression::best_fit_ols(&mut predictors, &mut outcomes)
                    };

                let p = matches
                    .value_of("precision")
                    .map(|s| s.parse::<usize>().expect("we check this using clap"));

                print_regression(&model, x_iter.clone(), y_iter.clone(), len, p);

                if debug_performance {
                    println!("Regression analysis took {}µs.", now.elapsed().as_micros());
                }

                if config.is_present("plot") {
                    let now = Instant::now();

                    let mut num_samples = config
                        .value_of("plot_samples")
                        .map(|s| {
                            if let Ok(i) = s.parse() {
                                i
                            } else {
                                eprintln!("You must supply an integer to `plot-samples`.");
                                exit(1)
                            }
                        })
                        .unwrap_or(500);
                    if config.is_present("linear")
                        || config.value_of("degree").map_or(false, |o| o == "1")
                    {
                        num_samples = 2;
                    }

                    let x_min = x_iter.clone().map(std_dev::F64OrdHash).min().unwrap().0;
                    let x_max = x_iter.clone().map(std_dev::F64OrdHash).max().unwrap().0;
                    let y_min = y_iter.clone().map(std_dev::F64OrdHash).min().unwrap().0;
                    let y_max = y_iter.clone().map(std_dev::F64OrdHash).max().unwrap().0;
                    let range = x_max - x_min;
                    let x_min = x_min - range * 0.1;
                    let range = range * 1.2;
                    let y_min = y_min - range * 0.2;
                    let y_max = y_max + range * 0.2;

                    let x = (0..num_samples)
                        .into_iter()
                        .map(|current| (current as f64 / (num_samples - 1) as f64) * range + x_min);

                    let mut plot = poloto::data();
                    plot.line(
                        format!("{model:.*}", p.unwrap_or(2)),
                        x.filter_map(|x| {
                            let y = model.predict_outcome(x);
                            Some((
                                x,
                                if num_samples < 5 || (y_min..y_max).contains(&y) {
                                    y
                                } else {
                                    return None;
                                },
                            ))
                        }),
                    );
                    plot.scatter("", x_iter.clone().zip(y_iter.clone()));
                    plot.text(format!(
                        "R² = {:.4}",
                        model.determination(x_iter, y_iter, len)
                    ));

                    let mut plotter = plot.build().plot(
                        config.value_of("plot_title").unwrap_or("Regression"),
                        config.value_of("plot_x_axis").unwrap_or("predictors"),
                        config.value_of("plot_y_axis").unwrap_or("outcomes"),
                    );
                    let data = poloto::disp(|a| plotter.render(a));
                    // Some scuffed styling to remove bar above R² value, move that closer to the
                    // equation, and to increase the width of the SVG.
                    // The styles are very dependent on only having 1 line.
                    let data = format!(
                        "{}<style>{}{}</style>{}{}",
                        r##"<svg class="poloto" width="1100" height="500" viewBox="0 0 1100 500" xmlns="http://www.w3.org/2000/svg">"##,
                        poloto::simple_theme::STYLE_CONFIG_DARK_DEFAULT,
                        r##".poloto_legend_text[y="200"] { transform: translate(0, -60px); }"##,
                        data,
                        poloto::simple_theme::SVG_END,
                    );

                    {
                        let path = if let Some(path) = config.value_of("plot_filename") {
                            let mut path = std::path::Path::new(path).to_path_buf();
                            path.set_extension("svg");
                            path
                        } else {
                            "plot.svg".into()
                        };
                        let mut file =
                            std::fs::File::create(&path).expect("failed to create plot file");
                        file.write_all(data.as_bytes())
                            .expect("failed to write plot file");
                    }

                    println!("Wrote plot file.");
                    if debug_performance {
                        println!("Plotting took {}µs.", now.elapsed().as_micros());
                    }
                }
            }
            Some(_) => unreachable!("invalid subcommand"),
            None => {
                let mut values = {
                    match input {
                        InputValue::Count(count) => std_dev::OwnedClusterList::new(count),
                        InputValue::List(list) => {
                            let mut count = Vec::with_capacity(list.len());
                            for item in list {
                                if item.len() != 1 && item.len() != 2 {
                                    eprintln!("Expected one or two values per line.");
                                    continue 'main;
                                }
                                let first = item[0];
                                let second = item.get(1).map_or(1, |f| f.round() as usize);
                                count.push((first, second))
                            }
                            std_dev::OwnedClusterList::new(count)
                        }
                    }
                };

                let now = Instant::now();

                values = values.borrow().optimize_values();

                if debug_performance {
                    println!("Optimizing input took {}µs", now.elapsed().as_micros());
                }

                let now = Instant::now();

                let mean = std_dev::standard_deviation_cluster(&values.borrow());

                if debug_performance {
                    println!(
                        "Standard deviation & mean took {}µs",
                        now.elapsed().as_micros()
                    );
                }
                let now = Instant::now();

                // Sort of clusters required.
                values.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

                let median = std_dev::percentiles_cluster(&mut values);

                if debug_performance {
                    println!("Median & quadrilles took {}µs", now.elapsed().as_micros());
                }

                let p = matches
                    .value_of("precision")
                    .map(|s| s.parse::<usize>().expect("we check this using clap"));

                if let Some(p) = p {
                    println!(
                        "Standard deviation: {:.5$}, mean: {:.5$}, median: {:.5$}{}{}",
                        mean.standard_deviation,
                        mean.mean,
                        median.median,
                        median
                            .lower_quadrille
                            .as_ref()
                            .map_or("".into(), |quadrille| {
                                format!(", lower quadrille: {:.1$}", *quadrille, p)
                            }),
                        median
                            .higher_quadrille
                            .as_ref()
                            .map_or("".into(), |quadrille| {
                                format!(", upper quadrille: {:.1$}", *quadrille, p)
                            }),
                        p
                    );
                } else {
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
        }
    }
}
