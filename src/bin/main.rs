use clap::{Arg, ArgAction, ValueHint};
use std::env;
use std::fmt::{Debug, Display};
#[cfg(feature = "regression")]
use std::io::Write;
use std::io::{stdin, BufRead};
use std::process::exit;
use std::str::FromStr;
use std::time::Instant;
use std_dev::regression::{
    BinarySearchOptions, CosecantEstimator, CosineEstimator, CotangentEstimator,
    ExponentialEstimator, GradientDescentParallelOptions, GradientDescentSimultaneousOptions,
    LogisticEstimator, PowerEstimator, SecantEstimator, SineEstimator, TangentEstimator,
};
#[cfg(feature = "regression")]
use std_dev::regression::{Determination, LinearEstimator, PolynomialEstimator, Predictive};

pub use std_dev;

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
            for segment in line.split(',').flat_map(|s| s.split_whitespace()) {
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
            .flat_map(|s| s.split_whitespace())
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
    x: impl Iterator<Item = f64> + Clone,
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
        .about(
            "Statistics calculation tool.\n\
            A common pattern is to cat files and pipe the data.",
        )
        .arg(
            Arg::new("debug-performance")
                .action(ArgAction::SetTrue)
                .long("debug-performance")
                .help(
                    "Print performance information. \
                    Can also be enabled by setting the \
                    DEBUG_PERFORMANCE environment variable.",
                ),
        )
        .arg(
            Arg::new("multiline")
                .short('m')
                .action(ArgAction::SetTrue)
                .long("multiline")
                .help(
                    "Accept multiple lines as one input. \
            Two consecutive newlines is treated as the series separator. \
            When not doing regression analysis the second 'column' \
            is the count of the first. Acts more like CSV.",
                ),
        )
        .arg(
            Arg::new("precision")
                .short('n')
                .long("precision")
                .help(
                    "Sets the precision of the output. When this isn't set, \
                    Rust decides how many digits to print. \
                    The determination will be 4 decimal places long. \
                    When this is set, all numbers are rounded.",
                )
                .num_args(1)
                .value_parser(clap::value_parser!(usize))
                .value_hint(ValueHint::Other),
        );

    #[cfg(feature = "completion")]
    {
        app = clap_autocomplete::add_subcommand(app);
    }

    #[cfg(feature = "regression")]
    {
        app = app.subcommand(
            clap::Command::new("regression")
                .about(
                    "Find a equation which describes the input data. \
                    Tries to automatically determine the model \
                    if no arguments specifying it are provided. \
                    Predictors are the independent values (usually denoted `x`) \
                    from which we want a equation to get the \
                    outcomes - the dependant variables, usually `y` or `f(x)`.",
                )
                .group(
                    clap::ArgGroup::new("model")
                        .arg("degree")
                        .arg("linear")
                        .arg("power")
                        .arg("exponential")
                        .arg("logistic")
                        .arg("sin")
                        .arg("cos")
                        .arg("tan")
                        .arg("sec")
                        .arg("csc")
                        .arg("cot"),
                )
                .group(
                    clap::ArgGroup::new("estimator")
                        .arg("theil_sen")
                        .arg("spiral")
                        .arg("binary")
                        .arg("ols"),
                )
                .arg(
                    Arg::new("degree")
                        .short('d')
                        .long("degree")
                        .help("Degree of polynomial.")
                        .num_args(1)
                        .value_parser(clap::value_parser!(usize))
                        .value_hint(ValueHint::Other),
                )
                .arg(
                    Arg::new("linear")
                        .short('l')
                        .action(ArgAction::SetTrue)
                        .long("linear")
                        .help("Tries to fit a line to the provided data."),
                )
                .arg(
                    Arg::new("power")
                        .short('p')
                        .action(ArgAction::SetTrue)
                        .long("power")
                        .help(
                            "Tries to fit a curve defined by \
                            the equation `a * x^b` to the data.\
                            If any of the predictors are below 1, x becomes (x+c), \
                            where c is an offset to the predictors. \
                            \
                            This is due to the arithmetic issue of taking the \
                            log of negative numbers and 0. A negative addition term \
                            will be appended if any of the outcomes are below 1.",
                        ),
                )
                .arg(
                    Arg::new("exponential")
                        .short('e')
                        .visible_alias("growth")
                        .long("exponential")
                        .action(ArgAction::SetTrue)
                        .help(
                            "Tries to fit a curve defined by the \
                            equation `a * b^x` to the data. \
                            If any of the predictors are below 1, x becomes (x+c), \
                            where c is an offset to the predictors. \
                            \
                            This is due to the arithmetic issue of taking the \
                            log of negative numbers and 0. A negative addition term \
                            will be appended if any of the outcomes are below 1.",
                        ),
                )
                .arg(
                    Arg::new("logistic")
                        .long("logistic")
                        .action(ArgAction::SetTrue)
                        .help(
                            "Tries to fit a curve defined by the logistic equation to the data. \
                    This requires the use of the spiral estimator.",
                        ),
                )
                .arg(
                    Arg::new("logistic_max")
                        .long("logistic-ceiling")
                        .help(
                            "Give the logistic regression the maximum value of the source. \
                            Say you know the population size and want to model the growth \
                            of a pandemic, use this to supply the population size.\n\
                            \n\
                            This gives much better performance than leaving it to the \
                            algorithm to figure out the ceiling.",
                        )
                        .requires("logistic")
                        .value_parser(|s: &str| {
                            parse::<f64>(s).ok_or("logistic-ceiling requites a float")
                        })
                        .value_hint(ValueHint::Other),
                )
                .group(
                    clap::ArgGroup::new("required_spiral")
                        .arg("logistic")
                        .arg("spiral")
                        .arg("sin")
                        .arg("cos")
                        .arg("tan")
                        .arg("sec")
                        .arg("csc")
                        .arg("cot")
                        .multiple(true)
                        .conflicts_with("ols")
                        .conflicts_with("theil_sen"),
                )
                .group(
                    clap::ArgGroup::new("trig")
                        .arg("sin")
                        .arg("cos")
                        .arg("tan")
                        .arg("sec")
                        .arg("csc")
                        .arg("cot"),
                )
                .arg(
                    Arg::new("sin")
                        .long("sin")
                        .action(ArgAction::SetTrue)
                        .help("Fit a sine wave."),
                )
                .arg(
                    Arg::new("cos")
                        .long("cos")
                        .action(ArgAction::SetTrue)
                        .help("Fit a cosine wave."),
                )
                .arg(
                    Arg::new("tan")
                        .long("tan")
                        .action(ArgAction::SetTrue)
                        .help("Fit a tangent function."),
                )
                .arg(
                    Arg::new("sec")
                        .long("sec")
                        .action(ArgAction::SetTrue)
                        .help("Fit a secant function."),
                )
                .arg(
                    Arg::new("csc")
                        .long("csc")
                        .action(ArgAction::SetTrue)
                        .help("Fit a cosecant function."),
                )
                .arg(
                    Arg::new("cot")
                        .long("cot")
                        .action(ArgAction::SetTrue)
                        .help("Fit a cotangent function."),
                )
                .arg(
                    Arg::new("trig_freq")
                        .long("trig-frequency-limit")
                        .help(
                            "Set the limit for frequency of the \
                              fitted trigonometric function.",
                        )
                        .requires("trig")
                        .default_value("1.0")
                        .value_parser(|v: &str| {
                            parse::<f64>(v)
                                .filter(|v| *v > 0.)
                                .ok_or("frequency needs to be a positive float")
                        })
                        .value_hint(ValueHint::Other),
                )
                .arg(
                    Arg::new("ols")
                        .long("ols")
                        .action(ArgAction::SetTrue)
                        .help("Use the ordinary least squares estimator. Linear time complexity."),
                )
                .arg(
                    Arg::new("theil_sen")
                        .long("theil-sen")
                        .short('t')
                        .action(ArgAction::SetTrue)
                        .help(
                            "Use the Theil-Sen estimator instead \
                            of OLS for all models. O(n^degree).",
                        ),
                )
                .arg(
                    Arg::new("spiral")
                        .long("spiral")
                        .short('s')
                        .action(ArgAction::SetTrue)
                        .help(
                            "Use the spiral estimator instead of OLS for all models \
                            (only supports polynomial of degree 1&2). \
                            A good result isn't guaranteed. Linear time complexity.",
                        ),
                )
                .arg(
                    Arg::new("spiral_level")
                        .long("spiral-level")
                        .help(
                            "Speed preset of spiral estimator. Lower are faster, \
                            but increase the risk of invalid output. \
                            You can expect a 2-4x decrease in performance \
                            for each additional level. \
                            Regressions with 3 variables require a higher level. \
                            The performance of these presets may change at any time.",
                        )
                        .requires("required_spiral")
                        .num_args(1)
                        .default_value("5")
                        .value_parser(|v: &str| {
                            parse::<u8>(v)
                                .filter(|v| (1..=9).contains(v))
                                .ok_or("spiral-level has to be in range [1..=9]")
                        })
                        .value_hint(ValueHint::Other),
                )
                .arg(
                    Arg::new("descent")
                        .long("gradient-descent")
                        .short('g')
                        .action(ArgAction::SetTrue)
                        .help(
                            "Use the gradient descent estimator instead of OLS for all models. \
                            A good result is guaranteed. Linear time complexity.",
                        ),
                )
                .arg(
                    Arg::new("simultaneous")
                        .long("gradient-descent-descent")
                        .short('u')
                        .action(ArgAction::SetTrue)
                        .help(
                            "Use the gradient descent estimator instead of OLS for all models. \
                            The simultaneous estimator is better at regressions where multiple \
                            variables affect the quality together. \
                            Linear time complexity.",
                        ),
                )
                .arg(
                    Arg::new("simultaneous_level")
                        .long("simultaneous-accuracy")
                        .help(
                            "Accuracy preset of gradient descent simultaneous \
                            estimator. Generally, when many variables are \
                            optimized (e.g. >8 degree polynomial), \
                            the accuracy needs to be more fine.",
                        )
                        .requires("simultaneous")
                        .num_args(1)
                        .default_value("1e-4")
                        .value_parser(|v: &str| {
                            parse::<f64>(v)
                                .filter(|v| v.is_finite())
                                .ok_or("simultaneous-accuracy needs to be a number")
                        })
                        .value_hint(ValueHint::Other),
                )
                .arg(
                    Arg::new("binary")
                        .long("binary-search")
                        .short('b')
                        .action(ArgAction::SetTrue)
                        .help(
                            "Use the binary search estimator instead of OLS for all models \
                            A good result isn't guaranteed. Linear time complexity.",
                        ),
                )
                .arg(
                    Arg::new("binary_precise")
                        .long("binary-full-precision")
                        .action(ArgAction::SetTrue)
                        .help(
                            "Get the full precision of 64-bit \
                            floats when calculating the binary-search",
                        )
                        .requires("binary"),
                )
                .arg(
                    Arg::new("binary_iterations")
                        .long("binary-iterations")
                        .num_args(1)
                        .requires("binary")
                        .help(
                            "Number of iterations for the binary search. \
                            Increasing this value is good in situations \
                            with many variables which are dependant.",
                        )
                        .value_parser(clap::value_parser!(usize))
                        .default_value("30"),
                )
                .arg(
                    Arg::new("binary_randomness")
                        .long("binary-randomness")
                        .num_args(1)
                        .requires("binary")
                        .help(
                            "Randomness factor in binary search.\
                            Larger values yield better and possibly more inconsistent results.",
                        )
                        .value_parser(|v: &str| {
                            parse::<f64>(v)
                                .filter(|v| *v <= 1. && *v > 0.)
                                .ok_or("--binary-randomness needs to be a number under 1.")
                        })
                        .default_value("1.0"),
                )
                .arg(
                    Arg::new("plot")
                        .long("plot")
                        .action(ArgAction::SetTrue)
                        .help("Plots the regression and input variables in a SVG."),
                )
                .arg(
                    Arg::new("plot_filename")
                        .long("plot-out")
                        .help("File name (without extension) for SVG plot.")
                        .num_args(1)
                        .requires("plot")
                        .value_hint(ValueHint::FilePath),
                )
                .arg(
                    Arg::new("plot_samples")
                        .long("plot-samples")
                        .help(
                            "Count of sample points when drawing the curve. \
                              Always set to 2 for linear regressions.",
                        )
                        .num_args(1)
                        .requires("plot")
                        .value_hint(ValueHint::Other),
                )
                .arg(
                    Arg::new("plot_title")
                        .long("plot-title")
                        .help("Title of plot.")
                        .num_args(1)
                        .requires("plot")
                        .value_hint(ValueHint::Other),
                )
                .arg(
                    Arg::new("plot_x_axis")
                        .long("plot-axis-x")
                        .help("Name of x axis of plot (the first column of data).")
                        .num_args(1)
                        .requires("plot")
                        .value_hint(ValueHint::Other),
                )
                .arg(
                    Arg::new("plot_y_axis")
                        .long("plot-axis-y")
                        .help("Name of y axis of plot (the second column of data).")
                        .num_args(1)
                        .requires("plot")
                        .value_hint(ValueHint::Other),
                ),
        );
    }

    #[cfg(feature = "regression")]
    let spiral_polynomial_degree_error = app.error(
        clap::error::ErrorKind::InvalidValue,
        "spiral only supports polynomials of degree 1 & 2",
    );

    #[cfg(feature = "completion")]
    let command = app.clone();
    let matches = app.get_matches();

    #[cfg(feature = "completion")]
    {
        match clap_autocomplete::test_subcommand(&matches, command) {
            Some(Ok(())) => exit(0),
            Some(Err(s)) => {
                eprintln!("{s}");
                exit(1)
            }
            None => {}
        }
    }

    let debug_performance = env::var("DEBUG_PERFORMANCE").ok().map_or_else(
        || matches.get_flag("debug-performance"),
        |s| !s.trim().is_empty(),
    );

    #[cfg(feature = "pretty")]
    let tty = std::io::stdin().is_terminal();
    #[cfg(not(feature = "pretty"))]
    let tty = false;

    let mut last_prompt = Instant::now();

    'main: loop {
        let multiline = {
            matches.get_flag("multiline") || matches!(matches.subcommand_name(), Some("regression"))
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

                let spiral_options = {
                    let level = *config
                        .get_one::<u8>("spiral_level")
                        .expect("we've provided a default value and validator");
                    std_dev::regression::spiral::Options::new(level)
                };
                let binary_options = {
                    let iterations = *config
                        .get_one("binary_iterations")
                        .expect("we've provided a default value and validator");
                    let randomness = *config
                        .get_one("binary_randomness")
                        .expect("we've provided a default value and validator");
                    let max_precision = config.get_flag("binary_precise");
                    let c = BinarySearchOptions {
                        iterations,
                        randomness_factor: randomness,
                        ..Default::default()
                    };
                    if max_precision {
                        c.max_precision()
                    } else {
                        c
                    }
                };
                let trig_freq: f64 = *config
                    .get_one("trig_freq")
                    .expect("we provided a default value and have a validator");

                let linear_estimator = {
                    if config.get_flag("theil_sen") {
                        std_dev::regression::LinearTheilSen.boxed_linear()
                    } else if config.get_flag("descent") {
                        GradientDescentParallelOptions::default().boxed_linear()
                    } else if config.get_flag("simultaneous") {
                        GradientDescentSimultaneousOptions::new(1e-6).boxed_linear()
                    } else if config.get_flag("spiral") {
                        spiral_options.clone().boxed_linear()
                    } else if config.get_flag("binary") {
                        binary_options.boxed_linear()
                    } else {
                        #[cfg(feature = "ols")]
                        {
                            std_dev::regression::OlsEstimator.boxed_linear()
                        }
                        #[cfg(not(feature = "ols"))]
                        {
                            eprintln!("No estimator specified. Consider enabling the OLS feature or explicitly specifying an estimator.");
                            exit(1);
                        }
                    }
                };

                let now = Instant::now();

                let model = if config.get_flag("power") {
                    if config.get_flag("spiral") {
                        spiral_options.model_power(&x, &y).boxed()
                    } else if config.get_flag("binary") {
                        binary_options.model_power(&x, &y).boxed()
                    } else {
                        std_dev::regression::derived::power(&mut x, &mut y, &&*linear_estimator)
                            .boxed()
                    }
                } else if config.get_flag("exponential") {
                    if config.get_flag("spiral") {
                        spiral_options.model_exponential(&x, &y).boxed()
                    } else if config.get_flag("binary") {
                        binary_options.model_exponential(&x, &y).boxed()
                    } else {
                        std_dev::regression::derived::exponential(
                            &mut x,
                            &mut y,
                            &&*linear_estimator,
                        )
                        .boxed()
                    }
                } else if config.get_flag("logistic") {
                    if let Some(ceiling) = config.get_one::<f64>("logistic_max").copied() {
                        std_dev::regression::SpiralLogisticWithCeiling::new(
                            spiral_options.clone(),
                            ceiling,
                        )
                        .model_logistic(&x, &y)
                        .boxed()
                    } else if config.get_flag("spiral") {
                        spiral_options.model_logistic(&x, &y).boxed()
                    } else {
                        binary_options.model_logistic(&x, &y).boxed()
                    }
                } else if config.get_flag("sin") {
                    if config.get_flag("spiral") {
                        spiral_options.model_sine(&x, &y, trig_freq).boxed()
                    } else {
                        binary_options.model_sine(&x, &y, trig_freq).boxed()
                    }
                } else if config.get_flag("cos") {
                    if config.get_flag("spiral") {
                        spiral_options.model_cosine(&x, &y, trig_freq).boxed()
                    } else {
                        binary_options.model_sine(&x, &y, trig_freq).boxed()
                    }
                } else if config.get_flag("tan") {
                    if config.get_flag("spiral") {
                        spiral_options.model_tangent(&x, &y, trig_freq).boxed()
                    } else {
                        binary_options.model_sine(&x, &y, trig_freq).boxed()
                    }
                } else if config.get_flag("sec") {
                    if config.get_flag("spiral") {
                        spiral_options.model_secant(&x, &y, trig_freq).boxed()
                    } else {
                        binary_options.model_sine(&x, &y, trig_freq).boxed()
                    }
                } else if config.get_flag("csc") {
                    if config.get_flag("spiral") {
                        spiral_options.model_cosecant(&x, &y, trig_freq).boxed()
                    } else {
                        binary_options.model_sine(&x, &y, trig_freq).boxed()
                    }
                } else if config.get_flag("cot") {
                    if config.get_flag("spiral") {
                        spiral_options.model_cotangent(&x, &y, trig_freq).boxed()
                    } else {
                        binary_options.model_sine(&x, &y, trig_freq).boxed()
                    }
                } else if config.get_flag("linear") || config.get_one::<usize>("degree").is_some() {
                    let degree = {
                        if let Some(degree) = config.get_one("degree") {
                            *degree
                        } else {
                            1
                        }
                    };
                    if degree + 1 > len {
                        eprintln!("Degree of polynomial is too large; add more datapoints.");
                        continue 'main;
                    }

                    if degree == 1 {
                        linear_estimator.model_linear(&x, &y).boxed()
                    } else {
                        let estimator = {
                            if config.get_flag("theil_sen") {
                                std_dev::regression::PolynomialTheilSen.boxed_polynomial()
                            } else if config.get_flag("descent") {
                                GradientDescentParallelOptions::default().boxed_polynomial()
                            } else if config.get_flag("simultaneous") {
                                let accuracy = *config
                                    .get_one("simultaneous_level")
                                    .expect("we provided a default value and checked the input");

                                GradientDescentSimultaneousOptions::new(accuracy).boxed_polynomial()
                            } else if config.get_flag("spiral") {
                                if !(1..=2).contains(&degree) {
                                    spiral_polynomial_degree_error.exit();
                                }
                                spiral_options.clone().boxed_polynomial()
                            } else if config.get_flag("binary") {
                                binary_options.boxed_polynomial()
                            } else {
                                #[cfg(feature = "ols")]
                                {
                                    std_dev::regression::OlsEstimator.boxed_polynomial()
                                }
                                #[cfg(not(feature = "ols"))]
                                {
                                    eprintln!("No estimator specified. Consider enabling the OLS feature or explicitly specifying an estimator.");
                                    exit(1);
                                }
                            }
                        };

                        estimator.model_polynomial(&x, &y, degree).boxed()
                    }
                } else {
                    std_dev::regression::best_fit(&x, &y, &&*linear_estimator)
                };

                let p = matches.get_one::<usize>("precision").copied();

                print_regression(&model, x_iter.clone(), y_iter.clone(), len, p);

                if debug_performance {
                    let elapsed = now.elapsed().as_micros();
                    if elapsed > 50_000 {
                        println!("Regression analysis took {}ms.", elapsed / 1000);
                    } else {
                        println!("Regression analysis took {}µs.", elapsed);
                    }
                }

                if config.get_flag("plot") {
                    let now = Instant::now();

                    let mut num_samples = config
                        .get_one::<&str>("plot_samples")
                        .map(|s| {
                            if let Ok(i) = s.parse() {
                                i
                            } else {
                                eprintln!("You must supply an integer to `plot-samples`.");
                                exit(1)
                            }
                        })
                        .unwrap_or(500);
                    if config.get_flag("linear")
                        || config.get_one::<usize>("degree").map_or(false, |o| *o == 1)
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
                        .map(|current| (current as f64 / (num_samples - 1) as f64) * range + x_min);

                    let line = poloto::build::plot(format!("{model:.*}", p.unwrap_or(2))).line(
                        x.map(|x| {
                            let y = model.predict_outcome(x);
                            (
                                x,
                                if num_samples < 5 || (y_min..y_max).contains(&y) {
                                    y
                                } else {
                                    // returning NAN makes the point disappear from the graph
                                    f64::NAN
                                },
                            )
                        }),
                    );
                    let scatter = poloto::build::plot("".to_owned())
                        .scatter(x_iter.clone().zip(y_iter.clone()));
                    let determination = poloto::build::plot(format!(
                        "R² = {:.4}",
                        model.determination(x_iter, y_iter, len)
                    ))
                    .text();

                    use tagu::elem::{Elem, Raw};

                    let plotter = poloto::frame_build()
                        .data(poloto::plots!(line, scatter, determination))
                        .build_and_label((
                            config
                                .get_one::<String>("plot_title")
                                .map_or("Regression", String::as_str),
                            config
                                .get_one::<String>("plot_x_axis")
                                .map_or("predictors", String::as_str),
                            config
                                .get_one::<String>("plot_y_axis")
                                .map_or("outcomes", String::as_str),
                        ))
                        .append_to(
                            poloto::header()
                                .with_dim([1100., 500.])
                                .with_viewbox([1100., 500.])
                                .dark_theme()
                                .append(tagu::elem::Element::new("style").append(Raw::new(
                                    ".poloto_legend[y=\"200\"] \
                                    { transform: translate(0, -60px); }",
                                ))),
                        );

                    let data = plotter.render_string().unwrap();

                    {
                        let path = if let Some(path) = config.get_one::<&str>("plot_filename") {
                            let mut path = std::path::Path::new(path).to_path_buf();
                            path.set_extension("svg");
                            path
                        } else {
                            "plot.svg".into()
                        };
                        let mut file =
                            std::fs::File::create(path).expect("failed to create plot file");
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

                let p = matches.get_one::<usize>("precision").copied();

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
