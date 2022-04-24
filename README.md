# std-dev

> Your Swiss Army knife for swiftly processing any amount of data. Implemented for industrial and educational purposes alike.

This codebase is well-documented and comments, in an effort to expose the wonderful algorithms of data analysis to the masses.

We're ever expanding, but for now the following are implemented.

-   Standard deviation, both for generic slices and [clusters](#clusters).
-   Fast median and mean for large datasets with limited options of values ([clusters](#clusters))
-   O(n) - linear time - algorithms, both for arbitrary generic lists (any type of number) and clusters:
    -   percentile
        -   median
    -   standard deviation
    -   mean
-   [Ordinary least square](https://en.wikipedia.org/wiki/Ordinary_least_squares) for linear and polynomial regression
-   Naive (O(n²))[Theil-Sen estimator](https://en.wikipedia.org/wiki/Theil%E2%80%93Sen_estimator) for both linear and polynomial (O(n^(m)), where m is the degree + 1) regression
-   Exponential/growth and power regression, with **correct handling of negatives** (most other applications silently ignores them)
-   "best fit" method if you don't know which regression model to use
-   (binary) A basic plotting feature to preview the equation in relation to the input data

# Usage

This application supports using it both as a **library** (with optional cargo features),
an interactive **CLI** program, and through **piping** data to it, through standard input.

It accepts any comma/space separated values. Scientific notation is supported.
This is minimalistic by design, as other programs may be used to produce/modify the data before it's processed by us.

## Shell completion

Using the subcommand `completion`, std-dev automatically generates shell completions for your shell and tries to put them in the appropriate location.

When using Bash or Zsh, you should run std-dev as root, as we need root privileges to write to their completion directories.
Alternatively, use the `--print` option to yourself write the completion file.

# Cargo features

When using this as a library, I recommend disabling all features (except `base`) (`std-dev = { version = "0.1", default-features = false, features = ["base"] }`)
and enabling those you need.

-   `bin` (default, binary feature): This enables the binary to compile.
-   `prettier` (default, binary feature): Makes the binary output prettier. Includes colours and prompts for interactive use.
-   `completion` (default, binary feature): Enable the ability to generate shell completions.
-   `regression` (default, library and binary feature): Enables all regression estimators. This requires `nalgebra`, which provides linear algebra.
-   `ols` (default, library feature): Enables the use of [OLS](https://en.wikipedia.org/wiki/Ordinary_least_squares), which is the "default" estimator. This also enables polynomial Theil-Sen for degrees > 2 & polynomial regression in `best_fit` functions.
-   `arbitrary-precision` (default, library feature): Uses arbitrary precision algebra for >10 degree polynomial regression.
-   `percentile-rand` (default, base, library feature): Enables the recommended `pivot_fn` for percentile-related functions.
-   `simplify-fraction` (default, base, library feature): Fractions are simplified. Relaxes the requirements for fraction input and implements Eq & Ord for fractions.
-   `generic-impls` (default, base, library feature): Makes `mean`, `standard_deviation`, and percentile resolving generic over numbers. This enables you to use numerical types from other libraries without hassle.

# Documentation

Documentation of the main branch can be found at [doc.icelk.dev](https://doc.icelk.dev/std-dev/std_dev/).

To document with information on which cargo features enables the code,
set the environment variable `RUSTDOCFLAGS` to `--cfg docsrs`
(e.g. in Fish `set -x RUSTDOCFLAGS "--cfg docsrs"`)
and then run `cargo +nightly doc`.

# Performance

This library aims to be as fast as possible while maintaining easily readable code.

## Clusters

> As all algorithms are executed in linear time now, this is not as useful, but nevertheless an interesting feature.
> If you already have clustered data, this feature is great.

When using the clusters feature (turning your list into a `ClusterList`),
calculations are done per _unique_ value.
Say you have a dataset of infant height, in centimeters.
That's probably only going to be some 40 different values, but potentially millions of entries.
Using clusters, all that data is only processed as `O(40)`, not `O(millions)`. (I know that notation isn't right, but you get my point).

Creating this cluster involves adding all the values to a map. This takes `O(n)` time, but is very slow compared to all other algorithms.
After creation, most operations in this library are executed in `O(m)` time, where m is the count of unique values.
