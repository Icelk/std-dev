# std-dev

> Fast statistics calculator, implemented for industrial and educational purposes alike.

This codebase is well-documented and comments, in an effort to expose the wonderful algorithms of data analysis to the masses.

We're ever expanding, but for now the following are implemented.

-   Standard deviation
-   Fast median and mean for large datasets with limited options of values ([clusters](#clusters))
-   O(n) - linear time - algorithms, both for arbitrary generic lists (any type of number) and clusters:
    -   percentile
        -   median
    -   standard deviation
    -   mean
-   [Ordinary least square](https://en.wikipedia.org/wiki/Ordinary_least_squares) for linear and polynomial regression
-   Exponential/growth and power regression, with **correct handling of negatives**
-   "best fit" method if you don't know which regression model to use.
-   (binary) A basic plotting feature to preview the equation in relation to the input data

# Usage

This application supports using it both as a **library** (with optional cargo features),
an interactive **CLI** program, and through **piping** data to it, through standard input.

It accepts any comma/space separated values. Scientific notation is supported.
This is minimalistic by design, as other programs may be used to produce/modify the data before it's processed by us.

# Cargo features

When using this as a library, I recommend disabling all features (`std-dev = { version = "0.1", default-features = false }`)
and enabling those you need.

## Bin

This enables the binary to compile.

## Prettier

Makes the binary output prettier. Includes colours and prompts for interactive use.

## Regression

Enables all regression estimators. This requires `nalgebra`, which provides linear algebra.

# Performance

## Clusters

> As all algorithms are executed in linear time now, this is virtually obsolete, but nevertheless an interesting feature.
> If you already have clustered data, this feature is great.

When using the clusters feature (turning your list into a `ClusterList`),
calculations are done per _unique_ value.
Say you have a dataset of infant height, in centimeters.
That's probably only going to be some 40 different values, but potentially millions of entries.
Using clusters, all that data is only processed as `O(40)`, not `O(millions)`. (I know that notation isn't right, but you get my point).

Creating this cluster involves adding all the values to a map. This takes `O(n)` time, but is very slow compared to all other algorithms.
After creation, most operations in this library are executed in `O(m)` time, where m is the count of unique values.
