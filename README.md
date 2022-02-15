# std-dev

> Fast statistics calculator, implemented for industrial and educational purposes alike.

This codebase is well-documented and comments, in an effort to expose the wonderful algorithms of data analysis to the masses.

We're ever expanding, but for now the following are implemented.

- Standard deviation
- Fast median and mean for large datasets with limited options of values
- [Ordinary least square](https://en.wikipedia.org/wiki/Ordinary_least_squares) for linear and polynomial regression
- Exponential/growth and power regression, with **correct handling of negatives**

# Usage

This application supports using it both as a **library** (with optional cargo features),
an interactive **CLI** program, and through **piping** data to it, through standard input.

It accepts any comma/space separated values. Scientific notation is supported.
This is minimalistic by design, as other programs may be used to produce/modify the data before it's processed by us.

# Performance

The n here isn't numer of elements, but rather number of *unique* elements.
If you have a range of possible values small compared to the count of values, this becomes stupidly fast.
Grouping the values has a O(n) time where n = number of elements. This is however fast - it takes less than half the time of parsing, so it isn't really affecting performance.

This means that n is the rate of which the range increases compared to input length. If the range doesn't expand, mean & standard deviation is O(1) and median is also O(1)! This isn't the case, as parsing takes time, but then it becomes O(n). If you use this as a library, you'd get O(1) performance.

O(n) for mean & standard deviation

O(n log n) for median and derivatives
