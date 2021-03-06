-   [x] Change precision in output.
-   [x] Arbitrary precision for high degree polynomials.
-   [x] Automatically choose process
-   [x] Linearization - solve all other.
-   [x] Fix input handling? (arrow up, sides, don't spam prompt when pasting (or don't have prompt on multiline?))
-   [x] Fix NaN on exponential and power regressions (add the min y + 1 to get all numbers above 1, then fit the curve to that)
-   [x] Slow ( O(n²) ) Theil-Sen estimator
-   [ ] O(n log n) Theil-Sen estimator (heavy-hitter, very hard)
-   [x] O(n) median (intermediate difficulty)
-   [x] All statistical tools for unique lists (not `Cluster`s)
-   [x] Performance logging in regression calculations.
-   [x] Plotting of data & regressions using [poloto](https://crates.io/crates/poloto) or [plotlib](https://crates.io/crates/plotlib)
-   [ ] Option for other plot lib.
-   [ ] Fix [bias](https://en.wikipedia.org/wiki/Nonlinear_regression#Transformation) in power and exponential regressions.
        Right now, it's biased towards errors in small values, as the large errors are, in the linear space, the log of what they are in reality.
-   [ ] [RANSAC](https://en.wikipedia.org/wiki/Random_sample_consensus) implementation? Iterations = lg(1-chance of success) \* (lg(number of data points) / lg(outliers in relation to total))
-   [ ] Support [covariance](https://en.wikipedia.org/wiki/Generalized_least_squares), for better estimation.
-   [ ] [Non](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)-[parametric](https://en.wikipedia.org/wiki/Local_regression) regression?
-   [ ] [Non-linear regression?](https://en.wikipedia.org/wiki/Non-linear_least_squares)
