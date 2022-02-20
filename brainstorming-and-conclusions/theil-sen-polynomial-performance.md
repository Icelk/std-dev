This document investigates how the performance of the Theil-Sen estimator behaves when we increase the degree of the polynomial from 1.

To make a curve from a polynomial, on needs `degree + 1` points. Therefore, I'm investigating how we can get lists of `degree + 1` points, to fit curves to.

[1,2,3,4,5]

[1,2]
[1,3]
[1,4]
[1,5]

[2,3]
[2,4]
[2,5]

[3,4]
[3,5]

[4,5]

n = 10

[1,2,3,4,5]

The second is just ↑, but we multiply by `n` again because of the first value.

n of slices which starts with 1: n-4 = 6
--||-- 2: n-4(4 slices above have 2 in them)
--||-- n: n-4 (same as above)

c = slice count
n = number of items
c = 5(n-4) = 30
c=c(n-(c-1))
1=n-(c-1)
0=n-c
c=n

# List of 6

[1,2,3,4,5,6]

[1,2]
[1,3]
[1,4]
[1,5]
[1,6]

[2,3]
[2,4]
[2,5]
[2,6]

[3,4]
[3,5]
[3,6]

[4,5]
[4,6]

[5,6]

n=15 (increased with n[old])
O(n²)

The length of the slice however increases by `c=n` per 1 length increase. This means increasing the degree does not affect output, and does not take more time - increasing degree should be O(d) where d is the polynomial degree.
