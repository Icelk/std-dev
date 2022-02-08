# std-dev

> Statistics calculator

# Performance

The n here isn't numer of elements, but rather number of *unique* elements.
If you have a range of possible values small compared to the count of values, this becomes stupidly fast.
Grouping the values has a O(n) time where n = number of elements. This is however fast - it takes less than half the time of parsing, so it isn't really affecting performance.

O(n) for mean & standard deviation

O(n log n) for median and derivatives
