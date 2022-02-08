# std-dev

> Statistics calculator

# Performance

The n here isn't numer of elements, but rather number of *unique* elements.
If you have a range of possible values small compared to the count of values, this becomes stupidly fast.
Grouping the values has a O(n) time where n = number of elements. This is however fast - it takes less than half the time of parsing, so it isn't really affecting performance.

This means that n is the rate of which the range increases compared to input length. If the range doesn't expand, mean & standard deviation is O(1) and median is also O(1)! This isn't the case, as parsing takes time, but then it becomes O(n). If you use this as a library, you'd get O(1) performance.

O(n) for mean & standard deviation

O(n log n) for median and derivatives
