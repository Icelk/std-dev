pub struct OwnedClusterList {
    list: Vec<(f64, usize)>,
    len: usize,
}
impl OwnedClusterList {
    pub fn borrow(&self) -> ClusterList {
        ClusterList {
            list: &self.list,
            len: self.len,
        }
    }
}

/// A list of clusters.
///
/// A cluster is a value and the count.
pub struct ClusterList<'a> {
    list: &'a [(f64, usize)],
    len: usize,
}
impl<'a> ClusterList<'a> {
    /// The float is the value. The integer is the count.
    pub fn new(list: &'a [(f64, usize)]) -> Self {
        let len = Self::size(list);
        Self { list, len }
    }

    fn size(list: &[(f64, usize)]) -> usize {
        list.iter().map(|(_, count)| *count).sum()
    }

    /// O(1)
    pub fn len(&self) -> usize {
        self.len
    }
    /// O(1)
    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }
    pub fn sum(&self) -> f64 {
        let mut sum = 0.0;
        for (v, count) in self.list.iter() {
            sum += v * *count as f64;
        }
        sum
    }
    fn sum_squared_diff(&self, base: f64) -> f64 {
        let mut sum = 0.0;
        for (v, count) in self.list.iter() {
            sum += (v - base).powi(2) * *count as f64;
        }
        sum
    }
    /// The inner list must be sorted by the `f64`.
    pub fn median(&self) -> f64 {
        let len = self.len();
        let even = len % 2 == 0;
        let mut len = len;
        let target = len / 2;

        for (pos, (v, count)) in self.list.iter().enumerate() {
            len -= *count;
            if len + 1 == target && even {
                let mean = (*v + self.list[pos - 1].0) / 2.0;
                return mean;
            }
            if len < target || len == target && !even {
                return *v;
            }
        }
        0.0
    }
    /// Can be used in [`Self::new`].
    pub fn split_start(&self, len: usize) -> OwnedClusterList {
        let mut sum = 0;
        let mut list = Vec::new();
        for (v, count) in self.list {
            sum += count;
            if sum >= len {
                list.push((*v, *count - (sum - len)));
                break;
            } else {
                list.push((*v, *count));
            }
        }
        debug_assert_eq!(len, Self::size(&list));
        OwnedClusterList { list, len }
    }
    /// Can be used in [`Self::new`].
    pub fn split_end(&self, len: usize) -> OwnedClusterList {
        let len = self.len() - len;
        let mut sum = self.len();
        let mut list = Vec::new();
        for (v, count) in self.list.iter().rev() {
            sum -= count;
            if sum <= len {
                list.insert(0, (*v, *count - (len - sum)));
                break;
            } else {
                list.insert(0, (*v, *count))
            }
        }
        debug_assert_eq!(len, Self::size(&list));
        OwnedClusterList { list, len }
    }
}

/// Returned from [`std_dev`].
pub struct MeanOutput {
    pub standard_deviation: f64,
    pub mean: f64,
}
/// Returned from [`median`]
pub struct MedianOutput {
    pub median: f64,
    pub lower_quadrille: Option<f64>,
    pub higher_quadrille: Option<f64>,
}

pub fn std_dev(values: ClusterList) -> MeanOutput {
    let m = values.sum() / values.len() as f64;
    let squared_deviations = values.sum_squared_diff(m);
    let variance: f64 = squared_deviations / (values.len() - 1) as f64;
    MeanOutput {
        standard_deviation: variance.sqrt(),
        mean: m,
    }
}
pub fn median(values: ClusterList) -> MedianOutput {
    let lower_half = values.split_start(values.len() / 2);
    let lower_half = lower_half.borrow();
    let upper_half = values.split_end(values.len() / 2);
    let upper_half = upper_half.borrow();
    MedianOutput {
        median: values.median(),
        lower_quadrille: if lower_half.len() > 1 {
            Some(lower_half.median())
        } else {
            None
        },
        higher_quadrille: if upper_half.len() > 1 {
            Some(upper_half.median())
        } else {
            None
        },
    }
}
