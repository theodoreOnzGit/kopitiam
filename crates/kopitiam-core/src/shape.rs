use std::fmt;

use crate::Error;

/// The dimensions of a tensor, outermost first.
///
/// A rank-0 shape (no dimensions) is a scalar and has exactly one element —
/// not zero. That is the convention every tensor library converges on, and
/// getting it wrong makes reductions (which produce scalars) special-cased
/// everywhere.
///
/// `Shape` owns its dimensions in a `Vec`. Inference shapes are small (rank
/// 4 at most, in practice) and created once per tensor, not per element, so
/// the allocation is not on any hot path — and a fixed-size inline array
/// would impose an arbitrary maximum rank for no measurable gain.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Shape(Vec<usize>);

impl Shape {
    pub fn new(dims: impl Into<Vec<usize>>) -> Self {
        Self(dims.into())
    }

    /// The scalar shape: rank 0, one element.
    pub fn scalar() -> Self {
        Self(Vec::new())
    }

    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    /// Number of dimensions. Zero for a scalar.
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// Total number of elements: the product of all dimensions.
    ///
    /// An empty product is 1, so a scalar correctly reports one element. A
    /// shape containing a zero dimension correctly reports zero.
    pub fn elem_count(&self) -> usize {
        self.0.iter().product()
    }

    /// Row-major (C-order) strides, in elements, for this shape.
    ///
    /// Row-major means the last dimension is contiguous — walking it steps
    /// one element at a time. This is the layout GGUF and SafeTensors both
    /// store weights in, so choosing it as the default avoids transposing
    /// every tensor at load time.
    pub fn strides(&self) -> Vec<usize> {
        let mut strides = vec![1; self.rank()];
        for i in (0..self.rank().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * self.0[i + 1];
        }
        strides
    }

    /// Reinterprets this shape as `dims`, which must describe the same number
    /// of elements.
    ///
    /// This is the shape-level half of a reshape; it says nothing about
    /// whether the underlying storage is contiguous enough to allow the
    /// reinterpretation without copying. That check belongs to the tensor.
    pub fn reshape(&self, dims: impl Into<Vec<usize>>) -> Result<Self, Error> {
        let candidate = Self::new(dims);
        if candidate.elem_count() != self.elem_count() {
            return Err(Error::ShapeMismatch {
                expected: self.clone(),
                actual: candidate,
            });
        }
        Ok(candidate)
    }

    /// The shape resulting from broadcasting `self` against `other`, or an
    /// error if the two are not broadcast-compatible.
    ///
    /// Follows the standard NumPy rule: align shapes from the right; two
    /// dimensions are compatible when they are equal or one of them is 1;
    /// the result takes the larger of each pair. Missing leading dimensions
    /// on the shorter shape are treated as 1.
    pub fn broadcast(&self, other: &Shape) -> Result<Shape, Error> {
        let rank = self.rank().max(other.rank());
        let mut dims = vec![0usize; rank];

        for i in 0..rank {
            // Align from the right: index `i` from the end of each shape.
            let a = self.dim_from_end(i).unwrap_or(1);
            let b = other.dim_from_end(i).unwrap_or(1);

            dims[rank - 1 - i] = match (a, b) {
                (a, b) if a == b => a,
                (1, b) => b,
                (a, 1) => a,
                _ => {
                    return Err(Error::NotBroadcastable {
                        left: self.clone(),
                        right: other.clone(),
                    });
                }
            };
        }

        Ok(Shape::new(dims))
    }

    /// The `i`th dimension counting from the last (0 = last dimension).
    fn dim_from_end(&self, i: usize) -> Option<usize> {
        (i < self.rank()).then(|| self.0[self.rank() - 1 - i])
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self(dims)
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(dims: [usize; N]) -> Self {
        Self(dims.to_vec())
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Self(dims.to_vec())
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[")?;
        for (i, dim) in self.0.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{dim}")?;
        }
        f.write_str("]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_scalar_has_rank_zero_and_exactly_one_element() {
        let s = Shape::scalar();
        assert_eq!(s.rank(), 0);
        assert_eq!(s.elem_count(), 1);
    }

    #[test]
    fn elem_count_is_the_product_of_dims() {
        assert_eq!(Shape::from([2, 3, 4]).elem_count(), 24);
        assert_eq!(Shape::from([5]).elem_count(), 5);
    }

    #[test]
    fn a_zero_dimension_means_zero_elements() {
        assert_eq!(Shape::from([3, 0, 4]).elem_count(), 0);
    }

    #[test]
    fn strides_are_row_major_with_the_last_dim_contiguous() {
        assert_eq!(Shape::from([2, 3, 4]).strides(), vec![12, 4, 1]);
        assert_eq!(Shape::from([5]).strides(), vec![1]);
        assert!(Shape::scalar().strides().is_empty());
    }

    #[test]
    fn reshape_preserves_element_count_or_errors() {
        let s = Shape::from([2, 6]);
        assert_eq!(s.reshape([3, 4]).unwrap(), Shape::from([3, 4]));
        assert_eq!(s.reshape([12]).unwrap(), Shape::from([12]));
        assert!(s.reshape([5, 5]).is_err());
    }

    #[test]
    fn broadcast_follows_the_numpy_right_aligned_rule() {
        // Equal ranks, a 1 that stretches.
        assert_eq!(
            Shape::from([3, 1]).broadcast(&Shape::from([3, 4])).unwrap(),
            Shape::from([3, 4])
        );
        // Different ranks: missing leading dims are treated as 1.
        assert_eq!(
            Shape::from([4]).broadcast(&Shape::from([3, 4])).unwrap(),
            Shape::from([3, 4])
        );
        // Both stretch, in different dimensions.
        assert_eq!(
            Shape::from([3, 1]).broadcast(&Shape::from([1, 4])).unwrap(),
            Shape::from([3, 4])
        );
    }

    #[test]
    fn broadcast_rejects_incompatible_dims() {
        assert!(Shape::from([3, 2]).broadcast(&Shape::from([3, 4])).is_err());
    }

    #[test]
    fn broadcasting_against_a_scalar_is_the_identity() {
        let s = Shape::from([2, 3]);
        assert_eq!(s.broadcast(&Shape::scalar()).unwrap(), s);
        assert_eq!(Shape::scalar().broadcast(&s).unwrap(), s);
    }
}
