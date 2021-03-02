use std::ops::{Add, Mul};

use crate::{FloatingScalar, Pair, Trig, Trio, Vector2, ZeroOneTwo};

/**
Trait for defining vector transformations

Transforms should be able to be chained without allocating
extra space. The standard way to do this is with a matrix.
For transforming 2D vectors, a 2×3 matrix can be used.
*/
pub trait Transform: Sized {
    /// The scalar type
    type Scalar: FloatingScalar;
    /// Create a new identity transform
    fn identity() -> Self;
    /// Chain this transform with another
    fn then(self, next: Self) -> Self;
    /// Chain another transform with this one
    fn but_first(self, prev: Self) -> Self {
        prev.then(self)
    }
    /// Apply this transform to a vector
    fn apply<V>(self, vector: V) -> V
    where
        V: Vector2<Scalar = Self::Scalar>;
    /// Create a translation from an offset vector
    fn new_translate<V>(offset: V) -> Self
    where
        V: Vector2<Scalar = Self::Scalar>;
    /// Create a rotation from a radian angle
    fn new_rotate(radians: Self::Scalar) -> Self;
    /// Create a scaling from a ratio vector
    fn new_scale<V>(ratio: V) -> Self
    where
        V: Vector2<Scalar = Self::Scalar>;
    /// Translate the transform
    fn translate<V>(self, offset: V) -> Self
    where
        V: Vector2<Scalar = Self::Scalar>,
    {
        self.then(Self::new_translate(offset))
    }
    /// Rotate the transform
    fn rotate(self, radians: Self::Scalar) -> Self {
        self.then(Self::new_rotate(radians))
    }
    /// Scale the transform
    fn scale<V>(self, ratio: V) -> Self
    where
        V: Vector2<Scalar = Self::Scalar>,
    {
        self.then(Self::new_scale(ratio))
    }
    /// Uniformly scale the transform
    fn zoom(self, ratio: Self::Scalar) -> Self {
        self.scale([ratio; 2])
    }
    /// Rotate the transform about a pivot
    fn rotate_about<V>(self, radians: Self::Scalar, pivot: V) -> Self
    where
        V: Vector2<Scalar = Self::Scalar>,
    {
        self.translate(pivot.neg()).rotate(radians).translate(pivot)
    }
}

impl<M, C> Transform for M
where
    M: Pair<Item = C>,
    C: Trio + Copy,
    C::Item: FloatingScalar,
{
    type Scalar = C::Item;
    fn identity() -> Self {
        M::from_items(
            C::from_items(C::Item::ONE, C::Item::ZERO, C::Item::ZERO),
            C::from_items(C::Item::ZERO, C::Item::ONE, C::Item::ZERO),
        )
    }
    fn then(self, next: Self) -> Self {
        let (a1, a2) = next.into_pair();
        let (b1, b2) = self.into_pair();
        let (a11, a12, a13) = a1.into_trio();
        let (a21, a22, a23) = a2.into_trio();
        let (b11, b12, b13) = b1.into_trio();
        let (b21, b22, b23) = b2.into_trio();
        M::from_items(
            C::from_items(
                a11 * b11 + a12 * b21,
                a11 * b12 + a12 * b22,
                a11 * b13 + a12 * b23 + a13,
            ),
            C::from_items(
                a21 * b11 + a22 * b21,
                a21 * b12 + a22 * b22,
                a21 * b13 + a22 * b23 + a23,
            ),
        )
    }
    fn apply<V>(self, vector: V) -> V
    where
        V: Vector2<Scalar = Self::Scalar>,
    {
        let vtrio = C::from_items(vector.x(), vector.y(), V::Scalar::ONE);
        let (a, b) = self.into_pair();
        let xp: C = a.pairwise(vtrio, Mul::mul);
        let yp: C = b.pairwise(vtrio, Mul::mul);
        let x = xp.trio_iter().fold(Self::Scalar::ZERO, Add::add);
        let y = yp.trio_iter().fold(Self::Scalar::ZERO, Add::add);
        V::new(x, y)
    }
    fn new_translate<V>(v: V) -> Self
    where
        V: Vector2<Scalar = Self::Scalar>,
    {
        M::from_items(
            C::from_items(C::Item::ONE, C::Item::ZERO, v.x()),
            C::from_items(C::Item::ZERO, C::Item::ONE, v.y()),
        )
    }
    fn new_rotate(radians: Self::Scalar) -> Self {
        let c = radians.cos();
        let s = radians.sin();
        M::from_items(
            C::from_items(c, -s, C::Item::ZERO),
            C::from_items(s, c, C::Item::ZERO),
        )
    }
    fn new_scale<V>(ratio: V) -> Self
    where
        V: Vector2<Scalar = Self::Scalar>,
    {
        M::from_items(
            C::from_items(ratio.x(), C::Item::ZERO, C::Item::ZERO),
            C::from_items(C::Item::ZERO, ratio.y(), C::Item::ZERO),
        )
    }
}
