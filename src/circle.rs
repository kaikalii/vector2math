use crate::{Abs, FloatingScalar, FloatingVector2, Pow, Rectangle, Vector2, ZeroOneTwo};

/// Trait for manipulating circles
pub trait Circle: Copy {
    /// The scalar type
    type Scalar: FloatingScalar;
    /// The vector type
    type Vector: FloatingVector2<Scalar = Self::Scalar>;
    /// Create a new circle from a center coordinate and a radius
    fn new(center: Self::Vector, radius: Self::Scalar) -> Self;
    /// Get the circle's center
    fn center(self) -> Self::Vector;
    /// Get the circle's radius
    fn radius(self) -> Self::Scalar;
    /// Map this circle to a circle of another type
    fn map<C>(self) -> C
    where
        C: Circle,
        C::Scalar: From<Self::Scalar>,
    {
        C::new(
            C::Vector::new(
                C::Scalar::from(self.center().x()),
                C::Scalar::from(self.center().y()),
            ),
            C::Scalar::from(self.radius()),
        )
    }
    /// Map this circle to a circle of another type using a function
    fn map_with<C, F>(self, mut f: F) -> C
    where
        C: Circle,
        F: FnMut(Self::Scalar) -> <<C as Circle>::Vector as Vector2>::Scalar,
    {
        C::new(
            C::Vector::new(f(self.center().x()), f(self.center().y())),
            f(self.radius()),
        )
    }
    /// Transform the circle into one with a different top-left corner position
    fn with_center(self, center: Self::Vector) -> Self {
        Self::new(center, self.radius())
    }
    /// Transform the circle into one with a different size
    fn with_radius(self, radius: Self::Scalar) -> Self {
        Self::new(self.center(), radius)
    }
    /// Get the circle's diameter
    fn diameter(self) -> Self::Scalar {
        self.radius() * Self::Scalar::TWO
    }
    /// Get the circle's circumference
    fn circumference(self) -> Self::Scalar {
        self.radius() * Self::Scalar::TAU
    }
    /// Get the circle's area
    fn area(self) -> Self::Scalar {
        self.radius().pow(Self::Scalar::TWO) * Self::Scalar::pi()
    }
    /// Get the circle that is this one translated by some vector
    fn translated(self, offset: Self::Vector) -> Self {
        self.with_center(self.center().add(offset))
    }
    /// Get the circle that is this one with a scalar-scaled size
    fn scaled(self, scale: Self::Scalar) -> Self {
        self.with_radius(self.radius() * scale)
    }
    /// Get the smallest square that this circle fits inside
    fn to_square<R>(self) -> R
    where
        R: Rectangle<Scalar = Self::Scalar, Vector = Self::Vector>,
    {
        R::new(
            self.center().sub(R::Vector::square(self.radius())),
            R::Vector::square(self.radius() * R::Scalar::TWO),
        )
    }
    /// Check that the circle contains the given point
    fn contains(self, point: Self::Vector) -> bool {
        self.center().dist(point) <= self.radius().abs()
    }
    /// Alias for `Rectangle::contains`
    ///
    /// Useful when `contains` is ambiguous
    fn cntains(self, point: Self::Vector) -> bool {
        self.contains(point)
    }
    /// Check that the circle contains all points
    fn contains_all<I>(self, points: I) -> bool
    where
        I: IntoIterator<Item = Self::Vector>,
    {
        points.into_iter().all(|point| self.contains(point))
    }
    /// Check that the circle contains any point
    fn contains_any<I>(self, points: I) -> bool
    where
        I: IntoIterator<Item = Self::Vector>,
    {
        points.into_iter().any(|point| self.contains(point))
    }
}

impl<S, V> Circle for (V, S)
where
    S: FloatingScalar,
    V: FloatingVector2<Scalar = S>,
{
    type Scalar = S;
    type Vector = V;
    fn new(center: Self::Vector, radius: Self::Scalar) -> Self {
        (center, radius)
    }
    fn center(self) -> Self::Vector {
        self.0
    }
    fn radius(self) -> Self::Scalar {
        self.1
    }
}