use crate::{Abs, FloatingScalar, FloatingVector2, Pow, Vector2, ZeroOneTwo};

type Scalar<T> = <<T as Circle>::Vector as Vector2>::Scalar;

/// Trait for manipulating circles
pub trait Circle: Copy
where
    Scalar<Self>: FloatingScalar,
{
    /// The vector type
    type Vector: FloatingVector2;
    /// Create a new circle from a center coordinate and a radius
    fn new(center: Self::Vector, radius: Scalar<Self>) -> Self;
    /// Get the circle's center
    fn center(self) -> Self::Vector;
    /// Get the circle's radius
    fn radius(self) -> Scalar<Self>;
    /// Map this circle to a circle of another type
    fn map_into<C>(self) -> C
    where
        C: Circle,
        Scalar<C>: FloatingScalar + From<Scalar<Self>>,
    {
        C::new(
            C::Vector::new(
                Scalar::<C>::from(self.center().x()),
                Scalar::<C>::from(self.center().y()),
            ),
            Scalar::<C>::from(self.radius()),
        )
    }
    /// Map this circle to a circle of another type using a function
    fn map_with<C, F>(self, mut f: F) -> C
    where
        C: Circle,
        Scalar<C>: FloatingScalar,
        F: FnMut(Scalar<Self>) -> <<C as Circle>::Vector as Vector2>::Scalar,
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
    fn with_radius(self, radius: Scalar<Self>) -> Self {
        Self::new(self.center(), radius)
    }
    /// Get the circle's diameter
    fn diameter(self) -> Scalar<Self> {
        self.radius() * Scalar::<Self>::TWO
    }
    /// Get the circle's circumference
    fn circumference(self) -> Scalar<Self> {
        self.radius() * Scalar::<Self>::TAU
    }
    /// Get the circle's area
    fn area(self) -> Scalar<Self> {
        self.radius().pow(Scalar::<Self>::TWO) * Scalar::<Self>::pi()
    }
    /// Get the circle that is this one translated by some vector
    fn translated(self, offset: Self::Vector) -> Self {
        self.with_center(self.center().add(offset))
    }
    /// Get the circle that is this one with a scalar-scaled size
    fn scaled(self, scale: Scalar<Self>) -> Self {
        self.with_radius(self.radius() * scale)
    }
    /// Get the smallest square that this circle fits inside
    fn to_square(self) -> [Scalar<Self>; 4] {
        let radius = self.radius();
        [
            self.center().x() - radius,
            self.center().y() - radius,
            radius * Scalar::<Self>::TWO,
            radius * Scalar::<Self>::TWO,
        ]
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
    type Vector = V;
    fn new(center: Self::Vector, radius: Scalar<Self>) -> Self {
        (center, radius)
    }
    fn center(self) -> Self::Vector {
        self.0
    }
    fn radius(self) -> Scalar<Self> {
        self.1
    }
}
