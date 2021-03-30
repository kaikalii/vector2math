//! Module for the [`Rectangle`] trait

use crate::{Pair, Scalar as _, Vector2};

/// The scalar type of a [`Rectangle`]
pub type Scalar<T> = <<T as Rectangle>::Vector as Vector2>::Scalar;

/**
Trait for manipulating axis-aligned rectangles

Because the primary expected use for this crate is in 2D graphics and alignment implementations,
a coordinate system where the positive Y direction is "down" is assumed.

# Note
Methods of the form `abs_*` account for the case where the size is negative.
If the size is not negative, they are identical to their non-`abs_*` counterparts.
```
use vector2math::*;

let pos_size = [1, 2, 3, 4];
assert_eq!(pos_size.right(), pos_size.abs_right());

let neg_size = [1, 2, -3, -4];
assert_ne!(neg_size.right(), neg_size.abs_right());

let points = vec![
    [-1, 0],
    [1, 5],
    [3, 2],
];
let bounding_rect: [i32; 4] = Rectangle::bounding(points).unwrap();
assert_eq!(
    bounding_rect,
    [-1, 0, 4, 5]
);
```
*/
pub trait Rectangle: Copy {
    /// The vector type
    type Vector: Vector2;
    /// Create a new rectangle from a top-left corner position and a size
    fn new(top_left: Self::Vector, size: Self::Vector) -> Self;
    /// Get the top-left corner position
    fn top_left(self) -> Self::Vector;
    /// Get the size
    fn size(self) -> Self::Vector;
    /// Create a new square from a top-left corner position and a side length
    fn square(top_left: Self::Vector, side_length: Scalar<Self>) -> Self {
        Self::new(top_left, Self::Vector::square(side_length))
    }
    /// Create a new rectangle from a center position and a size
    fn centered(center: Self::Vector, size: Self::Vector) -> Self {
        Self::new(center.sub(size.div(Scalar::<Self>::TWO)), size)
    }
    /// Create a new square from a top-left corner position and a side length
    fn square_centered(center: Self::Vector, side_length: Scalar<Self>) -> Self {
        Self::centered(center, Self::Vector::square(side_length))
    }
    /// Map this rectangle to a rectangle of another type
    fn map_into<R>(self) -> R
    where
        R: Rectangle,
        Scalar<R>: From<Scalar<Self>>,
    {
        R::new(
            R::Vector::new(
                Scalar::<R>::from(self.left()),
                Scalar::<R>::from(self.top()),
            ),
            R::Vector::new(
                Scalar::<R>::from(self.width()),
                Scalar::<R>::from(self.height()),
            ),
        )
    }
    /// Map this rectangle to a `[Scalar<Self>; 4]`
    ///
    /// This is an alias for `Rectangle::map_into::<[Scalar<Self>; 4]>()` that is more concise
    fn map_rect(self) -> [Scalar<Self>; 4] {
        self.map_into()
    }
    /// Map this rectangle to a rectangle of another type using a function
    fn map_with<R, F>(self, mut f: F) -> R
    where
        R: Rectangle,
        F: FnMut(Scalar<Self>) -> <<R as Rectangle>::Vector as Vector2>::Scalar,
    {
        R::new(
            R::Vector::new(f(self.left()), f(self.top())),
            R::Vector::new(f(self.width()), f(self.height())),
        )
    }
    /// Get the absolute size
    fn abs_size(self) -> Self::Vector {
        Self::Vector::new(self.size().x().abs(), self.size().y().abs())
    }
    /// Get the top-right corner position
    fn top_right(self) -> Self::Vector {
        Self::Vector::new(self.top_left().x() + self.size().x(), self.top_left().y())
    }
    /// Get the bottom-left corner position
    fn bottom_left(self) -> Self::Vector {
        Self::Vector::new(self.top_left().x(), self.top_left().y() + self.size().y())
    }
    /// Get the bottom-right corner position
    fn bottom_right(self) -> Self::Vector {
        self.top_left().add(self.size())
    }
    /// Get the absolute top-left corner position
    fn abs_top_left(self) -> Self::Vector {
        let tl = self.top_left();
        let size = self.size();
        Self::Vector::new(
            tl.x().minn(tl.x() + size.x()),
            tl.y().minn(tl.y() + size.y()),
        )
    }
    /// Get the absolute top-right corner position
    fn abs_top_right(self) -> Self::Vector {
        Self::Vector::new(
            self.abs_top_left().x() + self.abs_size().x(),
            self.abs_top_left().y(),
        )
    }
    /// Get the absolute bottom-left corner position
    fn abs_bottom_left(self) -> Self::Vector {
        Self::Vector::new(
            self.abs_top_left().x(),
            self.abs_top_left().y() + self.abs_size().y(),
        )
    }
    /// Get the absolute bottom-right corner position
    fn abs_bottom_right(self) -> Self::Vector {
        self.abs_top_left().add(self.abs_size())
    }
    /// Get the top y
    fn top(self) -> Scalar<Self> {
        self.top_left().y()
    }
    /// Get the bottom y
    fn bottom(self) -> Scalar<Self> {
        self.top_left().y() + self.size().y()
    }
    /// Get the left x
    fn left(self) -> Scalar<Self> {
        self.top_left().x()
    }
    /// Get the right x
    fn right(self) -> Scalar<Self> {
        self.top_left().x() + self.size().x()
    }
    /// Get the absolute top y
    fn abs_top(self) -> Scalar<Self> {
        self.abs_top_left().y()
    }
    /// Get the absolute bottom y
    fn abs_bottom(self) -> Scalar<Self> {
        self.abs_top_left().y() + self.abs_size().y()
    }
    /// Get the absolute left x
    fn abs_left(self) -> Scalar<Self> {
        self.abs_top_left().x()
    }
    /// Get the absolute right x
    fn abs_right(self) -> Scalar<Self> {
        self.abs_top_left().x() + self.abs_size().x()
    }
    /// Get the width
    fn width(self) -> Scalar<Self> {
        self.size().x()
    }
    /// Get the height
    fn height(self) -> Scalar<Self> {
        self.size().y()
    }
    /// Get the absolute width
    fn abs_width(self) -> Scalar<Self> {
        self.abs_size().x()
    }
    /// Get the absolute height
    fn abs_height(self) -> Scalar<Self> {
        self.abs_size().y()
    }
    /// Get the position of the center
    fn center(self) -> Self::Vector {
        self.top_left().add(self.size().div(Scalar::<Self>::TWO))
    }
    /// Transform the rectangle into one with a different size
    fn with_size(self, size: Self::Vector) -> Self {
        Self::new(self.top_left(), size)
    }
    /// Get the rectangle that is this one with a different top bound
    fn with_top(self, top: Scalar<Self>) -> Self {
        Self::new(
            Self::Vector::new(self.left(), top),
            Self::Vector::new(self.width(), self.bottom() - top),
        )
    }
    /// Get the rectangle that is this one with a different bottom bound
    fn with_bottom(self, bottom: Scalar<Self>) -> Self {
        Self::new(
            self.top_left(),
            Self::Vector::new(self.width(), bottom - self.top()),
        )
    }
    /// Get the rectangle that is this one with a different left bound
    fn with_left(self, left: Scalar<Self>) -> Self {
        Self::new(
            Self::Vector::new(left, self.top()),
            Self::Vector::new(self.right() - left, self.height()),
        )
    }
    /// Get the rectangle that is this one with a different right bound
    fn with_right(self, right: Scalar<Self>) -> Self {
        Self::new(
            self.top_left(),
            Self::Vector::new(right - self.left(), self.height()),
        )
    }
    /// Get the perimeter
    fn perimeter(self) -> Scalar<Self> {
        self.width() * Scalar::<Self>::TWO + self.height() * Scalar::<Self>::TWO
    }
    /// Get the area
    fn area(self) -> Scalar<Self> {
        self.width() * self.height()
    }
    /// Get the rectangle that is this one translated by some vector
    fn translated(self, offset: Self::Vector) -> Self {
        Self::new(self.top_left().add(offset), self.size())
    }
    /// Get the rectangle that is this one with a scalar-scaled size
    fn scaled(self, scale: Scalar<Self>) -> Self {
        self.with_size(self.size().mul(scale))
    }
    /// Get the rectangle that is this one with a vector-scaled size
    fn scaled2(self, scale: Self::Vector) -> Self {
        self.with_size(self.size().mul2(scale))
    }
    /// Get an array the rectangle's four corners, clockwise from top-left
    fn corners(self) -> [Self::Vector; 4] {
        [
            self.top_left(),
            self.top_right(),
            self.bottom_right(),
            self.bottom_left(),
        ]
    }
    /// Check that the rectangle contains the given point. Includes edges.
    fn contains(self, point: Self::Vector) -> bool {
        let in_x_bounds = self.abs_left() <= point.x() && point.x() <= self.abs_right();
        let in_y_bounds = || self.abs_top() <= point.y() && point.y() <= self.abs_bottom();
        in_x_bounds && in_y_bounds()
    }
    /// Check that the rectangle contains all points
    fn contains_all<I>(self, points: I) -> bool
    where
        I: IntoIterator<Item = Self::Vector>,
    {
        points.into_iter().all(|point| self.contains(point))
    }
    /// Check that the rectangle contains any point
    fn contains_any<I>(self, points: I) -> bool
    where
        I: IntoIterator<Item = Self::Vector>,
    {
        points.into_iter().any(|point| self.contains(point))
    }
    /// Get the smallest rectangle that contains all the points
    ///
    /// Returns `None` if the iterator is empty
    fn bounding<I>(points: I) -> Option<Self>
    where
        I: IntoIterator<Item = Self::Vector>,
    {
        let mut points = points.into_iter();
        if let Some(first) = points.next() {
            let mut tl = first;
            let mut br = first;
            for point in points {
                tl = Self::Vector::new(tl.x().minn(point.x()), tl.y().minn(point.y()));
                br = Self::Vector::new(br.x().maxx(point.x()), br.y().maxx(point.y()));
            }
            Some(Self::new(tl, br.sub(tl)))
        } else {
            None
        }
    }
    /// Get the rectangle that is inside this one with the given
    /// margin on all sides
    fn inner_margin(self, margin: Scalar<Self>) -> Self {
        self.inner_margins([margin; 4])
    }
    /// Get the rectangle that is inside this one with the given margins
    ///
    /// Margins should be ordered `[left, right, top, bottom]`
    fn inner_margins(self, [left, right, top, bottom]: [Scalar<Self>; 4]) -> Self {
        Self::new(
            self.abs_top_left().add(Self::Vector::new(left, top)),
            self.abs_size()
                .sub(Self::Vector::new(left + right, top + bottom)),
        )
    }
    /// Get the rectangle that is outside this one with the given
    /// margin on all sides
    fn outer_margin(self, margin: Scalar<Self>) -> Self {
        self.outer_margins([margin; 4])
    }
    /// Get the rectangle that is outside this one with the given margins
    ///
    /// Margins should be ordered `[left, right, top, bottom]`
    fn outer_margins(self, [left, right, top, bottom]: [Scalar<Self>; 4]) -> Self {
        Self::new(
            self.abs_top_left().sub(Self::Vector::new(left, top)),
            self.abs_size()
                .add(Self::Vector::new(left + right, top + bottom)),
        )
    }
}

impl<P> Rectangle for P
where
    P: Pair + Copy,
    P::Item: Vector2,
{
    type Vector = P::Item;
    fn new(top_left: Self::Vector, size: Self::Vector) -> Self {
        Self::from_items(top_left, size)
    }
    fn top_left(self) -> Self::Vector {
        self.into_pair().0
    }
    fn size(self) -> Self::Vector {
        self.into_pair().1
    }
}
