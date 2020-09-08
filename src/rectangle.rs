use std::vec;

use crate::{Abs, Pair, Scalar, Vector2, ZeroOneTwo};

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
    /// The scalar type
    type Scalar: Scalar;
    /// The vector type
    type Vector: Vector2<Scalar = Self::Scalar>;
    /// Create a new rectangle from a top-left corner position and a size
    fn new(top_left: Self::Vector, size: Self::Vector) -> Self;
    /// Get the top-left corner position
    fn top_left(self) -> Self::Vector;
    /// Get the size
    fn size(self) -> Self::Vector;
    /// Create a new square from a top-left corner position and a side length
    fn square(top_left: Self::Vector, side_length: Self::Scalar) -> Self {
        Self::new(top_left, Self::Vector::square(side_length))
    }
    /// Create a new rectangle from a center position and a size
    fn centered(center: Self::Vector, size: Self::Vector) -> Self {
        Self::new(center.sub(size.div(Self::Scalar::TWO)), size)
    }
    /// Create a new square from a top-left corner position and a side length
    fn square_centered(center: Self::Vector, side_length: Self::Scalar) -> Self {
        Self::centered(center, Self::Vector::square(side_length))
    }
    /// Map this rectangle to a rectangle of another type
    fn map<R>(self) -> R
    where
        R: Rectangle,
        R::Scalar: From<Self::Scalar>,
    {
        R::new(
            R::Vector::new(R::Scalar::from(self.left()), R::Scalar::from(self.top())),
            R::Vector::new(
                R::Scalar::from(self.width()),
                R::Scalar::from(self.height()),
            ),
        )
    }
    /// Map this rectangle to a `[f32;4]`
    ///
    /// This is an alias for `Rectangle::map::<[f32;4]>()` that is more concise
    fn map_f32(self) -> [f32; 4]
    where
        f32: From<Self::Scalar>,
    {
        self.map()
    }
    /// Map this rectangle to a `[f64;4]`
    ///
    /// This is an alias for `Rectangle::map::<[f64;4]>()` that is more concise
    fn map_f64(self) -> [f64; 4]
    where
        f64: From<Self::Scalar>,
    {
        self.map()
    }
    /// Map this rectangle to a rectangle of another type using a function
    fn map_with<R, F>(self, mut f: F) -> R
    where
        R: Rectangle,
        F: FnMut(Self::Scalar) -> <<R as Rectangle>::Vector as Vector2>::Scalar,
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
    fn top(self) -> Self::Scalar {
        self.top_left().y()
    }
    /// Get the bottom y
    fn bottom(self) -> Self::Scalar {
        self.top_left().y() + self.size().y()
    }
    /// Get the left x
    fn left(self) -> Self::Scalar {
        self.top_left().x()
    }
    /// Get the right x
    fn right(self) -> Self::Scalar {
        self.top_left().x() + self.size().x()
    }
    /// Get the absolute top y
    fn abs_top(self) -> Self::Scalar {
        self.abs_top_left().y()
    }
    /// Get the absolute bottom y
    fn abs_bottom(self) -> Self::Scalar {
        self.abs_top_left().y() + self.abs_size().y()
    }
    /// Get the absolute left x
    fn abs_left(self) -> Self::Scalar {
        self.abs_top_left().x()
    }
    /// Get the absolute right x
    fn abs_right(self) -> Self::Scalar {
        self.abs_top_left().x() + self.abs_size().x()
    }
    /// Get the width
    fn width(self) -> Self::Scalar {
        self.size().x()
    }
    /// Get the height
    fn height(self) -> Self::Scalar {
        self.size().y()
    }
    /// Get the absolute width
    fn abs_width(self) -> Self::Scalar {
        self.abs_size().x()
    }
    /// Get the absolute height
    fn abs_height(self) -> Self::Scalar {
        self.abs_size().y()
    }
    /// Get the position of the center
    fn center(self) -> Self::Vector {
        self.top_left().add(self.size().div(Self::Scalar::TWO))
    }
    /// Transform the rectangle into one with a different top-left corner position
    fn with_top_left(self, top_left: Self::Vector) -> Self {
        Self::new(top_left, self.size())
    }
    /// Transform the rectangle into one with a different center position
    fn with_center(self, center: Self::Vector) -> Self {
        Self::centered(center, self.size())
    }
    /// Transform the rectangle into one with a different size
    fn with_size(self, size: Self::Vector) -> Self {
        Self::new(self.top_left(), size)
    }
    /// Get the perimeter
    fn perimeter(self) -> Self::Scalar {
        self.width() * Self::Scalar::TWO + self.height() * Self::Scalar::TWO
    }
    /// Get the area
    fn area(self) -> Self::Scalar {
        self.width() * self.height()
    }
    /// Get the rectangle that is this one translated by some vector
    fn translated(self, offset: Self::Vector) -> Self {
        self.with_top_left(self.top_left().add(offset))
    }
    /// Get the rectangle that is this one with a scalar-scaled size
    fn scaled(self, scale: Self::Scalar) -> Self {
        self.with_size(self.size().mul(scale))
    }
    /// Get the rectangle that is this one with a vector-scaled size
    fn scaled2(self, scale: Self::Vector) -> Self {
        self.with_size(self.size().mul2(scale))
    }
    /// Get an iterator over the rectangle's four corners
    fn corners(self) -> vec::IntoIter<Self::Vector> {
        vec![
            self.top_left(),
            self.top_right(),
            self.bottom_right(),
            self.bottom_left(),
        ]
        .into_iter()
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
    fn inner_margin(self, margin: Self::Scalar) -> Self {
        self.inner_margins([margin; 4])
    }
    /// Get the rectangle that is inside this one with the given margins
    ///
    /// Margins should be ordered `[left, right, top, bottom]`
    fn inner_margins(self, [left, right, top, bottom]: [Self::Scalar; 4]) -> Self {
        Self::new(
            self.abs_top_left().add(Self::Vector::new(left, top)),
            self.abs_size()
                .sub(Self::Vector::new(left + right, top + bottom)),
        )
    }
    /// Get the rectangle that is outside this one with the given
    /// margin on all sides
    fn outer_margin(self, margin: Self::Scalar) -> Self {
        self.outer_margins([margin; 4])
    }
    /// Get the rectangle that is outside this one with the given margins
    ///
    /// Margins should be ordered `[left, right, top, bottom]`
    fn outer_margins(self, [left, right, top, bottom]: [Self::Scalar; 4]) -> Self {
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
    type Scalar = <P::Item as Vector2>::Scalar;
    type Vector = P::Item;
    fn new(top_left: Self::Vector, size: Self::Vector) -> Self {
        Self::from_items(top_left, size)
    }
    fn top_left(self) -> Self::Vector {
        self.to_pair().0
    }
    fn size(self) -> Self::Vector {
        self.to_pair().1
    }
}
