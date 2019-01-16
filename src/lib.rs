#![deny(missing_docs)]
#![deny(unsafe_code)]

/*!
This crate provides traits for doing 2D vector geometry operations using standard types

# Usage

Simple vector math is implemented for vectors with the following scalar types:
* `u8`-`u128`
* `usize`
* `i8`-`i128`
* `isize`
* `f32`
* `f64`
* Any type that implements one or more of this crate's `Scalar` traits

Vectors can be of the following forms:
* `[T; 2]`
* `(T, T)`
* Any type that implements one or more of this crate's `Vector2` traits

Many 2D Vector operations are supported. Vectors do not necessarily need
to be the same type to allow operation. They need only have the same `Scalar` type.
The output type will be the same as the first argument.
```
use vector2math::*;

let a = [2, 6];
let b = (4, -1);
assert_eq!(2, a.x());
assert_eq!(-1, b.y());
assert_eq!([-2, -6], a.neg());
assert_eq!([6, 5], a.add(b));
assert_eq!([-2, 7], a.sub(b));
assert_eq!((12, -3), b.mul(3));
assert_eq!((8, -6), b.mul2(a));
assert_eq!([1, 3], a.div(2));
assert_eq!([0, -6], a.div2(b));
```

Floating-point vectors have additional operations:
```
use vector2math::*;

assert_eq!(5.0, [3.0, 4.0].mag());
assert_eq!(10.0, [-1.0, -2.0].dist([5.0, 6.0]));
let rotation_calculation = [1.0, 0.0].rotate_about([0.0; 2], std::f64::consts::PI / 4.0);
let rotation_solution = [2f64.powf(0.5) / 2.0; 2];
assert!(rotation_calculation.sub(rotation_solution).mag() < std::f64::EPSILON);
```

Many types can be used to define axis-aligned rectangles:
* `[[T; 2]; 2]`
* `[(T, T); 2]`
* `((T, T), (T, T))`
* `([T; 2], [T; 2])`
* `[T; 4]`
* `(T, T, T, T)`
* Any type that implements this crate's `Pair` trait where the associated `Item` type implements `Vector2`.
```
use vector2math::*;

let rect = [1i32, 2, 4, 6];
assert_eq!([1, 2], rect.top_left());
assert_eq!([4, 6], rect.size());
assert_eq!([3, 5], rect.center());
assert_eq!(20, rect.perimeter());
assert_eq!(24, rect.area());
assert!(rect.contains([3, 5]));
```

Both vector and rectangle types can be easily mapped to different types:
```
use vector2math::*;

let arrayf32: [f32; 2] = [1.0, 2.0];
let arrayf64: [f64; 2] = arrayf32.map();
let pairf64: (f64, f64) = arrayf64.map();
let arrayi16: [i16; 2] = pairf64.map_with(|f| f as i16);
assert_eq!(arrayf32, arrayi16.map::<[f32; 2]>());

let weird_rect = [(0.0, 1.0), (2.0, 5.0)];
let normal_rectf32: [f32; 4] = weird_rect.map();
let normal_rectf64: [f32; 4] = normal_rectf32.map();
let normal_rectu8: [u8; 4] = normal_rectf32.map_with(|f| f as u8);
assert_eq!([0, 1, 2, 5], normal_rectu8);
```

Implementing `Vector2` and `Rectangle` traits for your own types is simple.
Just make sure that your type is `Copy`
```
use vector2math::*;

#[derive(Clone, Copy)]
struct MyVector {
    x: f64,
    y: f64,
}

impl Vector2 for MyVector {
    type Scalar = f64;
    fn new(x: f64, y: f64) -> Self {
        MyVector { x, y }
    }
    fn x(self) -> f64 {
        self.x
    }
    fn y(self) -> f64 {
        self.y
    }
}

#[derive(Clone, Copy)]
struct MyRectangle {
    top_left: MyVector,
    size: MyVector,
}

impl Rectangle for MyRectangle {
    type Scalar = f64;
    type Vector = MyVector;
    fn new(top_left: MyVector, size: MyVector) -> Self {
        MyRectangle { top_left, size }
    }
    fn top_left(self) -> MyVector {
        self.top_left
    }
    fn size(self) -> MyVector {
        self.size
    }
}

let rect: MyRectangle = [1, 2, 3, 4].map();
assert_eq!(12.0, rect.area());
assert_eq!(6.0, rect.bottom());
```
*/

use std::{
    ops::{Add, Div, Mul, Neg, Sub},
    vec,
};

/// Trait for defining a pair of items of the same type.
///
/// This trait is meant to generalize having two similar things.
/// It is implemented for `(T, T)` and `[T; 2]` with `Item = T`.
/// However, because a pair does not necessarily have to be an
/// Actual *pair* It is also implemented for `(T, T, T, T)` and
/// `[T; 4]` with `Item = (T, T)` and `Item = [T; 2]` respectively.
pub trait Pair {
    /// The type of the pair's item
    type Item;
    /// Get the first thing
    fn first(self) -> Self::Item;
    /// Get the second thing
    fn second(self) -> Self::Item;
    /// Create a pair from two items
    fn from_items(a: Self::Item, b: Self::Item) -> Self;
}

impl<T> Pair for (T, T)
where
    T: Clone,
{
    type Item = T;
    fn first(self) -> Self::Item {
        self.0.clone()
    }
    fn second(self) -> Self::Item {
        self.1.clone()
    }
    fn from_items(a: Self::Item, b: Self::Item) -> Self {
        (a, b)
    }
}

impl<T> Pair for [T; 2]
where
    T: Clone,
{
    type Item = T;
    fn first(self) -> Self::Item {
        self[0].clone()
    }
    fn second(self) -> Self::Item {
        self[1].clone()
    }
    fn from_items(a: Self::Item, b: Self::Item) -> Self {
        [a, b]
    }
}

impl<T> Pair for (T, T, T, T)
where
    T: Clone,
{
    type Item = (T, T);
    fn first(self) -> Self::Item {
        (self.0.clone(), self.1.clone())
    }
    fn second(self) -> Self::Item {
        (self.2.clone(), self.3.clone())
    }
    fn from_items(a: Self::Item, b: Self::Item) -> Self {
        (a.0, a.1, b.0, b.1)
    }
}

impl<T> Pair for [T; 4]
where
    T: Clone,
{
    type Item = [T; 2];
    fn first(self) -> Self::Item {
        [self[0].clone(), self[1].clone()]
    }
    fn second(self) -> Self::Item {
        [self[2].clone(), self[3].clone()]
    }
    fn from_items(a: Self::Item, b: Self::Item) -> Self {
        [a[0].clone(), a[1].clone(), b[0].clone(), b[1].clone()]
    }
}

/// Trait for getting the sine of a number
pub trait Sin {
    /// The output type
    type Output;
    /// Get the sine
    fn sin(self) -> Self::Output;
}

impl Sin for f32 {
    type Output = f32;
    fn sin(self) -> Self::Output {
        f32::sin(self)
    }
}

impl Sin for f64 {
    type Output = f64;
    fn sin(self) -> Self::Output {
        f64::sin(self)
    }
}

/// Trait for getting the cosine of a number
pub trait Cos {
    /// The output type
    type Output;
    /// Get the cosine
    fn cos(self) -> Self::Output;
}

impl Cos for f32 {
    type Output = f32;
    fn cos(self) -> Self::Output {
        f32::cos(self)
    }
}

impl Cos for f64 {
    type Output = f64;
    fn cos(self) -> Self::Output {
        f64::cos(self)
    }
}

/// Trait for retrieving an absolute value of a number
pub trait Abs {
    /// Get the absolute value of the number
    fn abs(self) -> Self;
}

macro_rules! abs_unsigned_impl {
    ($type:ty) => {
        impl Abs for $type {
            fn abs(self) -> Self {
                self
            }
        }
    };
}

macro_rules! abs_signed_impl {
    ($type:ty) => {
        impl Abs for $type {
            fn abs(self) -> Self {
                Self::abs(self)
            }
        }
    };
}

abs_unsigned_impl! {u8}
abs_unsigned_impl! {u16}
abs_unsigned_impl! {u32}
abs_unsigned_impl! {u64}
abs_unsigned_impl! {u128}
abs_unsigned_impl! {usize}

abs_signed_impl! {i8}
abs_signed_impl! {i16}
abs_signed_impl! {i32}
abs_signed_impl! {i64}
abs_signed_impl! {i128}
abs_signed_impl! {isize}

abs_signed_impl! {f32}
abs_signed_impl! {f64}

/// Trait for raising numbers to a power
pub trait Pow<P> {
    /// The output type
    type Output;
    /// Raise this number to a power
    fn pow(self, power: P) -> Self::Output;
}

macro_rules! pow_float_impl {
    ($type:ty) => {
        impl Pow<Self> for $type {
            type Output = Self;
            fn pow(self, power: Self) -> Self::Output {
                self.powf(power)
            }
        }
    };
}

pow_float_impl! {f32}
pow_float_impl! {f64}

/// Trait for defining small-number constants
pub trait ZeroOneTwo: Copy {
    /// This type's value for zero, i.e. `0`
    const ZERO: Self;
    /// This type's value for one, i.e. `1`
    const ONE: Self;
    /// This type's value for two, i.e. `2`
    const TWO: Self;
}

macro_rules! zot_int_impl {
    ($type:ty) => {
        impl ZeroOneTwo for $type {
            const ZERO: Self = 0;
            const ONE: Self = 1;
            const TWO: Self = 2;
        }
    };
}

zot_int_impl! {u8}
zot_int_impl! {u16}
zot_int_impl! {u32}
zot_int_impl! {u64}
zot_int_impl! {u128}
zot_int_impl! {usize}

zot_int_impl! {i8}
zot_int_impl! {i16}
zot_int_impl! {i32}
zot_int_impl! {i64}
zot_int_impl! {i128}
zot_int_impl! {isize}

macro_rules! zot_float_impl {
    ($type:ty) => {
        impl ZeroOneTwo for $type {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            const TWO: Self = 2.0;
        }
    };
}

zot_float_impl! {f32}
zot_float_impl! {f64}

/// Trait for math with scalar numbers
pub trait Scalar:
    Add<Self, Output = Self>
    + Copy
    + PartialEq
    + PartialOrd
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self>
    + Abs
    + ZeroOneTwo
{
    /// Get the max of this `Scalar` and another
    ///
    /// This function is named to not conflict with the
    /// `Scalar`'s default `max` function
    fn maxx(self, other: Self) -> Self {
        if self > other {
            self
        } else {
            other
        }
    }
    /// Get the min of this `Scalar` and another
    ///
    /// This function is named to not conflict with the
    /// `Scalar`'s default `min` function
    fn minn(self, other: Self) -> Self {
        if self < other {
            self
        } else {
            other
        }
    }
}

impl<T> Scalar for T where
    T: Copy
        + PartialEq
        + PartialOrd
        + Add<T, Output = T>
        + Sub<T, Output = T>
        + Mul<T, Output = T>
        + Div<T, Output = T>
        + Abs
        + ZeroOneTwo
{
}

/// Trait for floating-point scalar numbers
pub trait FloatingScalar:
    Scalar + Pow<Self, Output = Self> + Sin<Output = Self> + Cos<Output = Self>
{
}

impl<T> FloatingScalar for T where
    T: Scalar + Pow<Self, Output = Self> + Sin<Output = T> + Cos<Output = T>
{
}

/// Trait for manipulating 2D vectors
pub trait Vector2: Copy {
    /// The scalar type
    type Scalar: Scalar;
    /// Get the x component
    fn x(self) -> Self::Scalar;
    /// Get the y component
    fn y(self) -> Self::Scalar;
    /// Create a new vector from an x and y component
    fn new(x: Self::Scalar, y: Self::Scalar) -> Self;
    /// Create a new square vector
    fn square(s: Self::Scalar) -> Self {
        Self::new(s, s)
    }
    /// Map this vector to a vector of another type
    fn map<V>(self) -> V
    where
        V: Vector2,
        V::Scalar: From<Self::Scalar>,
    {
        V::new(V::Scalar::from(self.x()), V::Scalar::from(self.y()))
    }
    /// Map this vector to a vector of another type using a function
    fn map_with<V, F>(self, mut f: F) -> V
    where
        V: Vector2,
        F: FnMut(Self::Scalar) -> V::Scalar,
    {
        V::new(f(self.x()), f(self.y()))
    }
    /// Negate the vector
    fn neg(self) -> Self
    where
        Self::Scalar: Neg<Output = Self::Scalar>,
    {
        Self::new(-self.x(), -self.y())
    }
    /// Add the vector to another
    fn add<V>(self, other: V) -> Self
    where
        V: Vector2<Scalar = Self::Scalar>,
    {
        Self::new(self.x() + other.x(), self.y() + other.y())
    }
    /// Subtract another vector from this one
    fn sub<V>(self, other: V) -> Self
    where
        V: Vector2<Scalar = Self::Scalar>,
    {
        Self::new(self.x() - other.x(), self.y() - other.y())
    }
    /// Multiply this vector by a scalar
    fn mul(self, by: Self::Scalar) -> Self {
        Self::new(self.x() * by, self.y() * by)
    }
    /// Multiply this vector component-wise by another
    fn mul2<V>(self, other: V) -> Self
    where
        V: Vector2<Scalar = Self::Scalar>,
    {
        Self::new(self.x() * other.x(), self.y() * other.y())
    }
    /// Divide this vector by a scalar
    fn div(self, by: Self::Scalar) -> Self {
        Self::new(self.x() / by, self.y() / by)
    }
    /// Divide this vector component-wise by another
    fn div2<V>(self, other: V) -> Self
    where
        V: Vector2<Scalar = Self::Scalar>,
    {
        Self::new(self.x() / other.x(), self.y() / other.y())
    }
    /// Get the value of the dimmension with the higher magnitude
    fn max_dim(self) -> Self::Scalar {
        if self.x().abs() > self.y().abs() {
            self.x()
        } else {
            self.y()
        }
    }
}

impl<P> Vector2 for P
where
    P: Pair + Copy,
    P::Item: Scalar,
{
    type Scalar = P::Item;
    fn x(self) -> P::Item {
        self.first()
    }
    fn y(self) -> P::Item {
        self.second()
    }
    fn new(x: P::Item, y: P::Item) -> Self {
        Self::from_items(x, y)
    }
}

/// Trait for manipulating floating-point 2D vectors
pub trait FloatingVector2: Vector2
where
    Self::Scalar: FloatingScalar,
{
    /// Get the distance between this vector and another
    fn dist<V>(self, to: V) -> Self::Scalar
    where
        V: Vector2<Scalar = Self::Scalar>,
    {
        ((self.x() - to.x()).pow(Self::Scalar::TWO) + (self.y() - to.y()).pow(Self::Scalar::TWO))
            .pow(Self::Scalar::ONE / Self::Scalar::TWO)
    }
    /// Get the vector's magnitude
    fn mag(self) -> Self::Scalar {
        (self.x().pow(Self::Scalar::TWO) + self.y().pow(Self::Scalar::TWO))
            .pow(Self::Scalar::ONE / Self::Scalar::TWO)
    }
    /// Rotate the vector some number of radians about a pivot
    fn rotate_about<V>(self, pivot: V, radians: Self::Scalar) -> Self
    where
        V: Vector2<Scalar = Self::Scalar> + Clone,
    {
        let sin = radians.sin();
        let cos = radians.cos();
        let origin_point = self.sub(pivot);
        let rotated_point = Self::new(
            origin_point.x() * cos - origin_point.y() * sin,
            origin_point.x() * sin + origin_point.y() * cos,
        );
        rotated_point.add(pivot)
    }
}

impl<T> FloatingVector2 for T
where
    T: Vector2,
    T::Scalar: FloatingScalar,
{
}

/**
Trait for manipulating axis-aligned rectangles

Because the primary expected use for this crate is in 2D graphics and alignment implementations,
a coordinate system where the positive Y direction is "down" is assumed.

# Note
Methods of the form `abs_` account for the case where the size if negative.
If the size is not negative, they are identical to their non-`abs_` counterparts.
```
use vector2math::*;

let pos_size = [1, 2, 3, 4];
assert_eq!(pos_size.right(), pos_size.abs_right());

let neg_size = [1, 2, -3, -4];
assert_ne!(neg_size.right(), neg_size.abs_right());
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
    /// Check that the rectangle contains the given point
    fn contains(self, point: Self::Vector) -> bool {
        let in_x_bounds = self.left() <= point.x() && point.x() <= self.right();
        let in_y_bounds = self.top() <= point.y() && point.y() <= self.bottom();
        in_x_bounds && in_y_bounds
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
        self.first()
    }
    fn size(self) -> Self::Vector {
        self.second()
    }
}
