#![warn(missing_docs)]
#![cfg_attr(feature = "simd", feature(doc_cfg))]

/*!
This crate provides traits for doing 2D vector geometry operations using standard types

# Scalars

Simple vector math is implemented for vectors with the following scalar types:
* `u8`-`u128`
* `usize`
* `i8`-`i128`
* `isize`
* `f32`
* `f64`
* Any type that implements [`Scalar`]

`f32` and `f64` implement [`FloatingScalar`], which
gives some additional operations only applicable to floating-point numbers.

Each scalar type has an associated module that has type definitions for standard
geometric types using that scalar.

For example, instead of writing
```
# use vector2math::*;
let square = <[f32; 4]>::square([0.0; 2], 1.0);
```
You can instead write
```
# use vector2math::*;
let square = f32::Rect::square([0.0; 2], 1.0);
```

# Vectors

Vectors can be of the following forms:
* `[T; 2]`
* `(T, T)`
* Any type that implements [`Vector2`]

Many 2D Vector operations are supported.
```
use vector2math::*;

let a = [2, 6];
let b = [4, -1];
assert_eq!(2, a.x());
assert_eq!(-1, b.y());
assert_eq!([-2, -6], a.neg());
assert_eq!([6, 5], a.add(b));
assert_eq!([-2, 7], a.sub(b));
assert_eq!([12, -3], b.mul(3));
assert_eq!([8, -6], b.mul2(a));
assert_eq!([1, 3], a.div(2));
assert_eq!([0, -6], a.div2(b));
assert_eq!(2, a.dot(b));
```

Vectors that implement [`FloatingVector2`] have additional operations:
```
use vector2math::*;

assert_eq!(5.0, [3.0, 4.0].mag());
assert_eq!(10.0, [-1.0, -2.0].dist([5.0, 6.0]));
let rotation_calculation = [1.0, 0.0].rotate_about(f64::TAU / 8.0, [0.0; 2]);
let rotation_solution = [2f64.powf(0.5) / 2.0; 2];
assert!(rotation_calculation.sub(rotation_solution).mag() < std::f64::EPSILON);
```

# Rectangles

Many types can be used to define axis-aligned rectangles:
* `[[T; 2]; 2]`
* `[(T, T); 2]`
* `((T, T), (T, T))`
* `([T; 2], [T; 2])`
* `[T; 4]`
* `(T, T, T, T)`
* Any type that implements [`Pair`] where the associated
[`Pair::Item`] type implements [`Vector2`].
```
use vector2math::*;

let rect = [1i32, 2, 4, 6];
assert_eq!([1, 2], rect.top_left());
assert_eq!([4, 6], rect.size());
assert_eq!([3, 5], rect.center());
assert_eq!(20, rect.perimeter());
assert_eq!(24, rect.area());
assert!(rect.contains([3, 5]));
let corners = rect.corners();
assert_eq!(corners[0], [1, 2]);
assert_eq!(corners[1], [5, 2]);
assert_eq!(corners[2], [5, 8]);
assert_eq!(corners[3], [1, 8]);
```

# Circles

A few types can be used to define circles:
* `([T; 2], T)`
* `((T, T), T)`
* Any pair of types where the first implements [`FloatingVector2`]
and the second is the vector's [`Vector2::Scalar`] type.
```
use vector2math::*;
use std::f64;

let circle = ([2.0, 3.0], 4.0);
assert!((circle.circumference() - 25.132_741_228_718_345).abs() < f64::EPSILON);
assert!((circle.area() - 50.265_482_457_436_69).abs() < f64::EPSILON);
assert!(circle.contains([0.0, 1.0]));
assert!(!circle.contains([5.0, 6.0]));
```

# Mapping

Vector, rectangle, and circle types can be easily mapped to different types:
```
use vector2math::*;

let arrayf32: [f32; 2] = [1.0, 2.0];
let arrayf64: [f64; 2] = arrayf32.map_into();
let pairf64: (f64, f64) = arrayf64.map_into();
let arrayi16: [i16; 2] = pairf64.map_with(|f| f as i16);
assert_eq!(arrayf32, arrayi16.map_into::<f32::Vec2>());

let weird_rect = [(0.0, 1.0), (2.0, 5.0)];
let normal_rectf32: [f32; 4] = weird_rect.map_into();
let normal_rectf64: [f64; 4] = normal_rectf32.map_into();
let normal_rectu8: [u8; 4] = normal_rectf32.map_with(|f| f as u8);
assert_eq!([0, 1, 2, 5], normal_rectu8);

let pair_circlef32 = ((0.0, 1.0), 2.0);
let array_circlef32 = ([0.0, 1.0], 2.0);
assert_eq!(((0.0, 1.0), 2.0), array_circlef32.map_into::<((f64, f64), f64)>());
```

# Transforms

The [`Transform`] trait is used to define 2D vector transforms.
This crate implements [`Transform`] for all types that implement
[`Pair`](trait.Pair.html) where the [`Pair`](trait.Pair.html)'s
[`Item`](trait.Pair.html#associatedtype.Item) implments [`Trio`]
where the [`Trio`]'s [`Trio::Item`]
implements [`FloatingScalar`]. This type range includes
everything from `[[f32; 3]; 2]` to `(f64, f64, f64, f64, f64, f64)`.
[`Transform`]s can be chained and applied to vectors.
```
use vector2math::*;

let dis = [1.0; 2];
let rot = f32::TAU / 4.0;
let sc = [2.0; 2];

let transform = f32::Trans::identity().translate(dis).rotate(rot).scale(sc);

let v = [3.0, 5.0];
let v1 = v.transform(transform);
let v2 = v.add(dis).rotate(rot).mul2(sc);

assert_eq!(v1, v2);
```

# Implementing traits

Implementing these traits for your own types is simple.
Just make sure that your type is [`Copy`](https://doc.rust-lang.org/nightly/core/marker/trait.Copy.html).
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
    fn x(&self) -> f64 {
        self.x
    }
    fn y(&self) -> f64 {
        self.y
    }
}

#[derive(Clone, Copy)]
struct MyRectangle {
    top_left: MyVector,
    size: MyVector,
}

impl Rectangle for MyRectangle {
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

let rect: MyRectangle = [1, 2, 3, 4].map_into();
assert_eq!(12.0, rect.area());
assert_eq!(6.0, rect.bottom());
```
*/

#[cfg(feature = "simd")]
#[cfg_attr(feature = "simd", doc(cfg(feature = "simd")))]
pub mod simd;

pub mod circle;
pub use circle::Circle;
mod group;
pub use group::*;
pub mod rectangle;
pub use rectangle::Rectangle;
mod scalar;
pub use scalar::*;
mod transform;
pub use transform::*;

macro_rules! int_mod {
    ($T:ident) => {
        /// Standard geometric types for a scalar type
        pub mod $T {
            /// A dimension type
            pub type Dim = $T;
            /// A standard 2D vector type
            pub type Vec2 = [Dim; 2];
            /// A standard rectangle type
            pub type Rect = [Dim; 4];
        }
    };
}

int_mod!(u8);
int_mod!(u16);
int_mod!(u32);
int_mod!(u64);
int_mod!(u128);
int_mod!(usize);
int_mod!(i8);
int_mod!(i16);
int_mod!(i32);
int_mod!(i64);
int_mod!(i128);
int_mod!(isize);

macro_rules! float_mod {
    ($T:ident) => {
        /// Standard geometric types for a scalar type
        pub mod $T {
            /// A dimension type
            pub type Dim = $T;
            /// A standard 2D vector type
            pub type Vec2 = [Dim; 2];
            /// A standard rectangle type
            pub type Rect = [Dim; 4];
            /// A standard circle type
            pub type Circ = (Vec2, Dim);
            /// A standard transform type
            pub type Trans = [[Dim; 3]; 2];
        }
    };
}

float_mod!(f32);
float_mod!(f64);

use std::ops::Neg;

pub use Circle as _;
pub use Rectangle as _;
pub use Transform as _;

/// Trait for manipulating 2D vectors
pub trait Vector2: Copy {
    /// The scalar type
    type Scalar: Scalar;
    /// Get the x component
    fn x(&self) -> Self::Scalar;
    /// Get the y component
    fn y(&self) -> Self::Scalar;
    /// Create a new vector from an x and y component
    fn new(x: Self::Scalar, y: Self::Scalar) -> Self;
    /// Set the x component
    fn set_x(&mut self, x: Self::Scalar) {
        *self = Vector2::new(x, self.y())
    }
    /// Set the y component
    fn set_y(&mut self, y: Self::Scalar) {
        *self = Vector2::new(self.x(), y)
    }
    /// Get this vector with a different x component
    fn with_x(self, x: Self::Scalar) -> Self {
        Self::new(x, self.y())
    }
    /// Get this vector with a different y component
    fn with_y(self, y: Self::Scalar) -> Self {
        Self::new(self.x(), y)
    }
    /// Create a new square vector
    fn square(s: Self::Scalar) -> Self {
        Self::new(s, s)
    }
    /// Map this vector to a vector of another type
    #[inline(always)]
    fn map_into<V>(self) -> V
    where
        V: Vector2,
        V::Scalar: From<Self::Scalar>,
    {
        V::new(V::Scalar::from(self.x()), V::Scalar::from(self.y()))
    }
    /// Map this vector to a `[Self::Scalar; 2]`
    ///
    /// This is an alias for `Vector2::map_into::<[Self::Scalar; 2]>()` that is more concise
    fn map_vec2(self) -> [Self::Scalar; 2] {
        self.map_into()
    }
    /// Map the individual components of this vector
    fn map_dims<F>(self, mut f: F) -> Self
    where
        F: FnMut(Self::Scalar) -> Self::Scalar,
    {
        Self::new(f(self.x()), f(self.y()))
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
    #[inline(always)]
    fn neg(self) -> Self
    where
        Self::Scalar: Neg<Output = Self::Scalar>,
    {
        Self::square(Self::Scalar::ZERO).sub(self)
    }
    /// Add this vector to another
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        Self::new(self.x() + other.x(), self.y() + other.y())
    }
    /// Subtract another vector from this one
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        Self::new(self.x() - other.x(), self.y() - other.y())
    }
    /// Multiply this vector by a scalar
    #[inline(always)]
    fn mul(self, by: Self::Scalar) -> Self {
        self.mul2(Self::square(by))
    }
    /// Multiply this vector component-wise by another
    #[inline(always)]
    fn mul2(self, other: Self) -> Self {
        Self::new(self.x() * other.x(), self.y() * other.y())
    }
    /// Divide this vector by a scalar
    #[inline(always)]
    fn div(self, by: Self::Scalar) -> Self {
        self.div2(Self::square(by))
    }
    /// Divide this vector component-wise by another
    #[inline(always)]
    fn div2(self, other: Self) -> Self {
        Self::new(self.x() / other.x(), self.y() / other.y())
    }
    /// Add another vector into this one
    #[inline(always)]
    fn add_assign(&mut self, other: Self) {
        *self = self.add(other);
    }
    /// Subtract another vector into this one
    #[inline(always)]
    fn sub_assign(&mut self, other: Self) {
        *self = self.sub(other);
    }
    /// Multiply a scalar into this vector
    #[inline(always)]
    fn mul_assign(&mut self, by: Self::Scalar) {
        *self = self.mul(by);
    }
    /// Multiply another vector component-wise into this one
    #[inline(always)]
    fn mul2_assign(&mut self, other: Self) {
        *self = self.mul2(other);
    }
    /// Divide a scalar into this vector
    #[inline(always)]
    fn div_assign(&mut self, by: Self::Scalar) {
        *self = self.div(by);
    }
    /// Divide another vector component-wise into this one
    #[inline(always)]
    fn div2_assign(&mut self, other: Self) {
        *self = self.div2(other);
    }
    /// Get the value of the dimension with the higher magnitude
    fn max_dim(self) -> Self::Scalar {
        if self.x().abs() > self.y().abs() {
            self.x()
        } else {
            self.y()
        }
    }
    /// Get the value of the dimension with the lower magnitude
    fn min_dim(self) -> Self::Scalar {
        if self.x().abs() < self.y().abs() {
            self.x()
        } else {
            self.y()
        }
    }
    /// Get the dot product of this vector and another
    fn dot(self, other: Self) -> Self::Scalar {
        let sum = self.mul2(other);
        sum.x() + sum.y()
    }
}

impl<P> Vector2 for P
where
    P: Pair + Copy,
    P::Item: Scalar,
{
    type Scalar = P::Item;
    #[inline(always)]
    fn x(&self) -> P::Item {
        self.first()
    }
    #[inline(always)]
    fn y(&self) -> P::Item {
        self.second()
    }
    #[inline(always)]
    fn new(x: P::Item, y: P::Item) -> Self {
        Self::from_items(x, y)
    }
}

/// Trait for manipulating floating-point 2D vectors
pub trait FloatingVector2: Vector2
where
    Self::Scalar: FloatingScalar,
{
    /// Create a new unit vector from the given angle in radians
    fn from_angle(radians: Self::Scalar) -> Self {
        Self::new(radians.cos(), radians.sin())
    }
    /// Get the distance between this vector and another
    #[inline(always)]
    fn dist(self, to: Self) -> Self::Scalar {
        self.sub(to).mag()
    }
    /// Get the squared distance between this vector and another
    #[inline(always)]
    fn squared_dist(self, to: Self) -> Self::Scalar {
        self.sub(to).squared_mag()
    }
    /// Get the vector's magnitude
    #[inline(always)]
    fn mag(self) -> Self::Scalar {
        self.squared_mag().sqrt()
    }
    /// Get the vector's squared magnitude
    #[inline(always)]
    fn squared_mag(self) -> Self::Scalar {
        self.x().square() + self.y().square()
    }
    /// Get the unit vector
    #[inline(always)]
    fn unit(self) -> Self {
        let mag = self.mag();
        if mag < Self::Scalar::EPSILON {
            Self::new(Self::Scalar::ZERO, Self::Scalar::ZERO)
        } else {
            self.div(mag)
        }
    }
    /// Rotate the vector some number of radians about the origin
    fn rotate(self, radians: Self::Scalar) -> Self {
        self.rotate_about(radians, Self::square(Self::Scalar::ZERO))
    }
    /// Rotate the vector some number of radians about a pivot
    fn rotate_about(self, radians: Self::Scalar, pivot: Self) -> Self {
        let sin = radians.sin();
        let cos = radians.cos();
        let origin_point = self.sub(pivot);
        let rotated_point = Self::new(
            origin_point.x() * cos - origin_point.y() * sin,
            origin_point.x() * sin + origin_point.y() * cos,
        );
        rotated_point.add(pivot)
    }
    /// Linear interpolate the vector with another
    #[inline(always)]
    fn lerp(self, other: Self, t: Self::Scalar) -> Self {
        Self::square(Self::Scalar::ONE - t)
            .mul2(self)
            .add(Self::square(t).mul2(other))
    }
    /// Get the arctangent of the vector, which corresponds to
    /// the angle it represents bounded between -π to π
    fn atan(self) -> Self::Scalar {
        self.y().atan2(self.x())
    }
    /// Apply a transform to the vector
    fn transform<T>(self, transform: T) -> Self
    where
        T: Transform<Scalar = Self::Scalar>,
    {
        transform.apply(self)
    }
    /// Project this vector onto another
    fn project(self, other: Self) -> Self {
        other.unit().mul(self.dot(other) * self.mag())
    }
}

impl<T> FloatingVector2 for T
where
    T: Vector2,
    T::Scalar: FloatingScalar,
{
}

#[cfg(test)]
#[test]
fn margins() {
    let rect = [0, 0, 8, 8];
    assert!(rect.contains([1, 1]));
    assert!(!rect.inner_margin(2).contains([1, 1]));
}
#[cfg(test)]
#[test]
fn transforms() {
    let v = [1.0, 3.0];
    let rot = 1.0;
    let pivot = [5.0; 2];
    let transform = f32::Trans::identity().rotate_about(rot, pivot);
    let v1 = v.rotate_about(rot, pivot);
    let v2 = v.transform(transform);
    dbg!(v1.dist(v2) / f32::EPSILON);
    assert!(v1.dist(v2).is_near_zero(10.0));
}

#[cfg(test)]
#[test]
fn rect_with_bound() {
    let rect = [0, 0, 5, 5];
    let rt3 = rect.with_top(3);
    let rb8 = rect.with_bottom(8);
    let rl1 = rect.with_left(1);
    let rr1 = rect.with_right(1);
    assert_eq!(rt3, [0, 3, 5, 2]);
    assert_eq!(rb8, [0, 0, 5, 8]);
    assert_eq!(rl1, [1, 0, 4, 5]);
    assert_eq!(rr1, [0, 0, 1, 5]);
}
