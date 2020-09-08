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
assert_eq!(2, a.dot(b));
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

A few types can be used to define circles:
* `([T; 2], T)`
* `((T, T), T)`
* Any pair of types where the first implements `FloatingVector2` and the second is the vector's scalar type.
```
use vector2math::*;
use std::f64;

let circle = ([2.0, 3.0], 4.0);
assert!((circle.circumference() - 25.132_741_228_718_345).abs() < f64::EPSILON);
assert!((circle.area() - 50.265_482_457_436_69).abs() < f64::EPSILON);
assert!(circle.contains([0.0, 1.0]));
assert!(!circle.contains([5.0, 6.0]));
```

Vector, rectangle, and circle types can be easily mapped to different types:
```
use vector2math::*;

let arrayf32: [f32; 2] = [1.0, 2.0];
let arrayf64: [f64; 2] = arrayf32.map();
let pairf64: (f64, f64) = arrayf64.map();
let arrayi16: [i16; 2] = pairf64.map_with(|f| f as i16);
assert_eq!(arrayf32, arrayi16.map_f32());

let weird_rect = [(0.0, 1.0), (2.0, 5.0)];
let normal_rectf32: [f32; 4] = weird_rect.map();
let normal_rectf64: [f32; 4] = normal_rectf32.map();
let normal_rectu8: [u8; 4] = normal_rectf32.map_with(|f| f as u8);
assert_eq!([0, 1, 2, 5], normal_rectu8);

let pair_circlef32 = ((0.0, 1.0), 2.0);
let array_circlef32 = ([0.0, 1.0], 2.0);
assert_eq!(((0.0, 1.0), 2.0), array_circlef32.map::<((f64, f64), f64)>());
```

Implementing these traits for your own types is simple.
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

macro_rules! mods {
    ($($m:ident),*) => {
        $(mod $m; pub use $m::*;)*
    };
}

mods!(circle, group, rectangle, scalar);

use std::ops::Neg;

/// Module containing standard f32 types
///
/// Import the contents of this module if your project uses `f32`s in geometry
pub mod f32 {
    /// A standard dimension type
    pub type Dim = f32;
    /// A standard 2D vector type
    pub type Vec2 = [Dim; 2];
    /// A standard rectangle type
    pub type Rect = [Dim; 4];
    /// A standard circle type
    pub type Circ = ([Dim; 2], Dim);
}

/// Module containing standard f64 types
///
/// Import the contents of this module if your project uses `f64`s in geometry
pub mod f64 {
    /// A standard dimension type
    pub type Dim = f64;
    /// A standard 2D vector type
    pub type Vec2 = [Dim; 2];
    /// A standard rectangle type
    pub type Rect = [Dim; 4];
    /// A standard circle type
    pub type Circ = ([Dim; 2], Dim);
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
    fn map<V>(self) -> V
    where
        V: Vector2,
        V::Scalar: From<Self::Scalar>,
    {
        V::new(V::Scalar::from(self.x()), V::Scalar::from(self.y()))
    }
    /// Map this vector to a `[f32;2]`
    ///
    /// This is an alias for Vector2::map::<[f32;2]>() that is more concise
    fn map_f32(self) -> [f32; 2]
    where
        f32: From<Self::Scalar>,
    {
        self.map()
    }
    /// Map this vector to a `[f64;2]`
    ///
    /// This is an alias for Vector2::map::<[f64;2]>() that is more concise
    fn map_f64(self) -> [f64; 2]
    where
        f64: From<Self::Scalar>,
    {
        self.map()
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
    fn dot<V>(self, other: V) -> Self::Scalar
    where
        V: Vector2<Scalar = Self::Scalar>,
    {
        self.x() * other.x() + self.y() * other.y()
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
    /// Get the unit vector
    fn unit(self) -> Self {
        let mag = self.mag();
        if mag < Self::Scalar::EPSILON {
            Self::new(Self::Scalar::ZERO, Self::Scalar::ZERO)
        } else {
            self.div(mag)
        }
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
    /// Linear interpolate the vector with another
    fn lerp<V>(self, other: V, t: Self::Scalar) -> Self
    where
        V: Vector2<Scalar = Self::Scalar>,
    {
        Self::new(self.x().lerp(other.x(), t), self.y().lerp(other.y(), t))
    }
    /// Get the arctangent of the vector, which corresponds to
    /// the angle it represents bounded between -π to π
    fn atan(self) -> Self::Scalar {
        self.y().atan2(self.x())
    }
}

impl<T> FloatingVector2 for T
where
    T: Vector2,
    T::Scalar: FloatingScalar,
{
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn margins() {
        let rect = [0, 0, 8, 8];
        assert!(rect.contains([1, 1]));
        assert!(!rect.inner_margin(2).contains([1, 1]));
    }
}
