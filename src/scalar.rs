use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::Vector2;

/// Trait for trigonometric operations
pub trait Trig: Copy + Div<Output = Self> {
    /// Get the cosine
    fn cos(self) -> Self;
    /// Get the sine
    fn sin(self) -> Self;
    /// Get the tangent
    fn tan(self) -> Self {
        self.sin() / self.cos()
    }
    /// Get the four-quadrant arctangent
    fn atan2(self, other: Self) -> Self;
}

impl Trig for f32 {
    fn cos(self) -> Self {
        f32::cos(self)
    }
    fn sin(self) -> Self {
        f32::sin(self)
    }
    fn atan2(self, other: Self) -> Self {
        self.atan2(other)
    }
}

impl Trig for f64 {
    fn cos(self) -> Self {
        f64::cos(self)
    }
    fn sin(self) -> Self {
        f64::sin(self)
    }
    fn atan2(self, other: Self) -> Self {
        self.atan2(other)
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
    /// Create a square `Vector` from this `Scalar`
    fn square<V>(self) -> V
    where
        V: Vector2<Scalar = Self>,
    {
        V::square(self)
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
pub trait FloatingScalar: Scalar + Neg<Output = Self> + Pow<Self, Output = Self> + Trig {
    /// The value of Tau, or 2π
    const TAU: Self;
    /// The epsilon value
    const EPSILON: Self;
    /// Get the value of pi
    fn pi() -> Self {
        Self::TAU / Self::TWO
    }
    /// Linear interpolate the scalar with another
    fn lerp(self, other: Self, t: Self) -> Self {
        (Self::ONE - t) * self + t * other
    }
    /// Get the unit vector corresponding to an angle in radians defined by the scalar
    fn angle_as_vector(self) -> [Self; 2] {
        [self.cos(), self.sin()]
    }
    /// Check if the value is within its epsilon range
    fn is_zero(self) -> bool {
        self.is_near_zero(Self::ONE)
    }
    /// Check if the value is within a multiple epsilon range
    fn is_near_zero(self, n: Self) -> bool {
        self.abs() < Self::EPSILON * n
    }
}

impl FloatingScalar for f32 {
    const TAU: Self = std::f32::consts::PI * 2.0;
    const EPSILON: Self = std::f32::EPSILON;
}

impl FloatingScalar for f64 {
    const TAU: Self = std::f64::consts::PI * 2.0;
    const EPSILON: Self = std::f64::EPSILON;
}
