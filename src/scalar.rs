use std::ops::{Add, Div, Mul, Neg, Sub};

/// Trait for math with scalar numbers
pub trait Scalar:
    Add<Self, Output = Self>
    + Copy
    + PartialEq
    + PartialOrd
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self>
{
    /// The value of 0
    const ZERO: Self;
    /// The value of 1
    const ONE: Self;
    /// The value of 2
    const TWO: Self;
    /// Get the absolute value of the number
    fn abs(self) -> Self;
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

macro_rules! scalar_unsigned_impl {
    ($type:ty) => {
        impl Scalar for $type {
            const ZERO: Self = 0;
            const ONE: Self = 1;
            const TWO: Self = 2;
            fn abs(self) -> Self {
                self
            }
        }
    };
}

macro_rules! scalar_signed_impl {
    ($type:ty) => {
        impl Scalar for $type {
            const ZERO: Self = 0;
            const ONE: Self = 1;
            const TWO: Self = 2;
            fn abs(self) -> Self {
                self.abs()
            }
        }
    };
}

macro_rules! scalar_float_impl {
    ($type:ty) => {
        impl Scalar for $type {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            const TWO: Self = 2.0;
            fn abs(self) -> Self {
                self.abs()
            }
        }
    };
}

scalar_unsigned_impl!(u8);
scalar_unsigned_impl!(u16);
scalar_unsigned_impl!(u32);
scalar_unsigned_impl!(u64);
scalar_unsigned_impl!(u128);
scalar_unsigned_impl!(usize);

scalar_signed_impl!(i8);
scalar_signed_impl!(i16);
scalar_signed_impl!(i32);
scalar_signed_impl!(i64);
scalar_signed_impl!(i128);
scalar_signed_impl!(isize);

scalar_float_impl!(f32);
scalar_float_impl!(f64);

/// Trait for floating-point scalar numbers
pub trait FloatingScalar: Scalar + Mul<Output = Self> + Neg<Output = Self> {
    /// The value of Tau, or 2π
    const TAU: Self;
    /// The value of π
    const PI: Self;
    /// The epsilon value
    const EPSILON: Self;
    /// Get the sqare root of the scalar
    fn sqrt(self) -> Self;
    /// Square the scalar
    fn square(self) -> Self {
        self * self
    }
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

macro_rules! floating_scalar_impl {
    ($type:ty, $pi:expr, $epsilon:expr) => {
        impl FloatingScalar for $type {
            const PI: Self = $pi;
            const TAU: Self = $pi * 2.0;
            const EPSILON: Self = $epsilon;
            fn sqrt(self) -> Self {
                Self::sqrt(self)
            }
            fn cos(self) -> Self {
                Self::cos(self)
            }
            fn sin(self) -> Self {
                Self::sin(self)
            }
            fn atan2(self, other: Self) -> Self {
                self.atan2(other)
            }
        }
    };
}

floating_scalar_impl!(f32, std::f32::consts::PI, std::f32::EPSILON);
floating_scalar_impl!(f64, std::f64::consts::PI, std::f64::EPSILON);
