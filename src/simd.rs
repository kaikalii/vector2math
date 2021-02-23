/*!
Simd standard types
*/

use crate::Vector2;

pub use packed_simd::{f32x2, f64x2, i16x2, i32x2, i64x2, i8x2, u16x2, u32x2, u64x2, u8x2};

macro_rules! int_mod {
    ($T:ident, $V:ident) => {
        /// Standard geometric types for a scalar type
        pub mod $T {
            /// A standard 2D vector type
            pub type Vec2 = super::$V;
            /// A standard rectangle type
            pub type Rect = [super::$V; 2];
        }
    };
}

int_mod!(u8, u8x2);
int_mod!(u16, u16x2);
int_mod!(u32, u32x2);
int_mod!(u64, u64x2);
int_mod!(i8, i8x2);
int_mod!(i16, i16x2);
int_mod!(i32, i32x2);
int_mod!(i64, i64x2);

macro_rules! float_mod {
    ($T:ident, $V:ident) => {
        /// Standard geometric types for a scalar type
        pub mod $T {
            /// A standard 2D vector type
            pub type Vec2 = super::$V;
            /// A standard rectangle type
            pub type Rect = [super::$V; 2];
            /// A standard circle type
            pub type Circ = (super::$V, $T);
            /// A standard transform type
            pub type Trans = [$T; 6];
        }
    };
}

float_mod!(f32, f32x2);
float_mod!(f64, f64x2);

macro_rules! impl_simd_vector2 {
    ($Vector:ty, $Scalar:ty) => {
        impl Vector2 for $Vector {
            type Scalar = $Scalar;
            fn new(x: Self::Scalar, y: Self::Scalar) -> Self {
                <$Vector>::new(x, y)
            }
            fn x(self) -> Self::Scalar {
                unsafe { self.extract_unchecked(0) }
            }
            fn y(self) -> Self::Scalar {
                unsafe { self.extract_unchecked(1) }
            }
            fn with_x(self, x: $Scalar) -> Self {
                unsafe { self.replace_unchecked(0, x) }
            }
            fn with_y(self, y: $Scalar) -> Self {
                unsafe { self.replace_unchecked(1, y) }
            }
            fn square(s: Self::Scalar) -> Self {
                Self::splat(s)
            }
            fn add(self, other: Self) -> Self {
                self + other
            }
            fn sub(self, other: Self) -> Self {
                self - other
            }
            fn mul2(self, other: Self) -> Self {
                self * other
            }
            fn div2(self, other: Self) -> Self {
                self * other
            }
            fn add_assign(&mut self, other: Self) {
                *self += other;
            }
            fn sub_assign(&mut self, other: Self) {
                *self -= other;
            }
            fn mul2_assign(&mut self, other: Self) {
                *self *= other;
            }
            fn div2_assign(&mut self, other: Self) {
                *self /= other;
            }
        }
    };
}

impl_simd_vector2!(u8x2, u8);
impl_simd_vector2!(u16x2, u16);
impl_simd_vector2!(u32x2, u32);
impl_simd_vector2!(u64x2, u64);

impl_simd_vector2!(i8x2, i8);
impl_simd_vector2!(i16x2, i16);
impl_simd_vector2!(i32x2, i32);
impl_simd_vector2!(i64x2, i64);

impl_simd_vector2!(f32x2, f32);
impl_simd_vector2!(f64x2, f64);

#[cfg(test)]
#[test]
fn simd() {
    use crate::{FloatingVector2, Transform};
    let a = f32x2::new(1.0, 2.0);
    let b = f32x2::new(3.0, 5.0);
    let c = a.add(b);
    assert_eq!(f32x2::new(4.0, 7.0), c);
    let c = a.transform(f32::Trans::identity().translate(b));
    assert_eq!(f32x2::new(4.0, 7.0), c);
}
