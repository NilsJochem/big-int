use std::fmt::{Binary, Debug, Display, LowerExp, LowerHex, Octal, UpperExp, UpperHex};

use itertools::Either;

pub trait Primitive:
    Copy
    + Clone
    + Debug
    + Display
    + Default
    + Eq
    + Ord
    + Binary
    + Octal
    + LowerHex
    + UpperHex
    + LowerExp
    + UpperExp
{
    const BYTES: usize;
    const BITS: usize;
    const ZERO: Self;
    const ONE: Self;
    const MIN: Self;
    const MAX: Self;

    type Pos: UNum<Neg = Self::Neg>;
    type Neg: INum<Pos = Self::Pos>;

    #[allow(clippy::wrong_self_convention)]
    fn from_be(self) -> Self;
    #[allow(clippy::wrong_self_convention)]
    fn from_le(self) -> Self;
    fn to_be(self) -> Self;
    fn to_le(self) -> Self;
    fn to_le_bytes(self) -> impl ExactSizeIterator<Item = u8> + DoubleEndedIterator;
    fn push_le_bytes(self, buf: &mut impl Extend<u8>)
    where
        Self: Sized,
    {
        buf.extend(self.to_le_bytes());
    }

    fn select_sign(self) -> Either<Self::Pos, Self::Neg>;
}
pub trait UNum: Primitive {
    fn try_neg(self) -> Option<Self::Neg>;
}
pub trait INum: Primitive {
    fn is_negative(self) -> bool;
    fn abs(self) -> Self::Pos;

    #[allow(dead_code)]
    fn try_pos(self) -> Option<Self::Pos> {
        if self.is_negative() {
            None
        } else {
            Some(self.abs())
        }
    }
}

macro_rules! implPrim {
    (funcs, $bytes: literal) => {
        const BYTES: usize = $bytes;
        const BITS: usize = $bytes * 8;
        const ZERO: Self = 0;
        const ONE: Self = 1;
        const MIN: Self = Self::MIN;
        const MAX: Self = Self::MAX;

        fn to_le_bytes(self) -> impl ExactSizeIterator<Item = u8> + DoubleEndedIterator {
            self.to_le_bytes().into_iter()
        }

        implPrim!(relay, from_be, self, Self);
        implPrim!(relay, from_le, self, Self);
        implPrim!(relay, to_be, self, Self);
        implPrim!(relay, to_le, self, Self);
    };
    (relay, $name: ident, self, $return: ident) => {
        fn $name(self) -> $return {
            Self::$name(self)
        }
    };
    ($pos_type: ident, $neg_type: ident, $bytes: literal) => {
        impl Primitive for $pos_type {
            type Pos = $pos_type;
            type Neg = $neg_type;
            fn select_sign(self) -> Either<Self::Pos, Self::Neg> {
                Either::Left(self)
            }
            implPrim!(funcs, $bytes);
        }
        impl Primitive for $neg_type {
            type Pos = $pos_type;
            type Neg = $neg_type;

            fn select_sign(self) -> Either<Self::Pos, Self::Neg> {
                Either::Right(self)
            }
            implPrim!(funcs, $bytes);
        }
        impl UNum for $pos_type {
            fn try_neg(self) -> Option<Self::Neg> {
                $neg_type::try_from(self).ok()
            }
        }
        impl INum for $neg_type {
            fn is_negative(self) -> bool {
                self.is_negative()
            }
            fn abs(self) -> $pos_type {
                self.unsigned_abs()
            }
        }
    };
}

implPrim!(u8, i8, 1);
implPrim!(u16, i16, 16);
implPrim!(u32, i32, 32);
implPrim!(u64, i64, 64);
implPrim!(u128, i128, 128);

#[cfg(target_pointer_width = "64")]
implPrim!(usize, isize, 64);
#[cfg(target_pointer_width = "32")]
implPrim!(usize, isize, 32);
#[cfg(target_pointer_width = "16")]
implPrim!(usize, isize, 16);
