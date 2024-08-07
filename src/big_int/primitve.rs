use itertools::Either;

pub trait Primitive: Copy + Eq + Ord {
    const BYTES: usize;
    const ZERO: Self;
    const ONE: Self;

    type Pos: UNum<Neg = Self::Neg>;
    type Neg: INum<Pos = Self::Pos>;

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
    ($pos_type: tt, $neg_type: tt, $bytes: literal) => {
        impl Primitive for $pos_type {
            const BYTES: usize = $bytes;

            const ZERO: Self = 0;
            const ONE: Self = 1;

            type Pos = $pos_type;
            type Neg = $neg_type;

            fn to_le_bytes(self) -> impl ExactSizeIterator<Item = u8> + DoubleEndedIterator {
                self.to_le_bytes().into_iter()
            }
            fn select_sign(self) -> Either<Self::Pos, Self::Neg> {
                Either::Left(self)
            }
        }
        impl Primitive for $neg_type {
            const BYTES: usize = $bytes;

            const ZERO: Self = 0;
            const ONE: Self = 1;

            type Pos = $pos_type;
            type Neg = $neg_type;

            fn to_le_bytes(self) -> impl ExactSizeIterator<Item = u8> + DoubleEndedIterator {
                self.to_le_bytes().into_iter()
            }
            fn select_sign(self) -> Either<Self::Pos, Self::Neg> {
                Either::Right(self)
            }
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
