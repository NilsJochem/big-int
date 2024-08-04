use std::{
    iter,
    ops::{Deref, DerefMut, Index, IndexMut},
};

use itertools::{Either, Itertools};
pub mod util_traits {
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
}

#[cfg(target_endian = "little")]
const IS_LE: bool = true;
#[cfg(target_endian = "big")]
const IS_LE: bool = false;

#[cfg(target_pointer_width = "64")]
type HalfSizeNative = u32;
const HALF_SIZE_BYTES: usize = HalfSizeNative::BITS as usize / 8;

#[derive(Clone, Copy)]
pub union HalfSize {
    native: HalfSizeNative,
    ne_bytes: [u8; HALF_SIZE_BYTES],
}
impl std::fmt::Debug for HalfSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HalfSize").field("native", &**self).finish()
    }
}
impl std::fmt::Display for HalfSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", &**self)
    }
}
impl PartialEq for HalfSize {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}
impl Eq for HalfSize {}

impl Default for HalfSize {
    fn default() -> Self {
        Self { native: 0 }
    }
}

impl From<HalfSizeNative> for HalfSize {
    fn from(value: HalfSizeNative) -> Self {
        Self { native: value }
    }
}
impl From<[u8; HALF_SIZE_BYTES]> for HalfSize {
    fn from(value: [u8; HALF_SIZE_BYTES]) -> Self {
        Self { ne_bytes: value }
    }
}

impl HalfSize {
    fn format_index(index: usize) -> usize {
        assert!(index < HALF_SIZE_BYTES);
        if IS_LE {
            index
        } else {
            HALF_SIZE_BYTES - index
        }
    }
    pub const fn ne_bytes(self) -> [u8; HALF_SIZE_BYTES] {
        // SAFETY: union will always be properly initialized
        unsafe { self.ne_bytes }
    }
    pub fn le_bytes(self) -> [u8; HALF_SIZE_BYTES] {
        (*self).to_le_bytes()
    }
    pub fn be_bytes(self) -> [u8; HALF_SIZE_BYTES] {
        (*self).to_be_bytes()
    }
}
impl Deref for HalfSize {
    type Target = HalfSizeNative;

    fn deref(&self) -> &Self::Target {
        // SAFETY: union will always be properly initialized
        unsafe { &self.native }
    }
}
impl DerefMut for HalfSize {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: union will always be properly initialized
        unsafe { &mut self.native }
    }
}

/// access le ordered bytes
impl Index<usize> for HalfSize {
    type Output = u8;

    fn index(&self, index: usize) -> &Self::Output {
        // SAFETY: union will always be properly initialized
        unsafe { self.ne_bytes.index(Self::format_index(index)) }
    }
}
/// access le ordered bytes
impl IndexMut<usize> for HalfSize {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        // SAFETY: union will always be properly initialized
        unsafe { self.ne_bytes.index_mut(Self::format_index(index)) }
    }
}

#[derive(Clone, Default, PartialEq, Eq)]
pub struct BigInt {
    /// used to encode the length in bytes, the number of patial bytes in the last element and the sign of the number
    /// `abs()` can be any of `data.len()` * `HALF_SIZE_BYTES` - 3 and `data.len()` * `HALF_SIZE_BYTES`
    bytes: isize,
    /// holds the `HalfSize` values in LE order
    data: Vec<HalfSize>,
}

impl std::fmt::Debug for BigInt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Number {{ {} {:x?} , {} }}",
            if self.is_positive() { '+' } else { '-' },
            self.data,
            self.bytes
        )
    }
}
impl<P: util_traits::UNum> FromIterator<P> for BigInt {
    /// the iter should contain the parts in little endian order
    fn from_iter<T: IntoIterator<Item = P>>(iter: T) -> Self {
        let mut num = Self::default();
        num.extend(
            iter.into_iter()
                .flat_map(util_traits::Primitive::to_le_bytes),
        );
        num
    }
}

impl<N: util_traits::Primitive> From<N> for BigInt {
    fn from(value: N) -> Self {
        match value.select_sign() {
            Either::Left(pos) => iter::once(pos).collect(),
            Either::Right(neg) => {
                let mut num = iter::once(util_traits::INum::abs(neg)).collect::<Self>();
                if util_traits::INum::is_negative(neg) {
                    num.negate();
                }
                num
            }
        }
    }
}

impl Extend<u8> for BigInt {
    fn extend<T: IntoIterator<Item = u8>>(&mut self, iter: T) {
        let partial = self.partial();
        let mut iter = iter.into_iter();

        if partial != 0 {
            let last = self
                .data
                .last_mut()
                .expect("couldn't find last Halfsize, even if length was non zero");
            for (i, byte) in (partial..HALF_SIZE_BYTES).zip(iter.by_ref()) {
                last[i] = byte;
            }
            self.extent_length(HALF_SIZE_BYTES as isize - partial as isize);
        }

        self.extend_whole_unchecked(iter.chunks(HALF_SIZE_BYTES).into_iter().map(|chunk| {
            let mut buf = [0; HALF_SIZE_BYTES];

            for (place, byte) in buf.iter_mut().zip(chunk) {
                *place = byte;
            }
            buf
        }));
    }
}
impl<HSN: Into<HalfSize>> Extend<HSN> for BigInt {
    fn extend<T: IntoIterator<Item = HSN>>(&mut self, iter: T) {
        if self.partial() != 0 {
            self.extend(iter.into_iter().flat_map(|it| it.into().le_bytes()));
        } else {
            self.extend_whole_unchecked(iter);
        }
    }
}

impl BigInt {
    pub const fn is_negative(&self) -> bool {
        self.bytes.is_negative()
    }
    pub const fn is_positive(&self) -> bool {
        self.bytes.is_positive()
    }

    const fn partial(&self) -> usize {
        self.bytes.unsigned_abs() % HALF_SIZE_BYTES
    }

    /// extends the number with the given Halfsizes. Assumes, that no partial bytes exist
    fn extend_whole_unchecked<HSN, T>(&mut self, iter: T)
    where
        HSN: Into<HalfSize>,
        T: IntoIterator<Item = HSN>,
    {
        for next in iter {
            self.data.push(next.into());
            self.extent_length(HALF_SIZE_BYTES as isize);
        }

        self.strip_trailing_zeros();
    }
    fn extent_length(&mut self, offset: isize) {
        assert!(
            offset.is_positive() || self.bytes.abs() >= offset.abs(),
            "tried to remove to many elements"
        );
        if self.bytes.is_negative() {
            self.bytes -= offset;
        } else {
            self.bytes += offset;
        }
    }
    fn strip_trailing_zeros(&mut self) {
        while self.data.last().is_some_and(|&it| *it == 0) {
            self.data.pop();
            self.extent_length(-4);
        }
        if let Some(last) = self.data.last() {
            self.extent_length(
                -(last
                    .be_bytes()
                    .into_iter()
                    .take_while(|&it| it == 0)
                    .count() as isize),
            );
        }
    }

    fn negate(&mut self) {
        self.bytes *= -1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    mod create {
        use super::*;
        #[test]
        fn extend_only_partial() {
            let mut num = BigInt {
                bytes: 1,
                data: vec![HalfSize::from(0x11)],
            };
            num.extend(0x3322u16.to_le_bytes());
            assert_eq!(
                num,
                BigInt {
                    bytes: 3,
                    data: vec![HalfSize::from(0x00332211)]
                }
            );
        }
        #[test]
        fn extend_only_full() {
            let mut num = BigInt::default();
            num.extend(0x99887766554433221100u128.to_le_bytes());
            assert_eq!(
                num,
                BigInt {
                    bytes: 10,
                    data: vec![
                        HalfSize::from(0x33221100),
                        HalfSize::from(0x77665544),
                        HalfSize::from(0x00009988)
                    ]
                }
            );
        }
        #[test]
        fn extend_partial_and_full() {
            let mut num = BigInt {
                bytes: 5,
                data: vec![HalfSize::from(0x33221100), HalfSize::from(0x44)],
            };
            num.extend(0x9988776655u64.to_le_bytes());
            assert_eq!(
                num,
                BigInt {
                    bytes: 10,
                    data: vec![
                        HalfSize::from(0x33221100),
                        HalfSize::from(0x77665544),
                        HalfSize::from(0x00009988)
                    ]
                }
            );
        }
        #[test]
        fn extend_partial_and_full_negative() {
            let mut num = BigInt {
                bytes: -5,
                data: vec![HalfSize::from(0x33221100), HalfSize::from(0x44)],
            };
            num.extend(0x9988776655u64.to_le_bytes());
            assert_eq!(
                num,
                BigInt {
                    bytes: -10,
                    data: vec![
                        HalfSize::from(0x33221100),
                        HalfSize::from(0x77665544),
                        HalfSize::from(0x00009988)
                    ]
                }
            );
        }

        #[test]
        fn from_u32s() {
            assert_eq!(
                [0x33221100u32, 0x77665544, 0x9988]
                    .into_iter()
                    .collect::<BigInt>(),
                BigInt {
                    bytes: 10,
                    data: vec![
                        HalfSize::from(0x33221100),
                        HalfSize::from(0x77665544),
                        HalfSize::from(0x00009988)
                    ]
                }
            )
        }
        #[test]
        fn from_i128() {
            assert_eq!(
                BigInt::from(-0x99887766554433221100i128),
                BigInt {
                    bytes: -10,
                    data: vec![
                        HalfSize::from(0x33221100),
                        HalfSize::from(0x77665544),
                        HalfSize::from(0x00009988)
                    ]
                }
            )
        }
    }
}
