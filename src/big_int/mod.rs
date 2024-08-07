use core::fmt;
use std::{
    fmt::Write,
    iter,
    ops::{Deref, DerefMut, Index, IndexMut},
};

use itertools::{Either, Itertools};
use math_shortcuts::MathShortcut;

use crate::boo::{Boo, Moo};
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

const IS_LE: bool = cfg!(target_endian = "little");

#[cfg(target_pointer_width = "64")]
type HalfSizeNative = u32;
const HALF_SIZE_BYTES: usize = HalfSizeNative::BITS as usize / 8;
const FULL_SIZE_BYTES: usize = usize::BITS as usize / 8;

#[derive(Clone, Copy)]
pub union HalfSize {
    ne_bytes: [u8; HALF_SIZE_BYTES],
    native: HalfSizeNative,
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
impl std::fmt::LowerHex for HalfSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt::LowerHex::fmt(
            &if IS_LE {
                **self
            } else {
                HalfSizeNative::from_be(**self)
            },
            f,
        )
    }
}
impl std::fmt::UpperHex for HalfSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt::UpperHex::fmt(
            &if IS_LE {
                **self
            } else {
                HalfSizeNative::from_be(**self)
            },
            f,
        )
    }
}

impl PartialEq for HalfSize {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}
impl Eq for HalfSize {}
impl PartialOrd for HalfSize {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HalfSize {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (**self).cmp(&**other)
    }
}

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

#[derive(Clone, Copy)]
union FullSize {
    native: usize,
    halfs: [HalfSize; 2],
    #[allow(dead_code)]
    ne_bytes: [u8; FULL_SIZE_BYTES],
}
impl std::fmt::Debug for FullSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:#x}")
    }
}
impl std::fmt::Display for FullSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", **self)
    }
}
impl std::fmt::LowerHex for FullSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt::LowerHex::fmt(
            &if IS_LE {
                **self
            } else {
                usize::from_be(**self)
            },
            f,
        )
    }
}
impl std::fmt::UpperHex for FullSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt::UpperHex::fmt(
            &if IS_LE {
                **self
            } else {
                usize::from_be(**self)
            },
            f,
        )
    }
}

impl PartialEq for FullSize {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}
impl Eq for FullSize {}

impl From<usize> for FullSize {
    fn from(native: usize) -> Self {
        // SAFTY: access to native is always possible
        Self { native }
    }
}
impl From<HalfSize> for FullSize {
    fn from(lower: HalfSize) -> Self {
        Self::new(lower, HalfSize::default())
    }
}
/// SAFTY: access to part is always possible
#[allow(clippy::undocumented_unsafe_blocks)]
impl FullSize {
    const fn new(lower: HalfSize, higher: HalfSize) -> Self {
        if IS_LE {
            Self {
                halfs: [lower, higher],
            }
        } else {
            Self {
                halfs: [higher, lower],
            }
        }
    }
    const fn lower(self) -> HalfSize {
        if IS_LE {
            unsafe { self.halfs[0] }
        } else {
            unsafe { self.halfs[1] }
        }
    }
    const fn higher(self) -> HalfSize {
        if IS_LE {
            unsafe { self.halfs[1] }
        } else {
            unsafe { self.halfs[0] }
        }
    }
}

impl Deref for FullSize {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        // SAFETY: union will always be properly initialized
        unsafe { &self.native }
    }
}
impl DerefMut for FullSize {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: union will always be properly initialized
        unsafe { &mut self.native }
    }
}

macro_rules! implHalfMath {
    (a $($trait:tt)::*, $func:tt) => {
		implHalfMath!(a $($trait)::*, $func, Self);
		implHalfMath!(a $($trait)::*, $func, &Self);
		implHalfMath!(a $($trait)::*, $func, u32);
		implHalfMath!(a $($trait)::*, $func, &u32);
	};
    (a $($trait:tt)::*, $func:tt, Self) => {
		impl $($trait)::* for HalfSize {
			fn $func(&mut self, rhs: Self) {
				$($trait)::*::$func(&mut **self, *rhs)
			}
		}
	};
    (a $($trait:tt)::*, $func:tt, &Self) => {
		impl $($trait)::*<&Self> for HalfSize {
			fn $func(&mut self, rhs: &Self) {
				$($trait)::*::$func(&mut **self, **rhs)
			}
		}
	};
    (a $($trait:tt)::*, $func:tt, $rhs:tt) => {
		impl $($trait)::*<$rhs> for HalfSize {
			fn $func(&mut self, rhs: $rhs) {
				$($trait)::*::$func(&mut **self, rhs)
			}
		}
	};
    (a $($trait:tt)::*, $func:tt, &$rhs:tt) => {
		impl $($trait)::*<&$rhs> for HalfSize {
			fn $func(&mut self, rhs: &$rhs) {
				$($trait)::*::$func(&mut **self, *rhs)
			}
		}
	};

    ($($trait:tt)::*, $func:tt) => {
		implHalfMath!($($trait)::*, $func, Self);
		implHalfMath!($($trait)::*, $func, &Self);
		implHalfMath!($($trait)::*, $func, u32);
		implHalfMath!($($trait)::*, $func, &u32);
	};
    ($($trait:tt)::*, $func:tt, Self) => {
		impl $($trait)::* for HalfSize {
			type Output = Self;
			fn $func(self, rhs: Self) -> Self::Output  {
				Self::from($($trait)::*::$func(*self, *rhs))
			}
		}
		impl $($trait)::* for &HalfSize {
			type Output = HalfSize;
			fn $func(self, rhs: Self) -> Self::Output  {
				$($trait)::*::$func(*self, *rhs)
			}
		}
	};
    ($($trait:tt)::*, $func:tt, &Self) => {
		impl $($trait)::*<&Self> for HalfSize {
			type Output = Self;
			fn $func(self, rhs: &Self) -> Self::Output  {
				Self::from($($trait)::*::$func(*self, **rhs))
			}
		}
		impl $($trait)::*<&Self> for &HalfSize {
			type Output = HalfSize;
			fn $func(self, rhs: &Self) -> Self::Output  {
				$($trait)::*::$func(*self, **rhs)
			}
		}
	};
    ($($trait:tt)::*, $func:tt, $rhs:tt) => {
		impl $($trait)::*<$rhs> for HalfSize {
			type Output = Self;
			fn $func(self, rhs: $rhs) -> Self::Output  {
				Self::from($($trait)::*::$func( *self, rhs))
			}
		}
		impl $($trait)::*<$rhs> for &HalfSize {
			type Output = HalfSize;
			fn $func(self, rhs: $rhs) -> Self::Output  {
				$($trait)::*::$func(*self, rhs)
			}
		}
	};
    ($($trait:tt)::*, $func:tt, &$rhs:tt) => {
		impl $($trait)::*<&$rhs> for HalfSize {
			type Output = Self;
			fn $func(self, rhs: &$rhs) -> Self::Output  {
				Self::from($($trait)::*::$func( *self, *rhs))
			}
		}
		impl $($trait)::*<&$rhs> for &HalfSize {
			type Output = HalfSize;
			fn $func(self, rhs: &$rhs) -> Self::Output  {
				$($trait)::*::$func(*self, *rhs)
			}
		}
	};
}
implHalfMath!(a std::ops::BitOrAssign, bitor_assign);
implHalfMath!(std::ops::BitOr, bitor);
implHalfMath!(a std::ops::BitXorAssign, bitxor_assign);
implHalfMath!(std::ops::BitXor, bitxor);
implHalfMath!(a std::ops::BitAndAssign, bitand_assign);
implHalfMath!(std::ops::BitAnd, bitand);

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
            "Number {{ {} 0x[",
            if self.is_positive() { '+' } else { '-' },
        )?;
        for (pos, elem) in self.data.iter().rev().with_position() {
            write!(f, "{elem:08x}")?; // TODO hardcoded size
            if matches!(
                pos,
                itertools::Position::First | itertools::Position::Middle
            ) {
                f.write_str(", ")?;
            }
        }
        write!(f, "] , {} }}", self.bytes)
    }
}
// impl std::fmt::Display for BigInt {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         todo!("need divided / modulo")
//     }
// }
impl std::fmt::LowerHex for BigInt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut buf = String::new();
        for part in self.data.iter().rev() {
            write!(buf, "{part:x}")?;
        }
        f.pad_integral(
            !self.is_negative(),
            if f.alternate() { "0x" } else { "" },
            &buf,
        )
    }
}
impl std::fmt::UpperHex for BigInt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut buf = String::new();
        for part in self.data.iter().rev() {
            write!(buf, "{part:X}")?;
        }
        f.pad_integral(
            !self.is_negative(),
            if f.alternate() { "0X" } else { "" },
            &buf,
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
    pub const fn signum(&self) -> isize {
        self.bytes.signum()
    }

    pub const fn is_zero(&self) -> bool {
        self.bytes == 0
    }
    pub fn is_abs_one(&self) -> bool {
        self.data.len() == 1 && *self.data[0] == 1
    }

    pub fn ilog2(&self) -> Option<usize> {
        let mut canditate = None;
        for (i, part) in self.data.iter().enumerate().filter(|&(_, it)| **it != 0) {
            if canditate.is_some() || !(*part).is_power_of_two() {
                return None;
            }
            canditate = Some((*part).ilog2() as usize + i * HalfSizeNative::BITS as usize);
        }
        canditate
    }

    // number of partial bytes in the last `HalfSize`
    const fn partial(&self) -> usize {
        self.bytes.unsigned_abs() % HALF_SIZE_BYTES
    }
    // number of missing bytes in the last `HalfSize`, to have no partial remaining
    #[allow(dead_code)]
    const fn co_partial(&self) -> usize {
        (-self.bytes.abs()).rem_euclid(HALF_SIZE_BYTES as isize) as usize
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

        self.recalc_len();
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

    fn recalc_len(&mut self) {
        while self.data.last().is_some_and(|&it| *it == 0) {
            self.pop();
        }
        self.bytes =
            (self.data.len() * HALF_SIZE_BYTES) as isize * (!self.is_negative() as isize * 2 - 1);

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

    fn pop(&mut self) {
        self.data.pop();
        self.recalc_len();
    }
    fn push(&mut self, value: impl Into<HalfSize>) {
        let value = value.into();
        if *value == 0 {
            return;
        }
        self.data.push(value);
        self.recalc_len();
    }

    pub fn negate(&mut self) {
        self.bytes = std::ops::Neg::neg(self.bytes);
    }
    pub fn abs(&mut self) {
        self.bytes = self.bytes.abs();
    }
    #[must_use]
    pub fn abs_clone(&self) -> Self {
        let mut out = self.clone();
        out.abs();
        out
    }
    pub fn abs_ord(&self, rhs: &Self) -> std::cmp::Ordering {
        self.data.iter().rev().cmp(rhs.data.iter().rev())
    }
}

impl PartialOrd for BigInt {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for BigInt {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering as O;
        match self.bytes.cmp(&other.bytes) {
            O::Less => O::Less,
            O::Greater => O::Greater,
            O::Equal => self.abs_ord(other),
        }
    }
}

macro_rules! implBigMath {
    (a $($assign_trait:tt)::*, $assign_func:tt, $($trait:tt)::*, $func:tt) => {
        implBigMath!(a $($assign_trait)::*, $assign_func, $($trait)::*, $func, Self);
    };
    (a $($assign_trait:tt)::*, $assign_func:tt, $($trait:tt)::*, $func:tt, $rhs:tt) => {
		impl $($assign_trait)::*<$rhs> for BigInt {
			fn $assign_func(&mut self, rhs: $rhs) {
				$($assign_trait)::*::$assign_func(self, &rhs)
			}
		}
		impl $($trait)::*<&$rhs> for BigInt {
			type Output = Self;

			fn $func(mut self, rhs: &$rhs) -> Self::Output {
				$($assign_trait)::*::$assign_func(&mut self, rhs);
				self
			}
		}
		impl $($trait)::*<$rhs> for BigInt {
			type Output = Self;

			fn $func(mut self, rhs: $rhs) -> Self::Output {
				$($assign_trait)::*::$assign_func(&mut self, &rhs);
				self
			}
		}
	};

    ($($assign_trait:tt)::*, $assign_func:tt, $($trait:tt)::*, $func:tt) => {
        impl $($trait)::* for BigInt {
            implBigMath!(body $func, Self);
        }
        impl $($trait)::*<&BigInt> for BigInt {
            implBigMath!(body $func, &BigInt);
        }
        impl $($trait)::* for &BigInt {
            implBigMath!(body $func, Self);
        }
        impl $($trait)::*<BigInt> for &BigInt {
            implBigMath!(body $func, BigInt);
        }
        impl $($assign_trait)::* for BigInt {
            fn $assign_func(&mut self, rhs: Self) {
                BigInt::$func(Boo::from(self), Boo::from(rhs)).expect_mut_ref("did give &mut, shouldn't get result");
            }
        }
        impl $($assign_trait)::*<&Self> for BigInt {
            fn $assign_func(&mut self, rhs: &Self) {
                BigInt::$func(Boo::from(self), Boo::from(rhs)).expect_mut_ref("did give &mut, shouldn't get result");
            }
        }
    };
    (body $func:tt, $rhs:tt) => {
        type Output = BigInt;
        fn $func(self, rhs: $rhs) -> Self::Output {
            BigInt::$func(Boo::from(self), Boo::from(rhs)).expect_owned("didn't give &mut, should get result")
        }
    };
    (body $func:tt, &$rhs:tt) => {
        type Output = BigInt;
        fn $func(self, rhs: &$rhs) -> Self::Output {
            BigInt::$func(Boo::from(self), Boo::from(rhs)).expect_owned("didn't give &mut, should get result")
        }
    };
}

impl std::ops::Neg for BigInt {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.negate();
        self
    }
}

implBigMath!(a std::ops::ShlAssign, shl_assign, std::ops::Shl, shl, usize);
impl std::ops::ShlAssign<&usize> for BigInt {
    fn shl_assign(&mut self, rhs: &usize) {
        let partial = rhs % HalfSizeNative::BITS as usize;
        let full = rhs / HalfSizeNative::BITS as usize;

        let mut carry = HalfSize::default();
        if partial > 0 {
            for part in &mut self.data {
                let old_carry = carry;
                let result = FullSize::from(std::ops::Shl::shl(
                    *FullSize::new(*part, HalfSize::default()),
                    partial,
                ));
                *part = result.lower() | old_carry;
                carry = result.higher();
            }
        }
        let carry = Some(carry).filter(|&it| *it != 0);
        if carry.is_some() || full > 0 {
            self.data = std::iter::repeat(HalfSize::default())
                .take(full)
                .chain(self.data.iter().copied())
                .chain(carry)
                .collect();
        }
        self.recalc_len();
    }
}

implBigMath!(a std::ops::ShrAssign, shr_assign, std::ops::Shr, shr, usize);
impl std::ops::ShrAssign<&usize> for BigInt {
    fn shr_assign(&mut self, rhs: &usize) {
        let partial = rhs % HalfSizeNative::BITS as usize;
        let full = rhs / HalfSizeNative::BITS as usize;

        let mut carry = HalfSize::default();
        if partial > 0 {
            for part in self.data.iter_mut().rev() {
                let old_carry = carry;
                let result = FullSize::from(std::ops::Shr::shr(
                    *FullSize::new(HalfSize::default(), *part),
                    partial,
                ));
                *part = result.higher() | old_carry;
                carry = result.lower();
            }
        }
        if full > 0 {
            self.data = self.data.iter().copied().dropping(full).collect();
        }
        self.recalc_len();
    }
}

#[allow(clippy::multiple_inherent_impl)]
impl BigInt {
    const fn is_different_sign(&self, rhs: &Self) -> bool {
        !self.is_negative() ^ !rhs.is_negative()
    }

    fn assert_pair_valid(pair: (&Boo<'_, Self>, &Boo<'_, Self>)) {
        assert!(
            !matches!(pair.0, Boo::BorrowedMut(_)) || !matches!(pair.1, Boo::BorrowedMut(_)),
            "can't have to Borrow_mut's"
        );
    }

    fn refer_direct<'b, 'b1: 'b, 'b2: 'b, B1, B2>(
        lhs: B1,
        rhs: B2,
        func: impl FnOnce(&mut Self, &Self),
    ) -> Moo<'b, Self>
    where
        B1: Into<Boo<'b1, Self>>,
        B2: Into<Boo<'b2, Self>>,
    {
        let lhs: Boo<'_, Self> = lhs.into();
        let rhs: Boo<'_, Self> = rhs.into();
        Self::assert_pair_valid((&lhs, &rhs));

        match (lhs, rhs) {
            (Boo::BorrowedMut(borrow_mut), borrow) | (borrow, Boo::BorrowedMut(borrow_mut)) => {
                func(borrow_mut, &borrow);
                Moo::BorrowedMut(borrow_mut)
            }
            (Boo::Borrowed(lhs), rhs) => {
                let mut owned = rhs.cloned();
                func(&mut owned, lhs);
                Moo::Owned(owned)
            }
            (lhs, rhs) => {
                let mut owned = lhs.cloned();
                func(&mut owned, &rhs);
                Moo::Owned(owned)
            }
        }
    }

    fn bitor<'b, 'b1: 'b, 'b2: 'b, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b, Self>
    where
        B1: Into<Boo<'b1, Self>>,
        B2: Into<Boo<'b2, Self>>,
    {
        Self::refer_direct(lhs, rhs, math_algos::bit::bit_or_assign_internal)
    }
    fn bitxor<'b, 'b1: 'b, 'b2: 'b, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b, Self>
    where
        B1: Into<Boo<'b1, Self>>,
        B2: Into<Boo<'b2, Self>>,
    {
        Self::refer_direct(lhs, rhs, math_algos::bit::bit_xor_assign_internal)
    }

    fn bitand<'b, 'b1: 'b, 'b2: 'b, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b, Self>
    where
        B1: Into<Boo<'b1, Self>>,
        B2: Into<Boo<'b2, Self>>,
    {
        Self::refer_direct(lhs, rhs, math_algos::bit::bit_and_assign_internal)
    }

    fn add<'b, 'b1: 'b, 'b2: 'b, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b, Self>
    where
        B1: Into<Boo<'b1, Self>>,
        B2: Into<Boo<'b2, Self>>,
    {
        let lhs: Boo<'_, Self> = lhs.into();
        let rhs: Boo<'_, Self> = rhs.into();
        Self::assert_pair_valid((&lhs, &rhs));

        math_shortcuts::try_all!(lhs, rhs, math_shortcuts::add::Zero,);

        if lhs.is_different_sign(&rhs) {
            return match (lhs, rhs) {
                (lhs, Boo::BorrowedMut(rhs)) => {
                    rhs.negate();
                    Self::sub(lhs, rhs)
                }
                (Boo::Borrowed(lhs), rhs) => {
                    let mut owned = rhs.cloned();
                    owned.negate();
                    Self::sub(lhs, owned)
                }
                (mut lhs, rhs) => {
                    lhs.try_get_mut().unwrap().negate();
                    let mut ret = Self::sub(lhs, rhs);
                    ret.negate();
                    ret
                }
            };
        }
        Self::refer_direct(lhs, rhs, math_algos::add::assign_internal)
    }

    fn sub<'b, 'b1: 'b, 'b2: 'b, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b, Self>
    where
        B1: Into<Boo<'b1, Self>>,
        B2: Into<Boo<'b2, Self>>,
    {
        let lhs: Boo<'_, Self> = lhs.into();
        let rhs: Boo<'_, Self> = rhs.into();
        Self::assert_pair_valid((&lhs, &rhs));

        math_shortcuts::try_all!(lhs, rhs, math_shortcuts::sub::Zero,);

        if lhs.is_different_sign(&rhs) {
            return match (lhs, rhs) {
                (lhs, Boo::BorrowedMut(rhs)) => {
                    rhs.negate();
                    *rhs += lhs.as_ref();
                    Moo::BorrowedMut(rhs)
                }
                (Boo::Borrowed(lhs), rhs) => {
                    let mut owned = rhs.cloned();
                    owned.negate();
                    Self::add(lhs, owned)
                }
                (Boo::Owned(lhs), Boo::Owned(mut rhs)) => {
                    rhs.negate();
                    Self::add(lhs, rhs)
                }
                (mut lhs, rhs) => {
                    lhs.try_get_mut().unwrap().negate();
                    let mut ret = Self::add(lhs, rhs);
                    ret.negate();
                    ret
                }
            };
        }

        let (lhs, rhs, need_negate) = if lhs.abs_ord(&rhs).is_lt() {
            (rhs, lhs, true)
        } else {
            (lhs, rhs, false)
        };
        let finnish = |it: &mut Self| {
            if need_negate {
                it.negate();
            }
        };

        match (lhs, rhs) {
            (Boo::BorrowedMut(lhs), rhs) => {
                math_algos::sub::assign_smaller_same_sign(lhs, &rhs);
                finnish(lhs);
                Moo::BorrowedMut(lhs)
            }
            (lhs, Boo::BorrowedMut(borrowed)) => {
                let old_rhs = std::mem::replace(borrowed, lhs.cloned()); // lhs -> rhs, rhs -> old_rhs
                math_algos::sub::assign_smaller_same_sign(borrowed, &old_rhs);
                finnish(borrowed);
                Moo::BorrowedMut(borrowed)
            }
            (lhs, rhs) => {
                // can't really use storage in rhs (when existing) because algo can only sub smaller
                let mut lhs = lhs.cloned();
                math_algos::sub::assign_smaller_same_sign(&mut lhs, &rhs);
                finnish(&mut lhs);
                Moo::Owned(lhs)
            }
        }
    }

    fn mul<'b, 'b1: 'b, 'b2: 'b, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b, Self>
    where
        B1: Into<Boo<'b1, Self>>,
        B2: Into<Boo<'b2, Self>>,
    {
        let lhs: Boo<'_, Self> = lhs.into();
        let rhs: Boo<'_, Self> = rhs.into();
        Self::assert_pair_valid((&lhs, &rhs));

        math_shortcuts::try_all!(
            lhs,
            rhs,
            math_shortcuts::mul::ByZero,
            math_shortcuts::mul::ByOne,
            math_shortcuts::mul::ByPowerOfTwo,
        );

        match (lhs, rhs) {
            (Boo::BorrowedMut(borrow_mut), borrow) | (borrow, Boo::BorrowedMut(borrow_mut)) => {
                *borrow_mut = math_algos::mul::naive(borrow_mut, &borrow);
                Moo::BorrowedMut(borrow_mut)
            }
            (lhs, rhs) => Moo::Owned(math_algos::mul::naive(&lhs, &rhs)),
        }
    }
}

mod math_algos;
mod math_shortcuts;

// no `std::ops::Not`, cause implied zeros to the left would need to be flipped
implBigMath!(std::ops::BitOrAssign, bitor_assign, std::ops::BitOr, bitor);
implBigMath!(
    std::ops::BitXorAssign,
    bitxor_assign,
    std::ops::BitXor,
    bitxor
);
implBigMath!(
    std::ops::BitAndAssign,
    bitand_assign,
    std::ops::BitAnd,
    bitand
);
implBigMath!(std::ops::SubAssign, sub_assign, std::ops::Sub, sub);
implBigMath!(std::ops::AddAssign, add_assign, std::ops::Add, add);
implBigMath!(std::ops::MulAssign, mul_assign, std::ops::Mul, mul);

impl std::ops::Mul<&HalfSize> for &BigInt {
    type Output = BigInt;

    fn mul(self, rhs: &HalfSize) -> Self::Output {
        match **rhs {
            0 => return BigInt::default(),
            1 => return self.abs_clone(),
            x if x.is_power_of_two() => {
                let mut out = self.abs_clone();
                std::ops::ShlAssign::shl_assign(&mut out, x.ilog2() as usize);
                return out;
            }
            _ => {}
        }
        math_algos::mul::part_at_offset(self, *rhs, 0)
    }
}
impl std::ops::Mul<HalfSize> for &BigInt {
    type Output = BigInt;

    fn mul(self, rhs: HalfSize) -> Self::Output {
        std::ops::Mul::mul(self, &rhs)
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
    mod output {
        use super::*;

        #[test]
        fn lower_hex() {
            assert_eq!(
                format!("{:x}", BigInt::from(0x99887766554433221100u128)),
                "99887766554433221100"
            );
            assert_eq!(
                format!("{:#x}", BigInt::from(0x99887766554433221100u128)),
                "0x99887766554433221100"
            );

            assert_eq!(
                format!("{:x}", BigInt::from(-0x99887766554433221100i128)),
                "-99887766554433221100"
            );
            assert_eq!(
                format!("{:#x}", BigInt::from(-0x99887766554433221100i128)),
                "-0x99887766554433221100"
            );

            assert_eq!(
                format!("{:0>32x}", BigInt::from(0x99887766554433221100u128)),
                "00000000000099887766554433221100"
            );

            assert_eq!(
                format!("{:#032x}", BigInt::from(0x99887766554433221100u128)),
                "0x000000000099887766554433221100"
            );

            assert_eq!(
                format!(
                    "{:#032x}",
                    BigInt::from(0xeeddccbbaa99887766554433221100u128)
                ),
                "0xeeddccbbaa99887766554433221100"
            );
            assert_eq!(
                format!(
                    "{:#032X}",
                    BigInt::from(0xeeddccbbaa99887766554433221100u128)
                ),
                "0XEEDDCCBBAA99887766554433221100"
            );
        }
    }
    mod order {
        use std::cmp::Ordering;

        use super::*;
        #[test]
        fn same() {
            assert_eq!(
                BigInt::from(0x99887766554433221100u128)
                    .cmp(&BigInt::from(0x99887766554433221100u128)),
                Ordering::Equal
            );
            assert_eq!(
                BigInt::from(-0x99887766554433221100i128)
                    .cmp(&BigInt::from(-0x99887766554433221100i128)),
                Ordering::Equal
            );
        }
        #[test]
        fn negated() {
            assert_eq!(
                BigInt::from(0x99887766554433221100u128)
                    .cmp(&BigInt::from(-0x99887766554433221100i128)),
                Ordering::Greater
            );
            assert_eq!(
                BigInt::from(-0x99887766554433221100i128)
                    .cmp(&BigInt::from(0x99887766554433221100i128)),
                Ordering::Less
            );
        }
        #[test]
        fn middle_diff() {
            assert_eq!(
                BigInt::from(0x99888866554433221100u128)
                    .cmp(&BigInt::from(0x99887766554433221100i128)),
                Ordering::Greater
            );
            assert_eq!(
                BigInt::from(0x99887766554433221100i128)
                    .cmp(&BigInt::from(0x99888866554433221100i128)),
                Ordering::Less
            );
        }
        #[test]
        fn size_diff() {
            assert_eq!(
                BigInt::from(0xfffffffffffffffffffu128)
                    .cmp(&BigInt::from(0x99887766554433221100i128)),
                Ordering::Less
            );
        }
    }
    mod full_size {
        use super::*;

        #[test]
        fn load() {
            assert_eq!(
                FullSize::from(0x7766554433221100usize),
                FullSize::new(HalfSize::from(0x33221100), HalfSize::from(0x77665544))
            );
        }

        #[test]
        fn read() {
            assert_eq!(
                FullSize::from(0x7766554433221100usize).lower(),
                HalfSize::from(0x33221100)
            );
            assert_eq!(
                FullSize::from(0x7766554433221100usize).higher(),
                HalfSize::from(0x77665544)
            );
        }
    }
    pub(super) mod big_math {
        use super::*;
        pub fn test_op(
            lhs: impl Into<BigInt>,
            rhs: impl Into<BigInt>,
            op: impl for<'b> Fn(Boo<'b, BigInt>, Boo<'b, BigInt>) -> Moo<'b, BigInt>,
            result: impl Into<BigInt>,
        ) {
            let lhs = lhs.into();
            let rhs = rhs.into();
            let result = result.into();
            let validate = |res: Moo<BigInt>, dbg: &str| {
                assert_eq!(*res, result, "res equals with {dbg}");
            };
            let validate_mut = |res: Moo<BigInt>, dbg: &str| {
                assert!(matches!(res, Moo::BorrowedMut(_)), "res mut ref with {dbg}");
                validate(res, dbg);
            };
            let validate_non_mut = |res: Moo<BigInt>, dbg: &str| {
                assert!(matches!(res, Moo::Owned(_)), "res owned with {dbg}");
                validate(res, dbg);
            };
            {
                let mut lhs = lhs.clone();
                let res = op(Boo::from(&mut lhs), Boo::from(&rhs));
                validate_mut(res, "(&mut, &)");
                assert_eq!(lhs, result, "lhs assigned with (&mut, &)");
            }
            {
                let mut lhs = lhs.clone();
                let res = op(Boo::from(&mut lhs), Boo::from(rhs.clone()));
                validate_mut(res, "(&mut, o)");
                assert_eq!(lhs, result, "lhs assigned with (&mut, o)");
            }

            {
                let mut rhs = rhs.clone();
                let res = op(Boo::from(&lhs), Boo::from(&mut rhs));
                validate_mut(res, "(&, &mut)");
                assert_eq!(rhs, result, "rhs assigned with (&, &mut)");
            }
            {
                let mut rhs = rhs.clone();
                let res = op(Boo::from(lhs.clone()), Boo::from(&mut rhs));
                validate_mut(res, "(o, &mut)");
                assert_eq!(rhs, result, "rhs assigned with (o, &mut)");
            }

            let res = op(Boo::from(&lhs), Boo::from(&rhs));
            validate_non_mut(res, "res equals with (&, &)");

            let res = op(Boo::from(lhs.clone()), Boo::from(&rhs));
            validate_non_mut(res, "res equals with (o, &)");

            let res = op(Boo::from(&lhs), Boo::from(rhs.clone()));
            validate_non_mut(res, "res equals with (&, o)");

            let res = op(Boo::from(lhs), Boo::from(rhs));
            validate_non_mut(res, "res equals with (o, o)");
        }

        #[test]
        fn bit_or() {
            assert_eq!(
                BigInt::from(0x1111_00000000_00001111_01010101u128)
                    | BigInt::from(0x0101_01010101_11110000u128),
                BigInt::from(0x1111_00000101_01011111_11110101u128),
                "bigger lhs"
            );
            assert_eq!(
                BigInt::from(0x0101_01010101_11110000u128)
                    | BigInt::from(0x1111_00000000_00001111_01010101u128),
                BigInt::from(0x1111_00000101_01011111_11110101u128),
                "bigger rhs"
            );
        }
        #[test]
        fn bit_xor() {
            assert_eq!(
                BigInt::from(0x1111_00000000_00001111_01010101u128)
                    ^ BigInt::from(0x0101_01010101_11110000u128),
                BigInt::from(0x1111_00000101_01011010_10100101u128),
                "bigger lhs"
            );
            assert_eq!(
                BigInt::from(0x0101_01010101_11110000u128)
                    ^ BigInt::from(0x1111_00000000_00001111_01010101u128),
                BigInt::from(0x1111_00000101_01011010_10100101u128),
                "bigger rhs"
            );
        }

        #[test]
        fn bit_and() {
            assert_eq!(
                BigInt::from(0x1111_00000000_00001111_01010101u128)
                    & BigInt::from(0x0101_01010101_11110000u128),
                BigInt::from(0x0101_01010000u128),
                "bigger lhs"
            );
            assert_eq!(
                BigInt::from(0x0101_01010101_11110000u128)
                    & BigInt::from(0x1111_00000000_00001111_01010101u128),
                BigInt::from(0x0101_01010000u128),
                "bigger rhs"
            );
        }

        #[test]
        fn shl() {
            assert_eq!(
                BigInt::from(0x998877665544332211u128) << 4,
                BigInt::from(0x9988776655443322110u128)
            );
            assert_eq!(BigInt::from(1) << 1, BigInt::from(2));
        }
        #[test]
        fn shr() {
            assert_eq!(
                BigInt::from(0x998877665544332211u128) >> 4,
                BigInt::from(0x99887766554433221u128)
            )
        }
        #[test]
        fn add_overflow() {
            assert_eq!(
                BigInt::from(0xffff_ffff_ffff_ffffu64) + BigInt::from(1),
                BigInt::from(0x1_0000_0000_0000_0000u128),
                "bigger lhs"
            );
            assert_eq!(
                BigInt::from(1) + BigInt::from(0xffff_ffff_ffff_ffffu64),
                BigInt::from(0x1_0000_0000_0000_0000u128),
                "bigger rhs"
            );
        }
        #[test]
        fn add_middle_overflow() {
            assert_eq!(
                BigInt::from(0x1000_0000_ffff_ffff_ffff_ffffu128) + BigInt::from(1),
                BigInt::from(0x1000_0001_0000_0000_0000_0000u128),
                "bigger lhs"
            );
            assert_eq!(
                BigInt::from(1) + BigInt::from(0x1000_0000_ffff_ffff_ffff_ffffu128),
                BigInt::from(0x1000_0001_0000_0000_0000_0000u128),
                "bigger rhs"
            );
        }
        #[test]
        fn add_two_negative() {
            assert_eq!(
                BigInt::from(-0x11223344_55667788i128) + BigInt::from(-0x88776655_44332211i128),
                BigInt::from(-0x9999_9999_9999_9999i128)
            );
        }
        #[test]
        fn add() {
            assert_eq!(
                BigInt::from(0x11223344_55667788i128) - BigInt::from(-0x88776655_44332211i128),
                BigInt::from(0x9999_9999_9999_9999i128)
            );
            assert_eq!(
                BigInt::from(0x11223344_55667788i128) + BigInt::from(0x88776655_44332211i128),
                BigInt::from(0x9999_9999_9999_9999i128)
            );
        }

        #[test]
        fn sub() {
            assert_eq!(
                BigInt::from(0x9999_9999_9999_9999i128) - BigInt::from(0x88776655_44332211i128),
                BigInt::from(0x11223344_55667788i128)
            );
            assert_eq!(
                BigInt::from(0x9999_9999_9999_9999i128) + BigInt::from(-0x88776655_44332211i128),
                BigInt::from(0x11223344_55667788i128)
            );
        }
        #[test]
        fn sub_overflow() {
            assert_eq!(
                BigInt::from(0x1_0000_0000_0000_0000_0000_0000_0000i128) - BigInt::from(1),
                BigInt::from(0xffff_ffff_ffff_ffff_ffff_ffff_ffffi128)
            );
        }

        #[test]
        fn mul() {
            assert_eq!(BigInt::from(7) * BigInt::from(6), BigInt::from(42));
            assert_eq!(
                BigInt::from(30_000_000_700_000u128) * BigInt::from(60),
                BigInt::from(180_000_004_200_0000u128)
            );
        }
        #[test]
        fn mul_one_big() {
            assert_eq!(
                BigInt::from(0x0feeddcc_bbaa9988_77665544_33221100u128) * BigInt::from(2),
                [0x1fddbb9977553310eeccaa8866442200u128]
                    .into_iter()
                    .collect::<BigInt>(),
                "lhs big"
            );
            assert_eq!(
                BigInt::from(2) * BigInt::from(0x0feeddcc_bbaa9988_77665544_33221100u128),
                [0x1fddbb9977553310eeccaa8866442200u128]
                    .into_iter()
                    .collect::<BigInt>(),
                "rhs big"
            );
        }
        #[test]
        fn mul_sign_pow_two() {
            assert_eq!(BigInt::from(2) * BigInt::from(2), BigInt::from(4));
            assert_eq!(BigInt::from(-2) * BigInt::from(2), BigInt::from(-4));
            assert_eq!(BigInt::from(2) * BigInt::from(-2), BigInt::from(-4));
            assert_eq!(BigInt::from(-2) * BigInt::from(-2), BigInt::from(4));
        }
        #[test]
        fn mul_sign() {
            assert_eq!(BigInt::from(3) * BigInt::from(3), BigInt::from(9));
            assert_eq!(BigInt::from(-3) * BigInt::from(3), BigInt::from(-9));
            assert_eq!(BigInt::from(3) * BigInt::from(-3), BigInt::from(-9));
            assert_eq!(BigInt::from(-3) * BigInt::from(-3), BigInt::from(9));
        }
        #[test]
        fn mul_both_big() {
            assert_eq!(
                BigInt::from(0xffeeddcc_bbaa9988_77665544_33221100u128)
                    * BigInt::from(0xffeeddcc_bbaa9988_77665544_33221100u128),
                [
                    0x33432fd716ccd7135f999f4e85210000u128,
                    0xffddbcbf06b5eed38628ddc706bf1222u128,
                ]
                .into_iter()
                .collect::<BigInt>()
            );
        }

        #[test]
        fn log_2() {
            assert_eq!(BigInt::from(1).ilog2(), Some(0));
            assert_eq!(BigInt::from(2).ilog2(), Some(1));
            assert_eq!(BigInt::from(0x8000_0000_0000_0000u64).ilog2(), Some(63));
            assert_eq!(BigInt::from(0x1000_0001_0000_0000u64).ilog2(), None);
            assert_eq!(BigInt::from(0x1000_0000_1000_0000u64).ilog2(), None);
        }
    }
}
