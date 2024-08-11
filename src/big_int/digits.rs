use itertools::Itertools;
use std::{
    fmt::{Debug, LowerHex, UpperHex},
    ops::{Add, BitAndAssign, BitOrAssign, BitXor, BitXorAssign, Mul, Shl, Shr},
};

use super::SigNum;

pub trait Decomposable<D> {
    fn signum(&self) -> SigNum;
    fn into_le_bytes(self) -> impl ExactSizeIterator<Item = D> + DoubleEndedIterator;
    fn as_le_bytes<'s, 'd: 's>(&'s self) -> impl ExactSizeIterator<Item = &D> + DoubleEndedIterator
    where
        D: 'd;
}

/// A 'Digit' for a Bigint
/// is assumed to be exactly stored as a number of bytes (its basis is 2^8n for some Interger n)
pub trait Digit:
    Copy
    + Default
    + Debug
    + Eq
    + Ord
    + Decomposable<Self>
    + LowerHex
    + UpperHex
    + PartialOrd<u8>
    + PartialEq<u8>
    + From<bool>
where
    for<'r> Self: BitOrAssign<&'r Self>
        + BitAndAssign<&'r Self>
        + BitXorAssign<&'r Self>
        + BitXor<&'r Self, Output = Self>,
{
    const BYTES: usize;
    type Wide: Wide<Self>;

    fn from_le(bytes: impl Iterator<Item = u8>) -> Vec<Self>;
    fn ilog2(&self) -> u32;
    fn is_power_of_two(&self) -> bool;

    /// ((0, self) << rhl) | (0, in_carry) = (out_carry, res)
    /// carry should have `rhs` bits of data and is padded to the left with zeros
    fn widening_shl(self, rhs: usize, carry: Self) -> Self::Wide {
        debug_assert_eq!(
            (Self::Wide::new(carry, Self::default()) >> rhs)
                .split_le()
                .0,
            0u8,
            "carry has more than {rhs} bits info"
        );
        let mut full = Self::Wide::widen(self) << rhs;
        full.split_le_mut().0.bitor_assign(&carry);
        full
    }
    /// ((self, 0) >> rhl) | (in_carry, 0) = (res, out_carry)
    /// carry should have `rhs` bits of data and is padded to the right with zeros
    fn widening_shr(self, rhs: usize, carry: Self) -> Self::Wide {
        debug_assert_eq!(
            (Self::Wide::new(Self::default(), carry) << rhs)
                .split_le()
                .1,
            0u8,
            "carry has more than {rhs} bits info"
        );
        let mut full = Self::Wide::new(Self::default(), self) >> rhs;
        full.split_le_mut().1.bitor_assign(&carry);
        full
    }

    fn overflowing_add(self, rhs: Self) -> (Self, bool);
    fn carring_add(self, rhs: Self, in_carry: bool) -> (Self, bool) {
        let (res, carry_1) = self.overflowing_add(rhs);
        let (res, carry_2) = res.overflowing_add(Self::from(in_carry));
        (res, carry_1 | carry_2)
    }
    fn overflowing_sub(self, rhs: Self) -> (Self, bool);
    fn carring_sub(self, rhs: Self, in_carry: bool) -> (Self, bool) {
        let (res, carry_1) = self.overflowing_sub(rhs);
        let (res, carry_2) = res.overflowing_sub(Self::from(in_carry));
        (res, carry_1 | carry_2)
    }
    /// ((0, self) * (0, rhs)) + (0, carry_in) = (carry_out, result)
    fn widening_mul(self, rhs: Self, carry: Self) -> Self::Wide {
        Self::Wide::widen(self) * Self::Wide::widen(rhs) + Self::Wide::widen(carry)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn widening_shl() {
        assert_eq!(
            HalfSize::new(0x8765_4321).widening_shl(4, HalfSize::new(0x9)),
            0x8_7654_3219usize
        )
    }
    #[test]
    fn widening_shr() {
        assert_eq!(
            HalfSize::new(0x8765_4321).widening_shr(4, HalfSize::new(0x9000_0000)),
            0x9876_5432_1000_0000usize
        )
    }
}
pub trait Wide<Half>:
    Ord
    + Decomposable<Half>
    + Shl<usize, Output = Self>
    + Shr<usize, Output = Self>
    + Mul<Output = Self>
    + Add<Output = Self>
    + Sized
    + Debug
{
    fn new(lower: Half, upper: Half) -> Self;
    fn widen(value: Half) -> Self;
    fn split_le(self) -> (Half, Half);
    fn split_le_mut(&mut self) -> (&mut Half, &mut Half);
}
impl Wide<HalfSize> for usize {
    fn widen(value: HalfSize) -> Self {
        *value as Self
    }

    fn new(lower: HalfSize, upper: HalfSize) -> Self {
        *FullSize::new(lower, upper)
    }

    fn split_le(self) -> (HalfSize, HalfSize) {
        let full = FullSize::from(self);
        (full.lower(), full.higher())
    }
    fn split_le_mut(&mut self) -> (&mut HalfSize, &mut HalfSize) {
        let full = if cfg!(target_endian = "little") {
            unsafe { std::mem::transmute::<&mut usize, &mut FullSize>(self) }
        } else {
            unimplemented!()
        };
        full.split_le_mut()
    }
}

impl Decomposable<Self> for HalfSize {
    fn signum(&self) -> SigNum {
        SigNum::from_uint(**self == 0)
    }

    fn into_le_bytes(self) -> impl ExactSizeIterator<Item = Self> + DoubleEndedIterator {
        std::iter::once(self)
    }

    fn as_le_bytes<'s, 'd: 's>(
        &'s self,
    ) -> impl ExactSizeIterator<Item = &Self> + DoubleEndedIterator {
        std::iter::once(self)
    }
}
impl PartialEq<u8> for HalfSize {
    fn eq(&self, other: &u8) -> bool {
        self.partial_cmp(other).is_some_and(|it| it.is_eq())
    }
}
impl PartialOrd<u8> for HalfSize {
    fn partial_cmp(&self, other: &u8) -> Option<std::cmp::Ordering> {
        Some(u8::try_from(**self).map_or(std::cmp::Ordering::Greater, |it| it.cmp(other)))
    }
}
impl From<bool> for HalfSize {
    fn from(value: bool) -> Self {
        Self::new(value as HalfSizeNative)
    }
}
impl Digit for HalfSize {
    const BYTES: usize = HALF_SIZE_BYTES;
    type Wide = usize;

    fn from_le(bytes: impl Iterator<Item = u8>) -> Vec<Self> {
        let chunks = bytes.chunks(HALF_SIZE_BYTES);
        chunks
            .into_iter()
            .map(|chunk| {
                let mut buf = [0; Self::BYTES];

                for (place, byte) in buf.iter_mut().zip(chunk) {
                    *place = byte;
                }
                Self::from(buf)
            })
            .collect_vec()
    }
    fn ilog2(&self) -> u32 {
        (**self).ilog2()
    }

    fn is_power_of_two(&self) -> bool {
        (**self).is_power_of_two()
    }

    fn overflowing_add(self, rhs: Self) -> (Self, bool) {
        let (res, carry) = (*self).overflowing_add(*rhs);
        (Self::new(res), carry)
    }
    fn overflowing_sub(self, rhs: Self) -> (Self, bool) {
        let (res, carry) = (*self).overflowing_sub(*rhs);
        (Self::new(res), carry)
    }
}

#[cfg(target_pointer_width = "64")]
pub type HalfSizeNative = u32;
#[cfg(target_pointer_width = "32")]
pub type HalfSizeNative = u16;
#[cfg(target_pointer_width = "16")]
pub type HalfSizeNative = u8;
pub const HALF_SIZE_BYTES: usize = HalfSizeNative::BITS as usize / 8;
pub const FULL_SIZE_BYTES: usize = usize::BITS as usize / 8;
const _: () = {
    #[allow(clippy::manual_assert)]
    if HALF_SIZE_BYTES * 2 != FULL_SIZE_BYTES {
        panic!("what?");
    }
};
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
        std::fmt::LowerHex::fmt(
            &if cfg!(target_endian = "little") {
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
        std::fmt::UpperHex::fmt(
            &if cfg!(target_endian = "little") {
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
        Self::new(value)
    }
}
impl From<[u8; HALF_SIZE_BYTES]> for HalfSize {
    fn from(value: [u8; HALF_SIZE_BYTES]) -> Self {
        Self { ne_bytes: value }
    }
}

impl HalfSize {
    pub const fn new(native: HalfSizeNative) -> Self {
        Self { native }
    }
    fn format_index(index: usize) -> usize {
        assert!(index < HALF_SIZE_BYTES);
        if cfg!(target_endian = "little") {
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
impl std::ops::Deref for HalfSize {
    type Target = HalfSizeNative;

    fn deref(&self) -> &Self::Target {
        // SAFETY: union will always be properly initialized
        unsafe { &self.native }
    }
}
impl std::ops::DerefMut for HalfSize {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: union will always be properly initialized
        unsafe { &mut self.native }
    }
}

/// access le ordered bytes
impl std::ops::Index<usize> for HalfSize {
    type Output = u8;

    fn index(&self, index: usize) -> &Self::Output {
        // SAFETY: union will always be properly initialized
        unsafe { self.ne_bytes.index(Self::format_index(index)) }
    }
}
/// access le ordered bytes
impl std::ops::IndexMut<usize> for HalfSize {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        // SAFETY: union will always be properly initialized
        unsafe { self.ne_bytes.index_mut(Self::format_index(index)) }
    }
}

impl Decomposable<HalfSize> for usize {
    fn signum(&self) -> SigNum {
        SigNum::from_uint(*self == 0)
    }

    fn into_le_bytes(self) -> impl ExactSizeIterator<Item = HalfSize> + DoubleEndedIterator {
        let (lower, upper) = self.split_le();

        [lower, upper].into_iter()
    }

    fn as_le_bytes<'s, 'd: 's>(
        &'s self,
    ) -> impl ExactSizeIterator<Item = &HalfSize> + DoubleEndedIterator
    where
        HalfSize: 'd,
    {
        let full = unsafe { std::mem::transmute::<&usize, &FullSize>(self) };
        if cfg!(target_endian = "little") {
            [full.lower_ref(), full.higher_ref()].into_iter()
        } else {
            unimplemented!()
        }
    }
}
#[derive(Clone, Copy)]
pub union FullSize {
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
        std::fmt::LowerHex::fmt(
            &if cfg!(target_endian = "little") {
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
        std::fmt::UpperHex::fmt(
            &if cfg!(target_endian = "little") {
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

impl PartialOrd for FullSize {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for FullSize {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (**self).cmp(&**other)
    }
}

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
    pub const fn new(lower: HalfSize, higher: HalfSize) -> Self {
        if cfg!(target_endian = "little") {
            Self {
                halfs: [lower, higher],
            }
        } else {
            Self {
                halfs: [higher, lower],
            }
        }
    }
    pub const fn lower(self) -> HalfSize {
        *self.lower_ref()
    }
    pub const fn higher(self) -> HalfSize {
        *self.higher_ref()
    }
    pub const fn lower_ref(&self) -> &HalfSize {
        if cfg!(target_endian = "little") {
            unsafe { &self.halfs[0] }
        } else {
            unsafe { &self.halfs[1] }
        }
    }
    pub const fn higher_ref(&self) -> &HalfSize {
        if cfg!(target_endian = "little") {
            unsafe { &self.halfs[1] }
        } else {
            unsafe { &self.halfs[0] }
        }
    }
    pub fn split_le_mut(&mut self) -> (&mut HalfSize, &mut HalfSize) {
        let [a, b] = unsafe { &mut self.halfs };
        if cfg!(target_endian = "little") {
            (a, b)
        } else {
            (b, a)
        }
    }
}

impl std::ops::Deref for FullSize {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        // SAFETY: union will always be properly initialized
        unsafe { &self.native }
    }
}
impl std::ops::DerefMut for FullSize {
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
