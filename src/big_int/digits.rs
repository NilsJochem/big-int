use itertools::Itertools;
use std::{
    convert::Infallible,
    fmt::{Debug, LowerHex, UpperHex},
    ops::{Add, BitAndAssign, BitOrAssign, BitXor, BitXorAssign, Div, Mul, Shl, Shr, ShrAssign},
};

use crate::big_int::signed::SigNum;

pub trait Signed {
    fn signum(&self) -> SigNum;
}
pub trait Decomposable<D>: Convert<usize> {
    fn le_digits(&self) -> impl ExactSizeIterator<Item = D> + DoubleEndedIterator + '_;
}
pub trait Convert<D> {
    fn try_into(&self) -> Option<D>;
}
pub trait CombineTo<B> {
    fn try_combine(a: impl Iterator<Item = Self>) -> Option<B>;
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
    + Decomposable<u8>
    + CombineTo<usize>
    + LowerHex
    + UpperHex
    + From<u8>
    + From<bool>
    + ShrAssign<usize>
    + Shl<usize, Output = Self>
    + 'static
where
    for<'r> Self: BitOrAssign<&'r Self>
        + BitAndAssign<&'r Self>
        + BitXorAssign<&'r Self>
        + BitXor<&'r Self, Output = Self>,
{
    const BYTES: usize;
    const BASIS_POW: usize = (Self::BYTES * 8);
    const MAX: Self;
    type Wide: Wide<Self>;

    fn from_le(bytes: impl Iterator<Item = u8>) -> Vec<Self>;
    fn ilog2(&self) -> u32;
    fn is_power_of_two(&self) -> bool;
    fn is_even(&self) -> bool;
    fn get_bit(&self, i: usize) -> bool;
    fn iter_le_bits(
        &self,
        all: bool,
    ) -> impl ExactSizeIterator<Item = bool> + DoubleEndedIterator + '_ {
        (0..if all {
            Self::BASIS_POW
        } else {
            Self::ilog2(self) as usize + 1
        })
            .map(|i| self.get_bit(i))
    }

    fn cmp_u8(&self, other: u8) -> std::cmp::Ordering;
    fn eq_u8(&self, other: u8) -> bool {
        self.cmp_u8(other).is_eq()
    }

    /// ((0, `self`) << `rhl`) | (0, `in_carry`) = (`out_carry`, `res`)
    /// carry should have `rhs` bits of data and is padded to the left with zeros
    fn widening_shl(self, rhs: usize, carry: Self) -> Self::Wide {
        debug_assert!(
            (Self::Wide::new(carry, Self::default()) >> rhs)
                .split_le()
                .0
                .eq_u8(0),
            "carry has more than {rhs} bits info"
        );
        let mut full = Self::Wide::widen(self) << rhs;
        full.split_le_mut().0.bitor_assign(&carry);
        full
    }
    /// ((`self`, 0) >> `rhl`) | (`in_carry`, 0) = (`res`, `out_carry`)
    /// carry should have `rhs` bits of data and is padded to the right with zeros
    fn widening_shr(self, rhs: usize, carry: Self) -> Self::Wide {
        debug_assert!(
            (Self::Wide::new(Self::default(), carry) << rhs)
                .split_le()
                .1
                .eq_u8(0),
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
    /// ((0, `self`) * (0, `rhs`)) + (0, `carry_in`) = (`carry_out`, `result`)
    fn widening_mul(self, rhs: Self, carry: Self) -> Self::Wide {
        Self::Wide::widen(self) * Self::Wide::widen(rhs) + Self::Wide::widen(carry)
    }
}
pub trait Wide<Half>:
    Ord
    + Decomposable<Half>
    + Shl<usize, Output = Self>
    + Shr<usize, Output = Self>
    + Mul<Output = Self>
    + Add<Output = Self>
    + Div<Output = Self>
    + Sized
    + Debug
{
    fn new(lower: Half, upper: Half) -> Self;
    fn widen(value: Half) -> Self;
    fn split_le(self) -> (Half, Half);
    fn split_le_mut(&mut self) -> (&mut Half, &mut Half);
}

#[derive(Clone, Copy)]
union WideConvert<Half: Copy, Wide: Copy> {
    parts: WideHalfs<Half>,
    wide: Wide,
}
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct WideHalfs<Half> {
    #[cfg(target_endian = "little")]
    lower: Half,
    upper: Half,
    #[cfg(not(target_endian = "little"))]
    lower: Half,
}

#[allow(clippy::undocumented_unsafe_blocks)]
impl<H: Copy, W: Copy + Wide<H>> WideConvert<H, W> {
    const fn from_parts(lower: H, upper: H) -> Self {
        Self {
            parts: WideHalfs { lower, upper },
        }
    }
    const fn from_wide(wide: W) -> Self {
        Self { wide }
    }
    fn from_wide_mut(wide: &mut W) -> &mut Self {
        unsafe { &mut *((std::ptr::from_mut(wide)).cast()) }
    }

    const fn wide(self) -> W {
        unsafe { self.wide }
    }
    const fn halfs(self) -> WideHalfs<H> {
        unsafe { self.parts }
    }
}
#[allow(clippy::undocumented_unsafe_blocks)]
impl<H: Copy, W: Copy + Wide<H>> std::ops::Deref for WideConvert<H, W> {
    type Target = WideHalfs<H>;
    fn deref(&self) -> &Self::Target {
        unsafe { &self.parts }
    }
}
#[allow(clippy::undocumented_unsafe_blocks)]
impl<H: Copy, W: Copy + Wide<H>> std::ops::DerefMut for WideConvert<H, W> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut self.parts }
    }
}

macro_rules! implDigit {
    ($digit:ident, $wide: ident) => {
        impl Convert<usize> for $digit {
            fn try_into(&self) -> Option<usize> {
                usize::try_from(*self).ok()
            }
        }
        impl Signed for $digit {
            fn signum(&self) -> SigNum {
                SigNum::from_uint(*self == 0)
            }
        }
        impl Decomposable<Self> for $digit {
            fn le_digits(&self) -> impl ExactSizeIterator<Item = Self> + DoubleEndedIterator + '_ {
                std::iter::once(*self)
            }
        }
        impl Decomposable<bool> for $digit {
            fn le_digits(&self) -> impl ExactSizeIterator<Item = bool> + DoubleEndedIterator + '_ {
                self.iter_le_bits(false)
            }
        }
        impl Digit for $digit {
            const BYTES: usize = Self::BITS as usize / 8;
            const MAX: Self = Self::MAX;
            type Wide = $wide;

            fn from_le(bytes: impl Iterator<Item = u8>) -> Vec<Self> {
                let chunks = bytes.chunks(Self::BYTES);
                chunks
                    .into_iter()
                    .map(|chunk| {
                        let mut buf = [0; Self::BYTES];

                        for (place, byte) in buf.iter_mut().zip(chunk) {
                            *place = byte;
                        }
                        Self::from_le_bytes(buf)
                    })
                    .collect_vec()
            }
            fn ilog2(&self) -> u32 {
                (*self).ilog2()
            }
            fn is_power_of_two(&self) -> bool {
                (*self).is_power_of_two()
            }
            fn is_even(&self) -> bool {
                self % 2 == 0
            }
            fn get_bit(&self, i: usize) -> bool {
                (self & 1 << i) != 0
            }

            fn cmp_u8(&self, other: u8) -> std::cmp::Ordering {
                u8::try_from(*self).map_or(std::cmp::Ordering::Greater, |it| it.cmp(&other))
            }

            fn overflowing_add(self, rhs: Self) -> (Self, bool) {
                self.overflowing_add(rhs)
            }
            fn overflowing_sub(self, rhs: Self) -> (Self, bool) {
                self.overflowing_sub(rhs)
            }
        }

        impl Decomposable<$digit> for $wide {
            fn le_digits(
                &self,
            ) -> impl ExactSizeIterator<Item = $digit> + DoubleEndedIterator + '_ {
                <[_; 2]>::from(self.split_le()).into_iter()
            }
        }
        impl Wide<$digit> for $wide {
            fn widen(value: $digit) -> Self {
                value as Self
            }

            fn new(lower: $digit, upper: $digit) -> Self {
                WideConvert::from_parts(lower, upper).wide()
            }

            #[allow(clippy::tuple_array_conversions)]
            fn split_le(self) -> ($digit, $digit) {
                let halfs = WideConvert::from_wide(self).halfs();
                (halfs.lower, halfs.upper)
            }
            #[allow(clippy::tuple_array_conversions)]
            fn split_le_mut(&mut self) -> (&mut $digit, &mut $digit) {
                let halfs: &mut WideHalfs<$digit> = &mut *WideConvert::from_wide_mut(self);
                (&mut halfs.lower, &mut halfs.upper)
            }
        }
    };
}
// TODO add wrapper struct to add direct Ord<u8>
implDigit!(u8, u16);
implDigit!(u16, u32);
implDigit!(u32, u64);
implDigit!(u64, u128);

impl Signed for u128 {
    fn signum(&self) -> SigNum {
        SigNum::from_uint(*self == 0)
    }
}
impl Convert<usize> for u128 {
    fn try_into(&self) -> Option<usize> {
        usize::try_from(*self).ok()
    }
}
impl Decomposable<u8> for u32 {
    fn le_digits(&self) -> impl ExactSizeIterator<Item = u8> + DoubleEndedIterator + '_ {
        self.to_le_bytes().into_iter()
    }
}
impl Decomposable<u8> for u64 {
    fn le_digits(&self) -> impl ExactSizeIterator<Item = u8> + DoubleEndedIterator + '_ {
        self.to_le_bytes().into_iter()
    }
}
impl CombineTo<usize> for u8 {
    fn try_combine(iter: impl Iterator<Item = Self>) -> Option<usize> {
        let bytes: [Self; usize::BITS as usize / 8] = iter.collect::<Vec<_>>().try_into().ok()?;
        Some(usize::from_le_bytes(bytes))
    }
}
impl CombineTo<usize> for u16 {
    fn try_combine(iter: impl Iterator<Item = Self>) -> Option<usize> {
        if cfg!(target_pointer_width = "16") {
            iter.exactly_one().ok().map(|it| it as usize)
        } else {
            u8::try_combine(iter.flat_map(Self::to_le_bytes))
        }
    }
}
impl CombineTo<usize> for u32 {
    fn try_combine(iter: impl Iterator<Item = Self>) -> Option<usize> {
        if cfg!(target_pointer_width = "32") {
            iter.exactly_one().ok().map(|it| it as usize)
        } else {
            u8::try_combine(iter.flat_map(Self::to_le_bytes))
        }
    }
}
impl CombineTo<usize> for u64 {
    fn try_combine(iter: impl Iterator<Item = Self>) -> Option<usize> {
        if cfg!(target_pointer_width = "64") {
            iter.exactly_one().ok().map(|it| it as usize)
        } else {
            u8::try_combine(iter.flat_map(Self::to_le_bytes))
        }
    }
}

impl Convert<usize> for i32 {
    fn try_into(&self) -> Option<usize> {
        usize::try_from(*self).ok()
    }
}
impl Signed for i32 {
    fn signum(&self) -> SigNum {
        match self.cmp(&0) {
            std::cmp::Ordering::Less => SigNum::Negative,
            std::cmp::Ordering::Equal => SigNum::Zero,
            std::cmp::Ordering::Greater => SigNum::Positive,
        }
    }
}
impl Decomposable<bool> for i32 {
    fn le_digits(&self) -> impl ExactSizeIterator<Item = bool> + DoubleEndedIterator + '_ {
        <u32 as Decomposable<bool>>::le_digits(&self.unsigned_abs())
            .collect_vec()
            .into_iter()
    }
}

#[cfg(target_pointer_width = "64")]
type HalfSizeNative = u32;
#[cfg(target_pointer_width = "32")]
type HalfSizeNative = u16;
#[cfg(target_pointer_width = "16")]
type HalfSizeNative = u8;
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, derive_more::From)]
pub struct HalfSize(HalfSizeNative);

impl LowerHex for HalfSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        LowerHex::fmt(&self.0, f)
    }
}
impl UpperHex for HalfSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        UpperHex::fmt(&self.0, f)
    }
}
impl BitOrAssign<&Self> for HalfSize {
    fn bitor_assign(&mut self, rhs: &Self) {
        self.0 |= rhs.0;
    }
}
impl BitXorAssign<&Self> for HalfSize {
    fn bitxor_assign(&mut self, rhs: &Self) {
        self.0 ^= rhs.0;
    }
}
impl BitAndAssign<&Self> for HalfSize {
    fn bitand_assign(&mut self, rhs: &Self) {
        self.0 &= rhs.0;
    }
}
impl BitXor<&Self> for HalfSize {
    type Output = Self;

    fn bitxor(self, rhs: &Self) -> Self::Output {
        Self(self.0 ^ rhs.0)
    }
}
impl ShrAssign<usize> for HalfSize {
    fn shr_assign(&mut self, rhs: usize) {
        self.0 >>= rhs;
    }
}
impl Shl<usize> for HalfSize {
    type Output = Self;
    fn shl(self, rhs: usize) -> Self::Output {
        Self(self.0 << rhs)
    }
}

impl From<u8> for HalfSize {
    fn from(value: u8) -> Self {
        Self(value as HalfSizeNative)
    }
}
impl From<bool> for HalfSize {
    fn from(value: bool) -> Self {
        Self(value as HalfSizeNative)
    }
}
impl TryFrom<HalfSize> for usize {
    type Error = Infallible;

    fn try_from(value: HalfSize) -> Result<Self, Self::Error> {
        Ok(value.0 as Self)
    }
}
impl Convert<usize> for HalfSize {
    fn try_into(&self) -> Option<usize> {
        Some(self.0 as usize)
    }
}

impl PartialOrd<u8> for HalfSize {
    fn partial_cmp(&self, other: &u8) -> Option<std::cmp::Ordering> {
        self.partial_cmp(&Self::from(*other))
    }
}
impl PartialEq<u8> for HalfSize {
    fn eq(&self, other: &u8) -> bool {
        self.eq(&Self::from(*other))
    }
}
impl Signed for HalfSize {
    fn signum(&self) -> SigNum {
        SigNum::from_uint(self.0 == 0)
    }
}
impl Decomposable<Self> for HalfSize {
    fn le_digits(&self) -> impl ExactSizeIterator<Item = Self> + DoubleEndedIterator + '_ {
        std::iter::once(*self)
    }
}
impl Decomposable<u8> for HalfSize {
    fn le_digits(&self) -> impl ExactSizeIterator<Item = u8> + DoubleEndedIterator + '_ {
        self.0.to_le_bytes().into_iter()
    }
}
impl Decomposable<bool> for HalfSize {
    fn le_digits(&self) -> impl ExactSizeIterator<Item = bool> + DoubleEndedIterator + '_ {
        <HalfSizeNative as Decomposable<bool>>::le_digits(&self.0)
    }
}
impl CombineTo<usize> for HalfSize {
    fn try_combine(a: impl Iterator<Item = Self>) -> Option<usize> {
        let (low, high) = a.collect_tuple()?;
        Some(usize::new(low, high))
    }
}
impl Digit for HalfSize {
    const BYTES: usize = HalfSizeNative::BITS as usize / 8;
    const MAX: Self = Self(HalfSizeNative::MAX);
    type Wide = usize;
    fn from_le(bytes: impl Iterator<Item = u8>) -> Vec<Self> {
        let chunks = bytes.chunks(Self::BYTES);
        chunks
            .into_iter()
            .map(|chunk| {
                let mut buf = [0; Self::BYTES];
                for (place, byte) in buf.iter_mut().zip(chunk) {
                    *place = byte;
                }
                Self(HalfSizeNative::from_le_bytes(buf))
            })
            .collect_vec()
    }
    fn ilog2(&self) -> u32 {
        self.0.ilog2()
    }
    fn is_power_of_two(&self) -> bool {
        self.0.is_power_of_two()
    }
    fn is_even(&self) -> bool {
        self.0 % 2 == 0
    }
    fn get_bit(&self, i: usize) -> bool {
        (self.0 & 1 << i) != 0
    }

    fn cmp_u8(&self, other: u8) -> std::cmp::Ordering {
        u8::try_from(self.0).map_or(std::cmp::Ordering::Greater, |it| it.cmp(&other))
    }
    fn overflowing_add(self, rhs: Self) -> (Self, bool) {
        let (result, carry) = self.0.overflowing_add(rhs.0);
        (Self(result), carry)
    }
    fn overflowing_sub(self, rhs: Self) -> (Self, bool) {
        let (result, carry) = self.0.overflowing_sub(rhs.0);
        (Self(result), carry)
    }
}

impl Convert<Self> for usize {
    fn try_into(&self) -> Option<Self> {
        Some(*self)
    }
}
impl Signed for usize {
    fn signum(&self) -> SigNum {
        SigNum::from_uint(*self == 0)
    }
}
impl Decomposable<HalfSize> for usize {
    fn le_digits(&self) -> impl ExactSizeIterator<Item = HalfSize> + DoubleEndedIterator + '_ {
        <[_; 2]>::from(self.split_le()).into_iter()
    }
}
impl Wide<HalfSize> for usize {
    fn widen(value: HalfSize) -> Self {
        value.0 as Self
    }
    fn new(lower: HalfSize, upper: HalfSize) -> Self {
        WideConvert::from_parts(lower, upper).wide()
    }
    #[allow(clippy::tuple_array_conversions)]
    fn split_le(self) -> (HalfSize, HalfSize) {
        let halfs = WideConvert::from_wide(self).halfs();
        (halfs.lower, halfs.upper)
    }
    #[allow(clippy::tuple_array_conversions)]
    fn split_le_mut(&mut self) -> (&mut HalfSize, &mut HalfSize) {
        let halfs: &mut WideHalfs<HalfSize> = &mut *WideConvert::from_wide_mut(self);
        (&mut halfs.lower, &mut halfs.upper)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn widening_shl() {
        assert_eq!(
            HalfSize(0x8765_4321).widening_shl(4, HalfSize(0x9)),
            0x8_7654_3219usize
        );
    }
    #[test]
    fn widening_shr() {
        assert_eq!(
            HalfSize(0x8765_4321).widening_shr(4, HalfSize(0x9000_0000)),
            0x9876_5432_1000_0000usize
        );
    }

    mod full_size {
        use super::*;

        #[test]
        fn load() {
            assert_eq!(
                0x7766_5544_3322_1100usize,
                usize::new(HalfSize(0x3322_1100), HalfSize(0x7766_5544))
            );
        }

        #[test]
        fn read() {
            assert_eq!(
                0x7766_5544_3322_1100usize.split_le(),
                (HalfSize(0x3322_1100), HalfSize(0x7766_5544))
            );
        }
    }
}
