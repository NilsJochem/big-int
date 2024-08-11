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
    Copy + Default + Debug + Eq + Ord + Decomposable<Self> + LowerHex + UpperHex
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

    fn eq_u8(&self, other: u8) -> bool {
        self.cmp_u8(other).is_eq()
    }
    fn cmp_u8(&self, other: u8) -> std::cmp::Ordering;
    fn from_bool(value: bool) -> Self;

    /// ((0, `self`) << `rhl`) | (0, `in_carry`) = (`out_carry`, `res`)
    /// carry should have `rhs` bits of data and is padded to the left with zeros
    fn widening_shl(self, rhs: usize, carry: Self) -> Self::Wide {
        debug_assert!(
            (Self::Wide::new(carry, Self::default()) >> rhs)
                .split_le()
                .0
                .eq_u8(0u8),
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
                .eq_u8(0u8),
            "carry has more than {rhs} bits info"
        );
        let mut full = Self::Wide::new(Self::default(), self) >> rhs;
        full.split_le_mut().1.bitor_assign(&carry);
        full
    }

    fn overflowing_add(self, rhs: Self) -> (Self, bool);
    fn carring_add(self, rhs: Self, in_carry: bool) -> (Self, bool) {
        let (res, carry_1) = self.overflowing_add(rhs);
        let (res, carry_2) = res.overflowing_add(Self::from_bool(in_carry));
        (res, carry_1 | carry_2)
    }
    fn overflowing_sub(self, rhs: Self) -> (Self, bool);
    fn carring_sub(self, rhs: Self, in_carry: bool) -> (Self, bool) {
        let (res, carry_1) = self.overflowing_sub(rhs);
        let (res, carry_2) = res.overflowing_sub(Self::from_bool(in_carry));
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
    + Sized
    + Debug
{
    fn new(lower: Half, upper: Half) -> Self;
    fn widen(value: Half) -> Self;
    fn split_le(self) -> (Half, Half);
    fn split_le_mut(&mut self) -> (&mut Half, &mut Half);
}

#[cfg(target_pointer_width = "64")]
pub type HalfSize = u32;
#[cfg(target_pointer_width = "32")]
pub type HalfSizeNative = u16;
#[cfg(target_pointer_width = "16")]
pub type HalfSizeNative = u8;
const _: () = {
    #[allow(clippy::manual_assert)]
    if HalfSize::BYTES * 2 != usize::BITS as usize / 8 {
        panic!("what?");
    }
};

impl Decomposable<Self> for HalfSize {
    fn signum(&self) -> SigNum {
        SigNum::from_uint(*self == 0)
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
impl Digit for HalfSize {
    const BYTES: usize = Self::BITS as usize / 8;
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

    fn cmp_u8(&self, other: u8) -> std::cmp::Ordering {
        u8::try_from(*self).map_or(std::cmp::Ordering::Greater, |it| it.cmp(&other))
    }
    fn from_bool(value: bool) -> Self {
        value as Self
    }

    fn overflowing_add(self, rhs: Self) -> (Self, bool) {
        self.overflowing_add(rhs)
    }
    fn overflowing_sub(self, rhs: Self) -> (Self, bool) {
        self.overflowing_sub(rhs)
    }
}

impl Decomposable<HalfSize> for usize {
    fn signum(&self) -> SigNum {
        SigNum::from_uint(*self == 0)
    }

    fn into_le_bytes(self) -> impl ExactSizeIterator<Item = HalfSize> + DoubleEndedIterator {
        <[_; 2]>::from(self.split_le()).into_iter()
    }

    fn as_le_bytes<'s, 'd: 's>(
        &'s self,
    ) -> impl ExactSizeIterator<Item = &HalfSize> + DoubleEndedIterator
    where
        HalfSize: 'd,
    {
        #[allow(clippy::undocumented_unsafe_blocks)]
        let [a, b] = unsafe { &*((std::ptr::from_ref(self)).cast()) };
        if cfg!(target_endian = "little") {
            [a, b].into_iter()
        } else {
            [b, a].into_iter()
        }
    }
}
impl Wide<HalfSize> for usize {
    fn widen(value: HalfSize) -> Self {
        value as Self
    }

    fn new(lower: HalfSize, upper: HalfSize) -> Self {
        ((upper as Self) << HalfSize::BITS as Self) + lower as Self
    }

    #[allow(clippy::tuple_array_conversions)]
    fn split_le(self) -> (HalfSize, HalfSize) {
        #[allow(clippy::undocumented_unsafe_blocks)]
        let [a, b] = unsafe { std::mem::transmute::<Self, [HalfSize; 2]>(self) };
        if cfg!(target_endian = "little") {
            (a, b)
        } else {
            (b, a)
        }
    }
    #[allow(clippy::tuple_array_conversions)]
    fn split_le_mut(&mut self) -> (&mut HalfSize, &mut HalfSize) {
        #[allow(clippy::undocumented_unsafe_blocks)]
        let [a, b] = unsafe { &mut *((std::ptr::from_mut(self)).cast()) };
        if cfg!(target_endian = "little") {
            (a, b)
        } else {
            (b, a)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn widening_shl() {
        assert_eq!(0x8765_4321.widening_shl(4, 0x9), 0x8_7654_3219usize);
    }
    #[test]
    fn widening_shr() {
        assert_eq!(
            0x8765_4321.widening_shr(4, 0x9000_0000),
            0x9876_5432_1000_0000usize
        );
    }

    mod full_size {
        use super::*;

        #[test]
        fn load() {
            assert_eq!(
                0x7766_5544_3322_1100usize,
                usize::new(0x3322_1100, 0x7766_5544)
            );
        }

        #[test]
        fn read() {
            assert_eq!(
                0x7766_5544_3322_1100usize.split_le(),
                (0x3322_1100, 0x7766_5544)
            );
        }
    }
}
