use itertools::{Either, Itertools};
use std::fmt::Write;
use std::iter;
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Mul, MulAssign,
    Neg, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
};

use crate::boo::{Boo, Moo};
use math_shortcuts::MathShortcut;

mod digits;
mod math_algos;
mod math_shortcuts;
mod primitve;
use digits::{FullSize, HalfSize, HalfSizeNative, HALF_SIZE_BYTES};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(i8)]
pub enum SigNum {
    Negative = -1,
    Zero = 0,
    Positive = 1,
}
impl Default for SigNum {
    fn default() -> Self {
        Self::Zero
    }
}
impl From<SigNum> for i8 {
    fn from(value: SigNum) -> Self {
        value.into_i8()
    }
}
impl SigNum {
    const fn into_i8(self) -> i8 {
        self as i8
    }
    /// SAFETY: needs to be -1, 0 or 1
    const unsafe fn from_i8(value: i8) -> Self {
        #[allow(clippy::undocumented_unsafe_blocks)]
        unsafe {
            std::mem::transmute::<i8, Self>(value)
        }
    }
    const fn is_negative(self) -> bool {
        self.into_i8().is_negative()
    }
    const fn is_positive(self) -> bool {
        self.into_i8().is_positive()
    }
    const fn is_zero(self) -> bool {
        self.into_i8() == 0
    }
    const fn negate(self) -> Self {
        self.const_mul(Self::Negative)
    }
    const fn abs(self) -> Self {
        // SAFETY: can only be 0 or 1
        unsafe { Self::from_i8(self.into_i8().abs()) }
    }
    const fn const_mul(self, rhs: Self) -> Self {
        // SAFETY: can only be -1,0 or 1
        unsafe { Self::from_i8(self.into_i8() * rhs.into_i8()) }
    }
}
impl Neg for SigNum {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.negate()
    }
}
impl Mul for SigNum {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.const_mul(rhs)
    }
}
impl MulAssign for SigNum {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

trait TieBreaker {
    fn decide<'b>(lhs: Boo<'b, BigInt>, rhs: Boo<'b, BigInt>) -> (BigInt, Boo<'b, BigInt>);
}
struct TieSmaller;
impl TieBreaker for TieSmaller {
    fn decide<'b>(lhs: Boo<'b, BigInt>, rhs: Boo<'b, BigInt>) -> (BigInt, Boo<'b, BigInt>) {
        if *lhs <= *rhs {
            (lhs.cloned(), rhs)
        } else {
            (rhs.cloned(), lhs)
        }
    }
}
struct TieBigger;
impl TieBreaker for TieBigger {
    fn decide<'b>(lhs: Boo<'b, BigInt>, rhs: Boo<'b, BigInt>) -> (BigInt, Boo<'b, BigInt>) {
        if *lhs > *rhs {
            (lhs.cloned(), rhs)
        } else {
            (rhs.cloned(), lhs)
        }
    }
}

#[derive(Clone, Default, PartialEq, Eq)]
pub struct BigInt {
    signum: SigNum,
    /// holds the digits (in base 2^`HalfSize::BITS`) in LE order
    digits: Vec<HalfSize>,
}
impl std::fmt::Debug for BigInt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Number {{ {} 0x[",
            match self.signum {
                SigNum::Negative => "-",
                SigNum::Zero => "",
                SigNum::Positive => "+",
            }
        )?;
        for (pos, elem) in self.digits.iter().rev().with_position() {
            write!(f, "{elem:size$x}", size = HALF_SIZE_BYTES * 2)?;
            if matches!(
                pos,
                itertools::Position::First | itertools::Position::Middle
            ) {
                f.write_str(", ")?;
            }
        }
        write!(f, "]}}")
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
        for digit in self.digits.iter().rev() {
            write!(buf, "{digit:x}")?;
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
        for digit in self.digits.iter().rev() {
            write!(buf, "{digit:X}")?;
        }
        f.pad_integral(
            !self.is_negative(),
            if f.alternate() { "0X" } else { "" },
            &buf,
        )
    }
}

impl PartialOrd for BigInt {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for BigInt {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.signum
            .cmp(&other.signum)
            .then_with(|| self.abs_ord(other))
    }
}
impl From<HalfSize> for BigInt {
    fn from(value: HalfSize) -> Self {
        if *value == 0 {
            Self {
                signum: SigNum::Zero,
                digits: vec![],
            }
        } else {
            Self {
                signum: SigNum::Positive,
                digits: vec![value],
            }
        }
    }
}
impl FromIterator<HalfSize> for BigInt {
    fn from_iter<T: IntoIterator<Item = HalfSize>>(iter: T) -> Self {
        let mut num = Self::default();
        iter.into_iter()
            .for_each(|next| num.digits.push(next.into()));
        num.signum = SigNum::Positive;
        num.truncate_leading_zeros();
        num
    }
}
impl<POSITIVE: primitve::UNum> FromIterator<POSITIVE> for BigInt {
    /// the iter should contain the digits in little endian order
    fn from_iter<T: IntoIterator<Item = POSITIVE>>(iter: T) -> Self {
        let binding = iter
            .into_iter()
            .flat_map(primitve::Primitive::to_le_bytes)
            .chunks(HALF_SIZE_BYTES);

        Self::from_iter(binding.into_iter().map(|chunk| {
                let mut buf = [0; HALF_SIZE_BYTES];

                for (place, byte) in buf.iter_mut().zip(chunk) {
                    *place = byte;
                }
            HalfSize::from(buf)
        }))
    }
}
impl<PRIMITIVE: primitve::Primitive> From<PRIMITIVE> for BigInt {
    fn from(value: PRIMITIVE) -> Self {
        match value.select_sign() {
            Either::Left(pos) => iter::once(pos).collect(),
            Either::Right(neg) => {
                let mut num = iter::once(primitve::INum::abs(neg)).collect::<Self>();
                if primitve::INum::is_negative(neg) {
                    num.negate();
                }
                num
            }
        }
    }
}

impl BigInt {
    pub const BASIS_POW: usize = HalfSizeNative::BITS as usize;
    pub const BASIS: usize = 1 << Self::BASIS_POW;

    fn truncate_leading_zeros(&mut self) {
        while self.digits.last().is_some_and(|&it| *it == 0) {
            self.digits.pop();
        }

        // recalculates the sign
        if self.digits.is_empty() {
            self.signum = SigNum::Zero;
        } else {
            assert!(!self.signum.is_zero(), "found {self:?} with Signnum::Zero");
        }
    }
    fn pop(&mut self) {
        self.digits.pop();
        self.truncate_leading_zeros();
    }
    fn push(&mut self, value: impl Into<HalfSize>) {
        let value = value.into();
        if *value == 0 {
            return;
        }
        self.digits.push(value);
    }

    pub const fn is_negative(&self) -> bool {
        self.signum.is_negative()
    }
    pub const fn is_positive(&self) -> bool {
        self.signum.is_positive()
    }
    pub const fn is_zero(&self) -> bool {
        self.signum.is_zero()
    }
    pub fn is_abs_one(&self) -> bool {
        self.digits.len() == 1 && *self.digits[0] == 1
    }
    const fn is_different_sign(&self, rhs: &Self) -> bool {
        !self.is_negative() ^ !rhs.is_negative()
    }

    /// currently only works with powers of two
    pub fn digits(&self, radix: usize) -> usize {
        if radix == Self::BASIS {
            return self.digits.len();
        }
        if radix == 2 {
            return self.digits.last().map_or(0, |last| {
                self.digits(Self::BASIS) * Self::BASIS_POW - last.leading_zeros() as usize
            });
        }
        if radix.is_power_of_two() {
            return self.digits(2).div_ceil(radix / 2);
        }

        unimplemented!("need division");
    }

    pub fn ilog2(&self) -> Option<usize> {
        let mut canditate = None;
        for (i, digit) in self.digits.iter().enumerate().filter(|&(_, it)| **it != 0) {
            if canditate.is_some() || !(*digit).is_power_of_two() {
                return None;
            }
            canditate = Some((*digit).ilog2() as usize + i * HalfSizeNative::BITS as usize);
        }
        canditate
    }
    pub fn abs_ord(&self, rhs: &Self) -> std::cmp::Ordering {
        self.digits
            .len()
            .cmp(&rhs.digits.len())
            .then_with(|| self.digits.iter().rev().cmp(rhs.digits.iter().rev()))
    }
    #[must_use]
    pub fn abs_clone(&self) -> Self {
        let mut out = self.clone();
        out.abs();
        out
    }

    pub fn negate(&mut self) {
        self.signum = -self.signum;
    }
    pub fn abs(&mut self) {
        self.signum = self.signum.abs();
    }

    fn assert_pair_valid(lhs: &Boo<'_, Self>, rhs: &Boo<'_, Self>) {
        assert!(
            !matches!(lhs, Boo::BorrowedMut(_)) || !matches!(rhs, Boo::BorrowedMut(_)),
            "can't have to Borrow_mut's"
        );
    }
    fn refer_direct<'b, 'b1: 'b, 'b2: 'b, B1, B2, T>(
        lhs: B1,
        rhs: B2,
        func: impl FnOnce(&mut Self, &Self),
    ) -> Moo<'b, Self>
    where
        B1: Into<Boo<'b1, Self>>,
        B2: Into<Boo<'b2, Self>>,
        T: TieBreaker,
    {
        let lhs: Boo<'_, Self> = lhs.into();
        let rhs: Boo<'_, Self> = rhs.into();
        Self::assert_pair_valid(&lhs, &rhs);

        match (lhs, rhs) {
            (Boo::BorrowedMut(borrow_mut), borrow) | (borrow, Boo::BorrowedMut(borrow_mut)) => {
                func(borrow_mut, &borrow);
                Moo::BorrowedMut(borrow_mut)
            }
            (Boo::Borrowed(borrowed), Boo::Owned(mut owned))
            | (Boo::Owned(mut owned), Boo::Borrowed(borrowed)) => {
                func(&mut owned, borrowed);
                Moo::Owned(owned)
            }
            (lhs, rhs) => {
                let (mut owned, borrowed) = T::decide(lhs, rhs);
                func(&mut owned, &borrowed);
                Moo::Owned(owned)
            }
        }
    }

    fn bitor<'b, 'b1: 'b, 'b2: 'b, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b, Self>
    where
        B1: Into<Boo<'b1, Self>>,
        B2: Into<Boo<'b2, Self>>,
    {
        Self::refer_direct::<'_, '_, '_, _, _, TieBigger>(
            lhs,
            rhs,
            math_algos::bit_math::bit_or_assign,
        )
    }
    fn bitxor<'b, 'b1: 'b, 'b2: 'b, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b, Self>
    where
        B1: Into<Boo<'b1, Self>>,
        B2: Into<Boo<'b2, Self>>,
    {
        Self::refer_direct::<'_, '_, '_, _, _, TieBigger>(
            lhs,
            rhs,
            math_algos::bit_math::bit_xor_assign,
        )
    }

    fn bitand<'b, 'b1: 'b, 'b2: 'b, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b, Self>
    where
        B1: Into<Boo<'b1, Self>>,
        B2: Into<Boo<'b2, Self>>,
    {
        Self::refer_direct::<'_, '_, '_, _, _, TieSmaller>(
            lhs,
            rhs,
            math_algos::bit_math::bit_and_assign,
        )
    }

    fn shl<'b, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b, Self>
    where
        B1: Into<Boo<'b, Self>>,
        B2: Into<Boo<'b, usize>>,
    {
        let mut lhs = Moo::<Self>::from(lhs.into());
        let rhs = rhs.into().copied();

        let partial = rhs % HalfSizeNative::BITS as usize;
        let full = rhs / HalfSizeNative::BITS as usize;

        let mut carry = HalfSize::default();
        if partial > 0 {
            for digit in &mut lhs.digits {
                let old_carry = carry;
                let result = FullSize::from(*FullSize::new(*digit, HalfSize::default()) << partial);
                *digit = result.lower() | old_carry;
                carry = result.higher();
            }
        }
        let carry = Some(carry).filter(|&it| *it != 0);
        if carry.is_some() || full > 0 {
            lhs.digits = std::iter::repeat(HalfSize::default())
                .take(full)
                .chain(lhs.digits.iter().copied())
                .chain(carry)
                .collect();
        }
        lhs
    }
    fn shr<'b, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b, Self>
    where
        B1: Into<Boo<'b, Self>>,
        B2: Into<Boo<'b, usize>>,
    {
        Self::shr_internal(lhs, rhs).0
    }

    fn shr_internal<'b, B1, B2>(lhs: B1, rhs: B2) -> (Moo<'b, Self>, BigInt)
    where
        B1: Into<Boo<'b, Self>>,
        B2: Into<Boo<'b, usize>>,
    {
        let mut lhs = Moo::<Self>::from(lhs.into());
        let rhs = rhs.into().copied();

        let partial = rhs % HalfSizeNative::BITS as usize;
        let full = rhs / HalfSizeNative::BITS as usize;

        let mut carry = HalfSize::default();
        if partial > 0 {
            for digit in lhs.digits.iter_mut().rev() {
                let old_carry = carry;
                let result = FullSize::from(*FullSize::new(HalfSize::default(), *digit) >> partial);
                *digit = result.higher() | old_carry;
                carry = result.lower();
            }
        }
        let mut overflow;
        if full > 0 {
            let mut iter = lhs.digits.iter().copied();
            overflow = BigInt::from_iter(iter::once(carry).chain(iter.by_ref().take(full)));
            if partial != 0 {
                overflow >>= HalfSizeNative::BITS as usize - partial;
            } else {
                overflow.pop();
            }
            lhs.digits = iter.collect();
        } else {
            overflow = BigInt::from(carry);
        }
        lhs.truncate_leading_zeros();
        (lhs, overflow)
    }

    fn add<'b, 'b1: 'b, 'b2: 'b, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b, Self>
    where
        B1: Into<Boo<'b1, Self>>,
        B2: Into<Boo<'b2, Self>>,
    {
        let lhs: Boo<'_, Self> = lhs.into();
        let rhs: Boo<'_, Self> = rhs.into();
        Self::assert_pair_valid(&lhs, &rhs);

        math_shortcuts::try_all!(lhs, rhs, math_shortcuts::add::Zero,);

        if lhs.is_different_sign(&rhs) {
            return match (lhs, rhs) {
                (Boo::Borrowed(lhs), rhs) => {
                    let mut either = Moo::<Self>::from(rhs);
                    either.negate();
                    Self::sub(lhs, either)
                }
                (Boo::Owned(lhs), Boo::Owned(mut rhs)) => {
                    rhs.negate();
                    Self::sub(lhs, rhs)
                }
                (lhs, rhs) => {
                    let mut either = Moo::<Self>::from(lhs);
                    either.negate();
                    either = Self::sub(either, rhs);
                    either.negate();
                    either
                }
            };
        }
        Self::refer_direct::<'_, '_, '_, _, _, TieSmaller>(
            lhs,
            rhs,
            math_algos::add::assign_same_sign,
        )
    }
    fn sub<'b, 'b1: 'b, 'b2: 'b, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b, Self>
    where
        B1: Into<Boo<'b1, Self>>,
        B2: Into<Boo<'b2, Self>>,
    {
        let lhs: Boo<'_, Self> = lhs.into();
        let rhs: Boo<'_, Self> = rhs.into();
        Self::assert_pair_valid(&lhs, &rhs);

        math_shortcuts::try_all!(lhs, rhs, math_shortcuts::sub::Zero,);

        if lhs.is_different_sign(&rhs) {
            return match (lhs, rhs) {
                (Boo::Borrowed(lhs), rhs) => {
                    let mut either = Moo::<Self>::from(rhs);
                    either.negate();
                    Self::add(lhs, either)
                }
                (Boo::Owned(lhs), Boo::Owned(mut rhs)) => {
                    rhs.negate();
                    Self::add(lhs, rhs)
                }
                (lhs, rhs) => {
                    let mut either = Moo::<Self>::from(lhs);
                    either.negate();
                    either = Self::add(either, rhs);
                    either.negate();
                    either
                }
            };
        }

        let (lhs, rhs, signum) = if lhs.abs_ord(&rhs).is_lt() {
            (rhs, lhs, SigNum::Negative)
        } else {
            (lhs, rhs, SigNum::Positive)
        };

        let mut either = match (lhs, rhs) {
            (Boo::BorrowedMut(lhs), rhs) => {
                math_algos::sub::assign_smaller_same_sign(lhs, &rhs);
                Moo::BorrowedMut(lhs)
            }
            (lhs, Boo::BorrowedMut(borrowed)) => {
                let old_rhs = std::mem::replace(borrowed, lhs.cloned()); // lhs -> rhs, rhs -> old_rhs
                math_algos::sub::assign_smaller_same_sign(borrowed, &old_rhs);
                Moo::BorrowedMut(borrowed)
            }
            (lhs, rhs) => {
                // can't really use storage in rhs (when existing) because algo can only sub smaller
                let mut lhs = lhs.cloned();
                math_algos::sub::assign_smaller_same_sign(&mut lhs, &rhs);
                Moo::Owned(lhs)
            }
        };
        either.signum *= signum;
        either
    }

    fn mul_by_digit<'b, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b, Self>
    where
        B1: Into<Boo<'b, Self>>,
        B2: Into<Boo<'b, HalfSize>>,
    {
        let lhs: Boo<'_, Self> = lhs.into();
        let rhs: HalfSize = rhs.into().copied();

        if lhs.is_zero() {
            return lhs.into();
        }
        if *rhs == 0 {
            return match lhs {
                Boo::BorrowedMut(lhs) => {
                    *lhs = Self::default();
                    Moo::BorrowedMut(lhs)
                }
                _ => Moo::Owned(Self::default()),
            };
        }
        let mut lhs = Moo::from(lhs);
        if *rhs == 1 {
            return lhs;
        }
        if rhs.is_power_of_two() {
            return Self::shl(lhs, rhs.ilog2() as usize);
        }
        math_algos::mul::assign_mul_digit_at_offset(&mut lhs, rhs, 0);
        lhs
    }
    fn mul<'b, 'b1: 'b, 'b2: 'b, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b, Self>
    where
        B1: Into<Boo<'b1, Self>>,
        B2: Into<Boo<'b2, Self>>,
    {
        let lhs: Boo<'_, Self> = lhs.into();
        let rhs: Boo<'_, Self> = rhs.into();
        Self::assert_pair_valid(&lhs, &rhs);

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

macro_rules! implBigMath {
    ($($assign_trait:tt)::*, $assign_func:ident, $($trait:tt)::*, $func:ident) => {
        implBigMath!($($assign_trait)::*, $assign_func, $($trait)::*, $func, $func, BigInt);
    };
    ($($assign_trait:tt)::*, $assign_func:ident, $($trait:tt)::*, $func:ident, $ref_func:ident, $rhs: ident) => {
        impl $($trait)::*<$rhs> for BigInt {
            implBigMath!(body $func, $ref_func, $rhs);
        }
        impl $($trait)::*<&$rhs> for BigInt {
            implBigMath!(body $func, $ref_func, &$rhs);
        }
        impl $($trait)::*<$rhs> for &BigInt {
            implBigMath!(body $func, $ref_func, $rhs);
        }
        impl $($trait)::*<&$rhs> for &BigInt {
            implBigMath!(body $func, $ref_func, &$rhs);
        }
        impl $($assign_trait)::*<$rhs> for BigInt {
            fn $assign_func(&mut self, rhs: $rhs) {
                BigInt::$ref_func(self, rhs).expect_mut_ref("did give &mut, shouldn't get result");
            }
        }
        impl $($assign_trait)::*<&$rhs> for BigInt {
            fn $assign_func(&mut self, rhs: &$rhs) {
                BigInt::$ref_func(self, rhs).expect_mut_ref("did give &mut, shouldn't get result");
            }
        }
    };
    (body $func:tt, $ref_func:ident, $rhs:tt) => {
        type Output = BigInt;
        fn $func(self, rhs: $rhs) -> Self::Output {
            BigInt::$ref_func(self, rhs).expect_owned("didn't give &mut, should get result")
        }
    };
    (body $func:tt, $ref_func:ident, &$rhs:tt) => {
        type Output = BigInt;
        fn $func(self, rhs: &$rhs) -> Self::Output {
            BigInt::$ref_func(self, rhs).expect_owned("didn't give &mut, should get result")
        }
    };
}

// no `std::ops::Not`, cause implied zeros to the left would need to be flipped
impl Neg for BigInt {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.negate();
        self
    }
}
implBigMath!(BitOrAssign, bitor_assign, BitOr, bitor);
implBigMath!(BitXorAssign, bitxor_assign, BitXor, bitxor);
implBigMath!(BitAndAssign, bitand_assign, BitAnd, bitand);
implBigMath!(ShlAssign, shl_assign, Shl, shl, shl, usize);
implBigMath!(ShrAssign, shr_assign, Shr, shr, shr, usize);
implBigMath!(SubAssign, sub_assign, Sub, sub);
implBigMath!(AddAssign, add_assign, Add, add);
implBigMath!(MulAssign, mul_assign, Mul, mul, mul_by_digit, HalfSize);
implBigMath!(MulAssign, mul_assign, Mul, mul);

#[cfg(test)]
mod tests;
