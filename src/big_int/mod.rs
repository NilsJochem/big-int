use common::require;
use itertools::{Either, Itertools};
use rand::RngCore;
use std::fmt::{Debug, Write};
use std::iter;
use std::num::NonZero;
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Mul, MulAssign, Neg, RangeInclusive, Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub,
    SubAssign,
};

use crate::boo::{Boo, Moo};
use math_shortcuts::MathShortcut;

pub mod digits;
pub mod math_algos;
mod math_shortcuts;
mod primitve;
use digits::{Decomposable, Digit, Wide};

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
    const fn from_uint(is_zero: bool) -> Self {
        // SAFETY: either 0 or 1
        unsafe { Self::from_i8(!is_zero as i8) }
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
    fn decide<'b, D: Digit>(
        lhs: Boo<'b, BigInt<D>>,
        rhs: Boo<'b, BigInt<D>>,
    ) -> (BigInt<D>, Boo<'b, BigInt<D>>);
}
struct TieSmaller;
impl TieBreaker for TieSmaller {
    fn decide<'b, D: Digit>(
        lhs: Boo<'b, BigInt<D>>,
        rhs: Boo<'b, BigInt<D>>,
    ) -> (BigInt<D>, Boo<'b, BigInt<D>>) {
        if *lhs <= *rhs {
            (lhs.cloned(), rhs)
        } else {
            (rhs.cloned(), lhs)
        }
    }
}
struct TieBigger;
impl TieBreaker for TieBigger {
    fn decide<'b, D: Digit>(
        lhs: Boo<'b, BigInt<D>>,
        rhs: Boo<'b, BigInt<D>>,
    ) -> (BigInt<D>, Boo<'b, BigInt<D>>) {
        if *lhs > *rhs {
            (lhs.cloned(), rhs)
        } else {
            (rhs.cloned(), lhs)
        }
    }
}

#[derive(Clone, Default)]
pub struct BigInt<D> {
    /// the sign of the number or zero <=> `digits.is_empty()`
    signum: SigNum,
    /// holds the digits in LE order
    digits: Vec<D>,
}
impl<D: Digit> std::fmt::Debug for BigInt<D> {
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
            write!(f, "{elem:0size$x}", size = D::BYTES * 2)?;
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
impl<D: Digit> std::fmt::Display for BigInt<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let radix = Self::from(10);
        let mut number = self.clone();
        let has_sign = number.is_negative() || f.sign_plus();
        if number.is_negative() {
            f.write_char('-')?;
            number.signum = SigNum::Positive;
        } else if f.sign_plus() {
            f.write_char('+')?;
        }
        let mut buf = Vec::new();
        while !number.is_zero() {
            let (_, mut remainder) = Self::div_mod_euclid(&mut number, &radix);
            debug_assert!(remainder.digits.len() <= 1);
            buf.push(remainder.digits.pop().unwrap_or_default());
        }
        if buf.is_empty() {
            buf.push(D::default());
        }
        if let Some(pad) = f.width() {
            match f.align() {
                Some(std::fmt::Alignment::Left) => {
                    buf.extend(
                        std::iter::repeat(D::default())
                            .take(pad - has_sign as usize)
                            .skip(buf.len()),
                    );
                }
                Some(std::fmt::Alignment::Right) => {
                    todo!("need option to have ' ' in buf")
                    // buf = std::iter::repeat(' ')
                    //     .take(pad - has_sign as usize)
                    //     .skip(buf.len())
                    //     .chain(buf)
                    //     .collect();
                }
                Some(std::fmt::Alignment::Center) => todo!("not ready"),
                None => {}
            }
        }
        if f.alternate() {
            for (pos, digits) in buf
                .iter()
                .chunks(3)
                .into_iter()
                .collect_vec()
                .into_iter()
                .rev()
                .with_position()
            {
                for digit in digits.collect_vec().into_iter().rev() {
                    write!(f, "{digit:?}")?;
                }
                match pos {
                    itertools::Position::Middle | itertools::Position::First => {
                        f.write_char('_')?;
                    }
                    itertools::Position::Last | itertools::Position::Only => {}
                }
            }
        } else {
            for digit in buf.iter().rev() {
                write!(f, "{digit:?}")?;
            }
        }
        Ok(())
    }
}
impl<D: Digit> std::fmt::LowerHex for BigInt<D> {
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
impl<D: Digit> std::fmt::UpperHex for BigInt<D> {
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

impl<D: Digit> Eq for BigInt<D> {}
impl<D: Digit> Ord for BigInt<D> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl<D: Digit> Decomposable<D> for BigInt<D> {
    fn signum(&self) -> SigNum {
        self.signum
    }

    fn into_le_bytes(self) -> impl ExactSizeIterator<Item = D> + DoubleEndedIterator {
        self.digits.into_iter()
    }

    fn as_le_bytes<'s, 'd: 's>(&'s self) -> impl ExactSizeIterator<Item = &D> + DoubleEndedIterator
    where
        D: 'd,
    {
        self.digits.iter()
    }
}

impl<D: Digit, M: Decomposable<D>> PartialEq<M> for BigInt<D> {
    fn eq(&self, other: &M) -> bool {
        self.partial_cmp(other)
            .is_some_and(std::cmp::Ordering::is_eq)
    }
}
impl<D: Digit, M: Decomposable<D>> PartialOrd<M> for BigInt<D> {
    fn partial_cmp(&self, other: &M) -> Option<std::cmp::Ordering> {
        Some(self.signum.cmp(&other.signum()).then_with(|| {
            let digits = other.as_le_bytes();

            for elem in self.digits.iter().zip_longest(digits).rev() {
                match elem {
                    itertools::EitherOrBoth::Both(lhs, rhs) => {
                        let ord = lhs.cmp(rhs);
                        if ord.is_ne() {
                            return ord;
                        }
                    }
                    itertools::EitherOrBoth::Right(_) => return std::cmp::Ordering::Less,
                    itertools::EitherOrBoth::Left(_) => return std::cmp::Ordering::Greater,
                }
            }
            std::cmp::Ordering::Equal
        }))
    }
}

impl<POSITIVE: primitve::UNum, D: Digit> FromIterator<POSITIVE> for BigInt<D> {
    /// the iter should contain the digits in little endian order
    fn from_iter<T: IntoIterator<Item = POSITIVE>>(iter: T) -> Self {
        Self::from_digits(D::from_le(
            iter.into_iter().flat_map(primitve::Primitive::to_le_bytes),
        ))
    }
}
impl<PRIMITIVE: primitve::Primitive, D: Digit> From<PRIMITIVE> for BigInt<D> {
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Radix<D: Digit> {
    DigitBase,
    /// should not be `DigitBase`
    PowerOfTwo(NonZero<usize>),
    Other(BigInt<D>),
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RadixError {
    Zero,
    One,
    Negative,
}
impl<D: Digit> Radix<D> {
    const fn from_pow_2(power: NonZero<usize>) -> Self {
        if D::BYTES * 8 == power.get() {
            Self::DigitBase
        } else {
            Self::PowerOfTwo(power)
        }
    }
}
impl<D: Digit> TryFrom<usize> for Radix<D> {
    type Error = RadixError;

    fn try_from(value: usize) -> Result<Self, RadixError> {
        require!(value != 0, RadixError::Zero);
        require!(value != 1, RadixError::One);
        Ok(if value.is_power_of_two() {
            Self::from_pow_2(
                NonZero::new(value.ilog2() as usize).unwrap_or_else(|| {
                    unreachable!("ilog2 doesn't return 0 unless the value is 0")
                }),
            )
        } else {
            Self::Other(value.into())
        })
    }
}
impl<D: Digit> TryFrom<BigInt<D>> for Radix<D> {
    type Error = RadixError;

    fn try_from(value: BigInt<D>) -> Result<Self, RadixError> {
        require!(value.is_zero(), RadixError::Zero);
        require!(value.is_negative(), RadixError::Negative);
        require!(value.is_abs_one(), RadixError::One);
        Ok(if value.is_power_of_two() {
            Self::from_pow_2(
                NonZero::new(value.ilog(2)).unwrap_or_else(|| {
                    unreachable!("ilog2 doesn't return 0 unless the value is 0")
                }),
            )
        } else {
            Self::Other(value)
        })
    }
}

impl<D: Digit> BigInt<D> {
    const NONZERO_ONE: NonZero<usize> = {
        // SAFETY: 1 is non zero
        unsafe { NonZero::new_unchecked(1) }
    };

    /// generate a new random number with at least `bytes.start()` and at most `bytes.end()` bytes of information
    /// # Example
    /// `0x00_0100` <= `BigInt::new_random(2..=3, _)` <= `0xff_ffff`,
    pub fn new_random(bytes: RangeInclusive<usize>, mut rng: impl RngCore) -> Self {
        let bytes =
            bytes.start() + crate::rng::next_bound(*bytes.end() - *bytes.start(), &mut rng, 10);
        let mut rnd_bytes = crate::rng::random_bytes(rng);
        let last = rnd_bytes
            .by_ref()
            .take(5) // cap the number of tries
            .find(|&it| it > 0)
            .expect("only zeros found");
        rnd_bytes
            .take(bytes - 1)
            .chain(std::iter::once(last))
            .collect()
    }

    pub fn from_digit(value: D) -> Self {
        if value.eq_u8(0) {
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
    pub fn from_digits(iter: impl IntoIterator<Item = D>) -> Self {
        let mut num = Self {
            signum: SigNum::Positive,
            digits: iter.into_iter().collect_vec(),
        };
        num.truncate_leading_zeros();
        num
    }

    fn truncate_leading_zeros(&mut self) {
        while self.digits.last().is_some_and(|&it| it.eq_u8(0)) {
            self.digits.pop();
        }

        // recalculates the sign
        if self.digits.is_empty() {
            self.signum = SigNum::Zero;
        } else {
            assert!(!self.signum.is_zero(), "found {self:?} with Signnum::Zero");
        }
    }
    fn push(&mut self, value: impl Into<D>) {
        let value = value.into();
        if value.eq_u8(0) {
            return;
        }
        self.digits.push(value);
    }

    pub const fn signum(&self) -> SigNum {
        self.signum
    }
    pub const fn is_negative(&self) -> bool {
        self.signum().is_negative()
    }
    pub const fn is_positive(&self) -> bool {
        self.signum().is_positive()
    }
    pub const fn is_zero(&self) -> bool {
        self.signum().is_zero()
    }
    pub fn is_abs_one(&self) -> bool {
        self.digits.len() == 1 && self.digits[0].eq_u8(1)
    }
    const fn is_different_sign(&self, rhs: &Self) -> bool {
        !self.is_negative() ^ !rhs.is_negative()
    }

    pub fn digits<T>(&self, radix: T) -> usize
    where
        T: TryInto<Radix<D>>,
        T::Error: Debug,
    {
        self.try_digits(radix).unwrap()
    }

    pub fn try_digits<T>(&self, radix: T) -> Result<usize, T::Error>
    where
        T: TryInto<Radix<D>>,
    {
        fn inner<D: Digit>(number: &BigInt<D>, radix: Radix<D>) -> usize {
            match radix {
                Radix::DigitBase => number.digits.len(),
                Radix::PowerOfTwo(BigInt::<D>::NONZERO_ONE) => {
                    number.digits.last().map_or(0, |last| {
                        (inner(number, Radix::DigitBase) - 1) * D::BASIS_POW
                            + last.ilog2() as usize
                            + 1
                    })
                }
                Radix::PowerOfTwo(power) => {
                    inner(number, Radix::PowerOfTwo(BigInt::<D>::NONZERO_ONE))
                        .div_ceil(1 << (power.get() - 1))
                }
                Radix::Other(radix) => {
                    let mut n = 1;
                    let mut number = number.clone();
                    while number.abs_ord(&radix).is_ge() {
                        n += 1;
                        number /= &radix;
                    }
                    n
                }
            }
        }

        if self.is_zero() {
            return Ok(0);
        }
        Ok(inner(self, radix.try_into()?))
    }

    pub fn ilog<T>(&self, radix: T) -> usize
    where
        T: TryInto<Radix<D>>,
        T::Error: Debug,
    {
        assert!(!self.is_zero(), "can't 0.log(radix)");
        self.try_ilog(radix).unwrap()
    }

    pub fn try_ilog<T>(&self, radix: T) -> Result<usize, T::Error>
    where
        T: TryInto<Radix<D>>,
    {
        assert!(!self.is_zero(), "can't 0.log(radix)");
        self.try_digits(radix).map(|it| it - 1)
    }
    pub fn is_power_of_two(&self) -> bool {
        self.digits.last().map_or(false, Digit::is_power_of_two)
            && self.digits.iter().rev().skip(1).all(|&it| it.eq_u8(0))
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
    pub fn take_sign(&mut self) -> SigNum {
        let signum = self.signum;
        self.abs();
        signum
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
        D: 'b1 + 'b2,
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
        D: 'b1 + 'b2,
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
        D: 'b1 + 'b2,
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
        D: 'b1 + 'b2,
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

        let partial = rhs % D::BASIS_POW;
        let full = rhs / D::BASIS_POW;

        let mut carry = D::default();
        if partial > 0 {
            for digit in &mut lhs.digits {
                (*digit, carry) = digit.widening_shl(rhs, carry).split_le();
            }
        }
        let carry = Some(carry).filter(|&it| !it.eq_u8(0));
        if carry.is_some() || full > 0 {
            lhs.digits = std::iter::repeat(D::default())
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

    fn shr_internal<'b, B1, B2>(lhs: B1, rhs: B2) -> (Moo<'b, Self>, Self)
    where
        B1: Into<Boo<'b, Self>>,
        B2: Into<Boo<'b, usize>>,
    {
        let mut lhs = Moo::<Self>::from(lhs.into());
        let rhs = rhs.into().copied();

        let partial = rhs % D::BASIS_POW;
        let full = rhs / D::BASIS_POW;

        let mut carry = D::default();
        if partial > 0 {
            for digit in lhs.digits.iter_mut().rev() {
                (carry, *digit) = digit.widening_shr(partial, carry).split_le();
            }
        }
        let mut overflow;
        if full > 0 {
            let mut iter = lhs.digits.iter().copied();
            overflow = Self::from_digits(iter::once(carry).chain(iter.by_ref().take(full)));
            if partial != 0 {
                overflow >>= D::BASIS_POW - partial;
            } else if !overflow.is_zero() {
                overflow.digits.remove(0);
                overflow.truncate_leading_zeros();
            }
            lhs.digits = iter.collect();
        } else {
            if partial > 0 {
                carry >>= D::BASIS_POW - partial;
            }
            overflow = Self::from_digit(carry);
        }
        lhs.truncate_leading_zeros();
        (lhs, overflow)
    }

    pub(crate) fn add<'b, 'b1: 'b, 'b2: 'b, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b, Self>
    where
        B1: Into<Boo<'b1, Self>>,
        B2: Into<Boo<'b2, Self>>,
        D: 'b1 + 'b2,
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
    pub(crate) fn sub<'b, 'b1: 'b, 'b2: 'b, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b, Self>
    where
        B1: Into<Boo<'b1, Self>>,
        B2: Into<Boo<'b2, Self>>,
        D: 'b1 + 'b2,
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
        *either *= signum;
        either
    }

    pub(crate) fn mul_by_digit<'b, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b, Self>
    where
        B1: Into<Boo<'b, Self>>,
        B2: Into<Boo<'b, D>>,
    {
        let lhs: Boo<'_, Self> = lhs.into();
        let rhs: D = rhs.into().copied();

        if lhs.is_zero() {
            return lhs.into();
        }
        if rhs.eq_u8(0) {
            return Moo::from_with_value(lhs, Self::default());
        }
        let mut lhs = Moo::from(lhs);
        if rhs.eq_u8(1) {
            return lhs;
        }
        if rhs.is_power_of_two() {
            return Self::shl(lhs, rhs.ilog2() as usize);
        }
        math_algos::mul::assign_mul_digit_at_offset(&mut lhs, rhs, 0);
        lhs
    }

    pub(crate) fn mul_by_sign<'b, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b, Self>
    where
        B1: Into<Boo<'b, Self>>,
        B2: Into<Boo<'b, SigNum>>,
    {
        let mut lhs = Moo::<Self>::from(lhs.into());
        let rhs = rhs.into().copied();
        if rhs == SigNum::Zero {
            *lhs = Self::from(0);
        } else {
            lhs.signum *= rhs;
        }
        lhs
    }

    pub(crate) fn mul<'b, 'b1: 'b, 'b2: 'b, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b, Self>
    where
        B1: Into<Boo<'b1, Self>>,
        B2: Into<Boo<'b2, Self>>,
        D: 'b1 + 'b2,
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

    pub(crate) fn div_euclid<'b, 'b1: 'b, 'b2: 'b, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b, Self>
    where
        B1: Into<Boo<'b1, Self>>,
        B2: Into<Boo<'b2, Self>>,
        D: 'b1 + 'b2,
    {
        let lhs: Boo<'_, Self> = lhs.into();
        let rhs: Boo<'_, Self> = rhs.into();

        Self::assert_pair_valid(&lhs, &rhs);
        match (lhs, rhs) {
            (lhs, Boo::BorrowedMut(rhs)) => {
                let (result, _) = Self::div_mod_euclid(lhs, std::mem::take(rhs));
                Moo::from_with_value(rhs, result.expect_owned("did'nt hat mut ref"))
            }
            (lhs, rhs) => Self::div_mod_euclid(lhs, rhs).0,
        }
    }
    #[allow(dead_code)]
    pub(crate) fn rem_euclid<'b, 'b1: 'b, 'b2: 'b, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b, Self>
    where
        B1: Into<Boo<'b1, Self>>,
        B2: Into<Boo<'b2, Self>>,
        D: 'b1 + 'b2,
    {
        let lhs: Boo<'_, Self> = lhs.into();
        let rhs: Boo<'_, Self> = rhs.into();

        Self::assert_pair_valid(&lhs, &rhs);
        match (lhs, rhs) {
            (Boo::BorrowedMut(lhs), rhs) => {
                let (_, result) = Self::div_mod_euclid(std::mem::take(lhs), rhs);
                Moo::from_with_value(lhs, result.expect_owned("did'nt hat mut ref"))
            }
            (lhs, rhs) => Self::div_mod_euclid(lhs, rhs).1,
        }
    }
    pub fn div_mod_euclid<'b, 'b1: 'b, 'b2: 'b, B1, B2>(
        lhs: B1,
        rhs: B2,
    ) -> (Moo<'b, Self>, Moo<'b, Self>)
    where
        B1: Into<Boo<'b1, Self>>,
        B2: Into<Boo<'b2, Self>>,
        D: 'b1 + 'b2,
    {
        let lhs: Boo<'_, Self> = lhs.into();
        let rhs: Boo<'_, Self> = rhs.into();
        // here both can be allowed to be &muts in which case *lhs = lhs/rhs, *rhs = lhs%rhs
        // Self::assert_pair_valid(&lhs, &rhs);

        assert!(!rhs.is_zero(), "can't divide by zero");

        math_shortcuts::try_all!(
            lhs,
            rhs,
            left math_shortcuts::div::Same,
            left math_shortcuts::div::Smaller,
            right math_shortcuts::div::ByPowerOfTwo,
        );

        let (mut n, lhs) = lhs.take_keep_ref();
        let (mut d, rhs) = rhs.take_keep_ref();

let map_r = n.is_negative().then(|| d.abs_clone());
        let signum = n.take_sign() * d.take_sign();

        let (mut q, mut r) = math_algos::div::normalized_schoolbook(n, d);

        q.signum = signum;
if let Some(d) = map_r.filter(|_| !r.is_zero()) {
            q -= Self::from(1);
            r = d - r;
        }

        (Moo::from_with_value(lhs, q), Moo::from_with_value(rhs, r))
    }
}

macro_rules! implBigMath {
    ($($assign_trait:tt)::*, $assign_func:ident, $($trait:tt)::*, $func:ident) => {
        implBigMath!($($assign_trait)::*, $assign_func, $($trait)::*, $func, $func, BigInt<D>);
    };
    ($($assign_trait:tt)::*, $assign_func:ident, $($trait:tt)::*, $func:ident, $ref_func:ident, $rhs: ident$(<$gen:ident>)?) => {
        impl<D: Digit> $($trait)::*<$rhs$(<$gen>)?> for BigInt<D> {
            implBigMath!(body $func, $ref_func, $rhs$(<$gen>)?);
        }
        impl<D: Digit> $($trait)::*<&$rhs$(<$gen>)?> for BigInt<D> {
            implBigMath!(body $func, $ref_func, &$rhs$(<$gen>)?);
        }
        impl<D: Digit> $($trait)::*<$rhs$(<$gen>)?> for &BigInt<D> {
            implBigMath!(body $func, $ref_func, $rhs$(<$gen>)?);
        }
        impl<D: Digit> $($trait)::*<&$rhs$(<$gen>)?> for &BigInt<D> {
            implBigMath!(body $func, $ref_func, &$rhs$(<$gen>)?);
        }
        impl<D: Digit> $($assign_trait)::*<$rhs$(<$gen>)?> for BigInt<D> {
            fn $assign_func(&mut self, rhs: $rhs$(<$gen>)?) {
                BigInt::$ref_func(self, rhs).expect_mut("did give &mut, shouldn't get result");
            }
        }
        impl<D: Digit> $($assign_trait)::*<&$rhs$(<$gen>)?> for BigInt<D> {
            fn $assign_func(&mut self, rhs: &$rhs$(<$gen>)?) {
                BigInt::$ref_func(self, rhs).expect_mut("did give &mut, shouldn't get result");
            }
        }
    };
    (body $func:tt, $ref_func:ident, $rhs:ident$(<$gen:ident>)?) => {
        type Output = BigInt<D>;
        fn $func(self, rhs: $rhs$(<$gen>)?) -> Self::Output {
            BigInt::$ref_func(self, rhs).expect_owned("didn't give &mut, should get result")
        }
    };
    (body $func:tt, $ref_func:ident, &$rhs:ident$(<$gen:ident>)?) => {
        type Output = BigInt<D>;
        fn $func(self, rhs: &$rhs$(<$gen>)?) -> Self::Output {
            BigInt::$ref_func(self, rhs).expect_owned("didn't give &mut, should get result")
        }
    };
}

// no `std::ops::Not`, cause implied zeros to the left would need to be flipped
impl<D: Digit> Neg for BigInt<D> {
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
implBigMath!(MulAssign, mul_assign, Mul, mul, mul_by_digit, D);
implBigMath!(MulAssign, mul_assign, Mul, mul);
implBigMath!(DivAssign, div_assign, Div, div, div_euclid, BigInt<D>);
implBigMath!(RemAssign, rem_assign, Rem, rem, rem_euclid, BigInt<D>);

implBigMath!(MulAssign, mul_assign, Mul, mul, mul_by_sign, SigNum);

#[cfg(test)]
mod tests;
