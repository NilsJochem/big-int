// SPDX-FileCopyrightText: 2024 Nils Jochem
// SPDX-License-Identifier: MPL-2.0
use crate::{
    big_int::digits::{Convert, Decomposable, Digit, Signed},
    ops::{DivMod, Pow, PowAssign},
    util::boo::{Mob, Moo},
    BigUInt,
};

use common::boo::Boo;
use itertools::Either;
use std::{
    fmt::{Debug, Write},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign},
    str::FromStr,
};

use super::{AnyBigInt, AnyBigIntRef, BigInt, BigIntRef};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(i8)]
pub enum Sign {
    Negative = -1,
    Positive = 1,
}
impl From<Sign> for SigNum {
    fn from(value: Sign) -> Self {
        // SAFETY: will allways be either -1 or 1
        unsafe { Self::from_i8(value as i8) }
    }
}
impl From<SigNum> for Sign {
    fn from(value: SigNum) -> Self {
        match value {
            SigNum::Negative => Self::Negative,
            SigNum::Zero | SigNum::Positive => Self::Positive,
        }
    }
}
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
    pub(super) const unsafe fn from_i8(value: i8) -> Self {
        #[allow(clippy::undocumented_unsafe_blocks)]
        unsafe {
            std::mem::transmute::<i8, Self>(value)
        }
    }
    pub const fn from_uint(is_zero: bool) -> Self {
        // SAFETY: either 0 or 1
        unsafe { Self::from_i8(!is_zero as i8) }
    }
    pub const fn is_negative(self) -> bool {
        self.into_i8().is_negative()
    }
    pub const fn is_positive(self) -> bool {
        self.into_i8().is_positive()
    }
    pub const fn is_zero(self) -> bool {
        self.into_i8() == 0
    }
    #[must_use]
    pub const fn negate(self) -> Self {
        self.const_mul(Self::Negative)
    }
    #[must_use]
    pub const fn abs(self) -> Self {
        // SAFETY: can only be 0 or 1
        unsafe { Self::from_i8(self.into_i8().abs()) }
    }
    #[must_use]
    pub const fn const_mul(self, rhs: Self) -> Self {
        // SAFETY: can only be -1,0 or 1
        unsafe { Self::from_i8(self.into_i8() * rhs.into_i8()) }
    }
    pub const fn is_different(self, other: Self) -> bool {
        !self.is_negative() ^ !other.is_negative()
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

#[derive(Debug, derive_more::From)]
pub enum MaybeSignedBoo<'b, D: Digit> {
    BorrowedMut(&'b mut BigIInt<D>),
    Borrowed(BigIntRef<'b, D>),
    Owned(BigInt<D>),
}
impl<'b, D: Digit> From<Moo<'b, BigIInt<D>>> for MaybeSignedBoo<'b, D> {
    fn from(value: Moo<'b, BigIInt<D>>) -> Self {
        match value {
            Moo::Owned(it) => Self::Owned(it.into()),
            Moo::BorrowedMut(it) => Self::BorrowedMut(it),
        }
    }
}
impl<'b, D: Digit> From<Mob<'b, BigIInt<D>>> for MaybeSignedBoo<'b, D> {
    fn from(value: Mob<'b, BigIInt<D>>) -> Self {
        match value {
            Mob::Borrowed(it) => Self::Borrowed(it.into()),
            Mob::Owned(it) => Self::Owned(it.into()),
            Mob::BorrowedMut(it) => Self::BorrowedMut(it),
        }
    }
}
impl<'b, D: Digit> TryFrom<Moo<'b, BigUInt<D>>> for MaybeSignedBoo<'b, D> {
    type Error = &'b mut BigUInt<D>;
    fn try_from(value: Moo<'b, BigUInt<D>>) -> Result<Self, Self::Error> {
        match value {
            Moo::Owned(it) => Ok(Self::Owned(it.into())),
            Moo::BorrowedMut(it) => Err(it),
        }
    }
}
impl<'b, D: Digit> TryFrom<Mob<'b, BigUInt<D>>> for MaybeSignedBoo<'b, D> {
    type Error = &'b mut BigUInt<D>;
    fn try_from(value: Mob<'b, BigUInt<D>>) -> Result<Self, Self::Error> {
        match value {
            Mob::Borrowed(it) => Ok(Self::Borrowed(it.into())),
            Mob::Owned(it) => Ok(Self::Owned(it.into())),
            Mob::BorrowedMut(it) => Err(it),
        }
    }
}
impl<'b, D: Digit> From<Boo<'b, BigIInt<D>>> for MaybeSignedBoo<'b, D> {
    fn from(value: Boo<'b, BigIInt<D>>) -> Self {
        match value {
            Boo::Borrowed(it) => Self::Borrowed(it.into()),
            Boo::Owned(it) => Self::Owned(it.into()),
        }
    }
}
impl<'b, D: Digit> From<Boo<'b, BigUInt<D>>> for MaybeSignedBoo<'b, D> {
    fn from(value: Boo<'b, BigUInt<D>>) -> Self {
        match value {
            Boo::Borrowed(it) => Self::Borrowed(it.into()),
            Boo::Owned(it) => Self::Owned(it.into()),
        }
    }
}

impl<'b, D: Digit> From<&'b BigIInt<D>> for MaybeSignedBoo<'b, D> {
    fn from(value: &'b BigIInt<D>) -> Self {
        Self::Borrowed(value.into())
    }
}
impl<D: Digit> From<BigIInt<D>> for MaybeSignedBoo<'_, D> {
    fn from(value: BigIInt<D>) -> Self {
        Self::Owned(value.into())
    }
}
impl<'b, D: Digit> From<&'b BigUInt<D>> for MaybeSignedBoo<'b, D> {
    fn from(value: &'b BigUInt<D>) -> Self {
        Self::Borrowed(value.into())
    }
}
impl<D: Digit> From<BigUInt<D>> for MaybeSignedBoo<'_, D> {
    fn from(value: BigUInt<D>) -> Self {
        Self::Owned(value.into())
    }
}
impl<'b, D: Digit> From<MaybeSignedBoo<'b, D>> for Moo<'b, BigIInt<D>> {
    fn from(value: MaybeSignedBoo<'b, D>) -> Self {
        match value {
            MaybeSignedBoo::BorrowedMut(it) => Self::BorrowedMut(it),
            MaybeSignedBoo::Borrowed(it) => Self::Owned(it.into_owned().into_signed()),
            MaybeSignedBoo::Owned(it) => Self::Owned(it.into_signed()),
        }
    }
}
impl<'b, D: Digit> From<MaybeSignedBoo<'b, D>> for Mob<'b, BigUInt<D>> {
    fn from(value: MaybeSignedBoo<'b, D>) -> Self {
        match value {
            MaybeSignedBoo::BorrowedMut(it) => Self::BorrowedMut(it.abs_mut()),
            MaybeSignedBoo::Borrowed(BigIntRef::Signed(it)) => Self::Borrowed(it.abs()),
            MaybeSignedBoo::Borrowed(BigIntRef::Unsigned(it)) => Self::Borrowed(it),
            MaybeSignedBoo::Owned(it) => Self::Owned(it.into_abs()),
        }
    }
}
impl<'b, D: Digit> MaybeSignedBoo<'b, D> {
    fn get_lhs(self, rhs: Self) -> Moo<'b, BigIInt<D>> {
        // TODO rename
        match (self, rhs) {
            (lhs, MaybeSignedBoo::BorrowedMut(rhs)) => {
                *rhs = Moo::<BigIInt<D>>::from(lhs).cloned();
                Moo::BorrowedMut(rhs)
            }
            (lhs, _) => Moo::from(lhs),
        }
    }
}
mod trait_impl_block {
    #[allow(clippy::wildcard_imports)]
    use super::*;
    use crate::big_int::{Radix, RefEnum};
    RefEnum!(
        MaybeSignedBoo<'_, D>,
        BigInt<D1>,
        BigIInt<D>,
        BorrowedMut,
        Borrowed,
        Owned
    );
    impl<'a, D: Digit> From<&'a MaybeSignedBoo<'_, D>> for BigIntRef<'a, D> {
        fn from(value: &'a MaybeSignedBoo<'_, D>) -> Self {
            match value {
                MaybeSignedBoo::BorrowedMut(it) => BigIntRef::Signed(it),
                MaybeSignedBoo::Borrowed(it) => *it,
                MaybeSignedBoo::Owned(it) => it.into(),
            }
        }
    }

    impl<D: Digit> Convert<usize> for MaybeSignedBoo<'_, D> {
        fn try_into(&self) -> Option<usize> {
            match self {
                Self::BorrowedMut(it) => Convert::<usize>::try_into(*it),
                Self::Borrowed(it) => Convert::<usize>::try_into(it),
                Self::Owned(it) => Convert::<usize>::try_into(it),
            }
        }
    }
    impl<D: Digit> MaybeSignedBoo<'_, D> {
        fn inner_into_owned(self) -> <Self as AnyBigIntRef<D>>::Owned {
            match self {
                MaybeSignedBoo::BorrowedMut(it) => it.clone(),
                MaybeSignedBoo::Borrowed(it) => it.into_owned().into_signed(),
                MaybeSignedBoo::Owned(it) => it.into_signed(),
            }
        }
        fn inner_cloned(&self) -> <Self as AnyBigIntRef<D>>::Owned {
            match self {
                MaybeSignedBoo::BorrowedMut(it) => (*it).clone(),
                MaybeSignedBoo::Borrowed(it) => it.into_owned().into_signed(),
                MaybeSignedBoo::Owned(it) => it.clone().into_signed(),
            }
        }
    }
}

impl<D: Digit> From<BigIInt<D>> for BigUInt<D> {
    fn from(value: BigIInt<D>) -> Self {
        value.unsigned
    }
}
impl<'u, 's: 'u, D: Digit> From<&'s BigIInt<D>> for &'u BigUInt<D> {
    fn from(value: &'s BigIInt<D>) -> Self {
        value.abs()
    }
}
impl<D: Digit> AsRef<BigUInt<D>> for BigIInt<D> {
    fn as_ref(&self) -> &BigUInt<D> {
        self.abs()
    }
}

#[derive(Clone, Default, Hash)]
pub struct BigIInt<D> {
    /// the sign of the number or zero <=> `digits.is_empty()`
    signum: SigNum,
    /// holds the digits in LE order
    unsigned: BigUInt<D>,
}

impl<D: Digit> std::fmt::Debug for BigIInt<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Number {{ {} ",
            match self.signum {
                SigNum::Negative => "-",
                SigNum::Zero => "",
                SigNum::Positive => "+",
            }
        )?;
        self.unsigned.inner_debug(f)?;
        write!(f, "}}")
    }
}
impl<D: Digit> std::fmt::Display for BigIInt<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let has_sign = self.is_negative() || f.sign_plus();
        assert!(!has_sign || f.width().is_none() || f.fill() != ' ', "todo");
        if self.is_negative() {
            f.write_char('-')?;
        } else if f.sign_plus() {
            f.write_char('+')?;
        }
        self.unsigned.clone().write_with_radix(
            f,
            10,
            f.alternate().then_some((3, '_')),
            f.width().map(|w| {
                (
                    w - has_sign as usize,
                    f.align().unwrap_or(std::fmt::Alignment::Right),
                    f.fill(),
                )
            }),
        )
    }
}
impl<D: Digit> std::fmt::LowerHex for BigIInt<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.pad_integral(
            !self.is_negative(),
            if f.alternate() { "0x" } else { "" },
            &format!("{:x}", self.unsigned),
        )
    }
}
impl<D: Digit> std::fmt::UpperHex for BigIInt<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.pad_integral(
            !self.is_negative(),
            if f.alternate() { "0X" } else { "" },
            &format!("{:X}", self.unsigned),
        )
    }
}

impl<D: Digit> Eq for BigIInt<D> {}
impl<D: Digit> Ord for BigIInt<D> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl<D: Digit, M: Decomposable<D> + Signed> PartialEq<M> for BigIInt<D> {
    fn eq(&self, other: &M) -> bool {
        self.partial_cmp(other)
            .is_some_and(std::cmp::Ordering::is_eq)
    }
}
impl<D: Digit, M: Decomposable<D> + Signed> PartialOrd<M> for BigIInt<D> {
    fn partial_cmp(&self, other: &M) -> Option<std::cmp::Ordering> {
        Some(
            self.signum
                .cmp(&other.signum())
                .then_with(|| self.unsigned.partial_cmp(other).unwrap()),
        )
    }
}

impl<POSITIVE: super::primitve::UNum, D: Digit> FromIterator<POSITIVE> for BigIInt<D> {
    /// the iter should contain the digits in little endian order
    fn from_iter<T: IntoIterator<Item = POSITIVE>>(iter: T) -> Self {
        BigUInt::from_iter(iter).into()
    }
}
impl<PRIMITIVE: super::primitve::Primitive, D: Digit> From<PRIMITIVE> for BigIInt<D> {
    fn from(value: PRIMITIVE) -> Self {
        match value.select_sign() {
            Either::Left(pos) => BigUInt::from(pos).into(),
            Either::Right(neg) => BigUInt::from(super::primitve::INum::abs(neg)).with_sign(
                if super::primitve::INum::is_negative(neg) {
                    Sign::Negative
                } else {
                    Sign::Positive
                },
            ),
        }
    }
}

impl<D: Digit> From<BigUInt<D>> for BigIInt<D> {
    fn from(value: BigUInt<D>) -> Self {
        value.with_sign(Sign::Positive)
    }
}
impl<D: Digit> FromStr for BigIInt<D> {
    type Err = super::unsigned::FromStrErr;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (signum, rest) = strip_sign(s);
        let signum = signum.unwrap_or(Sign::Positive);

        rest.parse::<BigUInt<D>>().map(|it| it.with_sign(signum))
    }
}

fn strip_sign(s: &str) -> (Option<Sign>, &str) {
    match s.chars().next() {
        Some('-') => (Some(Sign::Negative), &s[1..]),
        Some('+') => (Some(Sign::Positive), &s[1..]),
        None | Some(_) => (None, s),
    }
}

impl<D: Digit> Convert<usize> for BigIInt<D> {
    fn try_into(&self) -> Option<usize> {
        if self.signum().is_negative() {
            return None;
        }
        <BigUInt<D> as Convert<usize>>::try_into(&self.unsigned)
    }
}
impl<D: Digit> Signed for BigIInt<D> {
    fn signum(&self) -> SigNum {
        self.signum
    }
}
impl<D: Digit> Decomposable<D> for BigIInt<D> {
    fn le_digits(&self) -> impl ExactSizeIterator<Item = D> + DoubleEndedIterator + '_ {
        <BigUInt<D> as Decomposable<D>>::le_digits(&self.unsigned)
    }
}
impl<D: Digit> Decomposable<bool> for BigIInt<D> {
    fn le_digits(&self) -> impl ExactSizeIterator<Item = bool> + DoubleEndedIterator + '_ {
        <BigUInt<D> as Decomposable<bool>>::le_digits(&self.unsigned)
    }
}

impl<D: Digit> BigIInt<D> {
    pub const ZERO: Self = Self {
        signum: SigNum::Zero,
        unsigned: BigUInt::ZERO,
    };
    pub const ONE: Self = Self {
        signum: SigNum::Positive,
        unsigned: BigUInt::ONE,
    };
    pub const NEG_ONE: Self = Self {
        signum: SigNum::Negative,
        unsigned: BigUInt::ONE,
    };
    pub fn new(sign: impl Into<Sign>, unsigned: impl Into<BigUInt<D>>) -> Self {
        let unsigned = unsigned.into();
        Self {
            signum: Self::get_new_signum(&unsigned, || sign.into()),
            unsigned,
        }
    }
    pub fn from_digit(value: D) -> Self {
        BigUInt::from_digit(value).into()
    }
    pub fn from_digits(iter: impl IntoIterator<Item = D>) -> Self {
        BigUInt::from_digits(iter).into()
    }
    pub fn split_sign(self) -> (SigNum, BigUInt<D>) {
        (self.signum, self.unsigned)
    }

    /// generate a new random number with at least `bytes.start()` and at most `bytes.end()` bytes of information
    /// # Example
    /// `0x00_0100` <= `BigInt::new_random(2..=3, _)` <= `0xff_ffff`,
    pub fn new_random(bytes: std::ops::RangeInclusive<usize>, mut rng: impl rand::RngCore) -> Self {
        let sign = if rng.next_u32() % 2 == 0 {
            Sign::Positive
        } else {
            Sign::Negative
        };
        BigUInt::new_random(bytes, rng).with_sign(sign)
    }

    fn recalc_sign(&mut self) {
        if self.abs().is_zero() {
            self.signum = SigNum::Zero;
        } else {
            assert!(!self.signum.is_zero(), "found {self:?} with Signnum::Zero");
        }
    }
    fn get_new_signum(unsigned: &BigUInt<D>, sign: impl FnOnce() -> Sign) -> SigNum {
        if unsigned.is_zero() {
            SigNum::Zero
        } else {
            sign().into()
        }
    }

    pub const fn abs(&self) -> &BigUInt<D> {
        &self.unsigned
    }
    pub(crate) fn abs_mut(&mut self) -> &mut BigUInt<D> {
        &mut self.unsigned
    }
    pub const fn signum(&self) -> SigNum {
        self.signum
    }
    pub fn set_sign(&mut self, sign: impl Into<Sign>) {
        self.signum = Self::get_new_signum(&self.unsigned, || sign.into());
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

    pub fn negate(&mut self) {
        self.signum = -self.signum;
    }
    pub fn take_sign(&mut self) -> SigNum {
        let signum = self.signum;
        self.signum = self.signum.abs();
        signum
    }

    /// needs to newly allocate on big endian systems
    /// will return the sign seperatly as this function cannot know which character isn't already used by the encoding, or otherwise not usable.
    #[cfg(feature = "base64")]
    pub fn as_base64(&self, engine: &impl base64::Engine) -> (SigNum, String) {
        (self.signum, self.unsigned.as_base64(engine))
    }
    #[cfg(feature = "base64")]
    pub fn from_base64(
        signum: SigNum,
        data: impl AsRef<[u8]>,
        engine: &impl base64::Engine,
    ) -> Result<Self, base64::DecodeError> {
        BigUInt::from_base64(data, engine).map(|it| {
            let num = Self::from(it);
            assert!(
                !signum.is_zero() || num.is_zero(),
                "given signum was zero, but decoded number not"
            );
            num * signum
        })
    }

    pub(super) fn assert_pair_valid(lhs: &MaybeSignedBoo<'_, D>, rhs: &MaybeSignedBoo<'_, D>) {
        assert!(
            !matches!(lhs, MaybeSignedBoo::BorrowedMut(_))
                || !matches!(rhs, MaybeSignedBoo::BorrowedMut(_)),
            "can't have to Borrow_mut's"
        );
    }
    pub(super) fn refer_to_abs<'b, 'b1: 'b, 'b2: 'b, B1, B2>(
        lhs: B1,
        rhs: B2,
        func: impl for<'u> FnOnce(Mob<'u, BigUInt<D>>, Mob<'u, BigUInt<D>>) -> Moo<'u, BigUInt<D>>,
        new_sign: SigNum,
    ) -> Moo<'b, Self>
    where
        B1: Into<MaybeSignedBoo<'b1, D>>,
        B2: Into<MaybeSignedBoo<'b2, D>>,
    {
        let lhs: MaybeSignedBoo<'_, D> = lhs.into();
        let rhs: MaybeSignedBoo<'_, D> = rhs.into();

        match (lhs, rhs) {
            (MaybeSignedBoo::BorrowedMut(borrow_mut), borrow) => {
                let _ = func(
                    Mob::BorrowedMut(&mut borrow_mut.unsigned),
                    Mob::from(borrow),
                );
                borrow_mut.signum = new_sign;
                borrow_mut.recalc_sign();
                Moo::BorrowedMut(borrow_mut)
            }
            (borrow, MaybeSignedBoo::BorrowedMut(borrow_mut)) => {
                let _ = func(
                    Mob::from(borrow),
                    Mob::BorrowedMut(&mut borrow_mut.unsigned),
                );
                borrow_mut.signum = new_sign;
                borrow_mut.recalc_sign();
                Moo::BorrowedMut(borrow_mut)
            }
            (lhs, rhs) => {
                let owned = func(Mob::from(lhs), Mob::from(rhs)).expect_owned("no mut ref given");
                Moo::Owned(owned.with_sign(new_sign))
            }
        }
    }

    pub fn add<'b, 'b1: 'b, 'b2: 'b, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b, Self>
    where
        B1: Into<MaybeSignedBoo<'b1, D>>,
        B2: Into<MaybeSignedBoo<'b2, D>>,
    {
        let lhs: MaybeSignedBoo<'_, D> = lhs.into();
        let rhs: MaybeSignedBoo<'_, D> = rhs.into();
        Self::assert_pair_valid(&lhs, &rhs);

        if lhs.signum().is_different(rhs.signum()) {
            return match (lhs, rhs) {
                (MaybeSignedBoo::Borrowed(lhs), rhs) => {
                    let mut either = Moo::<Self>::from(rhs);
                    either.negate();
                    Self::sub(lhs, either)
                }
                (MaybeSignedBoo::Owned(lhs), MaybeSignedBoo::Owned(rhs)) => {
                    let mut rhs = rhs.into_signed();
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
        let new_sign = lhs.signum();
        Self::refer_to_abs(lhs, rhs, |a, b| BigUInt::add(a, b), new_sign)
    }
    pub fn sub<'b, 'b1: 'b, 'b2: 'b, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b, Self>
    where
        B1: Into<MaybeSignedBoo<'b1, D>>,
        B2: Into<MaybeSignedBoo<'b2, D>>,
    {
        let lhs: MaybeSignedBoo<'_, D> = lhs.into();
        let rhs: MaybeSignedBoo<'_, D> = rhs.into();

        if lhs.is_zero() {
            let mut out = rhs.get_lhs(lhs);
            out.negate();
            return out;
        }

        if lhs.signum().is_different(rhs.signum()) {
            return match (lhs, rhs) {
                (MaybeSignedBoo::Borrowed(lhs), rhs) => {
                    let mut either = Moo::<Self>::from(rhs);
                    either.negate();
                    Self::add(lhs, either)
                }
                (MaybeSignedBoo::Owned(lhs), MaybeSignedBoo::Owned(rhs)) => {
                    let mut rhs = rhs.into_signed();
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

        let (lhs, rhs, signum) = if lhs.abs().cmp(rhs.abs()).is_lt() {
            (rhs, lhs, SigNum::Negative)
        } else {
            (lhs, rhs, SigNum::Positive)
        };

        let sign = lhs.signum();
        let mut either = Self::refer_to_abs(lhs, rhs, |a, b| BigUInt::sub(a, b), sign);
        *either *= signum;
        either
    }

    pub fn mul_by_digit<'b, 'b1: 'b, 'b2: 'b, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b, Self>
    where
        B1: Into<MaybeSignedBoo<'b1, D>>,
        B2: Into<Mob<'b2, D>>,
    {
        // let lhs: MaybeSignedBoo<'_, D> = lhs.into();
        // let rhs: MaybeSignedBoo<'_, D> = rhs.into();
        match lhs.into() {
            MaybeSignedBoo::BorrowedMut(lhs) => {
                let _ = BigUInt::mul_by_digit(&mut lhs.unsigned, rhs);
                lhs.recalc_sign();
                Moo::BorrowedMut(lhs)
            }
            lhs => {
                let sign = lhs.signum();
                Moo::Owned(
                    BigUInt::mul_by_digit(Mob::from(lhs), rhs)
                        .expect_owned("no mut ref")
                        .with_sign(sign),
                )
            }
        }
    }

    pub fn mul_by_sign<'b, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b, Self>
    where
        B1: Into<Mob<'b, Self>>,
        B2: Into<Mob<'b, SigNum>>,
    {
        let mut lhs = Moo::<Self>::from_mob_cloned(lhs.into());
        let rhs = rhs.into().copied();
        if rhs == SigNum::Zero {
            *lhs = Self::ZERO;
        } else {
            lhs.signum *= rhs;
        }
        lhs
    }
    pub fn mul<'b, 'b1: 'b, 'b2: 'b, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b, Self>
    where
        B1: Into<MaybeSignedBoo<'b1, D>>,
        B2: Into<MaybeSignedBoo<'b2, D>>,
    {
        let lhs: MaybeSignedBoo<'_, D> = lhs.into();
        let rhs: MaybeSignedBoo<'_, D> = rhs.into();

        let new_sign = lhs.signum() * rhs.signum();
        Self::refer_to_abs(lhs, rhs, |a, b| BigUInt::mul(a, b), new_sign)
    }

    pub fn div<'b1, 'b2: 'b1, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b1, Self>
    where
        B1: Into<MaybeSignedBoo<'b1, D>>,
        B2: Into<MaybeSignedBoo<'b2, D>>,
    {
        let lhs: MaybeSignedBoo<'_, D> = lhs.into();
        let rhs: MaybeSignedBoo<'_, D> = rhs.into();

        match (lhs, rhs) {
            (lhs, MaybeSignedBoo::BorrowedMut(rhs)) => {
                let (result, _) = Self::div_mod(lhs, std::mem::take(rhs));
                Moo::from_with_value(rhs, result.expect_owned("didn't hat mut ref"))
            }
            (lhs, rhs) => Self::div_mod(lhs, rhs).0,
        }
    }
    pub fn rem<'b2, 'b1: 'b2, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b2, Self>
    where
        B1: Into<MaybeSignedBoo<'b1, D>>,
        B2: Into<MaybeSignedBoo<'b2, D>>,
    {
        let lhs: MaybeSignedBoo<'_, D> = lhs.into();
        let rhs: MaybeSignedBoo<'_, D> = rhs.into();

        match (lhs, rhs) {
            (MaybeSignedBoo::BorrowedMut(lhs), rhs) => {
                let (_, result) = Self::div_mod(std::mem::take(lhs), rhs);
                *lhs = result.expect_owned("didn't hat mut ref");
                Moo::from(lhs)
            }
            (lhs, rhs) => Self::div_mod(lhs, rhs).1,
        }
    }
    pub fn div_mod<'b, 'b1: 'b, 'b2: 'b, B1, B2>(
        lhs: B1,
        rhs: B2,
    ) -> (Moo<'b1, Self>, Moo<'b2, Self>)
    where
        B1: Into<MaybeSignedBoo<'b1, D>>,
        B2: Into<MaybeSignedBoo<'b2, D>>,
    {
        let lhs: MaybeSignedBoo<'_, D> = lhs.into();
        let rhs: MaybeSignedBoo<'_, D> = rhs.into();

        #[cfg(debug_assertions)]
        let (n, d) = (lhs.cloned(), rhs.cloned());

        let signum_q = lhs.signum() * rhs.signum();
        let signum_r = lhs.signum();

        let (mut q, mut r) = match (lhs, rhs) {
            (MaybeSignedBoo::BorrowedMut(lhs), MaybeSignedBoo::BorrowedMut(rhs)) => {
                let (_, _) = BigUInt::div_mod_euclid(&mut lhs.unsigned, &mut rhs.unsigned);
                (Moo::BorrowedMut(lhs), Moo::BorrowedMut(rhs))
            }
            (MaybeSignedBoo::BorrowedMut(lhs), rhs) => {
                let (_, r) =
                    BigUInt::div_mod_euclid(Mob::BorrowedMut(&mut lhs.unsigned), Mob::from(rhs));
                (Moo::BorrowedMut(lhs), Moo::Owned(r.expect_owned("").into()))
            }
            (lhs, MaybeSignedBoo::BorrowedMut(rhs)) => {
                let (q, _) =
                    BigUInt::div_mod_euclid(Mob::from(lhs), Mob::BorrowedMut(&mut rhs.unsigned));
                (Moo::Owned(q.expect_owned("").into()), Moo::BorrowedMut(rhs))
            }
            (lhs, rhs) => {
                let (q, r) = BigUInt::div_mod_euclid(Mob::from(lhs), Mob::from(rhs));
                (
                    Moo::Owned(q.expect_owned("").into()),
                    Moo::Owned(r.expect_owned("").into()),
                )
            }
        };
        q.recalc_sign();
        r.recalc_sign();
        *q *= signum_q;
        *r *= signum_r;

        debug_assert!(
            r.abs() < d.abs(),
            "|r| < |d| failed for \nr: {}, d: {d}",
            *r
        );
        debug_assert_eq!(
            n,
            &*q * &d + &*r,
            "n = dq + r failed for \nn: {n}, d: {d}\nq: {}, r: {}",
            *q,
            *r
        );
        (q, r)
    }

    pub fn div_mod_euclid<'b, 'b1: 'b, 'b2: 'b, B1, B2>(
        lhs: B1,
        rhs: B2,
    ) -> (Moo<'b1, Self>, Moo<'b2, BigUInt<D>>)
    where
        B1: Into<MaybeSignedBoo<'b1, D>>,
        B2: Into<MaybeSignedBoo<'b2, D>>,
    {
        let lhs: MaybeSignedBoo<'_, D> = lhs.into();
        let rhs: MaybeSignedBoo<'_, D> = rhs.into();

        let map_r = lhs.signum().is_negative().then(|| rhs.abs().clone());
        let signum_q = lhs.signum() * rhs.signum();

        let (mut q, mut r) = Self::div_mod(lhs, rhs);

        if let Some(d) = map_r.filter(|_| !r.is_zero()) {
            *q += BigUInt::from(1u8).with_sign(signum_q);

            debug_assert!(r.is_negative(), "pre euclid shifted r:{} not negative", *r);
            *r = d + &*r;
            debug_assert!(r.is_positive(), "post euclid shifted r:{} not positve", *r);
        } else {
            debug_assert!(!r.is_negative(), "non euclid shifted r:{} negative", *r);
        }

        (q, Moo::from_mob_cloned(Mob::from(MaybeSignedBoo::from(r))))
    }

    pub fn pow<'b, 'b1: 'b, 'b2: 'b, B1, B2, P>(lhs: B1, pow: B2) -> Moo<'b, Self>
    where
        B1: Into<Mob<'b1, Self>>,
        P: Decomposable<bool> + 'b2 + Signed + Clone,
        B2: Into<Mob<'b2, P>>,
    {
        let pow = pow.into();
        let lhs = lhs.into();
        let sign = if lhs.signum().is_negative() && pow.le_digits().next().is_some_and(|it| it) {
            Sign::Negative
        } else {
            Sign::Positive
        };
        match lhs {
            Mob::BorrowedMut(lhs) => {
                let _ = BigUInt::pow::<'_, '_, '_, _, _, P>(&mut lhs.unsigned, pow);
                lhs.signum = sign.into();
                lhs.recalc_sign();
                Moo::BorrowedMut(lhs)
            }
            lhs => Moo::Owned(
                BigUInt::pow::<'_, '_, '_, _, _, P>(MaybeSignedBoo::from(lhs), pow)
                    .expect_owned("no mut ref")
                    .with_sign(sign),
            ),
        }
    }
}

macro_rules! implBigMath {
    ($($assign_trait:tt)::*, $assign_func:ident, $($trait:tt)::*, $func:ident) => {
        implBigMath!($($assign_trait)::*, $assign_func, $($trait)::*, $func, $func);
    };

    ($($assign_trait:tt)::*, $assign_func:ident, $($trait:tt)::*, $func:ident, $ref_func:ident) => {
        implBigMath!(assign $($assign_trait)::*, $assign_func, $($trait)::*, $func, $ref_func, BigIInt<D>, BigIInt<D>);
        implBigMath!(assign $($assign_trait)::*, $assign_func, $($trait)::*, $func, $ref_func, BigIInt<D>, BigUInt<D>);
        implBigMath!($($assign_trait)::*, $assign_func, $($trait)::*, $func, $ref_func, BigUInt<D>, BigIInt<D>);
    };
    (assign $($assign_trait:tt)::*, $assign_func:ident, $($trait:tt)::*, $func:ident, $ref_func:ident, $lhs:ident$(<$l_gen:ident>)?, $rhs:ident$(<$r_gen:ident>)?) => {
        implBigMath!($($assign_trait)::*, $assign_func, $($trait)::*, $func, $ref_func, $lhs$(<$l_gen>)?, $rhs$(<$r_gen>)?);
        impl<D: Digit> $($assign_trait)::*<$rhs$(<$r_gen>)?> for $lhs$(<$l_gen>)? {
            fn $assign_func(&mut self, rhs: $rhs$(<$r_gen>)?) {
                BigIInt::$ref_func(self, rhs).expect_mut("did give &mut, shouldn't get result");
            }
        }
        impl<D: Digit> $($assign_trait)::*<&$rhs$(<$r_gen>)?> for $lhs$(<$l_gen>)? {
            fn $assign_func(&mut self, rhs: &$rhs$(<$r_gen>)?) {
                BigIInt::$ref_func(self, rhs).expect_mut("did give &mut, shouldn't get result");
            }
        }
    };
    ($($assign_trait:tt)::*, $assign_func:ident, $($trait:tt)::*, $func:ident, $ref_func:ident, $lhs:ident$(<$l_gen:ident>)?, $rhs:ident$(<$r_gen:ident>)?) => {
        impl<D: Digit> $($trait)::*<$rhs$(<$r_gen>)?> for $lhs$(<$l_gen>)? {
            implBigMath!(body $func, $ref_func, $rhs$(<$r_gen>)?);
        }
        impl<D: Digit> $($trait)::*<&$rhs$(<$r_gen>)?> for $lhs$(<$l_gen>)? {
            implBigMath!(body $func, $ref_func, &$rhs$(<$r_gen>)?);
        }
        impl<D: Digit> $($trait)::*<$rhs$(<$r_gen>)?> for &$lhs$(<$l_gen>)? {
            implBigMath!(body $func, $ref_func, $rhs$(<$r_gen>)?);
        }
        impl<D: Digit> $($trait)::*<&$rhs$(<$r_gen>)?> for &$lhs$(<$l_gen>)? {
            implBigMath!(body $func, $ref_func, &$rhs$(<$r_gen>)?);
        }
    };
    (body $func:tt, $ref_func:ident, $rhs:ident$(<$gen:ident>)?) => {
        type Output = BigIInt<D>;
        fn $func(self, rhs: $rhs$(<$gen>)?) -> Self::Output {
            BigIInt::$ref_func(self, rhs).expect_owned("didn't give &mut, should get result")
        }
    };
    (body $func:tt, $ref_func:ident, &$rhs:ident$(<$gen:ident>)?) => {
        type Output = BigIInt<D>;
        fn $func(self, rhs: &$rhs$(<$gen>)?) -> Self::Output {
            BigIInt::$ref_func(self, rhs).expect_owned("didn't give &mut, should get result")
        }
    };
}

// no `std::ops::Not`, cause implied zeros to the left would need to be flipped
impl<D: Digit> Neg for BigIInt<D> {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.negate();
        self
    }
}
implBigMath!(
    assign
    MulAssign,
    mul_assign,
    Mul,
    mul,
    mul_by_sign,
    BigIInt<D>,
    SigNum
);
implBigMath!(
    assign
    MulAssign,
    mul_assign,
    Mul,
    mul,
    mul_by_digit,
    BigIInt<D>,
    D
);
implBigMath!(SubAssign, sub_assign, Sub, sub);
implBigMath!(AddAssign, add_assign, Add, add);
implBigMath!(MulAssign, mul_assign, Mul, mul);
implBigMath!(DivAssign, div_assign, Div, div);
implBigMath!(RemAssign, rem_assign, Rem, rem);

// manual impl of Pow as RHS is generic
impl<D: Digit, P: Decomposable<bool> + Signed + Clone> Pow<P> for BigIInt<D> {
    type Output = Self;

    fn pow(self, rhs: P) -> Self::Output {
        Self::pow(self, rhs).expect_owned("no mut ref given")
    }
}
impl<D: Digit, P: Decomposable<bool> + Signed + Clone> Pow<P> for &BigIInt<D> {
    type Output = BigIInt<D>;

    fn pow(self, rhs: P) -> Self::Output {
        BigIInt::pow(self, rhs).expect_owned("no mut ref given")
    }
}
impl<D: Digit, P: Decomposable<bool> + Signed + Clone> PowAssign<P> for BigIInt<D> {
    fn pow_assign(&mut self, rhs: P) {
        Self::pow(self, rhs).expect_mut("mut ref given");
    }
}

macro_rules! implBigDiv {
    (funcs, $rhs:ident$(<$gen:ident>)?) => {
        fn div_mod(self, rhs: $rhs$(<$gen>)?) -> (Self::Signed, Self::Signed) {
            implBigDiv!(body, div_mod, self, rhs)
        }
        fn  div_mod_euclid(self, rhs: $rhs$(<$gen>)?) -> (Self::Signed, Self::Unsigned) {
            implBigDiv!(body, div_mod_euclid, self, rhs)
        }
    };
    (funcs, &$rhs:ident$(<$gen:ident>)?) => {
        fn div_mod(self, rhs: &$rhs$(<$gen>)?) -> (Self::Signed, Self::Signed) {
            implBigDiv!(body, div_mod, self, rhs)
        }
        fn  div_mod_euclid(self, rhs: &$rhs$(<$gen>)?) -> (Self::Signed, Self::Unsigned) {
            implBigDiv!(body, div_mod_euclid, self, rhs)
        }
    };
    (body, $func: ident, $lhs: ident, $rhs:ident) => {{
        let (q, r) = BigIInt::$func($lhs, $rhs);
        (
            q.expect_owned("didn't give &mut, should get result"),
            r.expect_owned("didn't give &mut, should get result"),
        )
    }};
    ($lhs:ident$(<$l_gen:ident>)?, $rhs:ident$(<$r_gen:ident>)?) => {
        impl<D: Digit> DivMod<$rhs$(<$r_gen>)?> for $lhs$(<$l_gen>)? {
            type Signed = BigIInt<D>;
            type Unsigned = BigUInt<D>;

            implBigDiv!(funcs, $rhs$(<$r_gen>)?);
        }
        impl<D: Digit> DivMod<&$rhs$(<$r_gen>)?> for $lhs$(<$l_gen>)? {
            type Signed = BigIInt<D>;
            type Unsigned = BigUInt<D>;

            implBigDiv!(funcs, &$rhs$(<$r_gen>)?);
        }
        impl<D: Digit> DivMod<$rhs$(<$r_gen>)?> for &$lhs$(<$l_gen>)? {
            type Signed = BigIInt<D>;
            type Unsigned = BigUInt<D>;

            implBigDiv!(funcs, $rhs$(<$r_gen>)?);
        }
        impl<D: Digit> DivMod<&$rhs$(<$r_gen>)?> for &$lhs$(<$l_gen>)? {
            type Signed = BigIInt<D>;
            type Unsigned = BigUInt<D>;

            implBigDiv!(funcs, &$rhs$(<$r_gen>)?);
        }
    }
}

implBigDiv!(BigIInt<D>, BigIInt<D>);
implBigDiv!(BigUInt<D>, BigIInt<D>);
implBigDiv!(BigIInt<D>, BigUInt<D>);
implBigDiv!(BigUInt<D>, BigUInt<D>);
