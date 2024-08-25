use crate::boo::{Boo, Moo};

use crate::big_int::{
    digits::{Convert, Decomposable, Digit, Signed},
    unsigned::BigInt as BigUInt,
};

use itertools::{Either, Itertools};
use std::{
    fmt::{Debug, Write},
    ops::{
        Add, AddAssign, Deref, DerefMut, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub,
        SubAssign,
    },
    str::FromStr,
};

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

#[derive(Clone, Default)]
pub struct BigInt<D> {
    /// the sign of the number or zero <=> `digits.is_empty()`
    pub(super) signum: SigNum,
    /// holds the digits in LE order
    pub(super) unsigned: BigUInt<D>,
}

impl<D: Digit> std::fmt::Debug for BigInt<D> {
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
impl<D: Digit> std::fmt::Display for BigInt<D> {
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
impl<D: Digit> std::fmt::LowerHex for BigInt<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.pad_integral(
            !self.is_negative(),
            if f.alternate() { "0x" } else { "" },
            &format!("{:x}", self.unsigned),
        )
    }
}
impl<D: Digit> std::fmt::UpperHex for BigInt<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.pad_integral(
            !self.is_negative(),
            if f.alternate() { "0X" } else { "" },
            &format!("{:X}", self.unsigned),
        )
    }
}

impl<D: Digit> Eq for BigInt<D> {}
impl<D: Digit> Ord for BigInt<D> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl<D: Digit, M: Decomposable<D> + Signed> PartialEq<M> for BigInt<D> {
    fn eq(&self, other: &M) -> bool {
        self.partial_cmp(other)
            .is_some_and(std::cmp::Ordering::is_eq)
    }
}
impl<D: Digit, M: Decomposable<D> + Signed> PartialOrd<M> for BigInt<D> {
    fn partial_cmp(&self, other: &M) -> Option<std::cmp::Ordering> {
        Some(
            self.signum
                .cmp(&other.signum())
                .then_with(|| self.unsigned.partial_cmp(other).unwrap()),
        )
    }
}

impl<POSITIVE: super::primitve::UNum, D: Digit> FromIterator<POSITIVE> for BigInt<D> {
    /// the iter should contain the digits in little endian order
    fn from_iter<T: IntoIterator<Item = POSITIVE>>(iter: T) -> Self {
        BigUInt::from_iter(iter).with_sign(Sign::Positive)
    }
}
impl<PRIMITIVE: super::primitve::Primitive, D: Digit> From<PRIMITIVE> for BigInt<D> {
    fn from(value: PRIMITIVE) -> Self {
        match value.select_sign() {
            Either::Left(pos) => BigUInt::from(pos).with_sign(Sign::Positive),
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

impl<D: Digit> FromStr for BigInt<D> {
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

impl<D: Digit> Convert<usize> for BigInt<D> {
    fn try_into(&self) -> Option<usize> {
        if self.signum().is_negative() {
            return None;
        }
        <BigUInt<D> as Convert<usize>>::try_into(&self.unsigned)
    }
}
impl<D: Digit> Signed for BigInt<D> {
    fn signum(&self) -> SigNum {
        self.signum
    }
}
impl<D: Digit> Decomposable<D> for BigInt<D> {
    fn le_digits(&self) -> impl ExactSizeIterator<Item = D> + DoubleEndedIterator + '_ {
        <BigUInt<D> as Decomposable<D>>::le_digits(&self.unsigned)
    }
}
impl<D: Digit> Decomposable<bool> for BigInt<D> {
    fn le_digits(&self) -> impl ExactSizeIterator<Item = bool> + DoubleEndedIterator + '_ {
        <BigUInt<D> as Decomposable<bool>>::le_digits(&self.unsigned)
    }
}
impl<D: Digit> Deref for BigInt<D> {
    type Target = BigUInt<D>;

    fn deref(&self) -> &Self::Target {
        &self.unsigned
    }
}
impl<D: Digit> DerefMut for BigInt<D> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.unsigned
    }
}

impl<D: Digit> BigInt<D> {
    pub fn from_digit(value: D) -> Self {
        BigUInt::from_digit(value).with_sign(Sign::Positive)
    }
    pub fn from_digits(iter: impl IntoIterator<Item = D>) -> Self {
        BigUInt::from_digits(iter).with_sign(Sign::Positive)
    }
    pub fn split_sign(self) -> (SigNum, BigUInt<D>) {
        (self.signum, self.unsigned)
    }

    fn recalc_sign(&mut self) {
        if self.digits.is_empty() {
            self.signum = SigNum::Zero;
        } else {
            assert!(!self.signum.is_zero(), "found {self:?} with Signnum::Zero");
        }
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
    const fn is_different_sign(&self, rhs: &Self) -> bool {
        !self.is_negative() ^ !rhs.is_negative()
    }

    pub fn abs_ord(&self, rhs: &Self) -> std::cmp::Ordering {
        self.unsigned.cmp(&rhs.unsigned)
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

    pub fn rebase<D2: Digit>(&self) -> BigInt<D2> {
        BigUInt::rebase(&self.unsigned).with_sign(self.signum.into())
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
            let num = it.with_sign(Sign::Positive);
            assert!(
                !signum.is_zero() || num.is_zero(),
                "given signum was zero, but decoded number not"
            );
            num * signum
        })
    }

    fn abs_boo(value: Boo<'_, Self>) -> Boo<'_, BigUInt<D>> {
        match value {
            Boo::Owned(value) => Boo::Owned(value.unsigned),
            Boo::Borrowed(value) => Boo::Borrowed(&value.unsigned),
            Boo::BorrowedMut(value) => Boo::BorrowedMut(&mut value.unsigned),
        }
    }
    fn abs_moo(value: Moo<'_, Self>) -> Moo<'_, BigUInt<D>> {
        match value {
            Moo::Owned(value) => Moo::Owned(value.unsigned),
            Moo::BorrowedMut(value) => Moo::BorrowedMut(&mut value.unsigned),
        }
    }
    pub(super) fn refer_to_abs<'b, 'b1: 'b, 'b2: 'b, B1, B2>(
        lhs: B1,
        rhs: B2,
        func: impl for<'u> FnOnce(Boo<'u, BigUInt<D>>, Boo<'u, BigUInt<D>>) -> Moo<'u, BigUInt<D>>,
        new_sign: SigNum,
    ) -> Moo<'b, Self>
    where
        B1: Into<Boo<'b1, Self>>,
        B2: Into<Boo<'b2, Self>>,
    {
        let lhs: Boo<'_, Self> = lhs.into();
        let rhs: Boo<'_, Self> = rhs.into();

        match (lhs, rhs) {
            (Boo::BorrowedMut(borrow_mut), borrow) => {
                let _ = func(
                    Boo::BorrowedMut(&mut borrow_mut.unsigned),
                    Self::abs_boo(borrow),
                );
                borrow_mut.signum = new_sign;
                borrow_mut.recalc_sign();
                Moo::BorrowedMut(borrow_mut)
            }
            (borrow, Boo::BorrowedMut(borrow_mut)) => {
                let _ = func(
                    Self::abs_boo(borrow),
                    Boo::BorrowedMut(&mut borrow_mut.unsigned),
                );
                borrow_mut.signum = new_sign;
                borrow_mut.recalc_sign();
                Moo::BorrowedMut(borrow_mut)
            }
            (lhs, rhs) => {
                let owned =
                    func(Self::abs_boo(lhs), Self::abs_boo(rhs)).expect_owned("no mut ref given");
                Moo::Owned(owned.with_sign(new_sign.into()))
            }
        }
    }

    // fn shl<'b, 'b1: 'b, 'b2: 'b, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b, Self>
    // where
    //     B1: Into<Boo<'b1, Self>>,
    //     B2: Into<Boo<'b2, usize>>,
    // {
    //     match lhs.into() {
    //         Boo::BorrowedMut(lhs) => {
    //             let _ = BigUInt::shl(&mut lhs.unsigned, rhs);
    //             Moo::BorrowedMut(lhs)
    //         }
    //         lhs => {
    //             let sign = lhs.signum();
    //             Moo::Owned(
    //                 BigUInt::shl(Self::abs_boo(lhs), rhs)
    //                     .expect_owned("no mut ref")
    //                     .with_sign(sign.into()),
    //             )
    //         }
    //     }
    // }
    // fn shr<'b, 'b1: 'b, 'b2: 'b, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b, Self>
    // where
    //     B1: Into<Boo<'b1, Self>>,
    //     B2: Into<Boo<'b2, usize>>,
    // {
    //     match lhs.into() {
    //         Boo::BorrowedMut(lhs) => {
    //             let _ = BigUInt::shr(&mut lhs.unsigned, rhs);
    //             Moo::BorrowedMut(lhs)
    //         }
    //         lhs => {
    //             let sign = lhs.signum();
    //             Moo::Owned(
    //                 BigUInt::shr(Self::abs_boo(lhs), rhs)
    //                     .expect_owned("no mut ref")
    //                     .with_sign(sign.into()),
    //             )
    //         }
    //     }
    // }

    pub(crate) fn add<'b, 'b1: 'b, 'b2: 'b, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b, Self>
    where
        B1: Into<Boo<'b1, Self>>,
        B2: Into<Boo<'b2, Self>>,
    {
        let lhs: Boo<'_, Self> = lhs.into();
        let rhs: Boo<'_, Self> = rhs.into();
        BigUInt::<D>::assert_pair_valid(&lhs, &rhs);

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
        let new_sign = lhs.signum();
        Self::refer_to_abs(lhs, rhs, |a, b| BigUInt::add(a, b), new_sign)
    }
    pub(crate) fn sub<'b, 'b1: 'b, 'b2: 'b, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b, Self>
    where
        B1: Into<Boo<'b1, Self>>,
        B2: Into<Boo<'b2, Self>>,
    {
        let lhs: Boo<'_, Self> = lhs.into();
        let rhs: Boo<'_, Self> = rhs.into();

        if lhs.is_zero() {
            let mut out = super::math_shortcuts::get_lhs(rhs, lhs);
            out.negate();
            return out;
        }

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

        let sign = lhs.signum();
        let mut either = Self::refer_to_abs(lhs, rhs, |a, b| BigUInt::sub(a, b), sign);
        *either *= signum;
        either
    }

    pub(crate) fn mul_by_digit<'b, 'b1: 'b, 'b2: 'b, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b, Self>
    where
        B1: Into<Boo<'b1, Self>>,
        B2: Into<Boo<'b2, D>>,
    {
        match lhs.into() {
            Boo::BorrowedMut(lhs) => {
                let _ = BigUInt::mul_by_digit(&mut lhs.unsigned, rhs);
                lhs.recalc_sign();
                Moo::BorrowedMut(lhs)
            }
            lhs => {
                let sign = lhs.signum();
                Moo::Owned(
                    BigUInt::mul_by_digit(Self::abs_boo(lhs), rhs)
                        .expect_owned("no mut ref")
                        .with_sign(sign.into()),
                )
            }
        }
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
    {
        let lhs: Boo<'b, Self> = lhs.into();
        let rhs: Boo<'b, Self> = rhs.into();

        let new_sign = lhs.signum() * rhs.signum();
        Self::refer_to_abs(lhs, rhs, |a, b| BigUInt::mul(a, b), new_sign)
    }
    pub fn pow<'b, 'b1: 'b, 'b2: 'b, B1, B2, P>(lhs: B1, pow: B2) -> Moo<'b, Self>
    where
        B1: Into<Boo<'b1, Self>>,
        P: Decomposable<bool> + 'b2 + Signed + Clone,
        B2: Into<Boo<'b2, P>>,
    {
        let pow = pow.into();
        let lhs = lhs.into();
        let sign = if lhs.is_negative() && pow.le_digits().next().is_some_and(|it| it) {
            Sign::Negative
        } else {
            Sign::Positive
        };
        match lhs {
            Boo::BorrowedMut(lhs) => {
                let _ = BigUInt::pow::<'_, '_, '_, _, _, P>(&mut lhs.unsigned, pow);
                lhs.signum = sign.into();
                lhs.recalc_sign();
                Moo::BorrowedMut(lhs)
            }
            lhs => Moo::Owned(
                BigUInt::pow::<'_, '_, '_, _, _, P>(Self::abs_boo(lhs), pow)
                    .expect_owned("no mut ref")
                    .with_sign(sign),
            ),
        }
    }

    pub(crate) fn div_euclid<'b1, 'b2: 'b1, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b1, Self>
    where
        B1: Into<Boo<'b1, Self>>,
        B2: Into<Boo<'b2, Self>>,
    {
        let lhs: Boo<'_, Self> = lhs.into();
        let rhs: Boo<'_, Self> = rhs.into();

        match (lhs, rhs) {
            (lhs, Boo::BorrowedMut(rhs)) => {
                let (result, _) = Self::div_mod_euclid(lhs, std::mem::take(rhs));
                Moo::from_with_value(rhs, result.expect_owned("did'nt hat mut ref"))
            }
            (lhs, rhs) => Self::div_mod_euclid(lhs, rhs).0,
        }
    }
    #[allow(dead_code)]
    pub(crate) fn rem_euclid<'b2, 'b1: 'b2, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b2, BigUInt<D>>
    where
        B1: Into<Boo<'b1, Self>>,
        B2: Into<Boo<'b2, Self>>,
    {
        let lhs: Boo<'_, Self> = lhs.into();
        let rhs: Boo<'_, Self> = rhs.into();

        match (lhs, rhs) {
            (Boo::BorrowedMut(lhs), rhs) => {
                let (_, result) = Self::div_mod_euclid(std::mem::take(lhs), rhs);
                Moo::from_with_value(&mut lhs.unsigned, result.expect_owned("did'nt hat mut ref"))
            }
            (lhs, rhs) => Self::div_mod_euclid(lhs, rhs).1,
        }
    }
    pub fn div_mod_euclid<'b, 'b1: 'b, 'b2: 'b, B1, B2>(
        lhs: B1,
        rhs: B2,
    ) -> (Moo<'b1, Self>, Moo<'b2, BigUInt<D>>)
    where
        B1: Into<Boo<'b1, Self>>,
        B2: Into<Boo<'b2, Self>>,
    {
        let lhs: Boo<'_, Self> = lhs.into();
        let rhs: Boo<'_, Self> = rhs.into();

        #[cfg(debug_assertions)]
        let (n, d) = ((*lhs).clone(), (*rhs).clone());

        let map_r = lhs.is_negative().then(|| rhs.abs_clone());
        let signum_q = lhs.signum() * rhs.signum();

        let (mut q, mut r) = match (lhs, rhs) {
            (Boo::BorrowedMut(lhs), Boo::BorrowedMut(rhs)) => {
                let (_, _) = BigUInt::div_mod_euclid(&mut lhs.unsigned, &mut rhs.unsigned);
                (Moo::BorrowedMut(lhs), Moo::BorrowedMut(rhs))
            }
            (Boo::BorrowedMut(lhs), rhs) => {
                let (_, r) = BigUInt::div_mod_euclid(
                    Boo::BorrowedMut(&mut lhs.unsigned),
                    Self::abs_boo(rhs),
                );
                (
                    Moo::BorrowedMut(lhs),
                    Moo::Owned(r.expect_owned("").with_sign(Sign::Positive)),
                )
            }
            (lhs, Boo::BorrowedMut(rhs)) => {
                let (q, _) = BigUInt::div_mod_euclid(
                    Self::abs_boo(lhs),
                    Boo::BorrowedMut(&mut rhs.unsigned),
                );
                (
                    Moo::Owned(q.expect_owned("").with_sign(Sign::Positive)),
                    Moo::BorrowedMut(rhs),
                )
            }
            (lhs, rhs) => {
                let (q, r) = BigUInt::div_mod_euclid(Self::abs_boo(lhs), Self::abs_boo(rhs));
                (
                    Moo::Owned(q.expect_owned("").with_sign(Sign::Positive)),
                    Moo::Owned(r.expect_owned("").with_sign(Sign::Positive)),
                )
            }
        };
        q.recalc_sign();
        r.recalc_sign();
        *q *= signum_q;

        if let Some(d) = map_r.filter(|_| !r.is_zero()) {
            *q += BigUInt::from(1u8).with_sign(signum_q.into());
            *r = d - &*r;
        }

        debug_assert!(
            !r.is_negative() && r.abs_ord(&d).is_lt(),
            "0 <= r < |d| failed for \nr: {}, d: {d}",
            *r
        );
        debug_assert_eq!(
            n,
            &d * &*q + &*r,
            "n = dq + r failed for \nn: {n}, d: {d}\nq: {}, r: {}",
            *q,
            *r
        );
        (q, Self::abs_moo(r))
    }
}

macro_rules! implBigMath {
    ($($assign_trait:tt)::*, $assign_func:ident, $($trait:tt)::*, $func:ident) => {
        implBigMath!($($assign_trait)::*, $assign_func, $($trait)::*, $func, $func, BigInt<D>);
    };
    ($($assign_trait:tt)::*, $assign_func:ident, $($trait:tt)::*, $func:ident, $ref_func:ident, $rhs:ident$(<$gen:ident>)?) => {
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
// implBigMath!(BitOrAssign, bitor_assign, BitOr, bitor);
// implBigMath!(BitXorAssign, bitxor_assign, BitXor, bitxor);
// implBigMath!(BitAndAssign, bitand_assign, BitAnd, bitand);
// implBigMath!(ShlAssign, shl_assign, Shl, shl, shl, usize);
// implBigMath!(ShrAssign, shr_assign, Shr, shr, shr, usize);
implBigMath!(SubAssign, sub_assign, Sub, sub);
implBigMath!(AddAssign, add_assign, Add, add);
implBigMath!(MulAssign, mul_assign, Mul, mul, mul_by_digit, D);
implBigMath!(MulAssign, mul_assign, Mul, mul);
implBigMath!(DivAssign, div_assign, Div, div, div_euclid, BigInt<D>);

implBigMath!(MulAssign, mul_assign, Mul, mul, mul_by_sign, SigNum);

// manual impl of rem because of the always positive output
// implBigMath!(RemAssign, rem_assign, Rem, rem, rem_euclid, BigInt<D>);
impl<D: Digit> Rem for BigInt<D> {
    type Output = BigUInt<D>;
    fn rem(self, rhs: Self) -> Self::Output {
        Self::rem_euclid(self, rhs).expect_owned("didn't give &mut, should get result")
    }
}
impl<D: Digit> Rem<&Self> for BigInt<D> {
    type Output = BigUInt<D>;
    fn rem(self, rhs: &Self) -> Self::Output {
        Self::rem_euclid(self, rhs).expect_owned("didn't give &mut, should get result")
    }
}
impl<D: Digit> Rem<BigInt<D>> for &BigInt<D> {
    type Output = BigUInt<D>;
    fn rem(self, rhs: BigInt<D>) -> Self::Output {
        BigInt::rem_euclid(self, rhs).expect_owned("didn't give &mut, should get result")
    }
}
impl<D: Digit> Rem for &BigInt<D> {
    type Output = BigUInt<D>;
    fn rem(self, rhs: &BigInt<D>) -> Self::Output {
        BigInt::rem_euclid(self, rhs).expect_owned("didn't give &mut, should get result")
    }
}
impl<D: Digit> RemAssign<Self> for BigInt<D> {
    fn rem_assign(&mut self, rhs: Self) {
        Self::rem_euclid(self, rhs).expect_mut("did give &mut, shouldn't get result");
    }
}
impl<D: Digit> RemAssign<&Self> for BigInt<D> {
    fn rem_assign(&mut self, rhs: &Self) {
        Self::rem_euclid(self, rhs).expect_mut("did give &mut, shouldn't get result");
    }
}
