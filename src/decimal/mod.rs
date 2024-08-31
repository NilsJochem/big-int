// SPDX-FileCopyrightText: 2024 Nils Jochem
// SPDX-License-Identifier: MPL-2.0
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use crate::{
    big_int::{digits::Digit, math_algos::gcd::Gcd},
    ops::DivMod,
    util::boo::{Boo, Moo},
    BigIInt, BigUInt, SigNum,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decimal<D: Digit> {
    numerator: BigIInt<D>,
    denominator: BigUInt<D>,
}

impl<D: Digit> From<BigIInt<D>> for Decimal<D> {
    fn from(value: BigIInt<D>) -> Self {
        Self::new_coprime(value, 1u8)
    }
}
impl<D: Digit> TryFrom<Decimal<D>> for BigIInt<D> {
    type Error = Decimal<D>;

    fn try_from(value: Decimal<D>) -> Result<Self, Self::Error> {
        if value.denominator.is_one() {
            Ok(value.numerator)
        } else {
            Err(value)
        }
    }
}
impl<D: Digit> Decimal<D> {
    /// assumes gcd(n, d) == 1
    fn new_coprime(numerator: impl Into<BigIInt<D>>, denominator: impl Into<BigUInt<D>>) -> Self {
        let numerator = numerator.into();
        let denominator = denominator.into();
        debug_assert!(!denominator.is_zero());
        debug_assert!(!numerator.is_zero() || denominator.is_one());
        Self {
            numerator,
            denominator,
        }
    }
    pub fn new(
        numerator: impl Into<BigIInt<D>>,
        denominator: impl Into<BigUInt<D>>,
    ) -> Option<Self> {
        let denominator = denominator.into();
        if denominator.is_zero() {
            return None;
        }
        let (_, factors) = Gcd::new(numerator, denominator).factors();
        Some(Self::new_coprime(factors.a, factors.b))
    }
    fn extend(&mut self, rhs: &BigUInt<D>) {
        *self.numerator.abs_mut() *= rhs;
        self.denominator *= rhs;
    }
    pub const fn signum(&self) -> SigNum {
        self.numerator.signum()
    }
    pub fn abs_cmp_one(&self) -> std::cmp::Ordering {
        self.numerator.abs().cmp(&self.denominator)
    }

    pub fn div_mod_euclid(self) -> (BigIInt<D>, BigUInt<D>) {
        self.numerator.div_mod_euclid(self.denominator)
    }

    pub fn round(self) -> BigIInt<D> {
        let d = self.denominator;
        let (q, r) = self.numerator.div_mod_euclid(&d);
        if r * D::from(2u8) > d {
            q + BigIInt::ONE
        } else {
            q
        }
    }
    pub fn floor(self) -> BigIInt<D> {
        let (q, _) = self.div_mod_euclid();
        q
    }
    pub fn ceil(self) -> BigIInt<D> {
        let (q, r) = self.div_mod_euclid();
        if r.is_zero() {
            q
        } else {
            q + BigIInt::ONE
        }
    }

    pub fn round_to_numerator(self, new: impl Into<BigUInt<D>>) -> Option<Self> {
        let new = new.into();
        Self::new(
            Self::new(&new * self.numerator, self.denominator)
                .unwrap()
                .round(),
            new,
        )
    }

    pub fn negate(&mut self) {
        self.numerator.negate();
    }
    pub fn recip(&mut self) {
        assert!(!self.numerator.is_zero(), "can't invert 0");
        std::mem::swap(self.numerator.abs_mut(), &mut self.denominator);
    }

    fn split(value: Boo<'_, Self>) -> (Boo<'_, BigIInt<D>>, Boo<'_, BigUInt<D>>) {
        match value {
            Boo::Owned(value) => (Boo::Owned(value.numerator), Boo::Owned(value.denominator)),
            Boo::Borrowed(value) => (
                Boo::Borrowed(&value.numerator),
                Boo::Borrowed(&value.denominator),
            ),
            Boo::BorrowedMut(_) => panic!("can't split mut"),
        }
    }
    fn add_same_denominator<'b, 'b1: 'b, 'b2: 'b, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b, Self>
    where
        B1: Into<Boo<'b1, Self>>,
        B2: Into<Boo<'b2, Self>>,
        D: 'b1 + 'b2,
    {
        let lhs: Boo<'_, Self> = lhs.into();
        let rhs: Boo<'_, Self> = rhs.into();
        assert_eq!(lhs.denominator, rhs.denominator);

        match (lhs, rhs) {
            (Boo::BorrowedMut(borrow_mut), borrow) | (borrow, Boo::BorrowedMut(borrow_mut)) => {
                let (borrow_numerator, borrow_denominator) = Self::split(borrow);
                *borrow_mut = Self::new(
                    BigIInt::add(&borrow_mut.numerator, borrow_numerator)
                        .expect_owned("not mut given"),
                    borrow_denominator.cloned(),
                )
                .unwrap();
                Moo::BorrowedMut(borrow_mut)
            }
            (lhs, rhs) => {
                let (lhs_numerator, lhs_denominator) = Self::split(lhs);
                let (rhs_numerator, _) = Self::split(rhs);
                Moo::Owned(
                    Self::new(
                        BigIInt::add(lhs_numerator, rhs_numerator).expect_owned("not mut given"),
                        lhs_denominator.cloned(),
                    )
                    .unwrap(),
                )
            }
        }
    }
    pub fn add<'b, 'b1: 'b, 'b2: 'b, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b, Self>
    where
        B1: Into<Boo<'b1, Self>>,
        B2: Into<Boo<'b2, Self>>,
        D: 'b1 + 'b2,
    {
        let mut lhs = Moo::<Self>::from(lhs.into());
        let mut rhs = Moo::<Self>::from(rhs.into());

        if lhs.denominator != rhs.denominator {
            let lhs_denominator = lhs.denominator.clone();
            lhs.get_mut().extend(&rhs.denominator);
            rhs.get_mut().extend(&lhs_denominator);
        }
        Self::add_same_denominator(lhs, rhs)
    }

    pub fn sub<'b, 'b1: 'b, 'b2: 'b, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b, Self>
    where
        B1: Into<Boo<'b1, Self>>,
        B2: Into<Boo<'b2, Self>>,
        D: 'b1 + 'b2,
    {
        let mut lhs = Moo::<Self>::from(lhs.into());
        lhs.negate();
        Self::add(lhs, rhs)
    }
    pub fn mul<'b, 'b1: 'b, 'b2: 'b, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b, Self>
    where
        B1: Into<Boo<'b1, Self>>,
        B2: Into<Boo<'b2, Self>>,
        D: 'b1 + 'b2,
    {
        let lhs: Boo<'_, Self> = lhs.into();
        let rhs: Boo<'_, Self> = rhs.into();

        match (lhs, rhs) {
            (Boo::BorrowedMut(borrow_mut), borrow) | (borrow, Boo::BorrowedMut(borrow_mut)) => {
                let (borrow_numerator, borrow_denominator) = Self::split(borrow);
                *borrow_mut = Self::new(
                    BigIInt::mul(&borrow_mut.numerator, borrow_numerator)
                        .expect_owned("not mut given"),
                    BigUInt::mul(&borrow_mut.denominator, borrow_denominator)
                        .expect_owned("not mut given"),
                )
                .unwrap();
                Moo::BorrowedMut(borrow_mut)
            }
            (lhs, rhs) => {
                let (lhs_numerator, lhs_denominator) = Self::split(lhs);
                let (rhs_numerator, rhs_denominator) = Self::split(rhs);
                Moo::Owned(
                    Self::new(
                        BigIInt::mul(lhs_numerator, rhs_numerator).expect_owned("not mut given"),
                        BigUInt::mul(lhs_denominator, rhs_denominator)
                            .expect_owned("not mut given"),
                    )
                    .unwrap(),
                )
            }
        }
    }
    pub fn div<'b, 'b1: 'b, 'b2: 'b, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b, Self>
    where
        B1: Into<Boo<'b1, Self>>,
        B2: Into<Boo<'b2, Self>>,
        D: 'b1 + 'b2,
    {
        let mut rhs = Moo::<Self>::from(rhs.into());
        rhs.recip();
        Self::mul(lhs, rhs)
    }
}

macro_rules! implDecimalMath {
    ($($assign_trait:tt)::*, $assign_func:ident, $($trait:tt)::*, $func:ident) => {
        implDecimalMath!($($assign_trait)::*, $assign_func, $($trait)::*, $func, $func, Decimal<D>);
    };
    ($($assign_trait:tt)::*, $assign_func:ident, $($trait:tt)::*, $func:ident, $ref_func:ident, $rhs: ident$(<$gen:ident>)?) => {
        impl<D: Digit> $($trait)::*<$rhs$(<$gen>)?> for Decimal<D> {
            implDecimalMath!(body $func, $ref_func, $rhs$(<$gen>)?);
        }
        impl<D: Digit> $($trait)::*<&$rhs$(<$gen>)?> for Decimal<D> {
            implDecimalMath!(body $func, $ref_func, &$rhs$(<$gen>)?);
        }
        impl<D: Digit> $($trait)::*<$rhs$(<$gen>)?> for &Decimal<D> {
            implDecimalMath!(body $func, $ref_func, $rhs$(<$gen>)?);
        }
        impl<D: Digit> $($trait)::*<&$rhs$(<$gen>)?> for &Decimal<D> {
            implDecimalMath!(body $func, $ref_func, &$rhs$(<$gen>)?);
        }
        impl<D: Digit> $($assign_trait)::*<$rhs$(<$gen>)?> for Decimal<D> {
            fn $assign_func(&mut self, rhs: $rhs$(<$gen>)?) {
                Decimal::$ref_func(self, rhs).expect_mut("did give &mut, shouldn't get result");
            }
        }
        impl<D: Digit> $($assign_trait)::*<&$rhs$(<$gen>)?> for Decimal<D> {
            fn $assign_func(&mut self, rhs: &$rhs$(<$gen>)?) {
                Decimal::$ref_func(self, rhs).expect_mut("did give &mut, shouldn't get result");
            }
        }
    };
    (body $func:tt, $ref_func:ident, $rhs:ident$(<$gen:ident>)?) => {
        type Output = Decimal<D>;
        fn $func(self, rhs: $rhs$(<$gen>)?) -> Self::Output {
            Decimal::$ref_func(self, rhs).expect_owned("didn't give &mut, should get result")
        }
    };
    (body $func:tt, $ref_func:ident, &$rhs:ident$(<$gen:ident>)?) => {
        type Output = Decimal<D>;
        fn $func(self, rhs: &$rhs$(<$gen>)?) -> Self::Output {
            Decimal::$ref_func(self, rhs).expect_owned("didn't give &mut, should get result")
        }
    };
}
implDecimalMath!(AddAssign, add_assign, Add, add);
implDecimalMath!(SubAssign, sub_assign, Sub, sub);
implDecimalMath!(DivAssign, div_assign, Div, div);
implDecimalMath!(MulAssign, mul_assign, Mul, mul);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round() {
        assert_eq!(
            Decimal::<u32>::new_coprime(17, 5).round(),
            BigIInt::from_digit(3)
        );
        assert_eq!(
            Decimal::<u32>::new_coprime(18, 5).round(),
            BigIInt::from_digit(4)
        );
    }

    #[test]
    fn round_to_numerator() {
        assert_eq!(
            Decimal::<u32>::new(29, 69).unwrap().round_to_numerator(100),
            Decimal::new(42, 100)
        );
    }

    #[test]
    fn add() {
        assert_eq!(
            Decimal::<u32>::new_coprime(1, 2) + Decimal::new_coprime(1, 4),
            Decimal::new_coprime(3, 4)
        );
    }
    #[test]
    fn mul() {
        assert_eq!(
            Decimal::<u32>::new_coprime(2, 3) * Decimal::new_coprime(5, 7),
            Decimal::new_coprime(10, 21)
        );
    }
    #[test]
    fn div() {
        assert_eq!(
            Decimal::<u32>::new_coprime(2, 3) / Decimal::new_coprime(5, 7),
            Decimal::new_coprime(14, 15)
        );
    }
    #[test]
    fn recip() {
        let mut num = Decimal::<u32>::new_coprime(-2, 3);
        num.recip();
        assert_eq!(num, Decimal::new_coprime(-3, 2));
    }
}
