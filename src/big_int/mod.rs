use std::fmt::{Debug, Display};

use digits::{Convert, Decomposable, Digit, Signed};
use unsigned::radix::Radix;

// SPDX-FileCopyrightText: 2024 Nils Jochem
// SPDX-License-Identifier: MPL-2.0
pub mod digits;
pub mod math_algos;
mod math_shortcuts;
mod primitve;

pub mod signed;
pub mod unsigned;

use crate::{BigIInt, BigUInt};

pub trait AnyBigIntRef<D: Digit>:
    Signed + Convert<usize> + Decomposable<D> + Decomposable<bool> + Sized
where
    for<'a> BigIntRef<'a, D>: From<&'a Self>,
{
    type Base<D1: Digit>;

    fn is_zero(&self) -> bool;
    fn is_abs_one(&self) -> bool;
    fn is_even(&self) -> bool;
    fn is_power_of_two(&self) -> bool;

    fn try_digits<T>(&self, radix: T) -> Result<usize, T::Error>
    where
        T: TryInto<Radix<D>>;

    fn digits<T>(&self, radix: T) -> usize
    where
        T: TryInto<Radix<D>>,
        T::Error: Debug,
    {
        self.try_digits(radix).unwrap()
    }

    fn try_ilog<T>(&self, radix: T) -> Result<usize, T::Error>
    where
        T: TryInto<Radix<D>>,
    {
        assert!(!self.is_zero(), "can't 0.log(radix)");
        self.try_digits(radix).map(|it| it - 1)
    }
    fn ilog<T>(&self, radix: T) -> usize
    where
        T: TryInto<Radix<D>>,
        T::Error: Debug,
    {
        assert!(!self.is_zero(), "can't 0.log(radix)");
        self.try_ilog(radix).unwrap()
    }

    fn rebase<D2: Digit>(&self) -> Self::Base<D2>;

    fn abs(&self) -> &BigUInt<D>;
}

#[allow(clippy::module_name_repetitions)]
pub trait AnyBigInt<D: Digit>: AnyBigIntRef<D>
where
    BigInt<D>: From<Self>,
    for<'a> BigIntRef<'a, D>: From<&'a Self>,
{
    fn into_abs(self) -> BigUInt<D>;
}

#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Clone, Copy, derive_more::From)]
pub enum BigIntRef<'b, D: Digit> {
    Signed(&'b BigIInt<D>),
    Unsigned(&'b BigUInt<D>),
}
impl<D: Digit> Display for BigIntRef<'_, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Signed(it) => Debug::fmt(&it, f),
            Self::Unsigned(it) => Debug::fmt(&it, f),
        }
    }
}
impl<'a, D: Digit> From<&'a BigIntRef<'_, D>> for BigIntRef<'a, D> {
    fn from(value: &'a BigIntRef<'_, D>) -> Self {
        match value {
            Self::Signed(it) => Self::Signed(it),
            Self::Unsigned(it) => Self::Unsigned(it),
        }
    }
}
impl<D: Digit> Signed for BigIntRef<'_, D> {
    fn signum(&self) -> crate::SigNum {
        match self {
            Self::Signed(it) => it.signum(),
            Self::Unsigned(it) => it.signum(),
        }
    }
}
impl<D: Digit> Convert<usize> for BigIntRef<'_, D> {
    fn try_into(&self) -> Option<usize> {
        match self {
            Self::Signed(it) => Convert::<usize>::try_into(*it),
            Self::Unsigned(it) => Convert::<usize>::try_into(*it),
        }
    }
}
impl<D: Digit> Decomposable<D> for BigIntRef<'_, D> {
    fn le_digits(&self) -> impl ExactSizeIterator<Item = D> + DoubleEndedIterator + '_ {
        Decomposable::<D>::le_digits(self.abs())
    }
}
impl<D: Digit> Decomposable<bool> for BigIntRef<'_, D> {
    fn le_digits(&self) -> impl ExactSizeIterator<Item = bool> + DoubleEndedIterator + '_ {
        Decomposable::<bool>::le_digits(self.abs())
    }
}

impl<D: Digit> AnyBigIntRef<D> for BigIntRef<'_, D> {
    type Base<D1: Digit> = BigInt<D1>;
    fn is_zero(&self) -> bool {
        match self {
            Self::Signed(it) => it.is_zero(),
            Self::Unsigned(it) => it.is_zero(),
        }
    }
    fn is_abs_one(&self) -> bool {
        match self {
            Self::Signed(it) => it.is_abs_one(),
            Self::Unsigned(it) => it.is_abs_one(),
        }
    }
    fn is_even(&self) -> bool {
        match self {
            Self::Signed(it) => it.is_even(),
            Self::Unsigned(it) => it.is_even(),
        }
    }
    fn is_power_of_two(&self) -> bool {
        match self {
            Self::Signed(it) => it.is_power_of_two(),
            Self::Unsigned(it) => it.is_power_of_two(),
        }
    }

    fn try_digits<T>(&self, radix: T) -> Result<usize, T::Error>
    where
        T: TryInto<Radix<D>>,
    {
        match self {
            Self::Signed(it) => it.try_digits(radix),
            Self::Unsigned(it) => it.try_digits(radix),
        }
    }

    fn rebase<D2: Digit>(&self) -> Self::Base<D2> {
        match self {
            Self::Signed(it) => Self::Base::Signed(it.rebase()),
            Self::Unsigned(it) => Self::Base::Unsigned(it.rebase()),
        }
    }

    fn abs(&self) -> &BigUInt<D> {
        match self {
            Self::Signed(it) => it.abs(),
            Self::Unsigned(it) => it,
        }
    }
}

#[derive(Debug, Clone, derive_more::From)]
pub enum BigInt<D: Digit> {
    Signed(BigIInt<D>),
    Unsigned(BigUInt<D>),
}
impl<D: Digit> Display for BigInt<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Signed(it) => Debug::fmt(&it, f),
            Self::Unsigned(it) => Debug::fmt(&it, f),
        }
    }
}
impl<'a, D: Digit> From<&'a BigInt<D>> for BigIntRef<'a, D> {
    fn from(value: &'a BigInt<D>) -> Self {
        match value {
            BigInt::Signed(it) => Self::Signed(it),
            BigInt::Unsigned(it) => Self::Unsigned(it),
        }
    }
}
impl<D: Digit> Signed for BigInt<D> {
    fn signum(&self) -> crate::SigNum {
        match self {
            Self::Signed(it) => it.signum(),
            Self::Unsigned(it) => it.signum(),
        }
    }
}
impl<D: Digit> Convert<usize> for BigInt<D> {
    fn try_into(&self) -> Option<usize> {
        match self {
            Self::Signed(it) => Convert::<usize>::try_into(it),
            Self::Unsigned(it) => Convert::<usize>::try_into(it),
        }
    }
}
impl<D: Digit> Decomposable<D> for BigInt<D> {
    fn le_digits(&self) -> impl ExactSizeIterator<Item = D> + DoubleEndedIterator + '_ {
        Decomposable::<D>::le_digits(self.abs())
    }
}
impl<D: Digit> Decomposable<bool> for BigInt<D> {
    fn le_digits(&self) -> impl ExactSizeIterator<Item = bool> + DoubleEndedIterator + '_ {
        Decomposable::<bool>::le_digits(self.abs())
    }
}

impl<D: Digit> AnyBigIntRef<D> for BigInt<D> {
    type Base<D1: Digit> = BigInt<D1>;
    fn is_zero(&self) -> bool {
        match self {
            Self::Signed(it) => it.is_zero(),
            Self::Unsigned(it) => it.is_zero(),
        }
    }
    fn is_abs_one(&self) -> bool {
        match self {
            Self::Signed(it) => it.is_abs_one(),
            Self::Unsigned(it) => it.is_abs_one(),
        }
    }
    fn is_even(&self) -> bool {
        match self {
            Self::Signed(it) => it.is_even(),
            Self::Unsigned(it) => it.is_even(),
        }
    }
    fn is_power_of_two(&self) -> bool {
        match self {
            Self::Signed(it) => it.is_power_of_two(),
            Self::Unsigned(it) => it.is_power_of_two(),
        }
    }

    fn try_digits<T>(&self, radix: T) -> Result<usize, T::Error>
    where
        T: TryInto<Radix<D>>,
    {
        match self {
            Self::Signed(it) => it.try_digits(radix),
            Self::Unsigned(it) => it.try_digits(radix),
        }
    }

    fn rebase<D2: Digit>(&self) -> Self::Base<D2> {
        match self {
            Self::Signed(it) => Self::Base::Signed(it.rebase()),
            Self::Unsigned(it) => Self::Base::Unsigned(it.rebase()),
        }
    }

    fn abs(&self) -> &BigUInt<D> {
        match self {
            Self::Signed(it) => it.abs(),
            Self::Unsigned(it) => it,
        }
    }
}
impl<D: Digit> AnyBigInt<D> for BigInt<D> {
    fn into_abs(self) -> BigUInt<D> {
        match self {
            Self::Signed(it) => it.into(),
            Self::Unsigned(it) => it,
        }
    }
}

impl<D: Digit> AnyBigIntRef<D> for BigIInt<D> {
    type Base<D1: Digit> = BigIInt<D1>;
    fn is_zero(&self) -> bool {
        self.is_zero()
    }
    fn is_abs_one(&self) -> bool {
        self.abs().is_one()
    }
    fn is_even(&self) -> bool {
        self.abs().is_even()
    }
    fn is_power_of_two(&self) -> bool {
        self.abs().is_power_of_two()
    }

    fn try_digits<T>(&self, radix: T) -> Result<usize, T::Error>
    where
        T: TryInto<Radix<D>>,
    {
        self.abs().try_digits(radix)
    }

    fn rebase<D2: Digit>(&self) -> Self::Base<D2> {
        self.abs().rebase().with_sign(self.signum())
    }

    fn abs(&self) -> &BigUInt<D> {
        self.abs()
    }
}
impl<D: Digit> AnyBigInt<D> for BigIInt<D> {
    fn into_abs(self) -> BigUInt<D> {
        self.into()
    }
}

impl<D: Digit> AnyBigIntRef<D> for BigUInt<D> {
    type Base<D1: Digit> = BigUInt<D1>;
    fn is_zero(&self) -> bool {
        self.is_zero()
    }
    fn is_abs_one(&self) -> bool {
        self.is_one()
    }
    fn is_even(&self) -> bool {
        self.digits.last().map_or(true, D::is_even)
    }
    fn is_power_of_two(&self) -> bool {
        self.digits.last().map_or(false, Digit::is_power_of_two)
            && self.digits.iter().rev().skip(1).all(|&it| it.eq_u8(0))
    }

    fn try_digits<T>(&self, radix: T) -> Result<usize, T::Error>
    where
        T: TryInto<Radix<D>>,
    {
        self.try_digits(radix)
    }

    fn rebase<D2: Digit>(&self) -> Self::Base<D2> {
        self.digits
            .iter()
            .flat_map(<D as Decomposable<u8>>::le_digits)
            .collect()
    }

    fn abs(&self) -> &Self {
        self
    }
}
impl<D: Digit> AnyBigInt<D> for BigUInt<D> {
    fn into_abs(self) -> Self {
        self
    }
}

#[cfg(test)]
mod tests;
