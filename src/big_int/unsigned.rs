// SPDX-FileCopyrightText: 2024 Nils Jochem
// SPDX-License-Identifier: MPL-2.0
use crate::{
    big_int::{
        digits::{Convert, Decomposable, Digit, Wide},
        math_shortcuts::MathShortcut,
    },
    util::boo::{Boo, Moo},
    BigIInt, SigNum, Sign,
};

use common::{extensions::iter::IteratorExt, require};
use itertools::Itertools;
use rand::RngCore;
use std::{
    fmt::{Debug, Write},
    iter,
    ops::{
        Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div,
        DivAssign, Mul, MulAssign, RangeInclusive, Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign,
        Sub, SubAssign,
    },
    str::FromStr,
};
pub(super) mod digit_holder {
    use itertools::Itertools;
    use std::{
        fmt::Debug,
        ops::{Index, IndexMut},
    };

    #[derive(Clone, Hash)]
    pub enum DigitHolder<D> {
        None,
        One(D),
        Other(Vec<D>),
    }
    impl<D> Default for DigitHolder<D> {
        fn default() -> Self {
            Self::None
        }
    }
    impl<D: Debug> Debug for DigitHolder<D> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::None => f.debug_list().finish(),
                Self::One(arg0) => f.debug_list().entries(std::iter::once(arg0)).finish(),
                Self::Other(arg0) => Debug::fmt(arg0, f),
            }
        }
    }

    impl<D> From<Vec<D>> for DigitHolder<D> {
        fn from(mut values: Vec<D>) -> Self {
            match values.len() {
                0 => Self::new(),
                1 => Self::from_single(values.pop().unwrap_or_else(|| unreachable!())),
                _ => Self::Other(values),
            }
        }
    }
    impl<D> From<D> for DigitHolder<D> {
        fn from(value: D) -> Self {
            Self::One(value)
        }
    }
    impl<D> FromIterator<D> for DigitHolder<D> {
        fn from_iter<T: IntoIterator<Item = D>>(iter: T) -> Self {
            Self::from_vec(iter.into_iter().collect_vec())
        }
    }

    impl<D> From<DigitHolder<D>> for Vec<D> {
        fn from(value: DigitHolder<D>) -> Self {
            match value {
                DigitHolder::None => vec![],
                DigitHolder::One(digit) => vec![digit],
                DigitHolder::Other(digits) => digits,
            }
        }
    }

    impl<D> DigitHolder<D> {
        pub const fn new() -> Self {
            Self::None
        }
        pub const fn from_single(value: D) -> Self {
            Self::One(value)
        }
        pub fn from_vec(values: Vec<D>) -> Self {
            values.into()
        }
        pub fn len(&self) -> usize {
            match self {
                Self::None => 0,
                Self::One(_) => 1,
                Self::Other(digits) => {
                    debug_assert!(
                        digits.len() > 1,
                        "vec remained with 1 < {} elements",
                        digits.len()
                    );
                    digits.len()
                }
            }
        }
        pub const fn is_empty(&self) -> bool {
            matches!(self, Self::None)
        }
        pub fn is_in_bounds(&self, index: usize) -> bool {
            self.len() > index
        }
        pub fn assert_bounds(&self, index: usize) {
            assert!(
                self.is_in_bounds(index),
                "tried to access element {index} with len: {}",
                self.len()
            );
        }
        pub fn get(&self, index: usize) -> Option<&D> {
            match self {
                Self::None => None,
                Self::One(digit) => Some(digit).filter(|_| index == 0),
                Self::Other(digits) => digits.get(index),
            }
        }
        pub fn get_mut(&mut self, index: usize) -> Option<&mut D> {
            match self {
                Self::None => None,
                Self::One(digit) => Some(digit).filter(|_| index == 0),
                Self::Other(digits) => digits.get_mut(index),
            }
        }

        pub fn first(&self) -> Option<&D> {
            match self {
                Self::None => None,
                Self::One(digit) => Some(digit),
                Self::Other(digits) => digits.first(),
            }
        }
        pub fn first_mut(&mut self) -> Option<&mut D> {
            match self {
                Self::None => None,
                Self::One(digit) => Some(digit),
                Self::Other(digits) => digits.first_mut(),
            }
        }
        pub fn last(&self) -> Option<&D> {
            match self {
                Self::None => None,
                Self::One(digit) => Some(digit),
                Self::Other(digits) => digits.last(),
            }
        }
        pub fn last_mut(&mut self) -> Option<&mut D> {
            match self {
                Self::None => None,
                Self::One(digit) => Some(digit),
                Self::Other(digits) => digits.last_mut(),
            }
        }
        pub fn as_ptr(&self) -> *const D {
            match self {
                Self::None => [].as_slice().as_ptr(),
                Self::One(digit) => std::ptr::from_ref(digit),
                Self::Other(digits) => digits.as_ptr(),
            }
        }

        pub fn push(&mut self, value: D)
        where
            D: Copy,
        {
            match self {
                Self::None => {
                    *self = Self::from_single(value);
                }
                Self::One(digit) => {
                    *self = Self::from_vec(vec![*digit, value]);
                }
                Self::Other(digits) => digits.push(value),
            }
        }
        pub fn pop(&mut self) -> Option<D> {
            if self.len() <= 2 {
                match std::mem::take(self) {
                    Self::None => None,
                    Self::One(digit) => Some(digit),
                    Self::Other(mut digits) => {
                        let out = digits.pop();
                        debug_assert_eq!(digits.len(), 1);
                        *self = Self::from_single(digits.pop().unwrap());
                        out
                    }
                }
            } else {
                match self {
                    Self::None | Self::One(_) => unreachable!(),
                    Self::Other(digits) => digits.pop(),
                }
            }
        }
        pub fn remove(&mut self, index: usize) -> Option<D> {
            if !self.is_in_bounds(index) {
                return None;
            }
            if self.len() <= 2 {
                match std::mem::take(self) {
                    Self::None => unreachable!(),
                    Self::One(digit) => {
                        if index == 0 {
                            Some(digit)
                        } else {
                            *self = Self::from_single(digit);
                            None
                        }
                    }
                    Self::Other(mut digits) => {
                        let out = digits.remove(index);
                        debug_assert_eq!(digits.len(), 1);
                        *self = Self::from_single(digits.pop().unwrap());
                        Some(out)
                    }
                }
            } else {
                match self {
                    Self::None | Self::One(_) => unreachable!(),
                    Self::Other(digits) => Some(digits.remove(index)),
                }
            }
        }
        pub fn reverse(&mut self) {
            match self {
                Self::None | Self::One(_) => {}
                Self::Other(digits) => digits.reverse(),
            }
        }
        pub fn truncate(&mut self, len: usize) {
            match len {
                0 => *self = Self::new(),
                1 => match self {
                    Self::None | Self::One(_) => {}
                    Self::Other(digits) => {
                        digits.truncate(1);
                        *self = Self::One(digits.pop().unwrap());
                    }
                },
                _ => {
                    match self {
                        Self::None | Self::One(_) => {}
                        Self::Other(digits) => digits.truncate(len),
                    };
                }
            }
        }

        pub fn iter(&self) -> impl ExactSizeIterator<Item = &D> + DoubleEndedIterator {
            self.as_slice().iter()
        }
        pub fn as_slice(&self) -> &[D] {
            match self {
                Self::None => [].as_slice(),
                Self::One(digit) => std::slice::from_ref(digit),
                Self::Other(digits) => digits.as_slice(),
            }
        }
        pub fn iter_mut(&mut self) -> impl ExactSizeIterator<Item = &mut D> + DoubleEndedIterator {
            self.as_mut_slice().iter_mut()
        }
        pub fn as_mut_slice(&mut self) -> &mut [D] {
            match self {
                Self::None => [].as_mut_slice(),
                Self::One(digit) => std::slice::from_mut(digit),
                Self::Other(digits) => digits.as_mut_slice(),
            }
        }
    }
    impl<D> Index<usize> for DigitHolder<D> {
        type Output = D;

        fn index(&self, index: usize) -> &Self::Output {
            self.assert_bounds(index);
            self.get(index).unwrap()
        }
    }
    impl<D> IndexMut<usize> for DigitHolder<D> {
        fn index_mut(&mut self, index: usize) -> &mut Self::Output {
            self.get_mut(index).unwrap()
        }
    }
    pub enum DigitIter<D> {
        None,
        One(std::iter::Once<D>),
        Other(<Vec<D> as IntoIterator>::IntoIter),
    }
    impl<D> Iterator for DigitIter<D> {
        type Item = D;

        fn next(&mut self) -> Option<Self::Item> {
            match self {
                Self::None => None,
                Self::One(iter) => iter.next(),
                Self::Other(iter) => iter.next(),
            }
        }
        fn size_hint(&self) -> (usize, Option<usize>) {
            match self {
                Self::None => (0, Some(0)),
                Self::One(iter) => iter.size_hint(),
                Self::Other(iter) => iter.size_hint(),
            }
        }
    }
    impl<D> ExactSizeIterator for DigitIter<D> {}
    impl<D> DoubleEndedIterator for DigitIter<D> {
        fn next_back(&mut self) -> Option<Self::Item> {
            match self {
                Self::None => None,
                Self::One(iter) => iter.next_back(),
                Self::Other(iter) => iter.next_back(),
            }
        }
    }

    impl<D> IntoIterator for DigitHolder<D> {
        type Item = D;
        type IntoIter = DigitIter<D>;

        fn into_iter(self) -> Self::IntoIter {
            match self {
                Self::None => DigitIter::None,
                Self::One(digit) => DigitIter::One(std::iter::once(digit)),
                Self::Other(digits) => DigitIter::Other(digits.into_iter()),
            }
        }
    }
    impl<D: Copy> Extend<D> for DigitHolder<D> {
        fn extend<T: IntoIterator<Item = D>>(&mut self, iter: T) {
            match self {
                Self::None => *self = Self::from_vec(iter.into_iter().collect_vec()),
                Self::One(digit) => {
                    // TODO move self
                    *self = Self::from_vec(std::iter::once(*digit).chain(iter).collect_vec());
                }
                Self::Other(digits) => digits.extend(iter),
            }
        }
    }
}
use digit_holder::DigitHolder;
#[derive(Clone, Default, Hash)]
pub struct BigInt<D> {
    /// holds the digits in LE order
    pub(super) digits: DigitHolder<D>,
}

impl<D: Digit> Debug for BigInt<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Number {{ ")?;
        self.inner_debug(f)?;
        write!(f, "}}")
    }
}
impl<D: Digit> std::fmt::Display for BigInt<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.clone().write_with_radix(
            f,
            10,
            f.alternate().then_some((3, '_')),
            f.width()
                .map(|w| (w, f.align().unwrap_or(std::fmt::Alignment::Right), f.fill())),
        )
    }
}
impl<D: Digit> std::fmt::LowerHex for BigInt<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if f.alternate() {
            write!(f, "0x")?;
        }
        for digit in self.digits.iter().rev() {
            write!(f, "{digit:x}")?;
        }
        Ok(())
    }
}
impl<D: Digit> std::fmt::UpperHex for BigInt<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if f.alternate() {
            write!(f, "0X")?;
        }
        for digit in self.digits.iter().rev() {
            write!(f, "{digit:X}")?;
        }
        Ok(())
    }
}

impl<D: Digit> Eq for BigInt<D> {}
impl<D: Digit> Ord for BigInt<D> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
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
        let digits = other.le_digits();

        for elem in self.digits.iter().zip_longest(digits).rev() {
            match elem {
                itertools::EitherOrBoth::Both(lhs, rhs) => {
                    let ord = lhs.cmp(&rhs);
                    if ord.is_ne() {
                        return Some(ord);
                    }
                }
                itertools::EitherOrBoth::Right(_) => return Some(std::cmp::Ordering::Less),
                itertools::EitherOrBoth::Left(_) => return Some(std::cmp::Ordering::Greater),
            }
        }
        Some(std::cmp::Ordering::Equal)
    }
}

// From helper
enum StripRadix {
    UnkoneRadix(char),
    OnlyZero,
}
fn strip_radix(s: &str) -> Result<(Option<u8>, &str), StripRadix> {
    let mut chars = s.chars();

    if chars.next() == Some('0') {
        Ok((
            Some(match chars.next() {
                Some('x') => 16,
                Some('b') => 2,
                Some('d') => 10,
                Some('o') => 8,
                Some(c) => return Err(StripRadix::UnkoneRadix(c)),
                None => return Err(StripRadix::OnlyZero),
            }),
            &s[2..],
        ))
    } else {
        Ok((None, s))
    }
}
#[derive(Debug, PartialEq, Eq)]
pub enum FromStrErr {
    UnkownDigit { digit: char, position: usize },
    UnkoneRadix(char),
    Empty,
}

// From traits
impl<D: Digit> From<BigIInt<D>> for BigInt<D> {
    fn from(value: BigIInt<D>) -> Self {
        value.unsigned
    }
}
impl<POSITIVE: super::primitve::UNum, D: Digit> FromIterator<POSITIVE> for BigInt<D> {
    /// the iter should contain the digits in little endian order
    fn from_iter<T: IntoIterator<Item = POSITIVE>>(iter: T) -> Self {
        Self::from_digits(D::from_le(
            iter.into_iter()
                .flat_map(super::primitve::Primitive::to_le_bytes),
        ))
    }
}
cfg_if::cfg_if! {
    if #[cfg(all(
        feature = "uintFromAbsIPrimitive",
        feature = "uintFromAssertIPrimitive"
    ))] {
        compile_error!("feature \"uintFromAbsIPrimitive\" and feature \"uintFromAssertIPrimitive\" cannot be enabled at the same time");
    } else if #[cfg(any(
        feature = "uintFromAbsIPrimitive",
        feature = "uintFromAssertIPrimitive"
    ))] {
        use crate::big_int::primitve::{Either, INum};
        impl<PRIMITIVE: super::primitve::Primitive, D: Digit> From<PRIMITIVE> for BigInt<D> {
            fn from(value: PRIMITIVE) -> Self {
                iter::once(
                    match value.select_sign() {
                        Either::Left(pos) => pos,
                        Either::Right(neg) => {
                            #[cfg(feature = "uintFromAssertIPrimitive")]
                            assert!(!neg.is_negative() , "tried to get BigUInt from {value} < 0");
                            neg.abs()
                        },
                    }
                ).collect()
            }
        }
    } else {
        impl<POSITIVE: super::primitve::UNum, D: Digit> From<POSITIVE> for BigInt<D> {
            fn from(pos: POSITIVE) -> Self {
                iter::once(pos).collect()
            }
        }
    }
}

impl<D: Digit> FromStr for BigInt<D> {
    type Err = FromStrErr;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (radix, rest) = match strip_radix(s) {
            Ok((radix, rest)) => (radix.unwrap_or(10), rest),
            Err(StripRadix::UnkoneRadix(c)) => return Err(FromStrErr::UnkoneRadix(c)),
            Err(StripRadix::OnlyZero) => return Ok(Self::ZERO),
        };

        require!(!rest.is_empty(), FromStrErr::Empty);

        Self::from_base(rest, radix, |c| c.to_digit(radix as u32)).map_err(|mut err| {
            match &mut err {
                FromStrErr::UnkownDigit { digit: _, position } => {
                    *position += s.len() - rest.len();
                }
                FromStrErr::UnkoneRadix(_) | FromStrErr::Empty => {}
            };
            err
        })
    }
}

// Into traits
impl<D: Digit> Convert<usize> for BigInt<D> {
    fn try_into(&self) -> Option<usize> {
        Some(D::try_combine(self.digits.iter().copied()).unwrap())
    }
}
impl<D: Digit> Decomposable<D> for BigInt<D> {
    fn le_digits(&self) -> impl ExactSizeIterator<Item = D> + DoubleEndedIterator + '_ {
        self.digits.iter().copied()
    }
}
impl<D: Digit> Decomposable<bool> for BigInt<D> {
    fn le_digits(&self) -> impl ExactSizeIterator<Item = bool> + DoubleEndedIterator + '_ {
        self.digits
            .iter()
            .flat_map(|it| it.iter_le_bits(true))
            .with_size(self.digits.len() * D::BASIS_POW)
            .take(self.digits(2))
    }
}

pub mod radix {
    use core::num::NonZero;

    use super::BigInt;
    use crate::big_int::digits::Digit;
    use common::require;

    pub const NONZERO_ONE: NonZero<usize> = {
        // SAFETY: 1 is non zero
        unsafe { NonZero::new_unchecked(1) }
    };

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum Radix<D: Digit> {
        DigitBase,
        /// should not be `DigitBase`
        PowerOfTwo(NonZero<usize>),
        Other(BigInt<D>),
    }
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Error {
        Zero,
        One,
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
        type Error = Error;

        fn try_from(value: usize) -> Result<Self, Error> {
            require!(value != 0, Error::Zero);
            require!(value != 1, Error::One);
            Ok(if value.is_power_of_two() {
                Self::from_pow_2(NonZero::new(value.ilog2() as usize).unwrap_or_else(|| {
                    unreachable!("ilog2 doesn't return 0 unless the value is 0")
                }))
            } else {
                Self::Other(value.into())
            })
        }
    }

    impl<D: Digit> TryFrom<BigInt<D>> for Radix<D> {
        type Error = Error;

        fn try_from(value: BigInt<D>) -> Result<Self, Error> {
            require!(value.is_zero(), Error::Zero);
            require!(value.is_one(), Error::One);
            Ok(if value.is_power_of_two() {
                Self::from_pow_2(NonZero::new(value.ilog(2)).unwrap_or_else(|| {
                    unreachable!("ilog2 doesn't return 0 unless the value is 0")
                }))
            } else {
                Self::Other(value)
            })
        }
    }
}
use radix::Radix;

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

impl<D: Digit> BigInt<D> {
    pub const ZERO: Self = Self {
        digits: DigitHolder::new(),
    };
    pub const ONE: Self = Self {
        digits: DigitHolder::from_single(D::ONE),
    };

    // construction
    pub fn with_sign(self, sign: Sign) -> BigIInt<D> {
        BigIInt {
            signum: if self.is_zero() {
                SigNum::Zero
            } else {
                sign.into()
            },
            unsigned: self,
        }
    }
    /// generate a new random number with at least `bytes.start()` and at most `bytes.end()` bytes of information
    /// # Example
    /// `0x00_0100` <= `BigInt::new_random(2..=3, _)` <= `0xff_ffff`,
    pub fn new_random(bytes: RangeInclusive<usize>, mut rng: impl RngCore) -> Self {
        let bytes = bytes.start()
            + crate::util::rng::next_bound(*bytes.end() - *bytes.start(), &mut rng, 10);
        let mut rnd_bytes = crate::util::rng::random_bytes(rng);
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
                digits: DigitHolder::new(),
            }
        } else {
            Self {
                digits: DigitHolder::from_single(value),
            }
        }
    }
    pub fn from_digits(iter: impl IntoIterator<Item = D>) -> Self {
        let mut num = Self {
            digits: iter.into_iter().collect(),
        };
        num.truncate_leading_zeros();
        num
    }

    // inner utils
    pub(super) fn truncate_leading_zeros(&mut self) {
        while self.digits.last().is_some_and(|&it| it.eq_u8(0)) {
            self.digits.pop();
        }
    }
    pub(super) fn push(&mut self, value: impl Into<D>) {
        let value = value.into();
        if value.eq_u8(0) {
            return;
        }
        self.digits.push(value);
    }

    pub(super) fn inner_debug(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "0x[",)?;
        for (pos, elem) in self.digits.iter().rev().with_position() {
            write!(f, "{elem:0size$x}", size = D::BYTES * 2)?;
            if matches!(
                pos,
                itertools::Position::First | itertools::Position::Middle
            ) {
                f.write_str(", ")?;
            }
        }
        write!(f, "]")
    }
    pub fn write_with_radix(
        mut self,
        f: &mut std::fmt::Formatter,
        radix: u8,
        seperator: Option<(usize, char)>,
        pad: Option<(usize, std::fmt::Alignment, char)>,
    ) -> Result<(), std::fmt::Error> {
        use itertools::Either;
        assert_ne!(radix, 0, "can't print with radix == 0");
        assert!(
            !pad.is_some_and(|(_, _, it)| it == ' ') || seperator.is_none(),
            "todo alternate with space pad"
        );
        let big_radix = Self::from_digit(D::from(radix));
        let mut buf = Vec::new();
        while !self.is_zero() {
            let (_, mut remainder) = Self::div_mod_euclid(&mut self, &big_radix);
            debug_assert!(remainder.digits.len() <= 1);
            buf.push(Either::Left(remainder.digits.pop().unwrap_or_default()));
        }
        if buf.is_empty() {
            buf.push(Either::Left(D::default()));
        }
        if let Some((pad_size, align, pad_char)) = pad {
            match align {
                std::fmt::Alignment::Left => {
                    buf.extend(
                        std::iter::repeat(Either::Right(pad_char))
                            .take(pad_size)
                            .skip(buf.len()),
                    );
                }
                std::fmt::Alignment::Right => {
                    assert!(
                        !pad_char.is_digit(radix as u32),
                        "padding '{pad_char}' is a valid char with the radix {radix}"
                    );
                    buf = std::iter::repeat(Either::Right(pad_char))
                        .take(pad_size)
                        .skip(buf.len())
                        .chain(buf)
                        .collect();
                }
                std::fmt::Alignment::Center => todo!("not ready"),
            }
        }
        if let Some((seperator_distance, seperator)) = seperator {
            for (pos, digits) in buf
                .iter()
                .chunks(seperator_distance)
                .into_iter()
                .collect_vec()
                .into_iter()
                .rev()
                .with_position()
            {
                for digit in digits.collect_vec().into_iter().rev() {
                    match digit {
                        Either::Left(digit) => write!(f, "{digit:?}")?,
                        Either::Right(c) => write!(f, "{c}")?,
                    }
                }
                match pos {
                    itertools::Position::Middle | itertools::Position::First => {
                        f.write_char(seperator)?;
                    }
                    itertools::Position::Last | itertools::Position::Only => {}
                }
            }
        } else {
            for digit in buf.iter().rev() {
                match digit {
                    Either::Left(digit) => write!(f, "{digit:?}")?,
                    Either::Right(c) => write!(f, "{c}")?,
                }
            }
        }
        Ok(())
    }
    // getter
    pub const fn is_zero(&self) -> bool {
        self.digits.is_empty()
    }
    pub const fn signum(&self) -> SigNum {
        if self.is_zero() {
            SigNum::Zero
        } else {
            SigNum::Positive
        }
    }
    pub fn is_one(&self) -> bool {
        self.digits.len() == 1 && self.digits.first().unwrap().eq_u8(1)
    }
    pub fn is_even(&self) -> bool {
        self.digits.last().map_or(true, D::is_even)
    }
    pub fn is_power_of_two(&self) -> bool {
        self.digits.last().map_or(false, Digit::is_power_of_two)
            && self.digits.iter().rev().skip(1).all(|&it| it.eq_u8(0))
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
                Radix::PowerOfTwo(radix::NONZERO_ONE) => number.digits.last().map_or(0, |last| {
                    (inner(number, Radix::DigitBase) - 1) * D::BASIS_POW + last.ilog2() as usize + 1
                }),
                Radix::PowerOfTwo(power) => inner(number, Radix::PowerOfTwo(radix::NONZERO_ONE))
                    .div_ceil(1 << (power.get() - 1)),
                Radix::Other(radix) => {
                    let mut n = 1;
                    let mut number = number.clone();
                    while number.cmp(&radix).is_ge() {
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

    pub fn rebase<D2: Digit>(&self) -> BigInt<D2> {
        BigInt::<D2>::from_iter(
            self.digits
                .iter()
                .flat_map(<D as Decomposable<u8>>::le_digits),
        )
    }
    fn from_base(
        source: &str,
        radix: u8,
        map: impl Fn(char) -> Option<u32>,
    ) -> Result<Self, FromStrErr> {
        let mut num = Self::ZERO;
        let mut digits = source
            .chars()
            .enumerate()
            .filter(|(_, digit)| *digit != '_')
            .peekable();
        if digits.peek().is_none() {
            return Err(FromStrErr::Empty);
        }
        digits
            .by_ref()
            .take_while_ref(|(_, it)| *it == '0')
            .for_each(drop);
        if digits.peek().is_none() {
            return Ok(num);
        }

        if [2, 16].contains(&radix) {
            // optimize by trying to fill one D up and accumulate change
            let shift = radix.ilog2() as usize;
            let num_sub_digits = D::BASIS_POW / shift;
            let mut digit_buf = Vec::with_capacity(num_sub_digits);
            for (pos, n_digits) in digits
                .chunks(digit_buf.capacity())
                .into_iter()
                .with_position()
            {
                digit_buf.clear();
                debug_assert_eq!(digit_buf.capacity(), num_sub_digits);
                digit_buf.extend(n_digits);

                num.digits.push(D::default());
                let buf = if matches!(pos, itertools::Position::Last | itertools::Position::Only) {
                    num.digits.reverse();
                    num.truncate_leading_zeros();
                    num >>= D::BASIS_POW - (digit_buf.len() * shift);
                    if num.digits.is_empty() {
                        num.digits.push(D::default());
                    }
                    num.digits.first_mut().unwrap()
                } else {
                    num.digits.last_mut().unwrap()
                };

                for (j, &(i, digit)) in digit_buf.iter().rev().enumerate() {
                    match map(digit) {
                        Some(digit) => {
                            *buf |= &(D::from(digit as u8) << (shift * j));
                        }
                        None => return Err(FromStrErr::UnkownDigit { digit, position: i }),
                    }
                }
            }
        } else {
            for (i, digit) in digits {
                match map(digit) {
                    Some(digit) => {
                        num *= D::from(radix);
                        num += Self::from_digit(D::from(digit as u8));
                    }
                    None => return Err(FromStrErr::UnkownDigit { digit, position: i }),
                }
            }
        }

        Ok(num)
    }

    /// needs to newly allocate on big endian systems
    /// will return the sign seperatly as this function cannot know which character isn't already used by the encoding, or otherwise not usable.
    #[cfg(feature = "base64")]
    pub fn as_base64(&self, engine: &impl base64::Engine) -> String {
        if cfg!(target_endian = "little") {
            engine.encode(self.le_bytes_ref())
        } else {
            let buf = self
                .digits
                .iter()
                .flat_map(<D as Decomposable<u8>>::le_digits)
                .collect_vec();
            engine.encode(&buf)
        }
    }
    #[cfg(feature = "base64")]
    pub fn from_base64(
        data: impl AsRef<[u8]>,
        engine: &impl base64::Engine,
    ) -> Result<Self, base64::DecodeError> {
        engine.decode(data).map(Self::from_iter)
    }

    #[cfg(target_endian = "little")]
    #[cfg(feature = "base64")]
    fn le_bytes_ref(&self) -> &[u8] {
        // SAFETY: as digits are all numbers, accessing their bytes should be fine
        unsafe {
            std::slice::from_raw_parts(self.digits.as_ptr().cast(), self.digits.len() * D::BYTES)
        }
    }

    pub(super) fn assert_pair_valid<T>(lhs: &Boo<'_, T>, rhs: &Boo<'_, T>) {
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
    // math
    pub(super) fn bitor<'b, 'b1: 'b, 'b2: 'b, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b, Self>
    where
        B1: Into<Boo<'b1, Self>>,
        B2: Into<Boo<'b2, Self>>,
        D: 'b1 + 'b2,
    {
        Self::refer_direct::<'_, '_, '_, _, _, TieBigger>(
            lhs,
            rhs,
            super::math_algos::bit_math::bit_or_assign,
        )
    }
    pub(super) fn bitxor<'b, 'b1: 'b, 'b2: 'b, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b, Self>
    where
        B1: Into<Boo<'b1, Self>>,
        B2: Into<Boo<'b2, Self>>,
        D: 'b1 + 'b2,
    {
        Self::refer_direct::<'_, '_, '_, _, _, TieBigger>(
            lhs,
            rhs,
            super::math_algos::bit_math::bit_xor_assign,
        )
    }

    pub(super) fn bitand<'b, 'b1: 'b, 'b2: 'b, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b, Self>
    where
        B1: Into<Boo<'b1, Self>>,
        B2: Into<Boo<'b2, Self>>,
        D: 'b1 + 'b2,
    {
        Self::refer_direct::<'_, '_, '_, _, _, TieSmaller>(
            lhs,
            rhs,
            super::math_algos::bit_math::bit_and_assign,
        )
    }

    pub(super) fn shl<'b, 'b1: 'b, 'b2: 'b, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b, Self>
    where
        B1: Into<Boo<'b1, Self>>,
        B2: Into<Boo<'b2, usize>>,
    {
        let mut lhs = Moo::<Self>::from(lhs.into());
        let rhs = rhs.into().copied();

        let partial = rhs % D::BASIS_POW;
        let full = rhs / D::BASIS_POW;

        let mut carry = D::default();
        if partial > 0 {
            for digit in lhs.digits.iter_mut() {
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
    pub(super) fn shr<'b, 'b1: 'b, 'b2: 'b, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b, Self>
    where
        B1: Into<Boo<'b1, Self>>,
        B2: Into<Boo<'b2, usize>>,
    {
        Self::shr_internal(lhs, rhs).0
    }

    pub(super) fn shr_internal<'b, 'b1: 'b, 'b2: 'b, B1, B2>(
        lhs: B1,
        rhs: B2,
    ) -> (Moo<'b, Self>, Self)
    where
        B1: Into<Boo<'b1, Self>>,
        B2: Into<Boo<'b2, usize>>,
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

        super::math_shortcuts::try_all!(lhs, rhs, super::math_shortcuts::add::Zero,);

        Self::refer_direct::<'_, '_, '_, _, _, TieSmaller>(lhs, rhs, super::math_algos::add::assign)
    }
    pub(crate) fn sub<'b, 'b1: 'b, 'b2: 'b, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b, Self>
    where
        B1: Into<Boo<'b1, Self>>,
        B2: Into<Boo<'b2, Self>>,
    {
        let lhs: Boo<'_, Self> = lhs.into();
        let rhs: Boo<'_, Self> = rhs.into();

        Self::assert_pair_valid(&lhs, &rhs);
        assert!(*lhs >= *rhs, "result would be negative");
        if rhs.is_zero() {
            return super::math_shortcuts::get_lhs(lhs, rhs);
        }

        match (lhs, rhs) {
            (Boo::BorrowedMut(lhs), rhs) => {
                super::math_algos::sub::assign_smaller(lhs, &rhs);
                Moo::BorrowedMut(lhs)
            }
            (lhs, Boo::BorrowedMut(borrowed)) => {
                let old_rhs = std::mem::replace(borrowed, lhs.cloned()); // lhs -> rhs, rhs -> old_rhs
                super::math_algos::sub::assign_smaller(borrowed, &old_rhs);
                Moo::BorrowedMut(borrowed)
            }
            (lhs, rhs) => {
                // can't really use storage in rhs (when existing) because algo can only sub smaller
                let mut lhs = lhs.cloned();
                super::math_algos::sub::assign_smaller(&mut lhs, &rhs);
                Moo::Owned(lhs)
            }
        }
    }
    pub(crate) fn mul_by_digit<'b, 'b1: 'b, 'b2: 'b, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b, Self>
    where
        B1: Into<Boo<'b1, Self>>,
        B2: Into<Boo<'b2, D>>,
        D: 'b1 + 'b2,
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
        super::math_algos::mul::assign_mul_digit_at_offset(&mut lhs, rhs, 0);
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

        super::math_shortcuts::try_all!(
            lhs,
            rhs,
            super::math_shortcuts::mul::ByZero,
            super::math_shortcuts::mul::ByOne,
            super::math_shortcuts::mul::ByPowerOfTwo,
        );

        match (lhs, rhs) {
            (Boo::BorrowedMut(borrow_mut), borrow) | (borrow, Boo::BorrowedMut(borrow_mut)) => {
                *borrow_mut = super::math_algos::mul::naive(borrow_mut, &borrow);
                Moo::BorrowedMut(borrow_mut)
            }
            (lhs, rhs) => Moo::Owned(super::math_algos::mul::naive(&lhs, &rhs)),
        }
    }
    pub fn pow<'b, 'b1: 'b, 'b2: 'b, B1, B2, P>(lhs: B1, pow: B2) -> Moo<'b, Self>
    where
        B1: Into<Boo<'b1, Self>>,
        P: Decomposable<bool> + 'b2 + super::digits::Signed + Clone,
        B2: Into<Boo<'b2, P>>,
        D: 'b1 + 'b2,
    {
        let pow: Boo<P> = pow.into();
        assert!(
            !matches!(pow, Boo::BorrowedMut(_)),
            "will not assign to power"
        );
        assert!(!pow.signum().is_negative(), "can't pow ints by negatives");

        let lhs: Boo<_> = lhs.into();

        if pow.signum().is_zero() {
            return Moo::from_with_value(lhs, Self::ONE);
        }
        if lhs.is_zero() {
            return lhs.into();
        }
        if lhs.is_power_of_two() {
            let l_pow = lhs.digits(2) - 1;
            if let Some(pow) =
                <P as Convert<usize>>::try_into(&pow).and_then(|it| it.checked_mul(l_pow))
            {
                return Self::shl(Self::ONE, pow);
            }
        }

        let mut out: Moo<Self> = Moo::from(lhs);
        let mut x = std::mem::replace(&mut *out, Self::ONE);

        for bit in pow.le_digits() {
            if bit {
                *out.get_mut() *= &x;
            }
            x = &x * &x;
        }
        out
    }

    pub(crate) fn div_euclid<'b, 'b1: 'b, 'b2: 'b, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b1, Self>
    where
        B1: Into<Boo<'b1, Self>>,
        B2: Into<Boo<'b2, Self>>,
    {
        Self::div_mod_euclid(lhs, rhs).0
    }
    #[allow(dead_code)]
    pub(crate) fn rem_euclid<'b, 'b1: 'b, 'b2: 'b, B1, B2>(lhs: B1, rhs: B2) -> Moo<'b2, Self>
    where
        B1: Into<Boo<'b1, Self>>,
        B2: Into<Boo<'b2, Self>>,
    {
        Self::div_mod_euclid(lhs, rhs).1
    }
    pub fn div_mod_euclid<'b, 'b1: 'b, 'b2: 'b, B1, B2>(
        lhs: B1,
        rhs: B2,
    ) -> (Moo<'b1, Self>, Moo<'b2, Self>)
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

        if *lhs < *rhs {
            let (a, lhs) = lhs.take_keep_ref();
            return (
                Moo::from_with_value(lhs, Self::ZERO),
                Moo::from_with_value(rhs, a),
            );
        }
        if *lhs == *rhs {
            return (
                Moo::from_with_value(lhs, Self::ONE),
                Moo::from_with_value(rhs, Self::ZERO),
            );
        }
        if rhs.is_power_of_two() {
            let (q, r) = Self::shr_internal(lhs, rhs.ilog(2));
            return (q, Moo::from_with_value(rhs, r));
        }

        let (n, lhs) = lhs.take_keep_ref();
        let (d, rhs) = rhs.take_keep_ref();

        let (q, r) = super::math_algos::div::normalized_schoolbook(n, d);

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
