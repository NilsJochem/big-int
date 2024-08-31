// SPDX-FileCopyrightText: 2024 Nils Jochem
// SPDX-License-Identifier: MPL-2.0
pub mod big_int;
pub mod decimal;

pub use big_int::{
    signed::{BigInt as BigIInt, SigNum, Sign},
    unsigned::BigInt as BigUInt,
};

mod util {
    pub use common::boo;
    pub mod rng;
}
pub mod iter {
    use crate::{
        big_int::digits::{Convert, Digit},
        BigUInt,
    };

    /// a Range Iterator for `BigUInt`
    pub struct Range<D: Digit> {
        start: BigUInt<D>,
        end: Option<BigUInt<D>>,
    }
    impl<D: Digit> Iterator for Range<D> {
        type Item = BigUInt<D>;

        fn next(&mut self) -> Option<Self::Item> {
            self.has_next().then(|| {
                let out = self.start.clone();
                self.start += BigUInt::ONE;
                out
            })
        }
        fn size_hint(&self) -> (usize, Option<usize>) {
            self.len()
                .as_ref()
                .and_then(Convert::<usize>::try_into)
                .map_or((usize::MAX, None), |len| (len, Some(len)))
        }
    }
    impl<D: Digit> Range<D> {
        /// checks if this iterator has more elements
        pub fn has_next(&self) -> bool {
            !self.end.as_ref().is_some_and(|end| self.start >= *end)
        }

        /// builds a new instance of `Range`. Needs start to not be bigger than end.
        pub fn new<INTO1, INTO2>(
            start: impl Into<Option<INTO1>>,
            end: impl Into<Option<INTO2>>,
        ) -> Self
        where
            INTO1: Into<BigUInt<D>>,
            INTO2: Into<BigUInt<D>>,
        {
            let start = start.into().map_or(BigUInt::ZERO, INTO1::into);
            let end = end.into().map(INTO2::into);
            if let Some(end) = &end {
                assert!(
                    start <= *end,
                    "tried to build Range with start:{start} > end:{end}"
                );
            }
            Self { start, end }
        }
        /// calculates the exact length of this range, if possible
        pub fn len(&self) -> Option<BigUInt<D>> {
            self.end.as_ref().map(|end| {
                if self.start > *end {
                    BigUInt::ZERO
                } else {
                    end - &self.start
                }
            })
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        use itertools::Itertools;

        #[test]
        fn implicit_start() {
            assert_eq!(
                Range::<u8>::new::<u8, _>(None, 10).collect_vec(),
                (0..10).map(BigUInt::from).collect_vec()
            );
        }
        #[test]
        fn len_10() {
            let range = Range::<u8>::new::<u8, _>(None, 10);
            assert_eq!(range.len(), Some(BigUInt::from(10)));
            assert_eq!(range.size_hint(), (10, Some(10)));
        }
        #[test]
        fn len_unbounded() {
            let range = Range::<u8>::new::<u8, u8>(None, None);
            assert_eq!(range.len(), None);
            assert_eq!(range.size_hint(), (usize::MAX, None));
        }
        #[test]
        fn len_overflow() {
            let mut range = Range::<u8>::new::<u8, _>(None, BigUInt::ONE << usize::BITS as usize);
            assert_eq!(range.len(), Some(BigUInt::ONE << usize::BITS as usize));
            assert_eq!(range.size_hint(), (usize::MAX, None));

            assert_eq!(range.next(), Some(BigUInt::ZERO));
            assert_eq!(
                range.len(),
                Some((BigUInt::ONE << usize::BITS as usize) - BigUInt::ONE)
            );
            assert_eq!(range.size_hint(), (usize::MAX, Some(usize::MAX)));
        }
    }
}
pub mod ops {

    pub trait Pow<RHS> {
        type Output;
        fn pow(self, rhs: RHS) -> Self::Output;
    }
    pub trait PowAssign<RHS> {
        fn pow_assign(&mut self, rhs: RHS);
    }
    pub trait DivMod<RHS = Self> {
        type Signed;
        type Unsigned;
        fn div_mod(self, rhs: RHS) -> (Self::Signed, Self::Signed);
        fn div(self, rhs: RHS) -> Self::Signed
        where
            Self: Sized,
        {
            self.div_mod(rhs).0
        }
        fn rem(self, rhs: RHS) -> Self::Signed
        where
            Self: Sized,
        {
            self.div_mod(rhs).1
        }

        fn div_mod_euclid(self, rhs: RHS) -> (Self::Signed, Self::Unsigned);
        fn div_euclid(self, rhs: RHS) -> Self::Signed
        where
            Self: Sized,
        {
            self.div_mod_euclid(rhs).0
        }
        fn rem_euclid(self, rhs: RHS) -> Self::Unsigned
        where
            Self: Sized,
        {
            self.div_mod_euclid(rhs).1
        }
    }
}
