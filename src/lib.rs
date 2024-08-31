// SPDX-FileCopyrightText: 2024 Nils Jochem
// SPDX-License-Identifier: MPL-2.0
pub mod big_int;
pub mod decimal;

pub use big_int::{
    signed::{BigInt as BigIInt, SigNum, Sign},
    unsigned::BigInt as BigUInt,
};

mod util {
    pub mod boo;
    pub mod rng;
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
