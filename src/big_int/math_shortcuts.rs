#![allow(clippy::wildcard_imports)]
use super::BigInt;
use crate::boo::{Boo, Moo};

macro_rules! try_all {
    ($lhs:ident, $rhs:ident $(, )?) => {};
    ($lhs:ident, $rhs:ident, flip $($rule:tt)::*, $($tail:tt)*) => {
        math_shortcuts::one_side!($($rule)::*, $rhs, $lhs);
        math_shortcuts::try_all!($lhs, $rhs, $($rule)::*, $($tail)*);
    };
    ($lhs:ident, $rhs:ident, $($rule:tt)::*, $($tail:tt)*) => {
        math_shortcuts::one_side!($($rule)::*, $lhs, $rhs);
        math_shortcuts::try_all!($lhs, $rhs, $($tail)*);
    };
}
macro_rules! one_side {
    ($($rule:tt)::*, $lhs:ident, $rhs:ident) => {
        if let Some(sc) = $($rule)::*::can_shortcut(&$lhs, &$rhs) {
            return $($rule)::*::do_shortcut($lhs, $rhs, sc);
        }
    };
}
pub(crate) use {one_side, try_all};

pub trait MathShortcut {
    type SC;
    fn can_shortcut(lhs: &BigInt, rhs: &BigInt) -> Option<Self::SC>;
    fn do_shortcut<'b>(lhs: Boo<'b, BigInt>, rhs: Boo<'b, BigInt>, sc: Self::SC)
        -> Moo<'b, BigInt>;
}

fn get_lhs<'b>(lhs: Boo<'b, BigInt>, rhs: Boo<'b, BigInt>) -> Moo<'b, BigInt> {
    match (lhs, rhs) {
        (lhs, Boo::BorrowedMut(rhs)) => {
            *rhs = lhs.cloned();
            Moo::BorrowedMut(rhs)
        }
        (lhs, _) => Moo::<BigInt>::from(lhs),
    }
}

pub mod add {
    use super::*;
    pub struct ByZero;
    impl MathShortcut for ByZero {
        type SC = ();

        fn can_shortcut(_lhs: &BigInt, rhs: &BigInt) -> Option<Self::SC> {
            rhs.is_zero().then_some(())
        }

        fn do_shortcut<'b>(
            lhs: Boo<'b, BigInt>,
            rhs: Boo<'b, BigInt>,
            (): Self::SC,
        ) -> Moo<'b, BigInt> {
            super::get_lhs(lhs, rhs)
        }
    }
}

pub mod sub {
    use super::*;
    pub struct ByZero;
    impl MathShortcut for ByZero {
        type SC = bool;

        fn can_shortcut(lhs: &BigInt, rhs: &BigInt) -> Option<Self::SC> {
            rhs.is_zero()
                .then_some(false)
                .or_else(|| lhs.is_zero().then_some(true))
        }

        fn do_shortcut<'b>(
            lhs: Boo<'b, BigInt>,
            rhs: Boo<'b, BigInt>,
            need_negate: Self::SC,
        ) -> Moo<'b, BigInt> {
            let mut either = super::get_lhs(lhs, rhs);
            if need_negate {
                either.negate();
            }
            either
        }
    }
}

pub mod mul {
    use super::*;
    pub struct ByZero;
    impl MathShortcut for ByZero {
        type SC = ();

        fn can_shortcut(_lhs: &BigInt, rhs: &BigInt) -> Option<Self::SC> {
            rhs.is_zero().then_some(())
        }

        fn do_shortcut<'b>(
            lhs: Boo<'b, BigInt>,
            rhs: Boo<'b, BigInt>,
            (): Self::SC,
        ) -> Moo<'b, BigInt> {
            match (lhs, rhs) {
                (Boo::BorrowedMut(lhs), rhs) => {
                    *lhs = rhs.cloned();
                    Moo::BorrowedMut(lhs)
                }
                (_, Boo::BorrowedMut(rhs)) => Moo::BorrowedMut(rhs),
                (_, rhs) => Moo::Owned(rhs.cloned()),
            }
        }
    }
    pub struct ByOne;
    impl MathShortcut for ByOne {
        type SC = ();

        fn can_shortcut(_lhs: &BigInt, rhs: &BigInt) -> Option<Self::SC> {
            rhs.is_abs_one().then_some(())
        }

        fn do_shortcut<'b>(
            lhs: Boo<'b, BigInt>,
            rhs: Boo<'b, BigInt>,
            (): Self::SC,
        ) -> Moo<'b, BigInt> {
            let signum = rhs.signum();
            let mut either = super::get_lhs(lhs, rhs);
            either.bytes *= signum;
            either
        }
    }
    pub struct ByPowerOfTwo;
    impl MathShortcut for ByPowerOfTwo {
        type SC = usize;

        fn can_shortcut(_lhs: &BigInt, rhs: &BigInt) -> Option<Self::SC> {
            rhs.ilog2()
        }

        fn do_shortcut<'b>(
            lhs: Boo<'b, BigInt>,
            rhs: Boo<'b, BigInt>,
            pow: Self::SC,
        ) -> Moo<'b, BigInt> {
            let signum = rhs.signum();
            let mut either = super::get_lhs(lhs, rhs);
            either.bytes *= signum;
            *either <<= &pow;
            either
        }
    }
}
