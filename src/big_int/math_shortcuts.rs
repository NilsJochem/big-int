#![allow(clippy::wildcard_imports)]
use super::BigInt;
use crate::boo::{Boo, Moo};

macro_rules! try_all {
    ($lhs:ident, $rhs:ident $(, )?) => {};
    ($lhs:ident, $rhs:ident, $($rule:tt)::*, $($tail:tt)*) => {
        if let Some(sc) = $($rule)::*::can_shortcut_lhs(&$lhs, &$rhs) {
            return $($rule)::*::do_shortcut_lhs($lhs, $rhs, sc);
        }
        if let Some(sc) = $($rule)::*::can_shortcut_rhs(&$lhs, &$rhs) {
            return $($rule)::*::do_shortcut_rhs($lhs, $rhs, sc);
        }
        math_shortcuts::try_all!($lhs, $rhs, $($tail)*);
    };
}
pub(crate) use try_all;

pub trait MathShortcut {
    /// Support type if testing that a shortcut can be applied gives additional information used in the calculation
    type SC;

    /// can the operation be made significantly easier by using special info about the lhs.
    /// For example 0 - x = -x
    fn can_shortcut_lhs(lhs: &BigInt, rhs: &BigInt) -> Option<Self::SC>;
    /// apply the shortcut with the special lhs
    fn do_shortcut_lhs<'b>(
        lhs: Boo<'b, BigInt>,
        rhs: Boo<'b, BigInt>,
        sc: Self::SC,
    ) -> Moo<'b, BigInt>;

    /// can the operation be made significantly easier by using special info about the rhs.
    /// For example x - 0 = x
    fn can_shortcut_rhs(lhs: &BigInt, rhs: &BigInt) -> Option<Self::SC>;
    /// apply the shortcut with the special rhs
    fn do_shortcut_rhs<'b>(
        lhs: Boo<'b, BigInt>,
        rhs: Boo<'b, BigInt>,
        sc: Self::SC,
    ) -> Moo<'b, BigInt>;
}
/// refers its methods directly to *_rhs and with flipped parametes for *_lhs
pub trait MathShortcutFlip {
    /// See `MathShortcut`
    type SC;
    /// can the operation be made significantly easier by using special info about one side.
    /// For example 0 + x = x = x + 0
    fn can_shortcut(lhs: &BigInt, rhs: &BigInt) -> Option<Self::SC>;
    /// apply the shortcut
    fn do_shortcut<'b>(lhs: Boo<'b, BigInt>, rhs: Boo<'b, BigInt>, sc: Self::SC)
        -> Moo<'b, BigInt>;
}
impl<Flip: MathShortcutFlip> MathShortcut for Flip {
    type SC = Flip::SC;
    fn can_shortcut_rhs(lhs: &BigInt, rhs: &BigInt) -> Option<Self::SC> {
        Self::can_shortcut(lhs, rhs)
    }
    fn do_shortcut_rhs<'b>(
        lhs: Boo<'b, BigInt>,
        rhs: Boo<'b, BigInt>,
        sc: Self::SC,
    ) -> Moo<'b, BigInt> {
        Self::do_shortcut(lhs, rhs, sc)
    }
    fn can_shortcut_lhs(lhs: &BigInt, rhs: &BigInt) -> Option<Self::SC> {
        Self::can_shortcut(rhs, lhs)
    }
    fn do_shortcut_lhs<'b>(
        lhs: Boo<'b, BigInt>,
        rhs: Boo<'b, BigInt>,
        sc: Self::SC,
    ) -> Moo<'b, BigInt> {
        Self::do_shortcut(rhs, lhs, sc)
    }
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
    pub struct Zero;
    impl MathShortcutFlip for Zero {
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
    pub struct Zero;
    impl MathShortcut for Zero {
        type SC = ();

        fn can_shortcut_lhs(lhs: &BigInt, _: &BigInt) -> Option<Self::SC> {
            lhs.is_zero().then_some(())
        }

        fn do_shortcut_lhs<'b>(
            lhs: Boo<'b, BigInt>,
            rhs: Boo<'b, BigInt>,
            (): Self::SC,
        ) -> Moo<'b, BigInt> {
            let mut either = super::get_lhs(lhs, rhs);
            either.negate();
            either
        }

        fn can_shortcut_rhs(_: &BigInt, rhs: &BigInt) -> Option<Self::SC> {
            rhs.is_zero().then_some(())
        }

        fn do_shortcut_rhs<'b>(
            lhs: Boo<'b, BigInt>,
            rhs: Boo<'b, BigInt>,
            (): Self::SC,
        ) -> Moo<'b, BigInt> {
            super::get_lhs(lhs, rhs)
        }
    }
}

pub mod mul {
    use super::*;
    pub struct ByZero;
    impl MathShortcutFlip for ByZero {
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
    impl MathShortcutFlip for ByOne {
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
    impl MathShortcutFlip for ByPowerOfTwo {
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
