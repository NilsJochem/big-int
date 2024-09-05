#![allow(clippy::wildcard_imports)]
// SPDX-FileCopyrightText: 2024 Nils Jochem
// SPDX-License-Identifier: MPL-2.0
use crate::util::boo::{Mob, Moo};
use crate::{big_int::digits::Digit, BigUInt};

macro_rules! try_all {
    ($lhs:ident, $rhs:ident $(, )?) => {};
    ($lhs:ident, $rhs:ident, $($rule:tt)::*, $($tail:tt)*) => {
        if <$($rule)::* as MathShortcut<super::math_shortcuts::Left>>::can_shortcut(&$lhs, &$rhs) {
            return <$($rule)::* as MathShortcut<super::math_shortcuts::Left>>::do_shortcut($lhs, $rhs);
        }
        if <$($rule)::* as MathShortcut<super::math_shortcuts::Right>>::can_shortcut(&$lhs, &$rhs) {
            return <$($rule)::* as MathShortcut<super::math_shortcuts::Right>>::do_shortcut($lhs, $rhs);
        }
        super::math_shortcuts::try_all!($lhs, $rhs, $($tail)*);
    };
    ($lhs:ident, $rhs:ident, left $($rule:tt)::*, $($tail:tt)*) => {
        if <$($rule)::* as MathShortcut<super::math_shortcuts::Left>>::can_shortcut(&$lhs, &$rhs) {
            return <$($rule)::* as MathShortcut<super::math_shortcuts::Left>>::do_shortcut($lhs, $rhs);
        }
        super::math_shortcuts::try_all!($lhs, $rhs, $($tail)*);
    };
    ($lhs:ident, $rhs:ident, right $($rule:tt)::*, $($tail:tt)*) => {
        if <$($rule)::* as MathShortcut<super::math_shortcuts::Right>>::can_shortcut(&$lhs, &$rhs) {
            return <$($rule)::* as MathShortcut<super::math_shortcuts::Right>>::do_shortcut($lhs, $rhs);
        }
        super::math_shortcuts::try_all!($lhs, $rhs, $($tail)*);
    };
}
pub(crate) use try_all;

pub trait Side {
    #[allow(dead_code)]
    fn select<T>(l: T, r: T) -> T;
}
pub struct Left;
impl Side for Left {
    fn select<T>(l: T, _: T) -> T {
        l
    }
}
pub struct Right;
impl Side for Right {
    fn select<T>(_: T, r: T) -> T {
        r
    }
}

pub trait MathShortcut<S: Side> {
    type RES<'b, D: 'b>;

    /// can the operation be made significantly easier by using special info about the lhs.
    /// For example 0 - x = -x
    fn can_shortcut<D: Digit>(lhs: &BigUInt<D>, rhs: &BigUInt<D>) -> bool;
    /// apply the shortcut with the special lhs
    fn do_shortcut<'b, D: Digit>(
        lhs: Mob<'b, BigUInt<D>>,
        rhs: Mob<'b, BigUInt<D>>,
    ) -> Self::RES<'b, D>;
}
/// refers its methods directly to *_rhs and with flipped parametes for *_lhs
pub trait MathShortcutFlip {
    /// can the operation be made significantly easier by using special info about one side.
    /// For example 0 + x = x = x + 0
    fn can_shortcut<D: Digit>(lhs: &BigUInt<D>, rhs: &BigUInt<D>) -> bool;
    /// apply the shortcut
    fn do_shortcut<'b, D: Digit>(
        lhs: Mob<'b, BigUInt<D>>,
        rhs: Mob<'b, BigUInt<D>>,
    ) -> Moo<'b, BigUInt<D>>;
}
impl<Flip: MathShortcutFlip> MathShortcut<Right> for Flip {
    type RES<'b, D: 'b> = Moo<'b, BigUInt<D>>;

    fn can_shortcut<D: Digit>(lhs: &BigUInt<D>, rhs: &BigUInt<D>) -> bool {
        Self::can_shortcut(lhs, rhs)
    }
    fn do_shortcut<'b, D: Digit>(
        lhs: Mob<'b, BigUInt<D>>,
        rhs: Mob<'b, BigUInt<D>>,
    ) -> Moo<'b, BigUInt<D>> {
        Self::do_shortcut(lhs, rhs)
    }
}
impl<Flip: MathShortcutFlip> MathShortcut<Left> for Flip {
    type RES<'b, D: 'b> = Moo<'b, BigUInt<D>>;

    fn can_shortcut<D: Digit>(lhs: &BigUInt<D>, rhs: &BigUInt<D>) -> bool {
        Self::can_shortcut(rhs, lhs)
    }
    fn do_shortcut<'b, D: Digit>(
        lhs: Mob<'b, BigUInt<D>>,
        rhs: Mob<'b, BigUInt<D>>,
    ) -> Moo<'b, BigUInt<D>> {
        Self::do_shortcut(rhs, lhs)
    }
}

pub(super) fn get_lhs<'b, B: Clone>(lhs: Mob<'b, B>, rhs: Mob<'b, B>) -> Moo<'b, B> {
    match (lhs, rhs) {
        (lhs, Mob::BorrowedMut(rhs)) => {
            *rhs = lhs.cloned();
            Moo::BorrowedMut(rhs)
        }
        (lhs, _) => Moo::<B>::from_mob_cloned(lhs),
    }
}

pub mod add {
    use super::*;
    pub struct Zero;
    impl MathShortcutFlip for Zero {
        fn can_shortcut<D: Digit>(_lhs: &BigUInt<D>, rhs: &BigUInt<D>) -> bool {
            rhs.is_zero()
        }

        fn do_shortcut<'b, D: Digit>(
            lhs: Mob<'b, BigUInt<D>>,
            rhs: Mob<'b, BigUInt<D>>,
        ) -> Moo<'b, BigUInt<D>> {
            super::get_lhs(lhs, rhs)
        }
    }
}

pub mod sub {
    use super::*;
    pub struct Zero;
    impl MathShortcut<Right> for Zero {
        type RES<'b, D: 'b> = Moo<'b, BigUInt<D>>;

        fn can_shortcut<D: Digit>(_: &BigUInt<D>, rhs: &BigUInt<D>) -> bool {
            rhs.is_zero()
        }

        fn do_shortcut<'b, D: Digit>(
            lhs: Mob<'b, BigUInt<D>>,
            rhs: Mob<'b, BigUInt<D>>,
        ) -> Moo<'b, BigUInt<D>> {
            super::get_lhs(lhs, rhs)
        }
    }
}

pub mod mul {
    use crate::big_int::AnyBigIntRef;

    use super::*;
    pub struct ByZero;
    impl MathShortcutFlip for ByZero {
        fn can_shortcut<D: Digit>(_lhs: &BigUInt<D>, rhs: &BigUInt<D>) -> bool {
            rhs.is_zero()
        }

        fn do_shortcut<'b, D: Digit>(
            lhs: Mob<'b, BigUInt<D>>,
            rhs: Mob<'b, BigUInt<D>>,
        ) -> Moo<'b, BigUInt<D>> {
            match (lhs, rhs) {
                (Mob::BorrowedMut(lhs), rhs) => {
                    *lhs = rhs.cloned();
                    Moo::BorrowedMut(lhs)
                }
                (_, Mob::BorrowedMut(rhs)) => Moo::BorrowedMut(rhs),
                (_, rhs) => Moo::Owned(rhs.cloned()),
            }
        }
    }
    pub struct ByOne;
    impl MathShortcutFlip for ByOne {
        fn can_shortcut<D: Digit>(_lhs: &BigUInt<D>, rhs: &BigUInt<D>) -> bool {
            rhs.is_one()
        }

        fn do_shortcut<'b, D: Digit>(
            lhs: Mob<'b, BigUInt<D>>,
            rhs: Mob<'b, BigUInt<D>>,
        ) -> Moo<'b, BigUInt<D>> {
            super::get_lhs(lhs, rhs)
        }
    }
    pub struct ByPowerOfTwo;
    impl MathShortcutFlip for ByPowerOfTwo {
        fn can_shortcut<D: Digit>(_lhs: &BigUInt<D>, rhs: &BigUInt<D>) -> bool {
            rhs.is_power_of_two()
        }

        fn do_shortcut<'b, D: Digit>(
            lhs: Mob<'b, BigUInt<D>>,
            rhs: Mob<'b, BigUInt<D>>,
        ) -> Moo<'b, BigUInt<D>> {
            let pow = rhs.digits(2) - 1;
            let mut either = super::get_lhs(lhs, rhs);
            *either <<= pow;
            either
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::needless_pass_by_value)]
    fn can_shorcut<M, D: Digit>(
        lhs: impl Into<BigUInt<D>>,
        rhs: impl Into<BigUInt<D>>,
        l_result: bool,
        r_result: bool,
    ) where
        M: MathShortcut<Left> + MathShortcut<Right>,
    {
        let lhs = lhs.into();
        let rhs = rhs.into();
        assert_eq!(
            <M as MathShortcut<Left>>::can_shortcut(&lhs, &rhs),
            l_result,
            "lhs",
        );
        assert_eq!(
            <M as MathShortcut<Right>>::can_shortcut(&lhs, &rhs),
            r_result,
            "rhs",
        );
    }
    fn test_shorcut<M, S: Side, D: Digit + 'static>(
        lhs: impl Into<BigUInt<D>>,
        rhs: impl Into<BigUInt<D>>,
        result: impl Into<BigUInt<D>>,
        op_dbg: &str,
    ) where
        for<'b> M: MathShortcut<S, RES<'b, D> = Moo<'b, BigUInt<D>>>,
    {
        crate::big_int::tests::big_math::test_op(
            lhs,
            rhs,
            |lhs, rhs| M::do_shortcut(lhs, rhs),
            result,
            op_dbg,
            crate::big_int::tests::big_math::Side::Both,
        );
    }
    fn test_shorcut_commte<M, S: Side, D: Digit + 'static>(
        lhs: impl Into<BigUInt<D>>,
        rhs: impl Into<BigUInt<D>>,
        result: impl Into<BigUInt<D>>,
        op_dbg: &str,
    ) where
        for<'b> M: MathShortcut<S, RES<'b, D> = Moo<'b, BigUInt<D>>>,
    {
        crate::big_int::tests::big_math::test_op_commute(
            lhs,
            rhs,
            |lhs, rhs| M::do_shortcut(lhs, rhs),
            result,
            op_dbg,
        );
    }

    mod t_add {
        use super::*;
        const OP_DBG: &str = "+";

        const NON_ZERO: u8 = 42;
        const ZERO: u8 = 0;
        #[test]
        fn can_use_shortcut_both_zero() {
            can_shorcut::<add::Zero, u32>(ZERO, ZERO, true, true);
        }
        #[test]
        fn can_use_shortcut_rhs_zero() {
            can_shorcut::<add::Zero, u32>(NON_ZERO, ZERO, false, true);
        }
        #[test]
        fn can_use_shortcut_lhs_zero() {
            can_shorcut::<add::Zero, u32>(ZERO, NON_ZERO, true, false);
        }
        #[test]
        fn can_use_shortcut_none_zero() {
            can_shorcut::<add::Zero, u32>(NON_ZERO, NON_ZERO, false, false);
        }

        #[test]
        fn use_shortcut_lhs_zero() {
            test_shorcut::<add::Zero, Left, u32>(ZERO, NON_ZERO, NON_ZERO, OP_DBG);
        }
        #[test]
        fn use_shortcut_rhs_zero() {
            test_shorcut::<add::Zero, Right, u32>(NON_ZERO, ZERO, NON_ZERO, OP_DBG);
        }
        #[test]
        fn use_shortcut_both_zero() {
            test_shorcut::<add::Zero, Left, u32>(ZERO, ZERO, ZERO, OP_DBG);
            test_shorcut::<add::Zero, Right, u32>(ZERO, ZERO, ZERO, OP_DBG);
        }
    }

    mod t_mul {
        const OP_DBG: &str = "*";
        mod zero {
            use super::super::*;

            const ZERO: u8 = 0;
            const NON_ZERO: u8 = 42;

            #[test]
            fn can_use_shortcut_both_zero() {
                can_shorcut::<mul::ByZero, u32>(ZERO, ZERO, true, true);
            }
            #[test]
            fn can_use_shortcut_rhs_zero() {
                can_shorcut::<mul::ByZero, u32>(NON_ZERO, ZERO, false, true);
            }
            #[test]
            fn can_use_shortcut_lhs_zero() {
                can_shorcut::<mul::ByZero, u32>(ZERO, NON_ZERO, true, false);
            }
            #[test]
            fn can_use_shortcut_none_zero() {
                can_shorcut::<mul::ByZero, u32>(NON_ZERO, NON_ZERO, false, false);
            }

            #[test]
            fn use_shortcut_lhs_zero() {
                test_shorcut::<mul::ByZero, Left, u32>(ZERO, NON_ZERO, ZERO, super::OP_DBG);
            }
            #[test]
            fn use_shortcut_rhs_zero() {
                test_shorcut::<mul::ByZero, Right, u32>(NON_ZERO, ZERO, ZERO, super::OP_DBG);
            }
            #[test]
            fn use_shortcut_both_zero() {
                test_shorcut::<mul::ByZero, Left, u32>(ZERO, ZERO, ZERO, super::OP_DBG);
                test_shorcut::<mul::ByZero, Right, u32>(ZERO, ZERO, ZERO, super::OP_DBG);
            }
        }
        mod one {
            use super::super::*;

            const ONE: u8 = 1;
            const NON_ONE: u8 = 42;

            #[test]
            fn can_use_shortcut_both_one() {
                can_shorcut::<mul::ByOne, u32>(ONE, ONE, true, true);
            }
            #[test]
            fn can_use_shortcut_rhs_one() {
                can_shorcut::<mul::ByOne, u32>(NON_ONE, ONE, false, true);
            }
            #[test]
            fn can_use_shortcut_lhs_one() {
                can_shorcut::<mul::ByOne, u32>(ONE, NON_ONE, true, false);
            }
            #[test]
            fn can_use_shortcut_none_one() {
                can_shorcut::<mul::ByOne, u32>(NON_ONE, NON_ONE, false, false);
            }

            #[test]
            fn use_shortcut_lhs_one() {
                test_shorcut::<mul::ByOne, Left, u32>(ONE, NON_ONE, NON_ONE, super::OP_DBG);
            }
            #[test]
            fn use_shortcut_rhs_one() {
                test_shorcut::<mul::ByOne, Right, u32>(NON_ONE, ONE, NON_ONE, super::OP_DBG);
            }
            #[test]
            fn use_shortcut_both_one() {
                test_shorcut::<mul::ByOne, Left, u32>(ONE, ONE, ONE, super::OP_DBG);
                test_shorcut::<mul::ByOne, Right, u32>(ONE, ONE, ONE, super::OP_DBG);
            }
        }

        mod pow2 {

            use super::super::*;

            const POW: usize = 7;
            const POW2: u8 = 0b1000_0000;
            const NON_POW2: u8 = 42;

            #[test]
            fn can_use_shortcut_both() {
                can_shorcut::<mul::ByPowerOfTwo, u32>(POW2, POW2, true, true);
            }
            #[test]
            fn can_use_shortcut_rhs() {
                can_shorcut::<mul::ByPowerOfTwo, u32>(NON_POW2, POW2, false, true);
            }
            #[test]
            fn can_use_shortcut_lhs() {
                can_shorcut::<mul::ByPowerOfTwo, u32>(POW2, NON_POW2, true, false);
            }
            #[test]
            fn can_use_shortcut_none() {
                can_shorcut::<mul::ByPowerOfTwo, u32>(NON_POW2, NON_POW2, false, false);
            }

            #[test]
            fn use_shortcut_lhs() {
                test_shorcut::<mul::ByPowerOfTwo, Left, u32>(
                    POW2,
                    NON_POW2,
                    (NON_POW2 as u32) << (POW as u32),
                    super::OP_DBG,
                );
            }
            #[test]
            fn use_shortcut_rhs() {
                test_shorcut::<mul::ByPowerOfTwo, Right, u32>(
                    NON_POW2,
                    POW2,
                    (NON_POW2 as u32) << (POW as u32),
                    super::OP_DBG,
                );
            }
            #[test]
            fn use_shortcut_both() {
                test_shorcut::<mul::ByPowerOfTwo, Left, u32>(
                    POW2,
                    POW2,
                    (POW2 as u32) << (POW as u32),
                    super::OP_DBG,
                );
                test_shorcut::<mul::ByPowerOfTwo, Right, u32>(
                    POW2,
                    POW2,
                    (POW2 as u32) << (POW as u32),
                    super::OP_DBG,
                );
            }
            #[test]
            fn mul_sign_pow_two() {
                test_shorcut_commte::<mul::ByPowerOfTwo, Left, u32>(2, 2, 4, super::OP_DBG);
            }
        }
    }
}
