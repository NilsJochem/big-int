#![allow(clippy::wildcard_imports)]
use super::{digits::Digit, BigInt};
#[allow(unused_imports)]
use crate::boo::{Boo, Moo};

macro_rules! try_all {
    ($lhs:ident, $rhs:ident $(, )?) => {};
    ($lhs:ident, $rhs:ident, $($rule:tt)::*, $($tail:tt)*) => {
        if let Some(sc) = <$($rule)::* as MathShortcut<math_shortcuts::Left>>::can_shortcut(&$lhs, &$rhs) {
            return <$($rule)::* as MathShortcut<math_shortcuts::Left>>::do_shortcut($lhs, $rhs, sc);
        }
        if let Some(sc) = <$($rule)::* as MathShortcut<math_shortcuts::Right>>::can_shortcut(&$lhs, &$rhs) {
            return <$($rule)::* as MathShortcut<math_shortcuts::Right>>::do_shortcut($lhs, $rhs, sc);
        }
        math_shortcuts::try_all!($lhs, $rhs, $($tail)*);
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

#[allow(dead_code)]
pub trait MathShortcutBoth {
    type SC;
    fn can_shortcut_lhs<D: Digit>(lhs: &BigInt<D>, rhs: &BigInt<D>) -> Option<Self::SC>;
    fn do_shortcut_lhs<'b, D: Digit>(
        lhs: Boo<'b, BigInt<D>>,
        rhs: Boo<'b, BigInt<D>>,
        sc: Self::SC,
    ) -> Moo<'b, BigInt<D>>;
    fn can_shortcut_rhs<D: Digit>(lhs: &BigInt<D>, rhs: &BigInt<D>) -> Option<Self::SC>;
    fn do_shortcut_rhs<'b, D: Digit>(
        lhs: Boo<'b, BigInt<D>>,
        rhs: Boo<'b, BigInt<D>>,
        sc: Self::SC,
    ) -> Moo<'b, BigInt<D>>;
}
impl<M> MathShortcutBoth for M
where
    M: MathShortcut<Left> + MathShortcut<Right, SC = <Self as MathShortcut<Left>>::SC>,
{
    type SC = <Self as MathShortcut<Left>>::SC;

    fn can_shortcut_lhs<D: Digit>(lhs: &BigInt<D>, rhs: &BigInt<D>) -> Option<Self::SC> {
        <Self as MathShortcut<Left>>::can_shortcut(lhs, rhs)
    }
    fn do_shortcut_lhs<'b, D: Digit>(
        lhs: Boo<'b, BigInt<D>>,
        rhs: Boo<'b, BigInt<D>>,
        sc: Self::SC,
    ) -> Moo<'b, BigInt<D>> {
        <Self as MathShortcut<Left>>::do_shortcut(lhs, rhs, sc)
    }
    fn can_shortcut_rhs<D: Digit>(lhs: &BigInt<D>, rhs: &BigInt<D>) -> Option<Self::SC> {
        <Self as MathShortcut<Right>>::can_shortcut(lhs, rhs)
    }
    fn do_shortcut_rhs<'b, D: Digit>(
        lhs: Boo<'b, BigInt<D>>,
        rhs: Boo<'b, BigInt<D>>,
        sc: Self::SC,
    ) -> Moo<'b, BigInt<D>> {
        <Self as MathShortcut<Right>>::do_shortcut(lhs, rhs, sc)
    }
}

pub trait MathShortcut<S: Side> {
    /// Support type if testing that a shortcut can be applied gives additional information used in the calculation
    type SC;

    /// can the operation be made significantly easier by using special info about the lhs.
    /// For example 0 - x = -x
    fn can_shortcut<D: Digit>(lhs: &BigInt<D>, rhs: &BigInt<D>) -> Option<Self::SC>;
    /// apply the shortcut with the special lhs
    fn do_shortcut<'b, D: Digit>(
        lhs: Boo<'b, BigInt<D>>,
        rhs: Boo<'b, BigInt<D>>,
        sc: Self::SC,
    ) -> Moo<'b, BigInt<D>>;
}
/// refers its methods directly to *_rhs and with flipped parametes for *_lhs
pub trait MathShortcutFlip {
    /// See `MathShortcut`
    type SC;
    /// can the operation be made significantly easier by using special info about one side.
    /// For example 0 + x = x = x + 0
    fn can_shortcut<D: Digit>(lhs: &BigInt<D>, rhs: &BigInt<D>) -> Option<Self::SC>;
    /// apply the shortcut
    fn do_shortcut<'b, D: Digit>(
        lhs: Boo<'b, BigInt<D>>,
        rhs: Boo<'b, BigInt<D>>,
        sc: Self::SC,
    ) -> Moo<'b, BigInt<D>>;
}
impl<Flip: MathShortcutFlip> MathShortcut<Right> for Flip {
    type SC = Flip::SC;
    fn can_shortcut<D: Digit>(lhs: &BigInt<D>, rhs: &BigInt<D>) -> Option<Self::SC> {
        Self::can_shortcut(lhs, rhs)
    }
    fn do_shortcut<'b, D: Digit>(
        lhs: Boo<'b, BigInt<D>>,
        rhs: Boo<'b, BigInt<D>>,
        sc: Self::SC,
    ) -> Moo<'b, BigInt<D>> {
        Self::do_shortcut(lhs, rhs, sc)
    }
}
impl<Flip: MathShortcutFlip> MathShortcut<Left> for Flip {
    type SC = Flip::SC;
    fn can_shortcut<D: Digit>(lhs: &BigInt<D>, rhs: &BigInt<D>) -> Option<Self::SC> {
        Self::can_shortcut(rhs, lhs)
    }
    fn do_shortcut<'b, D: Digit>(
        lhs: Boo<'b, BigInt<D>>,
        rhs: Boo<'b, BigInt<D>>,
        sc: Self::SC,
    ) -> Moo<'b, BigInt<D>> {
        Self::do_shortcut(rhs, lhs, sc)
    }
}

fn get_lhs<'b, D: Digit>(lhs: Boo<'b, BigInt<D>>, rhs: Boo<'b, BigInt<D>>) -> Moo<'b, BigInt<D>> {
    match (lhs, rhs) {
        (lhs, Boo::BorrowedMut(rhs)) => {
            *rhs = lhs.cloned();
            Moo::BorrowedMut(rhs)
        }
        (lhs, _) => Moo::<BigInt<D>>::from(lhs),
    }
}

pub mod add {
    use super::*;
    pub struct Zero;
    impl MathShortcutFlip for Zero {
        type SC = ();

        fn can_shortcut<D: Digit>(_lhs: &BigInt<D>, rhs: &BigInt<D>) -> Option<Self::SC> {
            rhs.is_zero().then_some(())
        }

        fn do_shortcut<'b, D: Digit>(
            lhs: Boo<'b, BigInt<D>>,
            rhs: Boo<'b, BigInt<D>>,
            (): Self::SC,
        ) -> Moo<'b, BigInt<D>> {
            super::get_lhs(lhs, rhs)
        }
    }
}

pub mod sub {
    use super::*;
    pub struct Zero;
    impl MathShortcut<Left> for Zero {
        type SC = ();

        fn can_shortcut<D: Digit>(lhs: &BigInt<D>, _: &BigInt<D>) -> Option<Self::SC> {
            lhs.is_zero().then_some(())
        }

        fn do_shortcut<'b, D: Digit>(
            lhs: Boo<'b, BigInt<D>>,
            rhs: Boo<'b, BigInt<D>>,
            (): Self::SC,
        ) -> Moo<'b, BigInt<D>> {
            let mut either = super::get_lhs(lhs, rhs);
            either.negate();
            either
        }
    }
    impl MathShortcut<Right> for Zero {
        type SC = ();
        fn can_shortcut<D: Digit>(_: &BigInt<D>, rhs: &BigInt<D>) -> Option<Self::SC> {
            rhs.is_zero().then_some(())
        }

        fn do_shortcut<'b, D: Digit>(
            lhs: Boo<'b, BigInt<D>>,
            rhs: Boo<'b, BigInt<D>>,
            (): Self::SC,
        ) -> Moo<'b, BigInt<D>> {
            super::get_lhs(lhs, rhs)
        }
    }
}

pub mod mul {
    use super::*;
    pub struct ByZero;
    impl MathShortcutFlip for ByZero {
        type SC = ();

        fn can_shortcut<D: Digit>(_lhs: &BigInt<D>, rhs: &BigInt<D>) -> Option<Self::SC> {
            rhs.is_zero().then_some(())
        }

        fn do_shortcut<'b, D: Digit>(
            lhs: Boo<'b, BigInt<D>>,
            rhs: Boo<'b, BigInt<D>>,
            (): Self::SC,
        ) -> Moo<'b, BigInt<D>> {
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

        fn can_shortcut<D: Digit>(_lhs: &BigInt<D>, rhs: &BigInt<D>) -> Option<Self::SC> {
            rhs.is_abs_one().then_some(())
        }

        fn do_shortcut<'b, D: Digit>(
            lhs: Boo<'b, BigInt<D>>,
            rhs: Boo<'b, BigInt<D>>,
            (): Self::SC,
        ) -> Moo<'b, BigInt<D>> {
            let signum = rhs.signum;
            let mut either = super::get_lhs(lhs, rhs);
            either.signum *= signum;
            either
        }
    }
    pub struct ByPowerOfTwo;
    impl MathShortcutFlip for ByPowerOfTwo {
        type SC = ();

        fn can_shortcut<D: Digit>(_lhs: &BigInt<D>, rhs: &BigInt<D>) -> Option<Self::SC> {
            rhs.is_power_of_two().then_some(())
        }

        fn do_shortcut<'b, D: Digit>(
            lhs: Boo<'b, BigInt<D>>,
            rhs: Boo<'b, BigInt<D>>,
            (): Self::SC,
        ) -> Moo<'b, BigInt<D>> {
            let signum = rhs.signum;
            let pow = rhs.digits(2) - 1;
            let mut either = super::get_lhs(lhs, rhs);
            either.signum *= signum;
            *either <<= pow;
            either
        }
    }
}
pub mod div {
    use super::*;
    pub struct ByPowerOfTwo;
    impl MathShortcutFlip for ByPowerOfTwo {
        type SC = ();

        fn can_shortcut<D: Digit>(_lhs: &BigInt<D>, rhs: &BigInt<D>) -> Option<Self::SC> {
            rhs.is_power_of_two().then_some(())
        }

        fn do_shortcut<'b, D: Digit>(
            lhs: Boo<'b, BigInt<D>>,
            rhs: Boo<'b, BigInt<D>>,
            (): Self::SC,
        ) -> Moo<'b, BigInt<D>> {
            let signum = rhs.signum;
            let pow = rhs.digits(2) - 1;
            let mut either = super::get_lhs(lhs, rhs);
            either.signum *= signum;
            *either <<= &pow;
            either
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::needless_pass_by_value)]
    fn can_shorcut<M, D: Digit>(
        lhs: impl Into<BigInt<D>>,
        rhs: impl Into<BigInt<D>>,
        l_result: Option<M::SC>,
        r_result: Option<M::SC>,
    ) where
        M: MathShortcutBoth,
        M::SC: PartialEq + std::fmt::Debug,
    {
        let lhs = lhs.into();
        let rhs = rhs.into();
        assert_eq!(M::can_shortcut_lhs(&lhs, &rhs), l_result, "lhs",);
        assert_eq!(M::can_shortcut_rhs(&lhs, &rhs), r_result, "rhs",);
    }
    fn test_shorcut<M, S: Side, D: Digit>(
        lhs: impl Into<BigInt<D>>,
        rhs: impl Into<BigInt<D>>,
        sc: M::SC,
        result: impl Into<BigInt<D>>,
        op_dbg: &str,
    ) where
        M: MathShortcut<S>,
        M::SC: Clone,
    {
        crate::big_int::tests::big_math::test_op(
            lhs,
            rhs,
            |lhs, rhs| M::do_shortcut(lhs, rhs, sc.clone()),
            result,
            op_dbg,
        );
    }
    fn test_shorcut_commte<M, S: Side, D: Digit>(
        lhs: impl Into<BigInt<D>>,
        rhs: impl Into<BigInt<D>>,
        sc: M::SC,
        result: impl Into<BigInt<D>>,
        op_dbg: &str,
    ) where
        M: MathShortcut<S>,
        M::SC: Clone,
    {
        crate::big_int::tests::big_math::test_op_commute(
            lhs,
            rhs,
            |lhs, rhs| M::do_shortcut(lhs, rhs, sc.clone()),
            result,
            op_dbg,
        );
    }

    mod t_add {
        use super::*;
        const OP_DBG: &str = "+";

        const NON_ZERO: u8 = 42;
        #[test]
        fn can_use_shortcut_both_zero() {
            can_shorcut::<add::Zero, u32>(0, 0, Some(()), Some(()));
        }
        #[test]
        fn can_use_shortcut_rhs_zero() {
            can_shorcut::<add::Zero, u32>(NON_ZERO, 0, None, Some(()));
        }
        #[test]
        fn can_use_shortcut_lhs_zero() {
            can_shorcut::<add::Zero, u32>(0, NON_ZERO, Some(()), None);
        }
        #[test]
        fn can_use_shortcut_none_zero() {
            can_shorcut::<add::Zero, u32>(NON_ZERO, NON_ZERO, None, None);
        }

        #[test]
        fn use_shortcut_lhs_zero() {
            test_shorcut::<add::Zero, Left, u32>(0, NON_ZERO, (), NON_ZERO, OP_DBG);
        }
        #[test]
        fn use_shortcut_rhs_zero() {
            test_shorcut::<add::Zero, Right, u32>(NON_ZERO, 0, (), NON_ZERO, OP_DBG);
        }
        #[test]
        fn use_shortcut_both_zero() {
            test_shorcut::<add::Zero, Left, u32>(0, 0, (), 0, OP_DBG);
            test_shorcut::<add::Zero, Right, u32>(0, 0, (), 0, OP_DBG);
        }
    }
    mod t_sub {
        const OP_DBG: &str = "-";
        use super::*;

        const NON_ZERO: i8 = 42;

        #[test]
        fn can_use_shortcut_both_zero() {
            can_shorcut::<sub::Zero, u32>(0, 0, Some(()), Some(()));
        }
        #[test]
        fn can_use_shortcut_rhs_zero() {
            can_shorcut::<sub::Zero, u32>(NON_ZERO, 0, None, Some(()));
        }
        #[test]
        fn can_use_shortcut_lhs_zero() {
            can_shorcut::<sub::Zero, u32>(0, NON_ZERO, Some(()), None);
        }
        #[test]
        fn can_use_shortcut_none_zero() {
            can_shorcut::<sub::Zero, u32>(NON_ZERO, NON_ZERO, None, None);
        }

        #[test]
        fn use_shortcut_lhs_zero() {
            test_shorcut::<sub::Zero, Left, u32>(NON_ZERO, 0, (), -NON_ZERO, OP_DBG);
        }
        #[test]
        fn use_shortcut_rhs_zero() {
            test_shorcut::<sub::Zero, Right, u32>(NON_ZERO, 0, (), NON_ZERO, OP_DBG);
        }
        #[test]
        fn use_shortcut_both_zero() {
            test_shorcut::<sub::Zero, Left, u32>(0, 0, (), 0, OP_DBG);
            test_shorcut::<sub::Zero, Right, u32>(0, 0, (), 0, OP_DBG);
        }
    }

    mod t_mul {
        const OP_DBG: &str = "*";
        mod zero {
            use super::super::*;

            const NON_ZERO: u8 = 42;

            #[test]
            fn can_use_shortcut_both_zero() {
                can_shorcut::<mul::ByZero, u32>(0, 0, Some(()), Some(()));
            }
            #[test]
            fn can_use_shortcut_rhs_zero() {
                can_shorcut::<mul::ByZero, u32>(NON_ZERO, 0, None, Some(()));
            }
            #[test]
            fn can_use_shortcut_lhs_zero() {
                can_shorcut::<mul::ByZero, u32>(0, NON_ZERO, Some(()), None);
            }
            #[test]
            fn can_use_shortcut_none_zero() {
                can_shorcut::<mul::ByZero, u32>(NON_ZERO, NON_ZERO, None, None);
            }

            #[test]
            fn use_shortcut_lhs_zero() {
                test_shorcut::<mul::ByZero, Left, u32>(0, NON_ZERO, (), 0, super::OP_DBG);
            }
            #[test]
            fn use_shortcut_rhs_zero() {
                test_shorcut::<mul::ByZero, Right, u32>(NON_ZERO, 0, (), 0, super::OP_DBG);
            }
            #[test]
            fn use_shortcut_both_zero() {
                test_shorcut::<mul::ByZero, Left, u32>(0, 0, (), 0, super::OP_DBG);
                test_shorcut::<mul::ByZero, Right, u32>(0, 0, (), 0, super::OP_DBG);
            }
        }
        mod one {
            use super::super::*;

            const NON_ONE: u8 = 42;

            #[test]
            fn can_use_shortcut_both_one() {
                can_shorcut::<mul::ByOne, u32>(1, 1, Some(()), Some(()));
            }
            #[test]
            fn can_use_shortcut_rhs_one() {
                can_shorcut::<mul::ByOne, u32>(NON_ONE, 1, None, Some(()));
            }
            #[test]
            fn can_use_shortcut_lhs_one() {
                can_shorcut::<mul::ByOne, u32>(1, NON_ONE, Some(()), None);
            }
            #[test]
            fn can_use_shortcut_none_one() {
                can_shorcut::<mul::ByOne, u32>(NON_ONE, NON_ONE, None, None);
            }

            #[test]
            fn use_shortcut_lhs_one() {
                test_shorcut::<mul::ByOne, Left, u32>(1, NON_ONE, (), NON_ONE, super::OP_DBG);
            }
            #[test]
            fn use_shortcut_rhs_one() {
                test_shorcut::<mul::ByOne, Right, u32>(NON_ONE, 1, (), NON_ONE, super::OP_DBG);
            }
            #[test]
            fn use_shortcut_both_one() {
                test_shorcut::<mul::ByOne, Left, u32>(1, 1, (), 1, super::OP_DBG);
                test_shorcut::<mul::ByOne, Right, u32>(1, 1, (), 1, super::OP_DBG);
            }
        }

        mod pow2 {

            use super::super::*;

            const POW: usize = 7;
            const POW2: u8 = 0b1000_0000;
            const NON_POW2: u8 = 42;

            #[test]
            fn can_use_shortcut_both() {
                can_shorcut::<mul::ByPowerOfTwo, u32>(POW2, POW2, Some(()), Some(()));
            }
            #[test]
            fn can_use_shortcut_rhs() {
                can_shorcut::<mul::ByPowerOfTwo, u32>(NON_POW2, POW2, None, Some(()));
            }
            #[test]
            fn can_use_shortcut_lhs() {
                can_shorcut::<mul::ByPowerOfTwo, u32>(POW2, NON_POW2, Some(()), None);
            }
            #[test]
            fn can_use_shortcut_none() {
                can_shorcut::<mul::ByPowerOfTwo, u32>(NON_POW2, NON_POW2, None, None);
            }

            #[test]
            fn use_shortcut_lhs() {
                test_shorcut::<mul::ByPowerOfTwo, Left, u32>(
                    POW2,
                    NON_POW2,
                    (),
                    (NON_POW2 as u32) << (POW as u32),
                    super::OP_DBG,
                );
            }
            #[test]
            fn use_shortcut_rhs() {
                test_shorcut::<mul::ByPowerOfTwo, Right, u32>(
                    NON_POW2,
                    POW2,
                    (),
                    (NON_POW2 as u32) << (POW as u32),
                    super::OP_DBG,
                );
            }
            #[test]
            fn use_shortcut_both() {
                test_shorcut::<mul::ByPowerOfTwo, Left, u32>(
                    POW2,
                    POW2,
                    (),
                    (POW2 as u32) << (POW as u32),
                    super::OP_DBG,
                );
                test_shorcut::<mul::ByPowerOfTwo, Right, u32>(
                    POW2,
                    POW2,
                    (),
                    (POW2 as u32) << (POW as u32),
                    super::OP_DBG,
                );
            }
            #[test]
            fn mul_sign_pow_two() {
                test_shorcut_commte::<mul::ByPowerOfTwo, Left, u32>(2, 2, (), 4, super::OP_DBG);
                test_shorcut_commte::<mul::ByPowerOfTwo, Left, u32>(-2, 2, (), -4, super::OP_DBG);
                test_shorcut_commte::<mul::ByPowerOfTwo, Left, u32>(2, -2, (), -4, super::OP_DBG);
                test_shorcut_commte::<mul::ByPowerOfTwo, Left, u32>(-2, -2, (), 4, super::OP_DBG);
            }
        }
    }
}
