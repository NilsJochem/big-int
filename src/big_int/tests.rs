use super::*;
use tests::digits::{FullSize, HalfSize};
mod create {
    use super::*;
    #[test]
    fn cast_signum() {
        for i in i8::MIN..=i8::MAX {
            // Safety: explicitly testing, that the correct and only the correct values are cast
            unsafe {
                assert_eq!(
                    SigNum::from_i8(i) == SigNum::Negative,
                    i == -1,
                    "{i} failed to be/not be Negative"
                );
                assert_eq!(
                    SigNum::from_i8(i) == SigNum::Zero,
                    i == 0,
                    "{i} failed to be/not be Zero"
                );
                assert_eq!(
                    SigNum::from_i8(i) == SigNum::Positive,
                    i == 1,
                    "{i} failed to be/not be Positive"
                );
            }
        }
    }
    #[test]
    fn from_u32s() {
        assert_eq!(
            BigInt::from_iter([0x3322_1100_u32, 0x7766_5544, 0x9988]),
            BigInt {
                signum: SigNum::Positive,
                digits: vec![
                    HalfSize::from(0x3322_1100),
                    HalfSize::from(0x7766_5544),
                    HalfSize::from(0x0000_9988)
                ]
            }
        );
    }
    #[test]
    fn from_i128() {
        assert_eq!(
            BigInt::from(-0x9988_7766_5544_3322_1100_i128),
            BigInt {
                signum: SigNum::Negative,
                digits: vec![
                    HalfSize::from(0x3322_1100),
                    HalfSize::from(0x7766_5544),
                    HalfSize::from(0x0000_9988)
                ]
            }
        );
    }
}
mod output {
    use super::*;
    #[test]
    fn debug() {
        assert_eq!(
            format!(
                "{:?}",
                BigInt::<HalfSize>::from(0x0000_7766_0000_5544_3322_1100_u128)
            ),
            "Number { + 0x[00007766, 00005544, 33221100]}"
        );
    }
    #[test]
    fn lower_hex() {
        assert_eq!(
            format!(
                "{:x}",
                BigInt::<HalfSize>::from(0x9988_7766_5544_3322_1100_u128)
            ),
            "99887766554433221100"
        );
        assert_eq!(
            format!(
                "{:#x}",
                BigInt::<HalfSize>::from(0x9988_7766_5544_3322_1100_u128)
            ),
            "0x99887766554433221100"
        );

        assert_eq!(
            format!(
                "{:x}",
                BigInt::<HalfSize>::from(-0x9988_7766_5544_3322_1100_i128)
            ),
            "-99887766554433221100"
        );
        assert_eq!(
            format!(
                "{:#x}",
                BigInt::<HalfSize>::from(-0x9988_7766_5544_3322_1100_i128)
            ),
            "-0x99887766554433221100"
        );

        assert_eq!(
            format!(
                "{:0>32x}",
                BigInt::<HalfSize>::from(0x9988_7766_5544_3322_1100_u128)
            ),
            "00000000000099887766554433221100"
        );

        assert_eq!(
            format!(
                "{:#032x}",
                BigInt::<HalfSize>::from(0x9988_7766_5544_3322_1100_u128)
            ),
            "0x000000000099887766554433221100"
        );

        assert_eq!(
            format!(
                "{:#032x}",
                BigInt::<HalfSize>::from(0x00ee_ddcc_bbaa_9988_7766_5544_3322_1100_u128)
            ),
            "0xeeddccbbaa99887766554433221100"
        );
        assert_eq!(
            format!(
                "{:#032X}",
                BigInt::<HalfSize>::from(0x00ee_ddcc_bbaa_9988_7766_5544_3322_1100_u128)
            ),
            "0XEEDDCCBBAA99887766554433221100"
        );
    }
}
mod order {
    use std::cmp::Ordering;

    use super::*;
    #[test]
    fn same() {
        assert_eq!(
            BigInt::<HalfSize>::from(0x9988_7766_5544_3322_1100_u128)
                .cmp(&BigInt::<HalfSize>::from(0x9988_7766_5544_3322_1100_u128)),
            Ordering::Equal
        );
        assert_eq!(
            BigInt::<HalfSize>::from(-0x9988_7766_5544_3322_1100_i128)
                .cmp(&BigInt::<HalfSize>::from(-0x9988_7766_5544_3322_1100_i128)),
            Ordering::Equal
        );
    }
    #[test]
    fn negated() {
        assert_eq!(
            BigInt::<HalfSize>::from(0x9988_7766_5544_3322_1100_u128)
                .cmp(&BigInt::<HalfSize>::from(-0x9988_7766_5544_3322_1100_i128)),
            Ordering::Greater
        );
        assert_eq!(
            BigInt::<HalfSize>::from(-0x9988_7766_5544_3322_1100_i128)
                .cmp(&BigInt::<HalfSize>::from(0x9988_7766_5544_3322_1100_i128)),
            Ordering::Less
        );
    }
    #[test]
    fn middle_diff() {
        assert_eq!(
            BigInt::<HalfSize>::from(0x9988_8866_5544_3322_1100_u128)
                .cmp(&BigInt::<HalfSize>::from(0x9988_7766_5544_3322_1100_i128)),
            Ordering::Greater
        );
        assert_eq!(
            BigInt::<HalfSize>::from(0x9988_7766_5544_3322_1100_i128)
                .cmp(&BigInt::<HalfSize>::from(0x9988_8866_5544_3322_1100_i128)),
            Ordering::Less
        );
    }
    #[test]
    fn same_len() {
        assert_eq!(
            BigInt::<HalfSize>::from(0x0fff_ffff_ffff_ffff_ffff_u128)
                .cmp(&BigInt::<HalfSize>::from(0x9988_7766_5544_3322_1100_i128)),
            Ordering::Less
        );
    }

    #[test]
    fn differnd_len() {
        assert_eq!(
            BigInt::<HalfSize>::from(0x7766_5544_3322_1100u64)
                .cmp(&BigInt::<HalfSize>::from(0x0001_0000_0000_0000_0000u128)),
            Ordering::Less
        )
    }
}
mod full_size {
    use super::*;

    #[test]
    fn load() {
        assert_eq!(
            FullSize::from(0x7766_5544_3322_1100_usize),
            FullSize::new(HalfSize::from(0x3322_1100), HalfSize::from(0x7766_5544))
        );
    }

    #[test]
    fn read() {
        assert_eq!(
            FullSize::from(0x7766_5544_3322_1100_usize).lower(),
            HalfSize::from(0x3322_1100)
        );
        assert_eq!(
            FullSize::from(0x7766_5544_3322_1100_usize).higher(),
            HalfSize::from(0x7766_5544)
        );
    }
}
pub(super) mod big_math {

    use super::*;
    pub fn test_op_commute<D: Digit>(
        lhs: impl Into<BigInt<D>>,
        rhs: impl Into<BigInt<D>>,
        op: impl for<'b> Fn(Boo<'b, BigInt<D>>, Boo<'b, BigInt<D>>) -> Moo<'b, BigInt<D>>,
        result: impl Into<BigInt<D>>,
        op_dbg: &str,
    ) {
        let lhs = lhs.into();
        let rhs = rhs.into();
        let result = result.into();

        test_op(lhs.clone(), rhs.clone(), &op, result.clone(), op_dbg);

        test_op(rhs, lhs, op, result, op_dbg);
    }
    #[allow(clippy::similar_names)]
    pub fn test_op<D: Digit>(
        lhs: impl Into<BigInt<D>>,
        rhs: impl Into<BigInt<D>>,
        op: impl for<'b> Fn(Boo<'b, BigInt<D>>, Boo<'b, BigInt<D>>) -> Moo<'b, BigInt<D>>,
        result: impl Into<BigInt<D>>,
        op_dbg: impl AsRef<str>,
    ) {
        let lhs = lhs.into();
        let rhs = rhs.into();
        let result = result.into();
        let op_dbg = op_dbg.as_ref();
        let build_msg_id = |t1: &str, t2: &str| format!("{t1}{lhs:?} {op_dbg} {t2}{rhs:?}");
        let validate = |res: Moo<BigInt<D>>, dbg: &str| {
            assert_eq!(*res, result, "res equals with {dbg}");
        };
        let validate_mut = |res: Moo<BigInt<D>>, dbg: &str| {
            assert!(matches!(res, Moo::BorrowedMut(_)), "res mut ref with {dbg}");
            validate(res, dbg);
        };
        let validate_non_mut = |res: Moo<BigInt<D>>, dbg: &str| {
            assert!(matches!(res, Moo::Owned(_)), "res owned with {dbg}");
            validate(res, dbg);
        };
        {
            let mut lhs = lhs.clone();
            let res = op(Boo::from(&mut lhs), Boo::from(&rhs));
            let msg = build_msg_id("&mut", "&");
            validate_mut(res, &msg);
            assert_eq!(lhs, result, "assigned with {msg}");
        }
        {
            let mut lhs = lhs.clone();
            let res = op(Boo::from(&mut lhs), Boo::from(rhs.clone()));
            let msg = build_msg_id("&mut", "");
            validate_mut(res, &msg);
            assert_eq!(lhs, result, "assigned with {msg}");
        }

        {
            let mut rhs = rhs.clone();
            let res = op(Boo::from(&lhs), Boo::from(&mut rhs));
            let msg = build_msg_id("&", "&mut");
            validate_mut(res, &msg);
            assert_eq!(rhs, result, "assigned with {msg}");
        }
        {
            let mut rhs = rhs.clone();
            let res = op(Boo::from(lhs.clone()), Boo::from(&mut rhs));
            let msg = build_msg_id("", "&mut");
            validate_mut(res, &msg);
            assert_eq!(rhs, result, "assigned with {msg}");
        }

        let res = op(Boo::from(&lhs), Boo::from(&rhs));
        validate_non_mut(res, &format!("res equals with {}", build_msg_id("&", "&")));

        let res = op(Boo::from(lhs.clone()), Boo::from(&rhs));
        validate_non_mut(res, &format!("res equals with {}", build_msg_id("", "&")));

        let res = op(Boo::from(&lhs), Boo::from(rhs.clone()));
        validate_non_mut(res, &format!("res equals with {}", build_msg_id("&", "")));

        let res = op(Boo::from(lhs.clone()), Boo::from(rhs.clone()));
        validate_non_mut(res, &format!("res equals with {}", build_msg_id("", "")));
    }

    #[test]
    fn bit_or() {
        test_op_commute(
            0x1111_00000000_00001111_01010101u128,
            0x0101_01010101_11110000u128,
            |a, b| BigInt::<HalfSize>::bitor(a, b),
            0x1111_00000101_01011111_11110101u128,
            "|",
        );
    }
    #[test]
    fn bit_xor() {
        test_op_commute(
            0x1111_00000000_00001111_01010101u128,
            0x0101_01010101_11110000u128,
            |a, b| BigInt::<HalfSize>::bitxor(a, b),
            0x1111_00000101_01011010_10100101u128,
            "^",
        );
    }

    #[test]
    fn bit_and() {
        test_op_commute(
            0x1111_00000000_00001111_01010101u128,
            0x0101_01010101_11110000u128,
            |a, b| BigInt::<HalfSize>::bitand(a, b),
            0x0101_01010000u128,
            "&",
        );
    }

    #[test]
    fn shl() {
        assert_eq!(
            BigInt::<HalfSize>::from(0x0099_8877_6655_4433_2211_u128) << 4,
            BigInt::<HalfSize>::from(0x0998_8776_6554_4332_2110_u128)
        );
        assert_eq!(
            BigInt::<HalfSize>::from(1) << 1,
            BigInt::<HalfSize>::from(2)
        );
    }
    #[test]
    fn shr() {
        assert_eq!(
            BigInt::<HalfSize>::from(0x0099_8877_6655_4433_2211_u128) >> 4,
            BigInt::<HalfSize>::from(0x0009_9887_7665_5443_3221_u128)
        );
    }
    #[test]
    fn shr_overflow() {
        assert_eq!(
            BigInt::<HalfSize>::shr_internal(
                BigInt::<HalfSize>::from(0x0099_8877_6655_4433_2211_u128),
                40
            ),
            (
                Moo::from(BigInt::<HalfSize>::from(0x9988_7766_u32)),
                BigInt::<HalfSize>::from(0x55_4433_2211u64)
            )
        );
    }
    mod digits {
        use super::*;

        #[test]
        fn pow_2() {
            let zero = BigInt::<HalfSize>::from(0x0);
            let one = BigInt::<HalfSize>::from(0x1);
            let two_pow_9 = BigInt::<HalfSize>::from(0x100);
            let two_pow_9_minus_one = BigInt::<HalfSize>::from(0xff);
            for (pow, res) in [(2, 9), (4, 5), (8, 3), (16, 2)] {
                assert_eq!(zero.digits(pow), 0, "zero.digits({pow})");
                assert_eq!(one.digits(pow), 1, "one.digits({pow})");
                assert_eq!(two_pow_9.digits(pow), res, "(2^9).digits({pow})");
                assert_eq!(
                    two_pow_9_minus_one.digits(pow),
                    res - 1,
                    "(2^9-1).digits({pow})"
                );
            }
        }
        // TODO test non pow 2
    }

    #[test]
    fn add_overflow() {
        test_op_commute(
            0xffff_ffff_ffff_ffffu64,
            1,
            |a, b| BigInt::<HalfSize>::add(a, b),
            0x1_0000_0000_0000_0000u128,
            "+",
        );
    }
    #[test]
    fn add_middle_overflow() {
        test_op_commute(
            0x1000_0000_ffff_ffff_ffff_ffffu128,
            1,
            |a, b| BigInt::<HalfSize>::add(a, b),
            0x1000_0001_0000_0000_0000_0000u128,
            "+",
        );
    }
    #[test]
    fn add_two_negative() {
        test_op_commute(
            -0x1122_3344_5566_7788_i128,
            -0x8877_6655_4433_2211_i128,
            |a, b| BigInt::<HalfSize>::add(a, b),
            -0x9999_9999_9999_9999i128,
            "+",
        );
    }
    #[test]
    fn add() {
        test_op(
            0x1122_3344_5566_7788_i128,
            -0x8877_6655_4433_2211_i128,
            |a, b| BigInt::<HalfSize>::sub(a, b),
            0x9999_9999_9999_9999i128,
            "-",
        );
        test_op_commute(
            0x1122_3344_5566_7788_i128,
            0x8877_6655_4433_2211_i128,
            |a, b| BigInt::<HalfSize>::add(a, b),
            0x9999_9999_9999_9999i128,
            "+",
        );
    }

    #[test]
    fn sub_big() {
        test_op(
            0x9999_9999_9999_9999i128,
            0x8877_6655_4433_2211_i128,
            |a, b| BigInt::<HalfSize>::sub(a, b),
            0x1122_3344_5566_7788_i128,
            "-",
        );
        test_op_commute(
            0x9999_9999_9999_9999i128,
            -0x8877_6655_4433_2211_i128,
            |a, b| BigInt::<HalfSize>::add(a, b),
            0x1122_3344_5566_7788_i128,
            "+",
        );
    }
    #[test]
    fn sub_sign() {
        test_op(1, 2, |a, b| BigInt::<HalfSize>::sub(a, b), -1, "-");
        test_op(-1, -2, |a, b| BigInt::<HalfSize>::sub(a, b), 1, "-");
    }
    #[test]
    fn sub_overflow() {
        test_op(
            0x1_0000_0000_0000_0000_0000_0000_0000i128,
            1,
            |a, b| BigInt::<HalfSize>::sub(a, b),
            0xffff_ffff_ffff_ffff_ffff_ffff_ffffi128,
            "-",
        );
    }

    #[test]
    fn mul() {
        test_op_commute(7, 6, |a, b| BigInt::<HalfSize>::mul(a, b), 42, "*");
        test_op_commute(
            30_000_000_700_000u128,
            60,
            |a, b| BigInt::<HalfSize>::mul(a, b),
            1_800_000_042_000_000_u128,
            "*",
        );
    }
    #[test]
    fn mul_one_big() {
        test_op_commute(
            0x0fee_ddcc_bbaa_9988_7766_5544_3322_1100_u128,
            2,
            |a, b| BigInt::<HalfSize>::mul(a, b),
            0x1fdd_bb99_7755_3310_eecc_aa88_6644_2200_u128,
            "*",
        );
    }

    #[test]
    fn mul_with_digit() {
        assert_eq!(
            BigInt::<HalfSize>::from(0x0001_0000_0000_0000u64) * HalfSize::from(0x7766),
            BigInt::<HalfSize>::from(0x7766_0000_0000_0000u64)
        )
    }

    #[test]
    fn mul_sign() {
        test_op_commute(3, 3, |a, b| BigInt::<HalfSize>::mul(a, b), 9, "*");
        test_op_commute(-3, 3, |a, b| BigInt::<HalfSize>::mul(a, b), -9, "*");
        test_op_commute(3, -3, |a, b| BigInt::<HalfSize>::mul(a, b), -9, "*");
        test_op_commute(-3, -3, |a, b| BigInt::<HalfSize>::mul(a, b), 9, "*");
    }
    #[test]
    fn mul_both_big() {
        test_op_commute(
            0xffee_ddcc_bbaa_9988_7766_5544_3322_1100_u128,
            0xffee_ddcc_bbaa_9988_7766_5544_3322_1100_u128,
            |a, b| BigInt::<HalfSize>::mul(a, b),
            BigInt::<HalfSize>::from_iter([
                0x3343_2fd7_16cc_d713_5f99_9f4e_8521_0000_u128,
                0xffdd_bcbf_06b5_eed3_8628_ddc7_06bf_1222_u128,
            ]),
            "*",
        );
    }

    #[test]
    fn log_2() {
        assert!(BigInt::<HalfSize>::from(1).is_power_of_two());
        assert!(BigInt::<HalfSize>::from(2).is_power_of_two());
        assert!(BigInt::<HalfSize>::from(0x8000_0000_0000_0000u64).is_power_of_two());
        assert!(!BigInt::<HalfSize>::from(0x1000_0001_0000_0000u64).is_power_of_two());
        assert!(!BigInt::<HalfSize>::from(0x1000_0000_1000_0000u64).is_power_of_two());

        assert_eq!(BigInt::<HalfSize>::from(1).digits(2) - 1, 0);
        assert_eq!(BigInt::<HalfSize>::from(2).digits(2) - 1, 1);
        assert_eq!(
            BigInt::<HalfSize>::from(0x8000_0000_0000_0000u64).digits(2) - 1,
            63
        );
    }
}
