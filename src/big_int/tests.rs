use super::*;
mod create {
    use super::*;
    #[test]
    fn cast_signum() {
        for i in i8::MIN..=i8::MAX {
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
            BigInt::from_iter([0x33221100u32, 0x77665544, 0x9988]),
            BigInt {
                signum: SigNum::Positive,
                data: vec![
                    HalfSize::from(0x33221100),
                    HalfSize::from(0x77665544),
                    HalfSize::from(0x00009988)
                ]
            }
        )
    }
    #[test]
    fn from_i128() {
        assert_eq!(
            BigInt::from(-0x99887766554433221100i128),
            BigInt {
                signum: SigNum::Negative,
                data: vec![
                    HalfSize::from(0x33221100),
                    HalfSize::from(0x77665544),
                    HalfSize::from(0x00009988)
                ]
            }
        )
    }
}
mod output {
    use super::*;

    #[test]
    fn lower_hex() {
        assert_eq!(
            format!("{:x}", BigInt::from(0x99887766554433221100u128)),
            "99887766554433221100"
        );
        assert_eq!(
            format!("{:#x}", BigInt::from(0x99887766554433221100u128)),
            "0x99887766554433221100"
        );

        assert_eq!(
            format!("{:x}", BigInt::from(-0x99887766554433221100i128)),
            "-99887766554433221100"
        );
        assert_eq!(
            format!("{:#x}", BigInt::from(-0x99887766554433221100i128)),
            "-0x99887766554433221100"
        );

        assert_eq!(
            format!("{:0>32x}", BigInt::from(0x99887766554433221100u128)),
            "00000000000099887766554433221100"
        );

        assert_eq!(
            format!("{:#032x}", BigInt::from(0x99887766554433221100u128)),
            "0x000000000099887766554433221100"
        );

        assert_eq!(
            format!(
                "{:#032x}",
                BigInt::from(0xeeddccbbaa99887766554433221100u128)
            ),
            "0xeeddccbbaa99887766554433221100"
        );
        assert_eq!(
            format!(
                "{:#032X}",
                BigInt::from(0xeeddccbbaa99887766554433221100u128)
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
            BigInt::from(0x99887766554433221100u128).cmp(&BigInt::from(0x99887766554433221100u128)),
            Ordering::Equal
        );
        assert_eq!(
            BigInt::from(-0x99887766554433221100i128)
                .cmp(&BigInt::from(-0x99887766554433221100i128)),
            Ordering::Equal
        );
    }
    #[test]
    fn negated() {
        assert_eq!(
            BigInt::from(0x99887766554433221100u128)
                .cmp(&BigInt::from(-0x99887766554433221100i128)),
            Ordering::Greater
        );
        assert_eq!(
            BigInt::from(-0x99887766554433221100i128)
                .cmp(&BigInt::from(0x99887766554433221100i128)),
            Ordering::Less
        );
    }
    #[test]
    fn middle_diff() {
        assert_eq!(
            BigInt::from(0x99888866554433221100u128).cmp(&BigInt::from(0x99887766554433221100i128)),
            Ordering::Greater
        );
        assert_eq!(
            BigInt::from(0x99887766554433221100i128).cmp(&BigInt::from(0x99888866554433221100i128)),
            Ordering::Less
        );
    }
    #[test]
    fn size_diff() {
        assert_eq!(
            BigInt::from(0xfffffffffffffffffffu128).cmp(&BigInt::from(0x99887766554433221100i128)),
            Ordering::Less
        );
    }
}
mod full_size {
    use super::*;

    #[test]
    fn load() {
        assert_eq!(
            FullSize::from(0x7766554433221100usize),
            FullSize::new(HalfSize::from(0x33221100), HalfSize::from(0x77665544))
        );
    }

    #[test]
    fn read() {
        assert_eq!(
            FullSize::from(0x7766554433221100usize).lower(),
            HalfSize::from(0x33221100)
        );
        assert_eq!(
            FullSize::from(0x7766554433221100usize).higher(),
            HalfSize::from(0x77665544)
        );
    }
}
pub(super) mod big_math {
    use super::*;
    pub fn test_op_commute(
        lhs: impl Into<BigInt>,
        rhs: impl Into<BigInt>,
        op: impl for<'b> Fn(Boo<'b, BigInt>, Boo<'b, BigInt>) -> Moo<'b, BigInt>,
        result: impl Into<BigInt>,
        op_dbg: &str,
    ) {
        let lhs = lhs.into();
        let rhs = rhs.into();
        let result = result.into();

        test_op(lhs.clone(), rhs.clone(), &op, result.clone(), op_dbg);

        test_op(rhs, lhs, op, result, op_dbg);
    }
    pub fn test_op(
        lhs: impl Into<BigInt>,
        rhs: impl Into<BigInt>,
        op: impl for<'b> Fn(Boo<'b, BigInt>, Boo<'b, BigInt>) -> Moo<'b, BigInt>,
        result: impl Into<BigInt>,
        op_dbg: impl AsRef<str>,
    ) {
        let lhs = lhs.into();
        let rhs = rhs.into();
        let result = result.into();
        let op_dbg = op_dbg.as_ref();
        let build_msg_id = |t1: &str, t2: &str| format!("{t1}{lhs:?} {op_dbg} {t2}{rhs:?}");
        let validate = |res: Moo<BigInt>, dbg: &str| {
            assert_eq!(*res, result, "res equals with {dbg}");
        };
        let validate_mut = |res: Moo<BigInt>, dbg: &str| {
            assert!(matches!(res, Moo::BorrowedMut(_)), "res mut ref with {dbg}");
            validate(res, dbg);
        };
        let validate_non_mut = |res: Moo<BigInt>, dbg: &str| {
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
            |a, b| BigInt::bitor(a, b),
            0x1111_00000101_01011111_11110101u128,
            "|",
        );
    }
    #[test]
    fn bit_xor() {
        test_op_commute(
            0x1111_00000000_00001111_01010101u128,
            0x0101_01010101_11110000u128,
            |a, b| BigInt::bitxor(a, b),
            0x1111_00000101_01011010_10100101u128,
            "^",
        );
    }

    #[test]
    fn bit_and() {
        test_op_commute(
            0x1111_00000000_00001111_01010101u128,
            0x0101_01010101_11110000u128,
            |a, b| BigInt::bitand(a, b),
            0x0101_01010000u128,
            "&",
        );
    }

    #[test]
    fn shl() {
        assert_eq!(
            BigInt::from(0x998877665544332211u128) << 4,
            BigInt::from(0x9988776655443322110u128)
        );
        assert_eq!(BigInt::from(1) << 1, BigInt::from(2));
    }
    #[test]
    fn shr() {
        assert_eq!(
            BigInt::from(0x998877665544332211u128) >> 4,
            BigInt::from(0x99887766554433221u128)
        )
    }
    #[test]
    fn add_overflow() {
        test_op_commute(
            0xffff_ffff_ffff_ffffu64,
            1,
            |a, b| BigInt::add(a, b),
            0x1_0000_0000_0000_0000u128,
            "+",
        );
    }
    #[test]
    fn add_middle_overflow() {
        test_op_commute(
            0x1000_0000_ffff_ffff_ffff_ffffu128,
            1,
            |a, b| BigInt::add(a, b),
            0x1000_0001_0000_0000_0000_0000u128,
            "+",
        );
    }
    #[test]
    fn add_two_negative() {
        test_op_commute(
            -0x11223344_55667788i128,
            -0x88776655_44332211i128,
            |a, b| BigInt::add(a, b),
            -0x9999_9999_9999_9999i128,
            "+",
        );
    }
    #[test]
    fn add() {
        test_op(
            0x11223344_55667788i128,
            -0x88776655_44332211i128,
            |a, b| BigInt::sub(a, b),
            0x9999_9999_9999_9999i128,
            "-",
        );
        test_op_commute(
            0x11223344_55667788i128,
            0x88776655_44332211i128,
            |a, b| BigInt::add(a, b),
            0x9999_9999_9999_9999i128,
            "+",
        );
    }

    #[test]
    fn sub_big() {
        test_op(
            0x9999_9999_9999_9999i128,
            0x88776655_44332211i128,
            |a, b| BigInt::sub(a, b),
            0x11223344_55667788i128,
            "-",
        );
        test_op_commute(
            0x9999_9999_9999_9999i128,
            -0x88776655_44332211i128,
            |a, b| BigInt::add(a, b),
            0x11223344_55667788i128,
            "+",
        );
    }
    #[test]
    fn sub_sign() {
        test_op(1, 2, |a, b| BigInt::sub(a, b), -1, "-");
        test_op(-1, -2, |a, b| BigInt::sub(a, b), 1, "-");
    }
    #[test]
    fn sub_overflow() {
        test_op(
            0x1_0000_0000_0000_0000_0000_0000_0000i128,
            1,
            |a, b| BigInt::sub(a, b),
            0xffff_ffff_ffff_ffff_ffff_ffff_ffffi128,
            "-",
        );
    }

    #[test]
    fn mul() {
        test_op_commute(7, 6, |a, b| BigInt::mul(a, b), 42, "*");
        test_op_commute(
            30_000_000_700_000u128,
            60,
            |a, b| BigInt::mul(a, b),
            180_000_004_200_0000u128,
            "*",
        );
    }
    #[test]
    fn mul_one_big() {
        test_op_commute(
            0x0feeddcc_bbaa9988_77665544_33221100u128,
            2,
            |a, b| BigInt::mul(a, b),
            0x1fddbb9977553310eeccaa8866442200u128,
            "*",
        );
    }

    #[test]
    fn mul_sign() {
        test_op_commute(3, 3, |a, b| BigInt::mul(a, b), 9, "*");
        test_op_commute(-3, 3, |a, b| BigInt::mul(a, b), -9, "*");
        test_op_commute(3, -3, |a, b| BigInt::mul(a, b), -9, "*");
        test_op_commute(-3, -3, |a, b| BigInt::mul(a, b), 9, "*");
    }
    #[test]
    fn mul_both_big() {
        test_op_commute(
            0xffeeddcc_bbaa9988_77665544_33221100u128,
            0xffeeddcc_bbaa9988_77665544_33221100u128,
            |a, b| BigInt::mul(a, b),
            BigInt::from_iter([
                0x33432fd716ccd7135f999f4e85210000u128,
                0xffddbcbf06b5eed38628ddc706bf1222u128,
            ]),
            "*",
        );
    }

    #[test]
    fn log_2() {
        assert_eq!(BigInt::from(1).ilog2(), Some(0));
        assert_eq!(BigInt::from(2).ilog2(), Some(1));
        assert_eq!(BigInt::from(0x8000_0000_0000_0000u64).ilog2(), Some(63));
        assert_eq!(BigInt::from(0x1000_0001_0000_0000u64).ilog2(), None);
        assert_eq!(BigInt::from(0x1000_0000_1000_0000u64).ilog2(), None);
    }
}
