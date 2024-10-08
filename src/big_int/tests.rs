// SPDX-FileCopyrightText: 2024 Nils Jochem
// SPDX-License-Identifier: MPL-2.0
use crate::{BigIInt, BigUInt, SigNum};
use digits::Decomposable;
use itertools::Itertools;

use super::*;
mod create {
    use super::*;

    use crate::util::rng::seeded_rng;
    use unsigned::FromStrErr;
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
            BigIInt::<u32>::from_iter([0x3322_1100u32, 0x7766_5544, 0x9988, 0]).split_sign(),
            (
                SigNum::Positive,
                BigUInt {
                    digits: vec![0x3322_1100u32, 0x7766_5544, 0x0000_9988].into()
                }
            )
        );
    }
    #[test]
    fn from_i128() {
        assert_eq!(
            BigIInt::<u32>::from(-0x9988_7766_5544_3322_1100i128).split_sign(),
            (
                SigNum::Negative,
                BigUInt {
                    digits: vec![0x3322_1100u32, 0x7766_5544, 0x0000_9988].into()
                }
            )
        );
    }

    #[test]
    fn from_str_err() {
        assert_eq!("".parse::<BigIInt<u8>>(), Err(FromStrErr::Empty));
        assert_eq!("0x".parse::<BigIInt<u8>>(), Err(FromStrErr::Empty));

        assert_eq!(
            "0t".parse::<BigIInt<u8>>(),
            Err(FromStrErr::UnkoneRadix('t'))
        );

        assert_eq!(
            "123t".parse::<BigIInt<u8>>(),
            Err(FromStrErr::UnkownDigit {
                digit: 't',
                position: 3
            })
        );
        assert_eq!(
            "0b01210".parse::<BigIInt<u8>>(),
            Err(FromStrErr::UnkownDigit {
                digit: '2',
                position: 4
            })
        );
    }

    #[test]
    fn from_default_radix() {
        assert_eq!("0".parse::<BigIInt<u8>>(), Ok(BigIInt::from_digit(0)));
        assert_eq!("1234".parse::<BigIInt<u8>>(), Ok(BigIInt::from(1234)));
        assert_eq!("-1234".parse::<BigIInt<u8>>(), Ok(BigIInt::from(-1234)));
        assert_eq!("1_234".parse::<BigIInt<u8>>(), Ok(BigIInt::from(1234)));
        assert_eq!("-1_234".parse::<BigIInt<u8>>(), Ok(BigIInt::from(-1234)));

        assert_eq!(
            "a".parse::<BigIInt<u8>>(),
            Err(FromStrErr::UnkownDigit {
                digit: 'a',
                position: 0
            })
        );
    }
    #[test]
    fn from_radix_two() {
        assert_eq!(
            "0b0".parse::<BigIInt<u8>>(),
            Ok(BigIInt::from_digit(0)),
            "detecting zero"
        );
        assert_eq!(
            "0b00000".parse::<BigIInt<u8>>(),
            Ok(BigIInt::from_digit(0)),
            "detecting leading zeros"
        );
        assert_eq!(
            "0b1010101010".parse::<BigIInt<u8>>(),
            Ok(BigIInt::from(0b10_1010_1010))
        );
        assert_eq!(
            "-0b1010101010".parse::<BigIInt<u8>>(),
            Ok(BigIInt::from(-0b10_1010_1010)),
            "detecting negative"
        );
        assert_eq!(
            "0b10_1010_1010".parse::<BigIInt<u8>>(),
            Ok(BigIInt::from(0b10_1010_1010)),
            "ignoring underscores"
        );

        assert_eq!(
            "0b2".parse::<BigIInt<u8>>(),
            Err(FromStrErr::UnkownDigit {
                digit: '2',
                position: 2
            }),
            "complaining about unkown digits"
        );
    }
    #[test]
    fn from_radix_hex() {
        assert_eq!(
            "0x0".parse::<BigIInt<u8>>(),
            Ok(BigIInt::from_digit(0)),
            "detecting zero"
        );
        assert_eq!(
            "0x00000".parse::<BigIInt<u8>>(),
            Ok(BigIInt::from_digit(0)),
            "detecting leading zeros"
        );
        assert_eq!(
            "0x1234cdef".parse::<BigIInt<u8>>(),
            Ok(BigIInt::from(0x1234_cdef))
        );
        assert_eq!(
            "-0x1234cdef".parse::<BigIInt<u8>>(),
            Ok(BigIInt::from(-0x1234_cdef)),
            "detecting negative"
        );
        assert_eq!(
            "0x1234cdef".parse::<BigIInt<u8>>(),
            Ok(BigIInt::from(0x1234_cdef)),
            "ignoring underscores"
        );

        assert_eq!(
            "0xg".parse::<BigIInt<u8>>(),
            Err(FromStrErr::UnkownDigit {
                digit: 'g',
                position: 2
            }),
            "complaining about unkown digits"
        );
    }

    #[test]
    fn fuzz_new_random() {
        const TRIES: usize = 100_000;

        let (seed, mut rng) = seeded_rng();

        for i in 0..TRIES {
            let pick = BigUInt::<u32>::new_random(2..=3, &mut rng);
            assert!(pick.digits.len() == 1);
            let pick = pick.digits[0];
            assert!(
                (0x00_0100..=0xff_ffff).contains(&pick),
                "#{i} 0x0100 <= {pick:#x} <= 0xff_ffff; with seed {seed:?}"
            );
        }
    }
}
mod output {
    use super::*;
    #[test]
    fn debug() {
        assert_eq!(
            format!(
                "{:?}",
                BigIInt::<u32>::from(0x0000_7766_0000_5544_3322_1100u128)
            ),
            "Number { + 0x[00007766, 00005544, 33221100]}"
        );
    }
    mod display {
        use super::*;

        #[test]
        fn simple() {
            assert_eq!(format!("{}", BigIInt::<u8>::from(251)), "251");
        }
        #[test]
        fn negative() {
            assert_eq!(
                format!("{}", BigIInt::<u8>::from(-123_456_789i64)),
                "-123456789"
            );
            assert_eq!(
                format!("{:#}", BigIInt::<u8>::from(-23_456_789i64)),
                "-23_456_789"
            );
        }

        #[test]
        fn l_align() {
            assert_eq!(format!("{:0<#10}", BigIInt::<u8>::from(0)), "0_000_000_000");
            assert_eq!(
                format!("{:0<10}", BigIInt::<u8>::from(12_345i64)),
                "0000012345"
            );
            assert_eq!(
                format!("{:0<10}", BigIInt::<u8>::from(-12_345i64)),
                "-000012345"
            );
            assert_eq!(
                format!("{:0<#10}", BigIInt::<u8>::from(12_345i64)),
                "0_000_012_345"
            );
            assert_eq!(
                format!("{:0<#10}", BigIInt::<u8>::from(-12_345i64)),
                "-000_012_345"
            );
        }
    }
    #[test]
    fn lower_hex() {
        assert_eq!(
            format!("{:x}", BigIInt::<u32>::from(0x9988_7766_5544_3322_1100u128)),
            "99887766554433221100"
        );
        assert_eq!(
            format!(
                "{:#x}",
                BigIInt::<u32>::from(0x9988_7766_5544_3322_1100u128)
            ),
            "0x99887766554433221100"
        );

        assert_eq!(
            format!(
                "{:x}",
                BigIInt::<u32>::from(-0x9988_7766_5544_3322_1100i128)
            ),
            "-99887766554433221100"
        );
        assert_eq!(
            format!(
                "{:#x}",
                BigIInt::<u32>::from(-0x9988_7766_5544_3322_1100i128)
            ),
            "-0x99887766554433221100"
        );

        assert_eq!(
            format!(
                "{:0>32x}",
                BigIInt::<u32>::from(0x9988_7766_5544_3322_1100u128)
            ),
            "00000000000099887766554433221100"
        );

        assert_eq!(
            format!(
                "{:#032x}",
                BigIInt::<u32>::from(0x9988_7766_5544_3322_1100u128)
            ),
            "0x000000000099887766554433221100"
        );

        assert_eq!(
            format!(
                "{:#032x}",
                BigIInt::<u32>::from(0x00ee_ddcc_bbaa_9988_7766_5544_3322_1100u128)
            ),
            "0xeeddccbbaa99887766554433221100"
        );
        assert_eq!(
            format!(
                "{:#032X}",
                BigIInt::<u32>::from(0x00ee_ddcc_bbaa_9988_7766_5544_3322_1100u128)
            ),
            "0XEEDDCCBBAA99887766554433221100"
        );
    }
}
mod order {
    use super::*;

    use std::cmp::Ordering;

    #[test]
    fn same() {
        assert_eq!(
            BigIInt::<u32>::from(0x9988_7766_5544_3322_1100u128)
                .cmp(&BigIInt::<u32>::from(0x9988_7766_5544_3322_1100u128)),
            Ordering::Equal
        );
        assert_eq!(
            BigIInt::<u32>::from(-0x9988_7766_5544_3322_1100i128)
                .cmp(&BigIInt::<u32>::from(-0x9988_7766_5544_3322_1100i128)),
            Ordering::Equal
        );
    }
    #[test]
    fn negated() {
        assert_eq!(
            BigIInt::<u32>::from(0x9988_7766_5544_3322_1100u128)
                .cmp(&BigIInt::<u32>::from(-0x9988_7766_5544_3322_1100i128)),
            Ordering::Greater
        );
        assert_eq!(
            BigIInt::<u32>::from(-0x9988_7766_5544_3322_1100i128)
                .cmp(&BigIInt::<u32>::from(0x9988_7766_5544_3322_1100i128)),
            Ordering::Less
        );
    }
    #[test]
    fn middle_diff() {
        assert_eq!(
            BigIInt::<u32>::from(0x9988_8866_5544_3322_1100u128)
                .cmp(&BigIInt::<u32>::from(0x9988_7766_5544_3322_1100i128)),
            Ordering::Greater
        );
        assert_eq!(
            BigIInt::<u32>::from(0x9988_7766_5544_3322_1100i128)
                .cmp(&BigIInt::<u32>::from(0x9988_8866_5544_3322_1100i128)),
            Ordering::Less
        );
    }
    #[test]
    fn same_len() {
        assert_eq!(
            BigIInt::<u32>::from(0x0fff_ffff_ffff_ffff_ffffu128)
                .cmp(&BigIInt::<u32>::from(0x9988_7766_5544_3322_1100i128)),
            Ordering::Less
        );
    }

    #[test]
    fn differnd_len() {
        assert_eq!(
            BigIInt::<u32>::from(0x7766_5544_3322_1100u64)
                .cmp(&BigIInt::<u32>::from(0x0001_0000_0000_0000_0000u128)),
            Ordering::Less
        );
    }
}

#[cfg(feature = "base64")]
#[cfg(test)]
mod t_base64 {
    use super::*;

    #[test]
    fn criss_cross() {
        let num = BigUInt::<u16>::from(0u16);
        let engine = base64::engine::general_purpose::STANDARD;
        let encode = num.as_base64(&engine);
        assert_eq!(
            BigUInt::<u16>::from_base64(encode, &engine).unwrap(),
            num,
            "zero"
        );

        let num = BigUInt::<u16>::from(123_456_789u32);
        let encode = num.as_base64(&engine);
        assert_eq!(
            BigUInt::<u16>::from_base64(encode, &engine).unwrap(),
            num,
            "big"
        );
    }
}

#[test]
fn bits() {
    let mut num = BigUInt::<u16>::from(0x0123_4567u32);
    assert_eq!(
        <BigUInt::<_> as Decomposable<bool>>::le_digits(&num).collect_vec(),
        std::iter::from_fn(move || {
            if num.is_zero() {
                None
            } else {
                let (_, r) = BigUInt::shr_internal(&mut num, 1);
                Some(!r.is_zero())
            }
        })
        .collect_vec()
    );
    assert_eq!(
        <BigUInt<_> as Decomposable<bool>>::le_digits(&BigUInt::<u32>::from(0)).next(),
        None
    );
}
pub(super) mod big_math {
    use super::*;

    use crate::util::boo::{Mob, Moo};
    use std::fmt::Debug;

    pub fn test_op_commute<B: Clone + Eq + Debug>(
        lhs: impl Into<B>,
        rhs: impl Into<B>,
        op: impl for<'b> Fn(Mob<'b, B>, Mob<'b, B>) -> Moo<'b, B>,
        result: impl Into<B>,
        op_dbg: &str,
    ) {
        let lhs = lhs.into();
        let rhs = rhs.into();
        let result = result.into();

        test_op(
            lhs.clone(),
            rhs.clone(),
            &op,
            result.clone(),
            op_dbg,
            Side::Both,
        );

        test_op(rhs, lhs, op, result, op_dbg, Side::Both);
    }
    #[allow(dead_code)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Side {
        Left,
        Right,
        Both,
        Neither,
    }
    impl Side {
        const fn do_left(self) -> bool {
            matches!(self, Self::Left | Self::Both)
        }
        const fn do_right(self) -> bool {
            matches!(self, Self::Right | Self::Both)
        }
    }
    #[allow(clippy::similar_names)]
    pub fn test_op<B: Clone + Eq + Debug>(
        lhs: impl Into<B>,
        rhs: impl Into<B>,
        op: impl for<'b> Fn(Mob<'b, B>, Mob<'b, B>) -> Moo<'b, B>,
        result: impl Into<B>,
        op_dbg: impl AsRef<str>,
        test_mut: Side,
    ) {
        let lhs = lhs.into();
        let rhs = rhs.into();
        let result = result.into();
        let op_dbg = op_dbg.as_ref();
        let build_msg_id = |t1: &str, t2: &str| format!("{t1}{lhs:?} {op_dbg} {t2}{rhs:?}");
        let validate = |res: Moo<B>, dbg: &str| {
            assert_eq!(*res, result, "res equals with {dbg}");
        };
        let validate_mut = |res: Moo<B>, dbg: &str| {
            assert!(matches!(res, Moo::BorrowedMut(_)), "res mut ref with {dbg}");
            validate(res, dbg);
        };
        let validate_non_mut = |res: Moo<B>, dbg: &str| {
            assert!(matches!(res, Moo::Owned(_)), "res owned with {dbg}");
            validate(res, dbg);
        };
        if test_mut.do_left() {
            let mut lhs = lhs.clone();
            let res = op(Mob::from(&mut lhs), Mob::from(&rhs));
            let msg = build_msg_id("&mut", "&");
            validate_mut(res, &msg);
            assert_eq!(lhs, result, "assigned with {msg}");
        }
        if test_mut.do_left() {
            let mut lhs = lhs.clone();
            let res = op(Mob::from(&mut lhs), Mob::from(rhs.clone()));
            let msg = build_msg_id("&mut", "");
            validate_mut(res, &msg);
            assert_eq!(lhs, result, "assigned with {msg}");
        }

        if test_mut.do_right() {
            let mut rhs = rhs.clone();
            let res = op(Mob::from(&lhs), Mob::from(&mut rhs));
            let msg = build_msg_id("&", "&mut");
            validate_mut(res, &msg);
            assert_eq!(rhs, result, "assigned with {msg}");
        }
        if test_mut.do_right() {
            let mut rhs = rhs.clone();
            let res = op(Mob::from(lhs.clone()), Mob::from(&mut rhs));
            let msg = build_msg_id("", "&mut");
            validate_mut(res, &msg);
            assert_eq!(rhs, result, "assigned with {msg}");
        }

        let res = op(Mob::from(&lhs), Mob::from(&rhs));
        validate_non_mut(res, &format!("res equals with {}", build_msg_id("&", "&")));

        let res = op(Mob::from(lhs.clone()), Mob::from(&rhs));
        validate_non_mut(res, &format!("res equals with {}", build_msg_id("", "&")));

        let res = op(Mob::from(&lhs), Mob::from(rhs.clone()));
        validate_non_mut(res, &format!("res equals with {}", build_msg_id("&", "")));

        let res = op(Mob::from(lhs.clone()), Mob::from(rhs.clone()));
        validate_non_mut(res, &format!("res equals with {}", build_msg_id("", "")));
    }

    mod bit_math {
        use super::*;
        #[test]
        fn bit_or() {
            test_op_commute(
                0x1111_00000000_00001111_01010101u128,
                0x0101_01010101_11110000u128,
                |a, b| BigUInt::<u32>::bitor(a, b),
                0x1111_00000101_01011111_11110101u128,
                "|",
            );
        }
        #[test]
        fn bit_xor() {
            test_op_commute(
                0x1111_00000000_00001111_01010101u128,
                0x0101_01010101_11110000u128,
                |a, b| BigUInt::<u32>::bitxor(a, b),
                0x1111_00000101_01011010_10100101u128,
                "^",
            );
        }

        #[test]
        fn bit_and() {
            test_op_commute(
                0x1111_00000000_00001111_01010101u128,
                0x0101_01010101_11110000u128,
                |a, b| BigUInt::<u32>::bitand(a, b),
                0x0101_01010000u128,
                "&",
            );
        }

        #[test]
        fn shl() {
            assert_eq!(
                BigUInt::<u32>::from(0x0099_8877_6655_4433_2211u128) << 4,
                BigUInt::<u32>::from(0x0998_8776_6554_4332_2110u128)
            );
            assert_eq!(BigUInt::<u32>::from(1u32) << 1, BigUInt::<u32>::from(2u32));
        }
        #[test]
        fn shr() {
            assert_eq!(
                BigUInt::<u32>::from(0x0099_8877_6655_4433_2211u128) >> 4,
                BigUInt::<u32>::from(0x0009_9887_7665_5443_3221u128)
            );
        }
        #[test]
        fn shr_overflow() {
            assert_eq!(
                BigUInt::<u32>::shr_internal(
                    BigUInt::<u32>::from(0x0099_8877_6655_4433_2211u128),
                    40
                ),
                (
                    Moo::from(BigUInt::<u32>::from(0x9988_7766u32)),
                    BigUInt::<u32>::from(0x55_4433_2211u64)
                )
            );
            assert_eq!(
                BigUInt::<u8>::shr_internal(BigUInt::from(0b0000_0110u32), 2),
                (
                    Moo::from(BigUInt::from(0b0000_0001u32)),
                    BigUInt::from(0b0000_0010u32)
                )
            );
        }
        #[test]
        fn shr_overflow_no_partial() {
            assert_eq!(
                BigUInt::<u8>::shr_internal(BigUInt::from(0x4433_2211u32), 16),
                (
                    Moo::from(BigUInt::from(0x4433u32)),
                    BigUInt::from(0x2211u32)
                )
            );
        }
    }
    mod digits {
        use super::*;

        use std::num::NonZero;
        use unsigned::radix::Radix;

        #[test]
        fn radix() {
            assert_eq!(
                Radix::<u8>::try_from(2),
                Ok(Radix::PowerOfTwo(NonZero::new(1).unwrap()))
            );
            assert_eq!(
                Radix::<u8>::try_from(3),
                Ok(Radix::Other(BigUInt::from_digit(3)))
            );
            assert_eq!(
                Radix::<u8>::try_from(4),
                Ok(Radix::PowerOfTwo(NonZero::new(2).unwrap()))
            );
            assert_eq!(
                Radix::<u8>::try_from(8),
                Ok(Radix::PowerOfTwo(NonZero::new(3).unwrap()))
            );
            assert_eq!(Radix::<u8>::try_from(256), Ok(Radix::DigitBase));
            assert_eq!(
                Radix::<u32>::try_from(0x0001_0000_0000),
                Ok(Radix::DigitBase)
            );
        }
        #[test]
        fn pow_2() {
            let zero = BigIInt::<u32>::from(0x0);
            let one = BigIInt::<u32>::from(0x1);
            let two_pow_9 = BigIInt::<u32>::from(0x100);
            let two_pow_9_minus_one = BigIInt::<u32>::from(0xff);
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
        #[test]
        fn pow_3_minus_one() {
            assert_eq!(BigIInt::<u8>::from_digit(2).try_digits(3), Ok(1));
            assert_eq!(BigIInt::<u8>::from_digit(8).try_digits(3), Ok(2));
            assert_eq!(BigIInt::<u8>::from_digit(25).try_digits(3), Ok(3));
        }
        #[test]
        fn pow_3() {
            assert_eq!(BigIInt::<u8>::from_digit(3).try_digits(3), Ok(2));
            assert_eq!(BigIInt::<u8>::from_digit(9).try_digits(3), Ok(3));
            assert_eq!(BigIInt::<u8>::from_digit(27).try_digits(3), Ok(4));
        }
        #[test]
        fn pow_3_plus_one() {
            assert_eq!(BigIInt::<u8>::from_digit(4).try_digits(3), Ok(2));
            assert_eq!(BigIInt::<u8>::from_digit(10).try_digits(3), Ok(3));
            assert_eq!(BigIInt::<u8>::from_digit(28).try_digits(3), Ok(4));
        }
        // TODO test non pow 2
    }

    #[test]
    fn add_overflow() {
        test_op_commute(
            0xffff_ffff_ffff_ffffu64,
            1,
            |a, b| BigIInt::<u32>::add(a, b),
            0x1_0000_0000_0000_0000u128,
            "+",
        );
    }
    #[test]
    fn add_middle_overflow() {
        test_op_commute(
            0x1000_0000_ffff_ffff_ffff_ffffu128,
            1,
            |a, b| BigIInt::<u32>::add(a, b),
            0x1000_0001_0000_0000_0000_0000u128,
            "+",
        );
    }
    #[test]
    fn add_two_negative() {
        test_op_commute(
            -0x1122_3344_5566_7788i128,
            -0x8877_6655_4433_2211i128,
            |a, b| BigIInt::<u32>::add(a, b),
            -0x9999_9999_9999_9999i128,
            "+",
        );
    }
    #[test]
    fn add() {
        test_op(
            0x1122_3344_5566_7788i128,
            -0x8877_6655_4433_2211i128,
            |a, b| BigIInt::<u32>::sub(a, b),
            0x9999_9999_9999_9999i128,
            "-",
            Side::Both,
        );
        test_op_commute(
            0x1122_3344_5566_7788i128,
            0x8877_6655_4433_2211i128,
            |a, b| BigIInt::<u32>::add(a, b),
            0x9999_9999_9999_9999i128,
            "+",
        );
    }

    #[test]
    fn sub_big() {
        test_op(
            0x9999_9999_9999_9999i128,
            0x8877_6655_4433_2211i128,
            |a, b| BigIInt::<u32>::sub(a, b),
            0x1122_3344_5566_7788i128,
            "-",
            Side::Both,
        );
        test_op_commute(
            0x9999_9999_9999_9999i128,
            -0x8877_6655_4433_2211i128,
            |a, b| BigIInt::<u32>::add(a, b),
            0x1122_3344_5566_7788i128,
            "+",
        );
    }
    #[test]
    fn sub_sign() {
        test_op(1, 2, |a, b| BigIInt::<u32>::sub(a, b), -1, "-", Side::Both);
        test_op(-1, -2, |a, b| BigIInt::<u32>::sub(a, b), 1, "-", Side::Both);
    }
    #[test]
    fn sub_overflow() {
        test_op(
            0x1_0000_0000_0000_0000_0000_0000_0000i128,
            1,
            |a, b| BigIInt::<u32>::sub(a, b),
            0xffff_ffff_ffff_ffff_ffff_ffff_ffffi128,
            "-",
            Side::Both,
        );
    }

    #[test]
    fn mul() {
        test_op_commute(7, 6, |a, b| BigIInt::<u32>::mul(a, b), 42, "*");
        test_op_commute(
            30_000_000_700_000u128,
            60,
            |a, b| BigIInt::<u32>::mul(a, b),
            1_800_000_042_000_000u128,
            "*",
        );
    }
    #[test]
    fn mul_one_big() {
        test_op_commute(
            0x0fee_ddcc_bbaa_9988_7766_5544_3322_1100u128,
            2,
            |a, b| BigIInt::<u32>::mul(a, b),
            0x1fdd_bb99_7755_3310_eecc_aa88_6644_2200u128,
            "*",
        );
    }

    #[test]
    fn mul_with_digit() {
        assert_eq!(
            BigIInt::<u32>::from(0x0001_0000_0000_0000u64) * 0x7766,
            BigIInt::from(0x7766_0000_0000_0000u64)
        );
    }

    #[test]
    fn mul_sign() {
        test_op_commute(3, 3, |a, b| BigIInt::<u32>::mul(a, b), 9, "*");
        test_op_commute(-3, 3, |a, b| BigIInt::<u32>::mul(a, b), -9, "*");
        test_op_commute(3, -3, |a, b| BigIInt::<u32>::mul(a, b), -9, "*");
        test_op_commute(-3, -3, |a, b| BigIInt::<u32>::mul(a, b), 9, "*");
    }
    #[test]
    fn mul_both_big() {
        test_op_commute(
            0xffee_ddcc_bbaa_9988_7766_5544_3322_1100u128,
            0xffee_ddcc_bbaa_9988_7766_5544_3322_1100u128,
            |a, b| BigIInt::<u32>::mul(a, b),
            BigIInt::<u32>::from_iter([
                0x3343_2fd7_16cc_d713_5f99_9f4e_8521_0000u128,
                0xffdd_bcbf_06b5_eed3_8628_ddc7_06bf_1222u128,
            ]),
            "*",
        );
    }

    #[test]
    fn div_same() {
        assert_eq!(
            BigIInt::<u8>::div_mod_euclid(BigIInt::from_digit(10), BigIInt::from_digit(10)),
            (
                Moo::Owned(BigIInt::from_digit(1)),
                Moo::Owned(BigUInt::from_digit(0))
            )
        );
    }

    mod div_sign {
        use super::*;

        #[test]
        fn no_negative() {
            assert_eq!(
                BigIInt::div_mod_euclid(BigIInt::<u8>::from_digit(7), BigIInt::from_digit(4)),
                (
                    Moo::Owned(BigIInt::from_digit(1)),
                    Moo::Owned(BigUInt::from_digit(3))
                ),
                "no negative"
            );
        }
        #[test]
        fn rhs_negative() {
            assert_eq!(
                BigIInt::div_mod_euclid(BigIInt::<u8>::from_digit(7), BigIInt::from(-4)),
                (
                    Moo::Owned(BigIInt::from(-1)),
                    Moo::Owned(BigUInt::from_digit(3))
                ),
                "rhs negative"
            );
        }
        #[test]
        fn lhs_negative() {
            assert_eq!(
                BigIInt::div_mod_euclid(BigIInt::<u8>::from(-7), BigIInt::from(4)),
                (
                    Moo::Owned(BigIInt::from(-2)),
                    Moo::Owned(BigUInt::from_digit(1))
                ),
                "lhs negative"
            );
        }
        #[test]
        fn both_negative() {
            assert_eq!(
                BigIInt::div_mod_euclid(BigIInt::<u8>::from(-7), BigIInt::from(-4)),
                (
                    Moo::Owned(BigIInt::from(2)),
                    Moo::Owned(BigUInt::from_digit(1))
                ),
                "both negative"
            );
        }
    }

    #[test]
    fn log_2() {
        assert!(BigIInt::<u32>::from(1).is_power_of_two());
        assert!(BigIInt::<u32>::from(2).is_power_of_two());
        assert!(BigIInt::<u32>::from(0x8000_0000_0000_0000u64).is_power_of_two());
        assert!(!BigIInt::<u32>::from(0x1000_0001_0000_0000u64).is_power_of_two());
        assert!(!BigIInt::<u32>::from(0x1000_0000_1000_0000u64).is_power_of_two());

        assert_eq!(BigIInt::<u32>::from(1).digits(2) - 1, 0);
        assert_eq!(BigIInt::<u32>::from(2).digits(2) - 1, 1);
        assert_eq!(
            BigIInt::<u32>::from(0x8000_0000_0000_0000u64).digits(2) - 1,
            63
        );
    }
    #[test]
    fn pow() {
        assert_eq!(
            BigIInt::pow(BigIInt::from_digit(3), 21).cloned(),
            BigIInt::from_digit(3u64.pow(21))
        );
    }
    #[test]
    fn pow2() {
        assert_eq!(
            BigUInt::pow(BigUInt::from_digit(2), 21).cloned(),
            BigUInt::from_digit(1u32) << 21
        );
        assert_eq!(
            BigUInt::pow(BigUInt::from_digit(4), 21).cloned(),
            BigUInt::from_digit(1u32) << 42
        );
    }
    #[test]
    fn x_pow_zero() {
        assert_eq!(
            BigIInt::pow(BigIInt::from_digit(50u32), 0).cloned(),
            BigIInt::from_digit(1)
        );
    }
    #[test]
    fn zero_pow_zero() {
        assert_eq!(
            BigIInt::pow(BigIInt::from_digit(0u32), 0).cloned(),
            BigIInt::from_digit(1)
        );
    }
    #[test]
    fn zero_pow_n() {
        assert_eq!(
            BigIInt::pow(BigIInt::from_digit(0u32), 50).cloned(),
            BigIInt::from_digit(0)
        );
    }
}
