#![allow(clippy::wildcard_imports)]
use super::*;
use itertools::Itertools;

pub mod bit_math {
    use super::*;
    fn op_assign_zipped<'b, D: 'b>(
        lhs: &mut BigInt<D>,
        rhs: &'b BigInt<D>,
        op: impl for<'a> Fn(&'a mut D, &'b D),
    ) {
        for (digit, rhs) in lhs.digits.iter_mut().zip(rhs.digits.iter()) {
            op(digit, rhs);
        }
    }

    pub fn bit_or_assign<D: Digit>(lhs: &mut BigInt<D>, rhs: &BigInt<D>) {
        op_assign_zipped(lhs, rhs, std::ops::BitOrAssign::bitor_assign);
        lhs.digits.extend(rhs.digits.iter().skip(lhs.digits.len()));
    }
    pub fn bit_xor_assign<D: Digit>(lhs: &mut BigInt<D>, rhs: &BigInt<D>) {
        op_assign_zipped(lhs, rhs, std::ops::BitXorAssign::bitxor_assign);
        lhs.digits.extend(rhs.digits.iter().skip(lhs.digits.len()));
        lhs.truncate_leading_zeros();
    }
    pub fn bit_and_assign<D: Digit>(lhs: &mut BigInt<D>, rhs: &BigInt<D>) {
        op_assign_zipped(lhs, rhs, std::ops::BitAndAssign::bitand_assign);
        lhs.digits.truncate(rhs.digits.len());
        lhs.truncate_leading_zeros();
    }
}

pub mod add {
    use super::*;

    /// calculates `lhs` += `rhs`, both need to have the same sign, but either may be zero
    /// prefers lhs to be the longer number
    pub fn assign_same_sign<D: Digit>(lhs: &mut BigInt<D>, rhs: &BigInt<D>) {
        assert!(
            lhs.is_zero() || rhs.is_zero() || lhs.signum == rhs.signum,
            "lhs and rhs had differend signs"
        );
        let orig_lhs_len = lhs.digits.len();
        lhs.digits.extend(rhs.digits.iter().skip(orig_lhs_len));

        let mut carry = false;
        for elem in lhs
            .digits
            .iter_mut()
            .zip_longest(rhs.digits.iter().take(orig_lhs_len))
        {
            use itertools::EitherOrBoth as E;
            let (lhs_digit, rhs_digit) = match elem {
                E::Right(_rhs) => unreachable!("self was extendet"),
                E::Left(_digit) if !carry => {
                    break;
                }
                E::Left(digit) => (digit, D::default()),
                E::Both(digit, rhs) => (digit, *rhs),
            };
            (*lhs_digit, carry) = lhs_digit.carring_add(rhs_digit, carry);
        }
        lhs.push(carry);
        if lhs.is_zero() {
            lhs.signum = rhs.signum;
        }
        lhs.truncate_leading_zeros();
    }
}

pub mod sub {
    use super::*;

    /// calculates `lhs` -= `rhs`, both need to have the same sign, but either may be zero
    /// lhs needs to be the longer number
    pub fn assign_smaller_same_sign<D: Digit>(lhs: &mut BigInt<D>, rhs: &BigInt<D>) {
        assert!(
            lhs.is_zero() || rhs.is_zero() || lhs.signum == rhs.signum,
            "lhs and rhs had differend signs"
        );
        assert!(lhs.abs_ord(rhs).is_ge(), "lhs is smaller than rhs");

        let mut carry = false;
        for elem in lhs.digits.iter_mut().zip_longest(rhs.digits.iter()) {
            use itertools::EitherOrBoth as E;
            let (lhs_digit, rhs_digit) = match elem {
                E::Right(_rhs) => unreachable!("lhs is always bigger"),
                E::Left(_digit) if !carry => {
                    break;
                }
                E::Left(digit) => (digit, D::default()),
                E::Both(digit, rhs) => (digit, *rhs),
            };

            (*lhs_digit, carry) = lhs_digit.carring_sub(rhs_digit, carry);
        }

        lhs.truncate_leading_zeros();
    }
}

pub mod mul {
    use super::*;

    pub fn naive<D: Digit>(lhs: &BigInt<D>, rhs: &BigInt<D>) -> BigInt<D> {
        // try to minimize outer loops
        if lhs.digits.len() < rhs.digits.len() {
            return naive(rhs, lhs);
        }
        let mut out = BigInt::default();
        for (i, rhs_digit) in rhs.digits.iter().enumerate().rev() {
            let mut result = std::ops::Mul::mul(lhs.clone(), rhs_digit);
            result <<= i * D::BASIS_POW;
            out += result;
        }

        out.signum = lhs.signum * rhs.signum;
        out.truncate_leading_zeros();
        out
    }
    pub fn assign_mul_digit_at_offset<D: Digit>(lhs: &mut BigInt<D>, rhs: D, i: usize) {
        let mut carry = D::default();
        for digit in lhs.digits.iter_mut().skip(i) {
            (*digit, carry) = digit.widening_mul(rhs, carry).split_le();
        }
        lhs.digits.push(carry);
        lhs.truncate_leading_zeros();
    }
}

pub mod div {
    use super::*;

    /// computes (lhs/rhs, lhs%rhs)
    /// expects lhs and rhs to be non-negative and rhs to be non-zero
    pub fn normalized_schoolbook<D: Digit>(
        mut lhs: BigInt<D>,
        mut rhs: BigInt<D>,
    ) -> (BigInt<D>, BigInt<D>) {
        let shift =
            D::BASIS_POW - (rhs.digits.last().expect("can't divide by 0").ilog2() as usize + 1);
        lhs <<= shift;
        rhs <<= shift;
        let (q, mut r) = schoolbook(lhs, rhs);
        r >>= shift;
        (q, r)
    }
    #[allow(clippy::many_single_char_names)]
    pub(super) fn schoolbook<D: Digit>(lhs: BigInt<D>, rhs: BigInt<D>) -> (BigInt<D>, BigInt<D>) {
        let (m, n) = (lhs.digits.len(), rhs.digits.len());
        assert_eq!(
            rhs.digits.last().expect("can't divide by zero").ilog2(),
            (D::BASIS_POW) as u32 - 1,
            "base^{n}/2 <= {rhs:?} < base^{n}"
        );

        if m < n {
            return (BigInt::from(0), lhs);
        }
        if m == n {
            return if lhs < rhs {
                (BigInt::from(0), lhs)
            } else {
                (BigInt::from(1), lhs - rhs)
            };
        }
        if m == n + 1 {
            return schoolbook_sub(lhs, &rhs);
        }
        let power = D::BASIS_POW * (m - n - 1);
        let (lhs_prime, s) = BigInt::shr_internal(lhs, power);
        let (q_prime, r_prime) = schoolbook_sub(expect_owned(lhs_prime, "shr_internal"), &rhs);
        debug_assert!(s.digits.len() < (m - n));
        let (q, r) = BigInt::div_mod((r_prime << power) + s, rhs);
        debug_assert!(q.digits.len() < (m - n));
        (
            (q_prime << power) + expect_owned(q, "div_mod"),
            expect_owned(r, "div_mod"),
        )
    }
    fn expect_owned<T: Clone>(moo: Moo<T>, op: impl AsRef<str>) -> T {
        moo.expect_owned(format!("{} didn't get a mut ref", op.as_ref()))
    }
    pub(super) fn schoolbook_sub<D: Digit>(
        mut lhs: BigInt<D>,
        rhs: &BigInt<D>,
    ) -> (BigInt<D>, BigInt<D>) {
        let n = rhs.digits.len();
        assert!(lhs.digits.len() <= n + 1, "0 <= {lhs:?} < base^{}", n + 1);
        assert_eq!(
            rhs.digits.last().expect("rhs can't be zero").ilog2(),
            D::BASIS_POW as u32 - 1,
            "base^{n}/2 <= {rhs:?} < base^{n}"
        );

        match lhs.cmp(rhs) {
            std::cmp::Ordering::Less => return (BigInt::from(0), lhs.clone()),
            std::cmp::Ordering::Equal => return (BigInt::from(1), BigInt::from(0)),
            std::cmp::Ordering::Greater => {}
        }
        {
            let rhs_times_basis = rhs << D::BASIS_POW;
            let mut i = 0;
            while lhs >= rhs_times_basis {
                lhs -= &rhs_times_basis;
                i += 1;
            }
            if i > 0 {
                let (mut div_res, mod_res) = schoolbook_sub(lhs, rhs);
                div_res += BigInt::from(i) << D::BASIS_POW;
                return (div_res, mod_res);
            }
        }
        let (res_lower, res_upper) = (D::Wide::new(
            lhs.digits.get(n - 1).copied().unwrap_or_default(),
            lhs.digits.get(n).copied().unwrap_or_default(),
        ) / D::Wide::widen(rhs.digits[n - 1]))
        .split_le();

        let mut q = if res_upper.eq_u8(0) {
            res_lower
        } else {
            D::MAX
        };
        let mut t = rhs * q;
        for _ in 0..2 {
            if t <= lhs {
                break;
            }
            let carry;
            (q, carry) = q.overflowing_sub(D::from(true));
            debug_assert!(!carry, "q-1 has underflowed");
            t -= rhs;
        }
        (BigInt::from_digit(q), lhs - t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    mod t_mul {
        use super::*;

        #[test]
        fn both_big_naive() {
            assert_eq!(
                mul::naive::<u32>(
                    &BigInt::from(0xffee_ddcc_bbaa_9988_7766_5544_3322_1100u128),
                    &BigInt::from(0xffee_ddcc_bbaa_9988_7766_5544_3322_1100u128)
                ),
                BigInt::from_iter([
                    0x3343_2fd7_16cc_d713_5f99_9f4e_8521_0000u128,
                    0xffdd_bcbf_06b5_eed3_8628_ddc7_06bf_1222u128,
                ])
            );
        }
    }
    mod t_add {
        use super::*;

        #[test]
        fn add_smaller() {
            let mut lhs = BigInt::<u32>::from_iter([
                0x0000_0000u32,
                0x0000_0000,
                0x0000_0000,
                0xf5d2_8c00,
                0xb17e_4b17,
                0x7e4b_17e4,
                0x4b17_e4b1,
                0xffdd_bcbe,
            ]);
            add::assign_same_sign(
                &mut lhs,
                &BigInt::from_iter([
                    0x0000_0000u32,
                    0x0000_0000,
                    0xd042_0800,
                    0x07f6_e5d4,
                    0x4c3b_2a19,
                    0x907f_6e5d,
                    0x907f_6e5d,
                ]),
            );
            assert_eq!(
                lhs,
                BigInt::from_iter([
                    0x0000_0000u32,
                    0x0000_0000,
                    0xd042_0800,
                    0xfdc9_71d4,
                    0xfdb9_7530,
                    0x0eca_8641,
                    0xdb97_530f,
                    0xffdd_bcbe,
                ])
            );
        }

        #[test]
        fn assign_to_zero() {
            let mut lhs = BigInt::<u32>::from(0);
            add::assign_same_sign(&mut lhs, &BigInt::from(1));
            assert_eq!(lhs, BigInt::from(1));
        }
    }
    mod t_div {
        mod m_schoolbook {

            use super::super::super::*;

            #[test]
            fn rel_same_size() {
                assert_eq!(
                    div::normalized_schoolbook::<u32>(
                        BigInt::from(55_402_179_209_251_644_110_543_835_108_628_647_875u128),
                        BigInt::from(7_015_904_223_016_035_028_600_428_233_219_344_947u128)
                    ),
                    (
                        BigInt::from(7),
                        BigInt::from(6_290_849_648_139_398_910_340_837_476_093_233_246u128)
                    )
                );
            }

            #[test]
            fn differnt_size_remainder_zero() {
                assert_eq!(
                    div::normalized_schoolbook::<u32>(
                        BigInt::from_iter([
                            0x3343_2fd7_16cc_d713_5f99_9f4e_8521_0000u128,
                            0xffdd_bcbf_06b5_eed3_8628_ddc7_06bf_1222u128,
                        ]),
                        BigInt::from(0xffee_ddcc_bbaa_9988_7766_5544_3322_1100u128)
                    ),
                    (
                        BigInt::from(0xffee_ddcc_bbaa_9988_7766_5544_3322_1100u128),
                        BigInt::from(0)
                    )
                );
            }
            #[test]
            fn differnt_size_remainder_zero_smaller() {
                assert_eq!(
                    div::normalized_schoolbook::<u8>(
                        BigInt::from(0x8765_4321u32),
                        BigInt::from(0x060d)
                    ),
                    (BigInt::from(0x0016_6065), BigInt::from(0))
                );
            }
            #[test]
            fn t_schoolbook_simple() {
                assert_eq!(
                    div::normalized_schoolbook::<u32>(
                        BigInt::from(0x7766_5544_3322_1100u64),
                        BigInt::from(0x1_0000_0000u64)
                    ),
                    (BigInt::from(0x7766_5544), BigInt::from(0x3322_1100))
                );
            }
            #[test]
            fn t_schoolbook_sub() {
                assert_eq!(
                    div::schoolbook_sub::<u32>(
                        BigInt::from(0xbbaa_9988_7766_5544_3322_1100u128),
                        &BigInt::from(0x8000_0000_0000_0000u64)
                    ),
                    (
                        BigInt::from(0x1_7755_3310u64),
                        BigInt::from(0x7766_5544_3322_1100u64)
                    )
                );
            }
        }
    }
}
