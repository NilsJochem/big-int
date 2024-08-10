#![allow(clippy::wildcard_imports)]
use super::*;
use itertools::Itertools;

pub mod bit_math {
    use super::*;
    pub fn bit_or_assign(lhs: &mut BigInt, rhs: &BigInt) {
        for (digit, rhs) in lhs.digits.iter_mut().zip(rhs.digits.iter()) {
            std::ops::BitOrAssign::bitor_assign(digit, rhs);
        }
        if lhs.digits.len() < rhs.digits.len() {
            lhs.digits
                .extend(rhs.digits.iter().dropping(lhs.digits.len()));
        }
    }

    pub fn bit_xor_assign(lhs: &mut BigInt, rhs: &BigInt) {
        for (digit, rhs) in lhs.digits.iter_mut().zip(rhs.digits.iter()) {
            std::ops::BitXorAssign::bitxor_assign(digit, rhs);
        }
        if lhs.digits.len() < rhs.digits.len() {
            lhs.digits.extend(
                rhs.digits
                    .iter()
                    .dropping(lhs.digits.len())
                    .map(|it| std::ops::BitXor::bitxor(HalfSize::default(), it)),
            );
        }
        lhs.truncate_leading_zeros();
    }

    pub fn bit_and_assign(lhs: &mut BigInt, rhs: &BigInt) {
        for (digit, rhs) in lhs.digits.iter_mut().zip(rhs.digits.iter()) {
            std::ops::BitAndAssign::bitand_assign(digit, rhs);
        }

        if lhs.digits.len() > rhs.digits.len() {
            let to_remove = lhs.digits.len() - rhs.digits.len();
            for _ in 0..to_remove {
                lhs.pop();
            }
        }
        lhs.truncate_leading_zeros();
    }
}

pub mod add {
    use super::*;

    pub fn assign_same_sign(lhs: &mut BigInt, rhs: &BigInt) {
        assert!(
            lhs.is_zero() || rhs.is_zero() || lhs.signum == rhs.signum,
            "lhs and rhs had differend signs"
        );
        let orig_self_len = lhs.digits.len();

        if orig_self_len < rhs.digits.len() {
            lhs.digits
                .extend(rhs.digits.iter().skip(orig_self_len).copied());
        }

        let mut carry = HalfSize::default();
        for elem in lhs
            .digits
            .iter_mut()
            .zip_longest(rhs.digits.iter().take(orig_self_len).copied())
        {
            let (digit, rhs) = match elem {
                itertools::EitherOrBoth::Right(_rhs) => unreachable!("self was extendet"),
                itertools::EitherOrBoth::Left(_digit) if *carry == 0 => {
                    break;
                }
                itertools::EitherOrBoth::Left(digit) => (digit, None),
                itertools::EitherOrBoth::Both(digit, rhs) => (digit, Some(rhs)),
            };
            let result = **digit as usize + *carry as usize;
            let result = FullSize::from(match rhs {
                None => result,
                Some(rhs) => result + *rhs as usize,
            });

            *digit = result.lower();
            carry = result.higher();
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

    pub fn assign_smaller_same_sign(lhs: &mut BigInt, rhs: &BigInt) {
        assert!(
            lhs.is_zero() || rhs.is_zero() || lhs.signum == rhs.signum,
            "lhs and rhs had differend signs"
        );
        assert!(lhs.abs_ord(rhs).is_ge(), "lhs is smaller than rhs");

        let mut carry = false;
        for elem in lhs.digits.iter_mut().zip_longest(&rhs.digits) {
            let (digit, rhs) = match elem {
                itertools::EitherOrBoth::Right(_rhs) => unreachable!("lhs is always bigger"),
                itertools::EitherOrBoth::Left(_digit) if !carry => {
                    break;
                }
                itertools::EitherOrBoth::Left(digit) => (digit, None),
                itertools::EitherOrBoth::Both(digit, rhs) => (digit, Some(rhs)),
            };

            let result = FullSize::from(
                *FullSize::new(*digit, HalfSize::from(1))
                    - carry as usize
                    - rhs.map_or(0, |&rhs| *rhs as usize),
            );
            *digit = result.lower();
            carry = *result.higher() == 0; // extra bit was needed
        }

        lhs.truncate_leading_zeros();
    }
}

pub mod mul {
    use super::*;

    pub fn naive(lhs: &BigInt, rhs: &BigInt) -> BigInt {
        // try to minimize outer loops
        if lhs.digits.len() < rhs.digits.len() {
            return naive(rhs, lhs);
        }
        let mut out = BigInt::default();
        for (i, rhs_digit) in rhs.digits.iter().enumerate().rev() {
            let mut result = std::ops::Mul::mul(lhs.clone(), rhs_digit);
            result <<= i * HalfSizeNative::BITS as usize;
            out += result;
        }

        out.signum = lhs.signum * rhs.signum;
        out.truncate_leading_zeros();
        out
    }
    pub fn assign_mul_digit_at_offset(lhs: &mut BigInt, rhs: HalfSize, i: usize) {
        let mut carry = HalfSize::default();
        for elem in lhs.digits.iter_mut().skip(i) {
            let mul_result = FullSize::from((**elem) as usize * (*rhs) as usize);
            let add_result = FullSize::from((*mul_result.lower() as usize) + (*carry as usize));

            carry = HalfSize::from(*mul_result.higher() + *add_result.higher());
            *elem = add_result.lower();
        }
        lhs.digits.push(carry);
        lhs.truncate_leading_zeros();
    }
}

pub mod div {
    use super::*;
    use crate::big_int::digits::{FullSize, HalfSize, HalfSizeNative};

    /// computes (lhs/rhs, lhs%rhs)
    /// expects lhs and rhs to be non-negative and rhs to be non-zero
    pub fn normalized_schoolbook(mut lhs: BigInt, mut rhs: BigInt) -> (BigInt, BigInt) {
        let shift = rhs
            .digits
            .last()
            .expect("can't divide by 0")
            .leading_zeros() as usize;
        lhs <<= shift;
        rhs <<= shift;
        let (q, mut r) = schoolbook(lhs, rhs);
        r >>= shift;
        (q, r)
    }
    pub(super) fn schoolbook(lhs: BigInt, rhs: BigInt) -> (BigInt, BigInt) {
        let (m, n) = (lhs.digits.len(), rhs.digits.len());
        assert!(
            rhs.digits
                .last()
                .expect("can't divide by zero")
                .leading_zeros()
                == 0,
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
        let power = BigInt::BASIS_POW * (m - n - 1);
        let (lhs_prime, s) = BigInt::shr_internal(lhs, power);
        let (q_prime, r_prime) = schoolbook_sub(expect_owned(lhs_prime, "shr_internal"), &rhs);
        assert!(s.digits.len() < (m - n));
        let (q, r) = BigInt::div_mod((r_prime << power) + s, rhs);
        assert!(q.digits.len() < (m - n));
        (
            (q_prime << power) + expect_owned(q, "div_mod"),
            expect_owned(r, "div_mod"),
        )
    }
    fn expect_owned<T: Clone>(moo: Moo<T>, op: impl AsRef<str>) -> T {
        moo.expect_owned(format!("{} didn't get a mut ref", op.as_ref()))
    }
    pub(super) fn schoolbook_sub(mut lhs: BigInt, rhs: &BigInt) -> (BigInt, BigInt) {
        let n = rhs.digits.len();
        assert!(lhs.digits.len() <= n + 1, "0 <= {lhs:?} < base^{}", n + 1);
        assert!(
            rhs.digits
                .last()
                .expect("rhs can't be zero")
                .leading_zeros()
                == 0,
            "base^{n}/2 <= {rhs:?} < base^{n}"
        );

        match lhs.cmp(rhs) {
            std::cmp::Ordering::Less => return (BigInt::from(0), lhs.clone()),
            std::cmp::Ordering::Equal => return (BigInt::from(1), BigInt::from(0)),
            std::cmp::Ordering::Greater => {}
        }
        let rhs_times_basis = rhs << BigInt::BASIS_POW;
        if lhs >= rhs_times_basis {
            // let mut i = 0;
            // while lhs >= rhs_times_basis {
            lhs -= &rhs_times_basis;
            //     i += 1;
            // }
            // if i > 0 {
            let (mut div_res, mod_res) = schoolbook_sub(lhs, rhs);
            div_res += BigInt::from(BigInt::BASIS); // * HalfSize::from(i);
            return (div_res, mod_res);
            // }
        }
        let mut q = crate::big_int::digits::HalfSizeNative::try_from(
            *FullSize::new(
                lhs.digits.get(n - 1).cloned().unwrap_or_default(),
                lhs.digits.get(n).cloned().unwrap_or_default(),
            ) / (*rhs.digits[n - 1] as usize),
        )
        .unwrap_or(HalfSizeNative::MAX);
        let mut t = rhs * HalfSize::from(q);
        for _ in 0..=1 {
            if t > lhs {
                q -= 1;
                t -= rhs;
            }
        }
        return (BigInt::from(q), lhs - t);
    }
}

#[cfg(test)]
mod tests {
    mod t_mul {
        use super::super::*;

        #[test]
        fn both_big_naive() {
            assert_eq!(
                mul::naive(
                    &BigInt::from(0xffee_ddcc_bbaa_9988_7766_5544_3322_1100_u128),
                    &BigInt::from(0xffee_ddcc_bbaa_9988_7766_5544_3322_1100_u128)
                ),
                BigInt::from_iter([
                    0x3343_2fd7_16cc_d713_5f99_9f4e_8521_0000_u128,
                    0xffdd_bcbf_06b5_eed3_8628_ddc7_06bf_1222_u128,
                ])
            );
        }
    }
    mod t_add {
        use super::super::*;

        #[test]
        fn add_smaller() {
            let mut lhs = BigInt::from_iter([
                0x0000_0000_u32,
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
                    0x0000_0000_u32,
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
                    0x0000_0000_u32,
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
            let mut lhs = BigInt::from(0);
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
                    div::normalized_schoolbook(
                        BigInt::from(55402179209251644110543835108628647875u128),
                        BigInt::from(7015904223016035028600428233219344947u128)
                    ),
                    (
                        BigInt::from(7),
                        BigInt::from(6290849648139398910340837476093233246u128)
                    )
                );
            }

            #[test]
            fn differnt_size_remainder_zero() {
                assert_eq!(
                    div::normalized_schoolbook(
                        BigInt::from_iter([
                            0x3343_2fd7_16cc_d713_5f99_9f4e_8521_0000_u128,
                            0xffdd_bcbf_06b5_eed3_8628_ddc7_06bf_1222_u128,
                        ]),
                        BigInt::from(0xffee_ddcc_bbaa_9988_7766_5544_3322_1100_u128)
                    ),
                    (
                        BigInt::from(0xffee_ddcc_bbaa_9988_7766_5544_3322_1100_u128),
                        BigInt::from(0)
                    )
                );
            }
            #[test]
            fn t_schoolbook_simple() {
                assert_eq!(
                    div::normalized_schoolbook(
                        BigInt::from(0x7766_5544_3322_1100u64),
                        BigInt::from(0x1_0000_0000u64)
                    ),
                    (BigInt::from(0x7766_5544), BigInt::from(0x3322_1100))
                )
            }
            #[test]
            fn t_schoolbook_sub() {
                assert_eq!(
                    div::schoolbook_sub(
                        BigInt::from(0xbbaa_9988_7766_5544_3322_1100u128),
                        &BigInt::from(0x8000_0000_0000_0000u64)
                    ),
                    (
                        BigInt::from(0x1_7755_3310u64),
                        BigInt::from(0x7766_5544_3322_1100u64)
                    )
                )
            }
        }
    }
}
