#![allow(clippy::wildcard_imports)]
use super::*;
use itertools::Itertools;

pub mod bit_math {
    use super::*;
    pub fn bit_or_assign<D: Digit>(lhs: &mut BigInt<D>, rhs: &BigInt<D>) {
        for (digit, rhs) in lhs.digits.iter_mut().zip(rhs.digits.iter()) {
            std::ops::BitOrAssign::bitor_assign(digit, rhs);
        }
        if lhs.digits.len() < rhs.digits.len() {
            lhs.digits
                .extend(rhs.digits.iter().dropping(lhs.digits.len()));
        }
    }

    pub fn bit_xor_assign<D: Digit>(lhs: &mut BigInt<D>, rhs: &BigInt<D>) {
        for (digit, rhs) in lhs.digits.iter_mut().zip(rhs.digits.iter()) {
            std::ops::BitXorAssign::bitxor_assign(digit, rhs);
        }
        if lhs.digits.len() < rhs.digits.len() {
            lhs.digits.extend(
                rhs.digits
                    .iter()
                    .dropping(lhs.digits.len())
                    .map(|it| std::ops::BitXor::bitxor(D::default(), it)),
            );
        }
        lhs.truncate_leading_zeros();
    }

    pub fn bit_and_assign<D: Digit>(lhs: &mut BigInt<D>, rhs: &BigInt<D>) {
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

    pub fn assign_same_sign<D: Digit>(lhs: &mut BigInt<D>, rhs: &BigInt<D>) {
        assert!(
            lhs.is_zero() || rhs.is_zero() || lhs.signum == rhs.signum,
            "lhs and rhs had differend signs"
        );
        let orig_self_len = lhs.digits.len();

        if orig_self_len < rhs.digits.len() {
            lhs.digits
                .extend(rhs.digits.iter().skip(orig_self_len).copied());
        }

        let mut carry = false;
        for elem in lhs
            .digits
            .iter_mut()
            .zip_longest(rhs.digits.iter().take(orig_self_len).copied())
        {
            let (digit, rhs) = match elem {
                itertools::EitherOrBoth::Right(_rhs) => unreachable!("self was extendet"),
                itertools::EitherOrBoth::Left(_digit) if !carry => {
                    break;
                }
                itertools::EitherOrBoth::Left(digit) => (digit, None),
                itertools::EitherOrBoth::Both(digit, rhs) => (digit, Some(rhs)),
            };
            (*digit, carry) = digit.carring_add(rhs.unwrap_or_default(), carry);
        }
        lhs.push(D::from_bool(carry));
        if lhs.is_zero() {
            lhs.signum = rhs.signum;
        }
        lhs.truncate_leading_zeros();
    }
}

pub mod sub {
    use super::*;

    pub fn assign_smaller_same_sign<D: Digit>(lhs: &mut BigInt<D>, rhs: &BigInt<D>) {
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
                itertools::EitherOrBoth::Both(digit, rhs) => (digit, Some(*rhs)),
            };

            (*digit, carry) = digit.carring_sub(rhs.unwrap_or_default(), carry);
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
            result <<= i * BigInt::<D>::BASIS_POW;
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
}
