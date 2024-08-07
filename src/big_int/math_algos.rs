#![allow(clippy::wildcard_imports)]
use super::*;
use itertools::Itertools;

pub mod bit_math {
    use super::*;
    pub fn bit_or_assign(lhs: &mut BigInt, rhs: &BigInt) {
        for (part, rhs) in lhs.data.iter_mut().zip(rhs.data.iter()) {
            std::ops::BitOrAssign::bitor_assign(part, rhs);
        }
        if lhs.data.len() < rhs.data.len() {
            lhs.data.extend(rhs.data.iter().dropping(lhs.data.len()));
        }
    }

    pub fn bit_xor_assign(lhs: &mut BigInt, rhs: &BigInt) {
        for (part, rhs) in lhs.data.iter_mut().zip(rhs.data.iter()) {
            std::ops::BitXorAssign::bitxor_assign(part, rhs);
        }
        if lhs.data.len() < rhs.data.len() {
            lhs.data.extend(
                rhs.data
                    .iter()
                    .dropping(lhs.data.len())
                    .map(|it| std::ops::BitXor::bitxor(HalfSize::default(), it)),
            );
        }
        lhs.recalc_len();
    }

    pub fn bit_and_assign(lhs: &mut BigInt, rhs: &BigInt) {
        for (part, rhs) in lhs.data.iter_mut().zip(rhs.data.iter()) {
            std::ops::BitAndAssign::bitand_assign(part, rhs);
        }

        if lhs.data.len() > rhs.data.len() {
            let to_remove = lhs.data.len() - rhs.data.len();
            for _ in 0..to_remove {
                lhs.pop();
            }
        }
        lhs.recalc_len();
    }
}

pub mod add {
    use super::*;

    pub fn assign_same_sign(lhs: &mut BigInt, rhs: &BigInt) {
        assert!(
            lhs.is_zero() || rhs.is_zero() || lhs.signum == rhs.signum,
            "lhs and rhs had differend signs"
        );
        let orig_self_len = lhs.data.len();

        if orig_self_len < rhs.data.len() {
            lhs.data
                .extend(rhs.data.iter().skip(orig_self_len).copied());
        }

        let mut carry = HalfSize::default();
        for elem in lhs
            .data
            .iter_mut()
            .zip_longest(rhs.data.iter().take(orig_self_len).copied())
        {
            let (part, rhs) = match elem {
                itertools::EitherOrBoth::Right(_rhs) => unreachable!("self was extendet"),
                itertools::EitherOrBoth::Left(_part) if *carry == 0 => {
                    break;
                }
                itertools::EitherOrBoth::Left(part) => (part, None),
                itertools::EitherOrBoth::Both(part, rhs) => (part, Some(rhs)),
            };
            let result = **part as usize + *carry as usize;
            let result = FullSize::from(match rhs {
                None => result,
                Some(rhs) => result + *rhs as usize,
            });

            *part = result.lower();
            carry = result.higher();
        }
        lhs.push(carry);
        if lhs.is_zero() {
            lhs.signum = rhs.signum;
        }
        lhs.recalc_len();
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
        for elem in lhs.data.iter_mut().zip_longest(&rhs.data) {
            let (part, rhs) = match elem {
                itertools::EitherOrBoth::Right(_rhs) => unreachable!("lhs is always bigger"),
                itertools::EitherOrBoth::Left(_part) if !carry => {
                    break;
                }
                itertools::EitherOrBoth::Left(part) => (part, None),
                itertools::EitherOrBoth::Both(part, rhs) => (part, Some(rhs)),
            };

            let result = FullSize::from(
                *FullSize::new(*part, HalfSize::from(1))
                    - carry as usize
                    - rhs.map_or(0, |&rhs| *rhs as usize),
            );
            *part = result.lower();
            carry = *result.higher() == 0; // extra bit was needed
        }

        lhs.recalc_len();
    }
}

pub mod mul {
    use super::*;

    pub fn naive(lhs: &BigInt, rhs: &BigInt) -> BigInt {
        // try to minimize outer loops
        if lhs.data.len() < rhs.data.len() {
            return naive(rhs, lhs);
        }
        let mut out = BigInt::default();
        for (i, rhs_part) in rhs.data.iter().enumerate().rev() {
            let mut result = std::ops::Mul::mul(lhs, rhs_part);
            result <<= i * HalfSizeNative::BITS as usize;
            out += result;
        }

        out.signum = lhs.signum * rhs.signum;
        out.recalc_len();
        out
    }
    pub fn part_at_offset_abs(lhs: &BigInt, rhs: HalfSize, i: usize) -> BigInt {
        let mut out = BigInt::default();

        let mut carry = HalfSize::default();
        for elem in lhs.data.iter().skip(i) {
            let mul_result = FullSize::from((**elem) as usize * (*rhs) as usize);
            let add_result = FullSize::from((*mul_result.lower() as usize) + (*carry as usize));

            carry = HalfSize::from(*mul_result.higher() + *add_result.higher());
            out.data.push(add_result.lower());
        }
        out.data.push(carry);
        out.signum = SigNum::Positive;
        out.recalc_len();
        out
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
                    &BigInt::from(0xffeeddcc_bbaa9988_77665544_33221100u128),
                    &BigInt::from(0xffeeddcc_bbaa9988_77665544_33221100u128)
                ),
                [
                    0x33432fd716ccd7135f999f4e85210000u128,
                    0xffddbcbf06b5eed38628ddc706bf1222u128,
                ]
                .into_iter()
                .collect::<BigInt>()
            );
        }
    }
    mod t_add {
        use super::super::*;

        #[test]
        fn add_smaller() {
            let mut lhs = BigInt::from_iter([
                0x00000000u32,
                0x00000000,
                0x00000000,
                0xf5d28c00,
                0xb17e4b17,
                0x7e4b17e4,
                0x4b17e4b1,
                0xffddbcbe,
            ]);
            add::assign_same_sign(
                &mut lhs,
                &BigInt::from_iter([
                    0x00000000u32,
                    0x00000000,
                    0xd0420800,
                    0x07f6e5d4,
                    0x4c3b2a19,
                    0x907f6e5d,
                    0x907f6e5d,
                ]),
            );
            assert_eq!(
                lhs,
                BigInt::from_iter([
                    0x00000000u32,
                    0x00000000,
                    0xd0420800,
                    0xfdc971d4,
                    0xfdb97530,
                    0x0eca8641,
                    0xdb97530f,
                    0xffddbcbe,
                ])
            );
        }

        #[test]
        fn assign_to_zero() {
            let mut lhs = BigInt::from(0);
            add::assign_same_sign(&mut lhs, &BigInt::from(1));
            assert_eq!(lhs, BigInt::from(1))
        }
    }
}
