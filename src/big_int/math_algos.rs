use super::*;
use itertools::Itertools;

pub mod bit {
    use super::*;
    pub fn bit_or_assign_internal(lhs: &mut BigInt, rhs: &BigInt) {
        for (part, rhs) in lhs.data.iter_mut().zip(rhs.data.iter()) {
            std::ops::BitOrAssign::bitor_assign(part, rhs);
        }
        if lhs.data.len() < rhs.data.len() {
            lhs.data.extend(rhs.data.iter().dropping(lhs.data.len()));
            lhs.extent_length(HALF_SIZE_BYTES as isize);
        }
    }

    pub fn bit_xor_assign_internal(lhs: &mut BigInt, rhs: &BigInt) {
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
            lhs.extent_length(HALF_SIZE_BYTES as isize);
        }
        lhs.recalc_len();
    }

    pub fn bit_and_assign_internal(lhs: &mut BigInt, rhs: &BigInt) {
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

    pub fn assign_internal(lhs: &mut BigInt, rhs: &BigInt) {
        let orig_self_len = lhs.data.len();

        if orig_self_len < rhs.data.len() {
            lhs.data
                .extend(rhs.data.iter().skip(orig_self_len).copied());
            lhs.bytes = rhs.bytes;
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
    }
}

pub mod sub {
    use super::*;

    pub fn assign_smaller_same_sign(lhs: &mut BigInt, rhs: &BigInt) {
        assert!(!lhs.is_different_sign(rhs), "lhs has different sign as rhs");
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
        for (i, rhs_part) in rhs.data.iter().enumerate() {
            let mut result = std::ops::Mul::mul(lhs, rhs_part);
            result <<= i * HalfSizeNative::BITS as usize;
            out += result;
        }

        out.recalc_len();
        out.bytes *= lhs.signum() * rhs.signum();
        out
    }
    pub fn part_at_offset(lhs: &BigInt, rhs: HalfSize, i: usize) -> BigInt {
        let mut out = BigInt::default();

        let mut carry = HalfSize::default();
        for elem in lhs.data.iter().skip(i) {
            let mul_result = FullSize::from((**elem) as usize * (*rhs) as usize);
            let add_result = FullSize::from((*mul_result.lower() as usize) + (*carry as usize));

            carry = HalfSize::from(*mul_result.higher() + *add_result.higher());
            out.data.push(add_result.lower());
        }
        out.data.push(carry);
        out.recalc_len();
        out
    }
}
