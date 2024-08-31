#![allow(clippy::wildcard_imports)]
// SPDX-FileCopyrightText: 2024 Nils Jochem
// SPDX-License-Identifier: MPL-2.0
use crate::{
    big_int::digits::{Digit, Wide},
    BigUInt,
};
use itertools::Itertools;

pub mod bit_math {

    use super::*;
    fn op_assign_zipped<'b, D: 'b>(
        lhs: &mut BigUInt<D>,
        rhs: &'b BigUInt<D>,
        op: impl for<'a> Fn(&'a mut D, &'b D),
    ) {
        for (digit, rhs) in lhs.digits.iter_mut().zip(rhs.digits.iter()) {
            op(digit, rhs);
        }
    }

    pub fn bit_or_assign<D: Digit>(lhs: &mut BigUInt<D>, rhs: &BigUInt<D>) {
        op_assign_zipped(lhs, rhs, std::ops::BitOrAssign::bitor_assign);
        lhs.digits
            .extend(rhs.digits.iter().copied().dropping(lhs.digits.len()));
    }
    pub fn bit_xor_assign<D: Digit>(lhs: &mut BigUInt<D>, rhs: &BigUInt<D>) {
        op_assign_zipped(lhs, rhs, std::ops::BitXorAssign::bitxor_assign);
        lhs.digits
            .extend(rhs.digits.iter().copied().dropping(lhs.digits.len()));
        lhs.truncate_leading_zeros();
    }
    pub fn bit_and_assign<D: Digit>(lhs: &mut BigUInt<D>, rhs: &BigUInt<D>) {
        op_assign_zipped(lhs, rhs, std::ops::BitAndAssign::bitand_assign);
        lhs.digits.truncate(rhs.digits.len());
        lhs.truncate_leading_zeros();
    }
}

pub mod add {
    use super::*;

    /// calculates `lhs` += `rhs`, both need to have the same sign, but either may be zero
    /// prefers lhs to be the longer number
    pub fn assign<D: Digit>(lhs: &mut BigUInt<D>, rhs: &BigUInt<D>) {
        let orig_lhs_len = lhs.digits.len();
        lhs.digits
            .extend(rhs.digits.iter().copied().dropping(orig_lhs_len));

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
        lhs.truncate_leading_zeros();
    }
}

pub mod sub {
    use super::*;

    /// calculates `lhs` -= `rhs`, both need to have the same sign, but either may be zero
    /// lhs needs to be the longer number
    pub fn assign_smaller<D: Digit>(lhs: &mut BigUInt<D>, rhs: &BigUInt<D>) {
        assert!(*lhs >= *rhs, "lhs is smaller than rhs");

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

    pub fn naive<D: Digit>(lhs: &BigUInt<D>, rhs: &BigUInt<D>) -> BigUInt<D> {
        // try to minimize outer loops
        if lhs.digits.len() < rhs.digits.len() {
            return naive(rhs, lhs);
        }
        let mut out = BigUInt::default();
        for (i, rhs_digit) in rhs.digits.iter().enumerate().rev() {
            let mut result = std::ops::Mul::mul(lhs.clone(), rhs_digit);
            result <<= i * D::BASIS_POW;
            out += result;
        }

        out.truncate_leading_zeros();
        out
    }
    pub fn assign_mul_digit_at_offset<D: Digit>(lhs: &mut BigUInt<D>, rhs: D, i: usize) {
        let mut carry = D::default();
        for digit in lhs.digits.iter_mut().skip(i) {
            (*digit, carry) = digit.widening_mul(rhs, carry).split_le();
        }
        lhs.digits.push(carry);
        lhs.truncate_leading_zeros();
    }
}

pub mod div {

    use crate::util::boo::Moo;

    use super::*;

    /// computes (lhs/rhs, lhs%rhs)
    /// expects lhs and rhs to be non-negative and rhs to be non-zero
    pub fn normalized_schoolbook<D: Digit>(
        mut lhs: BigUInt<D>,
        mut rhs: BigUInt<D>,
    ) -> (BigUInt<D>, BigUInt<D>) {
        let shift =
            D::BASIS_POW - (rhs.digits.last().expect("can't divide by 0").ilog2() as usize + 1);
        lhs <<= shift;
        rhs <<= shift;
        let (q, mut r) = schoolbook(lhs, rhs);
        r >>= shift;
        (q, r)
    }
    #[allow(clippy::many_single_char_names)]
    pub(super) fn schoolbook<D: Digit>(
        lhs: BigUInt<D>,
        rhs: BigUInt<D>,
    ) -> (BigUInt<D>, BigUInt<D>) {
        let (m, n) = (lhs.digits.len(), rhs.digits.len());
        assert_eq!(
            rhs.digits.last().expect("can't divide by zero").ilog2(),
            (D::BASIS_POW) as u32 - 1,
            "base^{n}/2 <= {rhs:?} < base^{n}"
        );

        if m < n {
            return (BigUInt::ZERO, lhs);
        }
        if m == n {
            return if lhs < rhs {
                (BigUInt::ZERO, lhs)
            } else {
                (BigUInt::ONE, lhs - rhs)
            };
        }
        if m == n + 1 {
            return schoolbook_sub(lhs, &rhs);
        }
        let power = D::BASIS_POW * (m - n - 1);
        let (lhs_prime, s) = BigUInt::shr_internal(lhs, power);
        let (q_prime, r_prime) = schoolbook_sub(expect_owned(lhs_prime, "shr_internal"), &rhs);
        debug_assert!(s.digits.len() < (m - n));
        let (q, r) = BigUInt::div_mod_euclid((r_prime << power) + s, rhs);
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
        mut lhs: BigUInt<D>,
        rhs: &BigUInt<D>,
    ) -> (BigUInt<D>, BigUInt<D>) {
        let n = rhs.digits.len();
        assert!(lhs.digits.len() <= n + 1, "0 <= {lhs:?} < base^{}", n + 1);
        assert_eq!(
            rhs.digits.last().expect("rhs can't be zero").ilog2(),
            D::BASIS_POW as u32 - 1,
            "base^{n}/2 <= {rhs:?} < base^{n}"
        );

        match lhs.cmp(rhs) {
            std::cmp::Ordering::Less => return (BigUInt::ZERO, lhs.clone()),
            std::cmp::Ordering::Equal => return (BigUInt::ONE, BigUInt::ZERO),
            std::cmp::Ordering::Greater => {}
        }
        {
            let rhs_times_basis = rhs << D::BASIS_POW;
            let mut i = 0u32;
            while lhs >= rhs_times_basis {
                lhs -= &rhs_times_basis;
                i += 1;
            }
            if i > 0 {
                let (mut div_res, mod_res) = schoolbook_sub(lhs, rhs);
                div_res += BigUInt::from(i) << D::BASIS_POW;
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
        (BigUInt::from_digit(q), lhs - t)
    }
}
pub mod gcd {
    use crate::{ops::DivMod, BigIInt, BigUInt};

    use super::*;

    struct GCDHelper<D: Digit> {
        r: BigIInt<D>,
        s: BigIInt<D>,
    }
    fn euclid<D: Digit>(mut a: BigIInt<D>, mut b: BigIInt<D>) -> (GCDHelper<D>, GCDHelper<D>) {
        let sign_a = a.take_sign();
        let sign_b = b.take_sign();

        let mut old = GCDHelper {
            r: a,
            s: BigIInt::ONE,
        };
        let mut new = GCDHelper {
            r: b,
            s: BigIInt::ZERO,
        };

        while !new.r.is_zero() {
            let (qoutient, remainder) = old.r.div_mod_euclid(&new.r);

            let next = GCDHelper {
                r: BigIInt::from(remainder),
                s: old.s - qoutient * &new.s,
            };

            old = new;
            new = next;
        }
        old.s *= sign_a;
        new.s.signum = sign_b;
        (old, new)
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct BezoutBuilder<D: Digit>(Option<(BigIInt<D>, BigIInt<D>)>);
    impl<D: Digit> BezoutBuilder<D> {
        fn new(a: &BigIInt<D>, b: &BigIInt<D>) -> Self {
            if b.is_zero() {
                Self(None)
            } else {
                Self(Some((a.clone(), b.clone())))
            }
        }
        #[allow(dead_code)]
        fn calculate_factors(self, gcd: &BigUInt<D>, new_s: BigIInt<D>) -> Factors<D> {
            Factors {
                a: if let Some((a, b)) = self.0 {
                    (&new_s * a) / b
                } else {
                    gcd.clone().into() // to not get a 0 when gcd(a,0)
                },
                b: new_s,
            }
        }
        fn calculate_coefficients(
            self,
            gcd: &BigUInt<D>,
            old_s: BigIInt<D>,
        ) -> BezoutCoefficients<D> {
            if let Some((a, b)) = self.0 {
                BezoutCoefficients {
                    y: (gcd - &old_s * a) / b,
                    x: old_s,
                }
            } else {
                BezoutCoefficients {
                    x: gcd.clone().into(), // to not get a 1 when gcd(0,0)
                    y: BigIInt::ZERO,
                }
            }
        }
    }

    /// calculates just the gcd
    ///
    /// if Factors or Bezout coefficients are needed, use `Gcd::new`
    #[allow(dead_code)]
    pub fn gcd<D: Digit>(a: BigIInt<D>, b: BigIInt<D>) -> BigIInt<D> {
        let (old, _) = euclid(a, b);
        old.r
    }
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Gcd<D: Digit> {
        #[allow(clippy::struct_field_names)]
        gcd: BigUInt<D>,
        old_s: BigIInt<D>,
        new_s: BigIInt<D>,
        bezout_builder: BezoutBuilder<D>,
    }
    impl<D: Digit> Gcd<D> {
        pub fn new(a: impl Into<BigIInt<D>>, b: impl Into<BigIInt<D>>) -> Self {
            let a = a.into();
            let b = b.into();
            let bezout_builder = BezoutBuilder::new(&a, &b);
            let (old, new) = euclid(a, b);
            Self {
                gcd: old.r.into(),
                old_s: old.s,
                new_s: new.s,
                bezout_builder,
            }
        }

        pub fn gcd(self) -> BigUInt<D> {
            self.gcd
        }
        pub fn bezout_coefficients(self) -> (BigUInt<D>, BezoutCoefficients<D>) {
            let bezout = self
                .bezout_builder
                .calculate_coefficients(&self.gcd, self.old_s);
            (self.gcd, bezout)
        }
        pub fn factors(self) -> (BigUInt<D>, Factors<D>) {
            let factors = self.bezout_builder.calculate_factors(&self.gcd, self.new_s);
            (self.gcd, factors)
        }
        /// (`gcd`, `bezout_x`, `factor_b`)
        pub fn all_no_calc(self) -> (BigUInt<D>, BigIInt<D>, BigIInt<D>) {
            (self.gcd, self.old_s, self.new_s)
        }
        pub fn all(self) -> (BigUInt<D>, BezoutCoefficients<D>, Factors<D>) {
            let bezout = self
                .bezout_builder
                .clone()
                .calculate_coefficients(&self.gcd, self.old_s);
            let factors = self.bezout_builder.calculate_factors(&self.gcd, self.new_s);
            (self.gcd, bezout, factors)
        }
    }

    /// gcd(a, b) * factor.a = a
    /// gcd(a, b) * factor.b = b
    ///
    /// or
    /// a/factor.a = gcd(a, b) = b/factor.b
    ///
    /// or
    /// factor.b = factor.a*a/b
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Factors<D: Digit> {
        pub a: BigIInt<D>,
        pub b: BigIInt<D>,
    }

    /// ax + by = gcd(a,b)
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct BezoutCoefficients<D: Digit> {
        pub x: BigIInt<D>,
        pub y: BigIInt<D>,
    }

    #[cfg(test)]
    pub(super) mod test_helper {
        use super::*;

        #[cfg(test)]
        pub fn test_gcd<D: Digit>(
            a: impl Into<BigIInt<D>>,
            b: impl Into<BigIInt<D>>,
            gcd: impl Into<BigIInt<D>>,
            old_s: impl Into<BigIInt<D>>,
        ) {
            let b = b.into();
            let gcd = gcd.into();

            let res = Gcd::new(a.into(), b.clone());

            assert_eq!(res.gcd, gcd, "gcc correct");

            assert_eq!(res.old_s, old_s.into(), "old s");
            assert_eq!(res.new_s, &b / &gcd, "last s * gcc = b");
        }
        #[cfg(test)]
        pub fn test_bezout<D: Digit>(
            a: impl Into<BigIInt<D>>,
            b: impl Into<BigIInt<D>>,
            gcd: impl Into<BigIInt<D>>,
            old_s: impl Into<BigIInt<D>>,
            old_t: impl Into<BigIInt<D>>,
        ) {
            let a = a.into();
            let b = b.into();
            let gcd = gcd.into();

            let bezout = BezoutBuilder::new(&a, &b);
            assert_eq!(
                bezout.clone().calculate_coefficients(&gcd, old_s.into()).y,
                old_t.into(),
                "old t"
            );
            assert_eq!(bezout.calculate_factors(&gcd, b / &gcd).a, a / gcd, "new t");
        }
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
                    &BigUInt::from(0xffee_ddcc_bbaa_9988_7766_5544_3322_1100u128),
                    &BigUInt::from(0xffee_ddcc_bbaa_9988_7766_5544_3322_1100u128)
                ),
                BigUInt::from_iter([
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
            let mut lhs = BigUInt::<u32>::from_iter([
                0x0000_0000u32,
                0x0000_0000,
                0x0000_0000,
                0xf5d2_8c00,
                0xb17e_4b17,
                0x7e4b_17e4,
                0x4b17_e4b1,
                0xffdd_bcbe,
            ]);
            add::assign(
                &mut lhs,
                &BigUInt::from_iter([
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
                BigUInt::from_iter([
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
            let mut lhs = BigUInt::<u32>::from(0);
            add::assign(&mut lhs, &BigUInt::ONE);
            assert_eq!(lhs, BigUInt::ONE);
        }
    }
    mod t_div {
        mod m_schoolbook {

            use super::super::super::*;

            #[test]
            fn rel_same_size() {
                assert_eq!(
                    div::normalized_schoolbook::<u32>(
                        BigUInt::from(55_402_179_209_251_644_110_543_835_108_628_647_875u128),
                        BigUInt::from(7_015_904_223_016_035_028_600_428_233_219_344_947u128)
                    ),
                    (
                        BigUInt::from_digit(7),
                        BigUInt::from(6_290_849_648_139_398_910_340_837_476_093_233_246u128)
                    )
                );
            }

            #[test]
            fn differnt_size_remainder_zero() {
                assert_eq!(
                    div::normalized_schoolbook::<u32>(
                        BigUInt::from_iter([
                            0x3343_2fd7_16cc_d713_5f99_9f4e_8521_0000u128,
                            0xffdd_bcbf_06b5_eed3_8628_ddc7_06bf_1222u128,
                        ]),
                        BigUInt::from(0xffee_ddcc_bbaa_9988_7766_5544_3322_1100u128)
                    ),
                    (
                        BigUInt::from(0xffee_ddcc_bbaa_9988_7766_5544_3322_1100u128),
                        BigUInt::from_digit(0)
                    )
                );
            }
            #[test]
            fn differnt_size_remainder_zero_smaller() {
                assert_eq!(
                    div::normalized_schoolbook::<u8>(
                        BigUInt::from(0x8765_4321u32),
                        BigUInt::from(0x060du32)
                    ),
                    (BigUInt::from(0x0016_6065u32), BigUInt::from_digit(0))
                );
            }
            #[test]
            fn t_schoolbook_simple() {
                assert_eq!(
                    div::normalized_schoolbook::<u32>(
                        BigUInt::from(0x7766_5544_3322_1100u64),
                        BigUInt::from(0x1_0000_0000u64)
                    ),
                    (
                        BigUInt::from_digit(0x7766_5544u32),
                        BigUInt::from_digit(0x3322_1100u32)
                    )
                );
            }
            #[test]
            fn t_schoolbook_sub() {
                assert_eq!(
                    div::schoolbook_sub::<u32>(
                        BigUInt::from(0xbbaa_9988_7766_5544_3322_1100u128),
                        &BigUInt::from(0x8000_0000_0000_0000u64)
                    ),
                    (
                        BigUInt::from(0x1_7755_3310u64),
                        BigUInt::from(0x7766_5544_3322_1100u64)
                    )
                );
            }
        }
    }
    mod gcd {
        use crate::{
            big_int::math_algos::gcd::{
                test_helper::{test_bezout, test_gcd},
                BezoutCoefficients, Factors, Gcd,
            },
            BigIInt,
        };

        #[test]
        fn t_inner_gcd_both_positive() {
            test_gcd::<u32>(240, 46, 2, -9);
            test_gcd::<u32>(46, 240, 2, 47);
        }
        #[test]
        fn t_inner_gcd_rhs_negative() {
            test_gcd::<u32>(240, -46, 2, -9);
            test_gcd::<u32>(46, -240, 2, 47);
        }
        #[test]
        fn t_inner_gcd_lhs_negative() {
            test_gcd::<u32>(-240, 46, 2, 9);
            test_gcd::<u32>(-46, 240, 2, -47);
        }
        #[test]
        fn t_inner_gcd_both_negative() {
            test_gcd::<u32>(-240, -46, 2, 9);
            test_gcd::<u32>(-46, -240, 2, -47);
        }
        #[test]
        fn t_bezout_both_positive() {
            test_bezout::<u32>(240, 46, 2, -9, 47);
            test_bezout::<u32>(46, 240, 2, 47, -9);
        }
        #[test]
        fn t_bezout_lhs_negative() {
            test_bezout::<u32>(-240, 46, 2, 9, 47);
            test_bezout::<u32>(-46, 240, 2, -47, -9);
        }
        #[test]
        fn t_bezout_rhs_negative() {
            test_bezout::<u32>(240, -46, 2, -9, -47);
            test_bezout::<u32>(46, -240, 2, 47, 9);
        }
        #[test]
        fn t_bezout_both_negative() {
            test_bezout::<u32>(-240, -46, 2, 9, -47);
            test_bezout::<u32>(-46, -240, 2, -47, 9);
        }

        #[test]
        fn lhs_zero() {
            let (gcd, bezout, factor) =
                Gcd::new(BigIInt::<u32>::from_digit(0), BigIInt::from_digit(1)).all();
            assert_eq!(gcd, BigIInt::from_digit(1));
            assert_eq!(
                bezout,
                BezoutCoefficients {
                    x: BigIInt::ZERO,
                    y: BigIInt::ONE
                }
            );
            assert_eq!(
                factor,
                Factors {
                    b: BigIInt::ONE,
                    a: BigIInt::ZERO
                }
            );
        }
        #[test]
        fn rhs_zero() {
            let (gcd, bezout, factor) = Gcd::new(BigIInt::<u32>::ONE, BigIInt::ZERO).all();
            assert_eq!(gcd, BigIInt::ONE);
            assert_eq!(
                bezout,
                BezoutCoefficients {
                    x: BigIInt::ONE,
                    y: BigIInt::ZERO
                }
            );
            assert_eq!(
                factor,
                Factors {
                    b: BigIInt::ZERO,
                    a: BigIInt::ONE
                }
            );
        }
        #[test]
        fn both_zero() {
            let (gcd, bezout, factor) = Gcd::new(BigIInt::<u32>::ZERO, BigIInt::ZERO).all();
            assert_eq!(gcd, BigIInt::ZERO);
            assert_eq!(
                bezout,
                BezoutCoefficients {
                    x: BigIInt::ZERO,
                    y: BigIInt::ZERO
                }
            );
            assert_eq!(
                factor,
                Factors {
                    b: BigIInt::ZERO,
                    a: BigIInt::ZERO
                }
            );
        }
    }
}
