use std::ops::Mul;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Number {
    value: u64,
    modulus: u64,
}
impl std::fmt::Display for Number {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} mod ({})", self.value(), self.modulus())
    }
}
impl Number {
    pub fn from_i64(value: i64, modulus: u64) -> Self {
        assert!(i64::try_from(modulus).is_ok());
        Self::new_unchecked(value.rem_euclid(modulus as i64) as u64, modulus)
    }
    pub const fn from_u64(value: u64, modulus: u64) -> Self {
        Self::new_unchecked(value.rem_euclid(modulus), modulus)
    }
    pub const fn new_unchecked(value: u64, modulus: u64) -> Self {
        Self { value, modulus }
    }
    pub const fn value(&self) -> u64 {
        self.value
    }
    pub const fn modulus(&self) -> u64 {
        self.modulus
    }

    pub fn mul_inverse(&self) -> Option<Self> {
        let mut t = 0;
        let mut new_t = 1;
        let mut r = self.modulus() as i64;
        let mut new_r = self.value() as i64;

        while new_r != 0 {
            let quotient = r.div_euclid(new_r);
            let tmp = new_t;
            new_t = t - quotient * new_t;
            t = tmp;

            let tmp = new_r;
            new_r = r - quotient * new_r;
            r = tmp;
        }

        match () {
            () if r > 1 => None, //Number is not invertible
            () if r < 0 => Some(Self::from_i64(t + self.modulus() as i64, self.modulus())),
            () => Some(Self::from_i64(t, self.modulus())),
        }
    }
    fn div_unchecked(self, rhs: Self) -> Self {
        self.mul(rhs.mul_inverse().expect("no inverse of number").value())
    }
}

impl std::ops::Add<u64> for Number {
    type Output = Self;

    fn add(self, rhs: u64) -> Self::Output {
        if let Some(res) = self.value().checked_add(rhs) {
            Self::from_u64(res, self.modulus())
        } else {
            todo!("addition overflow occured")
        }
    }
}
impl std::ops::Add<i64> for Number {
    type Output = Self;

    fn add(self, rhs: i64) -> Self::Output {
        match u64::try_from(rhs) {
            Ok(rhs) => self.add(rhs),
            Err(_) => std::ops::Sub::sub(self, u64::try_from(-rhs).unwrap()),
        }
    }
}
impl std::ops::Add for Number {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        assert!(
            self.modulus() == rhs.modulus(),
            "tried to add with diffend modulus"
        );
        self.add(rhs.value())
    }
}
impl std::ops::Sub<u64> for Number {
    type Output = Self;

    fn sub(self, rhs: u64) -> Self::Output {
        if let Some(res) = self.value().checked_sub(rhs) {
            Self::from_u64(res, self.modulus())
        } else {
            Self::from_u64(
                self.modulus() - ((rhs - self.value()).rem_euclid(self.modulus())),
                self.modulus(),
            )
        }
    }
}
impl std::ops::Sub<i64> for Number {
    type Output = Self;

    fn sub(self, rhs: i64) -> Self::Output {
        match u64::try_from(rhs) {
            Ok(rhs) => self.sub(rhs),
            Err(_) => std::ops::Add::add(self, u64::try_from(-rhs).unwrap()),
        }
    }
}
impl std::ops::Sub for Number {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        assert!(
            self.modulus() == rhs.modulus(),
            "tried to subtract with diffend modulus"
        );
        self.sub(rhs.value())
    }
}

impl std::ops::Mul<u64> for Number {
    type Output = Self;

    fn mul(self, rhs: u64) -> Self::Output {
        if let Some(res) = self.value.checked_mul(rhs) {
            Self::from_u64(res, self.modulus())
        } else {
            todo!("multiplication overflow occured")
        }
    }
}
impl std::ops::Mul for Number {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        assert!(
            self.modulus() == rhs.modulus(),
            "tried to multiply with diffend modulus"
        );
        self.mul(rhs.value())
    }
}

impl std::ops::Div<i64> for Number {
    type Output = Self;

    fn div(self, rhs: i64) -> Self::Output {
        self.div_unchecked(Self::from_i64(rhs, self.modulus()))
    }
}
impl std::ops::Div for Number {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        assert!(
            self.modulus() == rhs.modulus(),
            "tried to divide with diffend modulus"
        );
        self.div_unchecked(rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add() {
        assert_eq!(Number::from_u64(5, 17), Number::from_u64(1, 17) + 4u64);
        assert_eq!(Number::from_u64(16, 17), Number::from_u64(1, 17) + 15u64);
    }
    #[test]
    fn add_overflow() {
        assert_eq!(Number::from_u64(0, 17), Number::from_u64(5, 17) + 12u64);
        assert_eq!(Number::from_u64(16, 17), Number::from_u64(1, 17) + 32u64);
    }

    #[test]
    fn sub() {
        assert_eq!(Number::from_u64(1, 17), Number::from_u64(5, 17) - 4u64);
        assert_eq!(Number::from_u64(15, 17), Number::from_u64(16, 17) - 1u64);
    }
    #[test]
    fn sub_overflow() {
        assert_eq!(Number::from_u64(0, 17), Number::from_u64(5, 17) - 22u64);
        assert_eq!(Number::from_u64(1, 17), Number::from_u64(16, 17) - 32u64);
    }

    #[test]
    fn mul() {
        assert_eq!(Number::from_u64(6, 17), Number::from_u64(2, 17) * 3);
        assert_eq!(Number::from_u64(16, 17), Number::from_u64(4, 17) * 4);
    }
    #[test]
    fn mul_overflow() {
        assert_eq!(Number::from_u64(0, 17), Number::from_u64(5, 17) * 17);
        assert_eq!(Number::from_u64(11, 17), Number::from_u64(15, 17) * 3);
    }

    #[test]
    fn inverse() {
        assert_eq!(
            Some(Number::from_u64(9, 17)),
            Number::from_u64(2, 17).mul_inverse()
        );
    }
}
