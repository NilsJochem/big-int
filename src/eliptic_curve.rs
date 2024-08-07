use std::iter;

use crate::{modular_arithmetic::Number, next_in_bounds};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Point {
    Actual { x: Number, y: Number },
    Infinity,
}
impl std::fmt::Debug for Point {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Actual { x, y } => {
                write!(f, "({}, {}) mod {}", x.value(), y.value(), x.modulus())
            }
            Self::Infinity => write!(f, "Infinity"),
        }
    }
}

impl Point {
    pub fn new(x: i64, y: i64, modulus: u64) -> Self {
        Self::new_unchecked(Number::from_i64(x, modulus), Number::from_i64(y, modulus))
    }
    pub fn from_numbers(x: Number, y: Number) -> Self {
        assert_eq!(x.modulus(), y.modulus());
        Self::new_unchecked(x, y)
    }
    pub const fn new_unchecked(x: Number, y: Number) -> Self {
        Self::Actual { x, y }
    }
    pub const fn modulus(&self) -> Option<u64> {
        match self {
            Self::Actual { x, y: _ } => Some(x.modulus()),
            Self::Infinity => None,
        }
    }
}
#[derive(Debug, Clone, Copy)]
pub struct ElipticCurve {
    a: Number,
    #[allow(dead_code)]
    b: Number,
}

impl ElipticCurve {
    pub fn new(a: i64, b: i64, modulus: u64) -> Self {
        Self {
            a: Number::from_i64(a, modulus),
            b: Number::from_i64(b, modulus),
        }
    }
    pub const fn modulus(&self) -> u64 {
        self.a.modulus()
    }

    pub fn generate(&self, start: Point) -> impl Iterator<Item = Point> + '_ {
        let mut point = start;
        iter::once(start).chain(iter::from_fn(move || {
            if matches!(point, Point::Infinity) {
                None
            } else {
                point = self.add(start, point);
                Some(point)
            }
        }))
    }

    pub fn add(&self, p: Point, q: Point) -> Point {
        if p == q {
            return self.double(p);
        }
        match (p, q) {
            (Point::Actual { x: px, y: _ }, Point::Actual { x: qx, y: _ }) if px == qx => {
                Point::Infinity
            }
            (Point::Actual { x: px, y: py }, Point::Actual { x: qx, y: qy }) => {
                let s = (py - qy) / (px - qx);
                #[allow(clippy::suspicious_operation_groupings)]
                let rx = s * s - (px + qx);
                Point::Actual {
                    x: rx,
                    y: s * (px - rx) - py,
                }
            }
            _ => Point::Infinity,
        }
    }
    pub fn double(&self, p: Point) -> Point {
        match p {
            Point::Actual { x, y: _ } if x.value() == 0 => Point::Infinity,
            Point::Actual { x, y } => {
                let s = (self.a + (x * x * 3)) / (y * 2);
                let rx = s * s - x * 2;
                Point::Actual {
                    x: rx,
                    y: (s * (x - rx) - y),
                }
            }
            Point::Infinity => Point::Infinity,
        }
    }
    pub fn scale(&self, p: Point, k: u64) -> Point {
        let mut res = p;
        for _ in 1..k {
            if matches!(res, Point::Infinity) {
                break;
            }
            res = self.add(res, p);
        }
        res
    }
    #[allow(clippy::unused_self)]
    pub const fn verfiy(&self, _p: Point) -> bool {
        //Todo
        true
    }
}

impl crate::Algo for (ElipticCurve, Generator) {
    type Secret = u64;
    type Intermediate = Point;

    fn pick_secret<Rand>(&self, rng: &mut Rand) -> Self::Secret
    where
        Rand: rand::CryptoRng + rand::RngCore,
    {
        next_in_bounds(rng, 1, self.1.order() - 1)
    }
    fn start(&self) -> Self::Intermediate {
        self.1.start()
    }
    fn convert(&self, intermediate: Self::Intermediate) -> u64 {
        match intermediate {
            Point::Actual { x, y: _ } => x.value(),
            Point::Infinity => panic!(),
        }
    }

    fn commute(
        &self,
        secret: Self::Secret,
        intermediate: Self::Intermediate,
    ) -> Self::Intermediate {
        self.0.scale(intermediate, secret)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Generator {
    start: Point,
    order: u64,
    cofactor: f64,
}
impl Generator {
    pub fn new(start: Point, curve: &ElipticCurve) -> Option<Self> {
        assert_eq!(start.modulus(), Some(curve.modulus()));
        let order = curve.generate(start).count() as u64;

        // use of Hasse's theorem for cofactor calculation
        (order > (4.0 * (curve.modulus() as f64).sqrt()) as u64).then(|| Self {
            start,
            order,
            cofactor: ((curve.modulus() + 1) as f64 / order as f64).round(),
        })
    }
    pub const fn start(&self) -> Point {
        self.start
    }
    pub const fn order(&self) -> u64 {
        self.order
    }
    pub const fn cofactor(&self) -> f64 {
        self.cofactor
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn double() {
        let curve = ElipticCurve::new(2, 2, 17);
        assert_eq!(Point::new(6, 3, 17), curve.double(Point::new(5, 1, 17)));
    }
    #[test]
    fn mul() {
        let curve = ElipticCurve::new(2, 2, 17);
        assert_eq!(Point::new(7, 6, 17), curve.scale(Point::new(5, 1, 17), 9));
    }
    #[test]
    fn add() {
        let curve = ElipticCurve::new(2, 2, 17);
        assert_eq!(
            Point::new(9, 16, 17),
            curve.add(Point::new(5, 1, 17), Point::new(3, 1, 17))
        );
    }

    #[test]
    fn generate() {
        let curve = ElipticCurve::new(2, 2, 17);
        assert_eq!(
            vec![
                Point::new(5, 1, 17),
                Point::new(6, 3, 17),
                Point::new(10, 6, 17),
                Point::new(3, 1, 17),
                Point::new(9, 16, 17),
                Point::new(16, 13, 17),
                Point::new(0, 6, 17),
                Point::new(13, 7, 17),
                Point::new(7, 6, 17),
                Point::new(7, 11, 17),
                Point::new(13, 10, 17),
                Point::new(0, 11, 17),
                Point::new(16, 4, 17),
                Point::new(9, 1, 17),
                Point::new(3, 16, 17),
                Point::new(10, 11, 17),
                Point::new(6, 14, 17),
                Point::new(5, 16, 17),
                Point::Infinity
            ],
            curve
                .generate(Point::new(5, 1, 17))
                .take(20)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn create_generator() {
        let curve = ElipticCurve::new(2, 2, 17);
        let gen = Generator::new(Point::new(5, 1, 17), &curve);

        assert_eq!(
            Some(Generator {
                start: Point::new(5, 1, 17),
                order: 19,
                cofactor: 1.0
            }),
            gen
        );
    }
}
