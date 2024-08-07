use std::{
    fmt::Debug,
    sync::{Arc, Mutex},
    thread::{self, sleep},
    time::Duration,
};

use rand::{CryptoRng, RngCore};

pub mod big_int;
mod boo;
mod eliptic_curve;
mod modular_arithmetic;
mod modular_potentiation;
pub trait Algo {
    type Secret: Clone;
    type Intermediate;

    fn pick_secret<Rand>(&self, rng: &mut Rand) -> Self::Secret
    where
        Rand: CryptoRng + RngCore;
    fn start(&self) -> Self::Intermediate;
    fn commute(&self, secret: Self::Secret, intermediate: Self::Intermediate)
        -> Self::Intermediate;

    fn convert(&self, intermediate: Self::Intermediate) -> u64;
}

pub fn next_in_bounds<Rand>(rng: &mut Rand, min: u64, max: u64) -> u64
where
    Rand: CryptoRng + RngCore,
{
    let percent = rng.next_u64() as f64 / u64::MAX as f64;
    ((percent * (max as f64)) as u64).checked_add(min).unwrap()
}

#[allow(dead_code)]
fn single_diffie_hellman<Rand, A>(
    rng1: Rand,
    rng2: Rand,
    algo: &A,
) -> (A::Intermediate, A::Intermediate)
where
    Rand: CryptoRng + RngCore,
    A: Algo,
{
    let place = std::cell::Cell::new(None);
    let mut key2 = None;

    let key = diffie_hellman(
        rng1,
        algo,
        || {
            let mut place2 = None;
            key2 = Some(diffie_hellman(
                rng2,
                algo,
                || place.take().expect("bob wasn't ready"),
                |it| place2 = Some(it),
            ));
            place2.expect("alice wasn't ready")
        },
        |it| {
            place.replace(Some(it));
        },
    );
    (key, key2.unwrap())
}

#[allow(dead_code)]
fn double_diffie_hellman<Rand, A>(rng1: Rand, rng2: Rand, algo: &A)
where
    Rand: CryptoRng + RngCore + Send + 'static,
    A: Algo + Clone + Send + Sync + 'static,
    A::Intermediate: PartialEq + Debug + Send + 'static,
{
    fn write<T>(tx: &Arc<Mutex<Option<T>>>, value: T) {
        let mut lock = tx.lock().unwrap();
        *lock = Some(value);
    }

    fn read<T>(rx: &Arc<Mutex<Option<T>>>) -> T {
        loop {
            let value = rx.lock().unwrap().take();
            if let Some(value) = value {
                break value;
            }
            sleep(Duration::from_millis(50));
        }
    }
    let place_a = Arc::new(Mutex::new(None));
    let place_b = Arc::new(Mutex::new(None));

    let handle = {
        let place_a = Arc::clone(&place_a);
        let place_b = Arc::clone(&place_b);
        let algo = algo.clone();

        thread::spawn(move || {
            diffie_hellman(rng1, &algo, || read(&place_a), |it| write(&place_b, it))
        })
    };

    let key = diffie_hellman(rng2, algo, || read(&place_b), |it| write(&place_a, it));
    assert_eq!(key, handle.join().unwrap());
}

pub fn diffie_hellman<Rand, A>(
    mut rng: Rand,
    algo: &A,
    rx: impl FnOnce() -> A::Intermediate,
    tx: impl FnOnce(A::Intermediate),
) -> A::Intermediate
where
    Rand: CryptoRng + RngCore,
    A: Algo,
{
    let secret = algo.pick_secret(&mut rng);

    let start = algo.start();
    let big_a = algo.commute(secret.clone(), start);

    tx(big_a);
    let big_b = rx();

    algo.commute(secret, big_b)
}

#[cfg(test)]
mod tests {
    use eliptic_curve::{ElipticCurve, Generator, Point};
    use rand::{
        rngs::{OsRng, StdRng},
        SeedableRng,
    };

    use crate::diffie_hellman;

    use super::*;

    struct MockRandom(Box<dyn Iterator<Item = u8>>);

    #[derive(Debug)]
    struct NoMoreData;
    impl std::fmt::Display for NoMoreData {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("no more data can be generated")
        }
    }
    impl std::error::Error for NoMoreData {}

    impl MockRandom {
        fn from_u64s<Iter>(data: Iter) -> Self
        where
            Iter: IntoIterator<Item = u64>,
            Iter::IntoIter: 'static,
        {
            Self(Box::new(data.into_iter().flat_map(u64::to_ne_bytes)))
        }
    }
    impl CryptoRng for MockRandom {}
    impl RngCore for MockRandom {
        fn next_u32(&mut self) -> u32 {
            let mut buf = [0; 4];
            self.try_fill_bytes(&mut buf).unwrap();
            u32::from_ne_bytes(buf)
        }

        fn next_u64(&mut self) -> u64 {
            let mut buf = [0; 8];
            self.try_fill_bytes(&mut buf).unwrap();
            u64::from_ne_bytes(buf)
        }

        fn fill_bytes(&mut self, dest: &mut [u8]) {
            self.try_fill_bytes(dest).unwrap();
        }

        fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
            for byte in dest.iter_mut() {
                *byte = self.0.next().ok_or_else(|| rand::Error::new(NoMoreData))?;
            }
            Ok(())
        }
    }

    #[test]
    fn diffie_hellman_t() {
        let curve = ElipticCurve::new(2, 2, 17);
        let g = Generator::new(Point::new(5, 1, 17), &curve).unwrap();
        let key = diffie_hellman(
            MockRandom::from_u64s(std::iter::once((8.0 / 17.0 * u64::MAX as f64) as u64)),
            &(curve, g),
            || Point::new(10, 6, 17),
            |it| {
                assert_eq!(
                    Point::new(7, 6, 17),
                    it,
                    "Bob calculated the wrong intermediate"
                );
            },
        );
        assert_eq!(Point::new(13, 7, 17), key, "Bob calculated the wrong key");

        let key = diffie_hellman(
            MockRandom::from_u64s(std::iter::once((2.0 / 17.0 * u64::MAX as f64) as u64)),
            &(curve, g),
            || Point::new(7, 6, 17),
            |it| {
                assert_eq!(
                    Point::new(10, 6, 17),
                    it,
                    "Alice calculated the wrong intermediate"
                );
            },
        );
        assert_eq!(Point::new(13, 7, 17), key, "Alice calculated the wrong key");
    }

    fn generate_array<const N: usize>(rng: &mut impl RngCore) -> Result<[u8; N], rand::Error> {
        let mut buf = [0; N];
        rng.try_fill_bytes(&mut buf)?;
        Ok(buf)
    }

    #[test]
    fn diffie_hellman_fuzz() {
        const N: usize = 100;

        let curve = ElipticCurve::new(2, 2, 17);
        let g = Generator::new(Point::new(5, 1, 17), &curve).unwrap();

        let seed = generate_array(&mut OsRng).expect("failed to generate seed");
        let mut rng = StdRng::from_seed(seed);

        for i in 0..N {
            let (key_bob, key_alice) = single_diffie_hellman(
                StdRng::from_seed(generate_array(&mut rng).expect("failed to seed bob")),
                StdRng::from_seed(generate_array(&mut rng).expect("failed to seed alice")),
                &(curve, g),
            );

            assert_eq!(
                key_bob, key_alice,
                "wrong keys generated with seed {seed:?} in iteration {i}"
            );
        }
    }

    #[test]
    fn diffie_hellman_special_1() {
        let seed = [
            167, 188, 233, 144, 226, 180, 193, 241, 245, 155, 46, 118, 192, 58, 173, 200, 245, 44,
            229, 102, 255, 78, 170, 94, 69, 6, 202, 168, 154, 109, 163, 141,
        ];
        let mut rng = StdRng::from_seed(seed);

        let curve = ElipticCurve::new(2, 2, 17);
        let g = Generator::new(Point::new(5, 1, 17), &curve).unwrap();

        let (key_bob, key_alice) = single_diffie_hellman(
            StdRng::from_seed(generate_array(&mut rng).expect("failed to seed bob")),
            StdRng::from_seed(generate_array(&mut rng).expect("failed to seed alice")),
            &(curve, g),
        );

        assert_eq!(
            key_bob, key_alice,
            "wrong keys generated with seed {seed:?}"
        );
    }

    #[test]
    fn next_in_bounds_t() {
        assert_eq!(
            9,
            next_in_bounds(
                &mut MockRandom::from_u64s(std::iter::once((8.0 / 17.0 * u64::MAX as f64) as u64)),
                1,
                17
            )
        );
    }
}
