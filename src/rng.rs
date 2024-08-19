use rand::RngCore;

#[cfg(test)]
pub fn generate_array<const N: usize>(rng: &mut impl RngCore) -> Result<[u8; N], rand::Error> {
    let mut buf = [0; N];
    rng.try_fill_bytes(&mut buf)?;
    Ok(buf)
}
pub fn random_bytes<'r>(mut rng: impl RngCore + 'r) -> impl Iterator<Item = u8> + 'r {
    std::iter::from_fn(move || Some(rng.next_u32())).flat_map(u32::to_ne_bytes)
}
pub fn next_usize(mut rng: impl RngCore) -> usize {
    if cfg!(target_pointer_width = "64") {
        rng.next_u64() as usize
    } else if cfg!(target_pointer_width = "32") {
        rng.next_u32() as usize
    } else {
        unimplemented!()
    }
}

pub fn next_bound(
    bound: usize,
    mut rng: impl RngCore,
    max_tries: impl Into<Option<usize>>,
) -> usize {
    if bound == 0 {
        return 0;
    }
    let mask = (1usize << (bound.ilog2() + 1)) - 1;
    if let Some(max_tries) = max_tries.into() {
        for _ in 0..max_tries {
            let pick = next_usize(&mut rng) & mask;
            if pick <= bound {
                return pick;
            }
        }
        panic!("to many tries");
    } else {
        loop {
            let pick = next_usize(&mut rng) & mask;
            if pick <= bound {
                return pick;
            }
        }
    }
}
#[allow(clippy::module_name_repetitions)]
#[cfg(test)]
pub fn seeded_rng() -> ([u8; 32], rand::rngs::StdRng) {
    let seed = generate_array(&mut rand::rngs::OsRng).expect("failed to generate seed");
    let rng = <rand::rngs::StdRng as rand::SeedableRng>::from_seed(seed);
    (seed, rng)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fuzz_next_bound() {
        const TRIES: usize = 100_000;
        const MAX: usize = 13;
        const DEVIATON: f64 = 0.04;

        let (seed, mut rng) = seeded_rng();

        let mut hits = [0u32; MAX + 1];
        for _ in 0..TRIES {
            hits[next_bound(MAX, &mut rng, None)] += 1;
        }
        let avg = TRIES as f64 / (MAX + 1) as f64;
        let lower_barrier = (avg * (1.0 - DEVIATON)) as u32;
        let upper_barrier = (avg * (1.0 + DEVIATON)) as u32;

        for (i, hit) in hits.iter().copied().enumerate() {
            assert!(
                lower_barrier <= hit && hit <= upper_barrier,
                "{i} was hit {lower_barrier} <= {hit} <= {upper_barrier}; rest is {hits:?} with seed {seed:?}"
            );
        }
    }
}
