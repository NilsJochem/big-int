struct ModularPotentiation;
impl crate::Algo for ModularPotentiation {
    type Secret = u64;
    type Intermediate = u64;

    fn pick_secret<Rand>(&self, _rng: &mut Rand) -> Self::Secret
    where
        Rand: rand::CryptoRng + rand::RngCore,
    {
        todo!()
    }

    fn start(&self) -> Self::Intermediate {
        todo!()
    }

    fn commute(
        &self,
        _secret: Self::Secret,
        _intermediate: Self::Intermediate,
    ) -> Self::Intermediate {
        todo!()
    }

    fn convert(&self, _intermediate: Self::Intermediate) -> u64 {
        todo!()
    }
}
