from starccato_jax import StarccatoVAE
from starccato_sampler.sampler import sample
import jax
import paths
import os

RNG = jax.random.PRNGKey(0)
NOISE_SIGMA = 2


def main():
    vae = StarccatoVAE()
    true_z = jax.random.normal(RNG, (1, vae.latent_dim))
    true_signal = vae.generate(z=true_z)[0]
    data = true_signal + jax.random.normal(RNG, true_signal.shape) * NOISE_SIGMA
    samples = sample(
        data=data,
        model_path=None,
        rng_int=0,
        outdir=os.path.join(paths.data, "out_mcmc"),
        num_warmup=2000,
        num_samples=200,
        num_chains=2,
        noise_sigma=NOISE_SIGMA,
        truth=dict(signal=true_signal, latent=true_z.ravel()),
        stepping_stone_lnz=False,
    )


if __name__ == "__main__":
    main()
