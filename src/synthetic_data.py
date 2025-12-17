import numpy as np

def generate_synthetic_data(df, n_samples=800, noise_ratio=0.05):
    synthetic = df.sample(n=n_samples, replace=True).copy()

    noise = np.random.normal(
        0,
        noise_ratio * synthetic['groundwater_level'].std(),
        n_samples
    )

    synthetic['groundwater_level'] += noise
    synthetic['synthetic'] = 1

    return synthetic
