import numpy as np
import pandas as pd

def generate_synthetic_data(df, n_samples=800, noise_ratio=0.05):
    df = df.copy()

    # 🔒 HARD SAFETY: ensure numeric groundwater level
    df['groundwater_level'] = pd.to_numeric(
        df['groundwater_level'],
        errors='coerce'
    )

    # Drop any bad rows before sampling
    df = df.dropna(subset=['groundwater_level'])

    synthetic = df.sample(n=n_samples, replace=True).copy()

    noise = np.random.normal(
        0,
        noise_ratio * synthetic['groundwater_level'].std(),
        n_samples
    )

    synthetic['groundwater_level'] += noise
    synthetic['synthetic'] = 1

    return synthetic
