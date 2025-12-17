import pandas as pd

def load_and_preprocess(path):
    df = pd.read_csv(path)

    # Standardize column names
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

    # Rename expected columns (adjust if column names differ slightly)
    df.rename(columns={
        'water_level_mbgl': 'groundwater_level',
        'measurement_date': 'timestamp'
    }, inplace=True)

    # Convert date
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp', 'groundwater_level'])

    # Sort time-wise
    df = df.sort_values('timestamp')

    # Handle missing groundwater values
    df['groundwater_level'] = df['groundwater_level'].interpolate()

    # Time features
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year
    df['dayofyear'] = df['timestamp'].dt.dayofyear

    return df
