from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def train_model(df):
    features = ['month', 'year', 'dayofyear']
    X = df[features]
    y = df['groundwater_level']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=12,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test
