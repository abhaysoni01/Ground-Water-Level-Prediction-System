from src.preprocessing import load_and_preprocess
from src.model import train_model
from src.evaluation import evaluate

df = load_and_preprocess("data/Atal_Jal_Disclosed_Ground_Water_Level-2015-2022.csv")

model, X_test, y_test = train_model(df)
preds = model.predict(X_test)

mae, rmse = evaluate(y_test, preds)
print("MAE:", mae)
print("RMSE:", rmse)
