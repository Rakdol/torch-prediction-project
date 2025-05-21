from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import torch
import joblib
import numpy as np
from src.model.base_model import DeepResidualMLP

app = FastAPI()


## Test
# curl -X POST http://localhost:8000/predict/load \
#   -H "Content-Type: application/json" \
#   -d '{"data": [{"load_id": 1.0, "temperature": 9.4, "humidity": 40.0, "hour": 15.0, "month": 5.0, "dayofweek": 6.0, "quarter": 2.0, "dayofyear": 138.0, "dayofmonth": 18.0, "lag_1 days": 5072.3325, "lag_2 days": 7006.455, "lag_3 days": 7072.6675, "lag_7 days": 3449.255, "hour_mean": 5638.08475, "hour_std": 1837.5093718172157, "dayweek_hour_mean": 3399.3985000000002}]}'


# curl -X POST http://localhost:8000/predict/solar \
#   -H "Content-Type: application/json" \
#   -d '{"data": [{ "solar_id": 5, "temperature": 19.9, "humidity": 86.0, "cloud_cover": "Cloudy", "lag_1 days": 161.44648362717203, "lag_2 days": 10.81637092956853, "lag_3 days": 184.5831062534172, "lag_7 days": 93.0038740094482, "hour_mean": 138.37783855421733, "hour_std": 67.94128870592229, "cloud_hour_mean": 164.5389219412978, "hour_sin": -0.816969893010442, "hour_cos": -0.5766803221148672, "month_sin": 0.49999999999999994, "month_cos": -0.8660254037844387, "dayofweek_sin": -2.4492935982947064e-16, "dayofweek_cos": 1.0, "quarter_sin": 1.2246467991473532e-16, "quarter_cos": -1.0, "dayofyear_sin": 0.6932812268869777, "dayofyear_cos": -0.7206671495538609, "dayofmonth_sin": -0.4853019625310808, "dayofmonth_cos": -0.8743466161445822, "wind_x": -1.761902219173649,"wind_y": 1.4784115022790407}]}'

# --- 설정 ---
TASKS = {
    "load": {
        "model_path": "artifacts/load/model.pt",
        "preprocessor_path": "artifacts/load/preprocessor.pkl",
        "sample_input": {
            "load_id": 1.0,
            "temperature": 9.4,
            "humidity": 40.0,
            "hour": 15.0,
            "month": 5.0,
            "dayofweek": 6.0,
            "quarter": 2.0,
            "dayofyear": 138.0,
            "dayofmonth": 18.0,
            "lag_1 days": 5072.3325,
            "lag_2 days": 7006.455,
            "lag_3 days": 7072.6675,
            "lag_7 days": 3449.255,
            "hour_mean": 5638.08475,
            "hour_std": 1837.5093718172157,
            "dayweek_hour_mean": 3399.3985000000002,
        },
    },
    "solar": {
        "model_path": "artifacts/solar/model.pt",
        "preprocessor_path": "artifacts/solar/preprocessor.pkl",
        "sample_input": {
            "solar_id": 5,
            "temperature": 19.9,
            "humidity": 86.0,
            "cloud_cover": "Cloudy",
            "lag_1 days": 161.44648362717203,
            "lag_2 days": 10.81637092956853,
            "lag_3 days": 184.5831062534172,
            "lag_7 days": 93.0038740094482,
            "hour_mean": 138.37783855421733,
            "hour_std": 67.94128870592229,
            "cloud_hour_mean": 164.5389219412978,
            "hour_sin": -0.816969893010442,
            "hour_cos": -0.5766803221148672,
            "month_sin": 0.49999999999999994,
            "month_cos": -0.8660254037844387,
            "dayofweek_sin": -2.4492935982947064e-16,
            "dayofweek_cos": 1.0,
            "quarter_sin": 1.2246467991473532e-16,
            "quarter_cos": -1.0,
            "dayofyear_sin": 0.6932812268869777,
            "dayofyear_cos": -0.7206671495538609,
            "dayofmonth_sin": -0.4853019625310808,
            "dayofmonth_cos": -0.8743466161445822,
            "wind_x": -1.761902219173649,
            "wind_y": 1.4784115022790407,
        },
    },
}


LOADED_TASKS = {}


# --- 입력 스키마 정의 ---
class PredictRequest(BaseModel):
    data: List[Dict[str, Any]]


# --- 초기화 시 모델 및 전처리기 로딩 ---
@app.on_event("startup")
def load_all_tasks():
    for task_name, config in TASKS.items():
        preprocessor = joblib.load(config["preprocessor_path"])
        dummy_input = preprocessor.transform(pd.DataFrame([config["sample_input"]]))
        input_dim = dummy_input.shape[1]
        model = DeepResidualMLP(input_dim)
        model.load_state_dict(torch.load(config["model_path"]))
        model.eval()
        LOADED_TASKS[task_name] = {"model": model, "preprocessor": preprocessor}


@app.get("/ready")
def readiness_check():
    missing = [task for task in ["load", "solar"] if task not in LOADED_TASKS]
    if missing:
        raise HTTPException(status_code=503, detail=f"Model(s) not ready: {missing}")
    return {"status": "ready"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


# --- 예측 API ---
@app.post("/predict/{task_name}")
def predict(task_name: str, request: PredictRequest):
    if task_name not in LOADED_TASKS:
        raise HTTPException(status_code=404, detail=f"Unknown task: {task_name}")

    try:
        task = LOADED_TASKS[task_name]
        df_input = pd.DataFrame(request.data)
        x_input = task["preprocessor"].transform(df_input)
        x_tensor = torch.tensor(x_input, dtype=torch.float32)

        with torch.no_grad():
            predictions = task["model"](x_tensor).squeeze().tolist()

        return {"task": task_name, "prediction": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
