# -*- coding: utf-8 -*-
"""
api.py
======
FastAPI backend. مدل و artifact ها را یک‌بار در startup لود می‌کند و روی
اندپوینت POST /predict قیمت را برمی‌گرداند.

اجرا (لوکال):
    uvicorn api:app --reload --port 8000
"""

import os

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import preprocessing as pp

ARTIFACTS_PATH = os.environ.get("ARTIFACTS_PATH", "models/artifacts.joblib")

app = FastAPI(title="House Price Prediction API", version="1.0.0")

_artifacts = None


@app.on_event("startup")
def load_artifacts():
    global _artifacts
    if not os.path.exists(ARTIFACTS_PATH):
        raise RuntimeError(
            f"فایل مدل در مسیر {ARTIFACTS_PATH} پیدا نشد. اول باید train.py را اجرا کنید."
        )
    _artifacts = joblib.load(ARTIFACTS_PATH)


class PredictRequest(BaseModel):
    features_text: str = Field(
        "", description="ویژگی‌ها (اختیاری) مثل: پارکینگ، انباری، آسانسور", example="پارکینگ، انباری، آسانسور"
    )
    description: str = Field(
        "", description="توضیحات کامل آگهی (اختیاری ولی توصیه می‌شود)",
        example="واحد نوساز با کابینت ام‌دی‌اف و کف سرامیک، دارای بالکن و دوربین مداربسته",
    )
    rooms: int | None = Field(None, description="تعداد اتاق (خالی = نامشخص)", example=2)
    floor: int | None = Field(None, description="طبقه (خالی = نامشخص)", example=3)
    area: float = Field(..., gt=0, description="متراژ به متر مربع", example=80)
    built_year_shamsi: int | None = Field(
        None, description="سال ساخت شمسی (خالی = نامشخص)", example=1399
    )
    city: str = Field(..., description="شهر", example="اصفهان")
    district: str = Field("", description="محله / منطقه", example="ملک‌شهر")


class PredictResponse(BaseModel):
    predicted_price_toman: float
    predicted_price_per_meter_toman: float
    currency: str = "IRT"
    note: str = ""


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _artifacts is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _artifacts is None:
        raise HTTPException(status_code=503, detail="مدل هنوز لود نشده است.")

    import jdatetime

    current_shamsi_year = jdatetime.date.today().year
    built_year = req.built_year_shamsi if req.built_year_shamsi is not None else np.nan

    raw_row = {
        "features_text": req.features_text,
        "description": req.description,
        "rooms": req.rooms if req.rooms is not None else np.nan,
        "floor": req.floor if req.floor is not None else np.nan,
        "area": req.area,
        "built_year_shamsi": built_year,
        "current_shamsi_year": current_shamsi_year,
        "city": req.city.strip(),
        "district": req.district.strip() if req.district else "",
    }

    try:
        X = pp.build_feature_row(raw_row, _artifacts)
        log_price = _artifacts["model"].predict(X)[0]
        price = float(np.expm1(log_price))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"خطا در پیش‌بینی: {exc}") from exc

    price_per_meter = price / req.area if req.area else 0.0

    note = ""
    city_district = f"{raw_row['city']}_{raw_row['district']}"
    if city_district not in set(_artifacts["top_city_districts"]):
        note = "این منطقه در دیتاست آموزشی کم‌تکرار بوده؛ دقت پیش‌بینی برای این منطقه ممکن است کمتر باشد."

    return PredictResponse(
        predicted_price_toman=round(price),
        predicted_price_per_meter_toman=round(price_per_meter),
        note=note,
    )
