# -*- coding: utf-8 -*-
"""
train.py
========
پایپ‌لاین کامل train: از فایل خام divar_ads_full.csv تا مدل نهایی LightGBM.

اجرا:
    python train.py --csv data/divar_ads_full.csv --out models/artifacts.joblib

خروجی: یک فایل joblib شامل مدل + تمام artifact های لازم برای inference
(TF-IDF vectorizer ها، TargetEncoder، لیست دسته‌های categorical، میانه‌های
train برای imputation و ترتیب دقیق ستون‌های فیچر).

نکته: نسبت به نوت‌بوک اصلی یک بهبود کوچک اعمال شده — TF-IDF روی «فقط»
train split فیت می‌شود (نه کل دیتاست) تا از نشت اطلاعات ولیدیشن/تست به
واژگان مدل جلوگیری شود.
"""

import argparse
import json

import joblib
import jdatetime
import lightgbm as lgb
import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import preprocessing as pp

COLUMN_RENAME_MAP = {
    "عنوان": "title",
    "دسته بندی": "category",
    "توضیحات کامل": "description",
    "اتاق": "rooms",
    "اجارهٔ ماهانه": "monthly_rent",
    "طبقه": "floor",
    "قیمت کل": "total_price",
    "قیمت هر متر": "price_per_meter",
    "متراژ": "area",
    "متراژ زمین": "land_area",
    "ودیعه": "deposit",
    "ودیعه و اجاره": "deposit_and_rent_type",
    "ویژگی ها": "features",
    "زمین": "land",
    "ویلا": "villa",
    "آپارتمان": "apartment",
    "مغازه": "shop",
    "تجاری": "commercial",
    "ساخت": "builded_date",
    "آدرس": "location",
}

INITIAL_DROP_COLUMNS = [
    "ردیف", "کلید", "لینک", "زمان و مکان", "آخر هفته", "روزهای عادی",
    "زمان خاتمه", "سازنده", "شروع قیمت", "ظرفیت", "کمترین متراژ",
    "هزینهٔ هر نفرِ اضافه", "تعطیلات و مناسبت\u200cها", "تعداد عکس",
    "پوشه عکس\u200cها", "آدرس عکس\u200cها", "تصویر نقشه",
    "تصویر\u200cها برای همین ملک است؟", "اسکریپ جزییات", "پوشه عکس ها",
    "آدرس عکس ها", "تأیید حضوری در تاریخ", "اطلاعات هویتی",
    "مالکیت موبایل", "مطابقت چهره", "پیشرفت فیزیکی کل پروژه", "تحویل",
    "پرداختی در زمان تحویل", "پیش پرداخت اولیه", "وضعیت فعلی پروژه",
    "قیمت پایه برای هر متر مربع", "نوع واحد\u200cها", "تاریخ استعلام",
    "وضعیت", "قیمت", "تعداد اتاق",
]

NUMERIC_COLS_TO_CLEAN = [
    "monthly_rent", "floor", "total_price", "price_per_meter",
    "area", "land_area", "deposit", "rooms", "builded_date",
]

INSTALLMENT_KEYWORDS = ["قسطی", "اقساط", "سفتکاری", "قسط", "امتیاز"]

# محدوده‌های حذف داده پرت (دقیقاً مطابق نوت‌بوک)
PRICE_MIN, PRICE_MAX = 1.0e8, 1.0e13
AREA_MAX = 500
AGE_MAX = 100
FLOOR_MAX = 15.0


def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)
    print(f"[1/8] داده خام لود شد: {df.shape}")

    df = df.drop(columns=INITIAL_DROP_COLUMNS, errors="ignore")
    df = df.rename(columns=COLUMN_RENAME_MAP)

    for col in NUMERIC_COLS_TO_CLEAN:
        df = pp.clean_and_convert_numeric_column(df, col)

    # فقط آپارتمان‌های فروشی (مطابق فیلتر نوت‌بوک اصلی)
    def category_third_part(cat):
        if pd.isna(cat):
            return None
        parts = cat.split(">")
        return parts[2].strip() if len(parts) > 2 else None

    third_part = df["category"].apply(category_third_part)
    df = df[third_part.str.contains("فروش آپارتمان", na=False, case=False)].copy()
    print(f"[2/8] فیلتر به آپارتمان‌های فروشی: {df.shape}")

    df["rooms"] = df["rooms"].fillna(-1)
    df["floor"] = df["floor"].fillna(-1)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = pp.extract_parking_elevator_storage(df, features_col="features")
    df = pp.extract_materials_and_amenities(df, description_col="description")
    df = pp.extract_location(df, location_col="location")

    current_shamsi_year = jdatetime.date.today().year
    df["age"] = pp.calculate_age(df["builded_date"], current_shamsi_year)

    # حذف آگهی‌های قسطی/سفت‌کاری از روی متن توضیحات
    pattern = "|".join(INSTALLMENT_KEYWORDS)
    mask = df["description_processed"].astype(str).str.contains(pattern, case=False, na=False)
    df = df[~mask].copy()
    print(f"[3/8] حذف آگهی‌های قسطی/سفت‌کاری: {df.shape}")

    # فیلتر داده پرت
    initial = len(df)
    cond = (
        (df["total_price"] >= PRICE_MIN)
        & (df["total_price"] <= PRICE_MAX)
        & (df["age"] <= AGE_MAX)
        & (df["age"] >= 0)
        & (df["area"] <= AREA_MAX)
        & (df["floor"] <= FLOOR_MAX)
        & (df["floor"] >= -1)
    )
    df = df[cond].copy()
    df["log_total_price"] = np.log1p(df["total_price"])

    q1, q3 = df["log_total_price"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df = df[(df["log_total_price"] >= lower) & (df["log_total_price"] <= upper)].copy()
    print(f"[4/8] حذف داده پرت (IQR روی log_total_price): {initial} -> {len(df)}")

    df = pp.make_city_district(df)

    return df


def build_xy(df: pd.DataFrame):
    top_city_districts = set(df["city_district"].value_counts().nlargest(pp.TOP_N_CITY_DISTRICTS).index)
    df["city_district"] = pp.apply_rare_location_mapping(df["city_district"], top_city_districts)

    X = pp.build_raw_feature_frame(df)
    y = df["log_total_price"].copy()

    categorical_categories = {}
    for col in ["cabinet_type", "floor_type", "wall_type", "cooler_type", "property_age_status"]:
        X[col] = X[col].astype("category")
        categorical_categories[col] = list(X[col].cat.categories)

    for col in pp.NUMERIC_AND_BINARY:
        X[col] = X[col].replace(-1, np.nan)

    aux = df[["description_processed"]].copy()

    return X, y, aux, top_city_districts, categorical_categories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/divar_ads_full.csv")
    parser.add_argument("--out", default="models/artifacts.joblib")
    args = parser.parse_args()

    df = load_and_clean(args.csv)
    df = engineer_features(df)
    X, y, aux, top_city_districts, categorical_categories = build_xy(df)

    # --- تقسیم داده (85% train+val / 15% test, سپس 20% از باقیمانده برای val) ---
    X_train_val, X_test, y_train_val, y_test, aux_train_val, aux_test = train_test_split(
        X, y, aux, test_size=0.15, random_state=42
    )
    X_train, X_val, y_train, y_val, aux_train, aux_val = train_test_split(
        X_train_val, y_train_val, aux_train_val, test_size=0.20, random_state=42
    )
    print(f"[5/8] تقسیم داده -> train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")

    # --- TF-IDF: فقط روی train فیت می‌شود (بهبود نسبت به نوت‌بوک اصلی) ---
    tfidf_description = TfidfVectorizer(
        max_features=300, min_df=5, max_df=0.7, ngram_range=(1, 2), sublinear_tf=True
    )
    tfidf_description.fit(aux_train["description_processed"].fillna(""))

    X_train = pp.add_tfidf_features(X_train, aux_train, tfidf_description)
    X_val = pp.add_tfidf_features(X_val, aux_val, tfidf_description)
    X_test = pp.add_tfidf_features(X_test, aux_test, tfidf_description)
    print(f"[6/8] TF-IDF اضافه شد -> ستون‌های نهایی: {X_train.shape[1]}")

    # --- Target Encoding برای city_district ---
    te = TargetEncoder(cols=["city_district"])
    X_train["city_district"] = te.fit_transform(X_train[["city_district"]], y_train)
    X_val["city_district"] = te.transform(X_val[["city_district"]])
    X_test["city_district"] = te.transform(X_test[["city_district"]])

    # --- Imputation با میانه‌ی train ---
    train_medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(train_medians)
    X_val = X_val.fillna(train_medians)
    X_test = X_test.fillna(train_medians)

    feature_columns = list(X_train.columns)

    # --- آموزش مدل نهایی ---
    categorical_for_lgbm = [c for c in X_train.columns if X_train[c].dtype.name == "category"]
    model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=1000,
        learning_rate=0.01,
        num_leaves=40,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train.ravel() if hasattr(y_train, "ravel") else y_train.values,
        eval_set=[(X_val, y_val.values)],
        categorical_feature=categorical_for_lgbm,
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    print("[7/8] مدل LightGBM آموزش دید.")

    # --- ارزیابی روی تست ---
    y_pred = model.predict(X_test)
    actual = np.expm1(y_test.values)
    predicted = np.expm1(y_pred)
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-10))) * 100
    rmse = np.sqrt(np.mean((y_test.values - y_pred) ** 2))
    print(f"[8/8] ارزیابی روی Test -> MAPE: {mape:.2f}%  |  RMSE(log): {rmse:.4f}")

    artifacts = {
        "model": model,
        "tfidf_description": tfidf_description,
        "target_encoder": te,
        "top_city_districts": list(top_city_districts),
        "categorical_categories": categorical_categories,
        "train_medians": train_medians,
        "feature_columns": feature_columns,
        "metrics": {"mape": float(mape), "rmse_log": float(rmse), "n_train": len(X_train)},
    }
    joblib.dump(artifacts, args.out)
    print(f"\nهمه چیز در {args.out} ذخیره شد.")
    print(json.dumps(artifacts["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
