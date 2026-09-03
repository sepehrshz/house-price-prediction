# -*- coding: utf-8 -*-
"""
preprocessing.py
=================
تمام منطق پاکسازی و فیچر انجینیرینگ که از روی نوت‌بوک اصلی استخراج شده.
این ماژول توسط دو جا استفاده می‌شود:
  - train.py   : روی کل دیتاست CSV اجرا می‌شود (چند صد هزار ردیف)
  - api.py     : روی یک ردیف تکی (ورودی کاربر از داشبورد) اجرا می‌شود

نکته مهم طراحی: تابع build_feature_row() قلب مشترک هر دو مسیر است تا
تضمین شود دقیقاً همان تبدیلاتی که مدل با آن‌ها train شده، حین inference
هم اعمال می‌شود (جلوگیری از train/serve skew).
"""

import re
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# ثابت‌ها - نام فیچرهای نهایی مدل (دقیقاً همان‌هایی که در نوت‌بوک استفاده شد)
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = ["rooms", "floor", "area", "age"]

CATEGORICAL_FEATURES = [
    "cabinet_type",
    "floor_type",
    "wall_type",
    "cooler_type",
    "property_age_status",
    "city_district",
]

BINARY_FEATURES = [
    "has_yard",
    "has_balcony",
    "has_cctv",
    "has_kanaf_ceiling",
    "has_double_glazed_windows",
    "near_mosque_school",
    "parking",
    "elevator",
    "storage",
    "is_raw",
]

# فیچرهایی که مقدار -1 (نامشخص) در آن‌ها باید قبل از target encoding به NaN تبدیل شود
NUMERIC_AND_BINARY = NUMERIC_FEATURES + BINARY_FEATURES

RAW_FEATURE_ORDER = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

TOP_N_CITY_DISTRICTS = 160  # همان مقدار نوت‌بوک (nlargest(160))

# ---------------------------------------------------------------------------
# 1) اعداد فارسی -> انگلیسی و پاکسازی ستون‌های عددی
# ---------------------------------------------------------------------------
_PERSIAN_DIGITS = {
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
}


def persian_to_english_numerals(text):
    if pd.isna(text) or not isinstance(text, str):
        return text
    return "".join(_PERSIAN_DIGITS.get(ch, ch) for ch in text)


def clean_and_convert_numeric_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    if column_name in df.columns:
        df[column_name] = df[column_name].astype(str)
        df[column_name] = df[column_name].apply(persian_to_english_numerals)
        df[column_name] = df[column_name].str.replace(r"[^۰-۹0-9.]", "", regex=True)
        df[column_name] = pd.to_numeric(df[column_name], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# 2) استخراج پارکینگ / آسانسور / انباری از متن "ویژگی‌ها"
# ---------------------------------------------------------------------------
def extract_parking_elevator_storage(df: pd.DataFrame, features_col: str = "features") -> pd.DataFrame:
    df["parking"] = 0
    df["elevator"] = 0
    df["storage"] = 0

    if features_col not in df.columns:
        return df

    processed = df[features_col].astype(str).fillna("NaN_Feature_Placeholder")
    missing_mask = processed == "NaN_Feature_Placeholder"
    df.loc[missing_mask, ["parking", "elevator", "storage"]] = -1

    df.loc[processed.str.contains("پارکینگ", case=False, na=False), "parking"] = 1
    df.loc[processed.str.contains("پارکینگ ندارد", case=False, na=False), "parking"] = 0

    df.loc[processed.str.contains("آسانسور", case=False, na=False), "elevator"] = 1
    df.loc[processed.str.contains("آسانسور ندارد", case=False, na=False), "elevator"] = 0

    df.loc[processed.str.contains("انباری", case=False, na=False), "storage"] = 1
    df.loc[processed.str.contains("انباری ندارد", case=False, na=False), "storage"] = 0

    return df


# ---------------------------------------------------------------------------
# 3) استخراج متریال‌ها (کابینت/کف/دیوار) و امکانات از متن "توضیحات"
# ---------------------------------------------------------------------------
_CABINET_KEYWORDS = {
    "mdf": ["ام دی اف", "MDF", "mdf", "آم دی اف"],
    "high_glass": ["های گلاس", "هایگلاس", "هایگلس"],
    "metal": ["فلز", "فلزی"],
    "galvanized": ["گالوانیزه", "گالوانیز"],
    "aluminum": ["آلومینیوم", "آلمینیوم", "الومینیوم"],
}
_FLOOR_KEYWORDS = {"ceramic": ["سرامیک"], "parquet": ["پارکت"]}
_WALL_KEYWORDS = {"plaster": ["گچ", "گچی"], "wallpaper": ["کاغذ دیواری"]}
_COOLER_KEYWORDS = {
    "split_ac": ["کولر گازی", "اسپیلت", "اسپلیت"],
    "evaporative_cooler": ["کولر آبی", "آبی"],
}


def _extract_type_from_text(df: pd.DataFrame, feature_col: str, text_col: str, keyword_map: dict) -> pd.DataFrame:
    for type_name, keywords in keyword_map.items():
        pattern = "|".join(keywords)
        mask = df[text_col].astype(str).str.contains(pattern, case=False, na=False) & (df[feature_col] == "unknown")
        df.loc[mask, feature_col] = type_name
    return df


def extract_materials_and_amenities(df: pd.DataFrame, description_col: str = "description") -> pd.DataFrame:
    """معادل سلول‌های 27، 29 و 31 نوت‌بوک."""
    if description_col in df.columns:
        text = df[description_col].astype(str).fillna("")
    else:
        text = pd.Series([""] * len(df), index=df.index)
    df["description_processed"] = text

    # متریال‌ها
    df["cabinet_type"] = "unknown"
    df["floor_type"] = "unknown"
    df["wall_type"] = "unknown"
    df = _extract_type_from_text(df, "cabinet_type", "description_processed", _CABINET_KEYWORDS)
    df = _extract_type_from_text(df, "floor_type", "description_processed", _FLOOR_KEYWORDS)
    df = _extract_type_from_text(df, "wall_type", "description_processed", _WALL_KEYWORDS)

    # امکانات باینری
    df["cooler_type"] = "unknown"
    df["has_yard"] = -1
    df["has_balcony"] = -1
    df["has_cctv"] = -1
    df["has_kanaf_ceiling"] = -1
    df["has_double_glazed_windows"] = -1
    df["near_mosque_school"] = -1

    df = _extract_type_from_text(df, "cooler_type", "description_processed", _COOLER_KEYWORDS)

    t = df["description_processed"]
    df.loc[t.str.contains("حیاط|حیات", case=False, na=False), "has_yard"] = 1
    df.loc[t.str.contains("بدون حیاط", case=False, na=False), "has_yard"] = 0

    df.loc[t.str.contains("بالکن|تراس", case=False, na=False), "has_balcony"] = 1
    df.loc[t.str.contains("بدون بالکن", case=False, na=False), "has_balcony"] = 0

    df.loc[t.str.contains("دوربین", case=False, na=False), "has_cctv"] = 1
    df.loc[t.str.contains("بدون دوربین", case=False, na=False), "has_cctv"] = 0

    df.loc[t.str.contains("کناف|سقف کاذب", case=False, na=False), "has_kanaf_ceiling"] = 1
    df.loc[t.str.contains("بدون کناف|سقف ساده", case=False, na=False), "has_kanaf_ceiling"] = 0

    df.loc[t.str.contains("شیشه دوجداره|دوجداره", case=False, na=False), "has_double_glazed_windows"] = 1
    df.loc[t.str.contains("بدون شیشه دوجداره", case=False, na=False), "has_double_glazed_windows"] = 0

    df.loc[t.str.contains("مسجد|مدرسه", case=False, na=False), "near_mosque_school"] = 1
    df.loc[t.str.contains("دور از مسجد|دور از مدرسه", case=False, na=False), "near_mosque_school"] = 0

    df["property_age_status"] = -1
    df.loc[t.str.contains("نوساز|تازه ساخت", case=False, na=False), "property_age_status"] = 1
    df.loc[t.str.contains("قدیمی ساز|کلنگی", case=False, na=False), "property_age_status"] = 0

    binary_unknown_to_zero = [
        "has_yard", "has_balcony", "has_cctv", "has_kanaf_ceiling",
        "has_double_glazed_windows", "near_mosque_school", "parking", "elevator", "storage",
    ]
    for col in binary_unknown_to_zero:
        if col in df.columns:
            df[col] = df[col].replace(-1, 0)
    df["property_age_status"] = df["property_age_status"].replace(-1, 0)

    # is_raw (سلول 31)
    df["is_raw"] = 0
    raw_pattern = "|".join(["خام", "خشک کار"])
    df.loc[t.str.contains(raw_pattern, case=False, na=False), "is_raw"] = 1

    return df


# ---------------------------------------------------------------------------
# 4) موقعیت مکانی: استخراج شهر/محله از متن آدرس (فقط برای training - داده خام Divar)
# ---------------------------------------------------------------------------
def extract_location_levels(location_text):
    if pd.isna(location_text) or not isinstance(location_text, str):
        return None, None
    match = re.search(r"در\s*(.*)", location_text)
    if not match:
        return None, None
    parts_str = match.group(1).strip()
    parts = [p.strip() for p in re.split(r"[،,]", parts_str) if p.strip()]
    level1 = parts[0] if len(parts) > 0 else None
    level2 = parts[1] if len(parts) > 1 else None
    return level1, level2


def extract_location(df: pd.DataFrame, location_col: str = "location") -> pd.DataFrame:
    df[["city", "district"]] = df[location_col].apply(lambda x: pd.Series(extract_location_levels(x)))
    return df


# ---------------------------------------------------------------------------
# 5) سن بنا بر مبنای سال شمسی
# ---------------------------------------------------------------------------
def calculate_age(builded_year_shamsi, current_shamsi_year: int):
    """می‌تواند روی یک Series یا یک عدد تکی اجرا شود."""
    return current_shamsi_year - builded_year_shamsi


# ---------------------------------------------------------------------------
# 6) هسته‌ی مشترک: ساخت ماتریس نهایی فیچرها (train و inference هر دو از این استفاده می‌کنند)
# ---------------------------------------------------------------------------
def make_city_district(df: pd.DataFrame) -> pd.DataFrame:
    df["city_district"] = df["city"].astype(str) + "_" + df["district"].astype(str)
    return df


def apply_rare_location_mapping(city_district_series: pd.Series, top_city_districts: set) -> pd.Series:
    return city_district_series.apply(lambda x: x if x in top_city_districts else "rare_location")


def build_raw_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    از یک دیتافریم که ستون‌های city_district, cabinet_type, floor_type, wall_type,
    cooler_type, property_age_status, rooms, floor, area, age و ستون‌های باینری را
    دارد، دیتافریم X (بدون تی‌اف‌آی‌دی‌اف و بدون target-encoding) می‌سازد.
    """
    cols = RAW_FEATURE_ORDER
    return df[cols].copy()


def add_tfidf_features(X: pd.DataFrame, df_source: pd.DataFrame, tfidf_description) -> pd.DataFrame:
    desc_text = df_source["description_processed"].fillna("") if "description_processed" in df_source.columns else pd.Series([""] * len(df_source), index=df_source.index)

    desc_matrix = tfidf_description.transform(desc_text)
    desc_df = pd.DataFrame(
        desc_matrix.toarray(),
        columns=[f"tfidf_description_{c}" for c in tfidf_description.get_feature_names_out()],
        index=df_source.index,
    )

    return pd.concat([X, desc_df], axis=1)


def apply_categorical_dtypes(X: pd.DataFrame, categories_map: dict) -> pd.DataFrame:
    """هر ستون categorical را با همان دسته‌بندی‌های زمان train ست می‌کند."""
    for col, cats in categories_map.items():
        if col in X.columns:
            X[col] = pd.Categorical(X[col], categories=cats)
    return X


def build_feature_row(raw_row: dict, artifacts: dict) -> pd.DataFrame:
    """
    نقطه‌ی ورود inference: یک دیکشنری از ورودی خام کاربر می‌گیرد و دقیقاً همان
    ترتیب/دیتاتایپ ستون‌هایی که مدل با آن train شده را برمی‌گرداند.

    raw_row باید شامل این کلیدها باشد:
        title, features_text, description, rooms, floor, area,
        built_year_shamsi, current_shamsi_year, city, district
    """
    df = pd.DataFrame([raw_row])

    # نگاشت اسم فیلدهای ورودی کاربر به اسم ستون‌های داخلی pipeline
    df = df.rename(columns={"features_text": "features"})

    df = extract_parking_elevator_storage(df, features_col="features")
    df = extract_materials_and_amenities(df, description_col="description")

    df["rooms"] = df["rooms"].fillna(-1)
    df["floor"] = df["floor"].fillna(-1)

    df["age"] = calculate_age(df["built_year_shamsi"], raw_row["current_shamsi_year"])

    df["city_district"] = df["city"].astype(str) + "_" + df["district"].astype(str)
    df["city_district"] = apply_rare_location_mapping(df["city_district"], set(artifacts["top_city_districts"]))

    X = build_raw_feature_frame(df)

    # جایگزینی -1 با NaN برای فیچرهای عددی و باینری (دقیقاً مثل train)
    for col in NUMERIC_AND_BINARY:
        X[col] = X[col].replace(-1, np.nan)

    # دیتاتایپ categorical دقیقا مطابق train (city_district جدا و به صورت رشته target-encode می‌شود)
    X = apply_categorical_dtypes(X, artifacts["categorical_categories"])

    X["city_district"] = X["city_district"].astype(str)
    X["city_district"] = artifacts["target_encoder"].transform(X[["city_district"]])["city_district"]

    X = add_tfidf_features(X, df, artifacts["tfidf_description"])

    # imputation نهایی با میانه‌ی train
    X = X.fillna(artifacts["train_medians"])

    # ترتیب دقیق ستون‌ها مطابق زمان train
    X = X.reindex(columns=artifacts["feature_columns"])

    return X
