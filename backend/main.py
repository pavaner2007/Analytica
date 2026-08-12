from __future__ import annotations

import io
import json
import math
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
AUDIT_LOG = DATA_DIR / "audit_log.jsonl"

app = FastAPI(
    title="Intelligent Data Suite API",
    description="React-ready API for dataset upload, profiling, cleaning, visualization, and ML prediction.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory session store. For a student/demo project this is easy to run locally.
# A production version can swap this for Redis/PostgreSQL.
SESSIONS: Dict[str, pd.DataFrame] = {}
ORIGINAL_FILES: Dict[str, str] = {}


class CleanRequest(BaseModel):
    action: str = Field(..., description="fill_missing, drop_missing, drop_duplicates, remove_outliers")
    method: Optional[str] = Field(default="mean", description="mean, median, mode, constant, iqr, zscore")
    columns: Optional[List[str]] = None
    fill_value: Optional[str] = None


class TransformRequest(BaseModel):
    columns: List[str]
    transform_type: str = Field(..., description="log, sqrt, square, standardize, normalize")


class FilterRequest(BaseModel):
    column: str
    operator: str = Field(..., description="contains, equals, not_equals, greater_than, less_than, between")
    value: Optional[str] = None
    value2: Optional[str] = None


class TrainRequest(BaseModel):
    target: str
    features: Optional[List[str]] = None
    model_type: str = Field(default="auto", description="auto, linear_regression, logistic_regression, random_forest")
    test_size: float = 0.2


def _sanitize(value: Any) -> Any:
    """Convert NumPy/Pandas values into JSON-safe values."""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _records(df: pd.DataFrame, limit: int = 50) -> List[Dict[str, Any]]:
    preview = df.head(limit).copy()
    for col in preview.columns:
        if pd.api.types.is_datetime64_any_dtype(preview[col]):
            preview[col] = preview[col].dt.strftime("%Y-%m-%d %H:%M:%S")
    preview = preview.astype(object).where(pd.notnull(preview), None)
    return [{str(k): _sanitize(v) for k, v in row.items()} for row in preview.to_dict(orient="records")]


def log_action(action: str, details: str = "", session_id: Optional[str] = None) -> None:
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "session_id": session_id,
        "action": action,
        "details": details,
    }
    with AUDIT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def get_df(session_id: str) -> pd.DataFrame:
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Dataset session not found. Upload or load a sample dataset first.")
    return SESSIONS[session_id]


def calculate_quality(df: pd.DataFrame) -> Dict[str, float]:
    rows, cols = df.shape
    if rows == 0 or cols == 0:
        return {"score": 0.0, "missing_pct": 0.0, "duplicate_pct": 0.0, "outlier_pct": 0.0}

    missing_pct = float(df.isna().sum().sum() / (rows * cols) * 100)
    duplicate_pct = float(df.duplicated().sum() / rows * 100)
    numeric = df.select_dtypes(include=np.number)
    outlier_pct = 0.0
    if not numeric.empty:
        std = numeric.std(ddof=0).replace(0, np.nan)
        z = ((numeric - numeric.mean()) / std).abs()
        outlier_pct = float((z > 3).sum().sum() / max(1, numeric.size) * 100)
    score = max(0.0, 100 - (missing_pct * 0.5 + duplicate_pct * 0.2 + outlier_pct * 0.3))
    return {
        "score": round(score, 2),
        "missing_pct": round(missing_pct, 2),
        "duplicate_pct": round(duplicate_pct, 2),
        "outlier_pct": round(outlier_pct, 2),
    }


def generate_insights(df: pd.DataFrame) -> List[str]:
    insights = [f"Dataset contains {df.shape[0]:,} rows and {df.shape[1]} columns."]
    if df.empty:
        insights.append("The dataset is empty after the latest operation.")
        return insights

    missing_pct = df.isna().sum().sum() / max(1, df.size) * 100
    if missing_pct > 10:
        insights.append(f"High missing data detected: {missing_pct:.1f}% of all cells are empty.")
    elif missing_pct > 0:
        insights.append(f"Small amount of missing data detected: {missing_pct:.1f}% of all cells are empty.")
    else:
        insights.append("No missing values were found in the current dataset.")

    dup_count = int(df.duplicated().sum())
    if dup_count:
        insights.append(f"{dup_count:,} duplicate rows detected and can be removed from Data Cleaning.")

    numeric_cols = list(df.select_dtypes(include=np.number).columns)
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True).abs()
        pairs = []
        for i, left in enumerate(corr.columns):
            for right in corr.columns[i + 1 :]:
                val = corr.loc[left, right]
                if pd.notna(val) and val > 0.8:
                    pairs.append((left, right, val))
        if pairs:
            top = sorted(pairs, key=lambda x: x[2], reverse=True)[0]
            insights.append(f"Strong relationship found between '{top[0]}' and '{top[1]}' with correlation {top[2]:.2f}.")

    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
    high_card = [col for col in categorical_cols if df[col].nunique(dropna=True) > 50]
    if high_card:
        insights.append(f"{len(high_card)} categorical column(s) have high cardinality and may need encoding/grouping for ML.")

    return insights


def build_profile(df: pd.DataFrame, session_id: str) -> Dict[str, Any]:
    numeric_cols = list(df.select_dtypes(include=np.number).columns)
    categorical_cols = list(df.select_dtypes(include=["object", "category", "bool"]).columns)
    datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

    columns = []
    for col in df.columns:
        series = df[col]
        sample_values = [_sanitize(v) for v in series.dropna().head(4).tolist()]
        columns.append(
            {
                "name": str(col),
                "dtype": str(series.dtype),
                "missing": int(series.isna().sum()),
                "missing_pct": round(float(series.isna().mean() * 100), 2) if len(series) else 0,
                "unique": int(series.nunique(dropna=True)),
                "sample_values": sample_values,
            }
        )

    stats: Dict[str, Dict[str, Any]] = {}
    if numeric_cols:
        desc = df[numeric_cols].describe().replace({np.nan: None}).to_dict()
        for col, values in desc.items():
            stats[col] = {str(k): _sanitize(v) for k, v in values.items()}

    corr = []
    if len(numeric_cols) >= 2:
        matrix = df[numeric_cols].corr(numeric_only=True).replace({np.nan: None})
        corr = [
            {"x": str(i), "y": str(j), "value": _sanitize(matrix.loc[i, j])}
            for i in matrix.index
            for j in matrix.columns
        ]

    return {
        "session_id": session_id,
        "file_name": ORIGINAL_FILES.get(session_id, "sample_data.csv"),
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "quality": calculate_quality(df),
        "columns": columns,
        "column_names": [str(c) for c in df.columns],
        "numeric_columns": [str(c) for c in numeric_cols],
        "categorical_columns": [str(c) for c in categorical_cols],
        "datetime_columns": [str(c) for c in datetime_cols],
        "preview": _records(df, 50),
        "stats": stats,
        "correlation": corr,
        "insights": generate_insights(df),
    }


def parse_uploaded_file(file: UploadFile, payload: bytes) -> pd.DataFrame:
    file_name = file.filename or "dataset"
    suffix = Path(file_name).suffix.lower()
    try:
        if suffix == ".csv":
            try:
                return pd.read_csv(io.BytesIO(payload), encoding="utf-8")
            except (UnicodeDecodeError, ValueError):
                try:
                    return pd.read_csv(io.BytesIO(payload), encoding="utf-8-sig")
                except (UnicodeDecodeError, ValueError):
                    return pd.read_csv(io.BytesIO(payload), encoding="latin-1")
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(io.BytesIO(payload))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read dataset: {exc}") from exc
    raise HTTPException(status_code=400, detail="Unsupported file type. Please upload CSV, XLSX, or XLS files.")


def load_sample_dataframe() -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range(start="2025-01-01", periods=140, freq="D")
    df = pd.DataFrame(
        {
            "Date": dates,
            "Region": np.random.choice(["North", "South", "East", "West"], len(dates)),
            "Category": np.random.choice(["Electronics", "Fashion", "Grocery", "Books"], len(dates)),
            "Sales": np.random.randint(1200, 9500, len(dates)),
            "Profit": np.round(np.random.uniform(150, 2500, len(dates)), 2),
            "Customer_Rating": np.round(np.random.uniform(2.8, 5.0, len(dates)), 1),
            "Returned": np.random.choice(["Yes", "No"], len(dates), p=[0.12, 0.88]),
        }
    )
    # Add a few realistic data quality issues for demonstration.
    df.loc[[4, 17, 32], "Profit"] = np.nan
    df.loc[[8, 29], "Sales"] = np.nan
    df = pd.concat([df, df.iloc[[3, 12]]], ignore_index=True)
    return df


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Intelligent Data Suite API is running", "docs": "/docs"}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/load-sample")
def load_sample() -> Dict[str, Any]:
    session_id = str(uuid.uuid4())
    df = load_sample_dataframe()
    SESSIONS[session_id] = df
    ORIGINAL_FILES[session_id] = "sample_data.csv"
    log_action("Load Sample Dataset", f"Loaded {df.shape[0]} rows and {df.shape[1]} columns", session_id)
    return build_profile(df, session_id)


@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)) -> Dict[str, Any]:
    payload = await file.read()
    df = parse_uploaded_file(file, payload)
    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded dataset is empty.")
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = df
    ORIGINAL_FILES[session_id] = file.filename or "dataset"
    log_action("Upload Dataset", f"Uploaded {ORIGINAL_FILES[session_id]} with {df.shape[0]} rows and {df.shape[1]} columns", session_id)
    return build_profile(df, session_id)


@app.get("/api/profile/{session_id}")
def profile(session_id: str) -> Dict[str, Any]:
    return build_profile(get_df(session_id), session_id)


@app.post("/api/clean/{session_id}")
def clean_dataset(session_id: str, request: CleanRequest) -> Dict[str, Any]:
    df = get_df(session_id).copy()
    columns = request.columns or list(df.columns)
    valid_columns = [c for c in columns if c in df.columns]
    if not valid_columns and request.action not in {"drop_duplicates"}:
        raise HTTPException(status_code=400, detail="No valid columns selected.")

    before_shape = df.shape
    action = request.action.lower()
    method = (request.method or "mean").lower()

    if action == "fill_missing":
        for col in valid_columns:
            if not df[col].isna().any():
                continue
            if pd.api.types.is_numeric_dtype(df[col]) and method in {"mean", "median"}:
                value = df[col].mean() if method == "mean" else df[col].median()
            elif method == "constant":
                value = request.fill_value if request.fill_value is not None else "Unknown"
            else:
                mode = df[col].mode(dropna=True)
                value = mode.iloc[0] if not mode.empty else "Unknown"
            df[col] = df[col].fillna(value)
    elif action == "drop_missing":
        df = df.dropna(subset=valid_columns)
    elif action == "drop_duplicates":
        df = df.drop_duplicates()
    elif action == "remove_outliers":
        numeric_columns = [c for c in valid_columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_columns:
            raise HTTPException(status_code=400, detail="Outlier removal requires numeric columns.")
        mask = pd.Series(True, index=df.index)
        for col in numeric_columns:
            series = df[col]
            if method == "zscore":
                std = series.std(ddof=0)
                if std and not np.isnan(std):
                    z = ((series - series.mean()) / std).abs()
                    mask &= z.fillna(0) <= 3
            else:
                q1, q3 = series.quantile(0.25), series.quantile(0.75)
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                mask &= series.between(lower, upper) | series.isna()
        df = df.loc[mask]
    else:
        raise HTTPException(status_code=400, detail="Invalid clean action.")

    SESSIONS[session_id] = df.reset_index(drop=True)
    log_action("Clean Dataset", f"{request.action} using {method}. Shape {before_shape} -> {df.shape}", session_id)
    return build_profile(SESSIONS[session_id], session_id)


@app.post("/api/transform/{session_id}")
def transform_dataset(session_id: str, request: TransformRequest) -> Dict[str, Any]:
    df = get_df(session_id).copy()
    transform_type = request.transform_type.lower()
    created = []

    for col in request.columns:
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if transform_type == "log":
            new_col = f"{col}_log"
            min_value = df[col].min(skipna=True)
            if pd.notna(min_value) and min_value <= -1:
                df[new_col] = np.log1p(df[col] - min_value)
            else:
                df[new_col] = np.log1p(df[col])
        elif transform_type == "sqrt":
            new_col = f"{col}_sqrt"
            df[new_col] = np.sqrt(df[col].abs())
        elif transform_type == "square":
            new_col = f"{col}_squared"
            df[new_col] = df[col] ** 2
        elif transform_type == "standardize":
            new_col = f"{col}_standardized"
            if df[col].isna().all():
                raise HTTPException(status_code=400, detail=f"Column '{col}' contains only missing values and cannot be standardized.")
            scaler = StandardScaler()
            df[new_col] = scaler.fit_transform(df[[col]].fillna(df[col].mean())).ravel()
        elif transform_type == "normalize":
            new_col = f"{col}_normalized"
            if df[col].isna().all():
                raise HTTPException(status_code=400, detail=f"Column '{col}' contains only missing values and cannot be normalized.")
            scaler = MinMaxScaler()
            df[new_col] = scaler.fit_transform(df[[col]].fillna(df[col].mean())).ravel()
        else:
            raise HTTPException(status_code=400, detail="Invalid transformation type.")
        created.append(new_col)

    if not created:
        raise HTTPException(status_code=400, detail="No numeric columns were transformed.")
    SESSIONS[session_id] = df
    log_action("Transform Dataset", f"Created columns: {', '.join(created)}", session_id)
    return build_profile(df, session_id)


@app.post("/api/filter/{session_id}")
def filter_dataset(session_id: str, request: FilterRequest) -> Dict[str, Any]:
    df = get_df(session_id).copy()
    if request.column not in df.columns:
        raise HTTPException(status_code=400, detail="Column not found.")

    series = df[request.column]
    operator = request.operator.lower()
    value = request.value
    before_shape = df.shape

    if operator == "contains":
        mask = series.astype(str).str.contains(str(value or ""), case=False, na=False)
    elif operator == "equals":
        mask = series.astype(str) == str(value)
    elif operator == "not_equals":
        mask = series.astype(str) != str(value)
    elif operator in {"greater_than", "less_than", "between"}:
        numeric_series = pd.to_numeric(series, errors="coerce")
        try:
            number = float(value) if value is not None else None
            number2 = float(request.value2) if request.value2 is not None else None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Numeric filter value is invalid.") from exc
        if operator == "greater_than":
            mask = numeric_series > number
        elif operator == "less_than":
            mask = numeric_series < number
        else:
            if number is None or number2 is None:
                raise HTTPException(status_code=400, detail="Between filter requires two values.")
            low, high = sorted([number, number2])
            mask = numeric_series.between(low, high)
    else:
        raise HTTPException(status_code=400, detail="Invalid filter operator.")

    filtered = df.loc[mask].reset_index(drop=True)
    SESSIONS[session_id] = filtered
    log_action("Filter Dataset", f"{request.column} {operator} {value}. Shape {before_shape} -> {filtered.shape}", session_id)
    return build_profile(filtered, session_id)


@app.get("/api/chart/{session_id}")
def chart(session_id: str, chart_type: str = "histogram", x: Optional[str] = None, y: Optional[str] = None, color: Optional[str] = None) -> Dict[str, Any]:
    df = get_df(session_id)
    if df.empty:
        raise HTTPException(status_code=400, detail="Dataset is empty.")
    try:
        if chart_type == "histogram":
            if not x:
                x = next(iter(df.select_dtypes(include=np.number).columns), df.columns[0])
            fig = px.histogram(df, x=x, color=color if color in df.columns else None, template="plotly_dark")
        elif chart_type == "scatter":
            numeric = list(df.select_dtypes(include=np.number).columns)
            x = x or (numeric[0] if numeric else df.columns[0])
            y = y or (numeric[1] if len(numeric) > 1 else x)
            fig = px.scatter(df, x=x, y=y, color=color if color in df.columns else None, template="plotly_dark")
        elif chart_type == "bar":
            x = x or df.columns[0]
            if y and y in df.columns and pd.api.types.is_numeric_dtype(df[y]):
                data = df.groupby(x, dropna=False)[y].mean().reset_index().head(30)
                fig = px.bar(data, x=x, y=y, template="plotly_dark")
            else:
                data = df[x].astype(str).value_counts().head(30).reset_index()
                data.columns = [x, "count"]
                fig = px.bar(data, x=x, y="count", template="plotly_dark")
        elif chart_type == "box":
            numeric = list(df.select_dtypes(include=np.number).columns)
            y = y or x or (numeric[0] if numeric else None)
            if not y:
                raise HTTPException(status_code=400, detail="Box plot requires a numeric column.")
            fig = px.box(df, y=y, x=x if x and x != y else None, color=color if color in df.columns else None, template="plotly_dark")
        else:
            raise HTTPException(status_code=400, detail="Invalid chart type.")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e5e7eb")
        return json.loads(fig.to_json())
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not generate chart: {exc}") from exc


@app.post("/api/train/{session_id}")
def train_model(session_id: str, request: TrainRequest) -> Dict[str, Any]:
    df = get_df(session_id).copy()
    if request.target not in df.columns:
        raise HTTPException(status_code=400, detail="Target column not found.")

    features = request.features or [c for c in df.columns if c != request.target]
    features = [c for c in features if c in df.columns and c != request.target]
    if not features:
        raise HTTPException(status_code=400, detail="Please select at least one feature column.")

    model_df = df[features + [request.target]].dropna(subset=[request.target])
    if model_df.shape[0] < 10:
        raise HTTPException(status_code=400, detail="Need at least 10 valid rows to train a model.")

    # Filter out columns that have all missing values
    valid_features = [col for col in features if not model_df[col].isna().all()]
    if not valid_features:
        raise HTTPException(status_code=400, detail="All selected feature columns have only missing values. Please select features with valid data.")
    features = valid_features

    X = model_df[features]
    y = model_df[request.target]
    numeric_features = list(X.select_dtypes(include=np.number).columns)
    categorical_features = [c for c in X.columns if c not in numeric_features]

    target_is_numeric = pd.api.types.is_numeric_dtype(y)
    unique_target_values = y.nunique(dropna=True)
    is_classification = (not target_is_numeric) or unique_target_values <= min(20, max(2, len(y) // 10))

    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]
    )
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_pipeline, numeric_features), ("cat", categorical_pipeline, categorical_features)],
        remainder="drop",
    )

    model_type = request.model_type.lower()
    # Explicit user choices override the auto-detected classification flag
    if model_type == "logistic_regression":
        is_classification = True
    elif model_type == "linear_regression":
        is_classification = False

    if is_classification:
        if pd.api.types.is_numeric_dtype(y):
            y = y.astype(str)
        if model_type == "logistic_regression":
            estimator = LogisticRegression(max_iter=1000)
            selected_model = "Logistic Regression"
        else:
            estimator = RandomForestClassifier(n_estimators=120, random_state=42)
            selected_model = "Random Forest Classifier"
    else:
        if model_type == "linear_regression":
            estimator = LinearRegression()
            selected_model = "Linear Regression"
        else:
            estimator = RandomForestRegressor(n_estimators=120, random_state=42)
            selected_model = "Random Forest Regressor"

    test_size = min(max(request.test_size, 0.1), 0.4)
    stratify = y if is_classification and y.nunique() > 1 and y.value_counts().min() >= 2 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify)
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    if is_classification:
        accuracy = accuracy_score(y_test, predictions)
        metrics = {"accuracy": round(float(accuracy), 4), "task": "classification"}
    else:
        r2 = r2_score(y_test, predictions)
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        metrics = {"r2_score": round(float(r2), 4), "rmse": round(float(rmse), 4), "task": "regression"}

    sample_predictions = []
    for actual, pred in zip(list(y_test.head(8)), list(predictions[:8])):
        sample_predictions.append({"actual": _sanitize(actual), "predicted": _sanitize(pred)})

    result = {
        "model": selected_model,
        "target": request.target,
        "features": features,
        "rows_used": int(model_df.shape[0]),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "metrics": metrics,
        "sample_predictions": sample_predictions,
    }
    log_action("Train ML Model", f"{selected_model} target={request.target} rows={model_df.shape[0]}", session_id)
    return result


@app.get("/api/export/{session_id}")
def export_csv(session_id: str) -> StreamingResponse:
    df = get_df(session_id)
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    stream.seek(0)
    file_name = f"cleaned_dataset_{session_id[:8]}.csv"
    log_action("Export Dataset", file_name, session_id)
    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={file_name}"},
    )


@app.get("/api/audit")
def audit_log(limit: int = 100) -> Dict[str, Any]:
    if not AUDIT_LOG.exists():
        return {"logs": []}
    lines = AUDIT_LOG.read_text(encoding="utf-8").splitlines()[-limit:]
    logs = []
    for line in lines:
        try:
            logs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return {"logs": logs[::-1]}
