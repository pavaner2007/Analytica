# Intelligent Data Suite - React + FastAPI Version

This is the corrected and upgraded version of the original Streamlit project. The interface has been changed to a modern React dashboard, while the data processing and machine learning logic runs through a FastAPI backend.

## What Was Changed

- Replaced the incomplete Streamlit interface with a React + Vite frontend.
- Added a FastAPI backend for upload, profiling, cleaning, filtering, visualization, ML training, export, and audit trail.
- Removed the bulky `venv` folder from the final ZIP.
- Fixed missing page implementation issues from the original app.
- Fixed dataset serialization issues for missing values, NumPy values, and dates.
- Fixed outlier removal logic.
- Fixed ML pipeline handling for numeric and categorical columns.
- Added a clean `requirements.txt` instead of shipping virtual environment files.
- Kept the original Streamlit file inside `legacy_streamlit/` for reference only.

## Project Structure

```text
DS_Project_react_fixed/
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   └── data/
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx
│       ├── api.js
│       └── styles.css
├── legacy_streamlit/
│   └── original_streamlit_app.py
├── run_backend.bat
├── run_frontend.bat
├── run_backend.sh
├── run_frontend.sh
└── README.md
```

## Features

- Upload CSV, XLSX, or XLS datasets.
- Load an inbuilt sample dataset.
- View dataset quality score, missing value percentage, duplicate percentage, and outlier percentage.
- Preview dataset rows and column health.
- Clean missing values using mean, median, mode, or constant values.
- Drop missing rows and duplicate rows.
- Remove outliers using IQR or Z-score logic.
- Apply transformations such as log, square root, square, standardization, and normalization.
- Filter records using contains, equals, greater than, less than, and between.
- Generate interactive Plotly charts.
- Train baseline ML models using Linear Regression, Logistic Regression, and Random Forest.
- Export the processed dataset as CSV.
- View audit trail of user actions.

## How to Run

### 1. Start the Backend

Windows:

```bash
run_backend.bat
```

macOS/Linux:

```bash
./run_backend.sh
```

Manual backend command:

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Backend API will run at:

```text
http://127.0.0.1:8000
```

API docs:

```text
http://127.0.0.1:8000/docs
```

### 2. Start the Frontend

Open a second terminal.

Windows:

```bash
run_frontend.bat
```

macOS/Linux:

```bash
./run_frontend.sh
```

Manual frontend command:

```bash
cd frontend
npm install
npm run dev
```

React app will run at:

```text
http://127.0.0.1:5173
```

## Notes

- Do not upload the `venv`, `.venv`, or `node_modules` folders to GitHub.
- The backend currently stores active datasets in memory. If you restart the backend, upload/load the dataset again.
- Audit logs are stored as JSONL in `backend/data/audit_log.jsonl`.
- For production, use a database or object storage for datasets instead of in-memory storage.
