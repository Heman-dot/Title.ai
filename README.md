# Title.ai

Title.ai is a small full-stack web project that helps verify and suggest titles using a machine learning model.

The repository contains three main parts:

- Frontend (React) — a Create React App-based UI in `src/` with pages for Home, Features, About, Login/Register and Profile.
- Backend (Node/Express) — an API and auth in `title-ai-backend/` that handles user registration/login and serves as the bridge to the ML service.
- ML service (Python) — `title-ai-ml/` contains the model code, preprocessing and a small Flask app to serve the model for title verification.

This README explains how the pieces fit together and how to run the system locally.

## Features

- Title verification model (trained weights present locally under `title-ai-ml/model/` — note these are intentionally excluded from git)
- Web UI to enter and verify titles
- Authentication (register/login) via the backend
- Simple Flask endpoint to expose the ML model for inference

## Project layout

```
title-ai/
├─ src/                # React frontend
├─ title-ai-backend/   # Node/Express backend (auth & API)
└─ title-ai-ml/        # ML code, model files, scripts
```



## Quick start (development)

Prerequisites:

- Node.js (16+ recommended)
- npm or yarn
- Python 3.8+ and pip for the ML service

1) Install frontend dependencies

```bash
cd /Users/heman/Downloads/title-ai
npm install
npm start
```

2) Start backend (title-ai-backend)

```bash
cd title-ai-backend
npm install
node server.js
# or use nodemon if available
```

3) Run the ML service (title-ai-ml)

```bash
cd title-ai-ml/app
# it is recommended to create a virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

The ML app exposes endpoints defined in `title-ai-ml/app/app.py` (for example, a prediction route) that the backend can call for verification.

## Data & model files

- Trained model files (e.g. `*.h5`, `*.pt`) and raw datasets are intentionally excluded from git via `.gitignore` to prevent committing large binaries. Local copies may exist under `title-ai-ml/model/` or `title-ai-ml/data/`.
- If you want to include models in the repository, use Git LFS — otherwise keep them in a cloud artifact store and download during deployment.



---

