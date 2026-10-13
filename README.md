# PHYSIO-TWIN

PHYSIO-TWIN is a web application that models daily physiological state using routine, sleep, activity, and subjective observations. It estimates fatigue and recovery trends and provides forecasts.

This project is a research prototype, not a medical diagnostic system.

## Technology Stack

### Frontend
- React
- TypeScript
- Vite
- Plotly
- Axios

### Backend
- Python
- FastAPI
- Pydantic
- SQLAlchemy
- SQLite

### Modelling and Data Science
- NumPy
- Pandas
- SciPy
- Statsmodels
- Scikit-learn

## Repository Structure

- `backend/`: FastAPI application, routes, database access, and security
- `frontend/`: React web application
- `model/`: Modelling and forecasting components
- `tests/`: Automated tests
- `data/`: Research data and model outputs
- `database/`: SQLite database
- `docs/`: Project documentation

## Prerequisites

Install:
- Python 3.11
- Node.js and npm
- Git

## Run the Backend on Windows

Open PowerShell in the project root.

Create a virtual environment if one does not already exist:

```powershell
py -3.11 -m venv my_env
```

Activate it:

```powershell
.\my_env\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Start the API:

```powershell
python -m uvicorn backend.main:app --reload
```

Backend URL:

`http://127.0.0.1:8000`

Interactive API documentation:

`http://127.0.0.1:8000/docs`

## Run the Frontend

Open a second terminal in the project root.

Install frontend dependencies:

```powershell
cd frontend
npm ci
```

Start the development server:

```powershell
npm run dev
```

Open the URL displayed by Vite, normally:

`http://localhost:5173`

The frontend currently uses `http://127.0.0.1:8000` as its API URL. The frontend API configuration and backend CORS settings must be updated for deployment.

## Database

The backend currently uses SQLite at:

`database/physio_twin.db`

The database is initialized when the FastAPI application starts. Existing database contents may be retained when the application reuses this file.

Before deploying, verify the database contents and ensure that any required records are dummy or approved test data. Do not expose sensitive personal information.

A deployed SQLite database requires persistent writable storage. Do not assume that a hosting platform preserves local files across restarts or redeployments.

## Tests

From the project root, activate the Python virtual environment and run:

```powershell
python -m pytest -v
```

## Deployment Checklist

Before public deployment:

- Configure the frontend to use the deployed backend URL.
- Allow the deployed frontend origin in the backend CORS configuration.
- Configure persistent database storage.
- Use HTTPS.
- Review authentication, authorization, and input validation.
- Keep secrets out of Git.
- Test registration, login, observations, state estimates, and forecasts.
- Verify that a fresh installation works without relying on developer-specific files.

## Important Note


This is a research prototype. Forecasts are estimates based on available inputs and model assumptions; they are not medical diagnoses or a substitute for professional medical advice.

