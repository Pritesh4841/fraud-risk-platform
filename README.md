
Task is Completed!!


## 1. Virtual environment

Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Mac / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 2. Dependencies install

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Tests

```bash
pytest -v
```

## 4. Retrain (optional, models pehle se included hain)

```bash
python -m src.generate_data
python -m src.train        # it takes 2-5 min
python -m src.evaluate
```

Oracle comparison:
```bash
python -m src.oracle_check
```

## 5. API start

```bash
uvicorn app:app --reload
```
Swagger docs: http://127.0.0.1:8000/docs

## 6. Quick API test

Health check:
```bash
curl http://127.0.0.1:8000/health
```

Predict:
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d 
"{\"amount\": 249.5, \"merchant_category\": \"online\", \"transaction_hour\": 2, \"previous_declines_24h\": 3}"
```