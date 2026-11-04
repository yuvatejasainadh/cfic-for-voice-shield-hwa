@echo off
REM Start CFIC API Gateway locally
cd %~dp0\..\api-gateway
if not exist .env (
    echo Copying .env.example to .env...
    copy .env.example .env
)
echo Starting CFIC API Gateway on http://localhost:8000...
..\.venv\Scripts\uvicorn.exe app.main:app --reload --host 0.0.0.0 --port 8000
