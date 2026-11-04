@echo off
REM Run the full CFIC test suite across api-gateway and cloud-inference
cd %~dp0\..
echo [1/2] Running API Gateway test suite...
cd api-gateway
..\.venv\Scripts\python.exe -m pytest -v
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

echo.
echo [2/2] Running Cloud Inference Orchestrator test suite...
cd ..\cloud-inference\orchestrator
..\..\.venv\Scripts\python.exe -m pytest -v
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

echo.
echo All test suites passed successfully!
