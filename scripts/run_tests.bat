@echo off
REM Run the full CFIC test suite across api-gateway, cloud-inference orchestrator, and model microservices
set ROOT_DIR=%~dp0..
set PY_EXE=%ROOT_DIR%\.venv\Scripts\python.exe

echo [1/4] Running API Gateway test suite...
cd %ROOT_DIR%\api-gateway
%PY_EXE% -m pytest -v
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

echo.
echo [2/4] Running Cloud Inference Orchestrator test suite...
cd %ROOT_DIR%\cloud-inference\orchestrator
%PY_EXE% -m pytest -v
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

echo.
echo [3/4] Running Open-Vakgyata LID microservice tests...
cd %ROOT_DIR%\cloud-inference\models\lid
%PY_EXE% -m pytest -v tests
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

echo.
echo [4/4] Running English Whisper ASR microservice tests...
cd %ROOT_DIR%\cloud-inference\models\asr\indicconformer\english
%PY_EXE% -m pytest -v tests
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

echo.
echo All test suites passed successfully!
