# Setup script for isolated Whisper and IndicConformer environments
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RootDir = Split-Path -Parent $ScriptDir
Set-Location $RootDir

Write-Host "Setting up isolated virtual environments for CFIC x VoiceShield ASR Benchmark..." -ForegroundColor Cyan

# 1. Base Python detection
$BasePython = "python"
if (Get-Command "py" -ErrorAction SilentlyContinue) {
    $BasePython = "py -3.12"
}

# 2. Whisper Environment
Write-Host "`n[1/2] Creating and configuring Whisper environment (.venv-whisper)..." -ForegroundColor Yellow
if (-not (Test-Path ".venv-whisper")) {
    Invoke-Expression "$BasePython -m venv .venv-whisper"
}
& ".venv-whisper\Scripts\pip.exe" install --upgrade pip
& ".venv-whisper\Scripts\pip.exe" install -r requirements-whisper.txt

# Extract and configure ffmpeg
& ".venv-whisper\Scripts\python.exe" -c "
import os, sys, imageio_ffmpeg, shutil
exe = imageio_ffmpeg.get_ffmpeg_exe()
target = os.path.join(os.path.dirname(sys.executable), 'ffmpeg.exe')
if not os.path.exists(target) and os.path.exists(exe):
    shutil.copyfile(exe, target)
    print('ffmpeg deployed to:', target)
"

# 3. IndicConformer Environment
Write-Host "`n[2/2] Creating and configuring IndicConformer environment (.venv-indic)..." -ForegroundColor Yellow
if (-not (Test-Path ".venv-indic")) {
    Invoke-Expression "$BasePython -m venv .venv-indic"
}
& ".venv-indic\Scripts\pip.exe" install --upgrade pip
& ".venv-indic\Scripts\pip.exe" install -r requirements-indicconformer.txt

# Copy ffmpeg to indic venv as well
if (Test-Path ".venv-whisper\Scripts\ffmpeg.exe") {
    Copy-Item ".venv-whisper\Scripts\ffmpeg.exe" ".venv-indic\Scripts\ffmpeg.exe" -Force
}

# Apply compatibility patches to local NeMo checkout
Write-Host "`nApplying compatibility patches to local NeMo..." -ForegroundColor Green
& ".venv-indic\Scripts\python.exe" "src/patch_nemo.py"

Write-Host "`nSetup complete! Environments are ready." -ForegroundColor Green
