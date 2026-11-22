# PowerShell runner for Whisper benchmark
param (
    [string]$Model = "small",
    [string]$Manifest = "data/manifest_telugu.jsonl",
    [string]$Output = "outputs/results/whisper_telugu.csv",
    [string]$ChunkOutput = "outputs/results/chunk_results.csv",
    [int]$MaxSamples = 0
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RootDir = Split-Path -Parent $ScriptDir
Set-Location $RootDir

$VenvPython = Join-Path $RootDir ".venv-whisper\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Error "Whisper virtual environment not found at $VenvPython. Run scripts/setup.ps1 first."
    exit 1
}

$ArgsList = @(
    "src/run_whisper.py",
    "--manifest", $Manifest,
    "--model", $Model,
    "--output", $Output,
    "--chunk-output", $ChunkOutput
)

if ($MaxSamples -gt 0) {
    $ArgsList += @("--max-samples", $MaxSamples)
}

Write-Host "Executing Whisper ($Model) ASR Benchmark..." -ForegroundColor Cyan
& $VenvPython @ArgsList
