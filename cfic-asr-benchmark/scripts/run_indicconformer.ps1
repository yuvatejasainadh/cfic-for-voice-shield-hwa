# PowerShell runner for AI4Bharat IndicConformer benchmark
param (
    [string]$ModelPath = "models/indicconformer_stt_te_hybrid_rnnt_large.nemo",
    [string]$ModelName = "indicconformer_te_large",
    [string]$Decoder = "ctc",
    [string]$Manifest = "data/manifest_telugu.jsonl",
    [string]$Output = "outputs/results/indicconformer_telugu.csv",
    [string]$ChunkOutput = "outputs/results/chunk_results.csv",
    [int]$MaxSamples = 0
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RootDir = Split-Path -Parent $ScriptDir
Set-Location $RootDir

$VenvPython = Join-Path $RootDir ".venv-indic\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Error "IndicConformer virtual environment not found at $VenvPython. Run scripts/setup.ps1 first."
    exit 1
}

$ArgsList = @(
    "src/run_indicconformer.py",
    "--manifest", $Manifest,
    "--nemo-model", $ModelPath,
    "--model-name", $ModelName,
    "--decoder", $Decoder,
    "--output", $Output,
    "--chunk-output", $ChunkOutput
)

if ($MaxSamples -gt 0) {
    $ArgsList += @("--max-samples", $MaxSamples)
}

Write-Host "Executing IndicConformer ($ModelName, Decoder: $Decoder) ASR Benchmark..." -ForegroundColor Cyan
& $VenvPython @ArgsList
