param([int]$Port = 8501)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
python -m http.server $Port --bind 0.0.0.0 --directory ui
