param([string]$DataRoot = '', [int]$Port = 8000)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
if ($DataRoot) { $env:PATHSNAP_DATA_ROOT = $DataRoot }
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) { throw 'Run python -m venv .venv and install requirements.txt first.' }
New-Item -ItemType Directory -Force results | Out-Null
& '.venv\Scripts\python.exe' -m uvicorn api.app:app --host 0.0.0.0 --port $Port 2>&1 | Tee-Object -FilePath 'results\api_server.log'
