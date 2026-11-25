<#
.SYNOPSIS
  Run the Smart Conveyor cloud server on THIS PC. The PC plays the "cloud" for the Raspberry Pi,
  which must be on the same Wi-Fi network. (For the real AWS deployment use deploy_aws.ps1.)

.EXAMPLE
  .\cloud\start_local_server.ps1             # port 8000, keeps the same API key every time
  .\cloud\start_local_server.ps1 -NewKey     # make a new API key (give the new one to the Pi)

.NOTES
  - Leave this window open: closing it (or the PC going to sleep) stops the server.
  - The API key is stored in cloud\.secrets\api_key_local.txt (git-ignored). Every request from the Pi
    and from the dashboard must carry it.
  - Box logs and Q-table backups are stored in cloud_data\ inside the project.
#>
param(
    [int]$Port = 8000,
    [switch]$NewKey
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Secrets = Join-Path $PSScriptRoot ".secrets"
$keyFile = Join-Path $Secrets "api_key_local.txt"
$py = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    throw "Python environment not found at $py. In the project folder run:  py -m venv .venv ; .venv\Scripts\pip install -r requirements-laptop.txt"
}
New-Item -ItemType Directory -Force $Secrets | Out-Null
if ($NewKey -or -not (Test-Path $keyFile)) {
    $bytes = New-Object byte[] 16                                      # 128-bit random key, 32 hex characters
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $k = ([System.BitConverter]::ToString($bytes) -replace "-", "").ToLower()
    Set-Content -Path $keyFile -Value $k -Encoding ascii -NoNewline
    Write-Host "New API key created in $keyFile"
}
$key = (Get-Content $keyFile -Raw).Trim()

$ips = @(Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
         Where-Object { $_.InterfaceAlias -match '^(Wi-Fi|Ethernet)' -and $_.IPAddress -notlike '169.254.*' } |
         Select-Object -ExpandProperty IPAddress)

Write-Host ""
Write-Host "=================== Smart Conveyor - cloud server on this PC ==================="
foreach ($ip in $ips) { Write-Host "  Server address for the Pi : http://${ip}:$Port" }
Write-Host "  Dashboard on this PC      : http://localhost:$Port   (press 'Key' and paste the key)"
Write-Host "  API key                   : $key"
Write-Host "  Pi setup (once, on the Pi): sudo bash deploy/pi/install_pi.sh http://<address above>:$Port <API key>"
Write-Host "  Test from a phone on the same Wi-Fi: open  http://<address above>:$Port/health"
Write-Host "  Stop the server           : Ctrl+C in this window"
Write-Host "================================================================================="
Write-Host ""

$env:CONVEYOR_API_KEY = $key
$env:CONVEYOR_WHERE = "this PC"
$env:CONVEYOR_PORT = "$Port"
Set-Location $Root
& $py -m server.app --cpu-only
