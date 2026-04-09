# init.ps1 - project environment bootstrap
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "This project uses bash-based remote x86 bootstrap via init.sh and scripts/remote/*."
Write-Host "Use WSL or a POSIX shell and run: bash init.sh"
exit 1
