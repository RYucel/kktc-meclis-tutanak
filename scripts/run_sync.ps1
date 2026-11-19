# Silent wrapper for Windows Task Scheduler -> incremental KKTC Meclis sync to Hugging Face
$ErrorActionPreference = "Continue"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$LogDir = Join-Path $RepoRoot "data"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }
$LogFile = Join-Path $LogDir "sync_history.log"

# Keep log under ~2 MB
if ((Test-Path $LogFile) -and ((Get-Item $LogFile).Length -gt 2MB)) {
    Move-Item -Force $LogFile "$LogFile.old"
}

$ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Path $LogFile -Value "`n==================== $ts ===================="

# Locate Python cleanly
$PythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $PythonExe) {
    $PythonExe = "C:\Users\SuperComputers\AppData\Local\Programs\Python\Python312\python.exe"
}

try {
    & $PythonExe -u (Join-Path $RepoRoot "src\sync_incremental.py") *>> $LogFile
    $code = $LASTEXITCODE
    $endTs = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $LogFile -Value "[$endTs] ExitCode=$code"
    exit $code
} catch {
    Add-Content -Path $LogFile -Value "[ERROR] $_"
    exit 1
}
