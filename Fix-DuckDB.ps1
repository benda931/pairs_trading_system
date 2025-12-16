param(
  [string] = (Resolve-Path ".").Path,
  [switch] = False,
  [int] = 0
)

Write-Host "== Fix-DuckDB Lite ==" -ForegroundColor Cyan
Write-Host "ProjectRoot: "

# 1) קבע DB מחוץ ל-OneDrive
 = C:\Users\omrib\AppData\Local
if (-not ) { Write-Warning "LOCALAPPDATA לא מוגדר, אשתמש בשורש הפרויקט כגיבוי." }
 = if () { Join-Path  "pairs_trading_system\cache.duckdb" } else { Join-Path  "cache.duckdb" }
 = Split-Path  -Parent
New-Item -ItemType Directory -Force -Path  | Out-Null
[Environment]::SetEnvironmentVariable("OPT_CACHE_PATH", , "User")
 = 
Write-Host "OPT_CACHE_PATH => "

# 2) סגור תהליך נועל
if ( -gt 0) {
  try { Get-Process -Id  -ErrorAction Stop | Stop-Process -Force } catch { Write-Warning "לא נמצא PID " }
} else {
  # סגור רק python שרץ מתוך התיקייה הזו
  Get-CimInstance Win32_Process | Where-Object {
    (.Name -match '^python(\.exe)?$') -and (.CommandLine -like "**")
  } | ForEach-Object {
    try { Stop-Process -Id .ProcessId -Force } catch {}
  }
  if () {
    Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
  }
}

# 3) ניקוי __pycache__
Get-ChildItem -Path  -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# 4) ודא ייבוא חד-משמעי ב-dashboard.py
 = Join-Path  "root\dashboard.py"
if (Test-Path ) {
   = Get-Content  -Raw
   =  -replace 'from\s+optimization_tab\s+import\s+render_optimization_tab','from root.optimization_tab import render_optimization_tab'
  if ( -notmatch 'from\s+root\.optimization_tab\s+import\s+render_optimization_tab') {
     = "from root.optimization_tab import render_optimization_tab
"
  }
  if ( -ne ) { Set-Content -Path  -Value  -NoNewline }
}

# 5) הרץ Streamlit מהשורש
Write-Host "USING DB PATH => "
Start-Process -FilePath "streamlit" -ArgumentList "run", "root\dashboard.py" -WorkingDirectory 
Write-Host "== Done ==" -ForegroundColor Cyan
