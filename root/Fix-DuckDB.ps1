param(
  [string]$ProjectRoot = (Resolve-Path ".").Path,
  [switch]$KillAllPython = $false,
  [int]$PidToKill = 0
)

Write-Host "== Fix-DuckDB: התחלה ==" -ForegroundColor Cyan
Write-Host "ProjectRoot: $ProjectRoot"

# 1) קביעת נתיב DB מחוץ ל-OneDrive (פרסיסטנטי למשתמש)
$LocalApp = $env:LOCALAPPDATA
if (-not $LocalApp) {
  Write-Warning "LOCALAPPDATA לא מוגדר. אשתמש בשורש הפרויקט כגיבוי."
}
$SafeDb = if ($LocalApp) { Join-Path $LocalApp "pairs_trading_system\cache.duckdb" } else { Join-Path $ProjectRoot "cache.duckdb" }
$SafeDbDir = Split-Path $SafeDb -Parent
New-Item -ItemType Directory -Force -Path $SafeDbDir | Out-Null

# קבע משתנה סביבה קבוע (User) כדי שהקוד ישתמש בו
[Environment]::SetEnvironmentVariable("OPT_CACHE_PATH", $SafeDb, "User")
$env:OPT_CACHE_PATH = $SafeDb
Write-Host "OPT_CACHE_PATH => $SafeDb"

# 2) סגירת תהליך נועל (אופציונלי לפי PID או אוטומטי)
if ($PidToKill -gt 0) {
  try {
    Write-Host "הרג תהליך לפי PID: $PidToKill"
    Get-Process -Id $PidToKill -ErrorAction Stop | Stop-Process -Force
  } catch { Write-Warning "לא נמצא PID $PidToKill או שאין הרשאות." }
}

# חיפוש תהליכי python עם נתיב/קומנדליין שמכילים את תיקיית הפרויקט
$procs = Get-CimInstance Win32_Process | Where-Object {
  ($_.Name -match '^python(\.exe)?$') -and
  ($_.CommandLine -like "*$ProjectRoot*" -or $_.ExecutablePath -like "*miniconda*")
}

if ($procs) {
  Write-Host ("נמצאו {0} תהליכי python קשורים לפרויקט" -f $procs.Count)
  $procs | ForEach-Object {
    Write-Host ("-> PID {0} | {1}" -f $_.ProcessId, $_.CommandLine)
    try { Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop } catch {}
  }
} elseif ($KillAllPython) {
  Write-Warning "לא נמצאו תהליכי python קשורים. בגלל -KillAllPython אסגור את כולם."
  Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
}

# 3) ניקוי מטמוני __pycache__
Write-Host "מנקה __pycache__..."
Get-ChildItem -Path $ProjectRoot -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# 4) איתור מופעים מסוכנים של duckdb.connect( בכל הקוד (רק מציג; יש מצב שתרצה לאשר החלפה)
Write-Host "סורק duckdb.connect( בכל הפרויקט..."
$hits = Select-String -Path (Join-Path $ProjectRoot "*\*.py") -Pattern 'duckdb\.connect\(' -CaseSensitive -SimpleMatch | `
        Where-Object { $_.Path -notlike "*site-packages*" }

if ($hits) {
  Write-Host "נמצאו מופעים פוטנציאליים לפתיחה בזמן import:"
  $hits | ForEach-Object { Write-Host ("{0}:{1}  {2}" -f $_.Path, $_.LineNumber, $_.Line.Trim()) }

  # החלפה נקודתית בטוחה למקרה ה"גרוע" הנפוץ: DUCK = duckdb.connect(PROJECT_ROOT/..."cache.duckdb")
  Write-Host "מנסה לנטרל פתיחה ישירה ברמת מודול (החלפה מדויקת-שמורה)..."
  foreach ($f in ($hits | Select-Object -ExpandProperty Path -Unique)) {
    $content = Get-Content $f -Raw
    $orig = 'DUCK = duckdb.connect(str(PROJECT_ROOT / "cache.duckdb"))'
    if ($content -like "*$orig*") {
      $new = '# [auto-fix] disabled direct connect; use lazy factory in optimization_tab.py' + "`r`n" + '# ' + $orig
      $content = $content -replace [regex]::Escape($orig), [System.Text.RegularExpressions.Regex]::Escape($new)
      Set-Content -Path $f -Value $content -NoNewline
      Write-Host "עודכן: $f"
    }
  }
} else {
  Write-Host "לא נמצאו מופעים של duckdb.connect( מחוץ לתלויות."
}

# 5) וידוא ש-root\dashboard.py מייבא בצורה חד-משמעית מהחבילה
$Dashboard = Join-Path $ProjectRoot "root\dashboard.py"
if (Test-Path $Dashboard) {
  $dash = Get-Content $Dashboard -Raw
  if ($dash -notmatch 'from\s+root\.optimization_tab\s+import\s+render_optimization_tab') {
    # הוסף/עדכן import (לא מוחק קיים; רק מוסיף אם חסר)
    $dash = $dash -replace 'from\s+optimization_tab\s+import\s+render_optimization_tab','from root.optimization_tab import render_optimization_tab'
    if ($dash -notmatch 'from\s+root\.optimization_tab\s+import\s+render_optimization_tab') {
      $dash = "from root.optimization_tab import render_optimization_tab`r`n" + $dash
    }
    Set-Content -Path $Dashboard -Value $dash -NoNewline
    Write-Host "עודכן import ב-dashboard.py"
  } else {
    Write-Host "ייבוא ב-dashboard.py תקין."
  }
} else {
  Write-Warning "לא נמצא root\dashboard.py (דלג)."
}

# 6) הרצת Streamlit מהשורש
Write-Host "מריץ: streamlit run root\dashboard.py"
Write-Host "USING DB PATH => $env:OPT_CACHE_PATH"
Write-Host "== Fix-DuckDB: סיום הכנות, מפעיל אפליקציה ==" -ForegroundColor Cyan
Start-Process -FilePath "streamlit" -ArgumentList "run `"root\dashboard.py`"" -WorkingDirectory $ProjectRoot
