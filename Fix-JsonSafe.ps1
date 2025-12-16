param(
  [string]$ProjectRoot = (Resolve-Path ".").Path,
  [switch]$DryRun = $false
)

Write-Host "== JsonSafe Fix (Python) ==" -ForegroundColor Cyan
Write-Host "Root: $ProjectRoot"

$BackupDir = Join-Path $ProjectRoot "__backup_jsonsafe"
New-Item -ItemType Directory -Force -Path $BackupDir | Out-Null

# קבצים לסריקה (למעט וירטואלים/תלויות)
$files = Get-ChildItem -Path $ProjectRoot -Recurse -Filter *.py |
  Where-Object {
    $_.FullName -notmatch '\\(__pycache__|\.venv|venv|env|site-packages|build|dist)\\'
  }

# בלוק העזר שנוסיף בראש הקובץ אם אין make_json_safe
$helperBlock = @'
# ---- JSON safety helpers (auto-injected) ----
try:
    from common.json_safe import make_json_safe, json_default as _json_default  # type: ignore
except Exception:  # fallback local helpers
    from pathlib import Path as _Path
    import numpy as _np
    import pandas as _pd
    from datetime import date as _date, datetime as _dt

    def _json_default(obj):
        if isinstance(obj, (_Path, _pd.Timestamp, _date, _dt)):
            return str(obj)
        if isinstance(obj, (_np.integer, _np.floating)):
            return obj.item()
        return str(obj)

    def make_json_safe(obj):
        if isinstance(obj, dict):
            return {k: make_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [make_json_safe(v) for v in obj]
        if isinstance(obj, (_Path, _pd.Timestamp, _date, _dt)):
            return str(obj)
        if isinstance(obj, (_np.integer, _np.floating)):
            return obj.item()
        return obj
# ---------------------------------------------
'@

# פונקציה שמנסה להזריק בלוק עזר אחרי בלוק ה-importים
function Add-Helper-IfMissing {
  param([string]$content)
  if ($content -match '\bmake_json_safe\b') { return $content }  # כבר קיים

  # מצא מקום הזרקה (אחרי importים הראשונים או בתחילת הקובץ)
  $lines = $content -split "`r?`n"
  $insertAt = 0
  for ($i=0; $i -lt [Math]::Min($lines.Count,400); $i++) {
    $l = $lines[$i]
    if ($l -match '^\s*(import|from)\s+\S+') { $insertAt = $i }  # עד שורת import האחרונה
    if ($l -match '^\s*(def|class|@|if\s+__name__\s*==)') { break } # אל תעלה על קוד
  }
  $before = $lines[0..$insertAt] -join "`r`n"
  $after  = $lines[($insertAt+1)..($lines.Count-1)] -join "`r`n"
  return ($before + "`r`n" + $helperBlock + "`r`n" + $after)
}

# regexים זהירים (לא מושלמים, אבל עובדים ברוב המקרים הטיפוסיים)
$reStJson   = [regex]'st\.json\(\s*([^\)]*?)\s*\)'
$reDumps    = [regex]'json\.dumps\(\s*([^\)]*?)\s*\)'
$reDumpHead = [regex]'json\.dump\(\s*([^,]+)\s*,'  # עוטף את הפרמטר הראשון לפני הפסיק

$changed = 0
foreach ($f in $files) {
  $text = Get-Content $f.FullName -Raw
  $orig = $text
  $fileChanged = $false

  # 1) st.json(x) -> st.json(make_json_safe(x))
  if ($reStJson.IsMatch($text)) {
    $text = $reStJson.Replace($text, { param($m) "st.json(make_json_safe($($m.Groups[1].Value)))" })
    $fileChanged = $true
  }

  # 2) json.dumps(obj, ...) -> json.dumps(make_json_safe(obj), ...)
  if ($reDumps.IsMatch($text)) {
    # אם כבר יש make_json_safe בפנים, אל תכפיל
    $text = $reDumps.Replace($text, {
      param($m)
      $inside = $m.Groups[1].Value
      if ($inside -match '\bmake_json_safe\(') { "json.dumps($inside)" }
      else { "json.dumps(make_json_safe($inside))" }
    })
    $fileChanged = $true
  }

  # 3) json.dump(obj, fp, ...) -> json.dump(make_json_safe(obj), fp, ...)
  if ($reDumpHead.IsMatch($text)) {
    $text = $reDumpHead.Replace($text, {
      param($m)
      $first = $m.Groups[1].Value
      if ($first -match '\bmake_json_safe\(') { "json.dump($first," }
      else { "json.dump(make_json_safe($first)," }
    })
    $fileChanged = $true
  }

  # 4) אם שינינו—וודא שיש helper
  if ($fileChanged) {
    $text = Add-Helper-IfMissing -content $text
    if (-not $DryRun) {
      $rel = Resolve-Path $f.FullName
      $backup = Join-Path $BackupDir ($f.Name + ".bak")
      Copy-Item $rel $backup -Force
      Set-Content -Path $rel -Value $text -Encoding UTF8
      Write-Host ("✓ Updated " + $rel) -ForegroundColor Green
    } else {
      Write-Host ("~ Would update " + $f.FullName) -ForegroundColor Yellow
    }
    $changed++
  }
}

Write-Host ("Done. Files changed: {0}" -f $changed) -ForegroundColor Cyan
Write-Host ("Backups in: " + $BackupDir)
