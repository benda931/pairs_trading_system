# migrate_pydantic_v2_local.ps1
param([switch]$DryRun)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Get-Location).Path
Write-Host "== Pydantic v2 Migration (scoped to $ProjectRoot) ==" -ForegroundColor Cyan

# 0) תלות
Write-Host ">> Ensuring pydantic-settings is installed..." -ForegroundColor Yellow
pip show pydantic-settings 1>$null 2>$null; if ($LASTEXITCODE -ne 0) { pip install "pydantic-settings>=2.0" }

# 1) גיבוי
$backupDir = Join-Path $ProjectRoot (".backup_pydantic_migration_{0}" -f (Get-Date -Format "yyyyMMddHHmmss"))
Write-Host ">> Creating backup in $backupDir ..." -ForegroundColor Yellow
New-Item -ItemType Directory -Path $backupDir | Out-Null

# 2) קבצים לסריקה (ללא venv/__pycache__/backup)
$excludeDirs = @("\.venv\", "\__pycache__\", "\.backup_pydantic_migration_")
$files = Get-ChildItem -Path $ProjectRoot -Recurse -File -Filter *.py -ErrorAction SilentlyContinue |
  Where-Object { $full=$_.FullName; -not ($excludeDirs | ForEach-Object { $full -match $_ }) }

# גיבוי
foreach ($f in $files) {
  $dest = Join-Path $backupDir ($f.FullName.Substring($ProjectRoot.Length).TrimStart('\'))
  New-Item -ItemType Directory -Path (Split-Path $dest) -Force | Out-Null
  Copy-Item $f.FullName $dest
}

# 3) עיבוד
$changed = 0
foreach ($f in $files) {
  $content = Get-Content $f.FullName -Raw
  $original = $content
  $fileChanged = $false

  # 3.a) החלפת import
  if ($content -match 'from\s+pydantic\s+import\s+BaseSettings') {
    $content = $content -replace 'from\s+pydantic\s+import\s+BaseSettings','from pydantic_settings import BaseSettings'
    $fileChanged = $true
  }

  # 3.b) ודא SettingsConfigDict אם יש מחלקה שיורשת BaseSettings
  $usesBaseSettings = $content -match 'class\s+\w+\s*\(\s*BaseSettings\s*\)\s*:'
  $hasSettingsConfigImport = $content -match 'from\s+pydantic_settings\s+import\s+.*\bSettingsConfigDict\b'
  if ($usesBaseSettings -and -not $hasSettingsConfigImport) {
    # תוספת import אחרי בלוק ה-imports אם נמצא, אחרת בתחילת הקובץ
    if ($content -match '(^(\s*from\s+\S+\s+import[^\r\n]*|^\s*import\s+\S+)[^\r\n]*\r?\n)+') {
      $importsBlock = $Matches[0]
      $newImportsBlock = $importsBlock + "from pydantic_settings import SettingsConfigDict`r`n"
      $content = $content -replace [regex]::Escape($importsBlock), $newImportsBlock
    } else {
      $content = "from pydantic_settings import SettingsConfigDict`r`n" + $content
    }
    $fileChanged = $true
  }

  # 3.c) משוך ערכי env_file/extra אם קיימים בקובץ (ברירת מחדל אם לא)
  $envFile = ([regex]::Match($content,'env_file\s*=\s*["'']([^"''\r\n]+)["'']').Groups[1].Value); if (-not $envFile) { $envFile = ".env" }
  $extraBehavior = ([regex]::Match($content,'extra\s*=\s*["'']([^"''\r\n]+)["'']').Groups[1].Value); if (-not $extraBehavior) { $extraBehavior = "allow" }

  # 3.d) הזרקת model_config מיד אחרי כותרת מחלקה (BaseSettings) אם אין
  $content = [System.Text.RegularExpressions.Regex]::Replace(
    $content,
    'class\s+([A-Za-z_]\w*)\s*\(\s*BaseSettings\s*\)\s*:\s*\r?\n(?!\s*model_config\s*=)',
    { param($m) $m.Value + "    model_config = SettingsConfigDict(env_file=""$envFile"", extra=""$extraBehavior"")`r`n" },
    [System.Text.RegularExpressions.RegexOptions]::Multiline
  )

  if ($content -ne $original) { $fileChanged = $true }
  if ($fileChanged) {
    if ($DryRun) {
      Write-Host "DRY-RUN: Would update $($f.FullName)" -ForegroundColor DarkYellow
    } else {
      Set-Content -Path $f.FullName -Value $content -Encoding UTF8
      Write-Host "Updated: $($f.FullName)"
      $changed++
    }
  }
}

if ($DryRun) {
  Write-Host "== DRY RUN complete. ==" -ForegroundColor Yellow
} else {
  Write-Host "== Migration complete. Files changed: $changed ==" -ForegroundColor Green
  Write-Host "Backup: $backupDir" -ForegroundColor Green
}
Write-Host "`nNext: streamlit run root/dashboard.py`n" -ForegroundColor Cyan
