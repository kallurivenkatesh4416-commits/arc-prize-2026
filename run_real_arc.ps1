param(
  [string]$Game = "ls20",
  [string]$Commit = "0d6bfca5a8ee00c3ff349c3d5d78a2544f661172",
  [string]$KeyFile = ""
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

function Normalize-ArcKey {
  param([string]$Raw)

  $value = $Raw.Trim()
  if ($value -match '^\s*(ARC_API_KEY|arc_competition_api_key)\s*=\s*(.+)\s*$') {
    $value = $Matches[2].Trim()
  }
  return $value.Trim('"').Trim("'")
}

function Read-ArcKeyFile {
  param([string]$Path)

  $raw = Get-Content -LiteralPath $Path -Raw
  foreach ($line in ($raw -split "`r?`n")) {
    $trimmed = $line.Trim()
    if (-not $trimmed -or $trimmed.StartsWith("#")) {
      continue
    }
    if ($trimmed -match '^\s*ARC_API_KEY\s*=\s*(.+)\s*$') {
      return Normalize-ArcKey $Matches[1]
    }
  }
  return Normalize-ArcKey $raw
}

function Get-ArcApiKey {
  if ($env:ARC_API_KEY) {
    return Normalize-ArcKey $env:ARC_API_KEY
  }

  if ($KeyFile) {
    $path = $KeyFile
    if (-not [System.IO.Path]::IsPathRooted($path)) {
      $path = Join-Path $RepoRoot $path
    }
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
      throw "Key file not found: $path"
    }
    return Read-ArcKeyFile $path
  }

  $envPath = Join-Path $RepoRoot ".env"
  if (Test-Path -LiteralPath $envPath -PathType Leaf) {
    $key = Read-ArcKeyFile $envPath
    if ($key) {
      return $key
    }
  }

  $secureKey = Read-Host "Paste ARC API key; input will be hidden" -AsSecureString
  $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
  try {
    return Normalize-ArcKey ([Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr))
  }
  finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
  }
}

$env:ARC_API_KEY = Get-ArcApiKey
if ([string]::IsNullOrWhiteSpace($env:ARC_API_KEY)) {
  throw "ARC API key was empty."
}
Write-Host "ARC_API_KEY loaded. Secret value hidden."

New-Item -ItemType Directory -Force -Path "runs" | Out-Null

$shortCommit = $Commit.Substring(0, [Math]::Min(7, $Commit.Length))
$sourceUrl = "https://github.com/kallurivenkatesh4416-commits/arc-prize-2026/commit/$Commit"
$safeGame = $Game -replace '[^A-Za-z0-9_.-]', '_'
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$outPath = Join-Path "runs" "$safeGame-$shortCommit-$stamp.json"
$tags = @("offline-controller", "real-api", $Game, $shortCommit) | Select-Object -Unique

$python = ".\.venv\Scripts\python.exe"
$pythonArgs = @()
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
  $python = "py"
  $pythonArgs = @("-3.12")
}

Write-Host "Running offline controller for $Game through arc_agi.Arcade..."
& $python @pythonArgs -m agent.offline_controller `
  --game $Game `
  --record `
  --tags $tags `
  --source-url $sourceUrl `
  --out $outPath

if ($LASTEXITCODE -ne 0) {
  throw "offline_controller exited with code $LASTEXITCODE."
}

Write-Host "Run output JSON: $outPath"
