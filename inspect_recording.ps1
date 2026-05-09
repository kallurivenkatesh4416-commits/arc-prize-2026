param(
  [string]$Path = "",
  [int]$First = 8
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

if (-not $Path) {
  $latest = Get-ChildItem -LiteralPath $RepoRoot -File -Force -Filter "*.recording.jsonl" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

  if (-not $latest) {
    throw "No *.recording.jsonl files found in $RepoRoot."
  }

  $Path = $latest.FullName
}

if (-not [System.IO.Path]::IsPathRooted($Path)) {
  $Path = Join-Path $RepoRoot $Path
}

if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
  throw "Recording not found: $Path"
}

Write-Host "Recording: $Path"
Write-Host "Showing first $First frames. Frame/grid data omitted."

$index = 0
Get-Content -LiteralPath $Path -TotalCount $First | ForEach-Object {
  $index += 1
  $raw = $_
  if ([string]::IsNullOrWhiteSpace($raw)) {
    return
  }

  try {
    $json = $raw | ConvertFrom-Json
  }
  catch {
    Write-Host ("{0}: INVALID_JSON" -f $index)
    return
  }

  $data = $json.data
  if (-not $data) {
    $data = $json
  }

  $actionInput = $data.action_input
  $actionId = $null
  $actionDataKeys = @()
  if ($actionInput) {
    $actionId = $actionInput.id
    if ($actionInput.data) {
      $actionDataKeys = @($actionInput.data.PSObject.Properties.Name)
    }
  }

  $available = @()
  if ($data.available_actions) {
    $available = @($data.available_actions | ForEach-Object {
      if ($_ -is [string]) {
        $_
      }
      elseif ($_.name) {
        $_.name
      }
      elseif ($_.id) {
        $_.id
      }
      else {
        $_
      }
    })
  }

  [PSCustomObject]@{
    frame = $index
    game_id = $data.game_id
    state = $data.state
    score = $data.score
    levels_completed = $data.levels_completed
    action_id = $actionId
    action_data_keys = ($actionDataKeys -join ",")
    available_actions = ($available -join ",")
  }
} | Format-Table -AutoSize
