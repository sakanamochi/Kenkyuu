[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'Medium')]
param()

# 新構成のデータと照合結果を確認し、指定した旧構成だけを削除する。
$ErrorActionPreference = 'Stop'
$taskRoot = [System.IO.Path]::GetFullPath($PSScriptRoot)
$taskNewRoot = [System.IO.Path]::GetFullPath((Join-Path $taskRoot 'paf_rebuild'))
$taskPrefix = $taskNewRoot + [System.IO.Path]::DirectorySeparatorChar
$taskInventoryPath = Join-Path $taskNewRoot 'data/inventory.json'
$taskCheckPath = Join-Path $taskNewRoot 'outputs/standalone_training_check/verification.json'
$taskReferencePath = Join-Path $taskNewRoot 'docs/training_data_verification.json'

# Git履歴・仮想環境・研究関連書類・保護された設定フォルダは対象にしない。
$taskNames = @(
    'assets', 'blender', 'config', 'docs', 'paf_ring_detection',
    'parametric_paf', 'research', 'tests', 'output', '.pytest_cache',
    '.chart-data-3UVK98', '.chart-data-duBS9g', '.chart-data-kcyDwz',
    '.chart-data-KIaGUh', '.chart-data-l3uGrn', '.chart-data-q1mXRo',
    '.chart-data-YeWiIh', 'run.py', 'SYSTEM_MAP.md',
    'requirements.txt', 'requirements-cuda.txt'
)

$taskInventory = Get-Content -LiteralPath $taskInventoryPath -Raw -Encoding UTF8 | ConvertFrom-Json
$taskCheck = Get-Content -LiteralPath $taskCheckPath -Raw -Encoding UTF8 | ConvertFrom-Json
$taskReference = Get-Content -LiteralPath $taskReferencePath -Raw -Encoding UTF8 | ConvertFrom-Json
$taskCount = ($taskCheck.summaries | Measure-Object -Property samples -Sum).Sum
if ($taskCount -ne 5446 -or $taskCheck.seed -ne 20260907) {
    throw 'The complete training-input check is missing.'
}
foreach ($taskRow in $taskCheck.summaries) {
    $taskExpected = @($taskReference.summaries | Where-Object {
        $_.source -eq $taskRow.source -and $_.split -eq $taskRow.split
    })
    if ($taskExpected.Count -ne 1 -or
        $taskExpected[0].samples -ne $taskRow.samples -or
        $taskExpected[0].tensor_sequence_sha256 -ne $taskRow.tensor_sequence_sha256) {
        throw 'The relocated training inputs do not match the reference.'
    }
}

$taskFileCount = 0
foreach ($taskEntry in $taskInventory.files_sha256.PSObject.Properties) {
    $taskFile = [System.IO.Path]::GetFullPath((Join-Path $taskNewRoot $taskEntry.Name))
    if (-not $taskFile.StartsWith($taskPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Asset path outside paf_rebuild: $taskFile"
    }
    if ((Get-FileHash -LiteralPath $taskFile -Algorithm SHA256).Hash -ne $taskEntry.Value) {
        throw "Asset hash mismatch: $taskFile"
    }
    $taskFileCount++
    if ($taskFileCount % 500 -eq 0) { Write-Host "Verified $taskFileCount assets" }
}
if ($taskFileCount -ne 6396) { throw 'The required asset inventory is incomplete.' }

& git -C $taskRoot rev-parse --verify 'codex/hole-region-comparison' | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'The source branch is unavailable.' }

# 全対象の絶対パスを、削除前に作業フォルダ直下へ限定する。
$taskTargets = @()
foreach ($taskName in $taskNames) {
    $taskPath = [System.IO.Path]::GetFullPath((Join-Path $taskRoot $taskName))
    if ([System.IO.Path]::GetDirectoryName($taskPath) -ne $taskRoot) {
        throw "Unexpected target: $taskPath"
    }
    if (-not (Test-Path -LiteralPath $taskPath)) { continue }
    $taskItem = Get-Item -LiteralPath $taskPath -Force
    if ($taskItem.PSIsContainer) {
        if ($taskItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
            throw "Reparse target is not allowed: $taskPath"
        }
        $taskLinks = @(Get-ChildItem -LiteralPath $taskPath -Recurse -Directory -Force |
            Where-Object { $_.Attributes -band [System.IO.FileAttributes]::ReparsePoint })
        if ($taskLinks.Count -ne 0) { throw "Linked directory found inside: $taskPath" }
    }
    $taskTargets += $taskItem
}

foreach ($taskItem in $taskTargets) {
    if ($PSCmdlet.ShouldProcess($taskItem.FullName, 'Delete legacy files')) {
        if ($taskItem.PSIsContainer) {
            Remove-Item -LiteralPath $taskItem.FullName -Recurse -Force -ErrorAction Stop
        } else {
            Remove-Item -LiteralPath $taskItem.FullName -Force -ErrorAction Stop
        }
        Write-Host "Deleted: $($taskItem.Name)"
    }
}
Write-Host 'Finished. paf_rebuild, research documents, .venv and Git history are preserved.'
