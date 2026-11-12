param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Image,
    [string]$Location = "Not determined",
    [string]$SourceId = "Not determined",
    [ValidateRange(0.0, 1.0)]
    [double]$DetectionConfidence = 0.30,
    [ValidateRange(-1.0, 1.0)]
    [double]$MatchThreshold = 0.33
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Python = Join-Path $Root "sapiensid\.conda\sapiensid\python.exe"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python environment not found: $Python"
}
if (-not (Test-Path -LiteralPath $Image -PathType Leaf)) {
    throw "Input image not found: $Image"
}
$Image = (Resolve-Path -LiteralPath $Image).Path
& $Python -m src.auto_workflow $Image --location $Location --source-id $SourceId --detection-confidence $DetectionConfidence --match-threshold $MatchThreshold
if ($LASTEXITCODE -ne 0) {
    throw "Surveillance workflow failed with exit code $LASTEXITCODE"
}
