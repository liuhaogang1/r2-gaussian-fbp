param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$backend = Join-Path $root "r2_gaussian\submodules\self-xray-gaussian-rasterization-voxelization"

Push-Location $backend
try {
    & $Python setup.py build_ext --inplace
    if ($LASTEXITCODE -ne 0) {
        throw "Pose rasterizer build failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}
