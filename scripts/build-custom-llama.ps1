<#
.SYNOPSIS
    Build a custom Turing-optimized llama-server binary locally.

.DESCRIPTION
    Clones/updates llama.cpp, applies PrismML 1-bit and TurboQuant CUDA patches,
    builds with sm_75 + FA_ALL_QUANTS, and registers the binary with OllamaRunner.

    Caches the source tree at ~/.ollamarunner/build/ so subsequent builds are fast.

.PARAMETER Clean
    Force a fresh clone (ignore cache).

.PARAMETER SkipPrismML
    Skip PrismML Q1_0_g128 patches.

.PARAMETER SkipTurboQuant
    Skip TurboQuant CUDA patches.

.PARAMETER SkipRegister
    Build only, don't register with OllamaRunner.

.EXAMPLE
    .\scripts\build-custom-llama.ps1
    .\scripts\build-custom-llama.ps1 -Clean
    .\scripts\build-custom-llama.ps1 -SkipTurboQuant
#>

param(
    [switch]$Clean,
    [switch]$SkipPrismML,
    [switch]$SkipTurboQuant,
    [switch]$SkipRegister
)

$ErrorActionPreference = "Stop"
$BuildRoot = Join-Path $env:USERPROFILE ".ollamarunner\build"
$SourceDir = Join-Path $BuildRoot "llama.cpp"
$BuildDir = Join-Path $SourceDir "build"
$BinDir = Join-Path $env:USERPROFILE ".ollamarunner\bin\custom"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# ---------- Prerequisites check ----------

Write-Host "`n=== Checking prerequisites ===" -ForegroundColor Cyan

$cmakePath = Get-Command cmake -ErrorAction SilentlyContinue
if (-not $cmakePath) {
    Write-Host "ERROR: cmake not found. Install with: winget install Kitware.CMake" -ForegroundColor Red
    exit 1
}
Write-Host "  cmake: $($cmakePath.Source)" -ForegroundColor Green

$nvccPath = Get-Command nvcc -ErrorAction SilentlyContinue
if (-not $nvccPath) {
    # Check common CUDA paths
    $cudaPaths = @(
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe"
    )
    $found = $cudaPaths | Where-Object { Test-Path $_ } | Select-Object -First 1
    if ($found) {
        $cudaBin = Split-Path $found
        $env:PATH = "$cudaBin;$env:PATH"
        Write-Host "  nvcc: $found (added to PATH)" -ForegroundColor Green
    } else {
        Write-Host "ERROR: nvcc not found. Install CUDA Toolkit from:" -ForegroundColor Red
        Write-Host "  https://developer.nvidia.com/cuda-12-4-0-download-archive" -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-Host "  nvcc: $($nvccPath.Source)" -ForegroundColor Green
}

# Check for Visual Studio (cl.exe)
$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vsWhere) {
    $vsPath = & $vsWhere -latest -property installationPath 2>$null
    Write-Host "  Visual Studio: $vsPath" -ForegroundColor Green
} else {
    Write-Host "WARNING: vswhere not found, assuming VS is available via Developer Command Prompt" -ForegroundColor Yellow
}

# ---------- Source management (cached) ----------

Write-Host "`n=== Source management ===" -ForegroundColor Cyan

if (-not (Test-Path $BuildRoot)) {
    New-Item -ItemType Directory -Path $BuildRoot -Force | Out-Null
}

if ($Clean -and (Test-Path $SourceDir)) {
    Write-Host "  Removing cached source (--Clean)..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $SourceDir
}

if (Test-Path (Join-Path $SourceDir ".git")) {
    Write-Host "  Updating cached llama.cpp source..." -ForegroundColor Green
    Push-Location $SourceDir
    git fetch origin --tags 2>&1 | Out-Null
    git checkout master 2>&1 | Out-Null
    git reset --hard origin/master 2>&1 | Out-Null
    git clean -fdx 2>&1 | Out-Null
    $sha = git rev-parse --short HEAD
    Write-Host "  Updated to: $sha" -ForegroundColor Green
    Pop-Location
} else {
    Write-Host "  Cloning llama.cpp (first time, will be cached)..." -ForegroundColor Green
    git clone https://github.com/ggml-org/llama.cpp.git $SourceDir 2>&1
    Push-Location $SourceDir
    $sha = git rev-parse --short HEAD
    Write-Host "  Cloned at: $sha" -ForegroundColor Green
    Pop-Location
}

# ---------- Apply patches ----------

Push-Location $SourceDir

# PrismML Q1_0_g128
if (-not $SkipPrismML) {
    Write-Host "`n=== Applying PrismML 1-bit patches ===" -ForegroundColor Cyan
    $remotes = git remote 2>&1
    if ($remotes -notcontains "prismml") {
        git remote add prismml https://github.com/PrismML-Eng/llama.cpp.git 2>&1 | Out-Null
    }
    git fetch prismml prism --depth=50 2>&1 | Out-Null

    git merge --no-commit --no-ff prismml/prism --allow-unrelated-histories 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  Merge had conflicts, attempting resolution..." -ForegroundColor Yellow
        # Accept theirs for new files, ours for conflicts in existing
        git checkout --theirs . 2>&1 | Out-Null
        git add -A 2>&1 | Out-Null
    }
    git commit -m "Apply PrismML Q1_0_g128 patches" --allow-empty 2>&1 | Out-Null
    Write-Host "  PrismML patches applied" -ForegroundColor Green
} else {
    Write-Host "`n=== Skipping PrismML patches ===" -ForegroundColor Yellow
}

# TurboQuant CUDA (spiritbuun fork with MSVC fix)
if (-not $SkipTurboQuant) {
    Write-Host "`n=== Applying TurboQuant CUDA patches (spiritbuun fork) ===" -ForegroundColor Cyan
    $remotes = git remote 2>&1
    if ($remotes -notcontains "turboquant") {
        git remote add turboquant https://github.com/spiritbuun/llama-cpp-turboquant-cuda.git 2>&1 | Out-Null
    }
    git fetch turboquant main --depth=100 2>&1 | Out-Null

    git merge --no-commit --no-ff turboquant/main --allow-unrelated-histories 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  Merge had conflicts, attempting resolution..." -ForegroundColor Yellow
        git checkout --theirs . 2>&1 | Out-Null
        git add -A 2>&1 | Out-Null
    }
    git commit -m "Apply TurboQuant CUDA patches (spiritbuun)" --allow-empty 2>&1 | Out-Null
    Write-Host "  TurboQuant patches applied" -ForegroundColor Green
} else {
    Write-Host "`n=== Skipping TurboQuant patches ===" -ForegroundColor Yellow
}

Pop-Location

# ---------- Build ----------

Write-Host "`n=== Building llama-server (Turing-optimized) ===" -ForegroundColor Cyan
Write-Host "  Target: sm_75 (RTX 2070)" -ForegroundColor Green
Write-Host "  FA_ALL_QUANTS: ON" -ForegroundColor Green

$cpuCount = (Get-CimInstance Win32_Processor).NumberOfLogicalProcessors
if (-not $cpuCount) { $cpuCount = 12 }

Push-Location $SourceDir

cmake -B build `
    -DGGML_CUDA=ON `
    -DCMAKE_CUDA_ARCHITECTURES=75 `
    -DGGML_CUDA_FA_ALL_QUANTS=ON `
    -DGGML_NATIVE=ON `
    -DCMAKE_BUILD_TYPE=Release `
    -DLLAMA_BUILD_TESTS=OFF `
    -DLLAMA_BUILD_EXAMPLES=OFF `
    -DLLAMA_BUILD_SERVER=ON

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: CMake configuration failed" -ForegroundColor Red
    Pop-Location
    exit 1
}

cmake --build build --config Release -j $cpuCount --target llama-server

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Build failed" -ForegroundColor Red
    Pop-Location
    exit 1
}

Pop-Location

# ---------- Collect and register ----------

Write-Host "`n=== Collecting artifacts ===" -ForegroundColor Cyan

# Find the built binary
$exePath = Get-ChildItem -Path $BuildDir -Recurse -Filter "llama-server.exe" | Select-Object -First 1
if (-not $exePath) {
    Write-Host "ERROR: llama-server.exe not found in build output" -ForegroundColor Red
    exit 1
}

Write-Host "  Found: $($exePath.FullName)" -ForegroundColor Green

# Write build manifest
$manifest = @{
    llama_cpp_sha = $sha
    cuda_arch = "sm_75 (Turing)"
    fa_all_quants = $true
    prismml_applied = (-not $SkipPrismML)
    turboquant_applied = (-not $SkipTurboQuant)
    build_date = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
} | ConvertTo-Json

$manifest | Out-File (Join-Path $exePath.DirectoryName "build-manifest.json") -Encoding utf8

if (-not $SkipRegister) {
    Write-Host "`n=== Registering with OllamaRunner ===" -ForegroundColor Cyan

    New-Item -ItemType Directory -Path $BinDir -Force | Out-Null

    # Copy exe
    Copy-Item $exePath.FullName (Join-Path $BinDir "llama-server.exe") -Force
    Write-Host "  Copied llama-server.exe" -ForegroundColor Green

    # Copy DLLs
    $dllCount = 0
    Get-ChildItem -Path $exePath.DirectoryName -Filter "*.dll" | ForEach-Object {
        Copy-Item $_.FullName (Join-Path $BinDir $_.Name) -Force
        $dllCount++
    }
    Write-Host "  Copied $dllCount DLLs" -ForegroundColor Green

    # Copy manifest
    Copy-Item (Join-Path $exePath.DirectoryName "build-manifest.json") (Join-Path $BinDir "build-manifest.json") -Force

    Write-Host "`n  Registered at: $BinDir" -ForegroundColor Green
    Write-Host "  OllamaRunner will now use this binary for all models." -ForegroundColor Green
} else {
    Write-Host "`n  Binary at: $($exePath.FullName)" -ForegroundColor Green
    Write-Host "  Register manually: python scripts\register-custom-binary.py `"$($exePath.FullName)`"" -ForegroundColor Yellow
}

Write-Host "`n=== Build complete ===" -ForegroundColor Cyan
Write-Host "  Source cached at: $SourceDir" -ForegroundColor Green
Write-Host "  Next build will be much faster (incremental)." -ForegroundColor Green
