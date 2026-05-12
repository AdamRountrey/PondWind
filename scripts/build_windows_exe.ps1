param(
    [string]$PythonExe = "",
    [string]$WindNinjaHome = "",
    [string]$CondaPrefix = "",
    [string]$DistDir = "",
    [string]$WorkDir = ""
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

function Resolve-PythonExe {
    param([string]$RequestedPythonExe)

    if (![string]::IsNullOrWhiteSpace($RequestedPythonExe)) {
        if (!(Test-Path $RequestedPythonExe)) {
            throw "Specified -PythonExe was not found: $RequestedPythonExe"
        }
        return (Resolve-Path $RequestedPythonExe).Path
    }

    $candidatePaths = @()
    if ($env:CONDA_PREFIX) {
        $candidatePaths += (Join-Path $env:CONDA_PREFIX "python.exe")
    }
    $candidatePaths += @(
        (Join-Path $projectRoot ".conda-sitewind\python.exe"),
        "python"
    )

    foreach ($candidate in $candidatePaths) {
        if ($candidate -eq "python") {
            $command = Get-Command python -ErrorAction SilentlyContinue
            if ($command) {
                return $command.Source
            }
        } elseif (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }

    throw "Unable to locate python.exe for the build. Pass -PythonExe explicitly or activate the Conda environment first."
}

function Resolve-CondaPrefix {
    param(
        [string]$RequestedCondaPrefix,
        [string]$ResolvedPythonExe
    )

    $candidateRoots = @()
    if (![string]::IsNullOrWhiteSpace($RequestedCondaPrefix)) {
        $candidateRoots += $RequestedCondaPrefix
    }
    if ($env:CONDA_PREFIX) {
        $candidateRoots += $env:CONDA_PREFIX
    }
    $candidateRoots += @(
        (Split-Path -Parent $ResolvedPythonExe),
        (Join-Path $projectRoot ".conda-sitewind")
    )

    foreach ($candidate in ($candidateRoots | Select-Object -Unique)) {
        if (Test-Path (Join-Path $candidate "Library")) {
            return (Resolve-Path $candidate).Path
        }
    }

    throw "Unable to locate a Conda-style runtime root with a Library directory. Pass -CondaPrefix explicitly or activate the intended environment."
}

function Resolve-WindNinjaHome {
    param([string]$RequestedWindNinjaHome)

    $candidateRoots = @()
    if (![string]::IsNullOrWhiteSpace($RequestedWindNinjaHome)) {
        $candidateRoots += $RequestedWindNinjaHome
    }
    if ($env:PONDWIND_WINDNINJA_HOME) {
        $candidateRoots += $env:PONDWIND_WINDNINJA_HOME
    }
    $candidateRoots += (Join-Path $projectRoot "tools\WindNinjaApp")

    foreach ($candidate in ($candidateRoots | Select-Object -Unique)) {
        $cliPath = Join-Path $candidate "bin\WindNinja_cli.exe"
        if (Test-Path $cliPath) {
            return (Resolve-Path $candidate).Path
        }
    }

    throw "Unable to locate WindNinja runtime. Stage it under tools\\WindNinjaApp or pass -WindNinjaHome explicitly."
}

$PythonExe = Resolve-PythonExe -RequestedPythonExe $PythonExe
$CondaPrefix = Resolve-CondaPrefix -RequestedCondaPrefix $CondaPrefix -ResolvedPythonExe $PythonExe
$WindNinjaHome = Resolve-WindNinjaHome -RequestedWindNinjaHome $WindNinjaHome

$env:PONDWIND_CONDA_PREFIX = $CondaPrefix
$env:PONDWIND_WINDNINJA_HOME = $WindNinjaHome

if ([string]::IsNullOrWhiteSpace($DistDir)) {
    $DistDir = Join-Path $projectRoot "dist"
}
if ([string]::IsNullOrWhiteSpace($WorkDir)) {
    $WorkDir = Join-Path $projectRoot "build"
}

Write-Host "Building PondWind.exe from $projectRoot"
Write-Host "Using python: $PythonExe"
Write-Host "Using conda root: $CondaPrefix"
Write-Host "Using WindNinja runtime: $WindNinjaHome"
Write-Host "Using dist dir: $DistDir"
Write-Host "Using work dir: $WorkDir"

Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $WorkDir
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $DistDir
& $PythonExe -m PyInstaller --noconfirm --clean --distpath $DistDir --workpath $WorkDir PondWind.spec

$distRoot = Join-Path $DistDir "PondWind"
$pythonRoot = Split-Path -Parent $PythonExe
$dllSourceRoots = @()
if ($env:CONDA_PREFIX) {
    $dllSourceRoots += $env:CONDA_PREFIX
}
if ($pythonRoot) {
    $dllSourceRoots += $pythonRoot
}
$dllSourceRoots += @(
    $CondaPrefix,
    (Join-Path $projectRoot ".conda-sitewind")
)
$dllSourceRoots = $dllSourceRoots | Select-Object -Unique

$rootDlls = @(
    "python311.dll",
    "python3.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "vcruntime140_threads.dll"
)

foreach ($dllName in $rootDlls) {
    $destPath = Join-Path $distRoot $dllName
    foreach ($sourceRoot in $dllSourceRoots) {
        $sourcePath = Join-Path $sourceRoot $dllName
        if (Test-Path $sourcePath) {
            Copy-Item -LiteralPath $sourcePath -Destination $destPath -Force
            break
        }
    }
}

$docsToShip = @(
    "README.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "SECURITY.md"
)

foreach ($docName in $docsToShip) {
    $sourcePath = Join-Path $projectRoot $docName
    $destPath = Join-Path $distRoot $docName
    if (Test-Path $sourcePath) {
        Copy-Item -LiteralPath $sourcePath -Destination $destPath -Force
    }
}

Write-Host ""
Write-Host "Build complete."
Write-Host "App folder: $distRoot"
Write-Host "Launch: $(Join-Path $distRoot 'PondWind.exe')"
