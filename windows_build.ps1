<#
.SYNOPSIS
    Windows-native build helper for CUDA_Practice.
    Mirrors the Makefile targets: configure, build, debug, profile, ncu, clean, clean-all.

.DESCRIPTION
    Use this instead of `make` on Windows (PowerShell 5.1+).
    All targets call CMake/nvcc directly — no Unix tools required.

.PARAMETER Target
    configure | build | debug | profile | ncu | clean | clean-all | libwb
    Default: build

.PARAMETER File
    Optional .cu filename (e.g. hello_world.cu) to act on a single target.

.EXAMPLE
    .\windows_build.ps1 configure
    .\windows_build.ps1 build hello_world.cu
    .\windows_build.ps1 debug hello_world.cu
    .\windows_build.ps1 profile hello_world.cu
    .\windows_build.ps1 ncu    hello_world.cu
    .\windows_build.ps1 clean
#>

param(
    [Parameter(Position = 0)]
    [ValidateSet('configure', 'build', 'debug', 'profile', 'ncu', 'clean', 'clean-all', 'libwb')]
    [string]$Target = 'build',

    [Parameter(Position = 1)]
    [string]$File = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$BuildDir    = 'build'
$LibwbDir    = 'extern/libwb'
$ProfilesDir = "$BuildDir/profiles"

# Derive target executable name (strip .cu if provided)
$TargetName = if ($File) { [System.IO.Path]::GetFileNameWithoutExtension($File) } else { '' }

# ── helpers ──────────────────────────────────────────────────────────────────

function Invoke-Cmd {
    param([string]$Cmd, [string[]]$Args)
    Write-Host "> $Cmd $($Args -join ' ')" -ForegroundColor DarkGray
    & $Cmd @Args
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $Cmd $($Args -join ' ')" }
}

function Require-Binary {
    param([string]$Exe)
    $path = "$BuildDir/$Exe.exe"
    if (-not (Test-Path $path)) {
        throw "Binary not found: $path — run first: .\windows_build.ps1 build $File"
    }
    return $path
}

# ── targets ──────────────────────────────────────────────────────────────────

switch ($Target) {

    'configure' {
        Write-Host '→ Installing Python deps (Poetry)...' -ForegroundColor Cyan
        Invoke-Cmd poetry @('install')

        Write-Host '→ Initialising submodules...' -ForegroundColor Cyan
        Invoke-Cmd git @('submodule', 'update', '--init', 'extern/ECE408_SP25', $LibwbDir)
        Invoke-Cmd git @('-C', $LibwbDir, 'checkout', 'master')

        Write-Host '→ Building libwb (static, once)...' -ForegroundColor Cyan
        Invoke-Cmd cmake @('-S', $LibwbDir, '-B', "$LibwbDir/build", '-DCMAKE_BUILD_TYPE=Release', '-Wno-dev')
        Invoke-Cmd cmake @('--build', "$LibwbDir/build", '--target', 'wb')
        Write-Host "→ libwb built: $LibwbDir/build/wb.lib" -ForegroundColor Green
    }

    'libwb' {
        Write-Host '→ Building libwb...' -ForegroundColor Cyan
        New-Item -ItemType Directory -Force -Path "$LibwbDir/build" | Out-Null
        Invoke-Cmd cmake @('-S', $LibwbDir, '-B', "$LibwbDir/build", '-DCMAKE_BUILD_TYPE=Release', '-Wno-dev')
        Invoke-Cmd cmake @('--build', "$LibwbDir/build", '--target', 'wb')
        Write-Host "→ libwb built." -ForegroundColor Green
    }

    'build' {
        New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
        Invoke-Cmd cmake @('-S', '.', '-B', $BuildDir, '-DCMAKE_BUILD_TYPE=Release', '-Wno-dev')

        if ($TargetName) {
            Write-Host "Building $TargetName..." -ForegroundColor Cyan
            Invoke-Cmd cmake @('--build', $BuildDir, '--target', $TargetName)
            Write-Host "`n=== $TargetName output ===" -ForegroundColor Yellow
            $bin = "$BuildDir/$TargetName.exe"
            & $bin | Tee-Object -FilePath "$BuildDir/${TargetName}_output.txt"
        } else {
            Write-Host 'Building all CUDA targets...' -ForegroundColor Cyan
            Invoke-Cmd cmake @('--build', $BuildDir)
            Get-ChildItem src -Directory | ForEach-Object {
                $name = $_.Name
                $bin  = "$BuildDir/$name.exe"
                if (Test-Path $bin) {
                    Write-Host "`n=== $name output ===" -ForegroundColor Yellow
                    & $bin | Tee-Object -FilePath "$BuildDir/${name}_output.txt"
                }
            }
        }
    }

    'debug' {
        $DebugDir = "$BuildDir/debug"
        New-Item -ItemType Directory -Force -Path $DebugDir | Out-Null
        Invoke-Cmd cmake @('-S', '.', '-B', $DebugDir, '-DCMAKE_BUILD_TYPE=Debug', '-Wno-dev')

        if ($TargetName) {
            Write-Host "Building $TargetName (debug)..." -ForegroundColor Cyan
            Invoke-Cmd cmake @('--build', $DebugDir, '--target', $TargetName)
            Write-Host "Debug binary: $DebugDir/$TargetName.exe" -ForegroundColor Green
        } else {
            Write-Host 'Building all CUDA targets (debug)...' -ForegroundColor Cyan
            Invoke-Cmd cmake @('--build', $DebugDir)
            Write-Host "Debug binaries in: $DebugDir/" -ForegroundColor Green
        }
    }

    'profile' {
        if (-not $TargetName) { throw 'Usage: .\windows_build.ps1 profile <file.cu>' }
        $bin = Require-Binary $TargetName
        New-Item -ItemType Directory -Force -Path $ProfilesDir | Out-Null
        Write-Host "Profiling $TargetName with Nsight Systems..." -ForegroundColor Cyan
        Invoke-Cmd nsys @(
            'profile',
            "--output=$ProfilesDir/${TargetName}_nsys",
            '--stats=true',
            '--force-overwrite=true',
            $bin
        )
        Write-Host "Report saved: $ProfilesDir/${TargetName}_nsys.nsys-rep" -ForegroundColor Green
    }

    'ncu' {
        if (-not $TargetName) { throw 'Usage: .\windows_build.ps1 ncu <file.cu>' }
        $bin = Require-Binary $TargetName
        New-Item -ItemType Directory -Force -Path $ProfilesDir | Out-Null
        Write-Host "Analyzing $TargetName with Nsight Compute..." -ForegroundColor Cyan
        Invoke-Cmd ncu @(
            '--set', 'full',
            "--export=$ProfilesDir/${TargetName}_ncu",
            '--force-overwrite',
            $bin
        )
        Write-Host "Report saved: $ProfilesDir/${TargetName}_ncu.ncu-rep" -ForegroundColor Green
    }

    'clean' {
        if (Test-Path $BuildDir) {
            Remove-Item $BuildDir -Recurse -Force
        }
        Get-ChildItem -Recurse -Filter '__pycache__' | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Get-ChildItem -Recurse -Filter '*.pyc'       | Remove-Item        -Force -ErrorAction SilentlyContinue
        Write-Host 'Cleaned.' -ForegroundColor Green
    }

    'clean-all' {
        & "$PSScriptRoot/windows_build.ps1" clean
        poetry env remove --all 2>$null
        Write-Host 'Removed Poetry environment.' -ForegroundColor Green
    }
}
