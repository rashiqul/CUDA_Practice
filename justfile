# ── Shell configuration ───────────────────────────────────────────────────────
# Simple recipes (no shebang) run in sh on Unix or PowerShell on Windows.
# Complex recipes declare their own interpreter via a shebang line:
#   #!/usr/bin/env bash  (Unix)
#   #!/usr/bin/env pwsh  (Windows — requires PowerShell 7+)
#
# Install just:  cargo install just
#   Windows:     winget install Casey.Just
#   Ubuntu/WSL:  cargo install just  OR  apt install just

set windows-shell := ["powershell.exe", "-NoProfile", "-Command"]

# ── Variables ─────────────────────────────────────────────────────────────────

build_dir    := "build"
libwb_dir    := "extern/libwb"
profiles_dir := build_dir / "profiles"

# ── Default: list available recipes ───────────────────────────────────────────

[private]
default:
    @just --list

# ── configure ─────────────────────────────────────────────────────────────────
# Initialises submodules, builds libwb, then installs Python deps via Poetry.

configure: _init-submodules libwb
    poetry install

[private]
[unix]
_init-submodules:
    git submodule update --init {{libwb_dir}}
    git -C {{libwb_dir}} checkout master
    git submodule update --init extern/ECE408_SP25 || echo "⚠ Skipping ECE408_SP25 — SAML SSO required (authorize SSH key at github.com/orgs/illinois-cs-coursework)"

[private]
[windows]
_init-submodules:
    git submodule update --init {{libwb_dir}}
    git -C {{libwb_dir}} checkout master
    git submodule update --init extern/ECE408_SP25; if ($LASTEXITCODE -ne 0) { Write-Host "⚠ Skipping ECE408_SP25 — SAML SSO required (authorize SSH key at github.com/orgs/illinois-cs-coursework)" -ForegroundColor Yellow }

# ── libwb ─────────────────────────────────────────────────────────────────────

[private]
[unix]
libwb:
    @echo "→ Building libwb..."
    cmake -S {{libwb_dir}} -B {{libwb_dir}}/build -DCMAKE_BUILD_TYPE=Release -Wno-dev > /dev/null 2>&1
    cmake --build {{libwb_dir}}/build --target wb
    @echo "→ libwb built."

[private]
[windows]
libwb:
    Write-Host "→ Building libwb..." -ForegroundColor Cyan
    cmake -S {{libwb_dir}} -B {{libwb_dir}}/build -DCMAKE_BUILD_TYPE=Release -Wno-dev
    cmake --build {{libwb_dir}}/build --target wb
    Write-Host "→ libwb built." -ForegroundColor Green

# ── build ─────────────────────────────────────────────────────────────────────
# Build and run one target or all targets.
# Usage: just build             — build + run all
#        just build foo.cu      — build + run foo only

[unix]
build file="":
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p {{build_dir}}
    cmake -S . -B {{build_dir}} -DCMAKE_BUILD_TYPE=Release -Wno-dev > /dev/null 2>&1
    if [ -n "{{file}}" ]; then
        target=$(basename "{{file}}" .cu)
        echo "Building $target..."
        cmake --build {{build_dir}} --target "$target"
        echo ""
        echo "=== $target output ==="
        {{build_dir}}/$target 2>&1 | tee {{build_dir}}/${target}_output.txt
    else
        echo "Building all CUDA targets..."
        cmake --build {{build_dir}}
        for src in src/*/*.cu; do
            name=$(basename "$src" .cu)
            echo "=== $name output ==="
            {{build_dir}}/$name 2>&1 | tee {{build_dir}}/${name}_output.txt
            echo ""
        done
    fi

[windows]
build file="":
    #!/usr/bin/env pwsh
    $target = if ("{{file}}" -ne "") { [System.IO.Path]::GetFileNameWithoutExtension("{{file}}") } else { "" }
    New-Item -ItemType Directory -Force -Path "{{build_dir}}" | Out-Null
    cmake -S . -B {{build_dir}} -DCMAKE_BUILD_TYPE=Release -Wno-dev
    if ($target) {
        Write-Host "Building $target..." -ForegroundColor Cyan
        cmake --build {{build_dir}} --target $target
        Write-Host "`n=== $target output ===" -ForegroundColor Yellow
        & "{{build_dir}}/$target.exe" | Tee-Object -FilePath "{{build_dir}}/${target}_output.txt"
    } else {
        Write-Host "Building all CUDA targets..." -ForegroundColor Cyan
        cmake --build {{build_dir}}
        Get-ChildItem src -Directory | ForEach-Object {
            $name = $_.Name
            $bin  = "{{build_dir}}/$name.exe"
            if (Test-Path $bin) {
                Write-Host "`n=== $name output ===" -ForegroundColor Yellow
                & $bin | Tee-Object -FilePath "{{build_dir}}/${name}_output.txt"
            }
        }
    }

# ── debug ─────────────────────────────────────────────────────────────────────
# Build with -g -G then launch the debugger.
# On Linux/WSL: starts an interactive cuda-gdb session.
# On Windows:   builds the binary; press F5 in VS Code to start Nsight.
# Usage: just debug foo.cu

# Private build-only recipe — called by VS Code preLaunchTask so that
# F5 builds the debug binary without also spawning a cuda-gdb process.
[private]
[unix]
_build-debug file:
    cmake -S . -B {{build_dir}}/debug -DCMAKE_BUILD_TYPE=Debug -Wno-dev > /dev/null 2>&1
    cmake --build {{build_dir}}/debug --target {{file_stem(file)}}

[private]
[windows]
_build-debug file:
    cmake -S . -B {{build_dir}}/debug -DCMAKE_BUILD_TYPE=Debug -Wno-dev
    cmake --build {{build_dir}}/debug --target {{file_stem(file)}}

[unix]
debug file: (_build-debug file)
    cuda-gdb -q {{build_dir}}/debug/{{file_stem(file)}}

[windows]
debug file: (_build-debug file)
    Write-Host ""
    Write-Host "  Binary : {{build_dir}}\debug\{{file_stem(file)}}.exe" -ForegroundColor Green
    Write-Host "  Action : open {{file}} in VS Code then press F5" -ForegroundColor Cyan

# ── profile ───────────────────────────────────────────────────────────────────
# Profile with Nsight Systems. Saves .nsys-rep to build/profiles/.
# Usage: just profile foo.cu   (requires: just build foo.cu first)

[unix]
profile file:
    #!/usr/bin/env bash
    set -euo pipefail
    target=$(basename "{{file}}" .cu)
    bin="{{build_dir}}/$target"
    [ -f "$bin" ] || { echo "Binary not found — run: just build {{file}}"; exit 1; }
    mkdir -p "{{profiles_dir}}"
    echo "Profiling $target with Nsight Systems..."
    nsys profile \
        --output="{{profiles_dir}}/${target}_nsys" \
        --stats=true \
        --force-overwrite=true \
        "$bin"
    echo "Report saved: {{profiles_dir}}/${target}_nsys.nsys-rep"

[windows]
profile file:
    #!/usr/bin/env pwsh
    $target = [System.IO.Path]::GetFileNameWithoutExtension("{{file}}")
    $bin    = "{{build_dir}}/$target.exe"
    if (-not (Test-Path $bin)) { throw "Binary not found — run: just build {{file}}" }
    New-Item -ItemType Directory -Force -Path "{{profiles_dir}}" | Out-Null
    Write-Host "Profiling $target with Nsight Systems..." -ForegroundColor Cyan
    nsys profile `
        --output="{{profiles_dir}}/${target}_nsys" `
        --stats=true `
        --force-overwrite=true `
        $bin
    Write-Host "Report saved: {{profiles_dir}}/${target}_nsys.nsys-rep" -ForegroundColor Green

# ── metrics ──────────────────────────────────────────────────────────────────
# Kernel-level analysis with Nsight Compute. Saves .ncu-rep to build/profiles/.
# Usage: just metrics foo.cu   (requires: just build foo.cu first)

[unix]
metrics file:
    #!/usr/bin/env bash
    set -euo pipefail
    target=$(basename "{{file}}" .cu)
    bin="{{build_dir}}/$target"
    [ -f "$bin" ] || { echo "Binary not found — run: just build {{file}}"; exit 1; }
    mkdir -p "{{profiles_dir}}"
    echo "Analyzing $target with Nsight Compute..."
    ncu \
        --set full \
        --export "{{profiles_dir}}/${target}_ncu" \
        --force-overwrite \
        "$bin"
    echo "Report saved: {{profiles_dir}}/${target}_ncu.ncu-rep"

[windows]
metrics file:
    #!/usr/bin/env pwsh
    $target = [System.IO.Path]::GetFileNameWithoutExtension("{{file}}")
    $bin    = "{{build_dir}}/$target.exe"
    if (-not (Test-Path $bin)) { throw "Binary not found — run: just build {{file}}" }
    New-Item -ItemType Directory -Force -Path "{{profiles_dir}}" | Out-Null
    Write-Host "Analyzing $target with Nsight Compute..." -ForegroundColor Cyan
    ncu `
        --set full `
        "--export={{profiles_dir}}/${target}_ncu" `
        --force-overwrite `
        $bin
    Write-Host "Report saved: {{profiles_dir}}/${target}_ncu.ncu-rep" -ForegroundColor Green



# ── clean ─────────────────────────────────────────────────────────────────────

[unix]
clean:
    rm -rf {{build_dir}}
    find . -name "__pycache__" -type d -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -not -path "./.git/*" -delete 2>/dev/null || true
    @echo "Cleaned."

[windows]
clean:
    if (Test-Path "{{build_dir}}") { Remove-Item "{{build_dir}}" -Recurse -Force }
    Get-ChildItem -Recurse -Filter '__pycache__' | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Get-ChildItem -Recurse -Filter '*.pyc'       | Remove-Item        -Force -ErrorAction SilentlyContinue
    Write-Host "Cleaned." -ForegroundColor Green

# ── clean-all ─────────────────────────────────────────────────────────────────

[unix]
clean-all: clean
    poetry env remove --all 2>/dev/null || true
    echo "Removed Poetry environment."

[windows]
clean-all: clean
    if (Get-Command poetry -ErrorAction SilentlyContinue) { poetry env remove --all }
    Write-Host "Removed Poetry environment." -ForegroundColor Green
