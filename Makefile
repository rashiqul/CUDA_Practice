BUILD_DIR  = build
LIBWB_DIR  = extern/libwb

# ── Host OS detection ─────────────────────────────────────────────────────────
ifeq ($(OS),Windows_NT)
  HOST_OS   := Windows
  EXE       := .exe
  RMDIR     := powershell -NoProfile -Command "if (Test-Path '$(BUILD_DIR)') { Remove-Item '$(BUILD_DIR)' -Recurse -Force }" 2>NUL || true
  RMCACHE   := powershell -NoProfile -Command "Get-ChildItem -Recurse -Filter '__pycache__' | Remove-Item -Recurse -Force" 2>NUL || true
  LIBWB_LIB := $(LIBWB_DIR)/build/wb.lib
else
  HOST_OS   := $(shell uname -s)
  EXE       :=
  RMDIR     := rm -rf $(BUILD_DIR)
  RMCACHE   := find . -name "__pycache__" -type d -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null; true
  LIBWB_LIB := $(LIBWB_DIR)/build/libwb.a
endif

.PHONY: configure clean clean-all build debug profile ncu libwb

configure:
	@poetry install
	@echo "→ Initialising submodules..."
	@git submodule update --init extern/ECE408_SP25 $(LIBWB_DIR)
	@git -C $(LIBWB_DIR) checkout master
	@$(MAKE) --no-print-directory libwb

libwb: $(LIBWB_LIB)

$(LIBWB_LIB):
	@echo "→ Building libwb (static, once) on $(HOST_OS)..."
ifeq ($(OS),Windows_NT)
	@cmake -S $(LIBWB_DIR) -B $(LIBWB_DIR)/build -DCMAKE_BUILD_TYPE=Release -Wno-dev
else
	@cmake -S $(LIBWB_DIR) -B $(LIBWB_DIR)/build -DCMAKE_BUILD_TYPE=Release -Wno-dev > /dev/null 2>&1
endif
	@cmake --build $(LIBWB_DIR)/build --target wb
	@echo "→ libwb built: $@"

# Second word on the command line is always the optional <file.cu> argument.
# Works for: make build hello_world.cu / make debug hello_world.cu / etc.
_ARG    := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
_TARGET := $(basename $(_ARG))

ifneq ($(_ARG),)
  $(eval $(_ARG):;@true)
endif

# ── Release build ────────────────────────────────────────────────────────────

build:
ifeq ($(OS),Windows_NT)
	@powershell -NoProfile -Command "New-Item -ItemType Directory -Force -Path '$(BUILD_DIR)' | Out-Null"
	@cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -Wno-dev
else
	@mkdir -p $(BUILD_DIR)
	@cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -Wno-dev > /dev/null 2>&1
endif
	@if [ -n "$(_TARGET)" ]; then \
		echo "Building $(_TARGET)..."; \
		cmake --build $(BUILD_DIR) --target $(_TARGET); \
		echo ""; \
		echo "=== $(_TARGET) output ==="; \
		$(BUILD_DIR)/$(_TARGET)$(EXE) 2>&1 | tee $(BUILD_DIR)/$(_TARGET)_output.txt; \
	else \
		echo "Building all CUDA targets..."; \
		cmake --build $(BUILD_DIR); \
		echo ""; \
		for src in src/*/*.cu; do \
			name=$$(basename $$src .cu); \
			echo "=== $$name output ==="; \
			$(BUILD_DIR)/$$name$(EXE) 2>&1 | tee $(BUILD_DIR)/$${name}_output.txt; \
			echo ""; \
		done; \
	fi

# ── Debug build (cuda-gdb / breakpoints) ─────────────────────────────────────
# Binaries land in build/debug/ to stay separate from the release build.
# Use with: make debug hello_world.cu

debug:
ifeq ($(OS),Windows_NT)
	@powershell -NoProfile -Command "New-Item -ItemType Directory -Force -Path '$(BUILD_DIR)/debug' | Out-Null"
	@cmake -S . -B $(BUILD_DIR)/debug -DCMAKE_BUILD_TYPE=Debug -Wno-dev
else
	@mkdir -p $(BUILD_DIR)/debug
	@cmake -S . -B $(BUILD_DIR)/debug -DCMAKE_BUILD_TYPE=Debug -Wno-dev > /dev/null 2>&1
endif
	@if [ -n "$(_TARGET)" ]; then \
		echo "Building $(_TARGET) (debug)..."; \
		cmake --build $(BUILD_DIR)/debug --target $(_TARGET); \
		echo "Debug binary: $(BUILD_DIR)/debug/$(_TARGET)$(EXE)"; \
	else \
		echo "Building all CUDA targets (debug)..."; \
		cmake --build $(BUILD_DIR)/debug; \
		echo "Debug binaries in: $(BUILD_DIR)/debug/"; \
	fi

PROFILES_DIR = $(BUILD_DIR)/profiles

# ── Nsight Systems — timeline / system-level profile ─────────────────────────
# Usage: make profile hello_world.cu
# Output: build/profiles/<target>_nsys.nsys-rep  (open in Nsight Systems GUI)

profile:
	@[ -n "$(_TARGET)" ] || (echo "Usage: make profile <file.cu>"; exit 1)
	@[ -f "$(BUILD_DIR)/$(_TARGET)$(EXE)" ] || (echo "Binary not found — run: make build $(_ARG)"; exit 1)
ifeq ($(OS),Windows_NT)
	@powershell -NoProfile -Command "New-Item -ItemType Directory -Force -Path '$(PROFILES_DIR)' | Out-Null"
else
	@mkdir -p $(PROFILES_DIR)
endif
	@echo "Profiling $(_TARGET) with Nsight Systems..."
	@nsys profile \
		--output=$(PROFILES_DIR)/$(_TARGET)_nsys \
		--stats=true \
		--force-overwrite=true \
		$(BUILD_DIR)/$(_TARGET)$(EXE)
	@echo "Report saved: $(PROFILES_DIR)/$(_TARGET)_nsys.nsys-rep"

# ── Nsight Compute — kernel-level metrics ────────────────────────────────────
# Usage: make ncu hello_world.cu
# Output: build/profiles/<target>_ncu.ncu-rep  (open in Nsight Compute GUI)

ncu:
	@[ -n "$(_TARGET)" ] || (echo "Usage: make ncu <file.cu>"; exit 1)
	@[ -f "$(BUILD_DIR)/$(_TARGET)$(EXE)" ] || (echo "Binary not found — run: make build $(_ARG)"; exit 1)
ifeq ($(OS),Windows_NT)
	@powershell -NoProfile -Command "New-Item -ItemType Directory -Force -Path '$(PROFILES_DIR)' | Out-Null"
else
	@mkdir -p $(PROFILES_DIR)
endif
	@echo "Analyzing $(_TARGET) with Nsight Compute..."
	@ncu \
		--set full \
		--export $(PROFILES_DIR)/$(_TARGET)_ncu \
		--force-overwrite \
		$(BUILD_DIR)/$(_TARGET)$(EXE)
	@echo "Report saved: $(PROFILES_DIR)/$(_TARGET)_ncu.ncu-rep"

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	@$(RMDIR)
	@$(RMCACHE)
ifeq ($(OS),Windows_NT)
	@powershell -NoProfile -Command "Get-ChildItem -Recurse -Filter '*.pyc' | Remove-Item -Force" 2>NUL || true
else
	@find . -name "*.pyc" -not -path "./.git/*" -delete 2>/dev/null; true
endif
	@echo "Cleaned."

clean-all: clean
	@poetry env remove --all 2>/dev/null || true
	@echo "Removed Poetry environment."
