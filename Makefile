BUILD_DIR = build

.PHONY: configure clean clean-all build debug profile ncu

configure:
	@poetry install

# Second word on the command line is always the optional <file.cu> argument.
# Works for: make build hello_world.cu / make debug hello_world.cu / etc.
_ARG    := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
_TARGET := $(basename $(_ARG))

ifneq ($(_ARG),)
  $(eval $(_ARG):;@true)
endif

# ── Release build ────────────────────────────────────────────────────────────

build:
	@mkdir -p $(BUILD_DIR)
	@cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -Wno-dev > /dev/null 2>&1
	@if [ -n "$(_TARGET)" ]; then \
		echo "Building $(_TARGET)..."; \
		cmake --build $(BUILD_DIR) --target $(_TARGET); \
		echo ""; \
		echo "=== $(_TARGET) output ==="; \
		$(BUILD_DIR)/$(_TARGET) 2>&1 | tee $(BUILD_DIR)/$(_TARGET)_output.txt; \
	else \
		echo "Building all CUDA targets..."; \
		cmake --build $(BUILD_DIR); \
		echo ""; \
		for src in src/*.cu; do \
			name=$$(basename $$src .cu); \
			echo "=== $$name output ==="; \
			$(BUILD_DIR)/$$name 2>&1 | tee $(BUILD_DIR)/$${name}_output.txt; \
			echo ""; \
		done; \
	fi

# ── Debug build (cuda-gdb / breakpoints) ─────────────────────────────────────
# Binaries land in build/debug/ to stay separate from the release build.
# Use with: make debug hello_world.cu

debug:
	@mkdir -p $(BUILD_DIR)/debug
	@cmake -S . -B $(BUILD_DIR)/debug -DCMAKE_BUILD_TYPE=Debug -Wno-dev > /dev/null 2>&1
	@if [ -n "$(_TARGET)" ]; then \
		echo "Building $(_TARGET) (debug)..."; \
		cmake --build $(BUILD_DIR)/debug --target $(_TARGET); \
		echo "Debug binary: $(BUILD_DIR)/debug/$(_TARGET)"; \
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
	@[ -f "$(BUILD_DIR)/$(_TARGET)" ] || (echo "Binary not found — run: make build $(_ARG)"; exit 1)
	@mkdir -p $(PROFILES_DIR)
	@echo "Profiling $(_TARGET) with Nsight Systems..."
	@nsys profile \
		--output=$(PROFILES_DIR)/$(_TARGET)_nsys \
		--stats=true \
		--force-overwrite=true \
		$(BUILD_DIR)/$(_TARGET)
	@echo "Report saved: $(PROFILES_DIR)/$(_TARGET)_nsys.nsys-rep"

# ── Nsight Compute — kernel-level metrics ────────────────────────────────────
# Usage: make ncu hello_world.cu
# Output: build/profiles/<target>_ncu.ncu-rep  (open in Nsight Compute GUI)

ncu:
	@[ -n "$(_TARGET)" ] || (echo "Usage: make ncu <file.cu>"; exit 1)
	@[ -f "$(BUILD_DIR)/$(_TARGET)" ] || (echo "Binary not found — run: make build $(_ARG)"; exit 1)
	@mkdir -p $(PROFILES_DIR)
	@echo "Analyzing $(_TARGET) with Nsight Compute..."
	@ncu \
		--set full \
		--export $(PROFILES_DIR)/$(_TARGET)_ncu \
		--force-overwrite \
		$(BUILD_DIR)/$(_TARGET)
	@echo "Report saved: $(PROFILES_DIR)/$(_TARGET)_ncu.ncu-rep"

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	@rm -rf $(BUILD_DIR)
	@find . -name "__pycache__" -type d -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null; true
	@find . -name "*.pyc" -not -path "./.git/*" -delete 2>/dev/null; true
	@echo "Cleaned."

clean-all: clean
	@poetry env remove --all 2>/dev/null || true
	@echo "Removed Poetry environment."
