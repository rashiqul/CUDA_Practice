BUILD_DIR = build

.PHONY: configure clean clean-all build

configure:
	@poetry install

# Absorb the .cu filename as a goal: make build <file.cu>
ifeq ($(firstword $(MAKECMDGOALS)),build)
  _BUILD_FILE := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  ifneq ($(_BUILD_FILE),)
    _BUILD_TARGET := $(basename $(_BUILD_FILE))
    $(eval $(_BUILD_FILE):;@true)
  endif
endif

build:
	@mkdir -p $(BUILD_DIR)
	@cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -Wno-dev > /dev/null 2>&1
	@if [ -n "$(_BUILD_TARGET)" ]; then \
		echo "Building $(_BUILD_TARGET)..."; \
		cmake --build $(BUILD_DIR) --target $(_BUILD_TARGET); \
		echo ""; \
		echo "=== $(_BUILD_TARGET) output ==="; \
		$(BUILD_DIR)/$(_BUILD_TARGET) 2>&1 | tee $(BUILD_DIR)/$(_BUILD_TARGET)_output.txt; \
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

clean:
	@rm -rf $(BUILD_DIR)
	@find . -name "__pycache__" -type d -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null; true
	@find . -name "*.pyc" -not -path "./.git/*" -delete 2>/dev/null; true
	@echo "Cleaned."

clean-all: clean
	@poetry env remove --all 2>/dev/null || true
	@echo "Removed Poetry environment."
