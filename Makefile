# Makefile for evaluation_pipeline (macOS/Linux)
# - delegates setup logic to bootstrap.py
# - keeps the existing target names for workshop-friendly troubleshooting
# - uses CPU/default PyTorch by default, with optional auto/CUDA wheel overrides

ifeq ($(OS),Windows_NT)
$(error Windows is not supported by this Makefile. Use: python bootstrap_windows.py)
endif

SHELL := /bin/sh

BOOTSTRAP_PY := $(strip $(shell command -v python3 2>/dev/null))
VENV_PY := .venv/bin/python
TORCH_WHEEL ?= cpu
CONFIG ?=

.PHONY: help setup venv framework install-framework deps dev-deps torch lfs check-tools check-lfs check-torch install-hooks test lint type-check pre-commit require-venv train sweep clean distclean

define RUN_BOOTSTRAP
	@if [ -z "$(BOOTSTRAP_PY)" ]; then \
		echo "Missing required tool: python3 (Python 3.10-3.12)"; \
		echo "Install Python 3, reopen the terminal, and retry."; \
		exit 1; \
	fi
	@if ! "$(BOOTSTRAP_PY)" -c 'import sys; raise SystemExit(0 if ((3, 10) <= sys.version_info[:2] <= (3, 12)) else 1)' >/dev/null 2>&1; then \
		echo "Unsupported Python interpreter: $(BOOTSTRAP_PY)"; \
		echo "Require Python 3.10-3.12. Install a supported python3, reopen the terminal, and retry."; \
		exit 1; \
	fi
	@"$(BOOTSTRAP_PY)" bootstrap.py $(1)
endef

help:
	@echo "Main targets:"
	@echo "  make setup              Full workshop setup: venv, framework, LFS, installs, torch check"
	@echo "  make check-torch        Print torch backend/device status after setup"
	@echo "  make install-hooks      Install pre-commit and pre-push hooks into .git/hooks"
	@echo "  make test               Run the unit test suite"
	@echo "  make lint               Run low-churn Ruff correctness checks"
	@echo "  make type-check         Run the manual ty check"
	@echo "  make pre-commit         Run pre-commit hooks on staged files"
	@echo ""
	@echo "Root workflow targets:"
	@echo "  make train CONFIG=nf_config_mixed.yaml"
	@echo "                            Train a reflectorch config from the repo root via .venv"
	@echo "  make sweep CONFIG=sweep_configs/baseline.yaml"
	@echo "                            Run a batch sweep from the repo root via .venv"
	@echo ""
	@echo "Troubleshooting / partial reruns:"
	@echo "  make check-tools        Verify Python, Git, Git LFS, and installer availability"
	@echo "  make check-lfs          Verify Git LFS is installed"
	@echo "  make venv               Create or repair the local .venv and upgrade pip/setuptools/wheel"
	@echo "  make framework          Clone or update vendor/nflows_reflectorch"
	@echo "  make lfs                Pull Git LFS assets for the framework checkout"
	@echo "  make install-framework  Install vendor/nflows_reflectorch editable into .venv"
	@echo "  make torch              Install PyTorch using TORCH_WHEEL ($(TORCH_WHEEL))"
	@echo "  make deps               Install requirements.txt into .venv"
	@echo "  make dev-deps           Install requirements-dev.txt into .venv"
	@echo "  make clean              Remove .venv"
	@echo "  make distclean          Remove .venv and vendor/"
	@echo ""
	@echo "Torch wheel modes:"
	@echo "  TORCH_WHEEL=cpu|auto|cu118|cu121|cu126|cu128 (default: $(TORCH_WHEEL))"
	@echo "  auto selects the newest pinned CUDA backend supported by your NVIDIA driver,"
	@echo "  or falls back to the default CPU/macOS wheel."

setup:
	$(call RUN_BOOTSTRAP,setup --torch-wheel $(TORCH_WHEEL))

check-tools:
	$(call RUN_BOOTSTRAP,check-tools)

check-lfs:
	$(call RUN_BOOTSTRAP,check-lfs)

venv:
	$(call RUN_BOOTSTRAP,venv)

framework:
	$(call RUN_BOOTSTRAP,framework)

lfs: framework check-lfs
	$(call RUN_BOOTSTRAP,lfs)

install-framework: venv framework
	$(call RUN_BOOTSTRAP,install-framework)

torch: venv
	$(call RUN_BOOTSTRAP,torch --torch-wheel $(TORCH_WHEEL))

deps: venv torch
	$(call RUN_BOOTSTRAP,deps)

dev-deps: venv
	$(call RUN_BOOTSTRAP,dev-deps)

check-torch: require-venv
	$(call RUN_BOOTSTRAP,check-torch --torch-wheel $(TORCH_WHEEL))

install-hooks: dev-deps
	$(call RUN_BOOTSTRAP,install-hooks)

test: deps dev-deps
	$(call RUN_BOOTSTRAP,test)

lint: dev-deps
	$(call RUN_BOOTSTRAP,lint)

type-check: deps dev-deps
	$(call RUN_BOOTSTRAP,type-check)

pre-commit: dev-deps
	$(call RUN_BOOTSTRAP,pre-commit)

require-venv:
	@if [ ! -x "$(VENV_PY)" ]; then \
		echo "Missing root virtual environment: $(VENV_PY)"; \
		echo "Run 'make setup' first, then retry from the inverse-eval repo root."; \
		exit 1; \
	fi

train: require-venv
	@if [ -z "$(CONFIG)" ]; then \
		echo "Missing CONFIG. Example: make train CONFIG=nf_config_mixed.yaml"; \
		exit 1; \
	fi
	@"$(VENV_PY)" -m reflectorch.train "$(CONFIG)"

sweep: require-venv
	@if [ -z "$(CONFIG)" ]; then \
		echo "Missing CONFIG. Example: make sweep CONFIG=sweep_configs/baseline.yaml"; \
		exit 1; \
	fi
	@"$(VENV_PY)" batch_sweep_runner.py --config "$(CONFIG)"

clean:
	$(call RUN_BOOTSTRAP,clean)

distclean:
	$(call RUN_BOOTSTRAP,distclean)
