# Makefile for evaluation_pipeline (macOS/Linux)
# - delegates setup logic to bootstrap.py
# - keeps the existing target names for workshop-friendly troubleshooting
# - uses CPU/default PyTorch by default, with optional auto/CUDA wheel overrides

ifeq ($(OS),Windows_NT)
$(error Windows is not supported by this Makefile. Use: python bootstrap_windows.py)
endif

SHELL := /bin/sh

BOOTSTRAP_PY := $(strip $(shell command -v python3 2>/dev/null || command -v python 2>/dev/null))
TORCH_WHEEL ?= auto

.PHONY: help setup venv framework install-framework deps torch lfs check-tools check-lfs check-torch clean distclean

define RUN_BOOTSTRAP
	@if [ -z "$(BOOTSTRAP_PY)" ]; then \
		echo "Missing required tool: Python 3.10-3.12"; \
		echo "Install Python, reopen the terminal, and retry."; \
		exit 1; \
	fi
	@"$(BOOTSTRAP_PY)" bootstrap.py $(1)
endef

help:
	@echo "Main targets:"
	@echo "  make setup              Full workshop setup: venv, framework, LFS, installs, torch check"
	@echo "  make check-torch        Print torch backend/device status after setup"
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

check-torch: torch
	$(call RUN_BOOTSTRAP,check-torch --torch-wheel $(TORCH_WHEEL))

clean:
	$(call RUN_BOOTSTRAP,clean)

distclean:
	$(call RUN_BOOTSTRAP,distclean)
