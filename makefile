# Makefile for evaluation_pipeline
# - creates a local venv
# - clones nflows_reflectorch
# - (optionally) pulls Git LFS models
# - installs nflows_reflectorch editable into the venv
# - installs PyTorch (pinned legacy cu118 by default) + remaining deps

SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

VENV        := .venv
PY          := $(VENV)/bin/python
PIP         := $(VENV)/bin/pip

VENDOR_DIR  := vendor
FW_DIR      := $(VENDOR_DIR)/nflows_reflectorch

# Use HTTPS by default:
FW_REPO_URL ?= https://github.com/m-levytskyi/reflectorch-nflows
# Pin to a tag/commit for reproducibility:
FW_REF      ?= dev_ml

REQ_FILE            := requirements.txt

TORCH_WHEEL ?= cu118   # cu118 | cu121 | cpu
TORCH_REQ_cu118 := requirements.torch-cu118.txt
TORCH_REQ_cu121 := requirements.torch-cu121.txt

.PHONY: help setup venv framework install-framework deps torch lfs check-tools check-torch clean distclean

help:
	@echo "Targets:"
	@echo "  make setup              Create venv, clone framework, pull LFS, install framework editable, install deps"
	@echo "  make venv               Create/update venv"
	@echo "  make framework          Clone/update framework repo at FW_REF into $(FW_DIR)"
	@echo "  make lfs                Pull Git LFS objects in framework repo (if used)"
	@echo "  make install-framework  Install framework editable into venv"
	@echo "  make torch              Install torch according to TORCH_WHEEL ($(TORCH_WHEEL))"
	@echo "  make deps               Install torch + other dependencies"
	@echo "  make check-torch        Print torch/cuda status"
	@echo "  make clean              Remove venv"
	@echo "  make distclean          Remove venv + vendor/"
	@echo ""
	@echo "Vars:"
	@echo "  TORCH_WHEEL=cu118|cu121|cpu"
	@echo "  FW_REF=<branch|tag|commit>"
	@echo "  FW_REPO_URL=<url>"

setup: check-tools venv framework lfs install-framework deps

check-tools:
	@command -v git >/dev/null
	@command -v python3 >/dev/null
	@echo "OK: git + python3 found"

venv:
	@test -d "$(VENV)" || python3 -m venv "$(VENV)"
	$(PIP) install -U pip wheel setuptools

framework:
	mkdir -p "$(VENDOR_DIR)"
	if [ ! -d "$(FW_DIR)/.git" ]; then \
		echo "Cloning framework: $(FW_REPO_URL) -> $(FW_DIR)"; \
		git clone "$(FW_REPO_URL)" "$(FW_DIR)"; \
	fi
	cd "$(FW_DIR)"
	git fetch --all --tags --prune
	git checkout "$(FW_REF)"
	if git show-ref --verify --quiet "refs/remotes/origin/$(FW_REF)"; then \
		git reset --hard "origin/$(FW_REF)"; \
	fi

lfs:
	@if command -v git-lfs >/dev/null 2>&1; then \
		if [ -d "$(FW_DIR)/.git" ]; then \
			echo "Pulling Git LFS objects in $(FW_DIR)"; \
			cd "$(FW_DIR)" && git lfs pull; \
		fi; \
	else \
		echo "Note: git-lfs not found; skipping 'git lfs pull'."; \
		echo "      If models are stored via LFS, install git-lfs and run: make lfs"; \
	fi

install-framework: venv framework
	$(PIP) install -e "$(FW_DIR)"

torch: venv
ifeq ($(TORCH_WHEEL),cu118)
	$(PIP) install -r $(TORCH_REQ_cu118)
else ifeq ($(TORCH_WHEEL),cu121)
	$(PIP) install -r $(TORCH_REQ_cu121)
else ifeq ($(TORCH_WHEEL),cpu)
	$(PIP) install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
	@echo "Unknown TORCH_WHEEL=$(TORCH_WHEEL) (use cu118|cu121|cpu)"; exit 1
endif

deps: venv torch
	@if [ -f "$(REQ_FILE)" ]; then \
		$(PIP) install -r "$(REQ_FILE)"; \
	else \
		echo "No $(REQ_FILE) found; skipping."; \
	fi

check-torch: venv
	$(PY) -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

clean:
	rm -rf "$(VENV)"

distclean: clean
	rm -rf "$(VENDOR_DIR)"
