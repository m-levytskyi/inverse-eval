# Makefile for evaluation_pipeline
# - creates a local venv
# - clones nflows_reflectorch
# - installs it editable into the venv
# - installs remaining dependencies from requirements.txt (and/or Pipfile via pipenv)

SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

VENV        := .venv
PY          := $(VENV)/bin/python
PIP         := $(VENV)/bin/pip

VENDOR_DIR  := vendor
FW_DIR      := $(VENDOR_DIR)/nflows_reflectorch

# Use HTTPS by default; switch to SSH if your supervisors use SSH keys:
FW_REPO_URL ?= https://gitlab.lrz.de/thesis-levytskyi/nflows_reflectorch.git
# Pin to a tag/commit for reproducibility (recommended):
FW_REF      ?= dev_ml

REQ_FILE    := requirements.txt

.PHONY: help setup venv framework install-framework deps lfs check-tools clean distclean

help:
	@echo "Targets:"
	@echo "  make setup           Create venv, clone framework, install editable, install deps"
	@echo "  make venv            Create/update venv"
	@echo "  make framework       Clone/update framework repo at FW_REF into $(FW_DIR)"
	@echo "  make install-framework  Install framework editable into venv"
	@echo "  make deps            Install other dependencies"
	@echo "  make lfs             Pull Git LFS objects in framework repo (if used)"
	@echo "  make clean           Remove venv"
	@echo "  make distclean       Remove venv + vendor/"

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
	# checkout pinned ref (branch/tag/commit)
	git checkout "$(FW_REF)"
	# ensure working tree matches remote for branches
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
	# Editable install of the local clone
	$(PIP) install -e "$(FW_DIR)"

deps: venv
	@if [ -f "$(REQ_FILE)" ]; then \
		$(PIP) install -r "$(REQ_FILE)"; \
	else \
		echo "No $(REQ_FILE) found; skipping."; \
	fi
	# If you still want to support Pipfile, uncomment:
	# @if [ -f "Pipfile" ]; then \
	#   $(PIP) install pipenv; \
	#   PIPENV_VENV_IN_PROJECT=1 pipenv install --dev; \
	# fi

clean:
	rm -rf "$(VENV)"

distclean: clean
	rm -rf "$(VENDOR_DIR)"
