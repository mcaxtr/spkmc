.PHONY: help install install-dev lint format test test-cov build clean version pre-commit-install release-patch release-minor release-major

# Default target
help:
	@echo "SPKMC Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install package"
	@echo "  make install-dev      Install with dev dependencies"
	@echo "  make pre-commit-install  Install pre-commit hooks"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             Run all linters (black, isort, flake8, mypy)"
	@echo "  make format           Auto-format code with black and isort"
	@echo ""
	@echo "Testing:"
	@echo "  make test             Run tests"
	@echo "  make test-cov         Run tests with coverage report"
	@echo ""
	@echo "Build:"
	@echo "  make build            Build source and wheel distributions"
	@echo "  make clean            Remove build artifacts"
	@echo "  make version          Show current version"
	@echo ""
	@echo "Release (requires GitHub CLI 'gh'):"
	@echo "  make release-patch    Trigger patch release (1.0.0 -> 1.0.1)"
	@echo "  make release-minor    Trigger minor release (1.0.0 -> 1.1.0)"
	@echo "  make release-major    Trigger major release (1.0.0 -> 2.0.0)"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pip install pre-commit

# Pre-commit hooks
pre-commit-install:
	pre-commit install

# Linting
lint:
	@echo "Running black..."
	black --check --diff spkmc tests
	@echo "Running isort..."
	isort --check-only --diff spkmc tests
	@echo "Running flake8..."
	flake8 spkmc tests --max-line-length=100 --extend-ignore=E203,W503
	@echo "Running mypy..."
	mypy spkmc --ignore-missing-imports || true

# Formatting
format:
	@echo "Formatting with black..."
	black spkmc tests
	@echo "Sorting imports with isort..."
	isort spkmc tests

# Testing
test:
	pytest -v

test-cov:
	pytest --cov=spkmc --cov-report=term-missing --cov-report=html -v
	@echo "Coverage report generated in htmlcov/"

# Build
build: clean
	python -m build

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf spkmc/*.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Version
version:
	@python -c "import spkmc; print(spkmc.__version__)"

# Release targets (require GitHub CLI)
release-patch:
	@echo "Triggering patch release..."
	gh workflow run release.yml -f version_bump=patch

release-minor:
	@echo "Triggering minor release..."
	gh workflow run release.yml -f version_bump=minor

release-major:
	@echo "Triggering major release..."
	gh workflow run release.yml -f version_bump=major
