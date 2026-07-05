.PHONY: bootstrap lint test-fast coverage harness-check package check

bootstrap:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt
	python -m pip install -e .
	python -m pip install -r requirements-dev.txt

lint:
	python -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	python -m pydocstyle --count --convention=numpy
	python -m black --check -l 99 dphtools tests scripts setup.py versioneer.py

test-fast:
	python -m pytest -q tests

coverage:
	python -m pytest --cov=dphtools --cov-branch --cov-report=term-missing --cov-report=xml tests
	python scripts/agent_harness/coverage_gate.py

harness-check:
	python scripts/agent_harness/validate_harness.py
	python scripts/agent_harness/validate_references.py
	python scripts/agent_harness/validate_pr.py --local

package:
	python setup.py sdist bdist_wheel
	python -m twine check dist/*

check: lint coverage harness-check package
