UV_RUN_DEV := uv run --with-requirements requirements.dev.txt

dist: clean ## builds source and wheel package
	python setup.py sdist bdist_wheel
	ls -l dist

.PHONY: clean
clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

.PHONY: clean-build
clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

.PHONY: clean-test
clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr test_results/
	rm -f result.xml
	rm -rf prof/

.PHONY: clean-pyc
clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

.PHONY: test
test:
	$(MAKE) format
	$(MAKE) test-short

.PHONY: test-full
test-full:
	$(UV_RUN_DEV) tox

.PHONY: test-short
test-short:
	$(UV_RUN_DEV) tox -e py310

.PHONY: format
format:
	$(UV_RUN_DEV) pre-commit run --all-files

.PHONY: release
release: dist ## package and upload release
	uv pip install -qq twine
	twine upload dist/*

.PHONY: release-test
release-test: dist ## package and upload test release
	uv pip install -qq twine
	twine upload --repository testpypi dist/*

.PHONY: bumpversion-to-dev
bumpversion-to-dev:
	uv pip install -qq bumpversion==0.5.3
	uv run python ./tools/bumpversion-tool.py --to-dev

.PHONY: bumpversion-from-dev
bumpversion-from-dev:
	uv pip install -qq bumpversion==0.5.3
	uv run python ./tools/bumpversion-tool.py --from-dev
