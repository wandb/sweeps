UV_RUN_DEV := uv run --with-requirements requirements.dev.txt

dist: clean ## builds source and wheel package
	python setup.py sdist bdist_wheel
	ls -l dist

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr test_results/
	rm -f result.xml
	rm -rf prof/

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

test:
	$(MAKE) format
	$(MAKE) test-short

test-full:
	$(UV_RUN_DEV) tox

test-short:
	$(UV_RUN_DEV) tox -e py310

format:
	$(UV_RUN_DEV) pre-commit run --all-files

release: dist ## package and upload release
	uv pip install -qq twine
	twine upload dist/*

release-test: dist ## package and upload test release
	uv pip install -qq twine
	twine upload --repository testpypi dist/*

bumpversion-to-dev:
	uv pip install -qq bump2version
	uv run python ./tools/bumpversion-tool.py --to-dev

bumpversion-from-dev:
	uv pip install -qq bump2version
	uv run python ./tools/bumpversion-tool.py --from-dev
