lint:
	pre-commit run --all-files
	mypy --scripts-are-modules --ignore-missing-imports .

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
	tox -e "black,flake8,docstrings,py37"

test-full:
	tox

test-short:
	tox -e "py37"

format:
	tox -e format

release: dist ## package and upload release
	pip install -qq twine
	twine upload dist/*

release-test: dist ## package and upload test release
	pip install -qq twine
	twine upload --repository testpypi dist/*

bumpversion-to-dev:
	pip install -qq bumpversion==0.5.3
	python ./tools/bumpversion-tool.py --to-dev

bumpversion-from-dev:
	pip install -qq bumpversion==0.5.3
	python ./tools/bumpversion-tool.py --from-dev
