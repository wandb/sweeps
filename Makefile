lint:
	pre-commit run --all-files
	mypy --scripts-are-modules --ignore-missing-imports .
