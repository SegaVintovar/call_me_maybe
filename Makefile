install:
	uv sync --all-groups
run:
	uv run python -m src
debug:
# 	pdb
clean:
	find . -path ./.venv -prune -o -type d -name __pycache__ -exec rm -rf {} +
	find . -path ./.venv -prune -o -type d -name .mypy_cache -exec rm -rf {} +
lint:
	flake8 . --exclude=.venv,llm_sdk
	mypy . --warn-return-any\
		--warn-unused-ignores\
		--ignore-missing-imports\
		--disallow-untyped-defs\
		--check-untyped-defs\
		--explicit-package-bases \
		--exclude '^(venv|\.venv|env|llm_sdk)/'