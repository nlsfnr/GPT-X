PY_VERSION=3.12.7
VENV=.venv
PY=uv run -- python
PIP=uv pip
ECHO=>&2 echo
SRC_FILES=./gptx.py
LOCAL_DEPS=.requirements.freeze.txt

all: lint typecheck


${LOCAL_DEPS}: requirements.txt
	@${ECHO} "Installing dependencies..."
	@python3 -m pip install --upgrade --user uv
	@uv python install ${PY_VERSION}
	@uv venv --python ${PY_VERSION} ${VENV}
	@${PIP} install \
		--requirement requirements.txt \
		pip  # To allow mypy to install type stubs
	@${PIP} freeze > ${LOCAL_DEPS}


.PHONY: lint
lint: ${LOCAL_DEPS}
	@${ECHO} "Linting..."
	@${PY} -m isort \
		--add-import "from __future__ import annotations" \
		--append-only *.py \
		-- ${SRC_FILES}
	@${PY} -m flake8 \
		--ignore E501,W503,E203,E704 \
		-- ${SRC_FILES}


.PHONY: typecheck
typecheck: ${LOCAL_DEPS}
	@${ECHO} "Running type checker..."
	@${PY} -m mypy \
		--install-types \
		--non-interactive \
		-- ${SRC_FILES}


.PHONY: clean
clean:
	@${ECHO} "Cleaning up..."
	@rm -rf \
        ${VENV} \
        ${LOCAL_DEPS} \
        .mypy_cache
