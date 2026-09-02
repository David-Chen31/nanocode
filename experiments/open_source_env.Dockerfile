ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG PROFILE
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /opt/bootstrap
COPY . /opt/bootstrap

RUN python -m pip install "pip==24.2" \
      "flit_core<4" \
      "editables" \
      "hatchling" \
      "hatch-fancy-pypi-readme>=22.5.0" \
      "setuptools>=61" \
      "setuptools-scm[toml]>=6.2.3" \
      "poetry-core>=1.0.0" && \
    case "$PROFILE" in \
      click) python -m pip install -r requirements/tests.txt && python -m pip install -e . ;; \
      httpx) python -m pip install -r requirements.txt ;; \
      pytest) SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PYTEST=8.4.0.dev0 python -m pip install -e ".[dev]" ;; \
      pydantic) python -m pip install -e . && python -m pip install \
        cloudpickle "coverage[toml]" dirty-equals eval-type-backport \
        "pytest>=8.2.2" pytest-mock pytest-pretty pytest-examples \
        "faker>=18.13.0" "pytest-benchmark>=4.0.0" \
        "pytest-codspeed~=2.2.0" "packaging>=23.2" ;; \
      requests) python -m pip install -r requirements-dev.txt ;; \
      rich) python -m pip install -e . "pytest>=7,<8" pytest-cov ;; \
      *) echo "unknown profile: $PROFILE" >&2; exit 2 ;; \
    esac && \
    python -m pip freeze --all > /opt/pip-freeze.txt

WORKDIR /task
