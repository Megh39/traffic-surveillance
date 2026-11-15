# 14/11/26

- download uv

  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

- update path

  ```powershell
  $env:Path = "C:\Users\palla\.local\bin;$env:Path"
  ```

- download dependencies

  ```powerquery
  uv sync
  ```

  - *Error handling*

    - update python version to <3.15 in `pyproject.toml`

    - create a README.md

    - correct `pyproject.toml`

      - to use it just for managing dependencies instead of building an installable Python library.

    - update python version to <3.14 in `pyproject.toml`

- run pipeline

  ```powershell
  python -m scripts.run_pipeline
  ```

  - *Error handling*

    - `from tvs. ...` $\rightarrow$ `from src.tvs. ...`  in multiple files


