# Contributing to mnemotree

Thank you for your interest in contributing to mnemotree! We welcome contributions from the community to help improve this project.

## Development Environment Setup

This project uses `uv` for dependency management.

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/kurcontko/mnemotree.git
    cd mnemotree
    ```

2.  **Install dependencies**:
    ```bash
    uv pip install -e ".[dev]"
    ```
    Or if you are just using pip:
    ```bash
    pip install -e ".[dev]"
    ```

## Code Quality

We strictly enforce code quality standards using `ruff` for linting and formatting, and `mypy` for static type checking.

Before submitting a Pull Request, please ensure you run the following commands:

### Formatting and Linting
```bash
ruff check .
ruff format .
```

### Type Checking
```bash
mypy src
```

### Tests
Run the test suite to ensure no regressions:
```bash
pytest
```

## Pull Request Process

1.  Fork the repository and branch off from `main`.
2.  Make your changes, ensuring you add tests for any new functionality or bug fixes.
3.  Run the linters and tests as described above.
4.  Submit a Pull Request targeting the `main` branch.
5.  Provide a clear description of your changes and reference any related issues.

## Reporting Issues

If you find a bug or have a feature request, please verify that it hasn't already been reported. If not, please submit a new issue using the provided templates.
