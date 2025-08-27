# Contributing to Our Project

Thank you for your interest in contributing to our project! We use Git pre-commit hooks to ensure code quality and consistency. Before contributing, please follow these guidelines to enable and use the pre-commit hooks.

## Pre-Commit Hooks

We have set up pre-commit hooks to check that the files you're committing meet our coding and formatting standards. These checks include:

- Ensuring there are no trailing spaces.
- Formatting code with [black](https://github.com/psf/black).
- Checking compliance with PEP8 using [flake8](https://flake8.pycqa.org/).
- Verifying that files end with a newline character (and only a newline).
- Sorting imports using [isort](https://pycqa.github.io/isort/).

Please note that these hooks are disabled by default. To enable them, follow these steps:

### Installation (Run only once)

1. Install the `pre-commit` package using pip:
   ```bash
   pip install pre-commit
   ```
1. Install the Git hooks using:
   ```bash
   pre-commit install
   ```
### Making a Commit
Once you have enabled the pre-commit hooks, follow these steps when making a commit:
1. Make your changes to the codebase.
2. Stage your changes by using git add for the files you modified.
3. Commit your changes using git commit. The pre-commit hooks will run automatically at this point.
4. If all hooks run successfully, you can write your commit message, and your changes will be successfully committed.
5. If any hook fails, your commit will not be successful. Please read and follow the error messages provided, make the necessary changes, and then re-run git add and git commit.

### Your Contribution
Your contributions are valuable to us, and by following these guidelines, you help maintain code consistency and quality in our project. We appreciate your dedication to ensuring high-quality code. If you have questions or need assistance, feel free to reach out to us. Thank you for being part of our open-source community!

