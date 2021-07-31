
## Pre-commit hooks

We use [git][git] [pre-commit][pre-commit] [hooks][hooks] to check that files
going to be committed:

  - contain no trailing spaces
  - are formatted with [black][black]
  - are compatible to [PEP8][PEP8] (checked by [flake8][flake8])
  - end in a newline and only a newline
  - contain sorted `imports` (checked by [isort][isort])

These hooks are disabled by default. Please use the following commands to enable them:

```bash
pip install pre-commit  # run it only once
pre-commit install      # run it only once, it will install all hooks

# modify some files
git add <some files>
git commit              # It runs all hooks automatically.

# If all hooks run successfully, you can write the commit message now. Done!
#
# If any hook failed, your commit was not successful.
# Please read the error messages and make changes accordingly.
# And rerun

git add <some files>
git commit
```

[git]: https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks
[flake8]: https://github.com/PyCQA/flake8
[PEP8]: https://www.python.org/dev/peps/pep-0008/
[black]: https://github.com/psf/black
[hooks]: https://github.com/pre-commit/pre-commit-hooks
[pre-commit]: https://github.com/pre-commit/pre-commit
[isort]: https://github.com/PyCQA/isort
