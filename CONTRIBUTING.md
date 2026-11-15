# Contributing Guidelines

Please follow the guidelines below when contributing to this project to ensure consistency, readability, and maintainability.

## 1. Language

All code must be written in English, including variable names, function names, class names, comments, docstrings, commits, and file names. Do not mix languages.

## 2. Naming Conventions

- Use `snake_case` for variables, functions, and file names.
- Use complete and descriptive names, avoiding abbreviations.
- Use `PascalCase` for class names.

## 3. Code Formatting

- Use **Black** as the automatic code formatter.
- Set the maximum line length to **88 characters**.

## 4. Docstrings

- All functions and .py files must have docstrings, even the smallest/simplest ones.
- Use the **Google style**.
- Always use triple double quotes (`"""`).

## 5. Comments

- Keep comments (`#`) short and use them only when relevant.
- Start comments with an uppercase letter.
- Use proper spelling and punctuation conventions.
- Do not end comments with a period.

## 6. Git Workflow

- Direct commits to the `main` branch are not allowed.
- Each contributor must work on their own branch.
- Follow the branch naming convention: `type/description-with-hyphens`.
- Submit changes through Pull Requests (PRs).
- Each PR must be briefly reviewed by another contributor before merging.
- Use **Merge commits** to integrate changes into `main`.

## 7. Commit Messages

- Keep commit messages short.
- Write commit messages in English, using proper accents.
- Do not end commit messages with a period.
- Use standardized prefixes, such as:
  - `feat:` — new features
  - `fix:` — bug fixes
  - `docs:` — documentation changes
  - `refactor:` — code restructuring
  - `chore:` — maintenance tasks

## 8. Pull Requests

- Use a short, readable title in English, starting with a verb 
  (e.g. "Add exploratory data analysis"), with no trailing period.
- Edit the automatic title generated from the branch name.
- Always fill in the description with a brief summary of the changes.
- One PR per task, small enough to be reviewed quickly.
- Run `pre-commit run --all-files` before opening.
- Every PR needs at least one approval from another contributor.
- Never merge your own PR without review.
- Delete the branch after merging.

## 9. Development Environment

- Use a Python virtual environment (`venv`).
- Manage dependencies through `requirements.txt`, with fixed versions.
- All contributors must use the same Python version.

## 10. Reproducibility

Use `random_state=42` in all operations involving randomness whenever the parameter is available.

## 11. File Paths

- Always use relative paths with `pathlib`.
- Never use absolute paths.
