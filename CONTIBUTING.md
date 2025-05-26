# Contributing to `deepmirror`

First off, thanks for taking the time to contribute! 🎉

This project aims to provide a lightweight CLI for interacting with deepmirror’s API. Whether you’re fixing bugs, adding new features, improving documentation, or helping with testing—we appreciate your help.

## 🛠 Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone https://github.com/deepmirror-ai/deepmirror-client.git
   cd deepmirror-client
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   bash setup.sh
   ```

3. **Install dependencies:**

   ```bash
   pip install -e .[dev]
   ```

4. **Run tests:**

   ```bash
   pytest
   ```

## 💡 Ways to Contribute

- 🚨 **Bug reports** — Use [GitHub Issues](https://github.com/deepmirror-ai/deepmirror-client/issues)
- 🧪 **Test coverage** — Add unit tests for uncovered code paths
- 🧩 **Feature requests** — Use issues or open a discussion
- 📚 **Docs improvements** — Clarify usage or improve examples
- 🧹 **Code cleanup** — Remove dead code, improve formatting, type hints

## 🧾 Coding Guidelines

- Use pre-commit hooks for code formatting and linting (includes Ruff, Prettier, Pylint).
- Follow existing patterns (click CLI commands, consistent error handling).
- Make pull requests small and focused.

## 🧪 Testing

- We use `pytest`.
- Place new tests under the `tests/` directory.
- Run tests locally before submitting a PR.

## 🤝 Pull Requests

- Fork the repo and create your branch: `feature/my-new-feature`
- Open a PR against `main` with a clear description
- Ensure all tests pass before review

## 🙏 Acknowledgements

Thanks for helping us make deepmirror better for the community!
