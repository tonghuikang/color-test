# Development Guidelines

## Core Development Rules

- Package Management
   - ONLY use uv, NEVER pip
   - Installation: `uv add package`
   - Running tools: `uv run tool`
   - Upgrading: `uv add --dev package --upgrade-package package`
   - FORBIDDEN: `uv pip install`, `@latest` syntax

- Code Quality
   - Type hints required for all code
   - Public APIs must have docstrings
   - Functions must be focused and small
   - Follow existing patterns exactly
   - Line length: 88 chars maximum


## Code Formatting

- Ruff
   - Format: `uv run --frozen ruff format .`
   - Check: `uv run --frozen ruff check .`
   - Fix: `uv run --frozen ruff check . --fix`
   - Critical issues:
     - Line length (88 chars)
     - Import sorting (I001)
     - Unused imports
   - Line wrapping:
     - Strings: use parentheses
     - Function calls: multi-line with proper indent
     - Imports: split into multiple lines

- Type Checking
   - Tool: `uv run --frozen pyright`
   - Requirements:
     - Explicit None checks for Optional
     - Type narrowing for strings
     - Version warnings can be ignored if checks pass


## Testing

- Modifications to Python files should be tested with `uv run python3 {file.py}` before reporting success


## Tools

To search code, use the file tool.
