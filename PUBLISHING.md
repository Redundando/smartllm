# Publishing to PyPI

## Prerequisites

```bash
pip install build twine
```

## Build the Package

```bash
python -m build
```

This creates distribution files in `dist/`:
- `smartllm-2.0.0.tar.gz` (source distribution)
- `smartllm-2.0.0-py3-none-any.whl` (wheel distribution)

## Test on TestPyPI (Optional but Recommended)

```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ smartllm
```

## Publish to PyPI

```bash
twine upload dist/*
```

You'll be prompted for your PyPI credentials.

## Install from PyPI

```bash
pip install smartllm
```

## Version Updates

When releasing a new version:

1. Update version in `smartllm/__init__.py`
2. Update version in `setup.py`
3. Update version in `pyproject.toml`
4. Update CHANGELOG in README.md
5. Commit changes
6. Create git tag: `git tag v2.0.0`
7. Push tag: `git push origin v2.0.0`
8. Build and publish as above
