---
inclusion: always
---

# Virtual Environment

Always use the project's virtual environment when running Python commands (tests, scripts, linters, etc.):

```
.venv\Scripts\python.exe
.venv\Scripts\pip.exe
```

For example:
- Run tests: `.venv\Scripts\python.exe -m pytest tests/unit/ -v`
- Run a script: `.venv\Scripts\python.exe some_script.py`
- Install packages: `.venv\Scripts\pip.exe install -r requirements.txt`
