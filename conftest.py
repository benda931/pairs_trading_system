# -*- coding: utf-8 -*-
"""Root conftest — loaded first by pytest.

CRITICAL: duckdb_engine must be imported before pandas to avoid segfault
on Python 3.13 + DuckDB 1.3.x.  The dialect's has_comment_support()
calls duckdb.connect() which crashes if pandas has already loaded its
own duckdb C extension into the process.
"""
try:
    import duckdb_engine  # noqa: F401
except ImportError:
    pass
