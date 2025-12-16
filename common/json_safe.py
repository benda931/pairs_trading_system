# -*- coding: utf-8 -*-
from pathlib import Path as _Path
from datetime import date as _date, datetime as _dt

try:
    import numpy as _np
except Exception:
    class _np:  # type: ignore
        integer = ()
        floating = ()

try:
    import pandas as _pd
except Exception:
    class _pd:  # type: ignore
        class Timestamp: pass

def json_default(obj):
    if isinstance(obj, (_Path, _pd.Timestamp, _date, _dt)):
        return str(obj)
    if isinstance(obj, getattr(_np, "integer", ())) or isinstance(obj, getattr(_np, "floating", ())):
        try:
            return obj.item()
        except Exception:
            return float(obj) if hasattr(obj, "__float__") else int(obj)
    return str(obj)

def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, (_Path, _pd.Timestamp, _date, _dt)):
        return str(obj)
    if isinstance(obj, getattr(_np, "integer", ())) or isinstance(obj, getattr(_np, "floating", ())):
        try:
            return obj.item()
        except Exception:
            return float(obj) if hasattr(obj, "__float__") else int(obj)
    return obj
