"""
Utilidades internas del módulo model.
"""
try:
    from .. import config
except ImportError:
    config = None


def _get_param(name, default):
    if config is not None and hasattr(config, name):
        return getattr(config, name)
    return default
