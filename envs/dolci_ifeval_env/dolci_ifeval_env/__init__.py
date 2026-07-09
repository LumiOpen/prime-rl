def load_environment(*args, **kwargs):
    from .env import load_environment as _load
    return _load(*args, **kwargs)


def load_environment_fi(*args, **kwargs):
    from .env import load_environment_fi as _load
    return _load(*args, **kwargs)


__all__ = ["load_environment", "load_environment_fi"]
