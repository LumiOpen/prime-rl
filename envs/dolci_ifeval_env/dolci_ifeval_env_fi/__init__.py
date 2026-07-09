def load_environment(*args, **kwargs):
    from dolci_ifeval_env.env import load_environment_fi as _load
    return _load(*args, **kwargs)


__all__ = ["load_environment"]
