def load_environment(*args, **kwargs):
    from dolci_ifeval_env.env_fi import load_environment as _load
    return _load(*args, **kwargs)


__all__ = ["load_environment"]
