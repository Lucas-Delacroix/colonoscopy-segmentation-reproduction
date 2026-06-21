from models.base_model import BaseModel

_REGISTRY: dict[str, type[BaseModel]] = {}


def register_model(name: str):
    def decorator(cls: type[BaseModel]) -> type[BaseModel]:
        if name in _REGISTRY:
            raise ValueError(f"Model '{name}' is already registered.")
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_model(name: str, **kwargs) -> BaseModel:
    if name not in _REGISTRY:
        available = ", ".join(list_models()) or "none"
        raise ValueError(
            f"Local model '{name}' is not registered. "
            f"Available local models: {available}. "
            "Upstream models should be run through `make train MODEL=<model>`."
        )
    return _REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    return list(_REGISTRY.keys())
