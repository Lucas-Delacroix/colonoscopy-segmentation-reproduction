from models.base_model import BaseModel

_REGISTRY: dict[str, type[BaseModel]] = {}


def register_model(name: str):
    """
    Decorator that registers a model class under a given name.

    Usage:
        @register_model("esfpnet")
        class ESFPNet(BaseModel):
            ...
    """
    def decorator(cls: type[BaseModel]) -> type[BaseModel]:
        if name in _REGISTRY:
            raise ValueError(f"Model '{name}' is already registered.")
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_model(name: str, **kwargs) -> BaseModel:
    """
    Instantiates a registered model by name.

    Usage:
        model = get_model("esfpnet", num_classes=1)
    """
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise ValueError(
            f"Model '{name}' not found in registry. "
            f"Available models: {available}"
        )
    return _REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    """Returns all registered model names."""
    return list(_REGISTRY.keys())

# Import built-in models so decorators populate the registry.
from models import esfpnet  # noqa: E402,F401
