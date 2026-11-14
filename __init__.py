"""Public provider entrypoint; discovery does not install or start Rapid-MLX."""
from providers import register_provider

from .provider import RapidMlxProfile

register_provider(RapidMlxProfile())
