"""Package-scoped console logging; root logger settings are untouched."""

import logging


class _ConsoleHandler(logging.StreamHandler):
    pass


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure once during startup; repeated calls update the level."""
    logger = logging.getLogger("modelaudit")
    logger.setLevel(level)
    if not any(isinstance(h, _ConsoleHandler) for h in logger.handlers):
        handler = _ConsoleHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        logger.addHandler(handler)
    logger.propagate = False
    return logger
