import logging

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(level: int | str = logging.INFO) -> None:
    """Configure logging once for the whole app. Call from entry points only."""
    logging.basicConfig(level=level, format=LOG_FORMAT)
    # Quiet noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "google"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
