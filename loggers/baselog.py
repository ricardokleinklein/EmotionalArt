""" Logger

Logging of activities and experiments.
"""
import calendar
import datetime
import logging
import socket
import sys

from pathlib import Path

from typing import Optional, Union


def read_cmd_line() -> str:
    return ' '.join(sys.argv)


def get_hostname() -> str:
    return socket.gethostname()


def log_timestamp() -> str:
    today = datetime.date.today()
    month = today.month
    day = today.day
    timestamp = datetime.datetime.now().strftime("%H_%M")
    return f"{calendar.month_abbr[month]}{day}-{timestamp}"


class Logger(logging.Logger):

    level = logging.NOTSET
    fmt = "%(asctime)s - [Job: %(process)d]: %(message)s"

    def __init__(self, name: str = "info.log",
                 log_dir: Optional[Union[str, Path]] = None,
                 verbose: bool = True) -> None:
        """

        Args:
            name
            log_dir:
        """
        super().__init__(name)

        if not log_dir:
            log_dir = f"runs/{log_timestamp()}-{get_hostname()}"
        if not isinstance(log_dir, Path):
            log_dir = Path(log_dir)
        self.log_dir = log_dir
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / name
        hdlr = [logging.FileHandler(log_file), logging.StreamHandler()]
        logging.basicConfig(level=self.level,
                            format=self.fmt,
                            handlers=hdlr if verbose else [hdlr[0]])
        self.logger = logging.getLogger()
        self.logger.info("Job launched")
        self.logger.info(read_cmd_line())

    def __call__(self, msg: str) -> None:
        """ Log a given piece of information without regards to its importance.

        TODO: Add error / exception handling

        Args:
            msg: Message to be logged.
        """
        self.logger.info(msg=msg)

    def close(self) -> None:
        self.logger.info("Closing logger.")
        handlers = self.logger.handlers[:]
        for handler in handlers:
            self.logger.removeHandler(handler)
            handler.close()
