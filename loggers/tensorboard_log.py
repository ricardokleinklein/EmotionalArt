""" Tensorboard log

Logging based on Tensorboard panel.

"""
import argparse
import sys
from pathlib import Path
from datetime import datetime
from functools import wraps
from torch.utils.tensorboard.writer import SummaryWriter

from typing import Optional


def timing(func):
    @wraps(func)
    def timing_wrapper(*args, **kwargs):
        now = datetime.now().strftime("%H:%M:%S")
        result = func(*args, **kwargs)
        return f'{now} -- {result}'
    return timing_wrapper


class Logger(SummaryWriter):

    def __init__(self, log_dir: Optional[str] = None,
                 args: argparse.Namespace = None) -> None:
        """ Logging tools based on pytorch's tensorboard, but incorporating
        additional methods to include loggers on other aspects such as
        experiment hyperparameters.

        Args:
            log_dir: Save directory location. Default is
                runs/CURRENT_DATETIME_HOSTNAME
        """
        super(Logger, self).__init__(log_dir=log_dir)
        self.stdout_file = Path(self.log_dir) / "stdout.log"

        with open(self.stdout_file, 'w') as fout:
            fout.write(f"{self._cmd_line()}\nArgs Namespace:\n{args}")

    @timing
    def _cmd_line(self) -> str:
        return f"Executed:{' '.join(sys.argv)}"
