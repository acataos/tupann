# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import os
import time
import warnings
from typing import Any, List, Optional


def parse_dates_argument(dates_str: str, format: str = "%Y%m%d") -> list:
    pre_dates = dates_str.split(",")
    dates = []
    for pre_date in pre_dates:
        if "-" in pre_date:
            date_range = pre_date.split("-")
            if len(date_range) != 2:
                raise Exception("Error: wrong formatting for dates")
            start = datetime.datetime.strptime(date_range[0], "%Y%m%d").date()
            end = datetime.datetime.strptime(date_range[1], "%Y%m%d").date()
            delta = end - start
            dates_between = [(start + datetime.timedelta(days=i)).strftime(format) for i in range(delta.days + 1)]
            dates = list(set(dates).union(set(dates_between)))

        else:
            try:
                new_pre_date = datetime.datetime.strptime(pre_date, "%Y%m%d").strftime(format)
            except ValueError:
                raise Exception("Error: wrong formatting for dates")

            if pre_date not in dates:
                dates.append(new_pre_date)
    return sorted(dates)


def parse_lags_argument(lags_str: str) -> list:
    pre_lags = lags_str.replace("'", "").replace('"', "").replace("\\", "")
    pre_lags = pre_lags.split(",")
    lags = []
    for pre_lag in pre_lags:
        if ":" in pre_lag:
            lag_range = pre_lag.split(":")
            if len(lag_range) != 2:
                raise Exception("Error: wrong formatting for lags")
            start = int(lag_range[0])
            end = int(lag_range[1])
            lags_between = list(range(start, end + 1))
            lags = list(set(lags).union(set(lags_between)))

        else:
            try:
                lag = int(pre_lag)
            except ValueError:
                raise ValueError("Lags string should be ints separated by ':' and ','.")

            if lag not in lags:
                lags.append(lag)
    return sorted(lags)


def is_strictly_increasing(lst):
    stack = []
    for i in lst:
        if stack and i <= stack[-1]:
            return False
        stack.append(i)
    return True


class _TicToc(object):
    """
    Author: Hector Sanchez
    Date: 2018-07-26
    Description: Class that allows you to do 'tic toc' to your code.
    This class was based on https://github.com/hector-sab/ttictoc, which is
    distributed under the MIT license. It prints time information between
    successive tic() and toc() calls.
    Example:
        from src.utils.general_utils import tic,toc
        tic()
        tic()
        toc()
        toc()
    """

    def __init__(
        self,
        name: str = "",
        method: Any = "time",
        nested: bool = False,
        print_toc: bool = True,
    ) -> None:
        """
        Args:
            name (str): Just informative, not needed
            method (int|str|ftn|clss): Still trying to understand the default
                options. 'time' uses the 'real wold' clock, while the other
                two use the cpu clock. To use your own method,
                do it through this argument
                Valid int values:
                    0: time.time | 1: time.perf_counter | 2: time.proces_time
                    3: time.time_ns | 4: time.perf_counter_ns
                    5: time.proces_time_ns
                Valid str values:
                  'time': time.time | 'perf_counter': time.perf_counter
                  'process_time': time.proces_time | 'time_ns': time.time_ns
                  'perf_counter_ns': time.perf_counter_ns
                  'proces_time_ns': time.proces_time_ns
                Others:
                  Whatever you want to use as time.time
            nested (bool): Allows to do tic toc with nested with a
                single object. If True, you can put several tics using the
                same object, and each toc will correspond to the respective tic.
                If False, it will only register one single tic, and
                return the respective elapsed time of the future tocs.
            print_toc (bool): Indicates if the toc method will print
                the elapsed time or not.
        """
        self.name = name
        self.nested = nested
        self.tstart: Any[List, None] = None
        if self.nested:
            self.set_nested(True)

        self._print_toc = print_toc

        self._int2strl = [
            "time",
            "perf_counter",
            "process_time",
            "time_ns",
            "perf_counter_ns",
            "process_time_ns",
        ]
        self._str2fn = {
            "time": [time.time, "s"],
            "perf_counter": [time.perf_counter, "s"],
            "process_time": [time.process_time, "s"],
            "time_ns": [time.time_ns, "ns"],
            "perf_counter_ns": [time.perf_counter_ns, "ns"],
            "process_time_ns": [time.process_time_ns, "ns"],
        }

        if type(method) is not int and type(method) is not str:
            self._get_time = method

        if type(method) is int and method < len(self._int2strl):
            method = self._int2strl[method]
        elif type(method) is int and method > len(self._int2strl):
            method = "time"

        if type(method) is str and method in self._str2fn:
            self._get_time = self._str2fn[method][0]
            self._measure = self._str2fn[method][1]
        elif type(method) is str and method not in self._str2fn:
            self._get_time = self._str2fn["time"][0]
            self._measure = self._str2fn["time"][1]

    def _print_elapsed(self) -> None:
        """
        Prints the elapsed time
        """
        if self.name != "":
            name = "[{}] ".format(self.name)
        else:
            name = self.name
        print("-{0}elapsed time: {1:.3g} ({2})".format(name, self.elapsed, self._measure))

    def tic(self) -> None:
        """
        Defines the start of the timing.
        """
        if self.nested:
            self.tstart.append(self._get_time())
        else:
            self.tstart = self._get_time()

    def toc(self, print_elapsed: Optional[bool] = None) -> None:
        """
        Defines the end of the timing.
        """
        self.tend = self._get_time()
        if self.nested:
            if len(self.tstart) > 0:
                self.elapsed = self.tend - self.tstart.pop()
            else:
                self.elapsed = None
        else:
            if self.tstart:
                self.elapsed = self.tend - self.tstart
            else:
                self.elapsed = None

        if print_elapsed is None:
            if self._print_toc:
                self._print_elapsed()
        else:
            if print_elapsed:
                self._print_elapsed()

        # return(self.elapsed)

    def set_print_toc(self, set_print: bool) -> None:
        """
        Indicate if you want the timed time printed out or not.
        Args:
          set_print (bool): If True, a message with the elapsed time
            will be printed.
        """
        if type(set_print) is bool:
            self._print_toc = set_print
        else:
            warnings.warn(
                "Parameter 'set_print' not boolean. Ignoring the command.",
                Warning,
            )

    def set_nested(self, nested: bool) -> None:
        """
        Sets the nested functionality.
        """
        # Assert that the input is a boolean
        if type(nested) is bool:
            # Check if the request is actually changing the
            # behaviour of the nested tictoc
            if nested != self.nested:
                self.nested = nested

                if self.nested:
                    self.tstart = []
                else:
                    self.tstart = None
        else:
            warnings.warn(
                "Parameter 'nested' not boolean. Ignoring the command.",
                Warning,
            )


class TicToc(_TicToc):
    def tic(self, nested: bool = True) -> None:
        """
        Defines the start of the timing.
        """
        if nested:
            self.set_nested(True)

        if self.nested:
            self.tstart.append(self._get_time())
        else:
            self.tstart = self._get_time()


__TICTOC_8320947502983745 = TicToc()
tic = __TICTOC_8320947502983745.tic
toc = __TICTOC_8320947502983745.toc
logger_initialized = {}


def get_logger(name, save_dir, distributed_rank, filename="log.log", resume=False):
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger

    logger.propagate = False
    # don't log results for the non-master process
    if distributed_rank > 0:
        logger.setLevel(logging.ERROR)
        return logger
    formatter = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        if resume:
            fh = logging.FileHandler(
                os.path.join(save_dir, filename),
                mode="a",
            )
        else:
            fh = logging.FileHandler(
                os.path.join(save_dir, filename),
                mode="w",
            )
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.setLevel(logging.INFO)

    logger_initialized[name] = True

    return logger


def print_warning(
    message: str = "WARNING",
    verbose: bool = True,
    skip_line_before: bool = True,
    skip_line_after: bool = True,
    bold: bool = False,
) -> None:
    """Print message in yellow."""
    if verbose:
        string_before = "\n" if skip_line_before else ""
        string_after = "\n" if skip_line_after else ""
        if bold:
            print(
                f"{string_before}\x1b[1;30;43m[ {message} ]\x1b[0m{string_after}",
            )
        else:
            print(f"{string_before}\x1b[33m{message}\x1b[0m{string_after}")


def print_ok(
    message: str = "OK",
    verbose: bool = True,
    skip_line_before: bool = True,
    skip_line_after: bool = True,
    bold: bool = False,
) -> None:
    """Print message in green."""
    if verbose:
        string_before = "\n" if skip_line_before else ""
        string_after = "\n" if skip_line_after else ""
        if bold:
            print(
                f"{string_before}\x1b[1;30;42m[ {message} ]\x1b[0m{string_after}",
            )
        else:
            print(f"{string_before}\x1b[32m{message}\x1b[0m{string_after}")
