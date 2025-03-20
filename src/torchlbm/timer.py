import timeit
import threading

from torchlbm.logger import Logger

timer_context = threading.local()
timer_context.active = True


class Timer:
    def __init__(self, logger: Logger, name: str, indent_level: int):
        self.__total_time = 0
        self.__last_time = 0
        self.__logger = logger
        self.__name = name
        self.__indent_level = indent_level

    def __enter__(self):
        self.__start_time = timeit.default_timer()

    def __exit__(self, *args, **kwargs):
        self.__last_time = timeit.default_timer() - self.__start_time

        if timer_context.active:
            self.__total_time += self.__last_time

    def __str__(self) -> str:
        result = f"Last time:  {self.__last_time:.6e}\n"
        result += f"Total time: {self.__total_time:.6e}\n\n"
        return result

    @property
    def total_time(self):
        return self.__total_time

    @property
    def last_time(self):
        return self.__last_time

    def log(self) -> None:
        self.__logger.indent += self.__indent_level
        self.__logger.write(f"{self.__name}:")
        self.__logger.write(str(self))
        self.__logger.indent -= self.__indent_level
