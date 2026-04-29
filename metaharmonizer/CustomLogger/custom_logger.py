import logging
import logging.handlers
import inspect
import os
import sys


## have flexibility to pass log level
class CustomLogger:

    def custlogger(self, loglevel=logging.DEBUG):
        # Set class name from where logger is called
        stack = inspect.stack()
        the_class = stack[1][0].f_locals.get("self", None)
        logger_name = the_class.__class__.__name__ if the_class else "DefaultLogger"

        # Create or get logger
        logger = logging.getLogger(logger_name)
        logger.setLevel(loglevel)

        # Add handlers only if they are not already added
        if not logger.handlers:
            # Rotating file handler (configurable via LOG_FILE env var)
            log_path = os.getenv("LOG_FILE", "out.log")
            fh = logging.handlers.RotatingFileHandler(
                log_path, maxBytes=10 * 1024 * 1024, backupCount=3
            )
            # Stream handler
            stdout = logging.StreamHandler(stream=sys.stdout)

            # Formatter
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(name)s: %(message)s",
                datefmt="%d/%m/%Y %I:%M:%S %p",
            )

            fh.setFormatter(formatter)
            stdout.setFormatter(formatter)

            logger.addHandler(fh)
            logger.addHandler(stdout)

        return logger

    # Create a base class

    def log(self):
        stack = inspect.stack()
        logger = logging.getLogger(self.__class__.__name__)
        try:
            logger.debug("Whole stack is:")
            logger.debug("\n".join([str(x[4]) for x in stack]))
            logger.debug("-" * 20)
            logger.debug("Caller was %s" % (str(stack[2][4])))
        finally:
            del stack


## An easy logger to include class name
## Does not work with my setup currently
## Might want to figure out the exact handling in logger module
class LoggingHandler:

    def __init__(self, *args, **kwargs):
        self.log = logging.getLogger(self.__class__.__name__)
