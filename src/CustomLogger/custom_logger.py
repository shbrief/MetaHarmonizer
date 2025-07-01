import logging
import inspect
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
            # File handler
            fh = logging.FileHandler("out.log")
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
        try:
            print("Whole stack is:")
            print("\n".join([str(x[4]) for x in stack]))
            print("-" * 20)
            print("Caller was %s" % (str(stack[2][4])))
        finally:
            del stack


## An easy logger to include class name
## Does not work with my setup currently
## Might want to figure out the exact handling in logger module
class LoggingHandler:

    def __init__(self, *args, **kwargs):
        self.log = logging.getLogger(self.__class__.__name__)
