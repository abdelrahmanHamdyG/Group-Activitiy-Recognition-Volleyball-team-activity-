import logging

class Logger:
    def __init__(self, name, log_file="app.log", log_level=logging.DEBUG):
        """
        Initializes the logger with the given name, log file, and log level.
        
        :param name: The name of the logger.
        :param log_file: The log file where the logs will be saved.
        :param log_level: The logging level (default is DEBUG).
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # Prevent messages from being propagated to the root logger
        self.logger.propagate = False

        # Check if handlers are already added to avoid duplicate logs
        if not self.logger.handlers:
            # Create a file handler that logs messages to a file
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)

            # Create a formatter and set it for the file handler
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)

            # Add the file handler to the logger
            self.logger.addHandler(file_handler)

    def debug(self, msg):
        """Logs a message with DEBUG level."""
        self.logger.debug(msg)

    def info(self, msg):
        """Logs a message with INFO level."""
        self.logger.info(msg)

    def warning(self, msg):
        """Logs a message with WARNING level."""
        self.logger.warning(msg)

    def error(self, msg):
        """Logs a message with ERROR level."""
        self.logger.error(msg)

    def critical(self, msg):
        """Logs a message with CRITICAL level."""
        self.logger.critical(msg)


# Example usage:
if __name__ == "__main__":
    logger = Logger("MyAppLogger", log_file="app.log")
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")
