import logging

# Define the formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create handlers
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler('/logs/app.log')

# Set the formatter for the handlers
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Create a logger
logger = logging.getLogger('formattedLogger')

# Set the log level for the logger
logger.setLevel(logging.DEBUG)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)
