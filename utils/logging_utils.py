import logging

def setup_logging( log_file ):

    """
    Set up logging configurationn.

    Parameters:
    --------

    log_file: str
        - Path to the log file.

    Returns:
    --------

    logger: logging.Logger
        - The configured logger instance.
    """

    logging.basicConfig( filename=log_file, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s' )
    
    return logging.getLogger()