"""Application for log

Features:
    1. debug
    2. info
    3. warn
    4. error
    5. critical
    
"""
import os
import logging


class Log(object):
    """Define a log for record.
    """

    def __init__(self, name):
        """Create a logger for record.

        Args:
            name (str) : the file name of the generating log

        Example::

            >>> name = 'log.py'
            >>> logger = Log(name)

        Return:
            logger (Log) : an instance of class Log with log file {name}

        """
        self._init()
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler('./log/{}.log'.format(name))
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.info('Start {}'.format(name))

    def _init(self):
        """Ensure the log directory
        """
        if not os.path.exists('./log/'):
            os.makedirs('./log/')

    def debug(self, mess):
        """Add debug message. Level 10

        Args:
            mess (str): message for log

        Example::

            >>> logger = Log('log')
            >>> logger.debug('This is debug.')

        """
        self.logger.debug(mess)

    def info(self, mess):
        """Add info message. Level 20

        Args:
            mess (str): message for log

        Example::

            >>> logger = Log('log')
            >>> logger.info('This is info.')

        """
        self.logger.info(mess)

    def warn(self, mess):
        """Add warning message. Level 30

        Args:
            mess (str): message for log

        Example::

            >>> logger = Log('log')
            >>> logger.warn('This is warn.')

        """
        self.logger.warn(mess)

    def error(self, mess):
        """Add error message. Level 40

        Args:
            mess (str): message for log

        Example::

            >>> logger = Log('log')
            >>> logger.error('This is error.')

        """
        self.logger.error(mess)

    def exception(self, mess):
        """[Not work] Add exception message. Level 40
        """
        self.logger.exception(self.name)

    def critical(self, mess):
        """Add critical message. Level 50  

        Args:   
            mess (str): message for log

        Example::

            >>> logger = Log('log')
            >>> logger.critical('This is critical.')

        """
        self.logger.critical(mess)

    def close(self):
        """Add close message. Level 10
        """
        self.logger.info('Close {}'.format(self.name))


if __name__ == '__main__':
    print(__doc__)
    logger = Log(__name__)
    logger.close()
