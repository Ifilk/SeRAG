import logging

from colorlog import colorlog

logger = logging.getLogger('main')

log_colors_config = {
    'DEBUG': 'white',
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'bold_red',
}

console_handler = logging.StreamHandler()

logger.setLevel(logging.INFO)
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(filename='main.log', mode='a', encoding='utf8')

file_handler.setLevel(logging.NOTSET)

file_formatter = logging.Formatter(
    fmt='[%(asctime)s.%(msecs)03d] %(filename)s ->'
        ' %(funcName)s line:%(lineno)d [%(levelname)s] : %(message)s',
    datefmt='%Y-%m-%d  %H:%M:%S'
)
console_formatter = colorlog.ColoredFormatter(
    fmt='%(log_color)s[%(asctime)s.%(msecs)03d] %(filename)s ->'
        ' %(funcName)s line:%(lineno)d [%(levelname)s] : %(message)s',
    datefmt='%Y-%m-%d  %H:%M:%S',
    log_colors=log_colors_config
)
console_handler.setFormatter(console_formatter)

file_handler.setFormatter(file_formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

console_handler.close()
file_handler.close()