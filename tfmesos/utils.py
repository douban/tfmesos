import struct
import logging
from six.moves import cPickle as pickle


def send(fd, o):
    d = pickle.dumps(o)
    fd.send(struct.pack('>I', len(d)) + d)


def recv(fd):
    d = fd.recv(struct.calcsize('>I'))
    assert len(d) == struct.calcsize('>I'), repr(d)
    size, = struct.unpack('>I', d)
    return pickle.loads(fd.recv(size))


def setup_logger(logger):
    FORMAT = '%(asctime)-11s [%(levelname)s] [%(name)-9s] %(message)s'
    formatter = logging.Formatter(FORMAT)
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger
