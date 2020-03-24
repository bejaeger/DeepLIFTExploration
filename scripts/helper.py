def mkdir_p(path):
    import os
    import errno
    """create a directory, emulating the behavior of 'mkdir -p'"""
    try:
        os.makedirs(path)
    except OSError as exc: 
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise
