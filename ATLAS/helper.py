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

def getOutputDir():        
    import os
    outputdir = ""
    try:
        outputdir = os.environ["OUTPUTDIR"]
    except KeyError:
        print("ERROR: Environment variable 'OUTPUTDIR' " \
              "not found! Maybe you forgot to execute " \
              "'source setup.sh'?")
        os._exit(0)
    return outputdir
        
