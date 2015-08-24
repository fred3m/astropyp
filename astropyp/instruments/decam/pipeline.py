import datapyp
import warnings
import os

class DecamPipeError(Exception):
    pass

class Pipeline(datapyp.core.Pipeline):
    def __init__(self, **kwargs):
        from datapyp.utils import get_bool
        # Make sure that the user included a dictionary of paths to initialize the pipeline
        if 'paths' not in kwargs:
            raise DecamPipeError(
                "You must initialize a pipeline with the following paths: 'temp'")
        if('stacks' not in kwargs['paths'] or 'config' not in kwargs['paths'] 
                or 'log' not in kwargs['paths'] or 'decam' not in kwargs['paths']):
            warnings.warn(
                "It is recommended to initialize a Pipeline with "
                "'log', 'stacks', 'config', 'decam' paths")
        # Check for the decam file index
        if 'idx_connect_str' not in kwargs:
            warnings.warn("If you do not set an 'idx_connect_str' parts of the Pipeline may not work")
        else:
            if kwargs['idx_connect_str'].startswith('sqlite'):
                if not os.path.isfile(kwargs['idx_connect_str'][10:]):
                    logger.info('path', kwargs['idx_connect_str'][10:])
                    if 'create_idx' in kwargs:
                        if not create_idx:
                            raise PipelineError("Unable to locate DECam file index")
                    else:
                        if not get_bool(
                                "DECam file index does not exist, create it now? ('y'/'n')"):
                            raise PipelineError("Unable to locate DECam file index")
                    import astropyp.index as index
                    recursive = get_bool(
                        "Search '{0}' recursively for images? ('y'/'n')")
                    index.build(img_path, idx_connect_str, True, recursive, True)
        datapyp.core.Pipeline.__init__(self, **kwargs)