##############################################################################
# Config
# Config is used to set dataset path for training and testing
##############################################################################


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from utils.attr_dict import AttrDict


__C = AttrDict()
cfg = __C
__C.EPOCH = 0
# Use Class Uniform Sampling to give each class proper sampling
__C.CLASS_UNIFORM_PCT = 0.0

# Use class weighted loss per batch to increase loss for low pixel count classes per batch
__C.BATCH_WEIGHTING = False

# Border Relaxation Count
__C.BORDER_WINDOW = 1
# Number of epoch to use before turn off border restriction
__C.REDUCE_BORDER_EPOCH = -1
# Comma Seperated List of class id to relax
__C.STRICTBORDERCLASS = None


# Attribute Dictionary for Dataset
__C.DATASET = AttrDict()
# Cityscapes Dir Location
# iSAID Dataset Dir Location
__C.DATASET.iSAID_DIR = '/home/gmh/datasets/iSAID_patches'
__C.DATASET.GAOFENSAR = 'gaofen/sar'
__C.DATASET.GAOFENIMG = 'gaofen/img'
# Number of splits to support
__C.DATASET.CV_SPLITS = 3


__C.MODEL = AttrDict()
__C.MODEL.BN = 'regularnorm'
__C.MODEL.BNFUNC = None


def assert_and_infer_cfg(args, make_immutable=True, train_mode=True):
    """Call this function in your script after you have finished setting all cfg
    values that are necessary (e.g., merging a config from a file, merging
    command line config options, etc.). By default, this function will also
    mark the global cfg as immutable to prevent changing the global cfg settings
    during script execution (which can lead to hard to debug errors or code
    that's harder to understand than is necessary).
    """

    if not train_mode:
        cfg.immutable(True)
        return

    if args.batch_weighting:
        __C.BATCH_WEIGHTING = True

    if args.jointwtborder:
        if args.strict_bdr_cls != '':
            __C.STRICTBORDERCLASS = [int(i)
                                     for i in args.strict_bdr_cls.split(",")]
        if args.rlx_off_epoch > -1:
            __C.REDUCE_BORDER_EPOCH = args.rlx_off_epoch

    if make_immutable:
        cfg.immutable(True)
