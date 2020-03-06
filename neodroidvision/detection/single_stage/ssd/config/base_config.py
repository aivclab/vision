from neodroidvision import PROJECT_APP_PATH
from neodroidvision.detection.single_stage.ssd.architecture import SSD
from neodroidvision.detection.single_stage.ssd.architecture.backbone.vgg import vgg
from neodroidvision.detection.single_stage.ssd.architecture.box_head.box_head import (
    SSDBoxHead,
)
from neodroidvision.detection.single_stage.ssd.architecture.box_head.box_predictor import (
    SSDBoxPredictor,
)
from warg import NOD

base_cfg = NOD()

base_cfg.OUTPUT_DIR = PROJECT_APP_PATH.user_data / "results"

base_cfg.MODEL = NOD()
base_cfg.MODEL.META_ARCHITECTURE = SSD
base_cfg.MODEL.DEVICE = "cuda"
# match default boxes to any ground truth with jaccard overlap higher than a threshold (0.5)
base_cfg.MODEL.THRESHOLD = 0.5
base_cfg.MODEL.NUM_CLASSES = 21
# Hard negative mining
base_cfg.MODEL.NEG_POS_RATIO = 3
base_cfg.MODEL.CENTER_VARIANCE = 0.1
base_cfg.MODEL.SIZE_VARIANCE = 0.2

# ---------------------------------------------------------------------------- #
# Backbone
# ---------------------------------------------------------------------------- #
base_cfg.MODEL.BACKBONE = NOD()
base_cfg.MODEL.BACKBONE.NAME = vgg
base_cfg.MODEL.BACKBONE.OUT_CHANNELS = (512, 1024, 512, 256, 256, 256)
base_cfg.MODEL.BACKBONE.PRETRAINED = True

# -----------------------------------------------------------------------------
# PRIORS
# -----------------------------------------------------------------------------
base_cfg.MODEL.PRIORS = NOD()
base_cfg.MODEL.PRIORS.FEATURE_MAPS = [38, 19, 10, 5, 3, 1]
base_cfg.MODEL.PRIORS.STRIDES = [8, 16, 32, 64, 100, 300]
base_cfg.MODEL.PRIORS.MIN_SIZES = [30, 60, 111, 162, 213, 264]
base_cfg.MODEL.PRIORS.MAX_SIZES = [60, 111, 162, 213, 264, 315]
base_cfg.MODEL.PRIORS.ASPECT_RATIOS = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
# When has 1 aspect ratio, every location has 4 boxes, 2 ratio 6 boxes.
# #boxes = 2 + #ratio * 2
base_cfg.MODEL.PRIORS.BOXES_PER_LOCATION = [
    4,
    6,
    6,
    6,
    4,
    4,
]  # number of boxes per feature map location
base_cfg.MODEL.PRIORS.CLIP = True

# -----------------------------------------------------------------------------
# Box Head
# -----------------------------------------------------------------------------
base_cfg.MODEL.BOX_HEAD = NOD()
base_cfg.MODEL.BOX_HEAD.HEAD = SSDBoxHead
base_cfg.MODEL.BOX_HEAD.PREDICTOR = SSDBoxPredictor

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
base_cfg.INPUT = NOD()
# Image size
base_cfg.INPUT.IMAGE_SIZE = 300
# Values to be used for image normalization, RGB layout
base_cfg.INPUT.PIXEL_MEAN = [123, 117, 104]

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
base_cfg.DATASETS = NOD()
# List of the dataset names for training, as present in paths_catalog.py
base_cfg.DATASETS.TRAIN = ()
# List of the dataset names for testing, as present in paths_catalog.py
base_cfg.DATASETS.TEST = ()
base_cfg.dataset_type = ""

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
base_cfg.DATA_LOADER = NOD()
# Number of data loading threads
base_cfg.DATA_LOADER.NUM_WORKERS = 8
base_cfg.DATA_LOADER.PIN_MEMORY = True

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
base_cfg.SOLVER = NOD()
# train configs
base_cfg.SOLVER.MAX_ITER = 120000
base_cfg.SOLVER.LR_STEPS = [80000, 100000]
base_cfg.SOLVER.GAMMA = 0.1
base_cfg.SOLVER.BATCH_SIZE = 32
base_cfg.SOLVER.LR = 1e-3
base_cfg.SOLVER.MOMENTUM = 0.9
base_cfg.SOLVER.WEIGHT_DECAY = 5e-4
base_cfg.SOLVER.WARMUP_FACTOR = 1.0 / 3
base_cfg.SOLVER.WARMUP_ITERS = 500

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
base_cfg.TEST = NOD()
base_cfg.TEST.NMS_THRESHOLD = 0.45
base_cfg.TEST.CONFIDENCE_THRESHOLD = 0.01
base_cfg.TEST.MAX_PER_CLASS = -1
base_cfg.TEST.MAX_PER_IMAGE = 100
base_cfg.TEST.BATCH_SIZE = 10
