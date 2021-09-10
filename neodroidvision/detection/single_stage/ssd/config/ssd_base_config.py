from pathlib import Path

from warg import NOD

from neodroidvision import PROJECT_APP_PATH
from neodroidvision.detection import SSDBoxPredictor
from neodroidvision.detection.single_stage.ssd.architecture.backbones import vgg_factory

base_cfg = NOD()

base_cfg.data_dir = Path.home() / "Data" / "Vision" / "Detection"
# base_cfg.DATA_DIR = Path.home() / "Data" / "Datasets"
base_cfg.output_dir = PROJECT_APP_PATH.user_data / "results"

# ---------------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------------- #

base_cfg.model = NOD()
base_cfg.model.device = "cuda"

# ---------------------------------------------------------------------------- #
# Backbone
# ---------------------------------------------------------------------------- #
base_cfg.model.backbone = NOD()
base_cfg.model.backbone.name = vgg_factory
base_cfg.model.backbone.out_channels = (512, 1024, 512, 256, 256, 256)
base_cfg.model.backbone.pretrained = True
base_cfg.model.backbone.predictor_type = SSDBoxPredictor

# ---------------------------------------------------------------------------- #
# Head
# ---------------------------------------------------------------------------- #

base_cfg.model.box_head = NOD()
base_cfg.model.box_head.nms_threshold = 0.45
base_cfg.model.box_head.confidence_threshold = 0.01
base_cfg.model.box_head.max_per_class = -1
base_cfg.model.box_head.max_per_image = 100
base_cfg.model.box_head.batch_size = 10
base_cfg.model.box_head.iou_threshold = 0.5
# match default boxes to any ground truth with jaccard overlap higher than a
# threshold (0.5)
base_cfg.model.box_head.num_categories = 21

base_cfg.model.box_head.neg_pos_ratio = 3  # hard negative mining
base_cfg.model.box_head.center_variance = 0.1
base_cfg.model.box_head.size_variance = 0.2

# -----------------------------------------------------------------------------
# Priors
# -----------------------------------------------------------------------------
base_cfg.model.box_head.priors = NOD()
base_cfg.model.box_head.priors.feature_maps = (38, 19, 10, 5, 3, 1)
base_cfg.model.box_head.priors.strides = (8, 16, 32, 64, 100, 300)
base_cfg.model.box_head.priors.min_sizes = (30, 60, 111, 162, 213, 264)
base_cfg.model.box_head.priors.max_sizes = (60, 111, 162, 213, 264, 315)
base_cfg.model.box_head.priors.aspect_ratios = (
    (2,),
    (2, 3),
    (2, 3),
    (2, 3),
    (2,),
    (2,),
)  # when has 1 aspect ratio,
# every location has 4 boxes, 2 ratio 6 boxes. #boxes = 2 + #ratio * 2
base_cfg.model.box_head.priors.boxes_per_location = (
    4,
    6,
    6,
    6,
    4,
    4,
)  # number of boxes per feature map location
base_cfg.model.box_head.priors.clip = True

# -----------------------------------------------------------------------------
# Input
# -----------------------------------------------------------------------------
base_cfg.input = NOD()
base_cfg.input.image_size = 300  # Image size
base_cfg.input.pixel_mean = (
    123,
    117,
    104,
)  # Values to be used for image normalization, RGB layout

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
base_cfg.datasets = NOD()
base_cfg.datasets.train = ()  # list of the dataset names for training, as present in paths_catalog.py
base_cfg.datasets.test = ()  # List of the dataset names for testing, as present in paths_catalog.py
base_cfg.dataset_type = None

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
base_cfg.data_loader = NOD()
base_cfg.data_loader.num_workers = 0  # number of data loading threads
base_cfg.data_loader.pin_memory = True

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
base_cfg.solver = NOD()
base_cfg.solver.max_iter = 120000
base_cfg.solver.lr_steps = (80000, 100000)
base_cfg.solver.gamma = 0.1
base_cfg.solver.batch_size = 32
base_cfg.solver.lr = 1e-3
base_cfg.solver.momentum = 0.9
base_cfg.solver.weight_decay = 5e-4
base_cfg.solver.warmup_factor = 1.0 / 3
base_cfg.solver.warmup_iters = 500
