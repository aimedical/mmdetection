from .coco import CocoDataset
from .registry import DATASETS

@DATASETS.register_module
class EC_NO_MAG_Dataset(CocoDataset):

    CLASSES = ('CANCER_WLI',
               'CANCER_NBI',
               'NON_CANCER_WLI',
               'NON_CANCER_NBI')


@DATASETS.register_module
class EC_MAG_Dataset(CocoDataset):

    CLASSES = ('CANCER_WLI',
               'CANCER_NBI',
               'NON_CANCER_WLI',
               'NON_CANCER_NBI')


@DATASETS.register_module
class EC_DEPTH_Dataset(CocoDataset):

    CLASSES = ('M_WLI',
               'M_NBI',
               'SM1_WLI',
               'SM1_NBI',
               'SM2_WLI',
               'SM2_NBI',
               'ADV_WLI',
               'ADV_NBI')
