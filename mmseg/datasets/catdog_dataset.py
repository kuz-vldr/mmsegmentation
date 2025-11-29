from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class CatDogDataset(BaseSegDataset):
    """Cat and Dog segmentation dataset.

    The class names and palette are defined for:
        - 0: background
        - 1: cat
        - 2: dog
    """
    METAINFO = dict(
        classes=('background', 'cat', 'dog'),
        palette=[
            [0, 0, 0],
            [255, 0, 0],    
            [0, 0, 255]     
        ]
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)