from model.caption_segmentation import SegmentationModule

def train_seg(cfg):
    segmentation_model = SegmentationModule(cfg)

    return