from aim.utils import DetectionModel, VideoInference

import argparse
import os
import numpy as np

from pycocotools.coco import COCO
from mmdet.apis import init_detector, inference_detector

# from torchvision.datasets import VisionDataset

# class CocoRawFormat(VisionDataset):
#     def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None):
#         super(CocoDetection, self).__init__(root, transforms, transform, target_transform)
#         self.coco = COCO(annFile)
#         self.ids = list(sorted(self.coco.imgs.keys()))

#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index
#         Returns:
#             tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
#         """
#         coco = self.coco
#         img_id = self.ids[index]
#         ann_ids = coco.getAnnIds(imgIds=img_id)
#         target = coco.loadAnns(ann_ids)

#         path = coco.loadImgs(img_id)[0]['file_name']
#         return os.path.join(self.root, path), target

#     def __len__(self):
#         return len(self.ids)


class MMDetection(DetectionModel):

    def __init__(self, config_file, checkpoint_file):
        self.model = init_detector(config_file, checkpoint_file, device='cuda')

    def __call__(self, *args, **kwargs):
        result = inference_detector(self.model, args[0])
        if isinstance(result, tuple):
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None
            
        results = []
        for i, bboxes in enumerate(bbox_result):
            for bbox in bboxes:
                obj = dict()
                obj['category_id'] = i
                obj['score'] = float(bbox[-1])
                obj['bbox'] = [int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])]
                results.append(obj)
                
        return results


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Video Analysis.')
    parser.add_argument('--config_file', required=True)
    parser.add_argument('--checkpoint_file', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--input', default='test.mp4', type=str)
    parser.add_argument('--output', default='output.mp4', type=str)
    parser.add_argument('--threshold', default=0.4, type=float)
    parser.add_argument('--crop', default='0:0:0:0', type=str)
    args = parser.parse_args()

    # dataset = CocoRawFormat(os.path.join(args.dataset, 'val'), os.path.join(args.dataset, 'val.json'))
    model = MMDetection(args.config_file, args.checkpoint_file)

    crop = [int(c) for c in args.crop.split(':')]
    video_inference = VideoInference(os.path.join(args.dataset, 'val.json'), args.input, args.output, args.threshold, dict(x=crop[2], y=crop[3], w=crop[0], h=crop[1]))
    video_inference.run(model)
