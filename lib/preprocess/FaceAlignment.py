import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import cv2
import time
import face_recognition

from mmcv.parallel import collate
from mmpose.apis import *
from mmpose.datasets import DatasetInfo
from mmpose.utils.hooks import OutputHook
from mmpose.datasets.pipelines import Compose

from lib.preprocess.utils import *


def _inference_single_pose_model(model,
                                 img_or_path,
                                 bboxes,
                                 dataset_info=None,
                                 return_heatmap=False):
    """Inference human bounding boxes.

    Note:
        - num_bboxes: N
        - num_keypoints: K

    Args:
        model (nn.Module): The loaded pose model.
        img_or_path (str | np.ndarray): Image filename or loaded image.
        bboxes (list | np.ndarray): All bounding boxes (with scores),
            shaped (N, 4) or (N, 5). (left, top, width, height, [score])
            where N is number of bounding boxes.
        dataset (str): Dataset name. Deprecated.
        dataset_info (DatasetInfo): A class containing all dataset info.
        outputs (list[str] | tuple[str]): Names of layers whose output is
            to be returned, default: None

    Returns:
        ndarray[NxKx3]: Predicted pose x, y, score.
        heatmap[N, K, H, W]: Model output heatmap.
    """

    cfg = model.cfg
    device = next(model.parameters()).device

    # build the data pipeline
    class LoadImage:
        def __init__(self):
            pass

        def __call__(self, data):
            data['img'] = data['img_or_path']
            data['image_file'] = ''
            return data

    test_pipeline = [LoadImage()] + cfg.test_pipeline[1:]
    test_pipeline = Compose(test_pipeline)

    assert len(bboxes[0]) in [4, 5]

    dataset_name = dataset_info.dataset_name
    flip_pairs = dataset_info.flip_pairs
    

    batch_data = []
    for i in range(len(img_or_path)):
        center, scale = box2cs(cfg, bboxes[i])

        # prepare data
        data = {
            'img_or_path':
            img_or_path[i],
            'center':
            center,
            'scale':
            scale,
            'bbox_score':
            1,
            'bbox_id':
            0,  # need to be assigned if batch_size > 1
            'dataset':
            dataset_name,
            'joints_3d':
            np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
            'joints_3d_visible':
            np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
            'rotation':
            0,
            'ann_info': {
                'image_size': np.array(cfg.data_cfg['image_size']),
                'num_joints': cfg.data_cfg['num_joints'],
                'flip_pairs': flip_pairs
            }
        }
        data = test_pipeline(data)
        batch_data.append(data)

    batch_data = collate(batch_data, samples_per_gpu=1)

    if next(model.parameters()).is_cuda:
        # scatter not work so just move image to cuda device
        batch_data['img'] = batch_data['img'].to(device)
    # get all img_metas of each bounding box
    batch_data['img_metas'] = [
        img_metas[0] for img_metas in batch_data['img_metas'].data
    ]

    # forward the model
    with torch.no_grad():
        result = model(
            img=batch_data['img'],
            img_metas=batch_data['img_metas'],
            return_loss=False,
            return_heatmap=return_heatmap)

    return result['preds'], result['output_heatmap']


def inference_top_down_pose_model(model,
                                  img_or_path,
                                  person_results=None,
                                  bbox_thr=None,
                                  format='xywh',
                                  dataset_info=None,
                                  return_heatmap=False):
    """Inference a single image with a list of person bounding boxes.

    Note:
        - num_people: P
        - num_keypoints: K
        - bbox height: H
        - bbox width: W

    Args:
        model (nn.Module): The loaded pose model.
        img_or_path (str| np.ndarray): Image filename or loaded image.
        person_results (list(dict), optional): a list of detected persons that
            contains ``bbox`` and/or ``track_id``:

            - ``bbox`` (4, ) or (5, ): The person bounding box, which contains
                4 box coordinates (and score).
            - ``track_id`` (int): The unique id for each human instance. If
                not provided, a dummy person result with a bbox covering
                the entire image will be used. Default: None.
        bbox_thr (float | None): Threshold for bounding boxes. Only bboxes
            with higher scores will be fed into the pose detector.
            If bbox_thr is None, all boxes will be used.
        format (str): bbox format ('xyxy' | 'xywh'). Default: 'xywh'.

            - `xyxy` means (left, top, right, bottom),
            - `xywh` means (left, top, width, height).
        dataset (str): Dataset name, e.g. 'TopDownCocoDataset'.
            It is deprecated. Please use dataset_info instead.
        dataset_info (DatasetInfo): A class containing all dataset info.
        return_heatmap (bool) : Flag to return heatmap, default: False
        outputs (list(str) | tuple(str)) : Names of layers whose outputs
            need to be returned. Default: None.

    Returns:
        tuple:
        - pose_results (list[dict]): The bbox & pose info. \
            Each item in the list is a dictionary, \
            containing the bbox: (left, top, right, bottom, [score]) \
            and the pose (ndarray[Kx3]): x, y, score.
        - returned_outputs (list[dict[np.ndarray[N, K, H, W] | \
            torch.Tensor[N, K, H, W]]]): \
            Output feature maps from layers specified in `outputs`. \
            Includes 'heatmap' if `return_heatmap` is True.
    """
    # get dataset info
    if (dataset_info is None and hasattr(model, 'cfg')
            and 'dataset_info' in model.cfg):
        dataset_info = DatasetInfo(model.cfg.dataset_info)
    if dataset_info is None:
        warnings.warn(
            'dataset is deprecated.'
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663'
            ' for details.', DeprecationWarning)

    # only two kinds of bbox format is supported.
    assert format in ['xyxy', 'xywh']

    pose_results = []

    if len(person_results) == 0:
        return pose_results

    # Change for-loop preprocess each bbox to preprocess all bboxes at once.
    bboxes = np.array([box['bbox'] for box in person_results])

    if format == 'xyxy':
        bboxes_xyxy = bboxes
        bboxes_xywh = xyxy2xywh(bboxes)
    else:
        # format is already 'xywh'
        bboxes_xywh = bboxes
        bboxes_xyxy = xywh2xyxy(bboxes)

    # if bbox_thr remove all bounding box
    if len(bboxes_xywh) == 0:
        return [], []

    poses, heatmap = _inference_single_pose_model(
        model,
        img_or_path,
        bboxes_xywh,
        dataset_info=dataset_info,
        return_heatmap=return_heatmap)

    assert len(poses) == len(person_results), print(
        len(poses), len(person_results), len(bboxes_xyxy))
    for pose, person_result, bbox_xyxy in zip(poses, person_results,
                                              bboxes_xyxy):
        pose_result = person_result.copy()
        pose_result['keypoints'] = pose
        pose_result['bbox'] = bbox_xyxy
        pose_results.append(pose_result)

    return pose_results


class FaceAlignment():
    def __init__(self, device):

        pose_config = 'assets/mmpose/mobilenetv2_coco_wholebody_face_256x256.py'
        pose_checkpoint = 'assets/mmpose/mobilenetv2_coco_wholebody_face_256x256-4a3f096e_20210909.pth'
        self.pose_model = init_pose_model(pose_config, pose_checkpoint, device=device)

        dataset_info = self.pose_model.cfg.data['test'].get('dataset_info', None)
        self.dataset_info = DatasetInfo(dataset_info)

        self.device = device


        # test fa
        #self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device)

    def get_face(self, images):
        images = [cv2.resize((images[i].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8), (256, 256)) for i in range(images.shape[0])]
        face_det_results = [face_recognition.face_locations(image, number_of_times_to_upsample=1) for image in images]
        for result in face_det_results:
            if len(result) == 0:
                return None
        face_results = process_face_det_results(face_det_results)

        pose_results = inference_top_down_pose_model(
            self.pose_model,
            images,
            face_results,
            bbox_thr=None,
            format='xyxy',
            dataset_info=self.dataset_info,
            return_heatmap=False)

        results = []
        for i in range(len(images)):
            landmarks = pose_results[i]['keypoints'][:,0:2]
            image_to_face_mat = get_transform_mat(landmarks)
            mask = get_image_hull_mask(images[i].shape, landmarks)
            face = cv2.warpAffine(images[i], image_to_face_mat, (128, 128), cv2.INTER_LANCZOS4)
            mask = cv2.warpAffine(mask, image_to_face_mat, (128, 128), cv2.INTER_LANCZOS4)
            face = face * mask[:, :, None]
            results.append(face)
        results = np.array(results)

        results = torch.from_numpy(results / 255).permute(0,3,1,2).float().to(self.device)
        return results

