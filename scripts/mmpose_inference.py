# Copyright (c) OpenMMLab. All rights reserved.
import os
from argparse import ArgumentParser
import json_tricks as json
import mmcv
import tqdm
import numpy as np

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline


def load_labelme_annotation(file_path):
    """ Load labelme annotation to get bounding boxes for the specified categories. """
    with open(file_path, 'r') as f:
        labelme_data = json.load(f)

    bboxes = []
    for shape in labelme_data['shapes']:
        if shape['label'] in ['red_side', 'blue_side']:  # Detect only the required categories
            points = np.array(shape['points'])
            x_min, y_min = np.min(points, axis=0)
            x_max, y_max = np.max(points, axis=0)
            bbox = [x_min, y_min, x_max, y_max]  # x_min, y_min, x_max, y_max
            bboxes.append(bbox)

    return np.array(bboxes), labelme_data


def process_one_image(args,
                      img,
                      pose_estimator,
                      labelme_annotation_file,
                      visualizer=None,
                      show_interval=0):
    """Visualize predicted keypoints (and heatmaps) of one image."""

    # Load bounding boxes from labelme annotation
    bboxes, labelme_data = load_labelme_annotation(labelme_annotation_file)

    # Predict keypoints using pose estimator
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    data_samples = merge_data_samples(pose_results)

    # If visualizer is not None, visualize results
    if isinstance(img, str):
        img = mmcv.imread(img, channel_order='rgb')
    elif isinstance(img, np.ndarray):
        img = mmcv.bgr2rgb(img)

    if visualizer is not None:
        visualizer.add_datasample(
            'result',
            img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=args.draw_heatmap,
            draw_bbox=args.draw_bbox,
            show_kpt_idx=args.show_kpt_idx,
            skeleton_style=args.skeleton_style,
            show=args.show,
            wait_time=show_interval,
            kpt_thr=args.kpt_thr)

    # Return predicted keypoints (17 points)
    return data_samples.get('pred_instances', None), labelme_data


def convert_to_labelme_format(pred_instances, labelme_data):
    """ Convert predicted instances into Labelme format. """
    # labelme_data = {
    #     "version": "4.5.7",
    #     "flags": {},
    #     "shapes": [],
    #     "imagePath": "",  # Fill with actual image filename later
    #     "imageData": None,
    #     "imageHeight": image_height,
    #     "imageWidth": image_width
    # }

    # List of keypoint labels
    keypoint_labels = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]

    bboxes_shapes = labelme_data["shapes"]
    final_shapes = []

    for i, instance in enumerate(pred_instances):
        keypoints = instance['keypoints'][0]  # This would be a 17x2 array
        kpts_scores = instance['keypoint_scores'][0]
        kpts_shapes = []
        final_shapes.append(bboxes_shapes[i])

        for i, (x, y) in enumerate(keypoints):
            if x > 0 and y > 0 and kpts_scores[i] > 0.35:  # Only consider valid keypoints
                kpts_shapes.append({
                    'label': keypoint_labels[i],
                    'points': [[x, y]],
                    'shape_type': 'point',
                    'flags': {}
                })

        # Append shapes to labelme data
        final_shapes.extend(kpts_shapes)

    labelme_data['shapes'] = final_shapes

    return labelme_data


"""
/Users/zhaosilei/PycharmProjects/Benz/mmpose-main/configs/body_2d_keypoint/rtmpose/body8/rtmpose-l_8xb256-420e_body8-256x192.py
/Users/zhaosilei/PycharmProjects/Benz/mmpose-main/checkpoints/rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.pth
--input-dir /home/zhaosilei/Datasets/Benz/sanda_det_labelme
--labelme-dir /home/zhaosilei/Datasets/Benz/sanda_det_labelme
--output-dir /home/zhaosilei/Datasets/Benz/sanda_pose_labelme
--device cpu
"""


def main():
    """ Visualize the demo images and save Labelme annotations. """

    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument(
        '--input-dir', type=str, required=True, help='Directory containing images')
    parser.add_argument(
        '--labelme-dir', type=str, required=True, help='Directory containing Labelme annotations')
    parser.add_argument(
        '--output-dir', type=str, required=True, help='Directory to save new Labelme annotations')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id', type=int, default=0, help='Category id for bounding box detection model')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Visualizing keypoint thresholds')
    parser.add_argument(
        '--draw-heatmap', action='store_true', default=False, help='Draw heatmap predicted by the model')
    parser.add_argument(
        '--show-kpt-idx', action='store_true', default=False, help='Whether to show the index of keypoints')
    parser.add_argument(
        '--skeleton-style', default='mmpose', type=str, choices=['mmpose', 'openpose'], help='Skeleton style selection')
    parser.add_argument(
        '--radius', type=int, default=3, help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness', type=int, default=1, help='Link thickness for visualization')
    parser.add_argument(
        '--show-interval', type=int, default=0, help='Sleep seconds per frame')

    args = parser.parse_args()

    # Build detector
    # detector = init_detector(args.det_config, args.det_checkpoint, device=args.device)
    # detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # Build pose estimator
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device,
        cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))))

    # Build visualizer
    # pose_estimator.cfg.visualizer.radius = args.radius
    # pose_estimator.cfg.visualizer.alpha = args.alpha
    # pose_estimator.cfg.visualizer.line_width = args.thickness
    # visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    # visualizer.set_dataset_meta(pose_estimator.dataset_meta, skeleton_style=args.skeleton_style)

    # Process each image in the input directory
    for img_file in tqdm.tqdm(os.listdir(args.input_dir)):
        if img_file.endswith('.jpg'):
            img_path = os.path.join(args.input_dir, img_file)
            labelme_annotation_file = os.path.join(args.labelme_dir, os.path.splitext(img_file)[0] + '.json')
            output_annotation_file = os.path.join(args.output_dir, os.path.splitext(img_file)[0] + '.json')

            # img = cv2.imread(img_path)

            # Inference and get the predicted instances
            pred_instances, labelme_data = process_one_image(args, img_path, pose_estimator, labelme_annotation_file)

            # Convert to Labelme format and save
            if pred_instances is not None:
                labelme_data = convert_to_labelme_format(pred_instances, labelme_data)
                labelme_data['imagePath'] = img_file
                with open(output_annotation_file, 'w') as f:
                    json.dump(labelme_data, f, indent=2)

    print(f'Finished processing all images. The annotations have been saved in {args.output_dir}.')


if __name__ == '__main__':
    main()
