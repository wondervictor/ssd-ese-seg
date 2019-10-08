from __future__ import division
from __future__ import print_function

import os
import argparse
import logging
logging.basicConfig(level=logging.INFO)
import time
import numpy as np
import mxnet as mx
from tqdm import tqdm
from mxnet import nd
from mxnet import gluon
import gluoncv as gcv
from gluoncv import data as gdata
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.ssd import SSDNewValTransform
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.utils.metrics.voc_polygon_detection import VOC07PolygonMApMetric

from IPython import embed

def parse_args():
    parser = argparse.ArgumentParser(description='Eval SSD networks.')
    parser.add_argument('--network', type=str, default='resnet18_v1',
                        help="Base network name")
    parser.add_argument('--data-shape', type=int, default=512,
                        help="Input data shape")
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training mini-batch size')
    parser.add_argument('--dataset', type=str, default='voc',
                        help='Training dataset.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--pretrained', type=str, default='True',
                        help='Load weights from previously saved parameters.')
    parser.add_argument('--save-prefix', type=str, default='test_ssd',
                        help='Saving parameter prefix')

    parser.add_argument('--deg', type=int, default=8, help='cheby degree , actually 17')
    parser.add_argument('--val_voc2012', type=bool, default=False, help='val in pascal voc 2012')

    args = parser.parse_args()
    return args


def get_dataset(dataset, args):
    if dataset.lower() == 'voc':
        if args.val_voc2012:
            val_dataset = gdata.VOC_Val_Detection(
                splits=[('sbdche', 'val_2012_bboxwh')])
        else:
            val_dataset = gdata.VOC_Val_Detection(
                splits=[('sbdche', 'val' + '_' + str(args.deg) + '_bboxwh')])
        val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
        val_polygon_metric = VOC07PolygonMApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    return val_dataset, val_metric, val_polygon_metric


def get_dataloader(val_dataset, data_shape, batch_size, num_workers):
    """Get dataloader."""
    width, height = data_shape, data_shape
    batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(SSDNewValTransform(width, height)), batchify_fn=batchify_fn,
        batch_size=batch_size, shuffle=False, last_batch='keep', num_workers=num_workers)
    return val_loader


def validate(net, val_data, ctx, eval_metric, polygon_metric, args):
    """Test on validation dataset."""
    eval_metric.reset()
    # set nms threshold and topk constraint
    net.set_nms(nms_thresh=0.45, nms_topk=400)
    mx.nd.waitall()
    net.hybridize()
    for batch in tqdm(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        det_bboxes = []
        det_ids = []
        det_scores = []
        det_coef_centers = []
        det_coefs = []

        gt_bboxes = []
        gt_points_xs = []
        gt_points_ys = []
        gt_ids = []
        gt_difficults = []
        gt_widths = []
        gt_heights = []

        for x, y in zip(data, label):
            # get prediction results
            ids, scores, bboxes, coef_centers, coef = net(x)
            det_ids.append(ids)
            det_scores.append(scores)
            det_coef_centers.append(coef_centers)
            det_coefs.append(coef)
            # clip to image size
            det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))

            # split ground truths
            gt_ids.append(y.slice_axis(axis=-1, begin=4 + 720, end=5 + 720))
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            gt_points_xs.append(y.slice_axis(axis=-1, begin=4, end=4 + 360))
            gt_points_ys.append(y.slice_axis(axis=-1, begin=4 + 360, end=4 + 720))
            gt_difficults.append(y.slice_axis(axis=-1, begin=5 + 720, end=6 + 720) if y.shape[-1] > 5 else None)
            gt_widths.append(y.slice_axis(axis=-1, begin=6 + 720, end=7 + 720))
            gt_heights.append(y.slice_axis(axis=-1, begin=7 + 720, end=8 + 720))

            embed()

        # update metric
        eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)

        polygon_metric.update(det_bboxes, det_coef_centers, det_coefs, det_ids, det_scores, gt_bboxes, gt_points_xs,
                              gt_points_ys, gt_ids, gt_widths, gt_heights, gt_difficults)
    return eval_metric.get(), polygon_metric.get()


def demo_val(net, val_data, eval_metric, polygon_metric, ctx, args):
    """Training pipeline"""
    net.collect_params().reset_ctx(ctx)

    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = args.save_prefix + '_train.log'
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    tic = time.time()
    btic = time.time()
    mx.nd.waitall()
    net.hybridize()

    map_bbox, map_polygon = validate(net, val_data, ctx, eval_metric, polygon_metric, args)
    map_name, mean_ap = map_bbox
    polygonmap_name, polygonmean_ap = map_polygon
    val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
    logger.info('[Epoch {}] Validation: \n{}'.format(0, val_msg))
    polygonval_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(polygonmap_name, polygonmean_ap)])
    logger.info('[Epoch {}] PolygonValidation: \n{}'.format(0, polygonval_msg))


if __name__ == '__main__':
    args = parse_args()

    # training contexts
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]

    # network
    net_name = '_'.join(('ssd', str(args.data_shape), args.network, args.dataset))
    args.save_prefix += net_name
    if args.pretrained.lower() in ['true', '1', 'yes', 't']:
        net = gcv.model_zoo.get_model(net_name, pretrained=True)
    else:
        net = gcv.model_zoo.get_model(net_name, pretrained=False)
        net.load_parameters(args.pretrained.strip())

    # training data
    val_dataset, val_metric, val_polygon_metric = get_dataset(args.dataset, args)
    val_data = get_dataloader(
        val_dataset, args.data_shape, args.batch_size, args.num_workers)

    demo_val(net, val_data, val_metric, val_polygon_metric, ctx, args)
