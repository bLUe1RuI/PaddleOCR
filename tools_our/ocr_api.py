from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import os
import os.path as osp
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import json
import paddle
from tqdm import tqdm
import yaml

from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
from ppocr.utils.utility import get_image_file_list
from ppocr.utils.logging import get_logger

def load_config(file_path):
    """
    Load config from yml/yaml file.
    Args:
        file_path (str): Path of the config file to be loaded.
    Returns: global config
    """
    _, ext = os.path.splitext(file_path)
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"
    config = yaml.load(open(file_path, 'rb'), Loader=yaml.Loader)
    return config

def get_rotate_crop_image(img, points):
    points = np.array(points)
    assert len(points) == 4
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3]),
        )
    )
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2]),
        )
    )
    pts_std = np.float32([[0, 0], [img_crop_width, 0], [img_crop_width, img_crop_height], [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points.astype('float32'), pts_std)
    dst_img = cv2.warpPerspective(img, M, (img_crop_width, img_crop_height), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img

def run_first_det_api(imgname):
    _config = '../configs/det/det_r50_db++_icdar15.yml'
    config = load_config(_config)
    global_config = config['Global']
    # build model
    model = build_model(config['Architecture'])
    load_model(config, model)
    # build post process
    post_process_class = build_post_process(config['PostProcess'])
    # create data ops
    transforms = []
    for op in config['Eval']['dataset']['transforms']:
        op_name = list(op)[0]
        if 'Label' in op_name:
            continue
        elif op_name == 'KeepKeys':
            op[op_name]['keep_keys'] = ['image', 'shape']
        transforms.append(op)
        
    ops = create_operators(transforms, global_config)
    
    model.eval()

    with open(imgname, 'rb') as f:
        img = f.read()
        data = {'image': img}
    batch = transform(data, ops)
    
    images = np.expand_dims(batch[0], axis=0)
    shape_list = np.expand_dims(batch[1], axis=0)
    images = paddle.to_tensor(images)
    preds = model(images)
    post_result = post_process_class(preds, shape_list)

    det_box_json = {}
    boxes = post_result[0]['points']
    dt_boxes_json = []
    # write result
    for box in boxes:
        tmp_json = {"transcription": ""}
        tmp_json["points"] = np.array(box).tolist()
        dt_boxes_json.append(tmp_json)
    det_box_json['filename'] = imgname
    det_box_json['boxes'] = dt_boxes_json

    del model
    paddle.device.cuda.empty_cache()
    return det_box_json

def post_process_first_det_api(idx, image, det_box, saveroot):
    save_img_path = osp.join(saveroot, f'first_det_result_{idx}.png')
    text_boxes = det_box['points']
    try:
        crop_im = get_rotate_crop_image(image, text_boxes)
    except:
        raise ValueError(f"error rotate crop")
            
    cv2.imwrite(save_img_path, crop_im)

def ocr_det_api(idx, saveroot):
    first_det_filename = osp.join(saveroot, f'first_det_result_{idx}.png')
    # init
    _config = '../configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml'
    config = load_config(_config)
    global_config = config['Global']
    # build model
    model = build_model(config['Architecture'])
    load_model(config, model)
    # build post process
    post_process_class = build_post_process(config['PostProcess'])
    # create data ops
    transforms = []
    for op in config['Eval']['dataset']['transforms']:
        op_name = list(op)[0]
        if 'Label' in op_name:
            continue
        elif op_name == 'KeepKeys':
            op[op_name]['keep_keys'] = ['image', 'shape']
        transforms.append(op)
        
    ops = create_operators(transforms, global_config)
    
    model.eval()
    det_box_json_list = []

    crop_im = first_det_filename
            
    with open(crop_im, 'rb') as f:
        img = f.read()
        data = {'image': img}
    batch = transform(data, ops)

    images = np.expand_dims(batch[0], axis=0)
    shape_list = np.expand_dims(batch[1], axis=0)
    images = paddle.to_tensor(images)
    preds = model(images)
    post_result = post_process_class(preds, shape_list)

    boxes = post_result['Student'][0]['points']
    dt_boxes_json = []
    # write result
    for box in boxes:
        tmp_json = {"transcription": ""}
        tmp_json["points"] = np.array(box).tolist()
        dt_boxes_json.append(tmp_json)
            
    del model
    paddle.device.cuda.empty_cache()
    return dt_boxes_json, first_det_filename

def ocr_rec_api(ocr_boxes, saveroot):
    first_det_filename = osp.join(saveroot, 'first_det_result.png')
    # init
    _config = '../configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml'
    config = load_config(_config)
    config['Global']['character_dict_path'] = osp.join('..', config['Global']['character_dict_path'])
    global_config = config['Global']
    # build post process
    post_process_class = build_post_process(config['PostProcess'],
                                            global_config)
    if hasattr(post_process_class, 'character'):
        char_num = len(getattr(post_process_class, 'character'))
        if config['Architecture']["algorithm"] in ["Distillation",
                                                   ]:  # distillation model
            for key in config['Architecture']["Models"]:
                if config['Architecture']['Models'][key]['Head'][
                        'name'] == 'MultiHead':  # for multi head
                    out_channels_list = {}
                    if config['PostProcess'][
                            'name'] == 'DistillationSARLabelDecode':
                        char_num = char_num - 2
                    out_channels_list['CTCLabelDecode'] = char_num
                    out_channels_list['SARLabelDecode'] = char_num + 2
                    config['Architecture']['Models'][key]['Head'][
                        'out_channels_list'] = out_channels_list
                else:
                    config['Architecture']["Models"][key]["Head"][
                        'out_channels'] = char_num
        elif config['Architecture']['Head'][
                'name'] == 'MultiHead':  # for multi head loss
            out_channels_list = {}
            if config['PostProcess']['name'] == 'SARLabelDecode':
                char_num = char_num - 2
            out_channels_list['CTCLabelDecode'] = char_num
            out_channels_list['SARLabelDecode'] = char_num + 2
            config['Architecture']['Head'][
                'out_channels_list'] = out_channels_list
        else:  # base rec model
            config['Architecture']["Head"]['out_channels'] = char_num
            
    model = build_model(config['Architecture'])

    load_model(config, model)

    # create data ops
    transforms = []
    for op in config['Eval']['dataset']['transforms']:
        op_name = list(op)[0]
        if 'Label' in op_name:
            continue
        elif op_name in ['RecResizeImg']:
            op[op_name]['infer_mode'] = True
        elif op_name == 'KeepKeys':
            if config['Architecture']['algorithm'] == "SRN":
                op[op_name]['keep_keys'] = [
                    'image', 'encoder_word_pos', 'gsrm_word_pos',
                    'gsrm_slf_attn_bias1', 'gsrm_slf_attn_bias2'
                ]
            elif config['Architecture']['algorithm'] == "SAR":
                op[op_name]['keep_keys'] = ['image', 'valid_ratio']
            elif config['Architecture']['algorithm'] == "RobustScanner":
                op[op_name][
                    'keep_keys'] = ['image', 'valid_ratio', 'word_positons']
            else:
                op[op_name]['keep_keys'] = ['image']
        transforms.append(op)
    global_config['infer_mode'] = True
    ops = create_operators(transforms, global_config)

    model.eval()
    _transcriptions = []
    for idx, _ocr_box in enumerate(ocr_boxes):
        _first_im = first_det_filename
        src_im = cv2.imread(_first_im)
        
        text_boxes = _ocr_box['points']
        crop_im = get_rotate_crop_image(src_im, text_boxes)
                
        save_img_path = os.path.join(saveroot, f'second_det_result_{idx}.png')
        cv2.imwrite(save_img_path, crop_im)
                
        with open(save_img_path, 'rb') as f:
            img = f.read()
            data = {'image': img}

        batch = transform(data, ops)
        if config['Architecture']['algorithm'] == "SRN":
            encoder_word_pos_list = np.expand_dims(batch[1], axis=0)
            gsrm_word_pos_list = np.expand_dims(batch[2], axis=0)
            gsrm_slf_attn_bias1_list = np.expand_dims(batch[3], axis=0)
            gsrm_slf_attn_bias2_list = np.expand_dims(batch[4], axis=0)

            others = [
                paddle.to_tensor(encoder_word_pos_list),
                paddle.to_tensor(gsrm_word_pos_list),
                paddle.to_tensor(gsrm_slf_attn_bias1_list),
                paddle.to_tensor(gsrm_slf_attn_bias2_list)
            ]
        if config['Architecture']['algorithm'] == "SAR":
            valid_ratio = np.expand_dims(batch[-1], axis=0)
            img_metas = [paddle.to_tensor(valid_ratio)]
        if config['Architecture']['algorithm'] == "RobustScanner":
            valid_ratio = np.expand_dims(batch[1], axis=0)
            word_positons = np.expand_dims(batch[2], axis=0)
            img_metas = [
                paddle.to_tensor(valid_ratio),
                paddle.to_tensor(word_positons),
            ]
        if config['Architecture']['algorithm'] == "CAN":
            image_mask = paddle.ones(
                (np.expand_dims(
                    batch[0], axis=0).shape), dtype='float32')
            label = paddle.ones((1, 36), dtype='int64')
        images = np.expand_dims(batch[0], axis=0)
        images = paddle.to_tensor(images)
        if config['Architecture']['algorithm'] == "SRN":
            preds = model(images, others)
        elif config['Architecture']['algorithm'] == "SAR":
            preds = model(images, img_metas)
        elif config['Architecture']['algorithm'] == "RobustScanner":
            preds = model(images, img_metas)
        elif config['Architecture']['algorithm'] == "CAN":
            preds = model([images, image_mask, label])
        else:
            preds = model(images)
        post_result = post_process_class(preds)
        info = None
        if isinstance(post_result, dict):
            rec_info = dict()
            for key in post_result:
                if len(post_result[key][0]) >= 2:
                    rec_info[key] = {
                        "label": post_result[key][0][0],
                        "score": float(post_result[key][0][1]),
                    }
            info = json.dumps(rec_info, ensure_ascii=False)
        elif isinstance(post_result, list) and isinstance(post_result[0],
                                                        int):
            # for RFLearning CNT branch 
            info = str(post_result[0])
        else:
            if len(post_result[0]) >= 2:
                info = post_result[0][0] + "\t" + str(post_result[0][1])

        _ocr_box['transcription'] = post_result[0][0]
        _transcriptions.append(post_result[0][0])
    _transcriptions_str = '##'.join(_transcriptions)
        
    del model
    paddle.device.cuda.empty_cache()
    return _transcriptions_str