from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import json
import paddle
from tqdm import tqdm

from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
from ppocr.utils.utility import get_image_file_list
from ppocr.utils.logging import get_logger
import tools.program as load_config


def draw_det_res(dt_boxes, config, img, img_name, save_path):
    import cv2
    src_im = img
    for box in dt_boxes:
        box = np.array(box).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, os.path.basename(img_name))
    cv2.imwrite(save_path, src_im)
    logger.info("The detected Image saved in {}".format(save_path))
    
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

def run_gt_det_api(save_res_path, gtfile, imroot):
    assert save_res_path[-5:] == '.json'
    datas = open(gtfile, 'r').readlines()
    det_box_json_list = []
    for line in tqdm(datas):
        img_path, label = line.strip().split('\t')
        
        det_box_json = {}
        label = json.loads(label)
        dt_boxes_json = []
        for idx, anno in enumerate(label):
            text_boxes = anno['points']
            
            tmp_json = {"transcription": ""}
            tmp_json["points"] = np.array(text_boxes).tolist()
            dt_boxes_json.append(tmp_json)
        det_box_json['filename'] = os.path.join(imroot, img_path)
        det_box_json['boxes'] = dt_boxes_json
        det_box_json_list.append(det_box_json)
        
    with open(save_res_path, 'w') as f:
        json.dump(det_box_json_list, f)
    return save_res_path

def run_first_det_api(imlist, save_res_path, logger):
    assert save_res_path[-5:] == '.json'
    _config = 'configs/det/det_r50_db++_icdar15.yml'
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
    if not os.path.exists(os.path.dirname(save_res_path)):
        os.makedirs(os.path.dirname(save_res_path))
    
    model.eval()
    logger.info("infer first det...")
    det_box_json_list = []
    for im in tqdm(imlist):
        logger.info("infer_img: {}".format(im))
        with open(im, 'rb') as f:
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
        det_box_json['filename'] = im
        det_box_json['boxes'] = dt_boxes_json
        det_box_json_list.append(det_box_json)
        
    logger.info("infer first det complete~~~~~~")
    del model
    paddle.device.cuda.empty_cache()
    with open(save_res_path, 'w') as f:
        json.dump(det_box_json_list, f)
    return save_res_path

def post_process_first_det_api(save_res_path, save_img_dir, logger):
    post_save_res_path = save_res_path.replace('.json', '_post.json')
    docs = json.load(open(save_res_path))
    logger.info("infer postprocess...")
    for doc in tqdm(docs):
        im = doc['filename']
        first_boxes = doc['boxes']
        
        basename = os.path.splitext(os.path.basename(im))[0]
        src_im = cv2.imread(im)
        for idx, _box in enumerate(first_boxes):
            text_boxes = _box['points']
            try:
                crop_im = get_rotate_crop_image(src_im, text_boxes)
            except:
                import ipdb;ipdb.set_trace()
            
            save_img_path = os.path.join(save_img_dir, f'{basename}_{idx}.png')
            cv2.imwrite(save_img_path, crop_im)
            _box['crop_filename'] = save_img_path
    logger.info("infer postprocess complete~~~~~~~")
    with open(post_save_res_path, 'w') as f:
        json.dump(docs, f)
    return post_save_res_path

def ocr_det_api(post_save_res_path, logger):
    ocr_det_save_res_path = post_save_res_path.replace('.json', '_ocr_det.json')
    docs = json.load(open(post_save_res_path))
    
    # init
    _config = 'configs/det/ch_PP-OCRv3/cn_PP-OCRv3_det_cml.yml'
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
    logger.info("infer ocr det...")
    det_box_json_list = []
    for doc in tqdm(docs):
        first_boxes = doc['boxes']
        for _box in first_boxes:
            crop_im = _box['crop_filename']
            logger.info("infer_img: {}".format(crop_im))
            
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

            _box['ocr_boxes'] = dt_boxes_json
            
    logger.info("infer ocr det complete~~~~~~~~")
    del model
    paddle.device.cuda.empty_cache()
    with open(ocr_det_save_res_path, 'w') as f:
        json.dump(docs, f)
    return ocr_det_save_res_path

def ocr_rec_api(ocr_det_save_res_path, save_img_dir, logger):
    ocr_rec_save_res_path = ocr_det_save_res_path.replace('.json', '_ocr_rec.json')
    docs = json.load(open(ocr_det_save_res_path))
    
    # init
    _config = 'configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml'
    config = load_config(_config)
    global_config = config['Global']
    # build post process
    post_process_class = build_post_process(config['PostProcess'],
                                            global_config)
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
    logger.info("infer ocr rec...")
    for doc in tqdm(docs):
        _first_boxes = doc['boxes']
        _transcriptions_list = []
        for _first_box in _first_boxes:
            _first_im = _first_box['crop_filename']
            basename = os.path.splitext(os.path.basename(_first_im))[0]
            src_im = cv2.imread(_first_im)
            
            _ocr_boxes = _first_box['ocr_boxes']
            _transcriptions = []
            for idx, _ocr_box in enumerate(_ocr_boxes):
                text_boxes = _ocr_box['points']
                crop_im = get_rotate_crop_image(src_im, text_boxes)
                
                save_img_path = os.path.join(save_img_dir, f'{basename}_{idx}.png')
                cv2.imwrite(save_img_path, crop_im)
                _ocr_box['crop_filename'] = save_img_path
                
                logger.info(f"infer_img: {_first_im} at {idx}th box")
                
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

                _ocr_box['transcription'] = info
                _transcriptions.append(post_result[0][0])
            _transcriptions_str = '##'.join(_transcriptions)
            _transcriptions_list.append(_transcriptions_str)
            _first_box['transcription'] = _transcriptions_str
        _transcriptions_list_str = '\n'.join(_transcriptions_list)
        doc['transcription'] = _transcriptions_list_str
        
    logger.info("infer ocr rec complete~~~~~")
    del model
    paddle.device.cuda.empty_cache()
    with open(ocr_rec_save_res_path, 'w') as f:
        json.dump(docs, f)
    return ocr_rec_save_res_path

@paddle.no_grad()
def main():
    logger = get_logger(log_file=None)
    imroot = ''
    root = 'pipeline_test'
    save_res_path = os.path.join(root, 'first_det.json')
    save_img_dir_post = os.path.join(root, 'post_img')
    save_img_dir_ocr = os.path.join(root, 'ocr_img')
    os.makedirs(save_img_dir_post, exist_ok=True)
    os.makedirs(save_img_dir_ocr, exist_ok=True)
    imlist = get_image_file_list(imroot)
    # first det
    save_res_path = run_first_det_api(imlist, save_res_path, logger)
    save_res_path = run_gt_det_api(save_res_path, '/2023_06_organize_data/Label.txt', '/2023_06/organize_data')
    # postprocess
    save_res_path = post_process_first_det_api(save_res_path, save_img_dir_post, logger)
    save_res_path = save_res_path.replace('.json', '_post.json')
    # ocr det
    save_res_path = ocr_det_api(save_res_path, logger)
    save_res_path = save_res_path.replace('.json', '_ocr_det.json')
    # ocr rec
    save_res_path = ocr_rec_api(save_res_path, save_img_dir_ocr, logger)
    save_res_path = save_res_path.replace('.json', '_ocr_rec.json')
    
    logger.info("success!")


if __name__ == '__main__':
    main()
