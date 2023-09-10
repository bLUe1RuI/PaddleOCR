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
from paddle import inference
import yaml
import platform

from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
from ppocr.utils.utility import get_image_file_list
from ppocr.utils.logging import get_logger

def get_infer_gpuid():
    sysstr = platform.system()
    if sysstr == "Windows":
        return 0

    if not paddle.fluid.core.is_compiled_with_rocm():
        cmd = "env | grep CUDA_VISIBLE_DEVICES"
    else:
        cmd = "env | grep HIP_VISIBLE_DEVICES"
    env_cuda = os.popen(cmd).readlines()
    if len(env_cuda) == 0:
        return 0
    else:
        gpu_id = env_cuda[0].strip().split("=")[1]
        return int(gpu_id[0])
    
def get_output_tensors(mode, predictor, rec_algorithm='SVTR_LCNet'):
    output_names = predictor.get_output_names()
    output_tensors = []
    if mode == "rec" and rec_algorithm in ["CRNN", "SVTR_LCNet"]:
        output_name = 'softmax_0.tmp_0'
        if output_name in output_names:
            return [predictor.get_output_handle(output_name)]
        else:
            for output_name in output_names:
                output_tensor = predictor.get_output_handle(output_name)
                output_tensors.append(output_tensor)
    else:
        for output_name in output_names:
            output_tensor = predictor.get_output_handle(output_name)
            output_tensors.append(output_tensor)
    return output_tensors

def create_predictor(model_dir, mode, use_gpu=False, use_onnx=False, use_tensorrt=False,
                     use_npu=False, use_xpu=False, enable_mkldnn=False, cpu_threads=10, use_precision='fp32'):
    if use_onnx:
        import onnxruntime as ort
        model_file_path = model_dir
        if not os.path.exists(model_file_path):
            raise ValueError("not find model file path {}".format(
                model_file_path))
        sess = ort.InferenceSession(model_file_path)
        return sess, sess.get_inputs()[0], None, None

    else:
        file_names = ['model', 'inference']
        for file_name in file_names:
            model_file_path = '{}/{}.pdmodel'.format(model_dir, file_name)
            params_file_path = '{}/{}.pdiparams'.format(model_dir, file_name)
            if os.path.exists(model_file_path) and os.path.exists(
                    params_file_path):
                break
        if not os.path.exists(model_file_path):
            raise ValueError(
                "not find model.pdmodel or inference.pdmodel in {}".format(
                    model_dir))
        if not os.path.exists(params_file_path):
            raise ValueError(
                "not find model.pdiparams or inference.pdiparams in {}".format(
                    model_dir))

        config = inference.Config(model_file_path, params_file_path)


        if use_precision == "fp16" and use_tensorrt:
            precision = inference.PrecisionType.Half
        elif use_precision == "int8":
            precision = inference.PrecisionType.Int8
        else:
            precision = inference.PrecisionType.Float32

        if use_gpu:
            gpu_id = get_infer_gpuid()
            if gpu_id is None:
                print(
                    "GPU is not found in current device by nvidia-smi. Please check your device or ignore it if run on jetson."
                )
            config.enable_use_gpu('500', 0)
            if use_tensorrt:
                config.enable_tensorrt_engine(
                    workspace_size=1 << 30,
                    precision_mode=precision,
                    max_batch_size=10,
                    min_subgraph_size=15,  # skip the minmum trt subgraph
                    use_calib_mode=False)

                # collect shape
                trt_shape_f = os.path.join(model_dir,
                                           f"{mode}_trt_dynamic_shape.txt")

                if not os.path.exists(trt_shape_f):
                    config.collect_shape_range_info(trt_shape_f)
                    print(
                        f"collect dynamic shape info into : {trt_shape_f}")
                try:
                    config.enable_tuned_tensorrt_dynamic_shape(trt_shape_f,
                                                               True)
                except Exception as E:
                    print.info(E)
                    print.info("Please keep your paddlepaddle-gpu >= 2.3.0!")

        elif use_npu:
            config.enable_custom_device("npu")
        elif use_xpu:
            config.enable_xpu(10 * 1024 * 1024)
        else:
            config.disable_gpu()
            if enable_mkldnn:
                # cache 10 different shapes for mkldnn to avoid memory leak
                config.set_mkldnn_cache_capacity(10)
                config.enable_mkldnn()
                if use_precision == "fp16":
                    config.enable_mkldnn_bfloat16()
                if cpu_threads is not None:
                    config.set_cpu_math_library_num_threads(cpu_threads)
                else:
                    # default cpu threads as 10
                    config.set_cpu_math_library_num_threads(10)
        # enable memory optim
        config.enable_memory_optim()
        config.disable_glog_info()
        config.delete_pass("conv_transpose_eltwiseadd_bn_fuse_pass")
        config.delete_pass("matmul_transpose_reshape_fuse_pass")
        if mode == 're':
            config.delete_pass("simplify_with_basic_ops_pass")
        if mode == 'table':
            config.delete_pass("fc_fuse_pass")  # not supported for table
        config.switch_use_feed_fetch_ops(False)
        config.switch_ir_optim(True)

        # create predictor
        predictor = inference.create_predictor(config)
        input_names = predictor.get_input_names()
        if mode in ['ser', 're']:
            input_tensor = []
            for name in input_names:
                input_tensor.append(predictor.get_input_handle(name))
        else:
            for name in input_names:
                input_tensor = predictor.get_input_handle(name)
        output_tensors = get_output_tensors(mode, predictor)
        return predictor, input_tensor, output_tensors, config

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

def run_first_det_api(imgname, det_algorithm='DB++', use_onnx=False):
    _config = '../configs/det/det_r50_db++_icdar15.yml'
    config = load_config(_config)
    global_config = config['Global']
    # build model
    # model = build_model(config['Architecture'])
    # load_model(config, model)
    # predictor, input_tensor, output_tensors, _ = create_predictor('infer/det_r50_icdar15', 'det')
    predictor, input_tensor, output_tensors, _ = create_predictor('infer/det_r50_icdar15_bs4x2_model0831', 'det')
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
    
    # model.eval()

    with open(imgname, 'rb') as f:
        img = f.read()
        data = {'image': img}
    batch = transform(data, ops)
    
    images = np.expand_dims(batch[0], axis=0)
    shape_list = np.expand_dims(batch[1], axis=0)
    # images = paddle.to_tensor(images)
    # preds = model(images)
    images = images.copy()
    if use_onnx:
        input_dict = {}
        input_dict[input_tensor.name] = images
        outputs = predictor.run(output_tensors, input_dict)
    else:
        input_tensor.copy_from_cpu(images)
        predictor.run()
        outputs = []
        for output_tensor in output_tensors:
            output = output_tensor.copy_to_cpu()
            outputs.append(output)
    preds = {}
    if det_algorithm == "EAST":
        preds['f_geo'] = outputs[0]
        preds['f_score'] = outputs[1]
    elif det_algorithm == 'SAST':
        preds['f_border'] = outputs[0]
        preds['f_score'] = outputs[1]
        preds['f_tco'] = outputs[2]
        preds['f_tvo'] = outputs[3]
    elif det_algorithm in ['DB', 'PSE', 'DB++']:
        preds['maps'] = outputs[0]
    elif det_algorithm == 'FCE':
        for i, output in enumerate(outputs):
            preds['level_{}'.format(i)] = output
    elif det_algorithm == "CT":
        preds['maps'] = outputs[0]
        preds['score'] = outputs[1]
    else:
        raise NotImplementedError
    
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

    del predictor
    del input_tensor
    del output_tensors
    # paddle.device.cuda.empty_cache()
    return det_box_json

def post_process_first_det_api(idx, image, det_box, saveroot):
    save_img_path = osp.join(saveroot, f'first_det_result_{idx}.png')
    text_boxes = det_box['points']
    try:
        crop_im = get_rotate_crop_image(image, text_boxes)
    except:
        raise ValueError(f"error rotate crop")
            
    cv2.imwrite(save_img_path, crop_im)

def ocr_det_api(idx, saveroot, det_algorithm='DB', use_onnx=False):
    first_det_filename = osp.join(saveroot, f'first_det_result_{idx}.png')
    # init
    _config = '../configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml'
    config = load_config(_config)
    global_config = config['Global']
    # build model
    # model = build_model(config['Architecture'])
    # load_model(config, model)
    # predictor, input_tensor, output_tensors, _ = create_predictor('infer/ch_PP-OCR_v3_det/Student', 'det')
    predictor, input_tensor, output_tensors, _ = create_predictor('infer/ch_PP-OCR_v3_det_add_09data/Student', 'det')
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
    
    # model.eval()
    det_box_json_list = []

    crop_im = first_det_filename
            
    with open(crop_im, 'rb') as f:
        img = f.read()
        data = {'image': img}
    batch = transform(data, ops)

    images = np.expand_dims(batch[0], axis=0)
    shape_list = np.expand_dims(batch[1], axis=0)
    # images = paddle.to_tensor(images)
    # preds = model(images)
    images = images.copy()
    if use_onnx:
        input_dict = {}
        input_dict[input_tensor.name] = images
        outputs = predictor.run(output_tensors, input_dict)
    else:
        input_tensor.copy_from_cpu(images)
        predictor.run()
        outputs = []
        for output_tensor in output_tensors:
            output = output_tensor.copy_to_cpu()
            outputs.append(output)
    preds = {}
    if det_algorithm == "EAST":
        preds['f_geo'] = outputs[0]
        preds['f_score'] = outputs[1]
    elif det_algorithm == 'SAST':
        preds['f_border'] = outputs[0]
        preds['f_score'] = outputs[1]
        preds['f_tco'] = outputs[2]
        preds['f_tvo'] = outputs[3]
    elif det_algorithm in ['DB', 'PSE', 'DB++']:
        preds['maps'] = outputs[0]
    elif det_algorithm == 'FCE':
        for i, output in enumerate(outputs):
            preds['level_{}'.format(i)] = output
    elif det_algorithm == "CT":
        preds['maps'] = outputs[0]
        preds['score'] = outputs[1]
    else:
        raise NotImplementedError
    preds_dict  = {'Student': preds}
    post_result = post_process_class(preds_dict, shape_list)

    boxes = post_result['Student'][0]['points']
    dt_boxes_json = []
    # write result
    for box in boxes:
        tmp_json = {"transcription": ""}
        tmp_json["points"] = np.array(box).tolist()
        dt_boxes_json.append(tmp_json)
            
    # del model
    # paddle.device.cuda.empty_cache()
    del predictor
    del input_tensor
    del output_tensors
    return dt_boxes_json, first_det_filename

def ocr_rec_api(ocr_boxes, saveroot, use_onnx=False):
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
            
    # model = build_model(config['Architecture'])
    # load_model(config, model)
    # predictor, input_tensor, output_tensors, _ = create_predictor('infer/v3_en_mobile', 'rec')
    predictor, input_tensor, output_tensors, _ = create_predictor('infer/v3_en_mobile_add_09data', 'rec')

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

    # model.eval()
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
        # images = paddle.to_tensor(images)
        if config['Architecture']['algorithm'] == "SRN":
            # preds = model(images, others)
            if use_onnx:
                input_dict = {}
                input_dict[input_tensor.name] = images
                outputs = predictor.run(output_tensors,
                                                input_dict)
                preds = {"predict": outputs[2]}
            else:
                input_names = predictor.get_input_names()
                inputs = [images, encoder_word_pos_list, gsrm_word_pos_list, gsrm_slf_attn_bias1_list, gsrm_slf_attn_bias2_list]
                for i in range(len(input_names)):
                    input_tensor = predictor.get_input_handle(
                        input_names[i])
                    input_tensor.copy_from_cpu(inputs[i])
                predictor.run()
                outputs = []
                for output_tensor in output_tensors:
                    output = output_tensor.copy_to_cpu()
                    outputs.append(output)
                preds = {"predict": outputs[2]}
        elif config['Architecture']['algorithm'] == "SAR":
            # preds = model(images, img_metas)
            valid_ratios = np.concatenate(valid_ratios)
            inputs = [
                images,
                np.array(
                    [valid_ratios], dtype=np.float32),
            ]
            if use_onnx:
                input_dict = {}
                input_dict[input_tensor.name] = images
                outputs = predictor.run(output_tensors,
                                                input_dict)
                preds = outputs[0]
            else:
                input_names = predictor.get_input_names()
                for i in range(len(input_names)):
                    input_tensor = predictor.get_input_handle(
                        input_names[i])
                    input_tensor.copy_from_cpu(inputs[i])
                predictor.run()
                outputs = []
                for output_tensor in output_tensors:
                    output = output_tensor.copy_to_cpu()
                    outputs.append(output)
                preds = outputs[0]
        elif config['Architecture']['algorithm'] == "RobustScanner":
            # preds = model(images, img_metas)
            valid_ratios = np.concatenate(valid_ratios)
            word_positions_list = np.concatenate(word_positions_list)
            inputs = [images, valid_ratios, word_positions_list]

            if use_onnx:
                input_dict = {}
                input_dict[input_tensor.name] = images
                outputs = predictor.run(output_tensors,
                                                input_dict)
                preds = outputs[0]
            else:
                input_names = predictor.get_input_names()
                for i in range(len(input_names)):
                    input_tensor = predictor.get_input_handle(
                        input_names[i])
                    input_tensor.copy_from_cpu(inputs[i])
                predictor.run()
                outputs = []
                for output_tensor in output_tensors:
                    output = output_tensor.copy_to_cpu()
                    outputs.append(output)
                preds = outputs[0]
        elif config['Architecture']['algorithm'] == "CAN":
            # preds = model([images, image_mask, label])
            norm_image_mask = np.ones(images.shape, dtype='float32')
            word_label = np.ones([1, 36], dtype='int64')
            norm_img_mask_batch = []
            word_label_list = []
            norm_img_mask_batch.append(norm_image_mask)
            word_label_list.append(word_label)
            word_label_list = np.concatenate(word_label_list)
            
            norm_img_mask_batch = np.concatenate(norm_img_mask_batch)
            word_label_list = np.concatenate(word_label_list)
            inputs = [images, norm_img_mask_batch, word_label_list]
            if use_onnx:
                input_dict = {}
                input_dict[input_tensor.name] = images
                outputs = predictor.run(output_tensors,
                                                input_dict)
                preds = outputs
            else:
                input_names = predictor.get_input_names()
                input_tensor = []
                for i in range(len(input_names)):
                    input_tensor_i = predictor.get_input_handle(
                        input_names[i])
                    input_tensor_i.copy_from_cpu(inputs[i])
                    input_tensor.append(input_tensor_i)
                input_tensor = input_tensor
                predictor.run()
                outputs = []
                for output_tensor in output_tensors:
                    output = output_tensor.copy_to_cpu()
                    outputs.append(output)
                preds = outputs
        else:
            # preds = model(images)
            if use_onnx:
                input_dict = {}
                input_dict[input_tensor.name] = images
                outputs = predictor.run(output_tensors,
                                                input_dict)
                preds = outputs[0]
            else:
                input_tensor.copy_from_cpu(images)
                predictor.run()
                outputs = []
                for output_tensor in output_tensors:
                    output = output_tensor.copy_to_cpu()
                    outputs.append(output)
                if len(outputs) != 1:
                    preds = outputs
                else:
                    preds = outputs[0]
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
        _ocr_box['ocr_score'] = post_result[0][1]
        _transcriptions.append(post_result[0][0])
    _transcriptions_str = '##'.join(_transcriptions)
        
    # del model
    # paddle.device.cuda.empty_cache()
    del predictor
    del input_tensor
    del output_tensors
    return _transcriptions_str

MADE_IDS = [
            '101', '103', '105', '106', '107', '112', '114', '121', '131', '132', '161', '181', '183',
            '184', '191', '043', '047', '051', '053', '054', '055', '056', '057', '058', '059', '060',
            '061', '062', '063', '202', '043', '053', '057', '059', '060', '063', '105', '114', '183',
            '184', 'XRY'
            ]
AIXS_IDS = ['160077', '08396', '11282', '25991', '19132', '28480', '66542', '06312', '47873', '11879', '14110', '38647', '42396', '60738', '76338', '19980', '72496', '19596', '60162', '80895', '190668', '03975', '11133', '07826', '85861', '39768', '39923', '01810', '14471', '17449', '79436', '27412', '158360', '176843', '12014', '233794', '15579', '36743', '01936', '08400', '14263', '08507', '13245', '65920', '19764', '207564', '56702', '81315', '00699', '72899', '58018', '11437', '22540', '84694', '69360', '22166', '607173', '57175', '53030', '19013', '02108', '13523', '82086', '51041', '10726', '07784', '71097', '08884', '00080', '10288', '80506', '101080', '162316', '04736', '194526', '16317', '68192', '21953', '51421', '23255', '01797', '17002', '52869', '142013', '15279', '65620', '12912', '146277', '24254', '17284', '57879']
MADE_TIME = ['1008', '0910', '1101', '0510', '0810', '0912', '0704', '0808', '1010', '1107', '1112', '1310', '0907', '1301', '1111', '1304', '1411', '1003', '1006', '0603', '1211', '0809', '0812', '1009', '1004', '1205', '1305', '0902', '1405', '1202', '1306', '0612', '1106', '1302', '0905', '1702', '1208', '1104', '0806', '1103', '1308', '0901', '1011', '0904', '1206', '1804', '1102', '1612', '1209', '1303', '1207', '1204', '0908']
MADE_CLASS = ['RE2B', 'RE2A', 'RD2']

def lcs(str1, str2):
    m = len(str1)
    n = len(str2)
    dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

def modify_rec_result(text, ref_texts, limit_len=None):
    tmp_item = None
    for ref_text in ref_texts:
        joint_length = lcs(text, ref_text)
        if limit_len is None:
            _len = 0
        elif limit_len == -1:
            _len = len(ref_text)
        else:
            _len = limit_len
        if joint_length >= _len:
            _recall = joint_length / len(ref_text)
            _precision = joint_length / len(text)
            if tmp_item:
                if _recall + _precision <= tmp_item['_recall'] + tmp_item['_precision']:
                    continue
            tmp_item = {
                'ori_text': text,
                'ref_text': ref_text,
                '_recall': _recall,
                '_precision': _precision
            }
    if tmp_item is None:
        return None
    return tmp_item

def post_process(ocr_boxes):
    # get MADE_IDS and AIXS_IDS
    sort_boxes = sorted(ocr_boxes, key=lambda x: -len(x['transcription']))
    made_axis_line = None
    for _box in sort_boxes[:2]:
        ocr_text = _box['transcription']
        if 'W' in ocr_text:
            continue
        if ocr_text[3] == ' ':
            made_axis_line = ocr_text
            break
    if made_axis_line is None:
        for _box in sort_boxes[:2]:
            ocr_text = _box['transcription']
            _made_id = ocr_text[:3]
            if modify_rec_result(_made_id, MADE_IDS, -1) is not None:
                made_axis_line = ocr_text[:3] + ' ' + ocr_text[3:]
                break
    tmp_sort_boxes = sorted(ocr_boxes, key=lambda x: (x['points'][1][1] + x['points'][2][1])/2)
    if made_axis_line is None:
        # get the middle text
        _ref_made_id_item = None
        for _box in tmp_sort_boxes[1:-2]:
            ocr_text = _box['transcription'].replace(' ', '')
            _item = modify_rec_result(ocr_text, MADE_IDS, 2)
            if _item is None:
                continue
            if _ref_made_id_item:
                if _item['_recall'] + _item['_precision'] < _ref_made_id_item['_recall'] + _ref_made_id_item['_precision']:
                    continue
            _ref_made_id_item = _item
        if _ref_made_id_item is None:
            return False, 'None match made_ids'
        _ref_made_id = _ref_made_id_item['ref_text']
        
        # _ref_aixs_id_item = None
        # for _box in tmp_sort_boxes[1:-2]:
        #     ocr_text = _box['transcription'].replace(' ', '')
        #     _item = modify_rec_result(ocr_text, AIXS_IDS, 4)
        #     if _item is None:
        #         continue
        #     if _ref_aixs_id_item:
        #         if _item['_recall'] + _item['_precision'] < _ref_aixs_id_item['_recall'] + _ref_aixs_id_item['_precision']:
        #             continue
        #     _ref_aixs_id_item = _item
        # if _ref_aixs_id_item is None:
        #     return False, 'None match aixs_ids'
        # _ref_axis_id = _ref_aixs_id_item['ref_text']
        tmp_txts = [x['transcription'].replace(' ', '') for x in tmp_sort_boxes[1:-2]]
        _ref_aixs_id_item = list(filter(lambda x: len(x) == 5 or len(x) == 6, tmp_txts))
        if len(_ref_aixs_id_item) == 0:
            return False, 'None match aixs_ids'
        _ref_axis_id = _ref_aixs_id_item[0]   
    else:
        # _made_id, _axis_id = made_axis_line.split(' ')
        # _ref_made_id = modify_rec_result(_made_id, MADE_IDS)['ref_text']
        # _ref_axis_id = modify_rec_result(_axis_id, AIXS_IDS)['ref_text']
        _ref_made_id = made_axis_line[:3].replace(' ', '')
        _ref_axis_id = made_axis_line[4:].replace(' ', '')
    
    # get MADE_TIME
    # _ref_made_time_item = None
    # for _box in sort_boxes[2:]:
    #     ocr_text = _box['transcription'].replace(' ', '')
    #     _item = modify_rec_result(ocr_text, MADE_TIME, 3)
    #     if _item is None:
    #         continue
    #     if _ref_made_time_item:
    #         if _item['_recall'] + _item['_precision'] < _ref_made_time_item['_recall'] + _ref_made_time_item['_precision']:
    #             continue
    #     _ref_made_time_item = _item
    # if _ref_made_time_item is None:
    #     return False, 'None match made_time'
    # _ref_made_time = _ref_made_time_item['ref_text']
    tmp_txts = [x['transcription'].replace(' ', '') for x in tmp_sort_boxes[2:-1]]
    _ref_made_time_item = list(filter(lambda x: len(x)==4, tmp_txts))
    if len(_ref_made_time_item) == 0:
        _ref_made_time_item = list(filter(lambda x: len(x)==5, tmp_txts))
        if len(_ref_made_time_item) == 0:
            return False, 'None match made_time'
    _ref_made_time = _ref_made_time_item[-1][:4]
    # get MADE_CLASS
    _ref_made_class_item = None
    for _box in sort_boxes[2:]:
        ocr_text = _box['transcription'].replace(' ', '')
        _item = modify_rec_result(ocr_text, MADE_CLASS, 3)
        if _item is None:
            continue
        if _ref_made_class_item:
            if _item['_recall'] + _item['_precision'] <= _ref_made_class_item['_recall'] + _ref_made_class_item['_precision']:
                continue
        _ref_made_class_item = _item
    if _ref_made_class_item is None:
        return False, 'None match made_class'
    _ref_made_class = _ref_made_class_item['ref_text']
    return True, ','.join([_ref_made_id, _ref_axis_id, _ref_made_time, _ref_made_class])

def single_test_ocr_det_rec():
    saveroot = 'pic_results/cache_file/2023-9-2/tmp-result/'
    idx = 1
    import ipdb;ipdb.set_trace()
    second_dt_boxes_json, first_det_filename = ocr_det_api(idx, saveroot, det_algorithm='DB', use_onnx=False)
    # shutil.copyfile(first_det_filename, osp.join(saveroot, 'first_det_result.png'))
    _transcriptions_str = ocr_rec_api(second_dt_boxes_json, saveroot)
    
if __name__ == '__main__':
    single_test_ocr_det_rec()