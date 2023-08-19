import os
import yaml
import glob
import json
from pathlib import Path

import cv2
import numpy as np
import math
import paddle
from paddle.inference import Config
from paddle.inference import create_predictor
from PIL import Image

from circle_detect_preprocess import preprocess, Resize, NormalizeImage, Permute, PadStride, LetterBoxResize, WarpAffine, Pad, decode_image
from circle_detect_visualize import visualize_box_mask
from circle_detect_utils import Timer, get_current_memory_mb, multiclass_nms, coco_clsid2catid

# Global dictionary
SUPPORT_MODELS = {
    'YOLO', 'PPYOLOE', 'RCNN', 'SSD', 'Face', 'FCOS', 'SOLOv2', 'TTFNet',
    'S2ANet', 'JDE', 'FairMOT', 'DeepSORT', 'GFL', 'PicoDet', 'CenterNet',
    'TOOD', 'RetinaNet', 'StrongBaseline', 'STGCN', 'YOLOX', 'YOLOF', 'PPHGNet',
    'PPLCNet', 'DETR', 'CenterTrack'
}

TUNED_TRT_DYNAMIC_MODELS = {'DETR'}

def load_predictor(model_dir,
                   arch,
                   run_mode='paddle',
                   batch_size=1,
                   device='CPU',
                   min_subgraph_size=3,
                   use_dynamic_shape=False,
                   trt_min_shape=1,
                   trt_max_shape=1280,
                   trt_opt_shape=640,
                   trt_calib_mode=False,
                   cpu_threads=1,
                   enable_mkldnn=False,
                   enable_mkldnn_bfloat16=False,
                   delete_shuffle_pass=False,
                   tuned_trt_shape_file="shape_range_info.pbtxt"):
    """set AnalysisConfig, generate AnalysisPredictor
    Args:
        model_dir (str): root path of __model__ and __params__
        device (str): Choose the device you want to run, it can be: CPU/GPU/XPU, default is CPU
        run_mode (str): mode of running(paddle/trt_fp32/trt_fp16/trt_int8)
        use_dynamic_shape (bool): use dynamic shape or not
        trt_min_shape (int): min shape for dynamic shape in trt
        trt_max_shape (int): max shape for dynamic shape in trt
        trt_opt_shape (int): opt shape for dynamic shape in trt
        trt_calib_mode (bool): If the model is produced by TRT offline quantitative
            calibration, trt_calib_mode need to set True
        delete_shuffle_pass (bool): whether to remove shuffle_channel_detect_pass in TensorRT. 
                                    Used by action model.
    Returns:
        predictor (PaddlePredictor): AnalysisPredictor
    Raises:
        ValueError: predict by TensorRT need device == 'GPU'.
    """
    if device != 'GPU' and run_mode != 'paddle':
        raise ValueError(
            "Predict by TensorRT mode: {}, expect device=='GPU', but device == {}"
            .format(run_mode, device))
    infer_model = os.path.join(model_dir, 'model.pdmodel')
    infer_params = os.path.join(model_dir, 'model.pdiparams')
    if not os.path.exists(infer_model):
        infer_model = os.path.join(model_dir, 'inference.pdmodel')
        infer_params = os.path.join(model_dir, 'inference.pdiparams')
        if not os.path.exists(infer_model):
            raise ValueError(
                "Cannot find any inference model in dir: {},".format(model_dir))
    config = Config(infer_model, infer_params)
    if device == 'GPU':
        # initial GPU memory(M), device ID
        config.enable_use_gpu(200, 0)
        # optimize graph and fuse op
        config.switch_ir_optim(True)
    elif device == 'XPU':
        if config.lite_engine_enabled():
            config.enable_lite_engine()
        config.enable_xpu(10 * 1024 * 1024)
    elif device == 'NPU':
        if config.lite_engine_enabled():
            config.enable_lite_engine()
        config.enable_custom_device('npu')
    else:
        config.disable_gpu()
        config.set_cpu_math_library_num_threads(cpu_threads)
        if enable_mkldnn:
            try:
                # cache 10 different shapes for mkldnn to avoid memory leak
                config.set_mkldnn_cache_capacity(10)
                config.enable_mkldnn()
                if enable_mkldnn_bfloat16:
                    config.enable_mkldnn_bfloat16()
            except Exception as e:
                print(
                    "The current environment does not support `mkldnn`, so disable mkldnn."
                )
                pass

    precision_map = {
        'trt_int8': Config.Precision.Int8,
        'trt_fp32': Config.Precision.Float32,
        'trt_fp16': Config.Precision.Half
    }
    if run_mode in precision_map.keys():
        if arch in TUNED_TRT_DYNAMIC_MODELS:
            config.collect_shape_range_info(tuned_trt_shape_file)
        config.enable_tensorrt_engine(
            workspace_size=(1 << 25) * batch_size,
            max_batch_size=batch_size,
            min_subgraph_size=min_subgraph_size,
            precision_mode=precision_map[run_mode],
            use_static=False,
            use_calib_mode=trt_calib_mode)
        if arch in TUNED_TRT_DYNAMIC_MODELS:
            config.enable_tuned_tensorrt_dynamic_shape(tuned_trt_shape_file,
                                                       True)

        if use_dynamic_shape:
            min_input_shape = {
                'image': [batch_size, 3, trt_min_shape, trt_min_shape],
                'scale_factor': [batch_size, 2]
            }
            max_input_shape = {
                'image': [batch_size, 3, trt_max_shape, trt_max_shape],
                'scale_factor': [batch_size, 2]
            }
            opt_input_shape = {
                'image': [batch_size, 3, trt_opt_shape, trt_opt_shape],
                'scale_factor': [batch_size, 2]
            }
            config.set_trt_dynamic_shape_info(min_input_shape, max_input_shape,
                                              opt_input_shape)
            print('trt set dynamic shape done!')

    # disable print log when predict
    config.disable_glog_info()
    # enable shared memory
    config.enable_memory_optim()
    # disable feed, fetch OP, needed by zero_copy_run
    config.switch_use_feed_fetch_ops(False)
    if delete_shuffle_pass:
        config.delete_pass("shuffle_channel_detect_pass")
    predictor = create_predictor(config)
    return predictor, config

def create_inputs(imgs, im_info):
    """generate input for different model type
    Args:
        imgs (list(numpy)): list of images (np.ndarray)
        im_info (list(dict)): list of image info
    Returns:
        inputs (dict): input of model
    """
    inputs = {}

    im_shape = []
    scale_factor = []
    if len(imgs) == 1:
        inputs['image'] = np.array((imgs[0], )).astype('float32')
        inputs['im_shape'] = np.array(
            (im_info[0]['im_shape'], )).astype('float32')
        inputs['scale_factor'] = np.array(
            (im_info[0]['scale_factor'], )).astype('float32')
        return inputs

    for e in im_info:
        im_shape.append(np.array((e['im_shape'], )).astype('float32'))
        scale_factor.append(np.array((e['scale_factor'], )).astype('float32'))

    inputs['im_shape'] = np.concatenate(im_shape, axis=0)
    inputs['scale_factor'] = np.concatenate(scale_factor, axis=0)

    imgs_shape = [[e.shape[1], e.shape[2]] for e in imgs]
    max_shape_h = max([e[0] for e in imgs_shape])
    max_shape_w = max([e[1] for e in imgs_shape])
    padding_imgs = []
    for img in imgs:
        im_c, im_h, im_w = img.shape[:]
        padding_im = np.zeros(
            (im_c, max_shape_h, max_shape_w), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = img
        padding_imgs.append(padding_im)
    inputs['image'] = np.stack(padding_imgs, axis=0)
    return inputs

def get_test_images(infer_dir, infer_img):
    """
    Get image path list in TEST mode
    """
    assert infer_img is not None or infer_dir is not None, \
        "--image_file or --image_dir should be set"
    assert infer_img is None or os.path.isfile(infer_img), \
            "{} is not a file".format(infer_img)
    assert infer_dir is None or os.path.isdir(infer_dir), \
            "{} is not a directory".format(infer_dir)

    # infer_img has a higher priority
    if infer_img and os.path.isfile(infer_img):
        return [infer_img]

    images = set()
    infer_dir = os.path.abspath(infer_dir)
    assert os.path.isdir(infer_dir), \
        "infer_dir {} is not a directory".format(infer_dir)
    exts = ['jpg', 'jpeg', 'png', 'bmp']
    exts += [ext.upper() for ext in exts]
    for ext in exts:
        images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
    images = list(images)

    assert len(images) > 0, "no image found in {}".format(infer_dir)
    print("Found {} inference images in total.".format(len(images)))

    return images

def visualize(image_list, result, labels, output_dir='output/', threshold=0.5):
    # visualize the predict result
    start_idx = 0
    for idx, image_file in enumerate(image_list):
        im_bboxes_num = result['boxes_num'][idx]
        im_results = {}
        if 'boxes' in result:
            im_results['boxes'] = result['boxes'][start_idx:start_idx +
                                                  im_bboxes_num, :]
        if 'masks' in result:
            im_results['masks'] = result['masks'][start_idx:start_idx +
                                                  im_bboxes_num, :]
        if 'segm' in result:
            im_results['segm'] = result['segm'][start_idx:start_idx +
                                                im_bboxes_num, :]
        if 'label' in result:
            im_results['label'] = result['label'][start_idx:start_idx +
                                                  im_bboxes_num]
        if 'score' in result:
            im_results['score'] = result['score'][start_idx:start_idx +
                                                  im_bboxes_num]

        start_idx += im_bboxes_num
        im = visualize_box_mask(
            image_file, im_results, labels, threshold=threshold)
        img_name = os.path.split(image_file)[-1]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        out_path = os.path.join(output_dir, img_name)
        im.save(out_path, quality=95)
        print("save result to: " + out_path)

class PredictConfig():
    """set config of preprocess, postprocess and visualize
    Args:
        model_dir (str): root path of model.yml
    """

    def __init__(self, model_dir):
        # parsing Yaml config for Preprocess
        deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        self.check_model(yml_conf)
        self.arch = yml_conf['arch']
        self.preprocess_infos = yml_conf['Preprocess']
        self.min_subgraph_size = yml_conf['min_subgraph_size']
        self.labels = yml_conf['label_list']
        self.mask = False
        self.use_dynamic_shape = yml_conf['use_dynamic_shape']
        if 'mask' in yml_conf:
            self.mask = yml_conf['mask']
        self.tracker = None
        if 'tracker' in yml_conf:
            self.tracker = yml_conf['tracker']
        if 'NMS' in yml_conf:
            self.nms = yml_conf['NMS']
        if 'fpn_stride' in yml_conf:
            self.fpn_stride = yml_conf['fpn_stride']
        if self.arch == 'RCNN' and yml_conf.get('export_onnx', False):
            print(
                'The RCNN export model is used for ONNX and it only supports batch_size = 1'
            )
        self.print_config()

    def check_model(self, yml_conf):
        """
        Raises:
            ValueError: loaded model not in supported model type 
        """
        for support_model in SUPPORT_MODELS:
            if support_model in yml_conf['arch']:
                return True
        raise ValueError("Unsupported arch: {}, expect {}".format(yml_conf[
            'arch'], SUPPORT_MODELS))

    def print_config(self):
        print('-----------  Model Configuration -----------')
        print('%s: %s' % ('Model Arch', self.arch))
        print('%s: ' % ('Transform Order'))
        for op_info in self.preprocess_infos:
            print('--%s: %s' % ('transform op', op_info['type']))
        print('--------------------------------------------')


class Detector(object):
    """
    Args:
        pred_config (object): config of model, defined by `Config(model_dir)`
        model_dir (str): root path of model.pdiparams, model.pdmodel and infer_cfg.yml
        device (str): Choose the device you want to run, it can be: CPU/GPU/XPU, default is CPU
        run_mode (str): mode of running(paddle/trt_fp32/trt_fp16)
        batch_size (int): size of pre batch in inference
        trt_min_shape (int): min shape for dynamic shape in trt
        trt_max_shape (int): max shape for dynamic shape in trt
        trt_opt_shape (int): opt shape for dynamic shape in trt
        trt_calib_mode (bool): If the model is produced by TRT offline quantitative
            calibration, trt_calib_mode need to set True
        cpu_threads (int): cpu threads
        enable_mkldnn (bool): whether to open MKLDNN
        enable_mkldnn_bfloat16 (bool): whether to turn on mkldnn bfloat16
        output_dir (str): The path of output
        threshold (float): The threshold of score for visualization
        delete_shuffle_pass (bool): whether to remove shuffle_channel_detect_pass in TensorRT. 
                                    Used by action model.
    """

    def __init__(self,
                 model_dir,
                 device='CPU',
                 run_mode='paddle',
                 batch_size=1,
                 trt_min_shape=1,
                 trt_max_shape=1280,
                 trt_opt_shape=640,
                 trt_calib_mode=False,
                 cpu_threads=1,
                 enable_mkldnn=False,
                 enable_mkldnn_bfloat16=False,
                 output_dir='output',
                 threshold=0.5,
                 delete_shuffle_pass=False):
        self.pred_config = self.set_config(model_dir)
        self.predictor, self.config = load_predictor(
            model_dir,
            self.pred_config.arch,
            run_mode=run_mode,
            batch_size=batch_size,
            min_subgraph_size=self.pred_config.min_subgraph_size,
            device=device,
            use_dynamic_shape=self.pred_config.use_dynamic_shape,
            trt_min_shape=trt_min_shape,
            trt_max_shape=trt_max_shape,
            trt_opt_shape=trt_opt_shape,
            trt_calib_mode=trt_calib_mode,
            cpu_threads=cpu_threads,
            enable_mkldnn=enable_mkldnn,
            enable_mkldnn_bfloat16=enable_mkldnn_bfloat16,
            delete_shuffle_pass=delete_shuffle_pass)
        self.det_times = Timer()
        self.cpu_mem, self.gpu_mem, self.gpu_util = 0, 0, 0
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.threshold = threshold

    def set_config(self, model_dir):
        return PredictConfig(model_dir)

    def preprocess(self, image_list):
        preprocess_ops = []
        for op_info in self.pred_config.preprocess_infos:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop('type')
            preprocess_ops.append(eval(op_type)(**new_op_info))

        input_im_lst = []
        input_im_info_lst = []
        for im_path in image_list:
            im, im_info = preprocess(im_path, preprocess_ops)
            input_im_lst.append(im)
            input_im_info_lst.append(im_info)
        inputs = create_inputs(input_im_lst, input_im_info_lst)
        input_names = self.predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            if input_names[i] == 'x':
                input_tensor.copy_from_cpu(inputs['image'])
            else:
                input_tensor.copy_from_cpu(inputs[input_names[i]])

        return inputs

    def postprocess(self, inputs, result):
        # postprocess output of predictor
        np_boxes_num = result['boxes_num']
        assert isinstance(np_boxes_num, np.ndarray), \
            '`np_boxes_num` should be a `numpy.ndarray`'

        result = {k: v for k, v in result.items() if v is not None}
        return result

    def filter_box(self, result, threshold):
        np_boxes_num = result['boxes_num']
        boxes = result['boxes']
        start_idx = 0
        filter_boxes = []
        filter_num = []
        for i in range(len(np_boxes_num)):
            boxes_num = np_boxes_num[i]
            boxes_i = boxes[start_idx:start_idx + boxes_num, :]
            idx = boxes_i[:, 1] > threshold
            filter_boxes_i = boxes_i[idx, :]
            filter_boxes.append(filter_boxes_i)
            filter_num.append(filter_boxes_i.shape[0])
            start_idx += boxes_num
        boxes = np.concatenate(filter_boxes)
        filter_num = np.array(filter_num)
        filter_res = {'boxes': boxes, 'boxes_num': filter_num}
        return filter_res

    def predict(self, repeats=1, run_benchmark=False):
        '''
        Args:
            repeats (int): repeats number for prediction
        Returns:
            result (dict): include 'boxes': np.ndarray: shape:[N,6], N: number of box,
                            matix element:[class, score, x_min, y_min, x_max, y_max]
                            MaskRCNN's result include 'masks': np.ndarray:
                            shape: [N, im_h, im_w]
        '''
        # model prediction
        np_boxes_num, np_boxes, np_masks = np.array([0]), None, None

        if run_benchmark:
            for i in range(repeats):
                self.predictor.run()
                paddle.device.cuda.synchronize()
            result = dict(
                boxes=np_boxes, masks=np_masks, boxes_num=np_boxes_num)
            return result

        for i in range(repeats):
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            boxes_tensor = self.predictor.get_output_handle(output_names[0])
            np_boxes = boxes_tensor.copy_to_cpu()
            if len(output_names) == 1:
                # some exported model can not get tensor 'bbox_num' 
                np_boxes_num = np.array([len(np_boxes)])
            else:
                boxes_num = self.predictor.get_output_handle(output_names[1])
                np_boxes_num = boxes_num.copy_to_cpu()
            if self.pred_config.mask:
                masks_tensor = self.predictor.get_output_handle(output_names[2])
                np_masks = masks_tensor.copy_to_cpu()
        result = dict(boxes=np_boxes, masks=np_masks, boxes_num=np_boxes_num)
        return result

    def merge_batch_result(self, batch_result):
        if len(batch_result) == 1:
            return batch_result[0]
        res_key = batch_result[0].keys()
        results = {k: [] for k in res_key}
        for res in batch_result:
            for k, v in res.items():
                results[k].append(v)
        for k, v in results.items():
            if k not in ['masks', 'segm']:
                results[k] = np.concatenate(v)
        return results

    def get_timer(self):
        return self.det_times

    def predict_image_slice(self,
                            img_list,
                            slice_size=[640, 640],
                            overlap_ratio=[0.25, 0.25],
                            combine_method='nms',
                            match_threshold=0.6,
                            match_metric='ios',
                            run_benchmark=False,
                            repeats=1,
                            visual=True,
                            save_results=False):
        # slice infer only support bs=1
        results = []
        try:
            import sahi
            from sahi.slicing import slice_image
        except Exception as e:
            print(
                'sahi not found, plaese install sahi. '
                'for example: `pip install sahi`, see https://github.com/obss/sahi.'
            )
            raise e
        num_classes = len(self.pred_config.labels)
        for i in range(len(img_list)):
            ori_image = img_list[i]
            slice_image_result = sahi.slicing.slice_image(
                image=ori_image,
                slice_height=slice_size[0],
                slice_width=slice_size[1],
                overlap_height_ratio=overlap_ratio[0],
                overlap_width_ratio=overlap_ratio[1])
            sub_img_num = len(slice_image_result)
            merged_bboxs = []
            print('slice to {} sub_samples.', sub_img_num)

            batch_image_list = [
                slice_image_result.images[_ind] for _ind in range(sub_img_num)
            ]
            if run_benchmark:
                # preprocess
                inputs = self.preprocess(batch_image_list)  # warmup
                self.det_times.preprocess_time_s.start()
                inputs = self.preprocess(batch_image_list)
                self.det_times.preprocess_time_s.end()

                # model prediction
                result = self.predict(repeats=50, run_benchmark=True)  # warmup
                self.det_times.inference_time_s.start()
                result = self.predict(repeats=repeats, run_benchmark=True)
                self.det_times.inference_time_s.end(repeats=repeats)

                # postprocess
                result_warmup = self.postprocess(inputs, result)  # warmup
                self.det_times.postprocess_time_s.start()
                result = self.postprocess(inputs, result)
                self.det_times.postprocess_time_s.end()
                self.det_times.img_num += 1

                cm, gm, gu = get_current_memory_mb()
                self.cpu_mem += cm
                self.gpu_mem += gm
                self.gpu_util += gu
            else:
                # preprocess
                self.det_times.preprocess_time_s.start()
                inputs = self.preprocess(batch_image_list)
                self.det_times.preprocess_time_s.end()

                # model prediction
                self.det_times.inference_time_s.start()
                result = self.predict()
                self.det_times.inference_time_s.end()

                # postprocess
                self.det_times.postprocess_time_s.start()
                result = self.postprocess(inputs, result)
                self.det_times.postprocess_time_s.end()
                self.det_times.img_num += 1

            st, ed = 0, result['boxes_num'][0]  # start_index, end_index
            for _ind in range(sub_img_num):
                boxes_num = result['boxes_num'][_ind]
                ed = st + boxes_num
                shift_amount = slice_image_result.starting_pixels[_ind]
                result['boxes'][st:ed][:, 2:4] = result['boxes'][
                    st:ed][:, 2:4] + shift_amount
                result['boxes'][st:ed][:, 4:6] = result['boxes'][
                    st:ed][:, 4:6] + shift_amount
                merged_bboxs.append(result['boxes'][st:ed])
                st = ed

            merged_results = {'boxes': []}
            if combine_method == 'nms':
                final_boxes = multiclass_nms(
                    np.concatenate(merged_bboxs), num_classes, match_threshold,
                    match_metric)
                merged_results['boxes'] = np.concatenate(final_boxes)
            elif combine_method == 'concat':
                merged_results['boxes'] = np.concatenate(merged_bboxs)
            else:
                raise ValueError(
                    "Now only support 'nms' or 'concat' to fuse detection results."
                )
            merged_results['boxes_num'] = np.array(
                [len(merged_results['boxes'])], dtype=np.int32)

            if visual:
                visualize(
                    [ori_image],  # should be list
                    merged_results,
                    self.pred_config.labels,
                    output_dir=self.output_dir,
                    threshold=self.threshold)

            results.append(merged_results)
            print('Test iter {}'.format(i))

        results = self.merge_batch_result(results)
        if save_results:
            Path(self.output_dir).mkdir(exist_ok=True)
            self.save_coco_results(
                img_list, results, use_coco_category=FLAGS.use_coco_category)
        return results

    def predict_image(self,
                      image_list,
                      run_benchmark=False,
                      repeats=1,
                      visual=True,
                      save_results=False):
        batch_loop_cnt = math.ceil(float(len(image_list)) / self.batch_size)
        results = []
        for i in range(batch_loop_cnt):
            start_index = i * self.batch_size
            end_index = min((i + 1) * self.batch_size, len(image_list))
            batch_image_list = image_list[start_index:end_index]
            if run_benchmark:
                # preprocess
                inputs = self.preprocess(batch_image_list)  # warmup
                self.det_times.preprocess_time_s.start()
                inputs = self.preprocess(batch_image_list)
                self.det_times.preprocess_time_s.end()

                # model prediction
                result = self.predict(repeats=50, run_benchmark=True)  # warmup
                self.det_times.inference_time_s.start()
                result = self.predict(repeats=repeats, run_benchmark=True)
                self.det_times.inference_time_s.end(repeats=repeats)

                # postprocess
                result_warmup = self.postprocess(inputs, result)  # warmup
                self.det_times.postprocess_time_s.start()
                result = self.postprocess(inputs, result)
                self.det_times.postprocess_time_s.end()
                self.det_times.img_num += len(batch_image_list)

                cm, gm, gu = get_current_memory_mb()
                self.cpu_mem += cm
                self.gpu_mem += gm
                self.gpu_util += gu
            else:
                # preprocess
                self.det_times.preprocess_time_s.start()
                inputs = self.preprocess(batch_image_list)
                self.det_times.preprocess_time_s.end()

                # model prediction
                self.det_times.inference_time_s.start()
                result = self.predict()
                self.det_times.inference_time_s.end()

                # postprocess
                self.det_times.postprocess_time_s.start()
                result = self.postprocess(inputs, result)
                self.det_times.postprocess_time_s.end()
                self.det_times.img_num += len(batch_image_list)

                if visual:
                    visualize(
                        batch_image_list,
                        result,
                        self.pred_config.labels,
                        output_dir=self.output_dir,
                        threshold=self.threshold)
            results.append(result)
            print('Test iter {}'.format(i))
        results = self.merge_batch_result(results)
        if save_results:
            Path(self.output_dir).mkdir(exist_ok=True)
            self.save_coco_results(
                image_list, results, use_coco_category=FLAGS.use_coco_category)
        return results

    def predict_video(self, video_file, camera_id):
        video_out_name = 'output.mp4'
        if camera_id != -1:
            capture = cv2.VideoCapture(camera_id)
        else:
            capture = cv2.VideoCapture(video_file)
            video_out_name = os.path.split(video_file)[-1]
        # Get Video info : resolution, fps, frame count
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print("fps: %d, frame_count: %d" % (fps, frame_count))

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        out_path = os.path.join(self.output_dir, video_out_name)
        fourcc = cv2.VideoWriter_fourcc(* 'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        index = 1
        while (1):
            ret, frame = capture.read()
            if not ret:
                break
            print('detect frame: %d' % (index))
            index += 1
            results = self.predict_image([frame[:, :, ::-1]], visual=False)

            im = visualize_box_mask(
                frame,
                results,
                self.pred_config.labels,
                threshold=self.threshold)
            im = np.array(im)
            writer.write(im)
            if camera_id != -1:
                cv2.imshow('Mask Detection', im)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        writer.release()

    def save_coco_results(self, image_list, results, use_coco_category=False):
        bbox_results = []
        mask_results = []
        idx = 0
        print("Start saving coco json files...")
        for i, box_num in enumerate(results['boxes_num']):
            file_name = os.path.split(image_list[i])[-1]
            if use_coco_category:
                img_id = int(os.path.splitext(file_name)[0])
            else:
                img_id = i

            if 'boxes' in results:
                boxes = results['boxes'][idx:idx + box_num].tolist()
                bbox_results.extend([{
                    'image_id': img_id,
                    'category_id': coco_clsid2catid[int(box[0])] \
                        if use_coco_category else int(box[0]),
                    'file_name': file_name,
                    'bbox': [box[2], box[3], box[4] - box[2],
                         box[5] - box[3]],  # xyxy -> xywh
                    'score': box[1]} for box in boxes])

            if 'masks' in results:
                import pycocotools.mask as mask_util

                boxes = results['boxes'][idx:idx + box_num].tolist()
                masks = results['masks'][i][:box_num].astype(np.uint8)
                seg_res = []
                for box, mask in zip(boxes, masks):
                    rle = mask_util.encode(
                        np.array(
                            mask[:, :, None], dtype=np.uint8, order="F"))[0]
                    if 'counts' in rle:
                        rle['counts'] = rle['counts'].decode("utf8")
                    seg_res.append({
                        'image_id': img_id,
                        'category_id': coco_clsid2catid[int(box[0])] \
                        if use_coco_category else int(box[0]),
                        'file_name': file_name,
                        'segmentation': rle,
                        'score': box[1]})
                mask_results.extend(seg_res)

            idx += box_num

        if bbox_results:
            bbox_file = os.path.join(self.output_dir, "bbox.json")
            with open(bbox_file, 'w') as f:
                json.dump(bbox_results, f)
            print(f"The bbox result is saved to {bbox_file}")
        if mask_results:
            mask_file = os.path.join(self.output_dir, "mask.json")
            with open(mask_file, 'w') as f:
                json.dump(mask_results, f)
            print(f"The mask result is saved to {mask_file}")

def main(image_file):
    model_dir = 'infer/ppyolo_mbv3_small_voc'
    deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
    with open(deploy_file) as f:
        yml_conf = yaml.safe_load(f)
    arch = yml_conf['arch']
    detector_func = 'Detector'

    detector = eval(detector_func)(
        model_dir,
        device='cpu',
        run_mode='paddle',
        batch_size=1,
        trt_min_shape=1,
        trt_max_shape=1280,
        trt_opt_shape=640,
        trt_calib_mode=False,
        cpu_threads=1,
        enable_mkldnn=False,
        enable_mkldnn_bfloat16=False,
        threshold=0.6,
        output_dir='output_debug')

    # predict from video file or camera video stream
    # if FLAGS.video_file is not None or FLAGS.camera_id != -1:
    #     detector.predict_video(FLAGS.video_file, FLAGS.camera_id)
    # else:
    # predict from image
    img_list = get_test_images(None, image_file)
    # if FLAGS.slice_infer:
    #     detector.predict_image_slice(
    #         img_list,
    #         FLAGS.slice_size,
    #         FLAGS.overlap_ratio,
    #         FLAGS.combine_method,
    #         FLAGS.match_threshold,
    #         FLAGS.match_metric,
    #         visual=FLAGS.save_images,
    #         save_results=FLAGS.save_results)
    # else:
    results = detector.predict_image(
                img_list,
                False,
                repeats=1,
                visual=False,
                save_results=False)
    return results
    # if not FLAGS.run_benchmark:
    #     detector.det_times.info(average=True)
    # else:
    #     mode = FLAGS.run_mode
    #     model_dir = FLAGS.model_dir
    #     model_info = {
    #         'model_name': model_dir.strip('/').split('/')[-1],
    #         'precision': mode.split('_')[-1]
    #     }
    #     bench_log(detector, img_list, model_info, name='DET')
    
def contain_box(box, ref_box, ref_thre=0.7):
    x1, y1, x2, y2 = int(box[2]), int(box[3]), int(box[4]), int(box[5])
    ref_x1, ref_y1, ref_x2, ref_y2 = int(ref_box[2]), int(ref_box[3]), int(ref_box[4]), int(ref_box[5])
    
    inter_x1 = max(x1, ref_x1)
    inter_y1 = max(y1, ref_y1)
    inter_x2 = min(x2, ref_x2)
    inter_y2 = min(y2, ref_y2)
    if inter_x1 > inter_x2 or inter_y1 > inter_y2:
        return False
    inter_area = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
    area = (ref_x2 - ref_x1 + 1) * (ref_y2 - ref_y1 + 1)
    ratio = inter_area / area
    if ratio >= ref_thre:
        return True
    else:
        return False
    
def circle_detect_api(imgfile):
    results = main(imgfile)
    box_results = results['boxes'].tolist()
    box_results = list(filter(lambda x: x[1] > 0.6, box_results))
    big_box_results = []
    small_box_results = []
    for i in range(len(box_results)):
        _contain = False
        for j in range(len(box_results)):
            if i == j:
                continue
            if contain_box(box_results[i], box_results[j]):
                _contain = True
                break
        if _contain:
            big_box_results.append(box_results[i])
        else:
            small_box_results.append(box_results[i])
    # big box sorted by area, select biggest one
    if len(big_box_results) == 0:
        return None, None
    big_box_results = sorted(big_box_results, key=lambda x: -((x[4] - x[2] + 1) * (x[5] - x[3] + 1)))
    big_box_result = big_box_results[0]
    # small box sorted by score, select top3
    if len(small_box_results) == 0:
        return None, None
    small_box_results = sorted(small_box_results, key=lambda x: -x[1])
    small_box_results = small_box_results[:4]
    
    big_circle_centx = (big_box_result[2] + big_box_result[4]) / 2
    big_circle_centy = (big_box_result[3] + big_box_result[5]) / 2
    big_circle_radius = (big_box_result[4] - big_box_result[2] + 1) / 2
    big_circle = ((big_circle_centx, big_circle_centy), big_circle_radius, 5)
    
    small_circles = []
    for small_box_result in small_box_results:
        small_circle_centx = (small_box_result[2] + small_box_result[4]) / 2
        small_circle_centy = (small_box_result[3] + small_box_result[5]) / 2
        small_circle_radius = (small_box_result[4] - small_box_result[2] + 1) / 2
        small_circle = ((small_circle_centx, small_circle_centy), small_circle_radius, 5)
        small_circles.append(small_circle)
    return big_circle, small_circles

if __name__ == '__main__':
    # imgfile = '../../../2023_06_organize_data/seal_VOC/JPEGImages/08_25_22_690.png'
    # imroot = '../../../2023_06_organize_data/seal_VOC/JPEGImages'
    # imroot = 'D:/work/project/庄林/图像/空识别图像/2023_07_17'
    imroot = 'D:/work/project/庄林/图像/字符识别图像/2023_07_17'
    imlist = os.listdir(imroot)
    imlist = list(filter(lambda x: x[-4:] == '.bmp', imlist))
    for im in imlist:
        imroot = '../../../2023_06_organize_data/2023_06_12'
        im = '09_05_56_607.png'
        imgfile = os.path.join(imroot, im)
        results = main(imgfile)
        
        # image = cv2.imread(imgfile)
        image = np.array(Image.open(os.path.join(imroot, im)).convert('RGB'))
        height, width, _ = image.shape
        for res in results['boxes']:
            if res[1] > 0.6:
                x1 = int(res[2])
                y1 = int(res[3])
                x2 = int(res[4])
                y2 = int(res[5])
                cent_x = (x1 + x2)//2
                cent_y = (y1 + y2)//2
                print(cent_x, cent_y, (x2 - x1)/2)
                cv2.rectangle(image, (x1, y1), (x2, y2), [0, 255, 0], thickness=6)
                cv2.circle(image, (cent_x, cent_y), 6, [0, 255, 0], thickness=6)
        tgt_width = 512
        tgt_height = int(tgt_width / width * height)
        tgt_image = cv2.resize(image, (tgt_width, tgt_height))
        cv2.imshow('image', tgt_image)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    