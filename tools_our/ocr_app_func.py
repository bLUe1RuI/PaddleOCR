import argparse
import os
import os.path as osp
import time

import cv2
import gradio as gr
from gradio import processing_utils
import numpy as np
import json
import PIL
import requests
import sys
import serial
import snap7
from snap7 import util
import struct
import shutil

from app_debug.camera_utils.camera import Camera
from ocr_api_infermodel import run_first_det_api, post_process_first_det_api, ocr_det_api, ocr_rec_api, post_process
from util_class import AxisDetect

def clear_state_button_func(state):
    state = []
    return state, state

"""serial func____________________________"""

def check_serial_input(serial_port, serial_baudrate, serial_bytesize, serial_parity,
                     serial_stopbits, serial_xonxoff, serial_rtscts, serial_dsrdtr,
                     serial_read_timeout, serial_write_timeout):
    if not isinstance(serial_port, str):
        serial_port = str(serial_port)
    if not isinstance(serial_baudrate, str):
        serial_baudrate = int(serial_baudrate)
    if isinstance(serial_bytesize, str):
        if serial_bytesize == 'FIVEBITS':
            serial_bytesize = serial.FIVEBITS
        elif serial_bytesize == 'SIXBITS':
            serial_bytesize = serial.SIXBITS
        elif serial_bytesize == 'SEVENBITS':
            serial_bytesize = serial.SEVENBITS
        else:
            serial_bytesize = serial.EIGHTBITS
    else:
        raise ValueError(f'error type serial bytesize {serial_bytesize}')
    
    if isinstance(serial_parity, str): # '', 'PARITY_MARK', 'PARITY_SPACE'
        if serial_parity == 'PARITY_NONE':
            serial_parity = serial.PARITY_NONE
        elif serial_parity == 'PARITY_EVEN':
            serial_parity = serial.PARITY_EVEN
        elif serial_parity == 'PARITY_ODD':
            serial_parity = serial.PARITY_ODD
        elif serial_parity == 'PARITY_MARK':
            serial_parity = serial.PARITY_MARK
        elif serial_parity == 'PARITY_SPACE':
            serial_parity = serial.PARITY_SPACE
        else:
            serial_parity = serial.PARITY_NONE
    else:
        raise ValueError(f'error type serial parity {serial_parity}')
    
    if isinstance(serial_stopbits, str):
        if serial_stopbits == 'STOPBITS_ONE':
            serial_stopbits = serial.STOPBITS_ONE
        elif serial_stopbits == 'STOPBITS_ONE_POINT_FIVE':
            serial_stopbits = serial.STOPBITS_ONE_POINT_FIVE
        elif serial_stopbits == 'STOPBITS_TWO':
            serial_stopbits = serial.STOPBITS_TWO
        else:
            serial_stopbits = serial.STOPBITS_ONE
    else:
        raise ValueError(f'error type serial stopbits {serial_stopbits}')
    
    if not isinstance(serial_xonxoff, bool):
        raise ValueError(f'error type serial xonxoff {serial_xonxoff}')
    if not isinstance(serial_rtscts, bool):
        raise ValueError(f'error type serial xonxoff {serial_rtscts}')
    if not isinstance(serial_dsrdtr, bool):
        raise ValueError(f'error type serial xonxoff {serial_dsrdtr}')
    
    if serial_read_timeout == 'None' or serial_read_timeout == '':
        serial_read_timeout = None
    elif not isinstance(serial_read_timeout, float):
        serial_read_timeout = float(serial_read_timeout)
        
    if serial_write_timeout == 'None' or serial_write_timeout =='':
        serial_write_timeout = None
    elif not isinstance(serial_write_timeout, float):
        serial_write_timeout = float(serial_write_timeout)
        
    return dict(serial_port=serial_port, 
                serial_baudrate=serial_baudrate, 
                serial_bytesize=serial_bytesize, 
                serial_parity=serial_parity,
                serial_stopbits=serial_stopbits, 
                serial_xonxoff=serial_xonxoff, 
                serial_rtscts=serial_rtscts, 
                serial_dsrdtr=serial_dsrdtr,
                serial_read_timeout=serial_read_timeout, 
                serial_write_timeout=serial_write_timeout)
    
def open_serial_func(state, user_serial, serial_port, serial_baudrate, serial_bytesize, serial_parity,
                     serial_stopbits, serial_xonxoff, serial_rtscts, serial_dsrdtr,
                     serial_read_timeout, serial_write_timeout):
    if user_serial is not None:
        state = [(None, f"串口{serial_port}连接中，请关闭串口后重新连接")]
        return state, state, user_serial
    else:
        _serial_paramers = check_serial_input(serial_port, serial_baudrate, serial_bytesize, serial_parity,
                                              serial_stopbits, serial_xonxoff, serial_rtscts, serial_dsrdtr,
                                              serial_read_timeout, serial_write_timeout)
        user_serial = serial.Serial(port=_serial_paramers['serial_port'],
                                    baudrate=_serial_paramers['serial_baudrate'],
                                    bytesize=_serial_paramers['serial_bytesize'],
                                    parity=_serial_paramers['serial_parity'],
                                    stopbits=_serial_paramers['serial_stopbits'],
                                    xonxoff=_serial_paramers['serial_xonxoff'],
                                    rtscts=_serial_paramers['serial_rtscts'],
                                    dsrdtr=_serial_paramers['serial_dsrdtr'],
                                    timeout=_serial_paramers['serial_read_timeout'],
                                    write_timeout=_serial_paramers['serial_write_timeout'])
        if user_serial.isOpen():
            state += [(None, f"串口{serial_port} 连接成功 success")]
        else:
            state += [(None, f"串口{serial_port} 连接失败")]
        return state, state, user_serial
    
def close_serial_func(state, user_serial):
    if user_serial is None:
        state = [(None, f"串口已为空，已经关闭~~~")]
        return state, state, user_serial
    else:
        user_serial.close()
        if user_serial.isOpen():
            state += [(None, f"串口{user_serial.port} 关闭失败!，请重新尝试")]
            return state, state, user_serial
        else:
            state = [(None, f"串口{user_serial.port} 完成关闭 success")]
            return state, state, user_serial
        
def turn_light_serial_func(state, user_serial, serial_turnlight_cmd):
    if user_serial is None:
        state += [(None, "未存在连接的串口，请先连接串口")]
        return state, state
    
    if ";" in serial_turnlight_cmd:
        cmds = serial_turnlight_cmd.split(";")
    elif "；" in serial_turnlight_cmd:
        cmds = serial_turnlight_cmd.split("；")
    elif "\n" in serial_turnlight_cmd:
        cmds = serial_turnlight_cmd.split("\n")
    else:
        cmds = [serial_turnlight_cmd]
    for cmd in cmds:
        _cmd_len = user_serial.write(cmd.encode("utf-8"))
        state += [(f"发送串口命令{cmd}", f"串口发送成功，共{_cmd_len}字节")]
    return state, state

"""camera func____________________________"""

def open_camera_func(state, user_camera, camera_index, camera_stdcall, camera_active_way):
    camera_index = int(camera_index)
    # 创建相机
    if user_camera is not None:
        if user_camera.camera_index != camera_index:
            state += [(None, f"目前相机-{user_camera.camera_index}-连接中，要切换需先关闭原始连接")]
            return state, state, user_camera
        if user_camera.camera_stdcall != camera_stdcall or user_camera.camera_active_way != camera_active_way:
            state += [(None, f"目前相机-{user_camera.camera_index}-切换抓取图像方式-{user_camera.camera_stdcall}，获取图像流的方式-{user_camera.camera_active_way}")]
            user_camera.camera_stdcall = camera_stdcall
            user_camera.camera_active_way = camera_active_way
    else:
        user_camera = Camera(camera_index, camera_stdcall, camera_active_way)
        for _log in user_camera.all_logs:
            state += [(None, _log)]
    # 根据相机状态进行连接
    if user_camera.cam is None:
        connect_logs = user_camera.connect_camera(user_camera.camera_index)
        for _log in connect_logs:
            state += [(None, _log)]
    if user_camera.cam is None: # 再次判断
        state += [(None, "相机初始化失败，请检查设备连接情况 failed")]
        return state, state, None
    connect_value, connect_logs = user_camera.decide_divice_on_line()
    state += [(None, connect_logs)]
    if not connect_value:
        logs_open_device = user_camera.open_device()
        state += [(None, connect_logs)]
    if not user_camera.decide_divice_on_line()[0]: # 再次判断
        state += [(None, "相机初始化失败，请检查设备连接情况 failed")]
        return state, state, None
    
    return state, state, user_camera

def get_image_func(state, user_camera):
    image, logs_get_image = user_camera.get_image()
    state += [(None, logs_get_image)]
    return  state, state, image

def close_camera_func(state, user_camera):
    if user_camera is None or user_camera.cam is None:
        state = [(None, "设备连接已经关闭")]
        return state, state, None, None 
    _logs = user_camera.close_and_destroy_device()
    state = [(None, _logs)]
    return state, state, None, None

def camera_info_button_func(state, user_camera):
    if user_camera is None:
        all_logs = Camera.get_info()[1]
    else:
        all_logs = user_camera.all_logs
        
    for _log in all_logs:
        state += [(None, _log)]
    return state, state

def camera_connect_info_button_func(state, user_camera):
    if user_camera is None or user_camera.cam is None:
        state += [(None, "暂无连接相机，请先进行相机连接")]
        return state, state
    connect_value, connect_logs = user_camera.decide_divice_on_line()
    state += [(None, connect_logs)]
    return state, state

"""plc func____________________________"""

def open_plc_func(state, user_plc, plc_port, plc_rack, plc_slot):
    if user_plc is not None and user_plc.get_connected():
        state = [(None, f"PLC连接中，若需要连接新设备，请关闭后重新连接")]
        return state, state, user_plc
    else:
        user_plc = snap7.client.Client()
        user_plc.connect(plc_port, plc_rack, plc_slot)
        if user_plc.get_connected():
            state += [(None, f"PLC-prot{plc_port}-rack-{plc_rack}-slot-{plc_slot} connect success")]
        else:
            state += [(None, "PLC 连接失败，请检查设备")]
        return state, state, user_plc
    
def close_plc_func(state, user_plc):
    if user_plc is None or not user_plc.get_connected():
        state = [(None, f"PLC已经关闭连接~~~")]
        return state, state, None
    else:
        user_plc.disconnect()
        if user_plc.get_connected():
            state += [(None, f"PLC关闭连接失败 failed")]
        else:
            state = [(None, f"PLC关闭连接成功 success")]
        return state, state, None
    
def check_plc_input(plc_blocknum, plc_blocknum_start_pos, plc_length):
    # 处理数据类型
    if not isinstance(plc_blocknum, int):
        if isinstance(plc_blocknum, str) and 'DB' in plc_blocknum:
            plc_blocknum = plc_blocknum.replace('DB', '')
        plc_blocknum = int(plc_blocknum)
    if isinstance(plc_blocknum_start_pos, str):
        if '.' in plc_blocknum_start_pos:
            plc_blocknum_start_pos = plc_blocknum_start_pos.split('.')
            assert len(plc_blocknum_start_pos) == 2
            byte_index, bit_index = int(plc_blocknum_start_pos[0]), int(plc_blocknum_start_pos[1])
        else:
            byte_index = int(plc_blocknum_start_pos)
            bit_index = 0
    if not isinstance(plc_length, int):
        plc_length = int(plc_length)
    return dict(
        plc_blocknum=plc_blocknum,
        byte_index=byte_index,
        bit_index=bit_index,
        plc_length=plc_length
    )
    
def read_plc_func(state, user_plc, plc_read_blocknum, plc_read_blocknum_start_pos, plc_read_length, plc_read_dtype):
    if user_plc is None or not user_plc.get_connected():
        state += [(None, "PLC暂未连接，请先连接PLC")]
        return state, state, None
    
    plc_read_param = check_plc_input(plc_read_blocknum, plc_read_blocknum_start_pos, plc_read_length)
    # 读取数据
    data = user_plc.db_read(
        plc_read_param['plc_blocknum'],
        plc_read_param['byte_index'],
        plc_read_param['plc_length']
    )
    if plc_read_dtype == 'bool':
        # data = bool.from_bytes(data, byteorder='big')
        data = util.get_bool(data, plc_read_param['byte_index'], plc_read_param['bit_index'])
    elif plc_read_dtype == 'int':
        data = int.from_bytes(data, byteorder='big')
    elif plc_read_dtype == 'real':
        data = struct.unpack('>f', data)[0]
    elif plc_read_dtype == 'string':
        data = data.decode(encoding="ascii")
    elif plc_read_dtype == 'wstring':
        data = data.decode(encoding="utf-16be")
    else:
        raise ValueError(f"now not support data type{plc_read_dtype}")
    
    state += [(None, f"读取-block{plc_read_blocknum}-start_pos-{plc_read_blocknum_start_pos}-类型{plc_read_dtype}的数据为 {data}")]
    return state, state, data

def write_plc_func(state, user_plc, plc_write_blocknum, plc_write_blocknum_start_pos, plc_write_dtype, plc_write_content):
    if user_plc is None or not user_plc.get_connected():
        state += [(None, "PLC暂未连接，请先连接PLC")]
        return state, state
    
    plc_write_param = check_plc_input(plc_write_blocknum, plc_write_blocknum_start_pos, 0)
    
    if plc_write_dtype == 'bool':
        if plc_write_content == 'True' or plc_write_content == 'TRUE':
            tmpdata = True
        else:
            tmpdata = False
        # data = bool.to_bytes(data, 1, 'big')
        data = bytearray(1)
        util.set_bool(data, plc_write_param['byte_index'], plc_write_param['bit_index'], tmpdata)
    elif plc_write_dtype == 'int':
        if not isinstance(plc_write_content, int):
            plc_write_content = int(plc_write_content)
        data = plc_write_content
        data = int.to_bytes(data, 2, 'big')
    elif plc_write_dtype == 'real':
        if not isinstance(plc_write_content, float):
            plc_write_content = float(plc_write_content)
        data = plc_write_content
        data = struct.pack(">f", data)
    elif plc_write_dtype == 'string':
        if not isinstance(plc_write_content, str):
            plc_write_content = str(plc_write_content)
        data = plc_write_content
        data = int.to_bytes(254, 1, 'big') + int.to_bytes(len(data), 1, 'big') + data.encode(encoding='ascii')
    elif plc_write_dtype == 'wstring':
        if not isinstance(plc_write_content, str):
            plc_write_content = str(plc_write_content)
        data = plc_write_content
        data = int.to_bytes(508, 2, 'big') + int.to_bytes(len(data), 2, 'big') + data.encode(encoding='utf-16be')
    else:
        raise ValueError(f"now not support data type{plc_write_dtype}")
    # 写入数据
    user_plc.do_write(
        plc_write_param['plc_blocknum'],
        plc_write_param['byte_index'],
        data
    )
    state += [(None, f"写入-block{plc_write_blocknum}-start_pos-{plc_write_blocknum_start_pos}-类型{plc_write_dtype}的数据为 {plc_write_content}")]
    return state, state

"""server func____________________________"""

def send_server_func(state, server_ip, server_block, server_content):

    data={}
    data['id'] = int(server_block)
    data['data'] = server_content
    try:
        data = json.dumps(data)
        resp = requests.post(server_ip, data=data)

        jsonResult = json.loads(resp)
        state += [(None, f"数据发送成功\n发送数据为: {data}\n返回值为: {jsonResult}")]
    except Exception as error:
        state += [(None, "数据发送失败")]
    return state, state

"""auto func"""

def auto_recog_func(state, user_serial, user_camera, user_plc, op_select_button, serial_openlight_cmd, serial_clostlight_cmd, camera_root,
                    plc_write_blocknum, plc_write_blocknum_start_pos, plc_write_dtype, plc_state_write_blocknum, plc_state_write_blocknum_start_pos, plc_state_write_dtype,
                    server_ip, ocr_block_text, plc_read_blocknum, plc_read_blocknum_start_pos, plc_read_length, plc_read_dtype):
    # 开灯
    state, _ = turn_light_serial_func(state, user_serial, serial_openlight_cmd)
    # 获取相片, 并保存
    state, _, image = get_image_func(state, user_camera)
    if camera_root == '':
        camera_root = 'pic_results'
    t = time.localtime()
    save_root = osp.join(camera_root, f'{t.tm_year}-{t.tm_mon}-{t.tm_mday}')
    if not osp.exists(save_root):
        os.makedirs(save_root)
    save_file = osp.join(save_root, f'{t.tm_year}-{t.tm_mon}-{t.tm_mday}-{t.tm_hour}-{t.tm_min}-{t.tm_sec}.png')
    cv2.imwrite(save_file, image)
    # 关灯
    state, _ = turn_light_serial_func(state, user_serial, serial_clostlight_cmd)
    # 自动识别
    if op_select_button == "OCR识别":
        res = ocr_detect_func(image, save_file, save_root)
    else:
        res = axis_detect_func(image, save_file, save_root)
    if res.success:
        # 发送结果
        if op_select_button == "OCR识别":
            state, _ = send_server_func(state, server_ip, ocr_block_text, res['ocr_text'])
        else:
            # write plc axis
            state, _ = write_plc_func(state, user_plc, plc_write_blocknum, plc_write_blocknum_start_pos, plc_write_dtype, res['axis_text'])
            # write plc state
            state, _ = write_plc_func(state, user_plc, plc_state_write_blocknum, plc_state_write_blocknum_start_pos, plc_state_write_dtype, res['axis_err_text'])
        # 置位plc
        write_plc_func(state, user_plc, plc_read_blocknum, plc_read_blocknum_start_pos, plc_read_dtype, 'False')
    else:
        failed_info = state['state']
        state += [(None, f"识别失败，未得到想要结果\ninfo: {failed_info}")]
        # TODO 检测是否需要发送字段，说明识别失败
        
    return state, state, res


"""auto recog func"""

def ocr_detect_func(image, imgname, save_root):
    saveroot = osp.join(save_root, 'tmp-result')
    # 进行轴心的检测
    axisDetect = AxisDetect()
    circleaxis, log_info = axisDetect.infer(image)
    if circleaxis is None or len(circleaxis.axis_angle) == 0:
        res = {
            'success': False,
            'ocr_text': 0,
            'state': "failed: first step axis failed"
        }
        return res
    sectors = circleaxis.cal_sector()
    # infer first det
    det_box_json = run_first_det_api(imgname)
    circleaxis.modify_det_box(det_box_json)
    for det_box in det_box_json['boxes']:
        for sector in sectors:
            sector.cal_contail_ocr(det_box)
            
    # infer second det
    det_sectors =  list(filter(lambda x: x.with_det==True, sectors))
    for sector_i, sector in enumerate(det_sectors):
        post_process_first_det_api(sector_i, image, sector.det_box, saveroot)
        second_dt_boxes_json, first_det_filename = ocr_det_api(sector_i, saveroot)
        sector.ocr_boxes = second_dt_boxes_json
        sector.first_det_file = first_det_filename
    
    # inference twice
    ref_transcriptions_str = None
    sorted_det_sectors = sorted(det_sectors, key=lambda x: -len(x.ocr_boxes))
    for core_sector in sorted_det_sectors:
        if len(core_sector.ocr_boxes) >= 4:
            core_sector.is_core = True
            shutil.copyfile(core_sector.first_det_file, osp.join(saveroot, 'first_det_result.png'))
            _transcriptions_str = ocr_rec_api(core_sector.ocr_boxes, saveroot)
            # 正则处理, 根据正则匹配，将不准确的部分重跑
            succ, _transcriptions_str = post_process(core_sector.ocr_boxes)
            if not succ:
                continue
            ref_transcriptions_str = _transcriptions_str
    if ref_transcriptions_str is None:
        res = {
            'success': False,
            'ocr_text': 0,
            'state': f"failed: {_transcriptions_str}"
        }
        return res
    res = {
            'success': True,
            'ocr_text': ref_transcriptions_str,
            'state': "success"
        }
    return res
    
    
def axis_detect_func(image, imgname, save_root):
    saveroot = osp.join(save_root, 'tmp-result')
    # 进行轴心的检测
    axisDetect = AxisDetect()
    circleaxis, log_info = axisDetect.infer(image)
    if circleaxis is None or len(circleaxis.axis_angle) == 0:
        res = {
            'success': False,
            'axis_text': 0,
            'axis_err_text': True,
            'state': "failed: first step axis failed"
        }
        return res
    sectors = circleaxis.cal_sector()
    # infer first det
    det_box_json = run_first_det_api(imgname)
    circleaxis.modify_det_box(det_box_json)
    for det_box in det_box_json['boxes']:
        for sector in sectors:
            sector.cal_contail_ocr(det_box)
    # 选择合适的转角
    select_sectors = list(filter(lambda x: x.with_det==False, sectors))
    if len(select_sectors) == 0:
        res = {
            'success': True,
            'axis_text': 0,
            'axis_err_text': False,
            'state': 'success'
        }
        return res
    select_sector = select_sectors[0]
    res = {
            'success': True,
            'axis_text': select_sector.rotate_angle,
            'axis_err_text': True,
            'state': 'success'
        }
    return res