import argparse
import time
import os
import os.path as osp

import cv2
import json
import gradio as gr
from gradio import processing_utils
import numpy as np
import PIL
import requests
import sys
import serial
import snap7
from snap7 import util
import struct

from app_debug.camera_utils.camera import Camera
from ocr_app_func import (open_serial_func, open_camera_func, open_plc_func, camera_info_button_func, 
                          turn_light_serial_func, get_image_func, send_server_func, write_plc_func, 
                          read_plc_func, auto_recog_func, ocr_detect_func, axis_detect_func)

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=6086, help="only useful when running gradio applications")
parser.add_argument('--debug', action="store_true")
parser.add_argument('--gradio_share', action="store_true")

args = parser.parse_args()

def is_platform_win():
    return sys.platform == "win32"

def read_confg():
    pass

def ocr_api_key_func(ocr_api_key):
    if ocr_api_key == 'zhuanglin':
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), None, gr.update(value='管理员')
    elif ocr_api_key == 'local':
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), None, gr.update(value='用户')
    else:
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), "密码不对，请联系管理员", None

def enable_button_func(ocr_api_key):
    if ocr_api_key == 'zhuanglin':
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), None, gr.update(value='管理员')
    elif ocr_api_key == 'local':
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), None, gr.update(value='用户')
    else:
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), "密码不对，请联系管理员", None

def config_button_func(config_button):
    if config_button == "收起配置":
        return gr.update(visible=False)
    else:
        return gr.update(visible=True)
    
def op_select_button_func(op_select_button):
    if op_select_button == "OCR识别":
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)
    
def init_button_func(state, user_serial, user_camera, user_plc, serial_port, serial_baudrate, serial_bytesize, serial_parity, serial_stopbits,
                     serial_xonxoff, serial_rtscts, serial_dsrdtr, serial_read_timeout, serial_write_timeout,
                     camera_index, camera_stdcall, camera_active_way, plc_port, plc_rack, plc_slot):
    # connect serial
    state, _, user_serial = open_serial_func(state, user_serial, serial_port, serial_baudrate, serial_bytesize, serial_parity,
                                             serial_stopbits, serial_xonxoff, serial_rtscts, serial_dsrdtr, serial_read_timeout,
                                             serial_write_timeout)
    # connect camera
    state, _, user_camera = open_camera_func(state, user_camera, camera_index, camera_stdcall, camera_active_way)
    # connect plc
    state, _, user_plc = open_plc_func(state, user_plc, plc_port, plc_rack, plc_slot)
    
    return state, state, user_serial, user_camera, user_plc

def camera_state_button_func(state, user_camera):
    state, _ = camera_info_button_func(state, user_camera)
    return state, state

def clear_state_button_func(state):
    state = []
    return state, state

def open_camera_func(state, user_camera, camera_root, user_serial, serial_openlight_cmd, serial_clostlight_cmd):
    # open light
    state, _ = turn_light_serial_func(state, user_serial, serial_openlight_cmd)
    # catch image
    state, _, image = get_image_func(state, user_camera)
    # close light
    state, _ = turn_light_serial_func(state, user_serial, serial_clostlight_cmd)
    if camera_root == '':
        camera_root = 'pic_results'
    t = time.localtime()
    save_root = osp.join(camera_root, f'{t.tm_year}-{t.tm_mon}-{t.tm_mday}')
    if not osp.exists(save_root):
        os.makedirs(save_root)
    save_file = osp.join(save_root, f'{t.tm_year}-{t.tm_mon}-{t.tm_mday}-{t.tm_hour}-{t.tm_min}-{t.tm_sec}.png')
    cv2.imwrite(save_file, image)
    return state, state, image

def recog_func(state, image, op_select_button, camera_root, ocr_text, axis_text, axis_err_text):
    if camera_root == '':
        camera_root = 'pic_results'
    t = time.localtime()
    save_root = osp.join(camera_root, 'cache_file', f'{t.tm_year}-{t.tm_mon}-{t.tm_mday}')
    if not osp.exists(save_root):
        os.makedirs(save_root)
    save_file = osp.join(save_root, f'{t.tm_year}-{t.tm_mon}-{t.tm_mday}-{t.tm_hour}-{t.tm_min}-{t.tm_sec}.png')
    cv2.imwrite(save_file, image)
    # 自动识别
    if op_select_button == "OCR识别":
        res = ocr_detect_func(image, save_file, save_root)
    else:
        res = axis_detect_func(image, save_file, save_root)
    if res.success:
        if op_select_button == "OCR识别":
            state += [(None, f"识别结果为: {res['ocr_text']}")]
        else:
            state += [(None, f"识别结果为: {res['axis_text']}")]
    else:
        failed_info = state['state']
        state += [(None, f"识别失败，未得到想要结果\ninfo: {failed_info}")]
    # 返回值
    if op_select_button == "OCR识别":
        return state, state, res['ocr_text'], axis_text, axis_err_text
    else:
        return state, state, ocr_text, res['axis_text'], res['axis_err_text']

def write_op_func(state, user_plc, op_select_button, ocr_text, ocr_block_text, axis_text, axis_err_text,
                  plc_write_blocknum, plc_write_blocknum_start_pos, plc_write_dtype, plc_write_content,
                  plc_state_write_blocknum, plc_state_write_blocknum_start_pos, plc_state_write_dtype,
                  plc_state_write_content, server_ip):
    if op_select_button == "OCR识别":
        state, _ = send_server_func(state, server_ip, ocr_block_text, ocr_text)
    else:
        # write plc axis
        state, _ = write_plc_func(state, user_plc, plc_write_blocknum, plc_write_blocknum_start_pos, plc_write_dtype, axis_text)
        # write plc state
        state, _ = write_plc_func(state, user_plc, plc_state_write_blocknum, plc_state_write_blocknum_start_pos, plc_state_write_dtype, axis_err_text)
    return state

def auto_recog_button_func(state, user_serial, user_camera, user_plc, op_select_button, serial_openlight_cmd, serial_clostlight_cmd, camera_root,
                plc_write_blocknum, plc_write_blocknum_start_pos, plc_write_dtype, plc_state_write_blocknum, plc_state_write_blocknum_start_pos, plc_state_write_dtype,
                server_ip, ocr_block_text, plc_read_blocknum, plc_read_blocknum_start_pos, plc_read_length, plc_read_dtype,
                ocr_text, axis_text, axis_err_text):
    # 读取plc状态
    state, _, data = read_plc_func(state, user_plc, plc_read_blocknum, plc_read_blocknum_start_pos, plc_read_length, plc_read_dtype)
    if data:
        # 进行自动识别
        state, _, res = auto_recog_func(state, user_serial, user_camera, user_plc, op_select_button, serial_openlight_cmd, serial_clostlight_cmd, camera_root,
                    plc_write_blocknum, plc_write_blocknum_start_pos, plc_write_dtype, plc_state_write_blocknum, plc_state_write_blocknum_start_pos, plc_state_write_dtype,
                    server_ip, ocr_block_text, plc_read_blocknum, plc_read_blocknum_start_pos, plc_read_length, plc_read_dtype)
        if op_select_button == "OCR识别":
            return state, state, res['ocr_text'], axis_text, axis_err_text
        else:
            return state, state, ocr_text, res['axis_text'], res['axis_err_text']
    return state, state, ocr_text, axis_text, axis_err_text

def fresh_button_func(user_serial, user_camera, user_plc):
    if user_serial is not None and user_serial.isOpen():
        ser_state = '在线'
    else:
        ser_state = '离线'
        
    if user_camera is None or user_camera.cam is None:
        cam_state = '离线'
    else:
        connect_value, connect_logs = user_camera.decide_divice_on_line()
        if connect_value:
            cam_state = '在线'
        else:
            cam_state = '离线'
            
    if user_plc is not None and user_plc.get_connected():
        plc_state = '在线'
    else:
        plc_state = '离线'
    return gr.update(value=ser_state), gr.update(value=cam_state), gr.update(value=plc_state)

def create_ui():
    title = """<p><h1 align="center">OCR-Workshop-Alpha</h1></p>
    """
    description = """<p>Gradio demo for OCR-WorWorkshop, used for Industry_Ocr</p>"""

    with gr.Blocks() as iface:
        state = gr.State([])
        user_serial = gr.State(None)
        user_camera = gr.State(None)
        user_plc = gr.State(None)

        gr.Markdown(title)
        gr.Markdown(description)
        
        with gr.Row(visibile=True) as _open_key_conf:
            ocr_api_key = gr.Textbox(
                    placeholder="Input OCR API key",
                    show_label=False,
                    label="OCR API Key",
                    lines=1,
                    type="password")
            enable_button = gr.Button(value="OPEN OCR", interactive=True, variant='primary')
        with gr.Row(visible=False) as _open_key_state:
            wiki_output = gr.Textbox(lines=5, label="错误提示", max_lines=5)
        
        with gr.Row(visible=False) as _use_state1:
            with gr.Column(scale=3.0):
                camera_img = gr.Image(interactive=True).style(height=500)
            with gr.Column(scale=1.0):
                chatbot = gr.Chatbot(label="chat with message info").style(height=500)

        with gr.Tabs(elem_id='conf tabs', visiblie=False) as _root_state1:
            with gr.TabItem('基础配置'):
                with gr.Row():
                    config_button = gr.Radio(
                                    choices=["收起配置", "展开配置"],
                                    value="收起配置",
                                    label="configure",
                                    interactive=True)
                with gr.Row(visible=False) as _conf:
                    with gr.Tab("串口配置"):
                    # with gr.Column(scale=1.0):
                        with gr.Row():
                            serial_port = gr.Textbox(label='串口port名称', value="COM3")
                            serial_baudrate = gr.Dropdown(choices=['50', '75', '110', '134', '150', '200', '300', 
                                                                '600', '1200', '1800', '2400', '4800', '9600', 
                                                                '19200', '38400', '57600', '115200', '230400', 
                                                                '460800', '500000', '576000', '921600', 
                                                                '1000000', '1152000', '1500000', '2000000', 
                                                                '2500000', '3000000', '3500000', '4000000'],
                                                    value='9600',
                                                    label='串口波特率', interactive=True)
                            serial_bytesize = gr.Radio(choices=['FIVEBITS', 'SIXBITS', 'SEVENBITS', 'EIGHTBITS'],
                                                    value='EIGHTBITS',
                                                    label='串口数据位', interactive=True)
                            serial_parity = gr.Radio(choices=['PARITY_NONE', 'PARITY_EVEN', 'PARITY_ODD', 'PARITY_MARK', 'PARITY_SPACE'],
                                                    value='PARITY_NONE',
                                                    label='串口校验位', interactive=True)
                            serial_stopbits = gr.Radio(choices=['STOPBITS_ONE', 'STOPBITS_ONE_POINT_FIVE', 'STOPBITS_TWO'],
                                                    value='STOPBITS_ONE',
                                                    label='串口停止位', interactive=True)
                        with gr.Row():
                            serial_xonxoff = gr.Radio(choices=[True, False],
                                                    value=False,
                                                    label='串口软件流控', interactive=True)
                            serial_rtscts = gr.Radio(choices=[True, False],
                                                    value=False,
                                                    label='串口硬件RTS/CTS流控', interactive=True)
                            serial_dsrdtr = gr.Radio(choices=[True, False],
                                                    value=False,
                                                    label='串口硬件DSR/DTR流控', interactive=True)
                            serial_read_timeout = gr.Textbox(label='串口读超时时间', placeholder="None")
                            serial_write_timeout = gr.Textbox(label='串口写超时时间', placeholder="None")
                        with gr.Row():
                            serial_openlight_cmd = gr.Textbox(label='open light', value="$110ff14;$120ff17;$1306414", placeholder="开灯发送字符串")
                            serial_clostlight_cmd = gr.Textbox(label='clost light', value="$210ff17;$220ff14;$2306417", placeholder="关等发送字符串")
                # with gr.Row(visible=False) as camera_conf:
                    # with gr.Column(scale=1.0):
                    with gr.Tab("相机配置"):
                        with gr.Row():
                            camera_root = gr.Textbox(label='相机采集路径', value='')
                            camera_index = gr.Textbox(label='连接相机索引', value='0')
                            camera_stdcall = gr.Radio(choices=[0, 1],
                                                    value=1,
                                                    label='抓取图像方式，0-回调抓取，1-主动抓取', interactive=True)
                            camera_active_way = gr.Radio(choices=['getImagebuffer', 'getoneframetimeout'],
                                                    value='getImagebuffer',
                                                    label='获取图像流的方式', interactive=True)
                # with gr.Row(visible=False) as plc_conf:
                #     with gr.Column(scale=1.0):
                    with gr.Tab("PLC配置"):
                        with gr.Row():
                            plc_port = gr.Textbox(label='PLC的ip地址', value="192.168.1.10")
                            plc_rack = gr.Radio(choices=[0, 1],
                                                    value=0,
                                                    label='PLC机架号RACK', interactive=True)
                            plc_slot = gr.Radio(choices=[0, 1, 2, 3],
                                                    value=0,
                                                    label='PLC的CPU槽号Slot', interactive=True)
                        with gr.Row():
                            plc_read_blocknum = gr.Textbox(label='读取的数据块号', value="31")
                            plc_read_blocknum_start_pos = gr.Textbox(label='读取的数据块的起始地址', value="0")
                            plc_read_length = gr.Textbox(label='读取的数据长度', value="2")
                            plc_read_dtype = gr.Radio(choices=['bool', 'int', 'real', 'string', 'wstring'],
                                                    value='int',
                                                    label='读取数据类型', interactive=True)

                        with gr.Row():
                            plc_write_blocknum = gr.Textbox(label='写入的数据块号', value="1")
                            plc_write_blocknum_start_pos = gr.Textbox(label='写入的数据块的起始地址', value="0")
                            plc_write_dtype = gr.Radio(choices=['bool', 'int', 'real', 'string', 'wstring'],
                                                    value='int',
                                                    label='写入数据类型', interactive=True)
                            plc_write_content = gr.Textbox(label='写入的数据内容', value="10")
                        with gr.Row():
                            plc_state_write_blocknum = gr.Textbox(label='写入转轴状态的数据块号', value="1")
                            plc_state_write_blocknum_start_pos = gr.Textbox(label='写入转轴状态的数据块的起始地址', value="0")
                            plc_state_write_dtype = gr.Radio(choices=['bool', 'int', 'real', 'string', 'wstring'],
                                                    value='bool',
                                                    label='写入转轴状态的数据类型', interactive=True)
                            plc_state_write_content = gr.Textbox(label='写入转轴状态的数据内容', value="True")
                # with gr.Row(visible=False) as server_conf:
                    # with gr.Column(scale=1.0):
                    with gr.Tab("server配置"):
                        with gr.Row():
                            server_ip = gr.Textbox(label='server的ip地址', value="http://127.0.0.1:8888/jnlz/receiveHkdata")
                            # server_port = gr.Textbox(label='server的port', value="31")
                        # with gr.Row():
                        #     server_block = gr.Textbox(label='工号', value="1")
                        #     server_content = gr.Textbox(label='写入的数据内容', value="")
        with gr.Tabs(elem_id='conf tabs', visiblie=False) as _use_state2:
            with gr.TabItem('连接配置'):
                with gr.Row():
                    user_state = gr.Radio(choices=['管理员', '用户'],
                                        value='用户',
                                        label='用户状态', interactive=False)
                    serial_state = gr.Radio(choices=['在线', '离线'],
                                        value='离线',
                                        label='串口状态', interactive=False)
                    camera_state = gr.Radio(choices=['在线', '离线'],
                                        value='离线',
                                        label='相机状态', interactive=False)
                    plc_state = gr.Radio(choices=['在线', '离线'],
                                        value='离线',
                                        label='PLC状态', interactive=False)
                    
                with gr.Row():
                    init_button = gr.Button(value="初始化连接", interactive=True)
                    fresh_button = gr.Button(value="刷新连接状态", interactive=True)
                    camera_state_button = gr.Button(value="相机状态", interactive=True)
                    clear_state_button = gr.Button(value="清空infos", interactive=True)
                    
        with gr.Tabs(elem_id='op tabs', visible=False) as _use_state3:
            with gr.TabItem('基础操作'):
                with gr.Row():
                    op_select_button = gr.Radio(
                                        choices=["OCR识别", "转轴角度识别"],
                                        value="OCR识别",
                                        label="configure",
                                        interactive=True)
                with gr.Row():
                    with gr.Column(visible=True) as ocr_text_vis:
                        ocr_text =  gr.Textbox(label='OCR识别结果', value="")
                        ocr_block_text = gr.Textbox(label='工号', value="1")
                    with gr.Column(visible=False) as aixs_text_vis:
                        axis_text =  gr.Textbox(label='转轴角度识别结果', value="0")
                        axis_err_text =  gr.Textbox(label='转轴状态', value="True")
                with gr.Tab("自动识别"):
                    auto_recog_button = gr.Button(value="自动识别", interactive=True)
                with gr.Tab("手动识别"):
                    open_light_serial = gr.Button(value="开灯", interactive=True)
                    close_light_serial = gr.Button(value="关灯", interactive=True)
                    open_camera = gr.Button(value="采图", interactive=True)
                    recog = gr.Button(value="识别", interactive=True)
                    write_op = gr.Button(value="写入数据", interactive=True)
                    
        # 用户登录状态
        ocr_api_key.submit(ocr_api_key_func,
                           inputs=[ocr_api_key],
                           outputs=[_open_key_conf, _open_key_state, _use_state1, _root_state1, _use_state2, _use_state3, wiki_output, user_state])
        enable_button.click(enable_button_func,
                            inputs=[ocr_api_key],
                            outputs=[_open_key_conf, _open_key_state, _use_state1, _root_state1, _use_state2, _use_state3, wiki_output, user_state])
                
        config_button.change(config_button_func,
                                inputs=[config_button],
                                outputs=[_conf])
        
        op_select_button.change(op_select_button_func,
                                    inputs=[op_select_button],
                                    outputs=[ocr_text_vis, aixs_text_vis])
        init_button.click(init_button_func,
                          inputs=[state, user_serial, user_camera, user_plc, serial_port, serial_baudrate, serial_bytesize, serial_parity, serial_stopbits,
                                  serial_xonxoff, serial_rtscts, serial_dsrdtr, serial_read_timeout, serial_write_timeout,
                                  camera_index, camera_stdcall, camera_active_way, plc_port, plc_rack, plc_slot],
                          outputs=[state, chatbot, user_serial, user_camera, user_plc])
        fresh_button.click(fresh_button_func,
                           inputs=[user_serial, user_camera, user_plc],
                           outputs=[serial_state, camera_state, plc_state])
        camera_state_button.click(camera_state_button_func,
                                  inputs=[state, user_camera],
                                  outputs=[state, chatbot])
        clear_state_button.click(clear_state_button_func,
                                 inputs=[state],
                                 outputs=[state, chatbot])
        
        # should cancel auto process
        auto_env = auto_recog_button.click(auto_recog_button_func,
                                        inputs=[state, user_serial, user_camera, user_plc, op_select_button, serial_openlight_cmd, serial_clostlight_cmd, camera_root,
                                                plc_write_blocknum, plc_write_blocknum_start_pos, plc_write_dtype, plc_state_write_blocknum, plc_state_write_blocknum_start_pos, plc_state_write_dtype,
                                                server_ip, ocr_block_text, plc_read_blocknum, plc_read_blocknum_start_pos, plc_read_length, plc_read_dtype,
                                                ocr_text, axis_text, axis_err_text],
                                        outputs=[state, chatbot, ocr_text, axis_text, axis_err_text],
                                        every=30)
        open_light_serial.click(turn_light_serial_func,
                                inputs=[state, user_serial, serial_openlight_cmd],
                                outputs=[state, chatbot],
                                cancels=[auto_env])
        close_light_serial.click(turn_light_serial_func,
                                 inputs=[state, user_serial, serial_clostlight_cmd],
                                 outputs=[state, chatbot],
                                 cancels=[auto_env])
        open_camera.click(open_camera_func,
                          inputs=[state, user_camera, camera_root, user_serial, serial_openlight_cmd, serial_clostlight_cmd],
                          outputs=[state, chatbot, camera_img],
                          cancels=[auto_env])
        recog.click(recog_func,
                    inputs=[state, camera_img, op_select_button, camera_root, ocr_text, axis_text, axis_err_text],
                    outputs=[state, chatbot, ocr_text, axis_text, axis_err_text])
        write_op.click(write_op_func,
                       inputs=[state, user_plc, op_select_button, ocr_text, ocr_block_text, axis_text, axis_err_text,
                               plc_write_blocknum, plc_write_blocknum_start_pos, plc_write_dtype, plc_write_content,
                               plc_state_write_blocknum, plc_state_write_blocknum_start_pos, plc_state_write_dtype,
                               plc_state_write_content, server_ip],
                       outputs=[state, chatbot],
                       cancels=[auto_env])
        

        return iface


if __name__ == '__main__':
    iface = create_ui()
    iface.queue(concurrency_count=5, api_open=False, max_size=10)
    iface.launch(server_name="0.0.0.0" if not is_platform_win() else None, enable_queue=True, server_port=args.port,
                 share=args.gradio_share)