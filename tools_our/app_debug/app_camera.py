import argparse
import sys

import gradio as gr
import serial

from camera_utils.camera import Camera

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=6086, help="only useful when running gradio applications")
parser.add_argument('--debug', action="store_true")
parser.add_argument('--gradio_share', action="store_true")

args = parser.parse_args()

def is_platform_win():
    return sys.platform == "win32"

def serial_config_button_func(serial_config_button):
    if serial_config_button == "收起配置":
        return gr.update(visible=False)
    else:
        return gr.update(visible=True)
    
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
    
    import ipdb;ipdb.set_trace()
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
    _cmd_len = user_serial.write(serial_turnlight_cmd.encode("utf-8"))
    state += [(f"发送串口命令{serial_turnlight_cmd}", f"串口发送成功，共{_cmd_len}字节")]
    return state, state

def camera_config_button_func(camera_config_button):
    if camera_config_button == "收起配置":
        return gr.update(visible=False)
    else:
        return gr.update(visible=True)

def open_camera_func(state, user_camera, camera_index, camera_stdcall, camera_active_way, camera_img):
    camera_index = int(camera_index)
    # 创建相机
    if user_camera is not None:
        if user_camera.camera_index != camera_index:
            state += [(None, f"目前相机-{user_camera.camera_index}-连接中，要切换需先关闭原始连接")]
            return state, state, user_camera, camera_img
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
        return state, state, None, None
    connect_value, connect_logs = user_camera.decide_divice_on_line()
    state += [(None, connect_logs)]
    if not connect_value:
        logs_open_device = user_camera.open_device()
        state += [(None, connect_logs)]
    if not user_camera.decide_divice_on_line()[0]: # 再次判断
        state += [(None, "相机初始化失败，请检查设备连接情况 failed")]
        return state, state, None, None
    image, logs_get_image = user_camera.get_image()
    state += [(None, logs_get_image)]
    return state, state, user_camera, image

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

def clear_state_button_func(state):
    state = []
    return state, state

def create_ui():
    title = """<p><h1 align="center">OCR-Workshop-Alpha</h1></p>
    """
    description = """<p>Gradio demo for OCR-WorWorkshop, used for Industry_Ocr</p>"""

    with gr.Blocks() as iface:
        state = gr.State([])
        user_serial = gr.State(None)
        user_camera = gr.State(None)

        gr.Markdown(title)
        gr.Markdown(description)
        with gr.Row():
            with gr.Column(scale=3.0):
                camera_img = gr.Image(interactive=False).style(height=500)
            with gr.Column(scale=1.0):
                chatbot = gr.Chatbot(label="chat with message info").style(height=500)
        with gr.Tabs(elem_id='serial conf tabs'):
            with gr.TabItem('串口配置'):
                with gr.Row():
                    serial_config_button = gr.Radio(
                                    choices=["收起配置", "展开配置"],
                                    value="收起配置",
                                    label="serial configure",
                                    interactive=True)
                with gr.Row(visible=False) as serial_conf:
                    with gr.Column(scale=1.0):
                        with gr.Row():
                            serial_port = gr.Textbox(label='串口port名称', value="COM17")
                            serial_baudrate = gr.Dropdown(choices=[50, 75, 110, 134, 150, 200, 300, 
                                                                600, 1200, 1800, 2400, 4800, 9600, 
                                                                19200, 38400, 57600, 115200, 230400, 
                                                                460800, 500000, 576000, 921600, 
                                                                1000000, 1152000, 1500000, 2000000, 
                                                                2500000, 3000000, 3500000, 4000000],
                                                    value=9600,
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
                            serial_openlight_cmd = gr.Textbox(label='open light', placeholder="开灯发送字符串")
                            serial_clostlight_cmd = gr.Textbox(label='clost light', placeholder="关等发送字符串")
        with gr.Tabs(elem_id='camera conf tabs'):
            with gr.TabItem('相机配置'):
                with gr.Row():
                    camera_config_button = gr.Radio(
                                    choices=["收起配置", "展开配置"],
                                    value="收起配置",
                                    label="camera configure",
                                    interactive=True)
                with gr.Row(visible=False) as camera_conf:
                    with gr.Column(scale=1.0):
                        with gr.Row():
                            camera_index = gr.Textbox(label='连接相机索引', value='0')
                            camera_stdcall = gr.Radio(choices=[0, 1],
                                                    value=1,
                                                    label='抓取图像方式，0-回调抓取，1-主动抓取', interactive=True)
                            camera_active_way = gr.Radio(choices=['getImagebuffer', 'getoneframetimeout'],
                                                    value='getImagebuffer',
                                                    label='获取图像流的方式', interactive=True)
        with gr.Row():
            with gr.Column(scale=2.0):
                with gr.Tabs(elem_id='serial op tabs'):
                    with gr.TabItem('串口操作'):
                        with gr.Row():
                            open_serial = gr.Button(value="串口连接", interactive=True)
                            close_serial = gr.Button(value="串口关闭", interactive=True)
                        with gr.Row():
                            open_light_serial = gr.Button(value="开灯", interactive=True)
                            close_light_serial = gr.Button(value="关灯", interactive=True)
            with gr.Column(scale=2.0):
                with gr.Tabs(elem_id='camera op tabs'):
                    with gr.TabItem('相机操作'):
                        with gr.Row():
                            open_camera = gr.Button(value="相机连接", interactive=True)
                            close_camera = gr.Button(value="相机关闭", interactive=True)
                        with gr.Row():
                            camera_info_button = gr.Button(value="设备信息", interactive=True)
                            camera_connect_info_button = gr.Button(value="连接状态", interactive=True)
            with gr.Column(scale=1.0):
                with gr.Tabs(elem_id='logs op tabs'):
                    with gr.TabItem('logs info'):
                        clear_state_button = gr.Button(value="clear logs", interactive=True)
        # 串口函数
        serial_config_button.change(serial_config_button_func, 
                                    inputs=serial_config_button,
                                    outputs=serial_conf)
        open_serial.click(open_serial_func,
                          inputs=[state, user_serial, serial_port, serial_baudrate, serial_bytesize, serial_parity,
                                  serial_stopbits, serial_xonxoff, serial_rtscts, serial_dsrdtr,
                                  serial_read_timeout, serial_write_timeout],
                          outputs=[state, chatbot, user_serial])
        close_serial.click(close_serial_func,
                           inputs=[state, user_serial],
                           outputs=[state, chatbot, user_serial])
        open_light_serial.click(turn_light_serial_func,
                                inputs=[state, user_serial, serial_openlight_cmd],
                                outputs=[state, chatbot])
        close_light_serial.click(turn_light_serial_func,
                                inputs=[state, user_serial, serial_clostlight_cmd],
                                outputs=[state, chatbot])
        # 相机函数
        camera_config_button.change(camera_config_button_func,
                                    inputs=camera_config_button,
                                    outputs=camera_conf)
        open_camera.click(open_camera_func,
                          inputs=[state, user_camera, camera_index, camera_stdcall, camera_active_way, camera_img],
                          outputs=[state, chatbot, user_camera, camera_img])
        close_camera.click(close_camera_func,
                           inputs=[state, user_camera],
                           outputs=[state, chatbot, user_camera, camera_img])
        camera_info_button.click(camera_info_button_func,
                                 inputs=[state, user_camera],
                                 outputs=[state, chatbot])
        camera_connect_info_button.click(camera_connect_info_button_func,
                                 inputs=[state, user_camera],
                                 outputs=[state, chatbot])
        # 清空logs缓存
        clear_state_button.click(clear_state_button_func,
                                 inputs=[state],
                                 outputs=[state, chatbot])

        return iface


if __name__ == '__main__':
    iface = create_ui()
    iface.queue(concurrency_count=5, api_open=False, max_size=10)
    iface.launch()
    # iface.launch(server_name="0.0.0.0" if not is_platform_win() else None, enable_queue=True, server_port=args.port,
    #              share=args.gradio_share)