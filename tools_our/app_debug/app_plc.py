import argparse
import sys

import snap7
from snap7 import util
import struct
import gradio as gr

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=6086, help="only useful when running gradio applications")
parser.add_argument('--debug', action="store_true")
parser.add_argument('--gradio_share', action="store_true")

args = parser.parse_args()

def is_platform_win():
    return sys.platform == "win32"

def plc_config_button_func(plc_config_button_func):
    if plc_config_button_func == "收起配置":
        return gr.update(visible=False)
    else:
        return gr.update(visible=True)
  
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
        return state, state
    
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
    return state, state

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

def create_ui():
    title = """<p><h1 align="center">OCR-Workshop-Alpha</h1></p>
    """
    description = """<p>Gradio demo for OCR-WorWorkshop, used for Industry_Ocr</p>"""

    with gr.Blocks() as iface:
        state = gr.State([])
        user_plc = gr.State(None)

        gr.Markdown(title)
        gr.Markdown(description)
        with gr.Row():
            chatbot = gr.Chatbot(label="chat with message info")
        with gr.Tabs(elem_id='plc conf tabs'):
            with gr.TabItem('PLC配置'):
                with gr.Row():
                    plc_config_button = gr.Radio(
                                    choices=["收起配置", "展开配置"],
                                    value="收起配置",
                                    label="PLC configure",
                                    interactive=True)
                with gr.Row(visible=False) as plc_conf:
                    with gr.Column(scale=1.0):
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
                            
        with gr.Tabs(elem_id='plc op tabs'):
            with gr.TabItem('PLC操作'):
                with gr.Row():
                    open_plc = gr.Button(value="PLC连接", interactive=True)
                    close_plc = gr.Button(value="PLC关闭", interactive=True)
                    read_plc = gr.Button(value="读取数据", interactive=True)
                    write_plc = gr.Button(value="写入数据", interactive=True)
                
        plc_config_button.change(plc_config_button_func, 
                                    inputs=plc_config_button,
                                    outputs=plc_conf)
        open_plc.click(open_plc_func,
                        inputs=[state, user_plc, plc_port, plc_rack, plc_slot],
                        outputs=[state, chatbot, user_plc])
        close_plc.click(close_plc_func,
                        inputs=[state, user_plc],
                        outputs=[state, chatbot, user_plc])
        read_plc.click(read_plc_func,
                       inputs=[state, user_plc, plc_read_blocknum, plc_read_blocknum_start_pos, plc_read_length, plc_read_dtype],
                       outputs=[state, chatbot])
        write_plc.click(write_plc_func,
                        inputs=[state, user_plc, plc_write_blocknum, plc_write_blocknum_start_pos, plc_write_dtype, plc_write_content],
                        outputs=[state, chatbot])
       

        return iface


if __name__ == '__main__':
    iface = create_ui()
    iface.queue(concurrency_count=5, api_open=False, max_size=10)
    iface.launch()
    # iface.launch(server_name="0.0.0.0" if not is_platform_win() else None, enable_queue=True, server_port=args.port,
    #              share=args.gradio_share)