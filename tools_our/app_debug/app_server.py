import argparse
import sys
import requests

import json
import gradio as gr

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=6086, help="only useful when running gradio applications")
parser.add_argument('--debug', action="store_true")
parser.add_argument('--gradio_share', action="store_true")

args = parser.parse_args()

def is_platform_win():
    return sys.platform == "win32"

def server_config_button_func(plc_config_button_func):
    if plc_config_button_func == "收起配置":
        return gr.update(visible=False)
    else:
        return gr.update(visible=True)
  
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


def create_ui():
    title = """<p><h1 align="center">OCR-Workshop-Alpha</h1></p>
    """
    description = """<p>Gradio demo for OCR-WorWorkshop, used for Industry_Ocr</p>"""

    with gr.Blocks() as iface:
        state = gr.State([])

        gr.Markdown(title)
        gr.Markdown(description)
        with gr.Row():
            chatbot = gr.Chatbot(label="chat with message info")
        with gr.Tabs(elem_id='server conf tabs'):
            with gr.TabItem('server配置'):
                with gr.Row():
                    server_config_button = gr.Radio(
                                    choices=["收起配置", "展开配置"],
                                    value="收起配置",
                                    label="server configure",
                                    interactive=True)
                with gr.Row(visible=False) as server_conf:
                    with gr.Column(scale=1.0):
                        with gr.Row():
                            server_ip = gr.Textbox(label='server的ip地址', value="http://127.0.0.1:8888/jnlz/receiveHkdata")
                            # server_port = gr.Textbox(label='server的port', value="31")
                        with gr.Row():
                            server_block = gr.Textbox(label='工号', value="1")
                            server_content = gr.Textbox(label='写入的数据内容', value="")
                            
        with gr.Tabs(elem_id='server op tabs'):
            with gr.TabItem('server操作'):
                with gr.Row():
                    send_server = gr.Button(value="发送数据", interactive=True)
                
        server_config_button.change(server_config_button_func, 
                                    inputs=server_config_button,
                                    outputs=server_conf)
        send_server.click(send_server_func,
                        inputs=[state, server_ip, server_block, server_content],
                        outputs=[state, chatbot])
       

        return iface


if __name__ == '__main__':
    iface = create_ui()
    iface.queue(concurrency_count=5, api_open=False, max_size=10)
    iface.launch()
    # iface.launch(server_name="0.0.0.0" if not is_platform_win() else None, enable_queue=True, server_port=args.port,
    #              share=args.gradio_share)