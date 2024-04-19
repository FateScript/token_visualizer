#!/usr/bin/env python3

from typing import Tuple

import gradio as gr
from loguru import logger

import token_visualizer
from token_visualizer import css_style, ensure_os_env

BASE_URL = ensure_os_env("BASE_URL")
OPENAI_API_KEY = ensure_os_env("OPENAI_KEY")

# MODEL = token_visualizer.TransformerModel()
# MODEL = token_visualizer.OpenAIModel
MODEL = token_visualizer.OpenAIProxyModel(
    base_url=BASE_URL,
    api_key=OPENAI_API_KEY,
    model_name="gpt-4-turbo-preview",
)


@logger.catch(reraise=True)
def text_analysis(
    text: str,
    display_whitespace: bool,
    do_sample: bool, temperature: float,
    max_tokens: int, repetition_penalty: float,
    num_beams: int, topk: int, topp: float,
) -> Tuple[str, str]:
    MODEL.display_whitespace = display_whitespace
    MODEL.do_sample = do_sample
    MODEL.temperature = temperature
    MODEL.max_tokens = max_tokens
    MODEL.repetition_penalty = repetition_penalty
    MODEL.num_beams = num_beams
    MODEL.topk = topk
    MODEL.topp = topp

    tokens = MODEL.generate_topk_per_token(text)
    html = MODEL.html_to_visualize(tokens)
    return MODEL.generated_answer, html


def demo(share: bool = True):
    demo = gr.Interface(
        text_analysis,
        [
            gr.TextArea(placeholder="Please input text here"),
            gr.Checkbox(value=False, label="display whitespace"),
            gr.Checkbox(value=False, label="do_sample"),
            gr.Slider(minimum=0, maximum=1, step=0.05, value=1.0, label="temperature"),
            gr.Slider(minimum=1, maximum=4096, step=1, value=256, label="max tokens"),
            gr.Slider(minimum=1, maximum=2, step=0.1, value=1.0, label="repetition penalty"),
            gr.Slider(minimum=1, maximum=10, step=1, value=1, label="num beams"),
            gr.Slider(minimum=1, maximum=100, step=1, value=50, label="topk"),
            gr.Slider(minimum=0, maximum=1, step=0.05, value=1.0, label="topp"),
        ],
        [
            gr.TextArea(label="LLM answer"),
            "html",
        ],
        examples=[
            ["Who are Hannah Quinlivan's child?"],
            ["Write python code to read a file and print its content."],
        ],
        title="LLM inference analysis",
        css=css_style(),
    )
    demo.launch(share=share)


if __name__ == "__main__":
    demo(share=False)
