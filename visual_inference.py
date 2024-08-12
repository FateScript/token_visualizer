#!/usr/bin/env python3

from argparse import ArgumentParser
from typing import Tuple

import gradio as gr
from loguru import logger

import token_visualizer
from token_visualizer import css_style, ensure_os_env

BASE_URL = ensure_os_env("BASE_URL")
OPENAI_API_KEY = ensure_os_env("OPENAI_KEY")
TGI_URL = ensure_os_env("TGI_URL")


def make_parser() -> ArgumentParser:
    parser = ArgumentParser("Inference process visualizer")
    parser.add_argument(
        "-t", "--type",
        choices=["llm", "tgi", "oai", "oai-proxy"],
        default="oai-proxy",
        help="Type of model to use, default to openai-proxy"
    )
    parser.add_argument(
        "--hf-repo", type=str, default=None,
        help="Huggingface model repository, used when type is 'llm'. Default to None"
    )
    parser.add_argument(
        "--oai-model", type=str, default="gpt-4-turbo-2024-04-09",
        help="OpenAI model name, used when type is 'oai'/'oai-proxy'. "
        "Check https://platform.openai.com/docs/models for more details. "
        "Default to `gpt-4-turbo-2024-04-09`."
    )
    parser.add_argument(
        "--oai-key", type=str, default=None,
        help="OpenAI api key, used when type is 'oai'/'oai-proxy'. "
        "If provided, will override OPENAI_KEY env variable.",
    )
    parser.add_argument(
        "--tgi-url", type=str, default=None,
        help="Service url of TGI model, used when type is 'tgi'. "
        "If provided, will override TGI_URL env variable.",
    )
    parser.add_argument(
        "-s", "--share", action="store_true",
        help="Share service to the internet",
    )
    parser.add_argument(
        "-p", "--port", type=int, default=12123,
        help="Port to run the service, default to 12123"
    )
    return parser


args = make_parser().parse_args()
logger.info(f"Args: {args}")

MODEL: token_visualizer.TopkTokenModel = None

if args.type == "llm":
    MODEL = token_visualizer.TransformerModel(repo=args.hf_repo)
elif args.type == "tgi":
    if args.tgi_url:
        TGI_URL = args.tgi_url
    MODEL = token_visualizer.TGIModel(url=TGI_URL, details=True)
elif args.type == "oai":
    MODEL = token_visualizer.OpenAIModel(
        base_url=BASE_URL,
        api_key=OPENAI_API_KEY,
        model_name=args.oai_model,
    )
elif args.type == "oai-proxy":
    MODEL = token_visualizer.OpenAIProxyModel(
        base_url=BASE_URL,
        api_key=OPENAI_API_KEY,
        model_name="gpt-4-turbo-2024-04-09",
    )
else:
    raise ValueError(f"Unknown model type {args.type}")


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

    html += "<br>"
    if isinstance(MODEL, token_visualizer.TGIModel) and MODEL.num_prefill_tokens:
        html += f"<div><strong>input tokens: {MODEL.num_prefill_tokens}</strong></div>"
    html += f"<div><strong>output tokens: {len(tokens)}</strong></div>"

    return MODEL.generated_answer, html


def demo(share: bool = True):
    global args

    demo = gr.Interface(
        text_analysis,
        [
            gr.TextArea(placeholder="Please input text here"),
            gr.Checkbox(value=False, label="display whitespace"),
            gr.Checkbox(value=True, label="do_sample"),
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
    demo.launch(
        server_name="0.0.0.0",
        share=args.share,
        server_port=args.port,
    )


if __name__ == "__main__":
    demo()
