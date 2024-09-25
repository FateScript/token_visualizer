#!/usr/bin/env python3

import functools
from argparse import ArgumentParser
from typing import Tuple, Optional

import gradio as gr
from loguru import logger

import token_visualizer
from token_visualizer import css_style, ensure_os_env


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
        help="Share service to the internet.",
    )
    parser.add_argument(
        "-p", "--port", type=int, default=12123,
        help="Port to run the service, default to 12123."
    )
    return parser


def build_model_by_args(args) -> token_visualizer.TopkTokenModel:
    BASE_URL = ensure_os_env("BASE_URL")
    OPENAI_API_KEY = ensure_os_env("OPENAI_KEY")
    TGI_URL = ensure_os_env("TGI_URL")

    model: Optional[token_visualizer.TopkTokenModel] = None

    if args.type == "llm":
        model = token_visualizer.TransformerModel(repo=args.hf_repo)
    elif args.type == "tgi":
        if args.tgi_url:
            TGI_URL = args.tgi_url
        model = token_visualizer.TGIModel(url=TGI_URL, details=True)
    elif args.type == "oai":
        model = token_visualizer.OpenAIModel(
            base_url=BASE_URL,
            api_key=OPENAI_API_KEY,
            model_name=args.oai_model,
        )
    elif args.type == "oai-proxy":
        model = token_visualizer.OpenAIProxyModel(
            base_url=BASE_URL,
            api_key=OPENAI_API_KEY,
            model_name="gpt-4-turbo-2024-04-09",
        )
    else:
        raise ValueError(f"Unknown model type {args.type}")

    return model


@logger.catch(reraise=True)
def text_analysis(
    text: str,
    display_whitespace: bool,
    do_sample: bool,
    temperature: float,
    max_tokens: int,
    repetition_penalty: float,
    num_beams: int,
    topk: int,
    topp: float,
    topk_per_token: int,
    model,  # model should be built in the interface
) -> Tuple[str, str]:
    model.display_whitespace = display_whitespace
    model.do_sample = do_sample
    model.temperature = temperature
    model.max_tokens = max_tokens
    model.repetition_penalty = repetition_penalty
    model.num_beams = num_beams
    model.topk = topk
    model.topp = topp
    model.topk_per_token = topk_per_token

    tokens = model.generate_topk_per_token(text)
    html = model.html_to_visualize(tokens)

    html += "<br>"
    if isinstance(model, token_visualizer.TGIModel) and model.num_prefill_tokens:
        html += f"<div><strong>input tokens: {model.num_prefill_tokens}</strong></div>"
    html += f"<div><strong>output tokens: {len(tokens)}</strong></div>"

    return model.generated_answer, html


def build_inference_analysis_demo(args):
    model = build_model_by_args(args)
    inference_func = functools.partial(text_analysis, model=model)

    interface = gr.Interface(
        inference_func,
        [
            gr.TextArea(placeholder="Please input text here"),
            gr.Checkbox(value=False, label="display whitespace in output"),
            gr.Checkbox(value=True, label="do_sample"),
            gr.Slider(minimum=0, maximum=1, step=0.05, value=1.0, label="temperature"),
            gr.Slider(minimum=1, maximum=4096, step=1, value=512, label="max tokens"),
            gr.Slider(minimum=1, maximum=2, step=0.1, value=1.0, label="repetition penalty"),
            gr.Slider(minimum=1, maximum=10, step=1, value=1, label="num beams"),
            gr.Slider(minimum=1, maximum=100, step=1, value=50, label="topk"),
            gr.Slider(minimum=0, maximum=1, step=0.05, value=1.0, label="topp"),
            gr.Slider(minimum=1, maximum=10, step=1, value=5, label="per-token topk"),
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
    )
    return interface


@logger.catch(reraise=True)
def ppl_from_model(
    text: str,
    url: str,
    bos: str,
    eos: str,
    display_whitespace: bool,
    model,
) -> str:
    """Generate PPL visualization from model.

    Args:
        text (str): input text to visualize.
        url (str): tgi url.
        bos (str): begin of sentence token.
        eos (str): end of sentence token.
        display_whitespace (bool): whether to display whitespace for output text.
            If set to True, whitespace will be displayed as "␣".
    """
    url = url.strip()
    assert url, f"Please provide url of your tgi model. Current url: {url}"
    logger.info(f"Set url to {url}")
    model.url = url
    model.display_whitespace = display_whitespace
    model.max_tokens = 1

    text = bos + text + eos
    tokens = model.generate_inputs_prob(text)
    html = model.html_to_visualize(tokens)

    # display tokens and ppl at the end
    html += "<br>"
    html += f"<div><strong>total tokens: {len(tokens)}</strong></div>"
    ppl = tokens[-1].ppl
    html += f"<div><strong>ppl: {ppl:.4f}</strong></div>"
    return html


def build_ppl_visualizer_demo(args):
    model = build_model_by_args(args)
    ppl_func = functools.partial(ppl_from_model, model=model)

    ppl_interface = gr.Interface(
        ppl_func,
        [
            gr.TextArea(placeholder="Please input text to visualize here"),
            gr.TextArea(
                placeholder="Please input tgi url here (Error if not provided)",
                lines=1,
            ),
            gr.TextArea(placeholder="BOS token, default to empty string", lines=1),
            gr.TextArea(placeholder="EOS token, default to empty string", lines=1),
            gr.Checkbox(value=False, label="display whitespace in output, default to False"),
        ],
        "html",
        title="PPL Visualizer",
    )
    return ppl_interface


def demo():
    args = make_parser().parse_args()
    logger.info(f"Args: {args}")

    demo = gr.Blocks(css=css_style())
    with demo:
        with gr.Tab("Inference"):
            build_inference_analysis_demo(args)
        with gr.Tab("PPL"):
            build_ppl_visualizer_demo(args)

    demo.launch(
        server_name="0.0.0.0",
        share=args.share,
        server_port=args.port,
    )


if __name__ == "__main__":
    demo()
