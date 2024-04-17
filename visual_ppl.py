#!/usr/bin/env python3

import gradio as gr
from argparse import ArgumentParser
from loguru import logger

import token_visualizer

MODEL = token_visualizer.TGIModel(
    decoder_input_details=True,
)


@logger.catch(reraise=True)
def ppl_from_model(
    text: str,
    url: str,
    bos: str,
    eos: str,
    display_whitespace: bool,
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
    assert url, f"Please provide url of your tgi model. Current url: {url}"
    logger.info(f"Set url to {url}")
    MODEL.url = url
    MODEL.display_whitespace = display_whitespace
    MODEL.max_tokens = 1

    text = bos + text + eos
    tokens = MODEL.generate_inputs_prob(text)
    html = MODEL.html_to_visualize(tokens)
    return html


def make_parser() -> ArgumentParser:
    parser = ArgumentParser("PPL Visualizer")
    parser.add_argument("-s", "--share", action="store_true", help="share service to the internet")
    parser.add_argument("-p", "--port", type=int, default=12345, help="port to run the service")
    return parser


def ppl_demo():
    args = make_parser().parse_args()
    logger.info(f"Args: {args}")
    demo = gr.Interface(
        ppl_from_model,
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
        css=token_visualizer.css_style(),
    )
    demo.launch(
        server_name="0.0.0.0",
        share=args.share,
        server_port=args.port,
    )


if __name__ == "__main__":
    ppl_demo()
