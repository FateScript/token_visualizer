#!/usr/bin/env python3

import os
from functools import lru_cache
from typing import Dict

import gradio as gr
from loguru import logger
from transformers import AutoTokenizer

from sentencepiece import SentencePieceProcessor

CANDIDATES = [  # model name sorted by alphabet
    "baichuan-inc/Baichuan2-13B-Chat",
    "bigcode/starcoder2-15b",
    "deepseek-ai/deepseek-coder-33b-instruct",
    # "google/gemma-7b",
    "gpt2",
    # "meta-llama/Llama-2-7b-chat-hf",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "THUDM/chatglm3-6b",
]
SENTENCE_PIECE_MAPPING = {}
SP_PREFIX = "SentencePiece/"


def add_sp_tokenizer(name: str, tokenizer_path: str):
    """Add a sentence piece tokenizer to the list of available tokenizers."""
    model_key = SP_PREFIX + name
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    SENTENCE_PIECE_MAPPING[model_key] = tokenizer_path
    CANDIDATES.append(model_key)


# add_sp_tokenizer("LLaMa", "llama_tokenizer.model")
logger.info(f"SentencePiece tokenizer: {list(SENTENCE_PIECE_MAPPING.keys())}")


@lru_cache
def get_tokenizer_and_vocab(name):
    if name.startswith(SP_PREFIX):
        local_file_path = SENTENCE_PIECE_MAPPING[name]
        tokenizer = SentencePieceProcessor(local_file_path)
        rev_vocab = {id_: tokenizer.id_to_piece(id_) for id_ in range(tokenizer.get_piece_size())}  # noqa
    else:
        tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        rev_vocab = {v: k for k, v in tokenizer.get_vocab().items()}
    return tokenizer, rev_vocab


def tokenize(name: str, text: str) -> Dict:
    tokenizer, rev_vocab = get_tokenizer_and_vocab(name)

    ids = tokenizer.encode(text)
    s, entities = '', []
    for i in ids:
        entity = str(i)
        start = len(s)
        s += rev_vocab[i]
        end = len(s)
        entities.append({"entity": entity, "start": start, "end": end})

    return {
        "text": s + f"\n({len(ids)} tokens / {len(text)} characters)",
        "entities": entities
    }


@logger.catch(reraise=True)
def make_demo():
    logger.info("Creating Interface..")

    DEFAULT_TOKENIZER = CANDIDATES[0]
    DEFAULT_INPUTTEXT = "Hello world."

    demo = gr.Interface(
        fn=tokenize,
        inputs=[
            gr.Dropdown(
                CANDIDATES, value=DEFAULT_TOKENIZER,
                label="Tokenizer", allow_custom_value=False
            ),
            gr.TextArea(value=DEFAULT_INPUTTEXT, label="Input text"),
        ],
        outputs=[
            gr.HighlightedText(
                value=tokenize(DEFAULT_TOKENIZER, DEFAULT_INPUTTEXT),
                label="Tokenized results"
            )
        ],
        title="Tokenzier Visualizer",
        description="If you want to try more tokenizers, please contact the author@wangfeng",  # noqa
        examples=[
            [DEFAULT_TOKENIZER, "乒乓球拍卖完了，无线电法国别研究，我一把把把把住了"],
            ["bigcode/starcoder2-15b", "def print():\n    print('Hello')"],
        ],
        cache_examples=True,
        live=True,
    )
    return demo


if __name__ == "__main__":
    demo = make_demo()
    demo.launch(server_name="0.0.0.0")
