#!/usr/bin/env python3

import colorsys
import itertools
import math
import operator
import statistics
from dataclasses import dataclass, field
from typing import List, Tuple

__all__ = [
    "Token",
    "candidate_tokens_html",
    "color_token_by_logprob",
    "set_tokens_ppl",
    "single_token_html",
    "tokens_min_max_logprob",
    "tokens_info_to_html",
]


@dataclass
class Token:
    text: str
    prob: float
    top_candidates: List = field(default_factory=list)
    ppl: float = field(default=None)

    @property
    def logprob(self) -> float:
        return math.log(self.prob)


def candidate_tokens_html(topk_candidate_tokens: List[Token]) -> str:
    """HTML content of a single token's topk candidate tokens."""
    template = '<span class="ppl-prediction ppl-hud-row">' \
        '<span>{token}</span><span class="ppl-hud-label">{prob}</span></span>'

    html_text = "".join([
        template.format(token=token.text, prob=f"{token.prob:.3%}")
        for token in topk_candidate_tokens
    ])
    html_text = f'<div class="ppl-predictions">{html_text}</div>'
    return html_text


def color_token_by_logprob(
    log_prob: float, min_log_prob: float, max_log_prob: float,
    hue_red: int = 0, hue_green: int = 150, epsilon: float = 1e-5,
) -> str:
    """According to the token's log prob, assign RGB color to the token.
    reference: https://twitter.com/thesephist/status/1617909119423500288

    Args:
        log_prob (float): log prob of the token.
        min_log_prob (float): min log prob of all tokens.
        max_log_prob (float): max log prob of all tokens.
        hue_red (int, optional): hue value of the red color. Defaults to 0.
        hue_green (int, optional): hue value of the green color. Defaults to 150.
        epsilon (float, optional): avoid divide by zero. Defaults to 1e-5.
    """
    # clamp the log_prob and scale to (hsl_red, hsl_green)
    if min_log_prob == max_log_prob:
        ratio = 1  # set to green color
    else:
        log_prob = max(min_log_prob, min(log_prob, max_log_prob))
        ratio = (log_prob - min_log_prob) / max((max_log_prob - min_log_prob), epsilon)
    hue = ratio * (hue_green - hue_red) + hue_red
    red, green, blue = colorsys.hls_to_rgb(hue / 360.0, 0.85, 0.6)  # hls({hue}deg 85% 60%)
    rgb_string = f"rgb({int(red * 255)}, {int(green * 255)}, {int(blue * 255)})"
    return rgb_string


def set_tokens_ppl(tokens: List[Token]):
    """Set ppl value for each token in the list of tokens."""
    logprob_sum = itertools.accumulate([x.logprob for x in tokens], operator.add)
    for num_tokens, (token, logprob) in enumerate(zip(tokens, logprob_sum), 1):
        token.ppl = math.exp(-logprob / num_tokens)


def single_token_html(token: Token) -> str:
    """HTML text of single token."""
    template = '<div class="ppl-hud-row"><span class="ppl-hud-label">{label}</span><span>{value}</span></div>'  # noqa

    html_text = template.format(label="prob", value=f"{token.prob:.3%}")
    html_text += template.format(label="logprob", value=f"{token.logprob:.4f}")
    if token.ppl is not None:
        html_text += template.format(label="ppl", value=f"{token.ppl:.4f}")
    html_text += candidate_tokens_html(token.top_candidates)
    html_text = f"<div class='ppl-hud'>{html_text}</div>"
    return html_text


def tokens_min_max_logprob(tokens: List[Token]) -> Tuple[float, float]:
    """Calculate the normalized min and max logprob of a list of tokens."""
    if len(tokens) == 1:
        min_logprob = max_logprob = tokens[0].logprob
    else:
        logprob_mean = statistics.mean([token.logprob for token in tokens])
        logprob_stddev = statistics.stdev([token.logprob for token in tokens])
        min_logprob = logprob_mean - (2.5 * logprob_stddev)
        max_logprob = logprob_mean + (2.5 * logprob_stddev)
    return min_logprob, max_logprob


def tokens_info_to_html(tokens: List[Token], special_for_newline: bool = True) -> str:
    """
    Generate html for a list of token, include token color and hover text.

    Args:
        tokens (List[Token]): a list of tokens to generate html for.
        special_for_newline (bool, optional): whether to use special style for newline token.
            Defaults to True.
    """
    min_logprob, max_logprob = tokens_min_max_logprob(tokens)
    set_tokens_ppl(tokens)

    tokens_html = ""
    for token in tokens:
        hover_html = single_token_html(token)
        rgb = color_token_by_logprob(token.logprob, min_logprob, max_logprob)
        token_text = token.text
        token_text = token_text.replace("<", "&lt;").replace(">", "&gt;")  # make token like <func_call> display correctly  # noqa
        is_newline = repr("\n")[1:-1] in token_text
        if special_for_newline and is_newline:
            token_text = f'<span class="ppl-pseudo-token">{token_text}</span>'
        token_html = f'<span class="ppl-token" style="background: {rgb};">{token_text}{hover_html}</span>'  # noqa
        if is_newline:
            token_html += "<br>"
        tokens_html += token_html
    tokens_html = f'<div class="ppl-visualization-tokens" style="font-family: inherit;">{tokens_html}</div>'  # noqa
    return tokens_html
