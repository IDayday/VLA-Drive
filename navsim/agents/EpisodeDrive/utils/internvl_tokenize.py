"""Exact InternVL prompt expansion/tokenization shared by ranks and workers."""

from typing import List

from .conversation import get_conv_template


def build_internvl_model_inputs(
    tokenizer,
    questions: List[str],
    num_patches_list: List[int],
    system_message: str,
    num_image_token: int = 256,
):
    queries = []
    for question, num_patches in zip(questions, num_patches_list):
        if "<image>" not in question:
            question = "<image>\n" + question

        template = get_conv_template("internvl2_5")
        template.system_message = system_message
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        image_tokens = (
            "<img>"
            + "<IMG_CONTEXT>" * num_image_token * num_patches
            + "</img>"
        )
        queries.append(query.replace("<image>", image_tokens, 1))

    tokenizer.padding_side = "left"
    return tokenizer(
        queries,
        return_tensors="pt",
        padding="max_length",
        max_length=2800,
    )
