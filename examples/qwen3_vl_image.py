"""Run Hi-Lo Prune on one image with Qwen3-VL."""

import argparse

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoConfig, AutoProcessor

from hilo_prune.models.qwen3_vl import Qwen3VLForConditionalGeneration
from hilo_prune.patches import image_to_merged_patch_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default="Describe this image.")
    parser.add_argument("--keep-ratio", type=float, default=0.1)
    parser.add_argument("--candidate-ratio", type=float, default=0.3)
    parser.add_argument("--prune-layer", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AutoConfig.from_pretrained(args.model)
    config.text_config.hilo_keep_ratio = args.keep_ratio
    config.text_config.hilo_candidate_ratio = args.candidate_ratio
    config.text_config.hilo_prune_layer = args.prune_layer
    config.text_config.hilo_use_fusion = True
    config.text_config.hilo_fusion_strength = 0.3
    config.text_config.hilo_top_k_ratio = 0.1
    config.text_config.hilo_attention_threshold = 0.05
    config.text_config.hilo_fusion_layers = list(range(args.prune_layer))

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        config=config,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(args.model)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": args.image},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    raw_image = Image.open(args.image)
    patch_features = image_to_merged_patch_features(
        raw_image,
        inputs["image_grid_thw"][0].cpu(),
        patch_size=processor.image_processor.patch_size,
        merge_size=processor.image_processor.merge_size,
    ).to(model.device)

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            image_tensor=patch_features,
            max_new_tokens=args.max_new_tokens,
        )

    generated = [
        output[len(input_ids) :]
        for input_ids, output in zip(inputs.input_ids, generated)
    ]
    print(processor.batch_decode(generated, skip_special_tokens=True)[0])


if __name__ == "__main__":
    main()
