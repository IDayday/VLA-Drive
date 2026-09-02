import math
import time
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def tile_metadata_from_image_size(
    width,
    height,
    min_num=1,
    max_num=12,
    image_size=448,
    use_thumbnail=True,
):
    """Return the exact normalized dynamic-tile geometry without RGB decode."""
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError(f"Image dimensions must be positive, got {(width, height)}")
    aspect_ratio = width / height
    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda ratio: ratio[0] * ratio[1],
    )
    grid_width, grid_height = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, width, height, image_size
    )
    metadata = []
    for index in range(grid_width * grid_height):
        column = index % grid_width
        row = index // grid_width
        metadata.append(
            [
                (column + 0.5) / grid_width,
                (row + 0.5) / grid_height,
                1.0 / grid_width,
                1.0 / grid_height,
                0.0,
            ]
        )
    if use_thumbnail and len(metadata) != 1:
        metadata.append([0.5, 0.5, 1.0, 1.0, 1.0])
    return torch.tensor(metadata, dtype=torch.float32)

def dynamic_preprocess(
    image,
    min_num=1,
    max_num=12,
    image_size=448,
    use_thumbnail=False,
    return_tile_metadata=False,
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    tile_metadata = []
    grid_width, grid_height = target_aspect_ratio
    for i in range(blocks):
        column = i % grid_width
        row = i // grid_width
        box = (
            column * image_size,
            row * image_size,
            (column + 1) * image_size,
            (row + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
        tile_metadata.append(
            [
                (column + 0.5) / grid_width,
                (row + 0.5) / grid_height,
                1.0 / grid_width,
                1.0 / grid_height,
                0.0,
            ]
        )
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
        tile_metadata.append([0.5, 0.5, 1.0, 1.0, 1.0])
    if return_tile_metadata:
        expected_metadata = tile_metadata_from_image_size(
            orig_width,
            orig_height,
            min_num=min_num,
            max_num=max_num,
            image_size=image_size,
            use_thumbnail=use_thumbnail,
        )
        actual_metadata = torch.tensor(tile_metadata, dtype=torch.float32)
        if not torch.equal(actual_metadata, expected_metadata):
            raise RuntimeError("Dynamic image tiles and cached tile geometry diverged")
        return processed_images, tile_metadata
    return processed_images

def load_image(
    image_file,
    input_size=448,
    max_num=12,
    return_tile_metadata=False,
):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    processed = dynamic_preprocess(
        image,
        image_size=input_size,
        use_thumbnail=True,
        max_num=max_num,
        return_tile_metadata=return_tile_metadata,
    )
    if return_tile_metadata:
        images, tile_metadata = processed
    else:
        images = processed
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    if return_tile_metadata:
        return pixel_values, torch.tensor(tile_metadata, dtype=torch.float32)
    return pixel_values


def load_image_with_timings(
    image_file,
    input_size=448,
    max_num=12,
    return_tile_metadata=True,
):
    """Load one image while separating decode and tensor-transform timings."""
    decode_start = time.perf_counter()
    with Image.open(image_file) as source:
        image = source.convert("RGB")
        image.load()
    decode_seconds = time.perf_counter() - decode_start

    transform_start = time.perf_counter()
    transform = build_transform(input_size=input_size)
    processed = dynamic_preprocess(
        image,
        image_size=input_size,
        use_thumbnail=True,
        max_num=max_num,
        return_tile_metadata=return_tile_metadata,
    )
    if return_tile_metadata:
        images, tile_metadata = processed
    else:
        images = processed
    pixel_values = torch.stack([transform(tile) for tile in images])
    transform_seconds = time.perf_counter() - transform_start
    if return_tile_metadata:
        return (
            pixel_values,
            torch.tensor(tile_metadata, dtype=torch.float32),
            decode_seconds,
            transform_seconds,
        )
    return pixel_values, decode_seconds, transform_seconds
