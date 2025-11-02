import os
import json
import tempfile
from pathlib import Path
import torch
from PIL import Image
import numpy as np
from inference_artist import (
    get_args_parser, load_checkpoint_state, normalize_state_dict_keys,
    resolve_num_classes, resolve_feature_dim, load_model, process_single_image,
    load_class_mapping
)
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

def process_image(image_path, model='lsnet_t_artist', checkpoint='', num_classes=None, feature_dim=None, mode='classify', class_csv=None, device='cuda', top_k=5, threshold=0.0):
    """
    Process a single image for artist style inference.

    Args:
        image_path (str): Path to the input image
        model (str): Model architecture
        checkpoint (str): Path to model checkpoint
        num_classes (int): Number of classes
        feature_dim (int): Feature dimension
        mode (str): Inference mode ('classify', 'cluster', 'both')
        class_csv (str): Path to class mapping CSV
        device (str): Device to use
        top_k (int): Number of top predictions
        threshold (float): Probability threshold

    Returns:
        dict: Inference results
    """
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "output"
        output_dir.mkdir(exist_ok=True)

        # Prepare arguments
        args = get_args_parser().parse_args([
            '--model', model,
            '--checkpoint', checkpoint,
            '--input', image_path,
            '--output', str(output_dir),
            '--device', device,
            '--top-k', str(top_k),
            '--threshold', str(threshold),
            '--mode', mode
        ])

        if num_classes is not None:
            args.num_classes = num_classes
        if feature_dim is not None:
            args.feature_dim = feature_dim
        if class_csv is not None:
            args.class_csv = class_csv

        # Load checkpoint and state
        state_dict = load_checkpoint_state(checkpoint)
        state_dict = normalize_state_dict_keys(state_dict)

        # Load class mapping
        class_mapping = load_class_mapping(class_csv) if class_csv else None

        # Resolve num_classes
        args.num_classes = resolve_num_classes(num_classes, class_mapping, state_dict)

        # Resolve feature_dim
        args.feature_dim = resolve_feature_dim(feature_dim, state_dict)

        # Load model
        model_obj = load_model(args, state_dict)

        # 根据模型配置动态设置输入大小
        from lsnet_model.lsnet_artist import default_cfgs_artist
        if args.model in default_cfgs_artist:
            model_cfg = default_cfgs_artist[args.model]
            configured_input_size = model_cfg.get('input_size', (3, 224, 224))[1]  # 获取高度（假设正方形）
            if args.input_size != configured_input_size:
                args.input_size = configured_input_size
                print(f"Auto-setting input_size to {configured_input_size} for model {args.model}")

        # Prepare transform
        config = resolve_data_config({'input_size': (3, args.input_size, args.input_size)}, model=model_obj)
        transform = create_transform(**config)

        # Process single image
        results = process_single_image(args, model_obj, transform, class_mapping)

        return results

def process_image_from_pil(image, **kwargs):
    """
    Process a PIL image for artist style inference.

    Args:
        image (PIL.Image): Input image
        **kwargs: Other arguments for process_image

    Returns:
        dict: Inference results
    """
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        image.save(temp_file.name)
        try:
            return process_image(temp_file.name, **kwargs)
        finally:
            try:
                os.unlink(temp_file.name)
            except OSError:
                pass  # Ignore if file is still in use