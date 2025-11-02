"""
画师风格模型推理脚本
支持两种模式：
1. 聚类模式：提取特征向量用于聚类
2. 分类模式：直接输出分类结果
"""
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.models import create_model

# Ensure custom LSNet artist models are registered with timm
from lsnet_model import lsnet_artist  # noqa: F401


def get_args_parser():
    parser = argparse.ArgumentParser('Artist Style Inference', add_help=False)
    
    # 模型参数
    parser.add_argument('--model', default='lsnet_t_artist', type=str,
                        choices=['lsnet_t_artist', 'lsnet_s_artist', 'lsnet_b_artist', 'lsnet_l_artist', 'lsnet_xl_artist', 'lsnet_xl_artist_448'],
                        help='Model architecture')
    parser.add_argument('--checkpoint', required=True, type=str,
                        help='Path to model checkpoint')
    parser.add_argument('--num-classes', default=None, type=int,
                        help='Number of classes. If omitted, will try to infer from checkpoint or CSV mapping.')
    parser.add_argument('--feature-dim', default=None, type=int,
                        help='Feature dimension')
    parser.add_argument('--input-size', default=224, type=int,
                        help='Input image size')
    
    # 推理模式
    parser.add_argument('--mode', default='classify', type=str,
                        choices=['classify', 'cluster', 'both'],
                        help='Inference mode: classify (with head), cluster (features only), or both')
    
    # 输入输出
    parser.add_argument('--input', required=True, type=str,
                        help='Input image path or directory')
    parser.add_argument('--output', default='./output/inference', type=str,
                        help='Output directory')
    parser.add_argument('--class-csv', default=None, type=str,
                        help='Path to class mapping CSV exported during training')
    
    # 其他参数
    parser.add_argument('--device', default='cuda', type=str,
                        help='Device to use')
    parser.add_argument('--batch-size', default=32, type=int,
                        help='Batch size for batch inference')
    parser.add_argument('--allow-head-reinit', action='store_true', default=False,
                        help='Allow re-initializing classification head when checkpoint classes mismatch')
    parser.add_argument('--top-k', default=5, type=int,
                        help='Number of top predictions to show (default: 5)')
    parser.add_argument('--threshold', default=0.0, type=float,
                        help='Probability threshold to filter predictions (default: 0.0)')
    
    return parser


def load_checkpoint_state(checkpoint_path: str):
    """加载 checkpoint 并返回模型权重"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            return checkpoint['model']
        if 'model_ema' in checkpoint:
            return checkpoint['model_ema']
    return checkpoint


def normalize_state_dict_keys(state_dict):
    """移除分布式训练前缀等冗余标记"""
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[len('module.'):]
        else:
            new_key = key
        normalized[new_key] = value
    return normalized


def resolve_num_classes(num_classes_arg: Optional[int],
                        class_mapping: Optional[Dict[int, str]],
                        state_dict) -> int:
    """根据参数、CSV 或 checkpoint 推断类别数"""
    # 优先使用CSV中的类别数
    if class_mapping:
        csv_classes = len(class_mapping)
        if num_classes_arg is not None and num_classes_arg != csv_classes:
            print(f"[Warning] 提供的 num_classes={num_classes_arg} 与 CSV 中的类别数 {csv_classes} 不一致，已使用 CSV 的值。")
        return csv_classes

    # 如果没有CSV，使用参数
    if num_classes_arg is not None:
        return num_classes_arg

    # 最后尝试从权重中解析分类头大小
    for key, value in state_dict.items():
        if key.endswith('head.weight') or key.endswith('head.l.weight'):
            return value.shape[0]

    raise ValueError('无法推断 num_classes，请提供 CSV 映射文件或显式指定 num_classes 参数。')


def resolve_feature_dim(feature_dim_arg: Optional[int], state_dict) -> int:
    """根据参数或 checkpoint 推断特征维度"""
    if feature_dim_arg is not None:
        return feature_dim_arg

    # 尝试从权重中解析特征维度
    # 查找head.bn.weight的维度，这通常是特征维度
    for key, value in state_dict.items():
        if key.endswith('head.bn.weight'):
            return value.shape[0]

    # 如果找不到，尝试查找其他可能的特征维度指示器
    for key, value in state_dict.items():
        if 'head' in key and 'weight' in key and len(value.shape) >= 2:
            # 对于线性层，输入维度通常是特征维度
            return value.shape[1] if len(value.shape) > 1 else value.shape[0]

    # 默认值
    print("[Warning] 无法从checkpoint推断特征维度，使用默认值384")
    return 384


def load_model(args, state_dict):
    """加载模型"""
    print(f"Loading model: {args.model}")
    state_dict = normalize_state_dict_keys(state_dict)

    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.num_classes,
        feature_dim=args.feature_dim,
    )

    model_state = model.state_dict()
    adapted_state = {}
    mismatched = {}

    for key, value in state_dict.items():
        if key in model_state:
            if model_state[key].shape != value.shape:
                mismatched[key] = (model_state[key].shape, value.shape)
                continue
        adapted_state[key] = value

    classifier_keys = [key for key in mismatched if 'head' in key or 'classifier' in key]
    other_mismatched = {key: shapes for key, shapes in mismatched.items() if key not in classifier_keys}

    if other_mismatched:
        details = '\n'.join([
            f"  - {key}: checkpoint {found} -> model {expected}"
            for key, (expected, found) in other_mismatched.items()
        ])
        raise RuntimeError(
            "以下权重尺寸与当前模型不兼容，且无法自动处理，请检查 checkpoint 或模型配置:\n" + details
        )

    require_strict_head = (args.mode in ['classify', 'both']) and not args.allow_head_reinit

    if classifier_keys and require_strict_head:
        details = '\n'.join([
            f"  - {key}: checkpoint {mismatched[key][1]} -> model {mismatched[key][0]}"
            for key in classifier_keys
        ])
        raise RuntimeError(
            "分类模式下检测到 checkpoint 分类头与当前 num_classes 不一致，已终止加载以避免随机初始化结果。\n"
            "请使用与训练数据一致的 checkpoint，或在确认需要重新初始化分类头时添加 --allow-head-reinit，"
            "或者切换到 --mode cluster 仅提取特征。\n" + details
        )

    if classifier_keys:
        print("[Warning] 分类头权重尺寸与当前 num_classes 不一致，将重新初始化以下权重：")
        for key in classifier_keys:
            expected, found = mismatched[key]
            print(f"  - {key}: checkpoint {found} -> model {expected}")
        # 冲突键已在上文过滤掉，无需额外处理

    load_result = model.load_state_dict(adapted_state, strict=False)

    if load_result.missing_keys:
        print(f"[Info] Missing keys during load: {load_result.missing_keys}")
    if load_result.unexpected_keys:
        print(f"[Info] Unexpected keys ignored: {load_result.unexpected_keys}")

    if classifier_keys and args.mode in ['classify', 'both']:
        if args.allow_head_reinit:
            print("[Warning] 分类模式在 --allow-head-reinit 下运行，分类头为随机初始化；为获得可靠结果请提供匹配的数据集 checkpoint。")
        else:
            print("[Info] 分类模式未开启或无需分类头，已忽略冲突的分类权重。")

    model.to(args.device)
    model.eval()

    print(f"Model loaded from {args.checkpoint}")
    return model


def load_class_mapping(class_csv_path: Optional[str]) -> Optional[Dict[int, str]]:
    """加载 CSV 类别映射，返回 class_id -> name 的字典"""
    if not class_csv_path:
        return None

    csv_path = Path(class_csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Class mapping CSV not found: {csv_path}")

    with csv_path.open('r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or 'class_id' not in reader.fieldnames or 'class_name' not in reader.fieldnames:
            raise ValueError('CSV 必须包含 class_id 和 class_name 两列。')

        mapping: Dict[int, str] = {}
        for row in reader:
            class_id = int(row['class_id'])
            class_name = row['class_name']
            mapping[class_id] = class_name

    if not mapping:
        raise ValueError(f"CSV {csv_path} 中未找到任何类别映射。")

    return mapping


def preprocess_image(image_path, transform):
    """预处理单张图像"""
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image)
    return tensor.unsqueeze(0)


def classify_image(model, image_tensor, device, class_mapping: Optional[Dict[int, str]] = None, top_k=5, threshold=0.0):
    """对图像进行分类"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        # 使用分类头
        logits = model(image_tensor, return_features=False)
        
        # 计算概率
        probs = F.softmax(logits, dim=-1)
        
        # Top-K结果
        top_probs, top_indices = torch.topk(probs, k=min(top_k, probs.size(-1)), dim=-1)
        
        results = []
        for prob, idx in zip(top_probs[0].cpu().numpy(), top_indices[0].cpu().numpy()):
            if prob >= threshold:
                class_name = class_mapping.get(int(idx), f"Class {idx}") if class_mapping else f"Class {idx}"
                results.append({
                    'class_id': int(idx),
                    'class_name': class_name,
                    'probability': float(prob)
                })
        
        # 如果过滤后结果少于 top_k，保持原样；否则取前 top_k
        if len(results) > top_k:
            results = results[:top_k]
        
        return results


def extract_features(model, image_tensor, device):
    """提取特征向量用于聚类"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        # 不使用分类头，直接返回特征
        features = model(image_tensor, return_features=True)
        return features.cpu().numpy()


def process_single_image(args, model, transform, class_mapping: Optional[Dict[int, str]] = None):
    """处理单张图像"""
    image_path = Path(args.input)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return
    
    print(f"\nProcessing: {image_path.name}")
    
    # 预处理
    image_tensor = preprocess_image(image_path, transform)
    
    results = {}
    
    # 分类模式
    if args.mode in ['classify', 'both']:
        print("\n[Classification Results]")
        classification = classify_image(model, image_tensor, args.device, class_mapping, args.top_k, args.threshold)
        results['classification'] = classification
        
        for i, result in enumerate(classification, 1):
            print(f"{i}. {result['class_name']}: {result['probability']:.4f}")
    
    # 聚类模式（提取特征）
    if args.mode in ['cluster', 'both']:
        print("\n[Feature Extraction]")
        features = extract_features(model, image_tensor, args.device)
        results['features'] = features[0].tolist()
        print(f"Feature vector shape: {features.shape}")
        print(f"Feature vector (first 10 dims): {features[0][:10]}")
    
    return results


def process_directory(args, model, transform, class_mapping: Optional[Dict[int, str]] = None):
    """批量处理目录中的图像"""
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"Error: Directory not found: {input_dir}")
        return
    
    # 支持的图像格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_paths = [p for p in input_dir.glob('**/*') if p.suffix.lower() in image_extensions]
    
    if not image_paths:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_paths)} images")
    
    all_results = {}
    
    # 批量处理
    for i in range(0, len(image_paths), args.batch_size):
        batch_paths = image_paths[i:i + args.batch_size]
        
        # 预处理批次
        batch_tensors = []
        for path in batch_paths:
            try:
                tensor = preprocess_image(path, transform)
                batch_tensors.append(tensor)
            except Exception as e:
                print(f"Error processing {path.name}: {e}")
                continue
        
        if not batch_tensors:
            continue
        
        batch_tensor = torch.cat(batch_tensors, dim=0)
        
        # 推理
        with torch.no_grad():
            batch_tensor = batch_tensor.to(args.device)
            
            if args.mode in ['classify', 'both']:
                logits = model(batch_tensor, return_features=False)
                probs = F.softmax(logits, dim=-1)
                top_probs, top_indices = torch.topk(probs, k=min(args.top_k, probs.size(-1)), dim=-1)
            
            if args.mode in ['cluster', 'both']:
                features = model(batch_tensor, return_features=True)
        
        # 保存结果
        for j, path in enumerate(batch_paths):
            if j >= len(batch_tensors):
                continue
            
            result = {'image': path.name}
            
            if args.mode in ['classify', 'both']:
                # 获取该图像的 top-k 结果
                img_top_probs = top_probs[j].cpu().numpy()
                img_top_indices = top_indices[j].cpu().numpy()
                
                classifications = []
                for prob, idx in zip(img_top_probs, img_top_indices):
                    if prob >= args.threshold:
                        class_id = int(idx)
                        class_name = class_mapping.get(class_id, f"Class {class_id}") if class_mapping else f"Class {class_id}"
                        classifications.append({
                            'class_id': class_id,
                            'class_name': class_name,
                            'probability': float(prob)
                        })
                
                # 如果需要，取前 top_k
                if len(classifications) > args.top_k:
                    classifications = classifications[:args.top_k]
                
                result['classification'] = classifications
            
            if args.mode in ['cluster', 'both']:
                result['features'] = features[j].cpu().numpy().tolist()
            
            all_results[path.name] = result
        
        print(f"Processed {min(i + args.batch_size, len(image_paths))}/{len(image_paths)} images")
    
    return all_results


def main(args):
    # 根据模型配置动态设置输入大小
    from lsnet_model.lsnet_artist import default_cfgs_artist
    if args.model in default_cfgs_artist:
        model_cfg = default_cfgs_artist[args.model]
        configured_input_size = model_cfg.get('input_size', (3, 224, 224))[1]  # 获取高度（假设正方形）
        if args.input_size != configured_input_size:
            args.input_size = configured_input_size
            print(f"Auto-setting input_size to {configured_input_size} for model {args.model} (from config)")
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.mode in ['classify', 'both'] and not args.class_csv:
        raise ValueError('分类或混合模式下必须提供 --class-csv，且需使用训练阶段导出的映射文件。')

    # 加载类别映射
    class_mapping = load_class_mapping(args.class_csv)

    # 加载 checkpoint 并解析类别数
    state_dict = load_checkpoint_state(args.checkpoint)
    state_dict = normalize_state_dict_keys(state_dict)
    args.num_classes = resolve_num_classes(args.num_classes, class_mapping, state_dict)
    args.feature_dim = resolve_feature_dim(args.feature_dim, state_dict)

    # 加载模型
    model = load_model(args, state_dict)
    
    # 创建数据转换
    config = resolve_data_config({'input_size': (3, args.input_size, args.input_size)}, model=model)
    transform = create_transform(**config)
    
    # 判断输入类型
    input_path = Path(args.input)
    
    if input_path.is_file():
        # 单张图像
        results = process_single_image(args, model, transform, class_mapping)
        
        # 保存结果
        output_file = output_dir / f"{input_path.stem}_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_file}")
        
    elif input_path.is_dir():
        # 目录批量处理
        results = process_directory(args, model, transform, class_mapping)
        
        # 保存结果
        output_file = output_dir / "batch_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_file}")
        
        # 如果是聚类模式，额外保存特征矩阵
        if args.mode in ['cluster', 'both']:
            features_list = []
            image_names = []
            for name, result in results.items():
                if 'features' in result:
                    features_list.append(result['features'])
                    image_names.append(name)
            
            if features_list:
                features_array = np.array(features_list)
                np.save(output_dir / "features.npy", features_array)
                with open(output_dir / "image_names.txt", 'w') as f:
                    f.write('\n'.join(image_names))
                print(f"Feature matrix saved: {output_dir / 'features.npy'}")
                print(f"Feature matrix shape: {features_array.shape}")
    else:
        print(f"Error: Invalid input path: {input_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Artist Style Inference', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
