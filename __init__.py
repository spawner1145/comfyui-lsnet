import os
import sys
import json
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, Optional

sys.path.append(os.path.dirname(__file__))

import folder_paths

from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.models import create_model

from lsnet_model import lsnet_artist  # noqa: F401

from inference_artist import (
    load_checkpoint_state,
    normalize_state_dict_keys,
    resolve_num_classes,
    resolve_feature_dim,
    load_class_mapping
)

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class LSNetModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        base_dir = os.path.join(folder_paths.models_dir, 'lsnet')
        subfolders = []
        if os.path.exists(base_dir):
            subfolders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
        
        return {
            "required": {
                "model_folder": (subfolders, {"default": subfolders[0] if subfolders else ""}),
                "device": ("STRING", {"default": "cuda"}),
            }
        }

    RETURN_TYPES = ("LSNET_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "LSNet"

    def load(self, model_folder, device):
        base_dir = os.path.join(folder_paths.models_dir, 'lsnet')
        model_dir = os.path.join(base_dir, model_folder)
        checkpoint_path = os.path.join(model_dir, "best_checkpoint.pth")
        csv_path = os.path.join(model_dir, "class_mapping.csv")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Class mapping CSV not found: {csv_path}")
        class_mapping = load_class_mapping(csv_path)
        state_dict = load_checkpoint_state(checkpoint_path)
        state_dict = normalize_state_dict_keys(state_dict)
        num_classes = resolve_num_classes(None, class_mapping, state_dict)
        feature_dim = resolve_feature_dim(None, state_dict)
        
        # 自动从config.json读取model类型
        config_path = os.path.join(model_dir, "config.json")
        model_type = 'lsnet_xl_artist'  # 默认值
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    if 'model' in config and config['model'] in ['lsnet_t_artist', 'lsnet_s_artist', 'lsnet_b_artist', 'lsnet_l_artist', 'lsnet_xl_artist', 'lsnet_xl_artist_448']:
                        model_type = config['model']
                        print(f"Model type loaded from config: {model_type}")
            except Exception as e:
                print(f"Warning: Failed to load config.json: {e}")
        
        model = create_model(
            model_type,
            pretrained=False,
            num_classes=num_classes,
            feature_dim=feature_dim,
        )
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        
        # 根据模型配置动态设置输入大小
        from lsnet_model.lsnet_artist import default_cfgs_artist
        input_size = 224  # 默认值
        if model_type in default_cfgs_artist:
            model_cfg = default_cfgs_artist[model_type]
            configured_input_size = model_cfg.get('input_size', (3, 224, 224))[1]  # 获取高度（假设正方形）
            input_size = configured_input_size
            print(f"Auto-setting input_size to {input_size} for model {model_type}")
        
        config = resolve_data_config({'input_size': (3, input_size, input_size)}, model=model)
        transform = create_transform(**config)
        model_bundle = {
            'model': model,
            'transform': transform,
            'class_mapping': class_mapping,
            'device': device
        }

        return (model_bundle,)

class LSNetArtistInferenceNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("LSNET_MODEL",),
                "top_k": ("INT", {"default": 5, "min": 1, "max": 100}),
                "threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("tag_string", "json_output")
    FUNCTION = "process"
    CATEGORY = "LSNet"

    def process(self, image, model, top_k, threshold):
        model_bundle = model
        model = model_bundle['model']
        transform = model_bundle['transform']
        class_mapping = model_bundle['class_mapping']
        device = model_bundle['device']

        if image.ndim == 4:
            image = image[0]
        image = (image * 255).clamp(0, 255).byte().cpu().numpy()
        pil_image = Image.fromarray(image)

        # Preprocess image
        image_tensor = transform(pil_image).unsqueeze(0)  # Add batch dimension

        # Classify
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            logits = model(image_tensor, return_features=False)
            probs = F.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, k=min(top_k, probs.size(-1)), dim=-1)

            results = []
            for prob, idx in zip(top_probs[0].cpu().numpy(), top_indices[0].cpu().numpy()):
                if prob >= threshold:
                    class_id = int(idx)
                    class_name = class_mapping.get(class_id, f"Class {class_id}")
                    results.append({
                        'class_id': class_id,
                        'class_name': class_name,
                        'probability': float(prob)
                    })

            # Limit to top_k if more results after filtering
            if len(results) > top_k:
                results = results[:top_k]

        # Prepare outputs
        tags = [res['class_name'] for res in results]
        tag_string = ",".join(tags)
        tag_dict = {res['class_name']: res['probability'] for res in results}
        json_output = json.dumps(tag_dict, ensure_ascii=False)

        return (tag_string, json_output)

class LSNetArtistSimilarityNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "processed_image": ("IMAGE",),
                "reference_images": ("IMAGE",),
                "model": ("LSNET_MODEL",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("similarity_json",)
    FUNCTION = "process"
    CATEGORY = "LSNet"

    def process(self, processed_image, reference_images, model):
        model_bundle = model
        model = model_bundle['model']
        transform = model_bundle['transform']
        device = model_bundle['device']

        def image_to_tensor(img):
            if img.ndim == 4:
                img = img[0]
            img = (img * 255).clamp(0, 255).byte().cpu().numpy()
            pil_img = Image.fromarray(img)
            return transform(pil_img).unsqueeze(0)

        processed_tensor = image_to_tensor(processed_image)
        with torch.no_grad():
            processed_tensor = processed_tensor.to(device)
            processed_features = model(processed_tensor, return_features=True).cpu().numpy()[0]

        references = []
        similarities = []
        num_refs = reference_images.shape[0] if reference_images.ndim == 4 else 1
        for i in range(num_refs):
            ref_img = reference_images[i] if reference_images.ndim == 4 else reference_images
            ref_tensor = image_to_tensor(ref_img)
            with torch.no_grad():
                ref_tensor = ref_tensor.to(device)
                ref_features = model(ref_tensor, return_features=True).cpu().numpy()[0]
            references.append(ref_features.tolist())
            sim = np.dot(processed_features, ref_features) / (np.linalg.norm(processed_features) * np.linalg.norm(ref_features))
            similarities.append(float(sim))

        result = {
            "processed_features": processed_features.tolist(),
            "reference_features": references,
            "similarities": similarities
        }
        json_output = json.dumps(result, ensure_ascii=False)

        return (json_output,)

class LSNetCommonFeaturesNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "reference_images": ("IMAGE",),
                "model": ("LSNET_MODEL",),
            }
        }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("common_features",)
    FUNCTION = "process"
    CATEGORY = "LSNet"

    def process(self, reference_images, model):
        model_bundle = model
        model = model_bundle['model']
        transform = model_bundle['transform']
        device = model_bundle['device']

        def image_to_tensor(img):
            if img.ndim == 4:
                img = img[0]
            img = (img * 255).clamp(0, 255).byte().cpu().numpy()
            pil_img = Image.fromarray(img)
            return transform(pil_img).unsqueeze(0)

        references = []
        num_refs = reference_images.shape[0] if reference_images.ndim == 4 else 1
        for i in range(num_refs):
            ref_img = reference_images[i] if reference_images.ndim == 4 else reference_images
            ref_tensor = image_to_tensor(ref_img)
            with torch.no_grad():
                ref_tensor = ref_tensor.to(device)
                ref_features = model(ref_tensor, return_features=True).cpu().numpy()[0]
            references.append(ref_features)

        if references:
            common_features = np.mean(np.array(references), axis=0)
        else:
            common_features = np.zeros(384)
        return (torch.tensor(common_features),)

class LSNetClusteringNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "method": (["kmeans", "dbscan", "hierarchical"], {"default": "kmeans"}),
                "n_clusters": ("INT", {"default": 10, "min": 2, "max": 100}),
                "eps": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 10.0}),
                "min_samples": ("INT", {"default": 5, "min": 1, "max": 50}),
                "visualize": ("BOOLEAN", {"default": True}),
                "viz_method": (["tsne", "pca"], {"default": "tsne"}),
                "perplexity": ("INT", {"default": 30, "min": 5, "max": 100}),
            },
            "optional": {
                "group_1": ("TENSOR",),
                "group_2": ("TENSOR",),
                "group_3": ("TENSOR",),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("clustering_json", "visualization")
    FUNCTION = "cluster"
    CATEGORY = "LSNet"

    def cluster(self, method, n_clusters, eps, min_samples, visualize, viz_method, perplexity, group_1=None, group_2=None, group_3=None):
        groups = []
        group_sizes = []
        for g in [group_1, group_2, group_3]:
            if g is not None:
                groups.append(g.cpu().numpy())
                group_sizes.append(g.shape[0])

        if not groups:
            return (json.dumps({"error": "No groups provided"}), torch.zeros(1, 64, 64, 3))

        features_np = np.vstack(groups)
        if method == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            labels = clusterer.fit_predict(features_np)
            centers = clusterer.cluster_centers_
        elif method == "dbscan":
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
            labels = clusterer.fit_predict(features_np)
            centers = None
        elif method == "hierarchical":
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            labels = clusterer.fit_predict(features_np)
            centers = None

        result = {
            "method": method,
            "n_samples": len(features_np),
            "group_sizes": group_sizes,
            "labels": labels.tolist(),
        }
        if centers is not None:
            result["centers"] = centers.tolist()

        json_output = json.dumps(result, ensure_ascii=False)

        if visualize and len(features_np) > 1:
            if viz_method == "tsne":
                reducer = TSNE(n_components=2, perplexity=min(perplexity, len(features_np)-1), random_state=42)
            else:
                reducer = PCA(n_components=2, random_state=42)
            
            reduced_features = reducer.fit_transform(features_np)
            
            plt.figure(figsize=(10, 8))
            unique_labels = np.unique(labels)
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
            
            for label, color in zip(unique_labels, colors):
                mask = labels == label
                plt.scatter(reduced_features[mask, 0], reduced_features[mask, 1], 
                           color=color, label=f'Cluster {label}', alpha=0.7)
            
            plt.title(f'{method.upper()} Clustering ({viz_method.upper()})')
            plt.legend()
            plt.tight_layout()
            
            fig = plt.gcf()
            fig.canvas.draw()
            img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            pil_image = Image.fromarray(img_array)
            plt.close()
            
            viz_tensor = torch.from_numpy(np.array(pil_image)).float() / 255.0
            if viz_tensor.ndim == 3:
                viz_tensor = viz_tensor.unsqueeze(0)
        else:
            viz_tensor = torch.zeros(1, 64, 64, 3)

        return (json_output, viz_tensor)

class LSNetFeatureComparisonNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("LSNET_MODEL",),
            },
            "optional": {
                "group_1": ("TENSOR",),
                "group_2": ("TENSOR",),
                "group_3": ("TENSOR",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("comparison_json",)
    FUNCTION = "compare"
    CATEGORY = "LSNet"

    def compare(self, image, model, group_1=None, group_2=None, group_3=None):
        model_bundle = model
        model = model_bundle['model']
        transform = model_bundle['transform']
        device = model_bundle['device']

        if image.ndim == 4:
            image = image[0]
        image_np = (image * 255).clamp(0, 255).byte().cpu().numpy()
        pil_image = Image.fromarray(image_np)
        image_tensor = transform(pil_image).unsqueeze(0)

        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            query_features = model(image_tensor, return_features=True).cpu().numpy()[0]

        groups = []
        for g in [group_1, group_2, group_3]:
            if g is not None:
                groups.append(g.cpu().numpy())

        if not groups:
            return (json.dumps({"error": "No groups provided"}),)

        similarities = []
        for group_feat in groups:
            sim = np.dot(query_features, group_feat) / (np.linalg.norm(query_features) * np.linalg.norm(group_feat))
            similarities.append(float(sim))

        best_index = np.argmax(similarities)
        best_similarity = similarities[best_index]
        result = {
            "best_group_index": int(best_index),
            "best_similarity": best_similarity,
            "all_similarities": similarities
        }
        json_output = json.dumps(result, ensure_ascii=False)

        return (json_output,)

class LSNetArtistImageConnector:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("stacked_images",)
    FUNCTION = "connect"
    CATEGORY = "LSNet"

    def connect(self, image_1, image_2, image_3):
        def normalize_image(img):
            if img.ndim == 4:
                img = img[0]
            return img.unsqueeze(0)

        img1 = normalize_image(image_1)
        img2 = normalize_image(image_2)
        img3 = normalize_image(image_3)

        stacked = torch.cat([img1, img2, img3], dim=0)
        return (stacked,)

NODE_CLASS_MAPPINGS = {
    "LSNetModelLoader": LSNetModelLoader,
    "LSNetArtistInference": LSNetArtistInferenceNode,
    "LSNetArtistSimilarity": LSNetArtistSimilarityNode,
    "LSNetCommonFeatures": LSNetCommonFeaturesNode,
    "LSNetClustering": LSNetClusteringNode,
    "LSNetFeatureComparison": LSNetFeatureComparisonNode,
    "LSNetArtistImageConnector": LSNetArtistImageConnector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LSNetModelLoader": "LSNet Model Loader",
    "LSNetArtistInference": "LSNet Artist Inference",
    "LSNetArtistSimilarity": "LSNet Artist Similarity",
    "LSNetCommonFeatures": "LSNet Common Features",
    "LSNetClustering": "LSNet Clustering",
    "LSNetFeatureComparison": "LSNet Feature Comparison",
    "LSNetArtistImageConnector": "LSNet Image Connector"
}
