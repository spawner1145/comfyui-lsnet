import gradio as gr
from backend_lsnet.inference import process_image_from_pil
import os
import json
import glob

def get_available_models():
    """Get available model folders from models/lsnet/"""
    models_dir = "models/lsnet"
    if os.path.exists(models_dir):
        subdirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
        if subdirs:
            return subdirs

def get_available_checkpoints(model_name):
    """Get available checkpoint files for the model"""
    models_dir = "models/lsnet"
    model_dir = os.path.join(models_dir, model_name)
    if os.path.exists(model_dir):
        checkpoints = []
        for ext in ['*.pth', '*.ckpt', '*.safetensors']:
            checkpoints.extend(glob.glob(os.path.join(model_dir, ext)))
        return [os.path.basename(f) for f in checkpoints]
    return []

def get_available_csv(model_name):
    """Get available CSV files for the model"""
    models_dir = "models/lsnet"
    model_dir = os.path.join(models_dir, model_name)
    if os.path.exists(model_dir):
        csv_files = glob.glob(os.path.join(model_dir, "*.csv"))
        return [os.path.basename(f) for f in csv_files]
    return []

def get_checkpoint_path(model_name, checkpoint_name):
    """Get full checkpoint path"""
    models_dir = "models/lsnet"
    return os.path.join(models_dir, model_name, checkpoint_name)

def create_ui():
    css = """
    .contain-image img {
        object-fit: contain !important;
        width: 100% !important;
        height: 100% !important;
        background: #222;
    }
    """

    block = gr.Blocks(css=css)
    with block:
        gr.Markdown('# LSNet Artist Inference')
        with gr.Tabs():
            with gr.TabItem("Inference"):
                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(sources='upload', type="pil", label="Input Image", height=320, elem_classes="contain-image")
                        model = gr.Dropdown(
                            choices=get_available_models(),
                            label="Model Folder", value='Kaloscope'
                        )
                        device = gr.Dropdown(['cuda', 'cpu'], label="Device", value='cuda')
                        top_k = gr.Slider(label="Top K", minimum=1, maximum=20, value=5, step=1)
                        threshold = gr.Slider(label="Threshold", minimum=0.0, maximum=1.0, value=0.0, step=0.01)
                        infer_button = gr.Button(value="Infer")
                    with gr.Column():
                        tag_string = gr.Textbox(label="Formatted Tags", lines=3, interactive=False)
                        result_json = gr.Textbox(label="JSON Results", lines=15, interactive=False)
                        error_message = gr.Markdown("", visible=False)

        def infer(image, model, device, top_k, threshold):
            if image is None:
                return "Please upload an image.", "", gr.update(visible=True)
            checkpoints = get_available_checkpoints(model)
            if not checkpoints:
                return f"No checkpoints found for model {model}.", "", gr.update(visible=True)
            checkpoint_name = checkpoints[0]  # use first available
            checkpoint = get_checkpoint_path(model, checkpoint_name)
            if not os.path.exists(checkpoint):
                return f"Checkpoint not found: {checkpoint}", "", gr.update(visible=True)
            try:
                csv_files = get_available_csv(model)
                class_csv = None
                if csv_files:
                    class_csv = os.path.join("models/lsnet", model, csv_files[0])  # use first available
                
                # 自动从config.json读取model类型
                model_dir = os.path.join("models/lsnet", model)
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
                
                kwargs = {
                    'model': model_type,
                    'checkpoint': checkpoint,
                    'mode': 'classify',  # default to classify
                    'device': device,
                    'top_k': top_k,
                    'threshold': threshold,
                    'class_csv': class_csv
                }
                results = process_image_from_pil(image, **kwargs)
                tag_string = ",".join([r['class_name'] for r in results.get('classification', [])])
                json_str = json.dumps({r['class_name']: r['probability'] for r in results.get('classification', [])}, ensure_ascii=False)
                return tag_string, json_str, gr.update(visible=False)
            except Exception as e:
                return str(e), "", gr.update(visible=True)

        infer_button.click(
            infer,
            inputs=[input_image, model, device, top_k, threshold],
            outputs=[tag_string, result_json, error_message]
        )

    return block