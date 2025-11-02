import base64
import logging
from typing import Callable
from threading import Lock
from secrets import compare_digest
from io import BytesIO
import asyncio
import concurrent.futures
import os
import glob
import json

from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field
from PIL import Image
import numpy as np
from backend_lsnet.inference import process_image_from_pil

try:
    from modules import shared
    from modules.call_queue import queue_lock as webui_queue_lock
    IN_WEBUI = True
except ImportError:
    IN_WEBUI = False
    shared = type('Shared', (), {'cmd_opts': type('CmdOpts', (), {'api_auth': None})()})()
    webui_queue_lock = None

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class InferenceRequest(BaseModel):
    input_image: str = Field(..., description="Input image as Base64 encoded string")
    model_name: str = Field('Kaloscope', description="Model name (subfolder in models/lsnet/)")
    device: str = Field('cuda', description="Device to use")
    top_k: int = Field(5, ge=1, le=20, description="Number of top predictions")
    threshold: float = Field(0.0, ge=0.0, le=1.0, description="Probability threshold")

class InferenceResponse(BaseModel):
    results: dict = Field(..., description="Inference results")
    info: str = Field(..., description="Additional information")

class CancelResponse(BaseModel):
    info: str = Field(..., description="Cancel operation result")

class Api:
    def __init__(self, app: FastAPI, queue_lock: Lock = None, prefix: str = "/lsnet/v1"):
        self.app = app
        self.queue_lock = queue_lock or Lock()
        self.prefix = prefix
        self.credentials = {}
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        if IN_WEBUI and shared.cmd_opts.api_auth:
            for auth in shared.cmd_opts.api_auth.split(","):
                user, password = auth.split(":")
                self.credentials[user] = password

        self.add_api_route(
            "infer",
            self.endpoint_infer,
            methods=["POST"],
            response_model=InferenceResponse,
            summary="Perform artist style inference",
            description="Classify or cluster an image using LSNet artist model."
        )
        self.add_api_route(
            "cancel",
            self.endpoint_cancel,
            methods=["POST"],
            response_model=CancelResponse,
            summary="Cancel the current inference task",
            description="Terminates the ongoing inference task."
        )

    def auth(self, creds: HTTPBasicCredentials = Depends(HTTPBasic())):
        if not self.credentials:
            return True
        if creds.username in self.credentials:
            if compare_digest(creds.password, self.credentials[creds.username]):
                return True
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"}
        )

    def add_api_route(self, path: str, endpoint: Callable, **kwargs):
        path = f"{self.prefix}/{path}" if self.prefix else path
        if self.credentials:
            return self.app.add_api_route(path, endpoint, dependencies=[Depends(self.auth)], **kwargs)
        return self.app.add_api_route(path, endpoint, **kwargs)

    def decode_base64_image(self, base64_str: str) -> Image.Image:
        try:
            img_data = base64.b64decode(base64_str, validate=True)
            img = Image.open(BytesIO(img_data)).convert("RGB")
            return img
        except base64.binascii.Error:
            raise HTTPException(400, "Invalid Base64 string format")
        except Exception as e:
            raise HTTPException(400, f"Failed to decode image: {str(e)}")

    async def run_inference(self, image, **kwargs):
        """Run inference in a separate thread"""
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(self.executor, lambda: process_image_from_pil(image, **kwargs))
        except Exception as e:
            logger.error(f"Inference execution failed: {str(e)}")
            raise

    async def endpoint_infer(self, req: InferenceRequest):
        logger.info(f"Received inference request: model_name={req.model_name}")
        try:
            with self.queue_lock:
                input_image = self.decode_base64_image(req.input_image)

                checkpoints = get_available_checkpoints(req.model_name)
                if not checkpoints:
                    raise HTTPException(400, f"No checkpoints found for model {req.model_name}")
                checkpoint_name = checkpoints[0]  # use first available
                checkpoint = get_checkpoint_path(req.model_name, checkpoint_name)
                if not os.path.exists(checkpoint):
                    raise HTTPException(400, f"Checkpoint not found: {checkpoint}")

                # Prepare inference arguments
                csv_files = get_available_csv(req.model_name)
                class_csv = None
                if csv_files:
                    class_csv = os.path.join("models/lsnet", req.model_name, csv_files[0])  # use first available
                
                # 自动从config.json读取model类型
                model_dir = os.path.join("models/lsnet", req.model_name)
                config_path = os.path.join(model_dir, "config.json")
                model_type = 'lsnet_xl_artist'  # 默认值
                if os.path.exists(config_path):
                    try:
                        with open(config_path, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                            if 'model' in config and config['model'] in ['lsnet_t_artist', 'lsnet_s_artist', 'lsnet_b_artist', 'lsnet_l_artist', 'lsnet_xl_artist', 'lsnet_xl_artist_448']:
                                model_type = config['model']
                                logger.info(f"Model type loaded from config: {model_type}")
                    except Exception as e:
                        logger.warning(f"Failed to load config.json: {e}")
                
                infer_args = {
                    "model": model_type,
                    "checkpoint": checkpoint,
                    "mode": "classify",  # default to classify
                    "device": req.device,
                    "top_k": req.top_k,
                    "threshold": req.threshold,
                    "class_csv": class_csv
                }

            # Run inference
            results = await self.run_inference(input_image, **infer_args)

            return InferenceResponse(results=results, info="Inference completed successfully")
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise HTTPException(500, f"Inference failed: {str(e)}")

    async def endpoint_cancel(self):
        # For simplicity, just return a message since inference is quick
        return CancelResponse(info="No active inference to cancel")

def on_app_started(demo, app):
    """Called when the webui app starts"""
    queue_lock = webui_queue_lock or Lock()
    api = Api(app, queue_lock)
    logger.info("LSNet API routes added to webui")