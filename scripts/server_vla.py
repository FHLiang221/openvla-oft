import io, base64, numpy as np
import logging
import traceback
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

# OpenVLA-OFT helpers
from types import SimpleNamespace
from experiments.robot.openvla_utils import (
    get_action_head, get_processor, get_proprio_projector, get_vla, get_vla_action
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM

# -----------------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server_vla_dualcam")

# -----------------------------------------------------------------------------------
# Config: mirror your training flags
# IMPORTANT: set num_images_in_input=2 for the dual-cam model/checkpoint.
# -----------------------------------------------------------------------------------
cfg = SimpleNamespace(
    # Point this at your dual-cam finetuned or merged checkpoint
    # pretrained_checkpoint="/home/yachuanh/openvla-oft/checkpoints/success_oft/openvla-7b+kinova_coke_push_dualcam+b64+lr-0.0005+lora-r32+dropout-0.0--image_aug--test_kinova_oft--3000_chkpt",
    # pretrained_checkpoint="openvla/openvla-7b",
    # pretrained_checkpoint="/home/yachuanh/openvla-oft/checkpoints/success_oft/openvla-7b+oct21_push_coke_sponge+b64+lr-0.0005+lora-r32+dropout-0.0--image_aug--test_kinova_oft--10000_chkpt",
    pretrained_checkpoint="/home/yachuanh/openvla-oft/checkpoints/success_oft/grad1/openvla-7b+nov5_pick_up_blue_cup+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--nov5_lora32--10000_chkpt",
    # pretrained_checkpoint="/home/yachuanh/openvla-oft/checkpoints/success_oft/2images/openvla-7b+oct21_push_coke_sponge+b64+lr-0.0005+lora-r32+dropout-0.0--image_aug--test_kinova_oft--5000_chkpt",
    # pretrained_checkpoint="/home/yachuanh/openvla-oft/checkpoints/success_oft/2images/openvla-7b+oct27_pick_up_blue_cup+b64+lr-0.0005+lora-r32+dropout-0.0--image_aug--test_kinova_oft--5000_chkpt+oct27_pick_up_blue_cup+b64+lr-0.0005+lora-r32+dropout-0.0--image_aug--test_kinova_oft--5000_chkpt",
    # pretrained_checkpoint="/home/yachuanh/openvla-oft/checkpoints/success_oft/b64/openvla-7b+oct27_pick_up_blue_cup+b256+lr-0.0005+lora-r32+dropout-0.0--image_aug--test_kinova_oft--2000_chkpt",
    # pretrained_checkpoint="/home/yachuanh/openvla-oft/checkpoints/success_oft/openvla-7b+kinova_coke_push+b64+lr-0.0005+lora-r32+dropout-0.0--image_aug--test_kinova_oft--2000_chkpt",
    # pretrained_checkpoint="/home/yachuanh/openvla-oft/checkpoints/success_oft/openvla-7b+kinova_coke_push+b64+lr-0.0005+lora-r32+dropout-0.0--image_aug--test_kinova_oft--2000_chkpt+kinova_coke_push+b64+lr-0.0005+lora-r32+dropout-0.0--image_aug--test_kinova_oft--5000_chkpt",
    # pretrained_checkpoint="/home/yachuanh/openvla-oft/checkpoints/success_oft/openvla-7b+kinova_coke_push+b64+lr-0.0005+lora-r32+dropout-0.0--image_aug--test_kinova_oft--2000_chkpt+kinova_coke_push+b64+lr-0.0005+lora-r32+dropout-0.0--image_aug--test_kinova_oft--5000_chkpt",

    use_l1_regression=True, use_diffusion=False, use_film=False,
    num_images_in_input=2,           # <— set 2 for dualcam, 1 for single cam
    use_proprio=True,
    load_in_8bit=False, load_in_4bit=False,
    center_crop=True, num_open_loop_steps=NUM_ACTIONS_CHUNK,

    # this should match the key you used when unnormalizing actions in training
    # e.g., if you trained on "kinova_coke_push" dual-cam rlds:
    # unnorm_key="oct21_push_coke_sponge",
    unnorm_key="nov5_pick_up_blue_cup",
)

# -----------------------------------------------------------------------------------
# Models (lazy-loaded at startup)
# -----------------------------------------------------------------------------------
vla = None
processor = None
action_head = None
proprio_projector = None
models_loaded = False

def load_models():
    """Load all models with robust logging."""
    global vla, processor, action_head, proprio_projector, models_loaded
    try:
        logger.info("Loading VLA model...")
        vla = get_vla(cfg)
        logger.info("VLA model loaded")

        logger.info("Loading processor...")
        processor = get_processor(cfg)
        logger.info("Processor loaded")

        logger.info("Loading action head...")
        action_head = get_action_head(cfg, llm_dim=vla.llm_dim)
        logger.info("Action head loaded")

        logger.info("Loading proprio projector...")
        proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)
        logger.info("Proprio projector loaded")

        models_loaded = True
        logger.info("All models loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        logger.error(traceback.format_exc())
        raise e

# -----------------------------------------------------------------------------------
# FastAPI
# -----------------------------------------------------------------------------------
app = FastAPI(title="OpenVLA-OFT Inference Server (dual-cam ready)")

class InferReq(BaseModel):
    # Back-compat single-camera input:
    image_b64: Optional[str] = Field(default=None, description="Primary camera image (base64)")

    # Dual-camera convenience field:
    wrist_image_b64: Optional[str] = Field(default=None, description="Wrist camera image (base64)")

    # Alternative: pass all images explicitly (len must match cfg.num_images_in_input)
    images_b64: Optional[List[str]] = Field(default=None, description="List of images as base64; len=1 or 2")

    instruction: str
    state: Optional[List[float]] = Field(default_factory=list, description="8D proprio; zeros used if absent")
    return_chunk: bool = True   # True -> full chunk, False -> first action

    action_sum_zero: int = 0

class InferResp(BaseModel):
    actions: List[List[float]]  # [[dx,dy,dz,dRx,dRy,dRz,grip], ...]

@app.on_event("startup")
async def startup_event():
    try:
        load_models()
    except Exception as e:
        logger.error(f"Failed to load models on startup: {e}")
        # Let the server start; /infer will return 503 until loaded.

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if models_loaded else "models_not_loaded",
        "models_loaded": models_loaded,
        "expects_num_images": cfg.num_images_in_input,
        "use_proprio": cfg.use_proprio,
    }

# -----------------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------------
def _decode_b64_to_rgb(b64: str) -> Image.Image:
    """Decode base64 (with or without data URL prefix) to a PIL RGB image."""
    if b64 is None:
        raise ValueError("Missing image data")
    # Strip data URL prefix if present
    if "," in b64 and b64.strip().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    img_bytes = base64.b64decode(b64)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")

def _prepare_images(req: InferReq) -> List[np.ndarray]:
    """
    Build the list of images to feed the model, honoring cfg.num_images_in_input.
    Accepts:
      - images_b64 (list)
      - or image_b64 (+ optional wrist_image_b64)
    """
    expected = int(cfg.num_images_in_input)
    imgs: List[Image.Image] = []

    if req.images_b64 is not None:
        if not isinstance(req.images_b64, list) or len(req.images_b64) == 0:
            raise HTTPException(status_code=400, detail="images_b64 must be a non-empty list")
        for s in req.images_b64:
            imgs.append(_decode_b64_to_rgb(s))

    else:
        if not req.image_b64:
            raise HTTPException(status_code=400, detail="image_b64 (or images_b64) is required")
        imgs.append(_decode_b64_to_rgb(req.image_b64))
        if req.wrist_image_b64:
            imgs.append(_decode_b64_to_rgb(req.wrist_image_b64))

    # Adjust to expected count
    if len(imgs) < expected:
        logger.warning(f"Received {len(imgs)} image(s) but model expects {expected}. "
                       "Duplicating the primary image to fill missing views.")
        while len(imgs) < expected:
            imgs.append(imgs[0])

    if len(imgs) > expected:
        logger.warning(f"Received {len(imgs)} image(s) but model expects {expected}. "
                       f"Using the first {expected} images.")
        imgs = imgs[:expected]

    # PIL->np RGB
    return [np.array(im) for im in imgs]

def _prepare_state(state_list: Optional[List[float]]) -> np.ndarray:
    """Validate proprio size and build a PROPRIO_DIM vector (zeros if missing)."""
    if cfg.use_proprio:
        if state_list and len(state_list) > 0:
            if len(state_list) != PROPRIO_DIM:
                logger.warning(f"Expected state dim {PROPRIO_DIM}, got {len(state_list)}; will clip/pad.")
            x = np.asarray(state_list, dtype=np.float32)
            if x.shape[0] > PROPRIO_DIM:
                x = x[:PROPRIO_DIM]
            elif x.shape[0] < PROPRIO_DIM:
                x = np.pad(x, (0, PROPRIO_DIM - x.shape[0]), mode="constant", constant_values=0)
            return x
        else:
            return np.zeros((PROPRIO_DIM,), np.float32)
    else:
        return np.zeros((PROPRIO_DIM,), np.float32)

# -----------------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------------
@app.post("/infer", response_model=InferResp)
def infer(req: InferReq):
    if not models_loaded:
        logger.error("Models not loaded")
        raise HTTPException(status_code=503, detail="Models not loaded")

    # Validate instruction early
    if not req.instruction or not isinstance(req.instruction, str):
        raise HTTPException(status_code=400, detail="instruction is required")

    try:
        # Decode images (1 or 2 depending on cfg.num_images_in_input)
        np_images = _prepare_images(req)  # List[np.ndarray], len = 1 or 2
        state = _prepare_state(req.state)

        # Build observation dict mirroring training-time keys the utils expect.
        # During training, RLDSBatchTransform standardized to 'full_image' and optional 'wrist_image'.
        obs = {
            "full_image": np_images[0],
            "state": state,
            "task_description": req.instruction,
        }
        if cfg.num_images_in_input > 1:
            obs["wrist_image"] = np_images[1]

        # Run policy
        logger.info(f"Inference: instr='{req.instruction[:96]}' "
                    f"images={len(np_images)} state_dim={len(state)}")
        acts = get_vla_action(cfg, vla, processor, obs, req.instruction, action_head, proprio_projector)

        # Truncate if only first action requested
        if not req.return_chunk:
            acts = acts[:1]

        # Convert to plain Python
        actions_list = [list(map(float, a)) for a in acts]
        # sum of the actions in each dimension
        actions_sum = np.sum(actions_list, axis=0)
        # if np.sum(np.abs(actions_sum)) < 0.001:
        #     action_sum_zero += 1
        print(f"Actions sum: {actions_sum}")
        # print(f"Action sum zero: {action_sum_zero}")

        return {"actions": actions_list}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
