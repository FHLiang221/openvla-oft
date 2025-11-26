import io, base64, numpy as np
import logging
import traceback
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# OpenVLA-OFT helpers
from types import SimpleNamespace
from experiments.robot.openvla_utils import (
    get_action_head, get_processor, get_proprio_projector, get_vla, get_vla_action
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- config: mirror your training flags ---
cfg = SimpleNamespace(
    # pretrained_checkpoint="/home/yachuanh/openvla-oft/checkpoints/success_oft/openvla-7b+kinova_coke_pick_same_start+b64+lr-0.0005+lora-r32+dropout-0.0--image_aug--test_kinova_oft--5000_chkpt",  # or your merged ckpt
    pretrained_checkpoint="/home/yachuanh/openvla-oft/checkpoints/success_oft/openvla-7b+kinova_coke_push_one_camera+b64+lr-0.0005+lora-r32+dropout-0.0--image_aug--test_kinova_oft--3000_chkpt",  # or your merged ckpt
    # pretrained_checkpoint="openvla/openvla-7b",  # or your merged ckpt
    use_l1_regression=True, use_diffusion=False, use_film=False,
    num_images_in_input=1, use_proprio=True,
    load_in_8bit=False, load_in_4bit=False,
    center_crop=True, num_open_loop_steps=NUM_ACTIONS_CHUNK,
    unnorm_key="kinova_coke_push_one_camera"
)

# Global variables for models
vla = None
processor = None
action_head = None
proprio_projector = None
models_loaded = False

def load_models():
    """Load all models with error handling"""
    global vla, processor, action_head, proprio_projector, models_loaded
    
    try:
        logger.info("Loading VLA model...")
        vla = get_vla(cfg)
        logger.info("VLA model loaded successfully")
        
        logger.info("Loading processor...")
        processor = get_processor(cfg)
        logger.info("Processor loaded successfully")
        
        logger.info("Loading action head...")
        action_head = get_action_head(cfg, llm_dim=vla.llm_dim)
        logger.info("Action head loaded successfully")
        
        logger.info("Loading proprio projector...")
        proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)
        logger.info("Proprio projector loaded successfully")
        
        models_loaded = True
        logger.info("All models loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        logger.error(traceback.format_exc())
        raise e

app = FastAPI(title="OpenVLA-OFT Inference Server")

class InferReq(BaseModel):
    image_b64: str        # base64 JPEG/PNG RGB
    instruction: str
    state: Optional[List[float]] = []  # 8D proprio if you have it; else empty
    return_chunk: bool = True   # True -> full chunk, False -> first action

class InferResp(BaseModel):
    actions: List[List[float]]  # [[dx,dy,dz,dRx,dRy,dRz,grip], ...] (len>=1)

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    try:
        load_models()
    except Exception as e:
        logger.error(f"Failed to load models on startup: {e}")
        # Don't raise - let the server start but return errors on inference

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if models_loaded else "models_not_loaded",
        "models_loaded": models_loaded
    }

@app.post("/infer", response_model=InferResp)
def infer(req: InferReq):
    """Main inference endpoint with comprehensive error handling"""
    
    # Check if models are loaded
    if not models_loaded:
        logger.error("Models not loaded")
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Validate inputs
        if not req.image_b64:
            raise HTTPException(status_code=400, detail="image_b64 is required")
        
        if not req.instruction:
            raise HTTPException(status_code=400, detail="instruction is required")
        
        # Decode image
        try:
            img_bytes = base64.b64decode(req.image_b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            logger.debug(f"Image decoded successfully: {img.size}")
        except Exception as e:
            logger.error(f"Failed to decode image: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid image_b64: {e}")
        
        # Process state
        try:
            if req.state and len(req.state) > 0:
                if len(req.state) != PROPRIO_DIM:
                    logger.warning(f"Expected state dimension {PROPRIO_DIM}, got {len(req.state)}")
                state = np.asarray(req.state, dtype=np.float32)
            else:
                state = np.zeros((PROPRIO_DIM,), np.float32)
            logger.debug(f"State processed: {state.shape}")
        except Exception as e:
            logger.error(f"Failed to process state: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid state: {e}")
        
        # Prepare observation
        try:
            obs = {
                "full_image": np.array(img), 
                "state": state, 
                "task_description": req.instruction
            }
            logger.debug(f"Observation prepared: image shape {obs['full_image'].shape}, state shape {obs['state'].shape}")
        except Exception as e:
            logger.error(f"Failed to prepare observation: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to prepare observation: {e}")
        
        # Run inference
        try:
            logger.info(f"Running inference for instruction: '{req.instruction}'")
            acts = get_vla_action(cfg, vla, processor, obs, req.instruction, action_head, proprio_projector)
            logger.info(f"Actions: {acts}")
            logger.info(f"Inference completed, got {len(acts)} actions")
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
        
        # Process output
        try:
            if not req.return_chunk:
                acts = acts[:1]
            
            # Convert to list of lists
            actions_list = [list(map(float, a)) for a in acts]
            logger.info(f"Returning {len(actions_list)} actions")
            
            return {"actions": actions_list}
            
        except Exception as e:
            logger.error(f"Failed to process output: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to process output: {e}")
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"Unexpected error in inference: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

# Run: uvicorn improved_serve_vla:app --host 0.0.0.0 --port 8000
