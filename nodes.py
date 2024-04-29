import os
import torch

from omegaconf import OmegaConf
from .models.depth_normal_pipeline_clip import DepthNormalEstimationPipeline
from .models.unet_2d_condition import UNet2DConditionModel
from diffusers import  DDIMScheduler, DDPMScheduler, DEISMultistepScheduler, PNDMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler, AutoencoderKL


import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from torchvision import transforms
from contextlib import  nullcontext

import model_management as mm
import comfy.utils
import folder_paths
script_directory = os.path.dirname(os.path.abspath(__file__))

def convert_dtype(dtype_str):
    if dtype_str == 'fp32':
        return torch.float32
    elif dtype_str == 'fp16':
        return torch.float16
    elif dtype_str == 'bf16':
        return torch.bfloat16
    else:
        raise NotImplementedError
    
scheduler_mapping = {
                'DDIMScheduler': DDIMScheduler,
                'DDPMScheduler': DDPMScheduler,
                'DEISMultistepScheduler': DEISMultistepScheduler,
                'PNDMScheduler': PNDMScheduler,
                'DPMSolverMultistepScheduler': DPMSolverMultistepScheduler,
                'EulerDiscreteScheduler': EulerDiscreteScheduler
            }
def get_scheduler_class(scheduler_str, model_path, subfolder='scheduler'):
    if scheduler_str in scheduler_mapping:
        return scheduler_mapping[scheduler_str].from_pretrained(model_path, subfolder=subfolder)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_str}")
    
class geowizard_model_loader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "vae": ("VAE",),
            "dtype": (
                    [
                        'fp32',
                        'fp16',
                    ], {
                        "default": 'fp16'
                    }),
            
            },
        }

    RETURN_TYPES = ("GEOWIZMODEL",)
    RETURN_NAMES = ("geowizard_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "Geowizard"

    def loadmodel(self, vae, dtype):
        mm.soft_empty_cache()
 
        custom_config = {
            'dtype': dtype,
            'vae': vae
        }

        if not hasattr(self, 'model') or self.model == None or custom_config != self.current_config:
            self.current_config = custom_config
            # setup pretrained models
            original_config = OmegaConf.load(os.path.join(script_directory, f"configs/v1-inference.yaml"))

            dtype = convert_dtype(dtype)
            
            from diffusers.loaders.single_file_utils import (convert_ldm_vae_checkpoint, create_vae_diffusers_config)
            
            sd = vae.get_sd()
            converted_vae_config = create_vae_diffusers_config(original_config, image_size=512)
            converted_vae = convert_ldm_vae_checkpoint(sd, converted_vae_config)
            self.vae = AutoencoderKL(**converted_vae_config)
            self.vae.load_state_dict(converted_vae, strict=False)


            model_path = os.path.join(folder_paths.models_dir,'diffusers', 'geowizard')
            if not os.path.exists(model_path):
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id="lemonaddie/geowizard", ignore_patterns=["*vae*", "*.ckpt", "*.pt", "*.png", "*non_ema*", "*safety_checker*", "*.bin"], 
                                    local_dir=model_path, local_dir_use_symlinks=False)
            
            unet = UNet2DConditionModel.from_pretrained(model_path, subfolder='unet')
            scheduler = DDIMScheduler.from_pretrained(model_path, subfolder='scheduler')
            feature_extractor = CLIPImageProcessor.from_pretrained(model_path, subfolder='feature_extractor')
            image_enc = CLIPVisionModelWithProjection.from_pretrained(model_path, subfolder='image_encoder')

            self.model = DepthNormalEstimationPipeline(
                            vae=self.vae,
                            image_encoder=image_enc,
                            feature_extractor=feature_extractor,
                            unet=unet,
                            scheduler=scheduler)
            self.model = self.model.to(dtype)
            if mm.XFORMERS_IS_AVAILABLE:
                self.model.enable_xformers_memory_efficient_attention()

        return (self.model,)
    
class geowizard_sampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "geowizard_model": ("GEOWIZMODEL",),
            "image": ("IMAGE",),
            "steps": ("INT", {"default": 10, "min": 1, "max": 200, "step": 1}),
            "ensemble_size": ("INT", {"default": 3, "min": 1, "max": 200, "step": 1}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "domain": (
            [   
                'outdoor',
                'indoor',
                'object'
            ], {
               "default": 'indoor'
            }),
            "scheduler": (
                    [
                        'DDIMScheduler',
                        'DDPMScheduler',
                        'DEISMultistepScheduler',
                        'PNDMScheduler',
                        'DPMSolverMultistepScheduler',
                        'EulerDiscreteScheduler'
                    ], {
                        "default": 'DDIMScheduler'
                    }),
            "keep_model_loaded": ("BOOLEAN", {"default": True}),
            },
    
        }

    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("depth", "normal",)
    FUNCTION = "process"
    CATEGORY = "Geowizard"

    def process(self, geowizard_model, image, domain, ensemble_size, steps, seed, scheduler, keep_model_loaded):
        device = mm.get_torch_device()
        mm.unload_all_models()
        mm.soft_empty_cache()
        pipe = geowizard_model
        dtype = pipe.dtype

        torch.manual_seed(seed)
        model_path = os.path.join(folder_paths.models_dir,'diffusers', 'geowizard')
        pipe.scheduler = get_scheduler_class(scheduler, model_path)
        
        autocast_condition = (dtype != torch.float32) and not mm.is_device_mps(device)
        with torch.autocast(mm.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
            image = image.permute(0, 3, 1, 2)

            B, C, H, W = image.shape
            ratio = 8
            orig_H, orig_W = H, W
            if W % ratio != 0:
                W = W - (W % ratio)
            if H % ratio != 0:
                H = H - (H % ratio)
            if orig_H % ratio != 0 or orig_W % ratio != 0:
                image = F.interpolate(image, size=(H, W), mode="bilinear")
           
            B, C, H, W = image.shape

            pipe = pipe.to(device)
            depth_maps = []
            normal_maps = []
            if B > 1:
                batch_pbar = comfy.utils.ProgressBar(B)
            for img in image:
                pipe_out = pipe(
                    img,
                    device=device,
                    denoising_steps=steps,
                    ensemble_size=ensemble_size,
                    processing_res=H,
                    batch_size=0,
                    domain=domain,
                    show_progress_bar=True,
                )

                depth = pipe_out.depth_pred_tensor
                depth = depth.repeat(3,1,1).unsqueeze(0)
                depth_maps.append(depth)
                normal_colored = pipe_out.normal_colored
                normal_tensor = transforms.ToTensor()(normal_colored).unsqueeze(0)
                normal_maps.append(normal_tensor)
                if B > 1:
                    batch_pbar.update(1)
            
            depth_out = torch.cat(depth_maps, dim=0)
            depth_out = torch.clamp(depth_out, 0.0, 1.0)
            depth_out = 1.0 - depth_out

            normal_out = torch.cat(normal_maps, dim=0)

            if depth_out.shape[2] != orig_H or depth_out.shape[3] != orig_W:
                print("Restoring original dimensions: ", orig_W,"x",orig_H)
                depth_out = F.interpolate(depth_out, size=(orig_H, orig_W), mode="bicubic")
                normal_out = F.interpolate(normal_out, size=(orig_H, orig_W), mode="bicubic")

            depth_out = depth_out.permute(0, 2, 3, 1).cpu().to(torch.float32)
            normal_out = normal_out.permute(0, 2, 3, 1).cpu().to(torch.float32)
            
            if not keep_model_loaded:
                pipe = pipe.to('cpu')

            return (depth_out, normal_out)
class geowizard_image_embed:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "geowizard_model": ("GEOWIZMODEL",),
            "image": ("IMAGE",),
            "keep_model_loaded": ("BOOLEAN", {"default": True}),
            },    
        }
    RETURN_TYPES = ("CLIP_VISION_OUTPUT",)
    FUNCTION = "process"
    CATEGORY = "Geowizard"
    def process(self, geowizard_model, image, keep_model_loaded):
        device = mm.get_torch_device()
        mm.unload_all_models()
        mm.soft_empty_cache()
        pipe = geowizard_model
        pipe = pipe.to(device)
        dtype = pipe.dtype
        image = image.permute(0, 3, 1, 2)
        rgb_norm = image.to(device) * 2.0 - 1.0
        rgb_norm = torch.clamp(rgb_norm, min=-1.0, max=1.0)
        autocast_condition = (dtype != torch.float32) and not mm.is_device_mps(device)
        with torch.autocast(mm.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
            rtn = pipe.encode_img_embed(rgb_norm) * 0.5 + 0.5
        if not keep_model_loaded:
                pipe = pipe.to('cpu')
        return (rtn,)
    

class geowizard_depth_postprocess:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "ensemble_size": ("INT", {"default": 1, "min": 1, "max": 200, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "Geowizard"
    def process(self, image, ensemble_size):
        # image = torch.clip(image, -1.0, 1.0)
        device = mm.get_torch_device()
        rtn = []
        for img in image:
            img = img.to(device)
            img = torch.mean(img,2)
            m = img.min()
            M = img.max()
            img = (img - m) / (M - m)
            rtn.append(img.cpu()[None,...,None].repeat(1,1,1,3))
        return (torch.cat(rtn),)
        ##for i in range(0, image.size(0), ensemble_size):

class geowizard_normals_postprocess:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "ensemble_size": ("INT", {"default": 1, "min": 1, "max": 200, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "Geowizard"
    def process(self, image, ensemble_size):
        # image = torch.clip(image, -1.0, 1.0)
        device = mm.get_torch_device()
        image = image.to(device) * 2 - 1
        image = image / (torch.norm(image, p=2, dim=3, keepdim=True)+1e-5)
        image = image*0.5 + 0.5
        return (image.cpu(),)

class geowizard_latent_sampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "geowizard_model": ("GEOWIZMODEL",),
            "image_embedding": ("CLIP_VISION_OUTPUT",),
            "latent_image": ("LATENT",),
            "steps": ("INT", {"default": 10, "min": 1, "max": 200, "step": 1}),
            "ensemble_size": ("INT", {"default": 1, "min": 1, "max": 200, "step": 1}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "domain": (
            [   
                'outdoor',
                'indoor',
                'object'
            ], {
               "default": 'indoor'
            }),
            "scheduler": (
                    [
                        'DDIMScheduler',
                        'DDPMScheduler',


                        'PNDMScheduler',
                        'DPMSolverMultistepScheduler',
                        'EulerDiscreteScheduler'
                    ], {
                        "default": 'DDIMScheduler'
                    }),
            "keep_model_loaded": ("BOOLEAN", {"default": False}),
            },
    
        }

    RETURN_TYPES = ("LATENT", "LATENT",)
    RETURN_NAMES = ("depth", "normal",)
    FUNCTION = "process"
    CATEGORY = "Geowizard"
    def process(self, geowizard_model, image_embedding, latent_image, domain, ensemble_size, steps, seed, scheduler, keep_model_loaded):
        device = mm.get_torch_device()
        mm.unload_all_models()
        mm.soft_empty_cache()
        pipe = geowizard_model
        dtype = pipe.dtype

        torch.manual_seed(seed)
        model_path = os.path.join(folder_paths.models_dir,'diffusers', 'geowizard')
        pipe.scheduler = get_scheduler_class(scheduler, model_path)
        
        autocast_condition = (dtype != torch.float32) and not mm.is_device_mps(device)
        with torch.autocast(mm.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
            latents = latent_image["samples"]
            B = latents.size(0) * ensemble_size
            pipe = pipe.to(device)
            depth_maps = []
            normal_maps = []
            if B > 1:
                batch_pbar = comfy.utils.ProgressBar(B)
            
            for i in range(latents.size(0)):
                for j in range(ensemble_size):
                    depth, normal = pipe.latent_sample(
                        device,
                        latents[i:i+1],
                        image_embedding[i:i+1],
                        steps,
                        domain,
                        True,
                    )

                    depth_maps.append(depth.cpu())
                    normal_maps.append(normal.cpu())
                    if B > 1:
                        batch_pbar.update(1)
            
            depth_out = torch.cat(depth_maps, dim=0)
            normal_out = torch.cat(normal_maps, dim=0)
            
            if not keep_model_loaded:
                pipe = pipe.to('cpu')
            return ({"samples":depth_out}, {"samples":normal_out})

NODE_CLASS_MAPPINGS = {
    "geowizard_model_loader": geowizard_model_loader,
    "geowizard_sampler": geowizard_sampler,
    "geowizard_image_embed": geowizard_image_embed,
    "geowizard_latent_sampler": geowizard_latent_sampler,
    "geowizard_depth_postprocess": geowizard_depth_postprocess,
    "geowizard_normals_postprocess": geowizard_normals_postprocess,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "geowizard_model_loader": "Geowizard Model Loader",
    "geowizard_sampler": "Geowizard Sampler",
    "geowizard_image_embed": "Geowizard Get Image Embedding",
    "geowizard_latent_sampler": "Geowizard Latent Sampler",
    "geowizard_depth_postprocess": "Geowizard Depth Postprocess",
    "geowizard_normals_postprocess": "Geowizard Normals Postprocess",
}
