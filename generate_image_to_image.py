#pip3 install flask diffusers torch accelerate transformers bitsandbytes protobuf sentencepiece xformers scipy
from flask import Flask, request, jsonify, send_file
from diffusers import StableDiffusionDepth2ImgPipeline, StableDiffusionXLImg2ImgPipeline
# from diffusers import DiffusionPipeline
import torch
from io import BytesIO
import base64
from PIL import Image
from huggingface_hub import login
from accelerate import Accelerator
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel, StableDiffusionImg2ImgPipeline, StableDiffusion3ControlNetPipeline
from diffusers import LMSDiscreteScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler, DDIMScheduler
from diffusers import StableDiffusion3Pipeline, StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, AutoPipelineForImage2Image
from diffusers.utils import load_image
from transformers import T5EncoderModel
from threading import Semaphore
import gc
import copy
import logging
import numpy as np
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

login(token="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")

# Khởi tạo mô hình Stable Diffusion
# model_name = "radames/stable-diffusion-2-depth-img2img"
# model_name = "stabilityai/stable-diffusion-xl-refiner-1.0"
# model_name = "stabilityai/stable-diffusion-2-1-base"
# model_name = "runwayml/stable-diffusion-v1-5"

# Tự động kiểm tra GPU nếu có và tải mô hình lên đúng thiết bị
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Giới hạn số lượng yêu cầu song song
max_concurrent_requests = 2  # Chỉ cho phép 2 yêu cầu đồng thời
semaphore = Semaphore(max_concurrent_requests)

# API tạo hình ảnh từ prompt
@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        # Kiểm tra và yêu cầu semaphore
        if not semaphore.acquire(blocking=False):
            raise ValueError("Too many requests, please try again later")
        
        # Lấy prompt từ request
        # data = request.get_json()
        data = request.form
        
        # url = data.get("url", "")
        prompt = data.get("prompt", "")
        negative_prompt = data.get("negative_prompt", "")
        model_name = data.get("model_name", "radames/stable-diffusion-2-depth-img2img")
        response_type = data.get("response_type", "binary")
        strength = data.get("strength", 1.0)
        scheduler = data.get("scheduler", "")
        sampling_steps = data.get("sampling_steps", 50)
        height = data.get("height", 1024)
        width = data.get("width", 1024)

        if not prompt:
            raise ValueError("Prompt are required.")
        
        if not isinstance(strength, (int, float)) and not strength.replace('.', '', 1).isdigit():
            raise ValueError("'strength' must be a valid number.")
        else:
            strength = float(strength)
        
        if not sampling_steps.isdigit():
            raise ValueError("'sampling_steps' must be an integer.")
        else:
            sampling_steps = int(sampling_steps)

        if not height.isdigit() or not width.isdigit():
            raise ValueError("'height' and 'width' must be integers.")
        else:
            height = int(height)
            width = int(width)
        
        strength = float(strength)  
        sampling_steps = int(sampling_steps) 
        height = int(height)
        width = int(width)
    except ValueError as e:
        semaphore.release()
        return jsonify({"error": str(e)}), 400
    
    file = request.files.get('file', None)
    
    # Khởi tạo Accelerator
    accelerator = Accelerator()

    if (model_name == "stabilityai/stable-diffusion-3.5-large" or 
        model_name == "stabilityai/stable-diffusion-3.5-large-turbo") and device == "cuda":
        # Cấu hình quantization 4-bit với nf4
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model_nf4 = SD3Transformer2DModel.from_pretrained(
            model_name,
            subfolder="transformer",
            quantization_config=nf4_config,
            torch_dtype=torch.bfloat16
        )

        if model_name == "stabilityai/stable-diffusion-3.5-large-turbo":
            if file and file.filename:
                pipe = AutoPipelineForImage2Image.from_pretrained(
                    model_name, 
                    transformer=model_nf4,
                    variant="fp16",
                    torch_dtype=torch.bfloat16,
                    use_safetensors=True,
                    low_cpu_mem_usage=True
                )
            else:
                t5_nf4 = T5EncoderModel.from_pretrained("diffusers/t5-nf4", torch_dtype=torch.bfloat16)
                pipe = StableDiffusion3Pipeline.from_pretrained(
                    model_name, 
                    transformer=model_nf4,
                    text_encoder_3=t5_nf4,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True
                )
                pipe.enable_model_cpu_offload()
                # pipe.enable_xformers_memory_efficient_attention()
        else:
            if file and file.filename:
                # controlnet = ControlNetModel.from_pretrained("CrucibleAI/ControlNetMediaPipeFace", torch_dtype=torch.float16, variant="fp16")
                pipe = AutoPipelineForImage2Image.from_pretrained(
                    model_name, 
                    transformer=model_nf4,
                    variant="fp16",
                    torch_dtype=torch.bfloat16,
                    # controlnet=controlnet, 
                    # safety_checker=None,
                    use_safetensors=True,
                    low_cpu_mem_usage=True
                )
            else:
                pipe = StableDiffusion3Pipeline.from_pretrained(
                    model_name, 
                    transformer=model_nf4,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True
                )
        
        if scheduler == "EulerDiscreteScheduler":
            # scheduler = EulerDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
            scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        elif scheduler == "EulerAncestralDiscreteScheduler":
            # scheduler = EulerAncestralDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
            scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        elif scheduler == "DPMSolverMultistepScheduler":
            # scheduler = DPMSolverMultistepScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
            scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        elif scheduler == "LMSDiscreteScheduler":
            # scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
            scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        elif scheduler == "DDIMScheduler":
            scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
            # scheduler = DDIMScheduler.from_config(pipe.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing")

        ## Initializing a scheduler and Setting number of sampling steps
        # scheduler.set_timesteps(sampling_steps)

        # scheduler_config = copy.deepcopy(pipe.scheduler.config)
        sample_size = pipe.config.sample_size
        noise = torch.randn((1, 3, sample_size, sample_size), device="cuda")
        # pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        # pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        if scheduler:
            pipe.scheduler = scheduler

        # pipe.enable_xformers_memory_efficient_attention()
    elif model_name == "runwayml/stable-diffusion-v1-5" and device == "cuda":
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
        )
    elif model_name == "stabilityai/stable-diffusion-2-1-base" and device == "cuda":
        controlnet = ControlNetModel.from_pretrained("CrucibleAI/ControlNetMediaPipeFace", torch_dtype=torch.float16, variant="fp16")
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
        )

        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        # pipe.enable_xformers_memory_efficient_attention()
    elif model_name == "radames/stable-diffusion-2-depth-img2img" and device == "cuda":
        pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-depth",
            torch_dtype=torch.float16,
        )
    else:
        # Load model và chuẩn bị với Accelerator
        # pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32)
        # pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(model_name,
        #                                                         torch_dtype=torch.float16, 
        #                                                         # use_auth_token=True, 
        #                                                         # low_cpu_mem_usage=True,
        #                                                         # variant="fp16", 
        #                                                         # use_safetensors=True
        #                                                         )
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_name,
                                                                torch_dtype=torch.float16, 
                                                                # use_auth_token=True, 
                                                                # low_cpu_mem_usage=True,
                                                                variant="fp16", 
                                                                use_safetensors=True
                                                                )
        
    if file and file.filename:
        # Load ảnh ban đầu từ URL
        try:
            # init_image = load_image(url).convert("RGB")
            # Đọc ảnh từ file upload và chuyển đổi sang RGB
            init_image = Image.open(file).convert("RGB")

            max_size = 1024  # Giới hạn độ phân giải tối đa
            width, height = init_image.size
            
            # Thay đổi kích thước sao cho chiều rộng và chiều cao là bội số của 64
            new_width = (width // 64) * 64
            new_height = (height // 64) * 64

            # Nếu kích thước ảnh đã thay đổi
            if new_width != width or new_height != height:
                init_image = init_image.resize((new_width, new_height))

            # init_image = pipe.preprocess(init_image)
        except Exception as e:
            semaphore.release()
            return jsonify({"error": f"Error loading image from URL: {str(e)}"}), 400

    pipe = accelerator.prepare(pipe)

    # pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

    pipe = pipe.to(device)  # Dùng GPU nếu có

    num_inference_steps = 30
    guidance_scale = 5.0
    if model_name == "stabilityai/stable-diffusion-3.5-large-turbo":
        guidance_scale = 0.0
    
    image = None
    try:
        torch_seed = np.random.randint(low=-1000000000, high=1000000000, dtype=np.int64)
        print(f'seed: {torch_seed}')
        generator = torch.Generator("cuda").manual_seed(int(torch_seed))

        if file and file.filename:
            # Tạo hình ảnh từ prompt
            image = pipe(prompt=prompt, 
                        negative_prompt=negative_prompt, 
                        image=init_image, 
                        #  height=512, 
                        #  width=512,
                        num_inference_steps=num_inference_steps, # Chi tiết hình ảnh
                        guidance_scale=guidance_scale,
                        #  max_sequence_length=512,
                        strength=strength, # Thay đổi so với ảnh gốc 
                        generator=generator
                        ).images[0]
        else:
            # Tạo hình ảnh từ prompt
            image = pipe(prompt=prompt, 
                        negative_prompt=negative_prompt, 
                        height=height, 
                        width=width,
                        num_inference_steps=num_inference_steps, # Chi tiết hình ảnh
                        guidance_scale=guidance_scale,
                        max_sequence_length=512,
                        generator=generator
                        ).images[0]

        # Giải phóng bộ nhớ GPU sau khi tạo hình ảnh
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Chuyển hình ảnh sang Base64
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Chuyển hình ảnh thành Base64
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        
        if response_type == 'base64':
            # Trả về hình ảnh dưới dạng Base64
            semaphore.release()
            return jsonify({
                "image": img_base64
            })
        else:
            semaphore.release()
            return send_file(img_byte_arr, mimetype='image/png')
    
    except Exception as e:
        semaphore.release()
        del image
        del pipe
        torch.cuda.empty_cache()
        gc.collect()
        return jsonify({"error": str(e)}), 500
    
    finally:
        logging.debug("Releasing semaphore...")
        semaphore.release()  # Giải phóng semaphore sau khi xử lý xong

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

