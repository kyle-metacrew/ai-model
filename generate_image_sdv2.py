#pip3 install flask diffusers torch accelerate transformers bitsandbytes protobuf sentencepiece
from flask import Flask, request, jsonify, send_file
# from diffusers import StableDiffusionPipeline
from diffusers import DiffusionPipeline
import torch
from io import BytesIO
import base64
from PIL import Image
from huggingface_hub import login
from accelerate import Accelerator
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers import StableDiffusion3Pipeline
from transformers import T5EncoderModel
from threading import Semaphore
import logging
import gc
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

login(token="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")

# Khởi tạo mô hình Stable Diffusion
# model_name = "stabilityai/stable-diffusion-3.5-large-turbo"

# Tự động kiểm tra GPU nếu có và tải mô hình lên đúng thiết bị
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Giới hạn số lượng yêu cầu song song
max_concurrent_requests = 2  # Chỉ cho phép 2 yêu cầu đồng thời
semaphore = Semaphore(max_concurrent_requests)

# API tạo hình ảnh từ prompt
@app.route('/generate', methods=['POST'])
def generate_image():
    # Kiểm tra và yêu cầu semaphore
    if not semaphore.acquire(blocking=False):
        logging.debug("Too many concurrent requests.")
        return jsonify({"error": "Too many requests, please try again later"}), 429  # Quá nhiều yêu cầu
    
    # Lấy prompt từ request
    data = request.get_json()
    prompt = data.get("prompt", "")
    negative_prompt = data.get("negative_prompt", "")
    height = data.get("height", 512)
    width = data.get("width", 512)
    num_inference_steps = data.get("num_inference_steps", 10)
    guidance_scale = data.get("guidance_scale", 0.0)
    model_name = data.get("model_name", "stabilityai/stable-diffusion-3.5-large-turbo")
    
    if not prompt:
        semaphore.release()
        return jsonify({"error": "Prompt is required"}), 400
    
    # Khởi tạo Accelerator
    accelerator = Accelerator()

    if (model_name == "stabilityai/stable-diffusion-3.5-large" or model_name == "stabilityai/stable-diffusion-3.5-large-turbo") and device == "cuda":
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
            t5_nf4 = T5EncoderModel.from_pretrained("diffusers/t5-nf4", torch_dtype=torch.bfloat16)
            pipe = StableDiffusion3Pipeline.from_pretrained(
                model_name, 
                transformer=model_nf4,
                text_encoder_3=t5_nf4,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True
            )
        else:
            pipe = StableDiffusion3Pipeline.from_pretrained(
                model_name, 
                transformer=model_nf4,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True
            )
        pipe.enable_model_cpu_offload()
    else:
        # Load model và chuẩn bị với Accelerator
        # pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32)
        pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16, use_auth_token=True, low_cpu_mem_usage=True)

    pipe = accelerator.prepare(pipe)

    pipe = pipe.to(device)  # Dùng GPU nếu có
    
    image = None
    try:
        # Tạo hình ảnh từ prompt
        image = pipe(prompt, 
                     negative_prompt=negative_prompt,
                     height=height, 
                     width=width,
                     num_inference_steps=num_inference_steps, # Chi tiết hình 
                     guidance_scale=guidance_scale, # Mối liên hệ hình ảnh tạo ra và prompt 
                     max_sequence_length=512).images[0]

        # Giải phóng bộ nhớ GPU sau khi tạo hình ảnh
        torch.cuda.empty_cache()
        
        # Chuyển hình ảnh sang Base64
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        # Giải phóng bộ nhớ GPU
        torch.cuda.empty_cache()
        gc.collect()  # Dọn dẹp thêm bộ nhớ
        
        # Chuyển hình ảnh thành Base64
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        
        return send_file(img_byte_arr, mimetype='image/png')
        # Trả về hình ảnh dưới dạng Base64
        return jsonify({
            "image": img_base64
        })
    
    except Exception as e:
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

