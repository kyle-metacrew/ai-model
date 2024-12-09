from diffusers import StableDiffusionPipeline
import torch

model_name = "runwayml/stable-diffusion-v1-5"

# Dùng torch.float32 thay vì torch.float16 trên CPU
pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32)

# Chạy mô hình trên CPU
pipe = pipe.to("cpu")

prompt = "a photo of an astronaut riding a horse on mars"

# Tạo hình ảnh với kích thước nhỏ hơn
image = pipe(prompt, height=512, width=512).images[0]
image.save("generated_image.png")