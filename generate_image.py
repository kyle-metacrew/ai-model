from huggingface_hub import InferenceClient
client = InferenceClient("Jovie/Midjourney", token="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")

# output is a PIL.Image object
image = client.text_to_image("Astronaut riding a horse")

# Lưu ảnh vào file
image.save("astronaut_riding_horse.png")
