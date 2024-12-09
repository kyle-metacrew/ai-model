import { HfInference } from "@huggingface/inference";
import fs from 'fs';  // Import fs để ghi file
import path from 'path';  // Import path để xử lý đường dẫn
import { fileURLToPath } from 'url';  // Import fileURLToPath từ url module

// Lấy API token từ biến môi trường (có thể lưu trong file .env)
const HF_TOKEN = process.env.HUGGINGFACE_TOKEN;

// Mô phỏng __dirname trong ES Module
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Khởi tạo đối tượng inference
const inference = new HfInference("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");

// Hàm tạo hình ảnh từ văn bản và lưu tệp
async function generateImage() {
    try {
        const response = await inference.textToImage({
            model: "stabilityai/stable-diffusion-2",  // Mô hình Stable Diffusion mà bạn muốn sử dụng
            inputs: "award winning high resolution photo of a giant tortoise/((ladybird)) hybrid, [trending on artstation]",
            parameters: {
                negative_prompt: "blurry", // Prompt tiêu cực để tránh tạo hình ảnh mờ
            },
        });

        // Log phản hồi từ API
        console.log('API Response:', response);

        // Kiểm tra xem phản hồi có phải là một Blob
        if (response instanceof Blob) {
            // Lấy tên file và lưu nó trên ổ đĩa
            const filePath = path.join(__dirname, 'generated_image.jpg');
            
            // Chuyển Blob thành Buffer và ghi vào tệp
            const buffer = await response.arrayBuffer();
            fs.writeFileSync(filePath, Buffer.from(buffer));
            
            console.log('Image saved at:', filePath);
        } else {
            console.log('No image data received.');
        }
    } catch (error) {
        console.error('Error generating image:', error);
    }
}

// Gọi hàm generateImage
generateImage();
