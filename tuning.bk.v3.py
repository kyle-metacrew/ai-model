import os
import platform
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
    TrainingArguments,
    default_data_collator,
)
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, setup_chat_format
from huggingface_hub import login
from itertools import islice


### GLOBAL CONFIG ###
model_name = "google/gemma-2-2b-it"
local_file_path = "prepared_dataset.json"
new_model_path = "Gemma-2-9b-it-chat-doctor"
hf_token = "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"

# Cấu hình môi trường dựa trên hệ điều hành
device = (
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)
device = "cpu"


### FUNCTION DEFINITIONS ###

def configure_huggingface():
    """Đăng nhập Hugging Face và thiết lập môi trường."""
    login(token=hf_token)
    print(f"Using device: {device}")


def load_and_prepare_dataset(file_path, train_ratio=0.8, shuffle_seed=65, repeat=5):
    """Load và chuẩn bị dataset."""
    print("Loading dataset...")
    dataset = load_dataset("json", data_files=file_path, split="train", streaming=True)

    # Đếm tổng số dòng trong dataset
    total_size = sum(1 for _ in dataset)  # Đếm số dòng

    # Tính kích thước tập huấn luyện và kiểm tra
    train_size = int(0.8 * total_size)  # 80% cho train
    test_size = total_size - train_size  # 20% còn lại cho test

    # Reset iterator và chia tập dữ liệu
    dataset = load_dataset("json", data_files=file_path, split="train", streaming=True)
    train_dataset = list(islice(dataset, train_size))
    test_dataset = list(islice(dataset, test_size))

    # Tạo DatasetDict
    dataset = DatasetDict({
        "train": Dataset.from_list(train_dataset),
        "test": Dataset.from_list(test_dataset),
    })

    # Shuffle và repeat
    dataset["train"] = dataset["train"].shuffle(seed=shuffle_seed).flatten_indices()
    dataset["train"] = concatenate_datasets([dataset["train"]] * repeat)

    print(f"Training set size: {len(dataset['train'])}")
    print(f"Test set size: {len(dataset['test'])}")
    return dataset


def preprocess_data(row):
    # Kiểm tra và chuyển đổi input
    if not isinstance(row['input'], str):
        row['input'] = tokenizer.decode(row['input'], skip_special_tokens=True) if isinstance(row['input'], list) else str(row['input'])
    # Kiểm tra và chuyển đổi output
    if not isinstance(row['output'], str):
        row['output'] = tokenizer.decode(row['output'], skip_special_tokens=True) if isinstance(row['output'], list) else str(row['output'])

    # Tạo trường 'text' cho huấn luyện
    row['text'] = f"User: {row['input']}\nAssistant: {row['output']}"
    return row


def clean_dataset(dataset):
    """Loại bỏ các mẫu có giá trị None hoặc không phải chuỗi."""
    def is_valid(row):
        return row["input"] and row["output"]
    
    return dataset.filter(is_valid)


def configure_model(model_name, use_quantization=False, use_remote_code=False):
    """
    Cấu hình mô hình Transformer dựa trên:
    - Hệ điều hành (macOS, Linux, Windows)
    - Khả năng phần cứng (CPU, MPS, GPU NVIDIA)
    - Cấu hình bổ sung: lượng tử hóa (quantization), mã tùy chỉnh (trust_remote_code).

    Args:
        model_name (str): Tên mô hình trên Hugging Face Hub.
        use_quantization (bool): Bật lượng tử hóa 4-bit nếu khả dụng.
        use_remote_code (bool): Kích hoạt mã tùy chỉnh nếu mô hình yêu cầu.
    
    Returns:
        model, tokenizer, device: Mô hình, bộ tokenizer, và thiết bị được sử dụng.
    """
    # Phát hiện hệ điều hành
    os_type = platform.system().lower()
    device = "cpu"  # Mặc định là CPU
    torch_dtype = torch.float32  # Mặc định là float32
    attn_implementation = "eager"  # Mặc định cơ chế tính toán attention

    # Cấu hình lượng tử hóa nếu bật
    quantization_config = None
    if use_quantization and torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    # Kiểm tra khả năng phần cứng và thiết bị
    if os_type == "darwin":  # macOS
        if torch.backends.mps.is_available():
            # device = "mps"
            device = "cpu"
            torch_dtype = torch.float16  # MPS hỗ trợ float16
            print("Running on macOS with MPS")
        else:
            print("macOS detected, but MPS is not available. Using CPU.")
    elif torch.cuda.is_available():  # Linux/Windows với GPU NVIDIA
        device = "cuda"
        if torch.cuda.get_device_capability()[0] >= 8:
            torch_dtype = torch.bfloat16
            attn_implementation = "flash_attention_2"  # Tối ưu với Flash Attention
            print("Using Flash Attention 2 and bfloat16 for optimal performance")
        else:
            torch_dtype = torch.float16
            attn_implementation = "eager"  # Dùng cơ chế mặc định
            print("Using float16 with default attention implementation")
    else:
        print(f"{os_type.capitalize()} detected. Using CPU.")

    # Tải cấu hình mô hình
    config = AutoConfig.from_pretrained(model_name)

    # Lưu cấu hình vào thư mục mô hình
    config.save_pretrained(new_model_path)

    # Tải mô hình với các tùy chọn
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,  # Cấu hình lượng tử hóa nếu bật
        trust_remote_code=use_remote_code,       # Bật mã tùy chỉnh nếu cần
        torch_dtype=torch_dtype,                 # Định dạng dữ liệu tensor
        device_map="auto" if device in ["cuda", "mps"] else None,  # Phân bổ tự động
        low_cpu_mem_usage=True,  # Đảm bảo tối ưu CPU khi sử dụng device_map
        config=config
    )

    # Vô hiệu hóa gradient checkpointing nếu được bật
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
        print("Gradient checkpointing has been disabled.")

    # Tải tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        use_fast=True, 
        # trust_remote_code=use_remote_code
    )

    # Kiểm tra và thiết lập định dạng hội thoại
    if not hasattr(tokenizer, "chat_template"):
        model, tokenizer = setup_chat_format(model, tokenizer)

    device = "cpu"
    # Đưa mô hình về thiết bị
    model.to(device)

    print(f"Model loaded on device: {device}")
    print(f"Attention Implementation: {attn_implementation}")
    return model, tokenizer, device, attn_implementation


def train_model(model, tokenizer, dataset, output_dir, device):
    """Huấn luyện mô hình."""
    # Cấu hình LoRA (PEFT)
    peft_config = LoraConfig(
        r=16,  # Số rank (low-rank) trong LoRA
        lora_alpha=32,  # Hệ số nhân LoRA
        lora_dropout=0.05,  # Tỷ lệ dropout
        bias="none",  # Không áp dụng LoRA vào bias
        task_type="CAUSAL_LM",  # LoRA cho bài toán ngôn ngữ nhân quả (causal language modeling)
    )

    # Gắn PEFT vào mô hình
    model = get_peft_model(model, peft_config)

    # Tắt cache để tương thích với LoRA
    model.config.use_cache = False

    # Cấu hình TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4, # Tích lũy gradient để mô phỏng batch lớn hơn
        optim="adamw_torch",  # Trình tối ưu hóa AdamW
        num_train_epochs=5,
        eval_strategy="steps",  # Đánh giá mô hình sau mỗi số bước
        eval_steps=50,  # Số bước giữa các lần đánh giá
        logging_steps=10,
        warmup_steps=50,  # Số bước warmup giúp mô hình khởi động tốt hơn
        logging_strategy="steps",  # Log thông tin sau mỗi số bước
        learning_rate=2e-5,
        fp16=False, # macOS không hỗ trợ FP16 trên MPS. Nếu đang sử dụng GPU NVIDIA, hãy bật fp16
        bf16=False, # Với GPU hỗ trợ bfloat16, bật bf16
        group_by_length=True,
        max_grad_norm=1.0,  # Gradient clipping để tránh exploding gradients
        dataloader_num_workers=0,
        # remove_unused_columns=False,  # Không loại bỏ các cột trong tập dữ liệu
    )

    # Khởi tạo SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=512,  # Độ dài tối đa chuỗi input/output
        dataset_text_field="text", # Tên trường chứa dữ liệu văn bản trong dataset
        peft_config=peft_config,  # Cấu hình LoRA
        # data_collator=default_data_collator, # Hàm gộp dữ liệu
        packing=False,  # Không kích hoạt gộp chuỗi
    )

    # Huấn luyện mô hình
    model.train()
    trainer.train()

    # Quan sát log loss từng batch
    print("Training log history:")
    print(trainer.state.log_history)

    # Hợp nhất trọng số LoRA vào mô hình gốc
    print("Merging LoRA weights into the base model...")
    model = trainer.model.merge_and_unload()

    # Lưu mô hình và tokenizer
    print("Saving the fine-tuned model and tokenizer...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Model and tokenizer saved to: {output_dir}")
    return model, trainer


def test_model_response(prompt, max_length=1500):
    device = "cpu"
    model.to(device)
    input_text = f"User: {prompt}\nAssistant:"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # In ra thông tin thiết bị của mô hình và dữ liệu
    print(f"Model device: {model.device}")
    print(f"Input device: {inputs['input_ids'].device}")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assistant_response = generated_text.split("Assistant:")[-1].strip()
    return assistant_response


### MAIN EXECUTION ###

if __name__ == "__main__":
    # 1. Cấu hình Hugging Face
    configure_huggingface()

    # 2. Cấu hình mô hình và thiết bị (khởi tạo model và tokenizer)
    model, tokenizer, device, attn_implementation = configure_model(model_name)
    # Kiểm tra thiết lập
    print(f"Model loaded on device: {device}")
    print(f"Attention Implementation: {attn_implementation}")

    # 3. Load và chuẩn bị dataset
    dataset = load_and_prepare_dataset(local_file_path)

    # 4. Làm sạch dữ liệu
    dataset["train"] = clean_dataset(dataset["train"])
    dataset["test"] = clean_dataset(dataset["test"])
    # Kiểm tra kích thước tập dữ liệu sau làm sạch
    print("Train dataset size after cleaning:", len(dataset["train"]))
    print("Test dataset size after cleaning:", len(dataset["test"]))

    # 5. Tiền xử lý dataset (sử dụng tokenizer đã khởi tạo)
    dataset = dataset.map(
        preprocess_data,
        num_proc=4  # Tăng tốc bằng cách sử dụng đa tiến trình
    )
    print("Sample from train dataset:", dataset["train"][0])
    print("Sample from test dataset:", dataset["test"][0])

    # 6. Huấn luyện mô hình
    trainer = train_model(model, tokenizer, dataset, new_model_path, device)

    # 7. Kiểm tra mô hình trên dữ liệu huấn luyện
    print("\n===== Testing on Train Dataset =====")
    train_prompt = dataset["train"][0]["input"]
    train_response = test_model_response(train_prompt)
    print(f"Prompt: {train_prompt}")
    print(f"Expected: {dataset['train'][0]['output']}")
    print(f"Model Response: {train_response}")

    # 8. Kiểm tra mô hình trên dữ liệu kiểm tra
    print("\n===== Testing on Test Dataset =====")
    test_prompt = dataset["test"][0]["input"]
    test_response = test_model_response(test_prompt)
    print(f"Prompt: {test_prompt}")
    print(f"Expected: {dataset['test'][0]['output']}")
    print(f"Model Response: {test_response}")


