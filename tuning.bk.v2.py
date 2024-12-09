from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
from huggingface_hub import login
import os, torch, wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig, DataCollatorForSeq2Seq, default_data_collator
from datasets import load_dataset, DatasetDict
from trl import SFTTrainer, setup_chat_format, SFTConfig
from datasets import concatenate_datasets, Dataset, DatasetDict
from itertools import islice
import random
# import bitsandbytes as bnb

modelName = "google/gemma-2-2b-it"
dataset_name = "lavita/ChatDoctor-HealthCareMagic-100k"
new_model = "Gemma-2-9b-it-chat-doctor"
local_file_path = "prepared_dataset.json"

# load_dotenv()

# hf_token = os.getenv("HF_TOKEN")
hf_token = "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"
login(token = hf_token)

# Sử dụng các kỹ thuật giảm kích thước như 4-bit quantization
# cấu hình BitsAndBytesConfig không tương thích với macos. yêu cầu GPU Nvidia với CUDA 
# bnbConfig = BitsAndBytesConfig(
#     load_in_4bit = True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )

# Mục đích: Tối ưu hóa hiệu suất của mô hình Transformer dựa trên khả năng của GPU.
# Nếu GPU hỗ trợ CUDA >= 8:
# Sử dụng Flash Attention 2 và kiểu dữ liệu bfloat16 để đạt hiệu suất cao nhất.
# Nếu GPU không hỗ trợ CUDA >= 8:
# Sử dụng kiểu dữ liệu float16 và cơ chế mặc định (eager) để đảm bảo tính tương thích.
# if torch.cuda.get_device_capability()[0] >= 8:
#     torch_dtype = torch.bfloat16
#     attn_implementation = "flash_attention_2"
# else:
#     torch_dtype = torch.float16
#     attn_implementation = "eager"

# tải cấu hình mô hình từ hugging face model hub 
modelConfig = AutoConfig.from_pretrained(modelName)

# Lưu cấu hình vào thư mục mô hình
modelConfig.save_pretrained(new_model)

# tải mô hình được huấn luyện trước với các tham số 
model = AutoModelForCausalLM.from_pretrained(
    modelName,
    # quantization_config=bnbConfig,
    # config=modelConfig,
    # device_map="auto",
    # trust_remote_code=True,
    # attn_implementation=attn_implementation
    low_cpu_mem_usage=True, # Khi True, các trọng số mô hình sẽ được tải tuần tự và chỉ chuyển vào bộ nhớ khi cần, thay vì tải toàn bộ vào RAM một lúc.
    # return_dict=True,
    torch_dtype=torch.float16, # Sử dụng floating point 16-bit để lưu trọng số, giúp giảm dung lượng bộ nhớ và tăng tốc tính toán trên GPU.
    # device_map="cpu",
)

# tokenizer chịu trách nhiệm chuyển đổi văn bản thành chuỗi số để mô hình xử lý và ngược lại 
tokenizer = AutoTokenizer.from_pretrained(
    modelName, 
    # trust_remote_code=True
    use_fast=True
    )

# Tìm tất cả module trong model có kiểu bnb.nn.Linear4bit
# def find_all_linear_names(model):
#     cls = bnb.nn.Linear4bit
#     lora_module_names = set()
#     for name, module in model.named_modules():
#         if isinstance(module, cls):
#             names = name.split('.')
#             lora_module_names.add(names[0] if len(names) == 1 else names[-1])
#     if 'lm_head' in lora_module_names:  # needed for 16 bit
#         lora_module_names.remove('lm_head')
#     return list(lora_module_names)

# modules = find_all_linear_names(model)

# LoRA config
# Áp dụng fine-tuning lên model 
peft_config = LoraConfig(
    r=16, #16
    lora_alpha=32, #32
    lora_dropout=0.05, #0.05
    bias="none",
    task_type="CAUSAL_LM"
    # target_modules=modules
)

if not hasattr(tokenizer, "chat_template"):
    model, tokenizer = setup_chat_format(model, tokenizer)
model = get_peft_model(model, peft_config)

#Importing the dataset
# dataset = load_dataset(dataset_name, split="all")
# dataset = load_dataset("json", data_files=local_file_path, split="train")

# Đọc dataset từng dòng (streaming=True)
dataset = load_dataset("json", data_files=local_file_path, split="train", streaming=True)

# Đếm tổng số dòng trong dataset
total_size = sum(1 for _ in dataset)  # Đếm số dòng

# Tính kích thước tập huấn luyện và kiểm tra
train_size = int(0.8 * total_size)  # 80% cho train
test_size = total_size - train_size  # 20% còn lại cho test

# Reset lại iterator để chia dataset
dataset = load_dataset("json", data_files=local_file_path, split="train", streaming=True)

# Chia dataset thành train và test
train_dataset = list(islice(dataset, train_size))
test_dataset = list(islice(dataset, 1000 - train_size))

# Tạo DatasetDict
dataset = DatasetDict({
    "train": Dataset.from_list(train_dataset),
    "test": Dataset.from_list(test_dataset),
})

# Chia dữ liệu thành train và test
# train_test_split = dataset.train_test_split(test_size=0.2, seed=42)

# Tạo DatasetDict
# dataset = DatasetDict({
#     "train": train_test_split["train"],
#     "test": train_test_split["test"]
# })

print("Train Data:", dataset["train"])
print("Test Data:", dataset["test"])

# Shuffle và chọn 1000 mẫu cho mỗi tập con
# dataset["train"] = dataset["train"].shuffle(seed=65).select(range(4))
# dataset["test"] = dataset["test"].shuffle(seed=65).select(range(2))  # Số lượng mẫu cho test có thể nhỏ hơn

# Kiểm tra kích thước thực tế của dataset
train_size = len(dataset["train"])
test_size = len(dataset["test"])

dataset["train"] = dataset["train"].shuffle(seed=65).select(range(min(train_size, 16)))
dataset["test"] = dataset["test"].shuffle(seed=65).select(range(min(test_size, 4)))

# dataset = dataset.shuffle(seed=65).select(range(1000)) # Only use 1000 samples for quick demo

# Lặp lại dữ liệu huấn luyện nhiều lần để mô hình "ưu tiên" chúng hơn kiến thức gốc
# Flatten indices nếu cần thiết (loại bỏ chỉ số bị xáo trộn trước đó)
dataset["train"] = dataset["train"].flatten_indices()

# Lặp lại dữ liệu huấn luyện 10 lần
dataset["train"] = concatenate_datasets([dataset["train"]] * 5)

print(f"Số lượng mẫu sau khi lặp lại: {len(dataset['train'])}")

def format_chat_template(row):
    row_json = [
                # {"role": "system", "content": row["instruction"]},
                {"role": "user", "content": row["input"]},
                {"role": "assistant", "content": row["output"]}]
    # row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    row["text"] = f"User: {row['input']}\nAssistant: {row['output']}"
    return row

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

# Áp dụng hàm format_chat_template lên từng mẫu trong tập dữ liệu.
dataset = dataset.map(
    preprocess_data,
    num_proc= 4, # Sử dụng đa tiến trình để tiết kiệm bộ nhớ
)

print(dataset["train"][0])
print(dataset["test"][0])

# for split_name in ["train", "test"]:
#     for row in dataset[split_name]:
#         try:
#             format_chat_template(row)
#         except Exception as e:
#             print(f"Error with row in {split_name}: {row}")
#             print(str(e))

# Chuyển mô hình sang CPU hoặc MPS
device = "mps" if torch.backends.mps.is_available() else "cpu"
# device = "cpu"  # Hoặc "mps"
model = model.to(device)

# model.gradient_checkpointing_enable() giúp tiết kiệm bộ nhớ GPU nhưng lại gây thêm tải trên CPU (do cần tính lại một số phần của mô hình trong quá trình huấn luyện).
model.gradient_checkpointing_disable()

# Chuẩn bị data collator
# data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
data_collator = default_data_collator

# Setting Hyperparamter
training_arguments = TrainingArguments(
    output_dir=new_model, # Thư mục để lưu mô hình sau khi huấn luyện.
    per_device_train_batch_size=1, # Kích thước batch cho mỗi GPU/CPU trong quá trình huấn luyện.
    per_device_eval_batch_size=1, # Kích thước batch cho mỗi GPU/CPU trong quá trình đánh giá.
    gradient_accumulation_steps=2, # Tăng tích lũy gradient để bù lại batch nhỏ
    # optim="paged_adamw_32bit",
    optim="adamw_torch", # bitsandbytes không hỗ trợ GPU trên macOS, bạn nên chuyển sang trình tối ưu hóa tiêu chuẩn như AdamW.
    num_train_epochs=3, # tăng nếu dataset lớn hơn hoặc nếu muốn hiệu quả huấn luyện cao hơn. giúp mô hình học tập trung hơn vào dữ liệu huấn luyện
    eval_strategy="steps",
    eval_steps=50,
    logging_steps=10,
    # save_steps=100,
    warmup_steps=100, # Warmup có thể giúp mô hình khởi động tốt hơn
    logging_strategy="steps",
    learning_rate=5e-5, # Tăng learning_rate để mô hình học nhanh hơn trên dataset nhỏ.
    fp16=False, # macOS không hỗ trợ FP16 trên MPS. Nếu đang sử dụng GPU NVIDIA, hãy bật fp16
    bf16=False, # Với GPU hỗ trợ bfloat16, bật bf16
    group_by_length=True,
    max_grad_norm=1.0, # Gradient clipping giúp ổn định quá trình huấn luyện
    # report_to="wandb" # Gửi thông tin huấn luyện đến Weights & Biases (WandB), một công cụ theo dõi quá trình huấn luyện.
    dataloader_num_workers=0,
)

# Setting sft parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    max_seq_length= 512, # Độ dài tối đa của chuỗi
    dataset_text_field="text", # Tên trường văn bản trong dataset
    tokenizer=tokenizer,
    args=training_arguments,
    packing= False,
    # num_workers=4, # Đặt num_workers cao hơn trong quá trình chuẩn bị dữ liệu
)

# Thông báo cho WandB rằng quá trình theo dõi hiện tại đã kết thúc.
# wandb.finish()
# Tắt caching trong mô hình. Việc này cần thiết khi sử dụng kỹ thuật LoRA, 
# vì caching không phù hợp với các module bổ sung trong LoRA.
model.config.use_cache = False
model.train()
trainer.train()

# Quan sát loss từng batch để kiểm tra liệu mô hình có đang học tốt không
print(trainer.state.log_history)

# Merge LoRA vào mô hình gốc
model = trainer.model.merge_and_unload()

model.save_pretrained(new_model) # Lưu trọng số và cấu hình mô hình 
tokenizer.save_pretrained(new_model) # Lưu các tệp liên quan đến tokenizer tokenizer.json, tokenizer_config.json

# Chia sẻ mô hình trên Hugging Face Hub để sử dụng hoặc triển khai trên Cloud.
# model.push_to_hub(new_model, use_temp_dir=False)

def test_model_response(prompt, max_length=1000):
    device = "cpu"
    model.to(device)
    input_text = f"User: {prompt}\nAssistant:"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assistant_response = generated_text.split("Assistant:")[-1].strip()
    return assistant_response

# Kiểm tra trên dữ liệu huấn luyện 
train_prompt = dataset["train"][0]["input"]
train_response = test_model_response(train_prompt)
print(f"Prompt: {train_prompt}")
print(f"Expected: {dataset['train'][0]['output']}")
print(f"Model Response: {train_response}")

# Kiểm tra trên dữ liệu kiểm tra
test_prompt = dataset["test"][0]["input"]
test_response = test_model_response(test_prompt)
print(f"Prompt: {test_prompt}")
print(f"Expected: {dataset['test'][0]['output']}")
print(f"Model Response: {test_response}")

#python3 ./convert_hf_to_gguf.py ../Gemma-2-9b-it-chat-doctor --outfile gemma-2-2b.gguf --outtype f32