from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from data_prepare import samples
import json
import os
import torch
os.environ['HF_HOME'] = r'F:\deepseekModel'

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
cache_path="F:\deepseekModel"
tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir=cache_path)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_path)

#二. 准备数据
with open("data.jsonl", "w", encoding="utf-8") as f:
    for s in samples:
        json_line=json.dumps(s, ensure_ascii=False)
        f.write(json_line+"\n")
    else:
        print("prepare data finished")
        print(len(samples))

#三.准备训练集和验证集
from datasets import load_dataset
dataset = load_dataset(path="json",data_files={"train":"data.jsonl"},split="train")
print("数据数量:",len(dataset))

train_test_split=dataset.train_test_split(test_size=0.1)
train_dataset=train_test_split["train"]
eval_dataset=train_test_split["test"]

print(f"train_dataset size:{len(train_dataset)}")
print(f"eval_dataset size:{len(eval_dataset)}")

print("----------完成训练数据的准备工作")

#四.编写tokenizer处理工具 
def tokenize_function(examples):
    texts=[f"{prompt}\n{completion}" for prompt,completion in zip(examples["prompt"],examples["completion"])]
    tokenized=tokenizer(texts,max_length=512,truncation=True,padding="max_length")
    tokenized["labels"]=tokenized["input_ids"].copy()
    return tokenized

tokenized_train_dataset=train_dataset.map(tokenize_function,batched=True)
tokenized_eval_dataset=eval_dataset.map(tokenize_function,batched=True)
#print(tokenized_train_dataset[0])
print("----------完成tokenizer处理工具的编写")

#五.量化设置
quantization_config = BitsAndBytesConfig(load_in_4bit=True,
    bnb_4bit_quant_type="nf4",       # 4位量化类型
    bnb_4bit_use_double_quant=True,  # 嵌套量化进一步压缩
    bnb_4bit_compute_dtype=torch.float16)
model=AutoModelForCausalLM.from_pretrained(model_name,cache_dir=cache_path,quantization_config=quantization_config,device_map="auto")
print("----------完成量化设置")


#六.lora设置
from peft import LoraConfig,get_peft_model,TaskType
lora_config = LoraConfig(
   r=4,
   lora_alpha=8,
   lora_dropout=0.1,
   task_type=TaskType.CAUSAL_LM
)
model=get_peft_model(model,lora_config)
model.print_trainable_parameters()
print("------lora微调设置完毕")

#七
from transformers import TrainingArguments, Trainer
training_args = TrainingArguments(
    output_dir="./finetuned_models",
    num_train_epochs=2,
    per_device_train_batch_size=4, 
    fp16=True,
    logging_steps=10,
    save_strategy="steps",
    eval_steps=10,
    learning_rate=3e-5,
    logging_dir="./logs",
    run_name="deepseek-r1-distill-finetune"
)
print("------训练参数设置完毕")

trianer=Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
)
print("------开始训练")
trianer.train()
print("------训练完毕")

save_path=".\saved_model"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("------模型保存完毕")

#保存全量模型
final_save_path=".\sfinal_saved_model"
from peft import PeftModel
base_model=AutoModelForCausalLM.from_pretrained(model_name,cache_dir=cache_path)
model=PeftModel.from_pretrained(base_model,save_path)
model=model.merge_and_unload()

model.save_pretrained(final_save_path)
tokenizer.save_pretrained(final_save_path)
print("------全量模型保存完毕")
