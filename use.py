from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

final_save_path=".\sfinal_saved_model"
model=AutoModelForCausalLM.from_pretrained(final_save_path)
tokenizer=AutoTokenizer.from_pretrained(final_save_path)
print("------模型加载完毕")

#构建推理的流程pipeline
from transformers import pipeline
pipe=pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

prompt="如何学会唱歌"
result=pipe(prompt,max_length=512,num_return_sequences=1)
print("开始回答：",result[0]["generated_text"])
