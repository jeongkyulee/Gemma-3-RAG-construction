from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline

model_id = "google/gemma-3-12b-it"  # 예: "gpt2", "skt/kogpt2-base-v2" 등

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

pipe = pipeline("intfloat/e5-large-v2", model=model, tokenizer=tokenizer, device=0)  # device=-1 이면 CPU

llm = HuggingFacePipeline(pipeline=pipe)
