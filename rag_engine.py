from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import os

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = True
torch._dynamo.config.disable = True  
torch._C._jit_set_texpr_fuser_enabled(False)

template = """
You are a professional chatbot specialized in Korean construction safety manuals.  
Always respond strictly in the following structured format using clear, concise, and technical English.  
Respond only in English under all circumstances. Do not include any Korean text.  
Do not display or reference the original information source.

Respond in the following order:
1. Relevant Regulation Summary: Summarize the key content extracted from the context in English, even if the original text is in Korean.  
2. Expert Answer: Provide an accurate, technically sound response based on the provided context or general construction safety knowledge.  
3. Recommendation: Suggest follow-up actions, best practices, or specific regulations that may be relevant.

[Question]  
{Question}

[Context]  
{Context}

"""



prompt = PromptTemplate(
    input_variables=["Question", "Context"],
    template=template,
)


class RAGEngine:
    def __init__(self, hf_token):
        self.hf_token = hf_token

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "google/gemma-3-1b-it",
            use_auth_token=hf_token,
            trust_remote_code=True
        )

        # Model
        #self.model = AutoModelForCausalLM.from_pretrained(
            #"google/gemma-3-4b-it",
            #torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
            #device_map="auto",
            #offload_folder="offload",
            #use_auth_token=hf_token,
            #trust_remote_code=True
        #)
        self.model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-3-1b-it",
            torch_dtype=torch.float32,     # ← CPU에서는 fp32가 가장 안전
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            token=self.hf_token
        )
        self.model.to("cpu")   

    
        # Generation Pipeline
        #self.pipe = pipeline(
            #"text-generation",
            #model=self.model,
            #tokenizer=self.tokenizer,
           #max_new_tokens=1024,
            #do_sample=True,  
        #)
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1,                     
            max_new_tokens=512,
            do_sample=False,               
        )

        # Vectorstore
        self.embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli")
        self.vectorstore = FAISS.load_local("faiss_index", self.embeddings, allow_dangerous_deserialization=True)

        # Langchain QA chain
        self.llm = HuggingFacePipeline(pipeline=self.pipe)
        self.qa_chain = load_qa_chain(
            llm=self.llm,
            chain_type="stuff",
            prompt=prompt,
            document_variable_name="Context"
        )

    def query(self, question: str) -> str:
        tokens = self.tokenizer.encode(question, truncation=True, max_length=512)
        print("Tokenized question:", tokens)
        print("Token range:", min(tokens), max(tokens))
        print(f"Vocab size: {self.tokenizer.vocab_size}")
        assert max(tokens) < self.tokenizer.vocab_size, "Token index out of vocab range!"

        docs = self.vectorstore.similarity_search(question, k=1)
        print(f"Retrieved {len(docs)} documents.")
        for i, doc in enumerate(docs):
            print(f"Doc {i+1}: {doc.page_content[:300]}...\n")

    # docs가 비어있지 않다면 첫번째 문서 내용 가져오기, 없으면 빈 문자열
        context_text = docs[0].page_content if docs else ""

        answer = self.qa_chain.run(input_documents=docs, Question=question, Context=context_text)
        print(f"Answer:\n{answer}")
        return answer


if __name__ == "__main__":
    engine = RAGEngine(hf_token="your_token_here")
    result = engine.query("Tell me about reinforced earth retaining wall construction safety guidelines.")
    print(result)
