import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

st.title("📄 PDF 임베딩 및 FAISS 구축기")

# 입력: PDF 폴더 경로
pdf_folder = st.text_input("PDF 폴더 경로", " ")

# 버튼 클릭 시 임베딩 수행
if st.button("임베딩 시작"):
    if not os.path.exists(pdf_folder):
        st.error("❌ 입력한 폴더가 존재하지 않습니다.")
    else:
        st.info("📄 PDF 로딩 중...")
        documents = []
        pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, filename in enumerate(pdf_files):
            pdf_path = os.path.join(pdf_folder, filename)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())

            progress = (i + 1) / len(pdf_files)
            progress_bar.progress(progress)
            status_text.text(f"{i+1} / {len(pdf_files)} PDF 로드됨")

        st.success(f"✅ 총 {len(documents)} 개의 문서를 불러왔습니다.")

        st.info("📚 문서 분할 중...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = splitter.split_documents(documents)
        st.success(f"✅ 분할된 문서 수: {len(docs)}")

        st.info("🧠 임베딩 모델 로드 중...")
        embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli")

        st.info("🔍 FAISS 인덱스 생성 중... 시간이 다소 걸릴 수 있습니다.")
        with st.spinner("임베딩 중..."):
            vectorstore = FAISS.from_documents(docs, embedding_model)

        vectorstore.save_local("faiss_index")
        st.success("✅ FAISS 인덱스 저장 완료! (faiss_index 폴더에 저장됨)")

