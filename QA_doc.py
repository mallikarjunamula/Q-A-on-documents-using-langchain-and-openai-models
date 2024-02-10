import os
import streamlit as st
import pinecone
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

os.environ['OPENAI_API_KEY'] = 'your_openai_api key'
st.subheader("Q&A on our data using langchain and GPT-4:",divider='rainbow')
def get_similiar_docs(query, k=2, score=False):
  pinecone.init(
      api_key="f31578ed-57dc-4691-b49e-1a3462fa5067",
      environment="gcp-starter"
  )
  index_name = "doc-qa-langchain"
  index = pinecone.Index(index_name)
  if score:
    similar_docs = index.similarity_search_with_score(query, k=k)
  else:
    similar_docs = index.similarity_search(query, k=k)
  return similar_docs

def qa_doc(query):
  model_name = "gpt-4"
  llm = OpenAI(model_name=model_name, api_key=os.environ['OPENAI_API_KEY'])
  chain = load_qa_chain(llm, chain_type="stuff")
  similar_docs = get_similiar_docs(query)
  answer = chain.run(input_documents=similar_docs, question=query)
  return answer

input_text  = st.text_area("Write Question here!")
button = st.button("Generate Response")
if button and input_text:
    with st.spinner("Generating Answer"):
        resp = qa_doc(input_text)
    st.write(resp)