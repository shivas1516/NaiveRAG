from dotenv import load_dotenv
import os
from langchain_classic.prompts import ChatPromptTemplate
from langchain_classic.schema.output_parser import StrOutputParser
from langchain_classic.schema.runnable import RunnablePassthrough
from prompt import template
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template(template)

def generate_answer(vector_store, query: str):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=api_key)

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    rag_chain = (
        {"context": retriever,  "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    response = rag_chain.invoke(query)

    print("Generated Answer:\n", response)
    return response

# Example usage
if __name__ == "__main__":
    from ingestion_pipeline import ingest_pdf

    # Step 1: Ingest data
    file_path = "data/Sample-filled-in-MR.pdf"
    vector_store, _ = ingest_pdf(file_path)

    # Step 2: Generate answer
    query = "What is the Full name of patient and Doctor?"
    print("Query:\n", query)
    generate_answer(vector_store, query)
