import gradio as gr

from llama_index.llms.huggingface import (
    HuggingFaceInferenceAPI,
    HuggingFaceLLM,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
import torch

# Initialize LLM and embedding model
llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    tokenizer_name="pmking27/PrathameshLLM-2B",
    model_name="pmking27/PrathameshLLM-2B",
    device_map="auto",
    model_kwargs={"torch_dtype": torch.float16}
)

embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-large-instruct")

# Initialize LLaMa-Index settings
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 250

# Define function to process queries
def process_query(document_path, query):
    global documents, query_engine

    if document_path:
        # Load document from file
        documents = SimpleDirectoryReader(input_files=[document_path]).load_data()

        # Index documents
        index = VectorStoreIndex.from_documents(documents)

        # Create query engine
        query_engine = index.as_query_engine(similarity_top_k=2)

    # Query the document
    response = query_engine.query(query)

    return response

# Create Gradio interface
document_upload = gr.File(label="Upload Document (TXT/PDF)")
query_input = gr.Textbox(lines=2, label="Enter your query")
output_text = gr.Textbox(label="Response")

gr.Interface(
    fn=process_query,
    inputs=[document_upload, query_input],
    outputs=output_text,
    title="LLM FOR REGIONAL LANGUAGE",
    description="Upload a document and ask queries about it.",
).launch(share=True)