import gradio as gr
import fitz  # PyMuPDF
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from groq import Groq

# Setup
api_key = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=api_key)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Text Chunker
def split_text_into_chunks(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

# Extract + Chunk
def extract_text_from_pdfs(files):
    texts, sources, preview = [], [], []
    for file in files:
        try:
            doc = fitz.open(file.name)
            full_text = ''
            for page in doc:
                text = page.get_text().strip()
                if text:
                    full_text += text + '\n\n'
            preview.append((file.name, full_text[:1000]))
            chunks = split_text_into_chunks(full_text)
            texts.extend(chunks)
            sources.extend([file.name] * len(chunks))
        except Exception as e:
            texts.append(f"[Error reading {file.name}: {str(e)}]")
            sources.append(file.name)
            preview.append((file.name, f"[Error previewing {file.name}: {str(e)}]"))
    return texts, sources, preview

# Create Index
def create_index(texts):
    embeddings = model.encode(texts)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

# Search Top Chunks
def get_top_chunks(query, texts, index, sources, embeddings, k=3):
    q_emb = model.encode([query])
    _, I = index.search(np.array(q_emb), k)
    results = [texts[i] for i in I[0]]
    refs = [sources[i] for i in I[0]]
    return results, refs

# Ask Question Handler
def ask_question(files, query, model_name, k_chunks, show_sources):
    try:
        if not api_key:
            return "❌ GROQ_API_KEY is missing.", ""
        if not files:
            return "📂 Please upload at least one PDF.", ""
        if not query.strip():
            return "❓ Please enter a question.", ""

        texts, sources, _ = extract_text_from_pdfs(files)
        if not texts or all(t.startswith("[Error") or not t.strip() for t in texts):
            return "❌ Could not extract readable text from the uploaded PDFs.", ""

        index, embeddings = create_index(texts)
        chunks, refs = get_top_chunks(query, texts, index, sources, embeddings, k=k_chunks)
        context = "\n\n".join(chunks)

        prompt = f"""Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}"""

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.choices[0].message.content
        if show_sources:
            source_info = "\n\n📚 Sources:\n" + "\n".join(refs)
        else:
            source_info = ""
        return answer, source_info
    except Exception as e:
        import traceback
        return f"❌ Runtime Error:\n{traceback.format_exc()}", ""

# Interface Logic
def interface(files, query, model_name, k_chunks, show_sources):
    previews = extract_text_from_pdfs(files)[2]
    preview_md = "\n\n".join([f"**{name}**\n```\n{content}\n```" for name, content in previews])
    answer, sources = ask_question(files, query, model_name, k_chunks, show_sources)
    return preview_md, answer + sources

# Gradio UI
with gr.Blocks(title="📘 PDF Chatbot with Groq") as demo:
    gr.Markdown("## 🤖📄 PDF RAG Chatbot using Groq + Sentence Transformers")
    gr.Markdown("Upload your PDFs, choose settings, and get answers using Groq LLMs.")

    with gr.Row():
        with gr.Column():
            pdf_input = gr.File(file_types=[".pdf"], file_count="multiple", label="📂 Upload PDFs")
            question_input = gr.Textbox(lines=3, placeholder="Type your question here...", label="❓ Ask a Question")
            model_select = gr.Radio(
                choices=["llama3-8b-8192", "llama3-70b-8192"],
                value="llama3-8b-8192",
                label="🧠 Choose Groq Model"
            )
            k_slider = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="📚 Number of Chunks to Use")
            show_sources = gr.Checkbox(label="Show Source Files", value=True)
            submit_btn = gr.Button("🚀 Get Answer", variant="primary")
            clear_btn = gr.Button("🔁 Clear")

        with gr.Column():
            with gr.Accordion("📖 PDF Preview", open=False):
                preview_output = gr.Markdown()
            answer_output = gr.Textbox(label="💬 Answer", interactive=False, lines=15)

    submit_btn.click(
        fn=interface,
        inputs=[pdf_input, question_input, model_select, k_slider, show_sources],
        outputs=[preview_output, answer_output]
    )

    clear_btn.click(
        fn=lambda: ("", ""),
        inputs=[],
        outputs=[preview_output, answer_output]
    )

if __name__ == "__main__":
    demo.launch(share=True)
