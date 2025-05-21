# RAG Chatbot Enhancement Report

## ✅ Overview
This project is a Retrieval-Augmented Generation (RAG) chatbot powered by Groq’s LLaMA3 models and deployed on Hugging Face Spaces using Gradio. It allows users to upload documents, ask questions based on them, and get smart, contextual answers.

## ✨ Enhancements Added
- Redesigned UI with clearer layout and user-friendly controls.
- Added support for:
  - Multiple chunk slider & auto-limit
  - Chat history tracking
  - Markdown-rendered answers
  - Token limit handler (prevents overload)
  - Dark/Light mode toggle
- Smarter model selection panel with icons.

## 📸 Screenshots
![Screenshot 1](./886c46cd-e592-450c-b7d2-68327845110a.png)  
![Screenshot 2](./b0d1269a-e660-4cfa-898d-1c415d2bc1a6.png)

## 🧠 Challenges Faced
- Token limit errors from Groq API (`413 Error`) due to large context size.
- Handling asynchronous file uploads in Gradio.
- Styling with limited CSS customization in Gradio.
- Optimizing chunk size vs token limit for smooth performance.

---

