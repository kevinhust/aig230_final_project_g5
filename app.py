import gradio as gr
from src.rag_pipeline import RAGEngine
from src.sentiment import format_sentiment_badge

# Initialize RAG Engine
# Set use_ollama=True if you have Ollama running locally
engine = None
engine_error = None

try:
    engine = RAGEngine(use_ollama=False)
except Exception as e:
    engine_error = str(e)
    print(f"Error initializing RAG Engine: {e}")


def format_sources(sources):
    """Format sources as clickable badges."""
    if not sources:
        return "No sources found"
    badges = [f"📄 `{src}`" for src in sources]
    return " | ".join(badges)


def format_response(response):
    """Format the full response with metadata."""
    answer = response['answer']
    sources = response.get('sources', [])
    detected_lang = response.get('detected_lang', 'Unknown')
    is_code_switching = response.get('is_code_switching', False)
    sentiment = response.get('sentiment', {})
    escalated = response.get('escalated', False)

    # Build metadata section
    metadata_parts = []

    # Language badge
    lang_emoji = {
        'English': '🇬🇧', '中文': '🇨🇳', 'Français': '🇫🇷',
        'Español': '🇪🇸', 'Deutsch': '🇩🇪', 'Japanese': '🇯🇵'
    }
    lang_icon = lang_emoji.get(detected_lang, '🌐')
    metadata_parts.append(f"**Language:** {lang_icon} {detected_lang}")

    # Code-switching badge
    if is_code_switching:
        metadata_parts.append("**Code-switching:** 🔀 Yes")

    # Sentiment badge
    sentiment_badge = format_sentiment_badge(sentiment)
    metadata_parts.append(f"**Sentiment:** {sentiment_badge}")

    # Escalation badge
    if escalated:
        metadata_parts.append("**Status:** 🔔 Escalated to Human Support")

    # Format sources
    source_text = format_sources(sources)

    # Combine all parts
    full_response = f"""{answer}

---
{chr(10).join(metadata_parts)}

**Sources:** {source_text}
"""
    return full_response


def chat_response(message, history):
    """Main chat response function for Gradio."""
    global engine

    if engine is None:
        return f"⚠️ System Error: RAG Engine not initialized.\n\nError: {engine_error}\n\nPlease check your API keys or Ollama connection."

    try:
        response = engine.ask(message, include_scores=False)
        return format_response(response)
    except Exception as e:
        return f"❌ Error processing your request: {str(e)}\n\nPlease try again or contact support."


def clear_history():
    """Clear conversation memory."""
    global engine
    if engine:
        engine.clear_memory()
        return "✅ Conversation history cleared!"
    return "⚠️ Engine not initialized"


# Custom CSS for better styling
custom_css = """
.gradio-container {
    font-family: 'Segoe UI', system-ui, sans-serif;
}
.message.user {
    background-color: #e3f2fd !important;
}
.message.bot {
    background-color: #f5f5f5 !important;
}
.source-badge {
    display: inline-block;
    background: #e8f5e9;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.85em;
    margin: 2px;
}
"""

# Define Gradio Interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🛒 Multilingual E-Commerce Customer Support
    ### Ask about shipping, returns, warranty, or product troubleshooting

    **Supported Languages:** English | 中文 | Français | Español | + Code-switching (mixed languages)

    ---
    """)

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.ChatInterface(
                fn=chat_response,
                title="",
                examples=[
                    "我的 fridge 噪音大，保修怎么走？",
                    "How do I request a return for my broken washer?",
                    "Mon téléphone ne s'allume pas, que faire ?",
                    "¿Cómo solicito una devolución de mi smartphone?",
                    "Do you ship to Toronto?",
                    "我的laptop screen flickering，warranty能cover吗？",
                    "This is UNACCEPTABLE! I want to speak to a manager!",
                ],
                cache_examples=False,
            )

        with gr.Column(scale=1):
            gr.Markdown("### 📊 Session Info")
            clear_btn = gr.Button("🗑️ Clear History", variant="secondary")
            status_output = gr.Textbox(label="Status", interactive=False)

            gr.Markdown("""
            ### 💡 Tips
            - Use any language or mix them
            - Ask about products, shipping, returns
            - Be specific for better answers
            - Sources are always cited

            ### 🎯 Features
            - ✅ Multilingual support
            - ✅ Code-switching detection
            - ✅ Sentiment analysis
            - ✅ Source citations
            - ✅ Angry customer detection
            """)

            def on_clear():
                return clear_history()

            clear_btn.click(on_clear, outputs=status_output)

    gr.Markdown("""
    ---
    **Note:** This is an AI assistant. For complex issues, you may be escalated to human support.

    📚 **Knowledge Base:** 17 documents covering appliances, electronics, furniture, clothing, policies, and more.
    """)


if __name__ == "__main__":
    demo.launch()
