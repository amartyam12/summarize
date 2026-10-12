import os
import gradio as gr
from huggingface_hub import InferenceClient

hf_token = os.environ.get("HF_TOKEN")
client = InferenceClient("facebook/bart-large-cnn", token=hf_token)

def summarize_text(paragraph):
    if not paragraph.strip():
        return "Please enter a valid paragraph."

    try:
        # FIX: Removed the 'parameters' dictionary argument
        response = client.summarization(paragraph)
        return response.summary_text
    except Exception as e:
        return f"Error: {str(e)}"

# Build the Gradio Web UI
demo = gr.Interface(
    fn=summarize_text,
    inputs=gr.Textbox(
        lines=8,
        placeholder="Paste your long paragraph or article here to summarize...",
        label="Input Paragraph",
    ),
    outputs=gr.Textbox(lines=4, label="AI Generated Summary"),
    title="Cloud AI Text Summarizer",
    description="Running on Render using the Hugging Face Inference API.",
    theme="default",
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)