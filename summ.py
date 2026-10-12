import os
from dotenv import load_dotenv
import gradio as gr
from huggingface_hub import InferenceClient

# Load environment variables from .env file
load_dotenv()

# Fetch the Hugging Face token from Render's environment variables
hf_token = os.getenv("HF_TOKEN")

# Initialize the client to use Hugging Face's free serverless API
client = InferenceClient("facebook/bart-large-cnn", token=hf_token)

def summarize_text(paragraph):
    if not paragraph.strip():
        return "Please enter a valid paragraph."

    try:
        response = client.summarization(
            paragraph, parameters={"max_length": 150, "min_length": 30}
        )
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
    # Render assigns a dynamic port; default to 7860 for local testing
    port = int(os.environ.get("PORT", 7860))
    # Bind to 0.0.0.0 so Render's load balancer can route traffic to the app
    demo.launch(server_name="0.0.0.0", server_port=port)