import os
from huggingface_hub import snapshot_download
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import gradio as gr

MODEL_NAME = "facebook/bart-large-cnn"
LOCAL_MODEL_DIR = "./bart_model"

# Check if model weight files exist locally
has_weights = os.path.exists(
    os.path.join(LOCAL_MODEL_DIR, "model.safetensors")
) or os.path.exists(os.path.join(LOCAL_MODEL_DIR, "pytorch_model.bin"))

# Download model locally if missing
if not has_weights:
  print(f"Downloading {MODEL_NAME} into '{LOCAL_MODEL_DIR}'...")
  snapshot_download(repo_id=MODEL_NAME, local_dir=LOCAL_MODEL_DIR)
  print("Download complete!")
else:
  print("Model files found locally. Skipping download.")

# Load model and tokenizer once when the app starts
print("Loading model and tokenizer into memory...")
tokenizer = AutoTokenizer.from_pretrained(
    LOCAL_MODEL_DIR, local_files_only=True
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    LOCAL_MODEL_DIR, local_files_only=True
)
print("Model loaded successfully!")


# Summarization function linked to the UI
def summarize_text(paragraph):
  if not paragraph.strip():
    return "Please enter a valid paragraph."

  inputs = tokenizer(
      paragraph, return_tensors="pt", max_length=1024, truncation=True
  )
  summary_ids = model.generate(
      inputs["input_ids"],
      max_length=150,
      min_length=30,
      num_beams=4,
      early_stopping=True,
  )
  return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# Build the Gradio Web UI
demo = gr.Interface(
    fn=summarize_text,
    inputs=gr.Textbox(
        lines=8,
        placeholder=(
            "Paste your long paragraph or article here to summarize..."
        ),
        label="Input Paragraph",
    ),
    outputs=gr.Textbox(lines=4, label="AI Generated Summary"),
    title="Local AI Text Summarizer",
    description=(
        "Running completely offline using `facebook/bart-large-cnn` and"
        " Transformers."
    ),
    theme="default",
)

if __name__ == "__main__":
  demo.launch()