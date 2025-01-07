from transformers import pipeline
import gradio as gr

# Initialize summarization model
model = pipeline("summarization", model="facebook/bart-large-cnn")  # Explicitly specify the model

def predict(prompt):
    try:
        # Ensure input length is within model's token limit
        max_input_length = 512  # Token limit for most summarization models
        prompt = prompt[:max_input_length]

        # Generate summary
        summary = model(prompt, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]
        return summary
    except Exception as e:
        return f"Error: {str(e)}"

# Create an interface for the model
with gr.Interface(fn=predict, inputs="textbox", outputs="text", title="Text Summarization") as interface:
    interface.launch()
