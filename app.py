from transformers import pipeline, BartTokenizer
import gradio as gr

# Initialize model and tokenizer
model = pipeline("summarization", model="facebook/bart-large-cnn")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

def predict(prompt):
    try:
        # Tokenize and truncate input to 512 tokens
        inputs = tokenizer(prompt, max_length=512, truncation=True, return_tensors="pt")
        
        # Generate summary
        summary = model(prompt, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]
        return summary
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
with gr.Interface(
    fn=predict, 
    inputs="textbox", 
    outputs="text", 
    title="Text Summarization",
    description="Summarize your input text using Hugging Face's Transformers library.",
) as interface:
    interface.launch()
