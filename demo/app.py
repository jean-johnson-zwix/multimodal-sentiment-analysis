import gradio as gr
from src.attention_model_inference import load_model, predict, CLASS_NAMES

model, device = load_model()

def infer(image, text):
    if image is None or text is None:
        return "Provide both image and text.", {}, 0.5
    result = predict(model, device, image, text)
    return (
        f"Predicted: {result['prediction']}",
        result["class_probabilities"],
        result["image_vs_text_trust"]
    )

with gr.Blocks(title="Multimodal Sentiment Demo") as demo:
    gr.Markdown("# Multimodal Sentiment Analysis with Image and Text")
    with gr.Row():
        image = gr.Image(type="filepath", label="Upload Image")
        text  = gr.Textbox(label="Enter Caption / Tweet", lines=4, placeholder="Type text here...")
    btn   = gr.Button("Analyze Sentiment")
    pred  = gr.Markdown()
    probs = gr.Label(num_top_classes=len(CLASS_NAMES), label="Class Probabilities")
    gate  = gr.Slider(0, 1, value=0.5, step=0.01, label="Gate: Image→Text trust (mean)", interactive=False)

    btn.click(infer, inputs=[image, text], outputs=[pred, probs, gate])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8000)
