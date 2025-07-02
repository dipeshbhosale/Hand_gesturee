import gradio as gr
from main import detect_gesture

iface = gr.Interface(
    fn=detect_gesture,
    inputs=gr.Image(type="numpy", label="Upload an Image"),
    outputs=[gr.Image(label="Annotated Output"), gr.Textbox(label="Prediction")],
    title="Face Gesture Recognition",
    description="Upload an image to get gesture prediction."
)

if __name__ == "__main__":
    iface.launch()
