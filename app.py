import os
import base64
import time
import threading
import gradio as gr
import pyttsx3
from PIL import Image
from dotenv import load_dotenv
import tempfile
import whisper
from langsmith import Client
from langchain_core.tracers import LangChainTracer
from langchain_openai import ChatOpenAI

# ------------------ ENV + Tracing ------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")

print("🔐 OpenAI API Key:", bool(OPENAI_API_KEY))
print("🔎 LangSmith Tracing V2:", LANGCHAIN_TRACING_V2)
print("🔎 LangSmith Project:", LANGCHAIN_PROJECT)
print("🔎 LangSmith API Key:", bool(LANGCHAIN_API_KEY))

if LANGCHAIN_TRACING_V2 and LANGCHAIN_API_KEY and LANGCHAIN_PROJECT:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
    tracer = LangChainTracer()
    print("✅ LangSmith Tracing Enabled")
else:
    tracer = None
    print("⚠️ LangSmith Tracing Not Enabled")

# ------------------ TTS (pyttsx3) ------------------
def speak(text):
    def run_tts():
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run_tts).start()

# ------------------ GPT-4 Vision ------------------
try:
    llm = ChatOpenAI(
        model="gpt-4o",
        openai_api_key=OPENAI_API_KEY,
        temperature=0,
        max_tokens=1500
    )
    print("✅ GPT‑4 Vision model initialized")
except Exception as e:
    print("❌ GPT‑4 Vision failed:", e)
    exit()

# ------------------ Utilities ------------------
def preprocess_and_encode(image_path):
    image = Image.open(image_path)
    image = image.resize((512, 512))
    image.save(image_path, optimize=True)
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result['text']

# ------------------ Main Assistant ------------------
conversation_history = []

def explain_image(image_path: str, user_question: str) -> str:
    try:
        print("🔁 Preparing image + question...")

        b64 = preprocess_and_encode(image_path)

        if not conversation_history:
            conversation_history.append({
                "role": "system",
                "content": "You are an AI that explains academic visuals (graphs, posters, flowcharts) to blind users in descriptive and structured steps. Always detect the type first."
            })

        # Append image + optional question
        conversation_history.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_question or "Please describe this image in detail."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ]
        })

        print("📡 Sending to GPT‑4 Vision...")
        start = time.time()
        response = llm.invoke(conversation_history, config={"callbacks": [tracer]} if tracer else {})
        conversation_history.append({"role": "assistant", "content": response.content})

        print("✅ Received explanation! ⏱️", round(time.time() - start, 2), "s")
        return response.content

    except Exception as e:
        return f"❌ Error during analysis: {e}"

# ------------------ Full Pipeline ------------------
def analyze_inputs(image, audio):
    try:
        if image is None:
            return "❌ Please upload an academic image."

        image_path = "input.png"
        image.save(image_path)

        question = ""
        if audio is not None:
            print("🎙️ Transcribing audio...")
            question = transcribe_audio(audio)
            print("📝 You asked:", question)

        response = explain_image(image_path, question)
        speak(response)

        return response
    except Exception as e:
        return f"❌ Full pipeline error: {e}"

# ------------------ Gradio UI ------------------
gr_interface = gr.Interface(
    fn=analyze_inputs,
    inputs=[
        gr.Image(type="pil", label="📷 Upload Academic Visual"),
        gr.Audio(source="microphone", type="filepath", label="🎤 Ask Your Question (Optional)")
    ],
    outputs=gr.Textbox(label="🧠 Explanation (Spoken + Text)"),
    title="SceneScribe – AI Assistant for the Visually Impaired",
    description="Upload a graph, flowchart, or academic poster. Ask your question by voice. SceneScribe will detect the image type and describe it with GPT-4 Vision."
)

if __name__ == "__main__":
    gr_interface.launch(share=True)
