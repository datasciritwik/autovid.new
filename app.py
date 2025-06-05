import gradio as gr

# ----------------------------------------
# 🧰 Step 1: Define Dummy Tool Functions
# ----------------------------------------

def generate_video(message, style, duration, resolution):
    return f"[🎬 Video generated]\nPrompt: {message}\nStyle: {style}\nDuration: {duration}s\nResolution: {resolution}"

def improve_prompt(message):
    return f"[✨ Improved Prompt]\n{message.upper()} (automatically enhanced)"

def collect_feedback(message):
    return "[✅ Feedback Received] Thank you for your input!"

# ----------------------------------------
# 💬 Step 2: Define ChatInterface Handler
# ----------------------------------------

def chat_fn(message, history, style, duration, resolution):
    if "generate" in message.lower():
        return generate_video(message, style, duration, resolution)
    elif "improve" in message.lower():
        return improve_prompt(message)
    elif "feedback" in message.lower():
        return collect_feedback(message)
    else:
        return "[🤖 AutoVid] Please specify if you want to generate a video, improve a prompt, or give feedback."

# ----------------------------------------
# 🧠 Step 3: Create Chat Interface
# ----------------------------------------

demo = gr.ChatInterface(
    fn=chat_fn,
    type="messages",
    title="🎥 AutoVid: Text-to-Video AI Agent",
    description="Ask AutoVid to generate a video, improve a prompt, or give feedback.",
    additional_inputs=[
        gr.Textbox(label="🎨 Style", value="Cinematic"),
        gr.Slider(label="⏱ Duration (sec)", minimum=5, maximum=30, value=10),
        gr.Dropdown(["480p", "720p", "1080p"], label="📺 Resolution", value="720p")
    ],
    examples=[
        ["Generate a video of a cyberpunk Tokyo at night"],
        ["Improve: A lion running through the forest with epic music"],
        ["Feedback: Please make transitions smoother"],
    ],
    submit_btn="🚀 Submit",
    stop_btn="⏹️ Stop",
    fill_height=True
)

# ----------------------------------------
# 🚀 Step 4: Launch the App
# ----------------------------------------

if __name__ == "__main__":
    demo.launch(mcp_server=True)
