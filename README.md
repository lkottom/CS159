# Assembly Verification System

This project provides an intelligent assembly verification pipeline that (using LEGO for proof-of-concept):

- Transcribes spoken instructions from video using Mistral to generate structured task graphs.
- Use this structured task graph and verifies image-based LEGO assemblies using OpenAI's GPT-4o to compare user-submitted photos against step-by-step references.

---

## 🚀 Overview

This system is designed to aid in validating assembly steps through AI. It consists of two major components:

### 🗣️ Speech-to-Graph Pipeline (`transcribe_video.py`)
- Uses a local Mistral model (e.g., mistral-7b-instruct) to transcribe spoken assembly instructions.
- Structures instructions into a JSON-based task graph.

### 🧠 Visual Assembly Verification App (`main.py`)
A GUI tool built with Tkinter that allows users to:
- Load a task graph and pulls relevent info from nodes
- Compare their real assembly photos against reference images
- Get validation results via GPT-4o's Vision-Language Model
- Save verification metrics to CSV

---

## 🧩 Features

### Speech-to-Graph (`transcribe_video.py`)
- Transcribes `.mov`/`.wav` files using a Mistral model.
- Generates structured JSON task graphs.
- Stores intermediate enhanced visualizations and graphs.

### Assembly Verifier (`main.py`)
- GUI with image upload support
- Side-by-side visual comparison
- Calls GPT-4o with combined images and custom prompts
- Tracks inference time and saves metrics like VLM result and human marking
- Supports navigation between task graph steps

---


## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/lkottom/CS159.git
   cd your-repo
   ```

2. **Set up a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variable**
   You must set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY=your_key_here  # or use a .env file
   ```

---

## 🧪 Usage

### 1. Transcribe a Video and Generate Graph
```bash
python transcribe_video.py --input IMG_0501.mov
```
This will create a task graph JSON (e.g., `IMG_0501_task_graph.json`) in the root or `speech_to_graph/` folder.

### 2. Launch GUI for Verification
```bash
python main.py
```
- Load tasks from the task graph.
- Upload your assembly image per step.
- Run "Verify Assembly" to get a GPT-4o visual response.
- Save metrics including match status and your own evaluation.

---

## 🛠️ Dependencies (from `requirements.txt`)

**Key packages:**
- **Vision & Imaging:** `opencv-python`, `pillow`, `matplotlib`, `moviepy`
- **AI Models:** `torch`, `torch-geometric`, `llama_cpp_python`, `openai`
- **Speech & Audio:** `SpeechRecognition`, `standard-aifc`
- **GUI:** `tkinter`, `ttk` (via stdlib)
- **Visualization:** `seaborn`, `pandas`
- **Misc:** `aiohttp`, `tqdm`, `requests`, `python-dotenv`

For full list, see `requirements.txt`.

---

## 📊 Output Metrics

Saved to `4o_mini_verification_metrics.csv`:

| Timestamp           | Task Name | Inference Time (s) | VLM Result | Human Marked Correct |
|---------------------|-----------|---------------------|-------------|------------------------|
| 2025-05-16 14:00:00 | step_1    | 2.37                | match       | Yes                    |

---

## 🧠 Model Info

- **Transcription:** Mistral 7B, locally loaded. Direct download link: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/blob/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf
- **Vision-Language Verification:** GPT-4o-mini vs. GPT-4o (OpenAI)

---