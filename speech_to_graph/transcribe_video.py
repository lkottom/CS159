#!/usr/bin/env python3

import argparse
import os
import json
import moviepy.editor as mp
import speech_recognition as sr
from llama_cpp import Llama
import time
import torch

# Initialize Llama model
model_path_llama = "/Users/legomac/Desktop/LEGO_MARS/lego_ai/speech_to_graph/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
llm = Llama(
    model_path=model_path_llama,
    n_ctx=2048,
    n_threads=4,
    n_gpu_layers=0
)

def extract_audio(video_path: str, wav_path: str) -> None:
    clip = mp.VideoFileClip(video_path)
    clip.audio.write_audiofile(wav_path, logger=None)
    clip.close()
    print(f"🗒  Audio written to: {wav_path}")

def transcribe_wav(wav_path: str) -> str:
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio_data = recognizer.record(source)
    text = recognizer.recognize_google(audio_data, language="en-US")
    return text

def generate_task_graph(transcript: str, available_pieces: list) -> dict:
    system_prompt = """You are a task graph generator that converts assembly instructions into a structured task graph.
    The task graph should follow this format (starting with assembly_step_0). The first step MUST have two required objects.
    The rest of the steps MUST only have one required object from the available_pieces.
    {
        "nodes": [
            {
                "task_name": "assembly_step_X",
                "task_description": "Detailed description of the assembly step",
                "required_objects": ["list", "of", "required", "pieces"],
                "reference_image": "assembly_X.png"
            },
            ...
        ],
        "edges": [
            {
                "source_node": "assembly_step_X",
                "target_node": "assembly_step_Y",
                "vlm_response": "match"
            },
            ...
        ],
        "image_directory": "assembly_photos"
    }
    Only use the available pieces provided in the available_pieces list.
    """

    user_prompt = f"""Convert the following assembly instructions into a task graph.
    Available pieces: {available_pieces}
    
    Instructions:
    {transcript}
    
    Generate a task graph in JSON format that matches the structure above.
    """

    full_prompt = f"<s>[INST] {system_prompt}\n\n{user_prompt} [/INST]"

    print("\n🔄 Generating task graph...")
    print("This may take a few minutes.")

    start_time = time.time()
    last_update = start_time
    total_tokens = 1024

    response = llm(
        full_prompt,
        max_tokens=total_tokens,
        temperature=0.7,
        stop=["</s>"],
        stream=True
    )

    full_response = ""
    for chunk in response:
        if 'text' in chunk['choices'][0]:
            full_response += chunk['choices'][0]['text']
            current_time = time.time()
            if current_time - last_update > 0.5:
                last_update = current_time

    print("\n")

    try:
        start = full_response.find('{')
        end = full_response.rfind('}') + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON found in response")
        json_str = full_response[start:end]
        return json.loads(json_str)
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        print("Raw response:", full_response)
        return None

def main():
    p = argparse.ArgumentParser(description="Extract audio from video, transcribe, and generate task graph")
    p.add_argument("video", help="Path to input video (e.g. .mov, .mp4)")
    args = p.parse_args()

    base, _ = os.path.splitext(args.video)
    wav_path = base + ".wav"
    txt_path = base + "_sr_transcript.txt"
    graph_path = base + "_task_graph.json"

    available_pieces = ['Brown_4x4_plate', 'Green_2x2', 'Red_2x4', 'Yellow_2x4']

    extract_audio(args.video, wav_path)

    print("🎙  Transcribing audio with Google Speech API...")
    transcript = transcribe_wav(wav_path)

    print("\n=== Transcript ===\n")
    print(transcript)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(transcript + "\n")
    print(f"\n💾 Transcript saved to: {txt_path}")

    task_graph = generate_task_graph(transcript, available_pieces)

    if task_graph:
        with open(graph_path, "w", encoding="utf-8") as f:
            json.dump(task_graph, f, indent=4)
        print(f"💾 Task graph saved to: {graph_path}")
    else:
        print("❌ Failed to generate task graph")

if __name__ == "__main__":
    main()
