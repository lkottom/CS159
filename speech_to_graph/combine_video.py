from moviepy.editor import VideoFileClip, AudioFileClip

# Input file paths
video_path = "/Users/legomac/Desktop/LEGO_MARS/lego_ai/speech_to_graph/Updated_demo.mov"
audio_path = "/Users/legomac/Desktop/LEGO_MARS/lego_ai/speech_to_graph/IMG_0501.wav"
output_path = "output_video.mov"

# Load video and audio
video = VideoFileClip(video_path)
new_audio = AudioFileClip(audio_path)

# Overlay (replace) audio
video_with_new_audio = video.set_audio(new_audio)

# Write the result to a new .mov file
video_with_new_audio.write_videofile(output_path, codec="libx264", audio_codec="aac")

print("New video with overlaid audio saved to:", output_path)
