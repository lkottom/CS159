#!/usr/bin/env python3
"""
enhanced_viz.py - Enhanced visualization for LEGO assembly with interactive node details

Key features:
- Fixes macOS Tkinter/Pygame compatibility issue
- Adds interactive node selection to display task details
- Shows required pieces and descriptions when nodes are clicked
- Linear graph layout for clearer step progression
- Improved UI with details panel and numeric slider indicators
- Enhanced reset functionality for complete state reset
- Configurable animation timing using constants
- Larger default window size

Usage:
  python enhanced_viz.py [--video VIDEO_PATH]
"""

import os
import json
import time
import threading
import argparse
import sys

# IMPORTANT: Initialize a Tkinter root before importing pygame
# This is key to avoid the NSInvalidArgumentException on macOS
import tkinter as tk
root = tk.Tk()
root.withdraw()  # Hide the root window, we'll create our own later

# Now import pygame after Tkinter is initialized
import pygame

# Rest of the imports
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import moviepy.editor as mp
import speech_recognition as sr
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use("TkAgg")

# No default video path - require user to select a video
DEFAULT_VIDEO_PATH = ""

# Animation constants - configured at the top level
INITIAL_PAUSE_SECONDS = 5.8  # Initial pause before animation starts (0 for immediate start)
NODE_TRANSITION_DELAY = 6.4 # Additional delay between adding nodes
GRAPH_SPEED = 1.0  # Graph animation speed (higher = faster)
EMPTY_GRAPH_DRAW_DELAY = 0.1  # Short delay for empty graph to be drawn


class CompleteVisualizationApp:
    def __init__(self, root, video_path=None):
        self.root = root
        self.root.title("LEGO Assembly Visualization")
        self.root.geometry("1600x1000")  # Increased window size
        
        # Initialize pygame for audio playback
        pygame.mixer.init()
        
        # Set up the main frame
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create horizontal paned window for side-by-side layout
        self.paned_window = ttk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)
        
        # Create three equal frames for video, transcript, and graph
        self.video_frame = ttk.LabelFrame(self.paned_window, text="Video")
        self.transcript_frame = ttk.LabelFrame(self.paned_window, text="Live Transcript")
        self.graph_frame = ttk.LabelFrame(self.paned_window, text="Task Graph Visualization")
        
        self.paned_window.add(self.video_frame, weight=1)
        self.paned_window.add(self.transcript_frame, weight=1)
        self.paned_window.add(self.graph_frame, weight=1)
        
        # Set up video display
        self.video_display_frame = ttk.Frame(self.video_frame)
        self.video_display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.video_label = ttk.Label(self.video_display_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Set up video controls
        self.video_control_frame = ttk.Frame(self.video_frame)
        self.video_control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Video speed control with value label
        self.video_speed_frame = ttk.Frame(self.video_control_frame)
        self.video_speed_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        ttk.Label(self.video_speed_frame, text="Video Speed:").pack(side=tk.LEFT, padx=5)
        self.video_speed_var = tk.DoubleVar(value=1.0)
        self.video_speed_scale = ttk.Scale(
            self.video_speed_frame, 
            from_=0.1, 
            to=5.0,  # Wider range
            orient=tk.HORIZONTAL, 
            variable=self.video_speed_var,
            length=150,
            command=self.update_video_speed_label
        )
        self.video_speed_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Add value label for video speed
        self.video_speed_label = ttk.Label(self.video_speed_frame, text="1.0")
        self.video_speed_label.pack(side=tk.LEFT, padx=5)
        
        # Audio control
        self.audio_var = tk.BooleanVar(value=True)
        self.audio_check = ttk.Checkbutton(
            self.video_control_frame, 
            text="Audio", 
            variable=self.audio_var,
            command=self.toggle_audio
        )
        self.audio_check.pack(side=tk.LEFT, padx=5)
        
        # Improved volume control with label showing current value
        self.volume_frame = ttk.Frame(self.video_control_frame)
        self.volume_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        ttk.Label(self.volume_frame, text="Volume:").pack(side=tk.LEFT, padx=5)
        self.volume_var = tk.DoubleVar(value=0.7)  # Default volume 70%
        
        # Add a label to show current volume percentage
        self.volume_label = ttk.Label(self.volume_frame, text="70%")
        self.volume_label.pack(side=tk.RIGHT, padx=5)
        
        self.volume_scale = ttk.Scale(
            self.volume_frame, 
            from_=0.0, 
            to=1.0, 
            orient=tk.HORIZONTAL, 
            variable=self.volume_var,
            length=100,
            command=self.update_volume
        )
        self.volume_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Set up transcript display
        self.transcript_text = tk.Text(self.transcript_frame, wrap=tk.WORD)
        self.transcript_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Set up transcript controls with value label
        self.transcript_control_frame = ttk.Frame(self.transcript_frame)
        self.transcript_control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(self.transcript_control_frame, text="Transcript Speed:").pack(side=tk.LEFT, padx=5)
        self.transcript_speed_var = tk.DoubleVar(value=0.6)  # Slower default for transcript
        self.transcript_speed_scale = ttk.Scale(
            self.transcript_control_frame, 
            from_=0.01, 
            to=1.0,  # Wider range
            orient=tk.HORIZONTAL, 
            variable=self.transcript_speed_var,
            length=150,
            command=self.update_transcript_speed_label
        )
        self.transcript_speed_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Add value label for transcript speed
        self.transcript_speed_label = ttk.Label(self.transcript_control_frame, text="0.6")
        self.transcript_speed_label.pack(side=tk.LEFT, padx=5)
        
        # Set up graph display
        self.fig, self.ax = plt.subplots(figsize=(6, 6))  # Square figure for better graph layout
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add node details panel below the graph
        self.node_details_frame = ttk.LabelFrame(self.graph_frame, text="Node Details")
        self.node_details_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=5)
        
        # Create a text widget to display node details
        self.node_details_text = tk.Text(self.node_details_frame, wrap=tk.WORD, height=6)
        self.node_details_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.node_details_text.insert(tk.END, "Click on a node to see its details")
        self.node_details_text.config(state=tk.DISABLED)
        
        # Add info label about graph visualization settings
        self.graph_info_frame = ttk.Frame(self.graph_frame)
        self.graph_info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.graph_info_label = ttk.Label(
            self.graph_info_frame, 
            text=f"Graph Settings: Initial Pause = {INITIAL_PAUSE_SECONDS}s, Speed = {GRAPH_SPEED}x, Node Delay = {NODE_TRANSITION_DELAY}s",
            anchor=tk.CENTER
        )
        self.graph_info_label.pack(fill=tk.X, expand=True)
        
        # Control frame
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Add buttons
        self.load_button = ttk.Button(self.control_frame, text="Load Video", command=self.load_video)
        self.load_button.pack(side=tk.LEFT, padx=5)
        
        self.start_button = ttk.Button(self.control_frame, text="Start Playback", command=self.start_playback, state=tk.DISABLED)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        # Reset button
        self.reset_button = ttk.Button(self.control_frame, text="Reset", command=self.hard_reset)
        self.reset_button.pack(side=tk.LEFT, padx=5)
        
        # Initialize variables
        self.video_path = video_path if video_path else DEFAULT_VIDEO_PATH
        self.video_capture = None
        self.audio_player = None
        self.playing = False
        self.audio_playing = False
        self.transcript = ""
        self.graph_data = None
        self.current_node_index = 0
        self.G = nx.DiGraph()
        self.pos = None
        self.selected_node = None
        self.current_visible_nodes = []
        self.current_visible_edges = []
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready - Please load a video file to begin")
        self.status_bar = ttk.Label(self.main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=2)
        
        # If a default path is provided, load it automatically
        if self.video_path and os.path.exists(self.video_path):
            self.root.after(500, self.load_video_from_path)
        else:
            # Automatically open file dialog if no video path provided
            self.root.after(500, self.load_video)
    
    # Add methods to update slider value labels
    def update_video_speed_label(self, event=None):
        self.video_speed_label.configure(text=f"{self.video_speed_var.get():.1f}")
    
    def update_transcript_speed_label(self, event=None):
        self.transcript_speed_label.configure(text=f"{self.transcript_speed_var.get():.2f}")
    
    def update_node_details(self, text):
        """Update the node details text box"""
        self.node_details_text.config(state=tk.NORMAL)
        self.node_details_text.delete(1.0, tk.END)
        self.node_details_text.insert(tk.END, text)
        self.node_details_text.config(state=tk.DISABLED)
    
    def on_node_click(self, event):
        """Handle clicks on the graph to select nodes"""
        # Get coordinates
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        
        # Find the closest node from currently visible nodes only
        closest_node = None
        closest_dist = float('inf')
        
        # Only consider nodes that are currently visible in the visualization
        visible_nodes = []
        if hasattr(self, 'current_visible_nodes') and self.current_visible_nodes:
            visible_nodes = self.current_visible_nodes
        else:
            # If animation is complete, all nodes are visible
            visible_nodes = list(self.G.nodes())
        
        for node in visible_nodes:
            if node in self.pos:
                nx, ny = self.pos[node]
                dist = (nx - x)**2 + (ny - y)**2
                if dist < closest_dist:
                    closest_dist = dist
                    closest_node = node
        
        # Check if click is close to a node with a stricter threshold
        threshold = 0.05
        if closest_dist < threshold and closest_node:
            self.selected_node = closest_node
            
            # Get node data from the graph
            node_data = self.G.nodes[closest_node]
            
            # Format the node details
            details = f"Task: {closest_node}\n"
            if 'description' in node_data:
                details += f"Description: {node_data['description']}\n"
            
            # Find the node in the original data to get required objects
            for node in self.graph_data.get('nodes', []):
                if node.get('task_name') == closest_node:
                    if 'required_objects' in node:
                        objects_list = ', '.join(node.get('required_objects', []))
                        details += f"Required pieces: {objects_list}\n"
                    if 'reference_image' in node:
                        details += f"Reference image: {node.get('reference_image', '')}"
                    break
            
            self.update_node_details(details)
            
            # Update the visualization to highlight the selected node but maintain current visibility
            if hasattr(self, 'current_visible_edges'):
                self.update_graph_viz_with_current(self.current_visible_nodes, self.current_visible_edges)
            else:
                # Fallback if no current visibility information
                self.update_graph_viz(self.G, list(self.G.edges()))

    def update_empty_graph(self):
        """Draw an empty graph to ensure a clean slate"""
        # Clear everything
        self.ax.clear()
        
        # Set basic properties
        self.ax.set_title("LEGO Assembly Steps")
        self.ax.axis('off')
        
        # Force margins to be maintained even with no content
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        
        # Ensure the axis limits are set appropriately for future nodes
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        
        # Force a complete redraw
        self.canvas.draw()
        self.canvas.flush_events()
    
    def find_related_files(self, video_path):
        """Find related files (JSON, WAV) based on the video filename pattern"""
        # Get the base name without extension
        video_basename = os.path.basename(video_path)
        video_name, _ = os.path.splitext(video_basename)
        video_dir = os.path.dirname(video_path)
        
        # If video directory is empty, use current directory
        if not video_dir:
            video_dir = "."
            
        # Look for matching files in video directory
        related_files = {
            'json': None,
            'wav': None,
            'transcript': None
        }
        
        # Define patterns to search for
        json_patterns = [
            f"{video_name}_task_graph.json",
            f"{video_name}.json",
            f"{video_name}_graph.json"
        ]
        
        wav_patterns = [
            f"{video_name}.wav",
            f"{video_name}_audio.wav"
        ]
        
        transcript_patterns = [
            f"{video_name}_sr_transcript.txt",
            f"{video_name}_transcript.txt"
        ]
        
        # Search in video directory
        try:
            files = os.listdir(video_dir)
            
            # Find JSON file
            for pattern in json_patterns:
                if pattern in files:
                    related_files['json'] = os.path.join(video_dir, pattern)
                    break
                    
            # Find WAV file
            for pattern in wav_patterns:
                if pattern in files:
                    related_files['wav'] = os.path.join(video_dir, pattern)
                    break
                    
            # Find transcript file
            for pattern in transcript_patterns:
                if pattern in files:
                    related_files['transcript'] = os.path.join(video_dir, pattern)
                    break
        except Exception as e:
            self.update_status(f"Error searching for related files: {str(e)}")
            
        return related_files
    
    def load_video_from_path(self):
        """Load the video from the predefined path"""
        if not os.path.exists(self.video_path):
            self.update_status(f"Error: Video file not found: {self.video_path}")
            return
            
        # Ensure everything is reset before loading
        self.hard_reset()
            
        # Open the video file
        self.video_capture = cv2.VideoCapture(self.video_path)
        ret, frame = self.video_capture.read()
        
        if ret:
            # Display the first frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = self.resize_frame(frame_rgb)
            img = Image.fromarray(frame_resized)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            
            # Find related files based on video filename
            related_files = self.find_related_files(self.video_path)
            
            # Extract the base name for conventional paths
            self.base_path, _ = os.path.splitext(self.video_path)
            
            # Load JSON if available but DO NOT display it yet
            if related_files['json'] and os.path.exists(related_files['json']):
                try:
                    with open(related_files['json'], 'r') as f:
                        self.graph_data = json.load(f)
                    # Initialize the graph structure but don't show it
                    self.initialize_graph_structure()
                    self.update_status("Graph data loaded successfully")
                except Exception as e:
                    self.update_status(f"Error loading graph data: {str(e)}")
            
            # Load transcript file but DO NOT display it yet
            if related_files['transcript'] and os.path.exists(related_files['transcript']):
                try:
                    with open(related_files['transcript'], 'r') as f:
                        transcript_content = f.read()
                    # Store the transcript but don't show it
                    self.transcript = transcript_content
                    self.update_status("Transcript loaded successfully")
                except Exception as e:
                    self.update_status(f"Error loading transcript: {str(e)}")
            
            # Check for existing WAV file
            if related_files['wav'] and os.path.exists(related_files['wav']):
                self.audio_path = related_files['wav']
                self.update_status(f"Found audio file: {os.path.basename(self.audio_path)}")
            
            # Update status
            self.update_status(f"Loaded video: {os.path.basename(self.video_path)} - Ready to start playback")
            
            # Enable start button
            self.start_button.config(state=tk.NORMAL)
            
            # Store the related files paths
            self.related_files = related_files

    def hard_reset(self):
        """Perform a complete hard reset of the visualization to initial state"""
        # Stop playback
        self.playing = False
        
        # Stop audio if playing
        if hasattr(self, 'audio_playing') and self.audio_playing and pygame.mixer.get_init() and pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
            self.audio_playing = False
        
        # Reset transcript display and stored transcript data
        self.transcript_text.delete(1.0, tk.END)
        # Don't clear self.transcript as it's the source data
        
        # Reset selection and details
        self.selected_node = None
        self.update_node_details("Click on a node to see its details")
        
        # Completely reset all graph structures
        self.current_visible_nodes = []
        self.current_visible_edges = []
        
        # Create a new DiGraph instead of reusing the old one
        self.G = nx.DiGraph()
        
        # Clear positions dictionary
        self.pos = {}
        
        # Complete reset of matplotlib plot
        try:
            self.fig.clf()  # Clear the entire figure
            self.ax = self.fig.add_subplot(111)  # Create a new axes
            self.ax.clear()  # Clear any existing content
            self.ax.set_title("LEGO Assembly Steps")
            self.ax.axis('off')
            
            # Force a complete redraw of the canvas
            self.canvas.draw()
            self.canvas.flush_events()
        except Exception as e:
            print(f"Error resetting graph: {str(e)}")
            # Recreate figure and axes if there was an error
            plt.close(self.fig)
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            self.ax.set_title("LEGO Assembly Steps")
            self.ax.axis('off')
            self.canvas.draw()
        
        # Reset video to first frame
        if hasattr(self, 'video_capture') and self.video_capture:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.video_capture.read()
            if ret:
                # Display the first frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = self.resize_frame(frame_rgb)
                img = Image.fromarray(frame_resized)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
                # Rewind back to start
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Re-enable start button
        self.start_button.config(state=tk.NORMAL)
        
        # Only re-initialize the graph structure if we have graph data
        # but don't display any nodes until animation starts
        if hasattr(self, 'graph_data') and self.graph_data:
            self.initialize_graph_structure()
        
        # Disconnect any existing event connections to avoid duplicate handlers
        if hasattr(self, '_cid'):
            try:
                self.fig.canvas.mpl_disconnect(self._cid)
                del self._cid
            except:
                pass
        
        # Reconnect event handler for node clicks
        self._cid = self.fig.canvas.mpl_connect('button_press_event', self.on_node_click)
        
        self.update_status("Hard reset complete - Ready to play")


    def initialize_graph_structure(self):
        """Initialize the graph structure without displaying it"""
        if not self.graph_data:
            return
                
        # Get nodes
        nodes = self.graph_data.get('nodes', [])
        
        if not nodes:
            return
                
        # Always create a completely new graph
        self.G = nx.DiGraph()
        
        # Add all nodes to the graph
        for i, node in enumerate(nodes):
            node_name = node.get('task_name', f"Task {i}")
            self.G.add_node(
                node_name,
                description=node.get('task_description', ''),
                image=node.get('reference_image', '')
            )
        
        # Add all edges to the graph
        edges = self.graph_data.get('edges', [])
        for edge in edges:
            source = edge.get('source_node', '')
            target = edge.get('target_node', '')
            
            if source and target:
                self.G.add_edge(
                    source, 
                    target,
                    vlm_response=edge.get('vlm_response', '')
                )
        
        # Create a linear layout for the graph nodes (left to right)
        pos = {}
        total_nodes = len(nodes)
        
        # Calculate positions in a linear layout
        for i, node in enumerate(nodes):
            node_name = node.get('task_name', f"Task {i}")
            # X-coordinate spaced evenly along horizontal axis
            x = (i / (total_nodes - 1 if total_nodes > 1 else 1)) * 0.8 + 0.1
            # Y-coordinate with slight variation to prevent perfectly straight line
            y = 0.5 + (0.1 * (-1 if i % 2 == 0 else 1))
            pos[node_name] = (x, y)
        
        # Always create a completely new position dictionary
        self.pos = pos.copy()
        
        # Explicitly reset tracking variables
        self.current_visible_nodes = []
        self.current_visible_edges = []
        
        # Clear the graph visualization but DON'T draw any nodes
        self.ax.clear()
        self.ax.set_title("LEGO Assembly Steps")
        self.ax.axis('off')
        self.canvas.draw()
    
    def load_video(self):
        """Open file dialog to select a video file"""
        video_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.mov *.avi")]
        )
        
        if not video_path:
            return
            
        self.video_path = video_path
        self.load_video_from_path()
    
    def toggle_audio(self):
        """Toggle audio playback on/off"""
        if self.audio_var.get():
            if not self.audio_playing and pygame.mixer.get_init() and pygame.mixer.music.get_busy():
                pygame.mixer.music.unpause()
                self.audio_playing = True
                self.update_status("Audio resumed")
        else:
            if self.audio_playing and pygame.mixer.get_init() and pygame.mixer.music.get_busy():
                pygame.mixer.music.pause()
                self.audio_playing = False
                self.update_status("Audio paused")
    
    def update_volume(self, *args):
        """Update audio volume and display percentage"""
        volume = self.volume_var.get()
        
        # Update volume label with percentage
        volume_percentage = int(volume * 100)
        self.volume_label.configure(text=f"{volume_percentage}%")
        
        # Update pygame mixer volume
        if pygame.mixer.get_init():
            pygame.mixer.music.set_volume(volume)
    
    def resize_frame(self, frame, max_width=600):  # Increased max width for larger display
        """Resize the frame to fit in the UI"""
        height, width = frame.shape[:2]
        if width > max_width:
            ratio = max_width / width
            new_height = int(height * ratio)
            frame = cv2.resize(frame, (max_width, new_height))
        return frame
    
    def start_playback(self):
        """Start playing the video and audio using Tk’s after() scheduling."""
        if not self.video_capture:
            self.update_status("Please load a valid video first")
            return

        # Reset everything to a clean slate
        self.hard_reset()

        # Disable controls during playback
        self.load_button.config(state=tk.DISABLED)
        self.start_button.config(state=tk.DISABLED)

        # Rewind video
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Determine native FPS (fallback to 30)
        fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0

        # Kick off the frame loop on the main thread
        self.playing = True
        self._schedule_next_frame(fps)

        # Start audio if available
        if hasattr(self, 'audio_path') and self.audio_path and os.path.exists(self.audio_path):
            try:
                pygame.mixer.music.load(self.audio_path)
                pygame.mixer.music.set_volume(self.volume_var.get())
                pygame.mixer.music.play()
                self.audio_playing = True
            except Exception as e:
                self.update_status(f"Error playing audio: {e}")

        # Launch graph and transcript threads as before
        threading.Thread(target=self.animate_graph, daemon=True).start()
        if self.transcript:
            threading.Thread(target=self.simulate_transcript_display, daemon=True).start()

        self.update_status(f"Playing video at {fps:.1f} FPS")

    
    def simulate_transcript_display(self):
        """Simulate gradual display of the transcript"""
        if not self.transcript:
            return
            
        words = self.transcript.split()
        for i in range(len(words)):
            if not self.playing:  # Stop if playback has ended
                break
            current_speed = self.transcript_speed_var.get()
            current_transcript = " ".join(words[:i+1])
            self.update_transcript(current_transcript)
            time.sleep(0.3 / current_speed)  # Adjusted for wider speed range
    
    def _schedule_next_frame(self, fps):
        """
        Display one frame and schedule the next via Tk.after().
        fps: the video’s native frames-per-second.
        """
        if not self.playing:
            return  # stopped

        ret, frame = self.video_capture.read()
        if not ret:
            # EOF
            self.update_status("Video playback finished")
            self.playing = False
            # Re‑enable controls
            self.load_button.config(state=tk.NORMAL)
            self.start_button.config(state=tk.NORMAL)
            return

        # Convert & resize
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = self.resize_frame(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(frame_resized))
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        # Compute delay (ms) from fps and slider
        delay_ms = int(1000 / (fps * self.video_speed_var.get()))

        # Schedule next frame
        self.root.after(delay_ms, lambda: self._schedule_next_frame(fps))


        
    def update_video_label(self, imgtk):
        """Update the video label with a new frame"""
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
    
    def update_transcript(self, text):
        """Update the transcript text box"""
        self.root.after(0, lambda t=text: self._update_transcript_ui(t))
    
    def _update_transcript_ui(self, text):
        """Actually update the UI with transcript text"""
        self.transcript_text.delete(1.0, tk.END)
        self.transcript_text.insert(tk.END, text)
        self.transcript_text.see(tk.END)
    
    def update_status(self, text):
        """Update the status bar"""
        self.root.after(0, lambda t=text: self.status_var.set(t))
    
    def update_graph_viz_with_current(self, nodes_to_show, edges_to_show):
        """Update graph visualization maintaining current visibility state with improved reset handling"""
        # Clear any existing content
        self.ax.clear()
        
        # Handle the case of empty nodes gracefully
        if not nodes_to_show:
            # Draw empty graph with just the axes setup
            self.update_empty_graph()
            return
        
        # Create a subgraph with only the visible nodes
        G_partial = self.G.subgraph(nodes_to_show)
        
        # Create node color map with highlighting for selected node
        node_colors = []
        for node in G_partial.nodes():
            if node == self.selected_node:
                node_colors.append('orange')  # Highlight color for selected node
            else:
                node_colors.append('lightblue')  # Default color
        
        # Set tight layout with more padding
        plt.tight_layout(pad=3.0)
        
        # Make sure we have appropriate figure margins
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        
        # Draw nodes with dynamic colors - increased size
        nx.draw_networkx_nodes(
            G_partial, self.pos,
            ax=self.ax,
            node_color=node_colors,
            node_size=4000,  # Large nodes to accommodate text
            alpha=0.9
        )
        
        # Draw edges only if they exist
        if edges_to_show:
            nx.draw_networkx_edges(
                G_partial, self.pos,
                ax=self.ax,
                edgelist=edges_to_show,
                edge_color='gray',
                arrows=True,
                arrowsize=20,
                width=1.5  # Slightly thicker edges
            )
        
        # Create cleaner labels by removing "assembly_step_" prefix
        labels = {}
        for node in G_partial.nodes():
            if node.startswith("assembly_step_"):
                # Extract the number and make a cleaner label
                step_num = node.replace("assembly_step_", "")
                labels[node] = f"Step {step_num}"
            else:
                labels[node] = node
        
        # Draw labels with improved visibility
        nx.draw_networkx_labels(
            G_partial, self.pos,
            labels=labels,
            ax=self.ax,
            font_size=10,       # Increased font size
            font_weight='bold', # Make text bold
            font_color='black'  # Dark text for better contrast
        )
        
        # Remove axis
        self.ax.axis('off')
        
        # Set title
        self.ax.set_title("LEGO Assembly Steps")
        
        # Connect the click event if not already connected
        if not hasattr(self, '_cid'):
            self._cid = self.fig.canvas.mpl_connect('button_press_event', self.on_node_click)
        
        # Ensure the canvas is completely updated
        self.canvas.draw()
        self.canvas.flush_events()


    def update_graph_viz(self, G_partial, current_edges):
        """Update the graph visualization"""
        self.ax.clear()
        
        # Store currently visible nodes based on G_partial
        self.current_visible_nodes = list(G_partial.nodes())
        self.current_visible_edges = current_edges
        
        # Create node color map with highlighting for selected node
        node_colors = []
        for node in G_partial.nodes():
            if node == self.selected_node:
                node_colors.append('orange')  # Highlight color for selected node
            else:
                node_colors.append('lightblue')  # Default color
        
        # Set tight layout with more padding
        plt.tight_layout(pad=3.0)
        
        # Make sure we have appropriate figure margins
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        
        # Draw nodes with dynamic colors
        nx.draw_networkx_nodes(
            G_partial, self.pos,
            ax=self.ax,
            node_color=node_colors,
            node_size=4000,
            alpha=0.9
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            G_partial, self.pos,
            ax=self.ax,
            edgelist=current_edges,
            edge_color='gray',
            arrows=True,
            arrowsize=20,
            width=1.5  # Slightly thicker edges
        )
        
        # Create cleaner labels by removing "assembly_step_" prefix
        labels = {}
        for node in G_partial.nodes():
            if node.startswith("assembly_step_"):
                # Extract the number and make a cleaner label
                step_num = node.replace("assembly_step_", "")
                labels[node] = f"Step {step_num}"
            else:
                labels[node] = node
        
        # Draw labels with improved visibility
        nx.draw_networkx_labels(
            G_partial, self.pos,
            labels=labels,
            ax=self.ax,
            font_size=10,       # Increased font size
            font_weight='bold', # Make text bold
            font_color='black'  # Dark text for better contrast
        )
        
        # Remove axis
        self.ax.axis('off')
        
        # Set title explicitly
        self.ax.set_title("LEGO Assembly Steps")
        
        # Connect the click event if not already connected
        if not hasattr(self, '_cid'):
            self._cid = self.fig.canvas.mpl_connect('button_press_event', self.on_node_click)
        
        # Update the canvas
        self.canvas.draw()
        self.canvas.flush_events()  # Added to ensure complete rendering


    def animate_graph(self):
        """Animate the graph building process, aborting immediately on reset."""
        if not self.graph_data:
            self.update_status("No graph data available")
            return

        nodes = self.graph_data.get("nodes", [])
        edges = self.graph_data.get("edges", [])

        if not nodes:
            self.update_status("Graph data contains no nodes")
            return

        self.update_status("Starting graph animation...")

        # 1) Initial pause
        time.sleep(INITIAL_PAUSE_SECONDS)
        if not self.playing:
            return  # user reset before animation started

        # 2) Clear out any old graph
        self.current_visible_nodes = []
        self.current_visible_edges = []
        self.root.after(0, self.update_empty_graph)
        
        # 3) Wait for empty‐graph to render
        time.sleep(EMPTY_GRAPH_DRAW_DELAY)
        if not self.playing:
            return

        # 4) Now add nodes one by one
        for i, node in enumerate(nodes):
            if not self.playing:
                break

            # delay between nodes (constant speed)
            node_delay = 4.0 / GRAPH_SPEED
            time.sleep(node_delay + NODE_TRANSITION_DELAY)
            if not self.playing:
                break

            # build up the visible list
            name = node.get("task_name", f"Task {i}")
            self.current_visible_nodes = [nodes[j].get("task_name", f"Task {j}") for j in range(i + 1)]

            # pick out any edges among them
            self.current_visible_edges = [
                (e["source_node"], e["target_node"])
                for e in edges
                if e.get("source_node") in self.current_visible_nodes
                and e.get("target_node") in self.current_visible_nodes
            ]

            # schedule the draw on the main thread
            self.update_status(f"Adding node: {name}")
            visible_nodes = list(self.current_visible_nodes)
            visible_edges = list(self.current_visible_edges)
            self.root.after(0, lambda nv=visible_nodes, ev=visible_edges: 
                            self.update_graph_viz_with_current(nv, ev))

        self.update_status("Graph visualization complete")



def check_dependencies():
    """Check if required packages are installed"""
    required = {
        'opencv-python': 'cv2',
        'pillow': 'PIL',
        'moviepy': 'moviepy',
        'pygame': 'pygame',
        'SpeechRecognition': 'speech_recognition',
        'networkx': 'networkx',
        'matplotlib': 'matplotlib'
    }
    
    missing = []
    
    print("Checking dependencies...")
    for package, import_name in required.items():
        try:
            __import__(import_name.split('.')[0])
            print(f"✓ {package}")
        except ImportError:
            missing.append(package)
            print(f"✗ {package}")
    
    if missing:
        print("\nMissing dependencies. Please install with:")
        print(f"pip install {' '.join(missing)}")
        sys.exit(1)
    
    print("All dependencies satisfied!")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run the integrated visualization for LEGO assembly')
    parser.add_argument('--video', help='Path to the video file to process (optional)')
    args = parser.parse_args()
    
    # Check dependencies
    check_dependencies()
    
    # Use provided video path or let user select one
    video_path = args.video
    
    # Create a new Tk root (we already initialized Tk earlier)
    main_root = tk.Toplevel()
    main_root.protocol("WM_DELETE_WINDOW", lambda: sys.exit(0))  # Handle window close correctly
    
    app = CompleteVisualizationApp(main_root, video_path)
    
    # Start the main loop
    root.mainloop()


if __name__ == "__main__":
    main()