import os
import json
import time
import csv
import cv2
import base64
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from datetime import datetime
from openai import OpenAI
import io

class AssemblyVerificationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Assembly Verification Tool")
        self.root.geometry("1400x900")  # Increased window size
        
        # Initialize variables
        self.task_graph_path = "/Users/legomac/Desktop/CS 159/speech_to_graph/IMG_0501_task_graph.json"
        self.task_nodes = []
        self.current_task_index = 0
        self.reference_image = None
        self.user_image = None
        self.verification_result = None
        self.inference_time = 0
        self.metrics_file = "4o_mini_verification_metrics.csv"
        
        # Check if metrics file exists, create it if not
        if not os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Timestamp', 'Task Name', 'Inference Time (s)', 'VLM Result', 'Human Marked Correct'])
        
        # Load task graph
        self.load_task_graph()
        
        # Setup UI
        self.setup_ui()
        
        # Load first task
        if self.task_nodes:
            self.load_current_task()

    def load_task_graph(self):
        """Load the task graph from JSON file"""
        try:
            with open(self.task_graph_path, 'r') as f:
                task_graph_data = json.load(f)
                
            # Extract task nodes from the graph data
            self.task_nodes = []
            for node in task_graph_data.get('nodes', []):
                self.task_nodes.append({
                    'task_name': node.get('task_name', f"Task {len(self.task_nodes) + 1}"),
                    'task_description': node.get('task_description', ''),
                    'reference_image': node.get('reference_image', ''),
                    'required_objects': node.get('required_objects', {}),
                })
                
            # Get the image directory from the task graph
            self.image_directory = task_graph_data.get('image_directory', '')
                
            print(f"Loaded {len(self.task_nodes)} task nodes")
            print(f"Image directory: {self.image_directory}")
        except Exception as e:
            print(f"Error loading task graph: {e}")
            messagebox.showerror("Error", f"Failed to load task graph: {e}")
            self.task_nodes = []
            self.image_directory = ''

    def setup_ui(self):
        """Set up the user interface"""
        # Main container with two columns
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left column for task info and images
        left_column = ttk.Frame(main_frame)
        left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Right column for verification results
        right_column = ttk.Frame(main_frame)
        right_column.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        right_column.config(width=400)
                
        # Task info frame
        task_frame = ttk.LabelFrame(left_column, text="Current Task", padding=10)
        task_frame.pack(fill=tk.X, pady=10)
        
        self.task_label = ttk.Label(task_frame, text="No task loaded", font=("Arial", 12), wraplength=800, justify=tk.LEFT)
        self.task_label.pack(pady=5)
        
        # Add the required objects label
        self.required_objects_label = ttk.Label(task_frame, text="Required objects: None", font=("Arial", 10), wraplength=800, justify=tk.LEFT)
        self.required_objects_label.pack(pady=5)
        
        # Navigation buttons
        nav_frame = ttk.Frame(task_frame)
        nav_frame.pack(pady=5)
        
        self.prev_button = ttk.Button(nav_frame, text="Previous Task", command=self.prev_task)
        self.prev_button.pack(side=tk.LEFT, padx=5)
        
        self.next_button = ttk.Button(nav_frame, text="Next Task", command=self.next_task)
        self.next_button.pack(side=tk.LEFT, padx=5)
        
        # Image display frame - increased size
        image_frame = ttk.Frame(left_column)
        image_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Reference image section
        ref_frame = ttk.LabelFrame(image_frame, text="Reference Image", padding=10)
        ref_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.ref_image_label = ttk.Label(ref_frame)
        self.ref_image_label.pack(fill=tk.BOTH, expand=True)
        
        # User image section
        user_frame = ttk.LabelFrame(image_frame, text="Your Assembly", padding=10)
        user_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.user_image_label = ttk.Label(user_frame)
        self.user_image_label.pack(fill=tk.BOTH, expand=True)
        
        # Image selection button
        self.select_image_button = ttk.Button(user_frame, text="Select Your Assembly Image", command=self.select_image)
        self.select_image_button.pack(pady=10)
        
        # Verification controls on the right column
        verify_frame = ttk.LabelFrame(right_column, text="Verification", padding=10)
        verify_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.verify_button = ttk.Button(verify_frame, text="Verify Assembly", command=self.verify_assembly, state=tk.DISABLED)
        self.verify_button.pack(pady=5)
        
        # Results display - much larger space for results
        result_frame = ttk.Frame(verify_frame)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Add scrollbar for result text
        result_scroll = ttk.Scrollbar(result_frame)
        result_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Use Text widget instead of Label for better scrolling
        self.result_text = tk.Text(result_frame, wrap=tk.WORD, font=("Arial", 12), 
                                  width=30, height=20, 
                                  yscrollcommand=result_scroll.set)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.result_text.insert(tk.END, "No verification performed yet")
        self.result_text.config(state=tk.DISABLED)  # Make read-only
        
        result_scroll.config(command=self.result_text.yview)
        
        self.time_label = ttk.Label(verify_frame, text="Inference time: -", font=("Arial", 10))
        self.time_label.pack(pady=5)
        
        # Metrics frame
        metrics_frame = ttk.LabelFrame(right_column, text="Metrics", padding=10)
        metrics_frame.pack(fill=tk.X, pady=10)
        
        self.correct_var = tk.BooleanVar(value=False)
        self.correct_check = ttk.Checkbutton(metrics_frame, text="Mark as Correct", variable=self.correct_var)
        self.correct_check.pack(side=tk.LEFT, padx=5)
        
        self.save_metrics_button = ttk.Button(metrics_frame, text="Save Metrics", command=self.save_metrics, state=tk.DISABLED)
        self.save_metrics_button.pack(side=tk.LEFT, padx=5)

    def load_current_task(self):
        """Load the current task and its reference image"""
        if 0 <= self.current_task_index < len(self.task_nodes):
            current_task = self.task_nodes[self.current_task_index]
            
            # Update task label with name and description
            task_name = current_task.get('task_name', '')
            task_desc = current_task.get('task_description', '')
            
            self.task_label.config(
                text=f"Task {self.current_task_index+1}/{len(self.task_nodes)}: {task_name}\n\nInstructions: {task_desc}",
                wraplength=800,
                justify=tk.LEFT
            )
            
            # Update required objects label
            required_objects = current_task.get('required_objects', [])
            if required_objects:
                if isinstance(required_objects, list):
                    objects_str = ", ".join(required_objects)
                else:
                    objects_str = str(required_objects)
                self.required_objects_label.config(text=f"Required objects: {objects_str}")
            else:
                self.required_objects_label.config(text="Required objects: None")
            
            # Load reference image
            ref_image_path = current_task.get('reference_image', '')
            try:
                # Check if path is absolute or relative
                if not os.path.isabs(ref_image_path):
                    # Use the image directory from the task graph
                    # Fix: Use the base directory without adding speech_to_graph
                    base_dir = "/Users/legomac/Desktop/CS 159"  # Direct path to the base directory
                    img_dir = os.path.join(base_dir, self.image_directory)
                    ref_image_path = os.path.join(img_dir, ref_image_path)
                
                print(f"Loading reference image from: {ref_image_path}")
                self.reference_image = cv2.imread(ref_image_path)
                if self.reference_image is None:
                    raise Exception(f"Could not read image at {ref_image_path}")
                
                # Convert to PIL and display
                ref_image_rgb = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(ref_image_rgb)
                
                # Resize to fit display area (larger size)
                pil_image = self.resize_image_to_fit(pil_image, 600, 500)
                
                # Convert to PhotoImage and display
                tk_image = ImageTk.PhotoImage(pil_image)
                self.ref_image_label.config(image=tk_image)
                self.ref_image_label.image = tk_image  # Keep reference to prevent garbage collection
                
            except Exception as e:
                print(f"Error loading reference image: {e}")
                messagebox.showerror("Error", f"Failed to load reference image: {e}")
                self.ref_image_label.config(image='')
                self.reference_image = None
            
            # Reset user image and results
            self.user_image = None
            self.user_image_label.config(image='')
            
            # Update result text
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "No verification performed yet")
            self.result_text.config(state=tk.DISABLED)
            
            self.time_label.config(text="Inference time: -")
            self.verify_button.config(state=tk.DISABLED)
            self.save_metrics_button.config(state=tk.DISABLED)
            self.correct_var.set(False)
            
            # Update navigation buttons
            self.prev_button.config(state=tk.NORMAL if self.current_task_index > 0 else tk.DISABLED)
            self.next_button.config(state=tk.NORMAL if self.current_task_index < len(self.task_nodes) - 1 else tk.DISABLED)
        else:
            self.task_label.config(text="No task loaded")
            self.required_objects_label.config(text="Required objects: None")
            self.ref_image_label.config(image='')
            self.reference_image = None

    def select_image(self):
        """Open file dialog to select user's assembly image"""
        file_path = filedialog.askopenfilename(
            title="Select Assembly Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            try:
                self.user_image = cv2.imread(file_path)
                if self.user_image is None:
                    raise Exception(f"Could not read image at {file_path}")
                
                # Convert to PIL and display
                user_image_rgb = cv2.cvtColor(self.user_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(user_image_rgb)
                
                # Resize to fit display area (larger size)
                pil_image = self.resize_image_to_fit(pil_image, 600, 500)
                
                # Convert to PhotoImage and display
                tk_image = ImageTk.PhotoImage(pil_image)
                self.user_image_label.config(image=tk_image)
                self.user_image_label.image = tk_image  # Keep reference to prevent garbage collection
                
                # Enable verify button
                self.verify_button.config(state=tk.NORMAL)
                
            except Exception as e:
                print(f"Error loading user image: {e}")
                messagebox.showerror("Error", f"Failed to load image: {e}")
                self.user_image = None

    def resize_image_to_fit(self, pil_image, max_width, max_height):
        """Resize image to fit within max dimensions while preserving aspect ratio"""
        width, height = pil_image.size
        
        # Calculate aspect ratio
        aspect_ratio = width / height
        
        if width > max_width or height > max_height:
            if width / max_width > height / max_height:
                # Width is the limiting dimension
                new_width = max_width
                new_height = int(new_width / aspect_ratio)
            else:
                # Height is the limiting dimension
                new_height = max_height
                new_width = int(new_height * aspect_ratio)
            
            return pil_image.resize((new_width, new_height), Image.LANCZOS)
        
        return pil_image

    def combine_images_side_by_side(self, image1, image2):
        """Combine two PIL images side-by-side for VLM comparison"""
        # Use the smaller height to avoid upscaling
        target_height = min(image1.height, image2.height)
        
        # Resize both images to match height
        image1_aspect = image1.width / image1.height
        image2_aspect = image2.width / image2.height
        
        image1_new_width = int(target_height * image1_aspect)
        image2_new_width = int(target_height * image2_aspect)
        
        image1_resized = image1.resize((image1_new_width, target_height), Image.LANCZOS)
        image2_resized = image2.resize((image2_new_width, target_height), Image.LANCZOS)
        
        # Create new image to hold both
        combined_width = image1_resized.width + image2_resized.width
        combined_image = Image.new('RGB', (combined_width, target_height))
        
        # Paste images side-by-side
        combined_image.paste(image1_resized, (0, 0))
        combined_image.paste(image2_resized, (image1_resized.width, 0))
        
        return combined_image

    def verify_assembly(self):
        """Verify user's assembly against reference image using OpenAI's VLM"""
        if self.reference_image is None or self.user_image is None:
            messagebox.showerror("Error", "Both reference and user images are required")
            return
        
        # Update UI to show verification is in progress
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Verification in progress...")
        self.result_text.config(state=tk.DISABLED)
        self.root.update()
        
        # Convert images to PIL format
        ref_image_rgb = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2RGB)
        user_image_rgb = cv2.cvtColor(self.user_image, cv2.COLOR_BGR2RGB)
        
        ref_image_pil = Image.fromarray(ref_image_rgb)
        user_image_pil = Image.fromarray(user_image_rgb)
        
        # Combine images side-by-side
        combined_image = self.combine_images_side_by_side(ref_image_pil, user_image_pil)
        
        # Save combined image for debugging
        os.makedirs("verification_images", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_image_path = os.path.join("verification_images", f"combined_{timestamp}.png")
        combined_image.save(combined_image_path)
        print(f"Saved combined image to: {combined_image_path}")
        
        try:
            # Initialize OpenAI client
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("OPENAI_API_KEY environment variable is not set")
                messagebox.showerror("Error", "OPENAI_API_KEY environment variable is not set")
                self.result_text.config(state=tk.NORMAL)
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, "Error: OpenAI API key not set")
                self.result_text.config(state=tk.DISABLED, foreground="red")
                return
            
            client = OpenAI(api_key=api_key)
            print("OpenAI client initialized")
            
            # Prepare VLM prompt for assembly comparison
            prompt = (
                "Compare the provided image of an assembly (on the right) with the reference assembly shown on the left. \n"
                "Focus only on verifying whether the EXACT positioning and EXACT connection of the parts match. \n"
                "Pay very close attention to how the pieces are connected and MAKE SURE this connection matches the assembly\n"
                "Return the result in the following JSON format:\n"
                "```json\n"
                "{\n"
                '    "results": "match" or "no_match",\n'
                '    "description": "Detailed description of the differences or confirmation of a match."\n'
                "}\n"
                "```"
            )
            
            # Start timing
            start_time = time.time()
            print("Starting verification...")
            
            # Encode image to base64
            buffer = io.BytesIO()
            combined_image.save(buffer, format="PNG")
            encoded_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
            img_type = "image/png"
            print("Image encoded")
            
            # Prepare messages for OpenAI API
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{img_type};base64,{encoded_string}"}}
                    ],
                }
            ]
            
            # Call OpenAI API
            print("Calling OpenAI API...")
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )
            print("Response received")
            
            # End timing
            end_time = time.time()
            self.inference_time = end_time - start_time
            
            # Extract and parse JSON response
            response_text = response.choices[0].message.content.strip()
            print(f"Raw response: {response_text}")
            
            # Handle response that may contain JSON code blocks
            json_start = response_text.find("```json")
            json_end = response_text.find("```", json_start + 1) if json_start != -1 else -1
            
            if json_start != -1 and json_end != -1:
                # Extract JSON from code block
                json_str = response_text[json_start + 7: json_end].strip()
                print(f"Extracted JSON: {json_str}")
                result_json = json.loads(json_str)
            else:
                # Try to parse entire response as JSON
                try:
                    result_json = json.loads(response_text)
                    print("Parsed entire response as JSON")
                except Exception as e:
                    print(f"Failed to parse as JSON: {e}")
                    # Fallback if parsing fails
                    result_json = {
                        "results": "error",
                        "description": f"Failed to parse VLM response: {response_text}"
                    }
            
            # Update UI with result
            self.verification_result = result_json
            print(f"Final result: {result_json}")
            
            result_text = "Match" if result_json.get("results") == "match" else "No Match"
            result_color = "green" if result_json.get("results") == "match" else "red"
            
            # Update result in the Text widget
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Result: {result_text}\n\n{result_json.get('description', '')}")
            self.result_text.config(state=tk.DISABLED)
            
            # Add a tag to colorize the result
            self.result_text.tag_add("result", "1.0", "1.end")
            self.result_text.tag_config("result", foreground=result_color, font=("Arial", 14, "bold"))
            
            self.time_label.config(text=f"Inference time: {self.inference_time:.2f} seconds")
            
            # Enable save metrics button
            self.save_metrics_button.config(state=tk.NORMAL)
            
        except Exception as e:
            print(f"Error during verification: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Verification failed: {e}")
            
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Verification error: {e}")
            self.result_text.config(state=tk.DISABLED)
            
            self.time_label.config(text="Inference time: -")

    def save_metrics(self):
        """Save verification metrics to CSV file"""
        if not hasattr(self, 'verification_result') or self.verification_result is None:
            messagebox.showerror("Error", "No verification result to save")
            return
        
        try:
            current_task = self.task_nodes[self.current_task_index]
            task_name = current_task.get('task_name', f"Task {self.current_task_index+1}")
            
            # Prepare row data
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            vlm_result = self.verification_result.get("results", "error")
            human_marked = "Yes" if self.correct_var.get() else "No"
            
            # Debug print before writing
            print(f"Saving metrics to CSV: {timestamp}, {task_name}, {self.inference_time:.2f}, {vlm_result}, {human_marked}")
            
            # Write to CSV
            try:
                with open(self.metrics_file, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        timestamp,
                        task_name,
                        f"{self.inference_time:.2f}",
                        vlm_result,
                        human_marked
                    ])
                print(f"Successfully wrote to {self.metrics_file}")
            except Exception as csv_error:
                print(f"Error writing to CSV: {csv_error}")
                import traceback
                traceback.print_exc()
                raise
            
            messagebox.showinfo("Success", "Metrics saved successfully")
            
            # Optionally, move to next task
            # if self.current_task_index < len(self.task_nodes) - 1:
            #     if messagebox.askyesno("Next Task", "Move to the next task?"):
            #         self.next_task()
            
        except Exception as e:
            print(f"Error saving metrics: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to save metrics: {e}")

    def prev_task(self):
        """Move to the previous task"""
        if self.current_task_index > 0:
            self.current_task_index -= 1
            self.load_current_task()

    def next_task(self):
        """Move to the next task"""
        if self.current_task_index < len(self.task_nodes) - 1:
            self.current_task_index += 1
            self.load_current_task()

def main():
    # Create output directory for verification images
    os.makedirs("verification_images", exist_ok=True)
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY environment variable is not set!")
        print("Set it using: export OPENAI_API_KEY=your_api_key")
    
    # Initialize Tkinter root window
    root = tk.Tk()
    app = AssemblyVerificationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()