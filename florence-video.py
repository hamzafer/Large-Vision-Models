# %% [markdown]
# # Notebook Description  
# This notebook demonstrates video processing using the Florence2 Large Vision Model (LVM). It includes steps for setting up the environment, importing essential libraries, and configuring logging for streamlined operations.

# %% [markdown]
# ### **Usage Notes**
# 
# - **Supported Tasks**:  
#   A total of **10 tasks** are supported for processing videos. Each task performs a specific function, and the output videos are saved with numbered filenames for easier identification. Below is the list of tasks:
# 
#   1. **Object Detection (`<OD>`)**:  
#      Detects objects and draws bounding boxes around them.  
#      **Example**:  
#      ```python
#      task = "<OD>"
#      text = ""
#      ```
# 
#   2. **Caption to Phrase Grounding (`<CAPTION_TO_PHRASE_GROUNDING>`)**:  
#      Highlights specific objects (e.g., "person") using bounding boxes.  
#      **Example**:  
#      ```python
#      task = "<CAPTION_TO_PHRASE_GROUNDING>"
#      text = "person"
#      ```
# 
#   3. **Region Proposal (`<REGION_PROPOSAL>`)**:  
#      Identifies and marks object regions with bounding boxes, also displaying object counts.  
#      **Example**:  
#      ```python
#      task = "<REGION_PROPOSAL>"
#      text = ""
#      ```
# 
#   4. **Open Vocabulary Detection (`<OPEN_VOCABULARY_DETECTION>`)**:  
#      Detects objects, draws bounding boxes, and overlays captions describing the objects.  
#      **Example**:  
#      ```python
#      task = "<OPEN_VOCABULARY_DETECTION>"
#      text = "person"
#      ```
# 
#   5. **Detailed Caption (`<DETAILED_CAPTION>`)**:  
#      Adds detailed captions describing the scene in each frame of the video.  
#      **Example**:  
#      ```python
#      task = "<DETAILED_CAPTION>"
#      text = ""
#      ```
# 
#   6. **More Detailed Caption (`<MORE_DETAILED_CAPTION>`)**:  
#      Adds a more detailed caption, offering finer scene descriptions in the video.  
#      **Example**:  
#      ```python
#      task = "<MORE_DETAILED_CAPTION>"
#      text = ""
#      ```
# 
#   7. **OCR with Region (`<OCR_WITH_REGION>`)**:  
#      Extracts text regions and overlays OCR results on the video with bounding boxes.  
#      **Example**:  
#      ```python
#      task = "<OCR_WITH_REGION>"
#      text = ""
#      ```
# 
#   8. **OCR (`<OCR>`)**:  
#      Recognizes text in the video frames and overlays the recognized text directly.  
#      **Example**:  
#      ```python
#      task = "<OCR>"
#      text = ""
#      ```
# 
#   9. **Region to Segmentation (`<REGION_TO_SEGMENTATION>`)**:  
#      Segments regions of interest in the video, highlighting them minimally. This task primarily outputs subtle visual cues and is mainly experimental.  
#      Still not sure why it is not showing any changes on the video.
#      **Example**:  
#      ```python
#      task = "<REGION_TO_SEGMENTATION>"
#      text = "person"
#      ```
# 
#   10. **Referring Expression Segmentation (`<REFERRING_EXPRESSION_SEGMENTATION>`)**:  
#       Segments a specified object (e.g., "person") and highlights it with a mask. This task can take significantly longer processing times.  
#       **Example**:  
#       ```python
#       task = "<REFERRING_EXPRESSION_SEGMENTATION>"
#       text = "person"
#       ```
# 
# - **Processing Time**:  
#   The estimated processing time for each task is included in the task list. The script logs the actual time taken for processing each task after completion.
# 
# - **Customizing Frame Processing**:  
#   Modify the `frame_step` parameter in the `process_video` function to process fewer frames for faster results. For example:  
#   ```python
#   frame_step = 5  # Processes every 5th frame.
#   ```
# 
# - **Output Filenames**:  
#   The output videos are saved with a descriptive filename pattern:  
#   `"<task_number>_<task_description>_<input_video_name>.mp4"`  
#   Example:  
#   For task 1 (`<OD>`), processing an input video named `hot_air_balloons.mp4` generates:  
#   `01_object_detection_hot_air_balloons.mp4`
# 
# - **Viewing Output**:  
#   Use the `display_video` function to view processed videos directly in the Jupyter notebook. For example:  
#   ```python
#   display_video(output_video_path)
#   ```

# %% [markdown]
# Start Time

# %%
import time

# Record the start time
start_time = time.time()
print("Notebook execution started.")


# %%
# Import necessary libraries
import os
import cv2
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image, ImageDraw, ImageFont
import supervision as sv
import numpy as np
from IPython.display import HTML
import base64
import logging

# Set HOME directory
HOME = os.getcwd()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# %% [markdown]
# ### Function: `initialize_model`  
# Initializes the Florence-2 model and processor, automatically selecting GPU or CPU for execution.

# %%
def initialize_model(checkpoint="microsoft/Florence-2-large-ft", device=None):
    """
    Initialize the Florence-2 model and processor.

    Parameters:
    - checkpoint: The model checkpoint to use.
    - device: The device to run the model on.

    Returns:
    - model: The initialized model.
    - processor: The initialized processor.
    - device: The device used.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load the model and processor
    model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)

    return model, processor, device


# %% [markdown]
# ### Function: `process_frame`  
# Processes a single video frame using the Florence-2 model, applying tasks such as object detection, captioning, segmentation, or OCR, and annotates the frame accordingly.

# %%
def process_frame(frame, model, processor, device, task, text):
    """
    Process a single video frame with the specified task.

    Parameters:
    - frame: The video frame to process.
    - model: The initialized model.
    - processor: The initialized processor.
    - device: The device to run the model on.
    - task: The task to perform (e.g., "<OD>", "<DETAILED_CAPTION>", etc.).
    - text: The text input for the task.

    Returns:
    - frame_bgr: The processed frame in BGR format.
    """
    # Convert the frame to RGB and then to a PIL Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)

    # Preprocess the input for the model
    prompt = task + text
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    # Generate results
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )

    # Post-process the generated text
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    response = processor.post_process_generation(generated_text, task=task, image_size=image.size)

    # Handle different tasks based on the response
    if task in ["<OD>", "<OPEN_VOCABULARY_DETECTION>", "<CAPTION_TO_PHRASE_GROUNDING>", "<REGION_PROPOSAL>"]:
        # Tasks that output bounding boxes
        detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, response, resolution_wh=image.size)
        bounding_box_annotator = sv.BoundingBoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
        label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
        image = bounding_box_annotator.annotate(image, detections)
        image = label_annotator.annotate(image, detections)

    elif task == "<REFERRING_EXPRESSION_SEGMENTATION>":
        # Task that outputs segmentation masks
        detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, response, resolution_wh=image.size)
        mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
        label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
        image = mask_annotator.annotate(image, detections)
        image = label_annotator.annotate(image, detections)

    elif task in ["<OCR_WITH_REGION>"]:
        # Task that outputs OCR results with regions
        detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, response, resolution_wh=image.size)
        bounding_box_annotator = sv.BoundingBoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
        label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX, text_scale=1.5, text_thickness=2)
        image = bounding_box_annotator.annotate(image, detections)
        image = label_annotator.annotate(image, detections)

    elif task in ["<OCR>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"]:
        # Tasks that output text captions
        caption = response.get(task, "")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        text_position = (10, 10)
        draw.text(text_position, caption, fill="red", font=font)

    else:
        # For any other tasks, we can print the response or handle accordingly
        logger.warning(f"Unhandled task or output format for task: {task}")

    # Convert back to OpenCV format (BGR) for saving the video
    frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    return frame_bgr


# %% [markdown]
# ### Function: `process_video`  
# Processes a video frame by frame using the Florence-2 model, applying a specified task (e.g., object detection, captioning). Outputs an annotated video, with options to skip frames for faster processing.

# %%
def process_video(input_video_path, output_video_path, model, processor, device, task, text, frame_step=1):
    """
    Process a video with the specified task.

    Parameters:
    - input_video_path: Path to the input video file.
    - output_video_path: Path to save the processed video.
    - model: The initialized model.
    - processor: The initialized processor.
    - device: Device to run the model on.
    - task: The task to perform (e.g., "<OD>", "<DETAILED_CAPTION>", etc.).
    - text: The text input for the task.
    - frame_step: Process every Nth frame (default is 1, i.e., process every frame).
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        logger.error(f"Error opening video file {input_video_path}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Total number of frames: {frame_count}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare to save the processed video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(output_video_path, fourcc, fps / frame_step, (width, height))

    # Process frames
    frame_idx = 0
    processed_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            logger.info(f"Processing frame {frame_idx+1}/{frame_count}")
            processed_frame = process_frame(frame, model, processor, device, task, text)
            processed_frames += 1
        else:
            processed_frame = frame  # Use the original frame if not processing

        # Write the frame to the output video
        out.write(processed_frame)
        frame_idx += 1

    # Release video capture and writer objects
    cap.release()
    out.release()
    logger.info(f"Processed video saved to: {output_video_path}")
    logger.info(f"Total processed frames: {processed_frames}")


# %% [markdown]
# ### Function: `display_video`  
# Displays a video directly in the notebook with adjustable width for visualization.

# %%
def display_video(video_path, video_width=600):
    """
    Display a video inside the notebook.

    Parameters:
    - video_path: Path to the video file.
    - video_width: Width of the video in the display.

    Returns:
    - HTML object to display the video.
    """
    video_file = open(video_path, "rb").read()
    data_url = "data:video/mp4;base64," + base64.b64encode(video_file).decode()
    return HTML(f"""
    <video width={video_width} controls>
          <source src="{data_url}" type="video/mp4">
    </video>
    """)


# %% [markdown]
# ### Code: Initialize Model and Processor  
# Initializes the Florence-2 model and processor and sets the execution device (GPU or CPU).

# %%
# Initialize model and processor
model, processor, device = initialize_model()


# %% [markdown]
# ### Code: Check and Download Video  
# Checks if the video file already exists in the `data` directory. If not, downloads the video from the specified URL and logs the progress. Skips downloading if the file is already present.

# %%
import os
from urllib.parse import urlparse

# Define generic variables for data directory, video URL, and desired video name
data_dir_name = "data"
video_url = "https://videos.pexels.com/video-files/3015510/3015510-hd_1920_1080_24fps.mp4"
desired_video_name = "hot_air_balloons.mp4"

# Create data directory if it doesn't exist
data_dir = os.path.join(HOME, data_dir_name)
os.makedirs(data_dir, exist_ok=True)

# Define the final path for the video after renaming
input_video_path = os.path.join(data_dir, desired_video_name)

# Check if the video already exists
if not os.path.exists(input_video_path):
    logger.info(f"Downloading the video from {video_url}...")
    temp_video_path = os.path.join(data_dir, os.path.basename(urlparse(video_url).path))
    
    # Download the video to a temporary path
    os.system(f"wget -q {video_url} -O {temp_video_path}")
    logger.info("Download complete.")
    
    # Rename to the desired name
    os.rename(temp_video_path, input_video_path)
    logger.info(f"Video renamed to {desired_video_name}.")
else:
    logger.info("Video already exists. Skipping download.")

print("Video path:", input_video_path)

# %% [markdown]
# ### Code: Process Video for Multiple Tasks  
# Iterates through a predefined list of tasks, processes the video for each task using the Florence-2 model, and saves the output with task-specific filenames. Allows optional display of processed videos within the notebook.

# %%
import time

# Quick video path
input_video_path = os.path.join(data_dir, "november_leaves.mp4")

# Define a list of tasks and corresponding texts
# Supported tasks list
tasks = [
    # Object Detection and Proposal Tasks:
    # 1. Object Detection - works: good object detection and bounding boxes.
    {"task": "<OD>", "text": "", "time_minutes": 5, "description": "object_detection"},

    # 2. Caption to Phrase Grounding - works: detects the object and highlights using bounding box.
    {"task": "<CAPTION_TO_PHRASE_GROUNDING>", "text": "person", "time_minutes": 6, "description": "caption_to_phrase"},

    # 3. Region Proposal - works: draws bounding boxes around objects and puts count of objects.
    {"task": "<REGION_PROPOSAL>", "text": "", "time_minutes": 7, "description": "region_proposal"},

    # 4. Open Vocabulary Detection - works: draws bounding boxes and puts captions on the object.
    {"task": "<OPEN_VOCABULARY_DETECTION>", "text": "person", "time_minutes": 8, "description": "open_vocab_detection"},

    # Captioning Tasks:
    # 5. Detailed Caption - works: puts caption on the video.
    {"task": "<DETAILED_CAPTION>", "text": "", "time_minutes": 10, "description": "detailed_caption"},

    # 6. More Detailed Caption - works: puts a more detailed caption on the video.
    {"task": "<MORE_DETAILED_CAPTION>", "text": "", "time_minutes": 10, "description": "more_detailed_caption"},

    # Text Recognition (OCR) Tasks:
    # 7. OCR with Region - works: puts OCR text on the video with bounding boxes.
    {"task": "<OCR_WITH_REGION>", "text": "", "time_minutes": 4, "description": "ocr_with_region"},

    # 8. OCR - works: performs OCR without specific region marking.
    {"task": "<OCR>", "text": "", "time_minutes": 10, "description": "ocr"},

    # Segmentation and Highlighting Tasks:
    # 9. Region to Segmentation - works: does minimal segmentation on the video.
    # Also takes a long time to process.
    {"task": "<REGION_TO_SEGMENTATION>", "text": "person", "time_minutes": 10, "description": "region_to_segmentation"},

    # 10. Referring Expression Segmentation - works: takes a long time. 
    # Does segmentation of the object, highlighting it with a purple mask. not very accurate (50min for 19sec video).
    {"task": "<REFERRING_EXPRESSION_SEGMENTATION>", "text": "person", "time_minutes": 50, "description": "referring_expression_segmentation"}
]

# Loop through each task and process the video
for idx, item in enumerate(tasks, start=1):
    task = item["task"]
    text = item["text"]
    time_minutes = item["time_minutes"]
    description = item["description"]
    
    # Generate a numbered and descriptive output video file name
    output_video_path = os.path.join(data_dir, f"{idx:02d}_{description}_{os.path.basename(input_video_path)}")
    
    # Check if the output video already exists
    if os.path.exists(output_video_path):
        print(f"Skipping Task {idx}: {task} as output video already exists at {output_video_path}")
        continue

    print(f"Processing Task {idx}: {task} with text: '{text}' (Estimated time: {time_minutes} minutes)")
    
    # Process the video
    start_time = time.time()
    process_video(
        input_video_path=input_video_path,
        output_video_path=output_video_path,
        model=model,
        processor=processor,
        device=device,
        task=task,
        text=text,
        frame_step=1  # Process every frame; increase for faster processing
    )
    end_time = time.time()
    
    # Calculate and print the actual time taken
    elapsed_minutes = (end_time - start_time) / 60
    print(f"Task {idx} completed in {elapsed_minutes:.2f} minutes. Output video: {output_video_path}")


# %% [markdown]
# ### Code: Display Processed Video  
# Displays the processed video directly inside the notebook, allowing for immediate visualization of the results.

# %%
output_video_path = "data/hot_air_balloons.mp4"

# Display the processed video inside the notebook
display_video(output_video_path)


# %% [markdown]
# Captions

# %% [markdown]
# End Time and Total Time Logging

# %%
# Record the end time
end_time = time.time()

# output: Notebook execution completed in 99.19 minutes.

# Calculate and log the total execution time
total_time = end_time - start_time
print(f"Notebook execution completed in {total_time/60:.2f} minutes.")



