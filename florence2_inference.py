# Import necessary libraries
import os
import cv2
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import supervision as sv
import numpy as np

# Set HOME directory
HOME = os.getcwd()

# Function to initialize model and processor
def initialize_model(checkpoint="microsoft/Florence-2-large-ft", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)

    return model, processor, device

# Function to run inference on an image
def run_inference(image: Image, model, processor, device, task: str, text: str = ""):
    prompt = task + text
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    response = processor.post_process_generation(generated_text, task=task, image_size=image.size)
    return response

# Function to process a single frame
def process_frame(frame, model, processor, device, task, text):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)

    response = run_inference(image=image, model=model, processor=processor, device=device, task=task, text=text)
    detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, response, resolution_wh=image.size)

    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
    image = mask_annotator.annotate(image, detections)
    image = label_annotator.annotate(image, detections)

    frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return frame_bgr

# Function to process a video
def process_video(input_video_path, output_video_path, model, processor, device, task, text, frame_step=1):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error opening video file {input_video_path}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total number of frames: {frame_count}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps / frame_step, (width, height))

    frame_idx = 0
    processed_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            print(f"Processing frame {frame_idx+1}/{frame_count}...", end="\r")
            processed_frame = process_frame(frame, model, processor, device, task, text)
            processed_frames += 1
        else:
            processed_frame = frame

        out.write(processed_frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"\nProcessed video saved to: {output_video_path}")
    print(f"Total processed frames: {processed_frames}")

# Main code execution
model, processor, device = initialize_model()

# Create data directory if it doesn't exist
data_dir = os.path.join(HOME, "data")
os.makedirs(data_dir, exist_ok=True)

# Download the video
video_url = "https://videos.pexels.com/video-files/3015510/3015510-hd_1920_1080_24fps.mp4"
input_video_path = os.path.join(data_dir, "hot_air_balloons.mp4")
if not os.path.exists(input_video_path):
    print("Downloading video...")
    os.system(f"wget -q {video_url} -O {input_video_path}")
else:
    print("Video already downloaded.")

# Define task and text
task = "<REFERRING_EXPRESSION_SEGMENTATION>"
text = "person"

# Output video path
output_video_path = os.path.join(data_dir, "referring_expression_segmentation.mp4")

# Process the video
process_video(
    input_video_path=input_video_path,
    output_video_path=output_video_path,
    model=model,
    processor=processor,
    device=device,
    task=task,
    text=text,
    frame_step=1
)
