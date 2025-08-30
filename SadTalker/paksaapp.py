import os
import sys
import torch
import torch.nn as nn
import numpy as np
import gradio as gr
import cv2
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union
from src.config import *  # Import FFmpeg configuration
from src.utils.video_utils import process_uploaded_file, create_video_from_frames

# Import GRUGAN from gesture_models
from src.gesture_models import GRUGAN

# Import SadTalker after other imports to avoid circular imports
from src.gradio_demo import SadTalker

# Set environment variables for FFmpeg
os.environ['PATH'] = f"C:\\ffmpeg\\bin;{os.environ['PATH']}"
os.environ['FFMPEG_BINARY'] = FFMPEG_PATH
os.environ['FFPROBE_BINARY'] = FFPROBE_PATH


try:
    import webui  # in webui
    in_webui = True
except:
    in_webui = False


def toggle_audio_file(choice):
    if choice == False:
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)
    
def ref_video_fn(path_of_ref_video):
    if path_of_ref_video is not None:
        return gr.update(value=True)
    else:
        return gr.update(value=False)

class AudioFeatureExtractor:
    """Extracts audio features for gesture generation"""
    def __init__(self, sample_rate=16000, n_mfcc=80, n_fft=1024, hop_length=512):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        
    def extract_mfcc(self, audio_path: str) -> torch.Tensor:
        """Extract MFCC features from audio file"""
        try:
            import librosa
            # Load audio file
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(
                y=y,
                sr=sr,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            # Convert to PyTorch tensor and add batch dimension
            return torch.FloatTensor(mfcc).unsqueeze(0)
        except Exception as e:
            print(f"Error extracting audio features: {e}")
            return None

class GestureGenerator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.audio_processor = AudioFeatureExtractor()
        self.model = self._load_gesture_model()
        
    def _load_gesture_model(self):
        """Initialize the gesture generation model"""
        model = GRUGAN(self.device).to(self.device)
        # Load pre-trained weights if available
        try:
            # Check for pre-trained weights in the checkpoints directory
            if os.path.exists('checkpoints/gesture_generator.pth'):
                model.load_state_dict(torch.load(
                    'checkpoints/gesture_generator.pth',
                    map_location=self.device
                ))
                print("Loaded pre-trained gesture generator weights")
        except Exception as e:
            print(f"Could not load gesture generator weights: {e}")
            
        model.eval()
        return model
    
    def preprocess_audio_features(self, audio_features: torch.Tensor) -> torch.Tensor:
        """Preprocess audio features for gesture generation"""
        # Normalize features
        mean = audio_features.mean(dim=-1, keepdim=True)
        std = audio_features.std(dim=-1, keepdim=True) + 1e-6
        normalized = (audio_features - mean) / std
        
        # Pad or truncate to expected sequence length
        target_length = 240  # Adjust based on your model's expected input
        if normalized.shape[-1] < target_length:
            # Pad with zeros
            padding = target_length - normalized.shape[-1]
            normalized = torch.nn.functional.pad(normalized, (0, padding))
        elif normalized.shape[-1] > target_length:
            # Truncate
            normalized = normalized[..., :target_length]
            
        return normalized.unsqueeze(0)  # Add batch dimension
    
    def postprocess_gestures(self, gestures: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        """Apply post-processing to generated gestures based on user parameters"""
        # Apply intensity
        gestures = gestures * params['intensity']
        
        # Apply speed (temporal interpolation)
        if params['speed'] != 1.0:
            from torch.nn.functional import interpolate
            # (batch, channels, time)
            gestures = interpolate(
                gestures,
                scale_factor=1.0/params['speed'],
                mode='linear',
                align_corners=False
            )
            
        # Apply style-specific modifications
        if params['style'] == "Casual":
            pass  # Keep as is
        elif params['style'] == "Formal":
            gestures = gestures * 0.8  # More subtle movements
        elif params['style'] == "Expressive":
            gestures = gestures * 1.3  # Exaggerated movements
        elif params['style'] == "Subtle":
            gestures = gestures * 0.6  # Very subtle movements
            
        return gestures
    
    def generate_gestures(self, audio_path: str, params: Dict[str, Any]) -> torch.Tensor:
        """
        Generate gesture sequences from audio file
        
        Args:
            audio_path: Path to the input audio file
            params: Dictionary containing gesture generation parameters
            
        Returns:
            torch.Tensor: Generated gesture sequences
        """
        try:
            # Extract audio features
            audio_features = self.audio_processor.extract_mfcc(audio_path)
            if audio_features is None:
                raise ValueError("Failed to extract audio features")
                
            # Preprocess features
            processed_features = self.preprocess_audio_features(audio_features)
            
            # Generate gestures
            with torch.no_grad():
                # Move to device and add batch dimension if needed
                if len(processed_features.shape) == 2:
                    processed_features = processed_features.unsqueeze(0)
                processed_features = processed_features.to(self.device)
                
                # Generate raw gestures
                raw_gestures = self.model(processed_features)
                
                # Apply post-processing
                gestures = self.postprocess_gestures(raw_gestures, params)
                
            return gestures.cpu()
            
        except Exception as e:
            print(f"Error generating gestures: {e}")
            # Return a default gesture (neutral pose)
            return torch.zeros(1, 3, 240)  # Example: 3D positions for 240 frames

def process_gesture_parameters(
    use_gestures: bool,
    gesture_intensity: float,
    gesture_speed: float,
    gesture_style: str
) -> Dict[str, Any]:
    """Process and validate gesture generation parameters"""
    return {
        'use_gestures': use_gestures,
        'intensity': max(0.0, min(1.0, gesture_intensity)),
        'speed': max(0.1, min(2.0, gesture_speed)),
        'style': gesture_style
    }

def process_video(
    source_video,
    driven_audio,
    preprocess_type,
    is_still_mode,
    enhancer,
    batch_size,
    size_of_image,
    pose_style,
    exp_scale,
    use_gestures,
    gesture_intensity,
    gesture_speed,
    gesture_style,
    full_body
):
    """
    Process a video file by extracting frames, processing each frame, and combining them back.
    """
    try:
        print("🎥 Starting video processing...")
        
        # Create a unique output directory
        import time
        timestamp = int(time.time())
        output_dir = os.path.join("results", f"video_output_{timestamp}")
        frames_dir = os.path.join(output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # Extract frames from video
        print("📹 Extracting frames from video...")
        cap = cv2.VideoCapture(source_video)
        if not cap.isOpened():
            raise ValueError(f"❌ Could not open video file: {source_video}")
            
        frame_count = 0
        frame_paths = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_path = os.path.join(frames_dir, f"frame_{frame_count:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            frame_count += 1
            
            # Limit to first 300 frames (about 10 seconds at 30fps)
            if frame_count >= 300:
                break
                
        cap.release()
        
        if frame_count == 0:
            raise ValueError("❌ No frames could be extracted from the video")
        
        print(f"✅ Extracted {frame_count} frames from video")
        
        # Process each frame
        processed_frames = []
        for i, frame_path in enumerate(frame_paths):
            print(f"🔄 Processing frame {i+1}/{len(frame_paths)}")
            
            # Process the frame using the full pipeline
            output_frame = process_single_frame(
                frame_path=frame_path,
                output_dir=frames_dir,
                index=i,
                total_frames=len(frame_paths),
                source_image=frame_path,
                driven_audio=driven_audio,
                preprocess_type=preprocess_type,
                is_still_mode=is_still_mode,
                enhancer=enhancer,
                batch_size=batch_size,
                size_of_image=size_of_image,
                pose_style=pose_style,
                exp_scale=exp_scale,
                use_gestures=use_gestures,
                gesture_intensity=gesture_intensity,
                gesture_speed=gesture_speed,
                gesture_style=gesture_style,
                full_body=full_body
            )
            
            if output_frame and os.path.exists(output_frame):
                processed_frames.append(output_frame)
        
        if not processed_frames:
            raise ValueError("❌ No frames were processed successfully")
        
        # Create output video path
        output_video_path = os.path.join("results", f"output_{timestamp}.mp4")
        
        # Combine processed frames into video
        print("🎬 Combining frames into final video...")
        create_video_from_frames(processed_frames, output_video_path, fps=min(fps, 30))
        
        print(f"✅ Video processing complete: {output_video_path}")
        return output_video_path
        
    except Exception as e:
        error_msg = f"❌ Error in video processing: {str(e)}"
        print(error_msg)
        raise ValueError(error_msg) from e


def sadtalker_demo(checkpoint_path='checkpoints', config_path='src/config', warpfn=None):
    # Initialize SadTalker and GestureGenerator
    sad_talker = SadTalker(checkpoint_path, config_path, lazy_load=True)
    gesture_generator = GestureGenerator()
    
    # Make results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    with gr.Blocks(analytics_enabled=False, title="PaksaApp - Advanced Talking Head Animation") as sadtalker_interface:
        gr.Markdown("<div align='center'> <h1> PaksaApp: Advanced Talking Head Animation </h1> \
                    <p> Create realistic talking head videos with enhanced features and full body movement. </p> </div>")
        gr.Markdown("<div align='center'> <h2> SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation </span> </h2> \
                    <a style='font-size:18px;color: #efefef' href='https://arxiv.org/abs/2211.12194'>Arxiv</a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
                    <a style='font-size:18px;color: #efefef' href='https://sadtalker.github.io'>Homepage</a>  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
                     <a style='font-size:18px;color: #efefef' href='https://github.com/Winfredy/SadTalker'> Github </div>")
        
        with gr.Box():
            with gr.Row():
                with gr.Column(variant='panel'):  
                    with gr.Row():
                        gr.Markdown("""
                        # SadTalker v1.1
                        ## 1. Upload an image or video, and generate a talking head video with audio input.
                        """)
                    
                    # Upload section with both image and video options
                    with gr.Row():
                        source_image = gr.Image(
                            label="Upload Image or Video", 
                            source="upload", 
                            type="filepath", 
                            elem_id="img2img_image"
                        )
                        source_video = gr.Video(
                            label="Or Upload Video",
                            source="upload",
                            format="mp4",
                            elem_id="video_upload",
                            visible=False  # Initially hidden
                        )
                    
                    # Toggle between image and video
                    def toggle_inputs(source_image, source_video):
                        if source_image is not None:
                            return gr.update(visible=False), gr.update(visible=True)
                        elif source_video is not None:
                            return gr.update(visible=True), gr.update(visible=False)
                        return gr.update(visible=True), gr.update(visible=False)
                    
                    # Connect the toggle function to both inputs
                    source_image.change(
                        fn=toggle_inputs,
                        inputs=[source_image, source_video],
                        outputs=[source_image, source_video]
                    )
                    source_video.change(
                        fn=toggle_inputs,
                        inputs=[source_image, source_video],
                        outputs=[source_image, source_video]
                    )
                    
                    # Audio input section
                    with gr.Row():
                        with gr.Column(variant='panel'):
                            input_audio = gr.Audio(
                                sources=("upload", "microphone"), 
                                type="filepath", 
                                label="Input audio"
                            )
                            input_text = gr.Textbox(
                                label="Input Text (or put the script of input_audio here)", 
                                lines=5, 
                                placeholder="Please input the text"
                            )
                            with gr.Row():
                                tts = gr.Dropdown(
                                    ["Edge (Windows)", "gTTS"], 
                                    value="Edge (Windows)", 
                                    label="TTS Method"
                                )
                                tts_btn = gr.Button("Generate Audio", variant="primary")
                                tts_btn.click(
                                    fn=text_to_speech, 
                                    inputs=[input_text, tts], 
                                    outputs=[input_audio]
                                )

                    with gr.Tabs(elem_id="checkbox_instead_of_tabs"):
                        with gr.Row():
                            with gr.Column():
                                with gr.Row():
                                    # Full-body animation option
                                    full_body = gr.Checkbox(label="Full-Body Animation", value=False, 
                                                         info="Enable full-body animation (overrides face animation)")
                                    pose_style = gr.Slider(minimum=0, maximum=46, value=0, step=1, label="Pose style", 
                                                         info="The selected (real image, vid) pose will be performed by the generated video.")
                                    size_of_image = gr.Radio([256, 512], value=256, label='Face model resolution', 
                                                          info="Use 256/512 model? 256 is less memory and faster, while 512 has better quality.")
                                    preprocess_type = gr.Radio(['crop', 'resize', 'full'], value='full', label='Preprocess', 
                                                           info="How to handle input image?")
                                    is_still_mode = gr.Checkbox(label="Still Mode (fewer head motion, works with preprocess `full`)")
                                    enhancer = gr.Checkbox(label="GFPGAN as Face enhancer")
                                    batch_size = gr.Slider(label="Batch size in generation", step=1, maximum=10, value=2, interactive=True)
                                    exp_scale= gr.Slider(minimum=0.1, maximum=10, value=1, step=0.1, label="Expression scale")
                                    
                                    # Add gesture controls
                                    with gr.Column(visible=True) as gesture_controls:
                                        use_gestures = gr.Checkbox(label="Enable Gesture Generation", value=False)
                                        with gr.Row(visible=False) as gesture_params:
                                            gesture_intensity = gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.1, 
                                                                      label="Gesture Intensity")
                                            gesture_speed = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, 
                                                                  label="Gesture Speed")
                                            gesture_style = gr.Dropdown(choices=["Neutral", "Expressive", "Subtle"], 
                                                                    value="Neutral", label="Gesture Style")
                                        
                                        use_gestures.change(
                                            fn=lambda x: (gr.update(visible=x), gr.update(visible=not x)),
                                            inputs=[use_gestures],
                                            outputs=[gesture_params, gr.Textbox(visible=False)]
                                        )
                                    gesture_style = gr.Dropdown(
                                        ["Casual", "Formal", "Expressive", "Subtle"],
                                        value="Casual",
                                        label="Gesture Style"
                                                            )
                                    
                with gr.Tabs(elem_id="sadtalker_genearted"):
                        gen_video = gr.Video(label="Generated video", format="mp4").style(width=256)

        def process_single_frame(frame_path, output_dir, index, total_frames, **kwargs):
            """Process a single frame with the SadTalker model"""
            # Create a temporary directory for this frame's output
            frame_output_dir = os.path.join(output_dir, f"frame_{index:04d}")
            os.makedirs(frame_output_dir, exist_ok=True)
            
            # Update the source image path for this frame
            kwargs['source_image'] = frame_path
            
            # Generate the talking head video for this frame
            output_video = sad_talker_interface(**kwargs)
            
            # Extract the first frame from the output video
            cap = cv2.VideoCapture(output_video)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                output_frame_path = os.path.join(output_dir, f"output_frame_{index:04d}.jpg")
                cv2.imwrite(output_frame_path, frame)
                return output_frame_path
            return None

        def generate_video_with_gestures(
            source_image=None, 
            source_video=None,
            driven_audio=None, 
            preprocess_type='crop',
            is_still_mode=False,
            enhancer=False,
            batch_size=2,                            
            size_of_image=256,                         
            pose_style=0, 
            exp_scale=1.0,
            use_gestures=False,
            gesture_intensity=1.0,
            gesture_speed=1.0,
            gesture_style="Casual",
            full_body=False):
            try:
                # Create output directory
                output_dir = os.path.join("results", str(uuid.uuid4()))
                os.makedirs(output_dir, exist_ok=True)
                
                # Process gesture parameters
                gesture_params = process_gesture_parameters(
                    use_gestures=use_gestures,
                    gesture_intensity=gesture_intensity,
                    gesture_speed=gesture_speed,
                    gesture_style=gesture_style
                )
                
                # Prepare common kwargs for both image and video processing
                common_kwargs = {
                    'driven_audio': driven_audio,
                    'preprocess_type': preprocess_type,
                    'is_still_mode': is_still_mode,
                    'enhancer': enhancer,
                    'batch_size': batch_size,
                    'size_of_image': size_of_image,
                    'pose_style': pose_style,
                    'exp_scale': exp_scale,
                    'use_gestures': use_gestures,
                    'gesture_intensity': gesture_intensity,
                    'gesture_speed': gesture_speed,
                    'gesture_style': gesture_style,
                    'full_body': full_body
                }
                
                # Check if we're processing a video or an image
                if source_video is not None and os.path.exists(source_video):
                    # Process video
                    print(f"Processing video: {source_video}")
                    
                    # Extract frames from video
                    frames_dir = os.path.join(output_dir, "frames")
                    os.makedirs(frames_dir, exist_ok=True)
                    
                    # Extract frames from the video
                    cap = cv2.VideoCapture(source_video)
                    frame_count = 0
                    frame_paths = []
                    
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                            
                        frame_path = os.path.join(frames_dir, f"frame_{frame_count:05d}.jpg")
                        cv2.imwrite(frame_path, frame)
                        frame_paths.append(frame_path)
                        frame_count += 1
                        
                        # Limit the number of frames for testing
                        if frame_count >= 300:  # 10 seconds at 30fps
                            break
                            
                    cap.release()
                    
                    # Process each frame
                    processed_frames = []
                    for i, frame_path in enumerate(frame_paths):
                        print(f"Processing frame {i+1}/{len(frame_paths)}")
                        output_frame = process_single_frame(
                            frame_path=frame_path,
                            output_dir=output_dir,
                            index=i,
                            total_frames=len(frame_paths),
                            **common_kwargs
                        )
                        if output_frame:
                            processed_frames.append(output_frame)
                    
                    # Combine processed frames into a video
                    if processed_frames:
                        output_video_path = os.path.join(output_dir, "output_video.mp4")
                        create_video_from_frames(processed_frames, output_video_path, fps=25)
                        return output_video_path
                    else:
                        raise ValueError("No frames were processed successfully")
                
                # If not a video, process as a single image
                elif source_image is not None and os.path.exists(source_image):
                    print(f"Processing image: {source_image}")
                    return {
                        'source_image': source_image,
                        'source_video': source_video,
                        'input_audio': input_audio,
                        'preprocess_type': preprocess_type,
                        'is_still_mode': is_still_mode,
                        'enhancer': enhancer,
                        'batch_size': batch_size,
                        'size_of_image': size_of_image,
                        'pose_style': pose_style,
                        'exp_scale': exp_scale,
                        'use_gestures': use_gestures,
                        'gesture_intensity': gesture_intensity,
                        'gesture_speed': gesture_speed,
                        'gesture_style': gesture_style,
                        'full_body': full_body,
                        'submit': submit,
                        'gen_video': gen_video
                    }(source_image=source_image, **common_kwargs)
                else:
                    raise ValueError("No valid image or video file provided")
                
                # Add smoothness to params if advanced options are enabled
                if use_advanced_gestures:
                    gesture_params['smoothness'] = gesture_smoothness
                
                # Generate video using SadTalker
                test_kwargs = {
                    'source_image': source_image,
                    'driven_audio': driven_audio,
                    'preprocess': preprocess_type,
                    'still_mode': is_still_mode,  # Using the correct parameter name for SadTalker.test()
                    'use_enhancer': enhancer,
                    'batch_size': batch_size,
                    'size': size_of_image,
                    'pose_style': pose_style
                }
                video_output = sad_talker.test(**test_kwargs)
                
                # If gestures are enabled, process them
                if use_gestures and driven_audio is not None:
                    try:
                        # Generate gestures from audio
                        gestures = gesture_generator.generate_gestures(
                            audio_path=driven_audio,
                            params=gesture_params
                        )
                        
                        # TODO: Integrate gestures with the video output
                        # This would involve:
                        # 1. Extracting the face/head movements from the video
                        # 2. Combining them with the generated body gestures
                        # 3. Rendering the final video with both face and body movements
                        
                        # For now, we'll just print the gesture stats
                        print(f"Generated gestures with shape: {gestures.shape}")
                        print(f"Gesture parameters: {gesture_params}")
                        
                    except Exception as e:
                        print(f"Error during gesture generation: {e}")
                        # Continue with just the face animation if gesture generation fails
                
                return video_output
                
            except Exception as e:
                error_msg = f"Error in video generation: {str(e)}"
                print(error_msg)
                return None  # Return None to indicate failure
        
        def sad_talker_interface(
            source_image=None, 
            source_video=None,
            driven_audio=None, 
            preprocess_type='crop',
            is_still_mode=False,
            enhancer=False,
            batch_size=2,                           
            size_of_image=256,                      
            pose_style=0, 
            exp_scale=1.0,
            use_gestures=False,
            gesture_intensity=1.0,
            gesture_speed=1.0,
            gesture_style="Casual",
            full_body=False
        ):
            # Create results directory if it doesn't exist
            os.makedirs("results", exist_ok=True)
            
            # Check if we have a valid input
            is_video = source_video is not None and os.path.exists(str(source_video))
            is_image = source_image is not None and os.path.exists(str(source_image))
            
            if not (is_video or is_image):
                raise ValueError("❌ Please provide either a valid image or video file")
            
            try:
                if is_video:
                    print(f"🎥 Processing video: {source_video}")
                    
                    # Create directory for frames
                    import time
                    timestamp = int(time.time())
                    frames_dir = os.path.join("results", f"frames_{timestamp}")
                    os.makedirs(frames_dir, exist_ok=True)
                    
                    # Extract frames from video
                    cap = cv2.VideoCapture(source_video)
                    if not cap.isOpened():
                        raise ValueError(f"❌ Could not open video file: {source_video}")
                        
                    frame_count = 0
                    frame_paths = []
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                            
                        frame_path = os.path.join(frames_dir, f"frame_{frame_count:05d}.jpg")
                        cv2.imwrite(frame_path, frame)
                        frame_paths.append(frame_path)
                        frame_count += 1
                        
                        # Limit to first 300 frames (about 10 seconds at 30fps)
                        if frame_count >= 300:
                            break
                            
                    cap.release()
                    
                    if frame_count == 0:
                        raise ValueError("❌ No frames could be extracted from the video")
                    
                    print(f"✅ Extracted {frame_count} frames from video")
                    
                    # Process each frame
                    processed_frames = []
                    for i, frame_path in enumerate(frame_paths):
                        print(f"🔄 Processing frame {i+1}/{len(frame_paths)}")
                        
                        # Process the frame using the full pipeline
                        output_frame = process_single_frame(
                            frame_path=frame_path,
                            output_dir=frames_dir,
                            index=i,
                            total_frames=len(frame_paths),
                            source_image=frame_path,
                            driven_audio=driven_audio,
                            preprocess_type=preprocess_type,
                            is_still_mode=is_still_mode,
                            enhancer=enhancer,
                            batch_size=batch_size,
                            size_of_image=size_of_image,
                            pose_style=pose_style,
                            exp_scale=exp_scale,
                            use_gestures=use_gestures,
                            gesture_intensity=gesture_intensity,
                            gesture_speed=gesture_speed,
                            gesture_style=gesture_style,
                            full_body=full_body
                        )
                        
                        if output_frame and os.path.exists(output_frame):
                            processed_frames.append(output_frame)
                    
                    if not processed_frames:
                        raise ValueError("❌ No frames were processed successfully")
                    
                    # Create output video path with timestamp
                    output_video_path = os.path.join("results", f"output_{timestamp}.mp4")
                    
                    # Combine processed frames into video
                    print("🎬 Combining frames into final video...")
                    create_video_from_frames(processed_frames, output_video_path, fps=min(fps, 30))
                    
                    # Clean up temporary frame files
                    try:
                        import shutil
                        shutil.rmtree(frames_dir)
                    except Exception as e:
                        print(f"⚠️ Could not clean up temporary files: {e}")
                    
                    print(f"✅ Video processing complete: {output_video_path}")
                    return output_video_path
                
                else:  # Process single image
                    print(f"🖼️ Processing image: {source_image}")
                    return sad_talker.test(
                        source_image=source_image,
                        driven_audio=driven_audio,
                        preprocess=preprocess_type,
                        still_mode=is_still_mode,
                        use_enhancer=enhancer,
                        batch_size=batch_size,
                        size=size_of_image,
                        pose_style=pose_style,
                        exp_scale=exp_scale
                    )
                    
            except Exception as e:
                error_msg = f"❌ Error in video processing: {str(e)}"
                print(error_msg)
                raise ValueError(error_msg) from e
        
        def process_inputs(
            source_image=None,
            source_video=None,
            driven_audio=None,
            preprocess_type='crop',
            is_still_mode=False,
            enhancer=False,
            batch_size=2,
            size_of_image=256,
            pose_style=0,
            exp_scale=1.0,
            use_gestures=False,
            gesture_intensity=1.0,
            gesture_speed=1.0,
            gesture_style="Casual",
            full_body=False
        ):
            try:
                # Determine which input is being used
                if source_image is not None and os.path.exists(str(source_image)):
                    # Process as image
                    print("🖼️ Processing image input...")
                    return sad_talker.test(
                        source_image=source_image,
                        driven_audio=driven_audio,
                        preprocess=preprocess_type,
                        still_mode=is_still_mode,
                        use_enhancer=enhancer,
                        batch_size=batch_size,
                        size=size_of_image,
                        pose_style=pose_style,
                        exp_scale=exp_scale
                    )
                elif source_video is not None and os.path.exists(str(source_video)):
                    # Process as video
                    print("🎥 Processing video input...")
                    return process_video(
                        source_video=source_video,
                        driven_audio=driven_audio,
                        preprocess_type=preprocess_type,
                        is_still_mode=is_still_mode,
                        enhancer=enhancer,
                        batch_size=batch_size,
                        size_of_image=size_of_image,
                        pose_style=pose_style,
                        exp_scale=exp_scale,
                        use_gestures=use_gestures,
                        gesture_intensity=gesture_intensity,
                        gesture_speed=gesture_speed,
                        gesture_style=gesture_style,
                        full_body=full_body
                    )
                else:
                    raise ValueError("❌ Please provide either an image or a video file")
                    
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                print(error_msg)
                raise ValueError(error_msg) from e
        
        def get_input_source(source_image, source_video):
            """Determine which input source to use (image or video)"""
            if source_image is not None and os.path.exists(str(source_image)):
                return source_image, None
            elif source_video is not None and os.path.exists(str(source_video)):
                return None, source_video
            return None, None
            
        # Connect the submit button to the processing function
        submit.click(
            fn=lambda *args: process_inputs(*args[:-1]),
            inputs=[
                source_image,
                source_video,
                input_audio,
                preprocess_type,
                is_still_mode,
                enhancer,
                batch_size,                           
                size_of_image,                         
                pose_style, 
                exp_scale,
                use_gestures,
                gesture_intensity,
                gesture_speed,
                gesture_style,
                full_body
            ],
            outputs=[gen_video]
        )
        
        # Add some JavaScript to handle tab switching
        source_tabs.select(
            fn=None,
            inputs=None,
            outputs=None,
            _js="""
            function() {
                // Make sure only one source type is active at a time
                const tabs = document.querySelectorAll('.tab-nav button');
                tabs.forEach(tab => {
                    tab.addEventListener('click', function() {
                        // Clear the other input when switching tabs
                        const tabText = this.textContent.trim();
                        if (tabText === 'Image') {
                            document.getElementById('video_upload').value = '';
                        } else if (tabText === 'Video') {
                            document.getElementById('img2img_image').value = '';
                        }

# Connect the submit button to the generation function
submit.click(
    fn=generate_video_with_gestures,
    inputs=[
        source_image,
        source_video,
        driven_audio,
        preprocess_type,
        is_still_mode,
        enhancer,
        batch_size,                            
        size_of_image,                         
        pose_style, 
        exp_scale,
        use_gestures,
        gesture_intensity,
        gesture_speed,
        gesture_style,
        full_body
    ],
    outputs=[gen_video]
)

# Add some JavaScript to handle tab switching
source_tabs.select(
    fn=None,
    inputs=None,
    outputs=None,
    _js="""
    function() {
        // Make sure only one source type is active at a time
        const tabs = document.querySelectorAll('.tab-nav button');
        tabs.forEach(tab => {
            tab.addEventListener('click', function() {
                // Clear the other input when switching tabs
                const tabText = this.textContent.trim();
                if (tabText === 'Image') {
                    document.getElementById('video_upload').value = '';
                } else if (tabText === 'Video') {
                    document.getElementById('img2img_image').value = '';
                }
            });
        });
        return [];
    }
    """
)

def process_single_frame(
    frame_path,
    output_dir,
    index,
    total_frames,
    source_image,
    driven_audio,
    preprocess_type,
    is_still_mode,
    enhancer,
    batch_size,
    size_of_image,
    pose_style,
    exp_scale,
    use_gestures,
    gesture_intensity,
    gesture_speed,
    gesture_style,
    full_body
):
    """
    Process a single frame through the SadTalker pipeline.
    
    Args:
        frame_path: Path to the input frame
        output_dir: Directory to save processed frames
        index: Index of the current frame
        total_frames: Total number of frames being processed
        ... other parameters from the main interface ...
        
    Returns:
        Path to the processed frame
    """
    try:
        print(f"  Processing frame {index+1}/{total_frames}")
        
        # Create a temporary directory for this frame's output
        frame_output_dir = os.path.join(output_dir, f"frame_{index:05d}")
        os.makedirs(frame_output_dir, exist_ok=True)
        
        # Process the frame using SadTalker
        output_path = sad_talker.test(
            source_image=frame_path,
            driven_audio=driven_audio,
            preprocess=preprocess_type,
            still_mode=is_still_mode,
            use_enhancer=enhancer,
            batch_size=batch_size,
            size=size_of_image,
            pose_style=pose_style,
            exp_scale=exp_scale,
            result_dir=frame_output_dir
        )
        
        if output_path and os.path.exists(output_path):
            # Find the generated video file
            video_files = [f for f in os.listdir(output_path) 
                         if f.endswith('.mp4') or f.endswith('.avi')]
            
            if video_files:
                # Extract first frame from the video
                video_path = os.path.join(output_path, video_files[0])
                
                # Read the first frame of the output video
                cap = cv2.VideoCapture(video_path)
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    # Save the processed frame
                    output_frame_path = os.path.join(output_dir, f"processed_frame_{index:05d}.jpg")
                    cv2.imwrite(output_frame_path, frame)
                    
                    # Clean up temporary files
                    try:
                        shutil.rmtree(frame_output_dir)
                    except Exception as e:
                        print(f"  ⚠️ Could not clean up temporary files: {e}")
                    
                    return output_frame_path
        
        return None
        
    except Exception as e:
        print(f"  ❌ Error processing frame {index+1}: {str(e)}")
        return None


def create_video_from_frames(frame_paths, output_path, fps=25, codec='mp4v'):
    """
    Create a video from a list of image frames.
    
    Args:
        frame_paths: List of paths to image frames
        output_path: Path to save the output video
        fps: Frames per second for the output video
        codec: Video codec to use (default: 'mp4v' for MP4)
        
    Returns:
        Path to the created video file
    """
    if not frame_paths:
        raise ValueError("No frames provided to create video")
    
    # Filter out None values
    frame_paths = [f for f in frame_paths if f is not None and os.path.exists(f)]
    
    if not frame_paths:
        raise ValueError("No valid frames found to create video")
    
    # Read the first frame to get dimensions
    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        raise ValueError(f"Could not read first frame: {frame_paths[0]}")
        
    height, width, _ = first_frame.shape
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames to video
    for frame_path in tqdm(frame_paths, desc="Creating video"):
        frame = cv2.imread(frame_path)
        if frame is not None:
            # Resize frame if dimensions don't match
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height))
            out.write(frame)
    
    # Release everything when done
    out.release()
    
    # Verify the video was created
    if not os.path.exists(output_path):
        raise RuntimeError(f"Failed to create video at {output_path}")
        
    print(f"Video created successfully at: {output_path}")
    return output_path

if __name__ == "__main__":
    # Create the interface
    interface = sadtalker_demo()
    
    # Launch the app
    interface.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        debug=True
    )