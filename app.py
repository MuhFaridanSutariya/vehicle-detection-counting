import streamlit as st
from ultralytics import YOLO
from supervision.draw.color import ColorPalette
from supervision.tracker.byte_tracker.core import ByteTrack
from supervision.detection.annotate import BoxAnnotator
from supervision.detection.line_counter import LineZone, LineZoneAnnotator
from supervision.geometry.core import Point
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
from video_processor import VideoProcessor
from utils import process_image, process_video_realtime
import tempfile
import numpy as np
import cv2

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

def main():
    st.title('Vehicle Detection and Counting')

    model = YOLO("yolov8x.pt")
    model.fuse()
    label_map = model.model.names
    byte_tracker = ByteTrack()
    box_annotator = BoxAnnotator(color=ColorPalette.default(), thickness=4, text_thickness=4, text_scale=2)
    
    line_start = Point(50, 1500)
    line_end = Point(3840 - 50, 1500)
    line_counter = LineZone(start=line_start, end=line_end)
    line_annotator = LineZoneAnnotator(thickness=4, text_thickness=4, text_scale=2)

    source_option = st.selectbox("Select Source Data", ("Webcam", "Image file", "Video file"))

    if source_option == "Webcam":
        ctx = webrtc_streamer(
            key="example",
            video_processor_factory=lambda: VideoProcessor(model, label_map, byte_tracker, box_annotator),
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False}
        )
        if st.button('Take Screenshot and Process'):
            if ctx.video_processor and ctx.video_processor.frame is not None:
                image = ctx.video_processor.frame
                processed_image, object_counts = process_image(image, model, label_map, byte_tracker, box_annotator)
                st.image(processed_image, caption="Processed Image", use_column_width=True)
                st.write("Results:", object_counts)
            else:
                st.warning("No frame available yet.")
    elif source_option == "Image file":
        uploaded_image = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            if st.button('Process Image'):
                processed_image, object_counts = process_image(image, model, label_map, byte_tracker, box_annotator)
                st.image(processed_image, caption="Processed Image", use_column_width=True)
                st.write("Results:", object_counts)
    elif source_option == "Video file":
        uploaded_video = st.file_uploader("Upload a video file", type=["mp4"])
        demo_video = "vehicle-counting.mp4"
        
        if uploaded_video is not None:
            temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            with open(temp_video_path, 'wb') as f:
                f.write(uploaded_video.read())
            st.video(temp_video_path)
            if st.button('Process Uploaded Video'):
                process_video_realtime(temp_video_path, model, label_map, byte_tracker, box_annotator, line_counter, line_annotator)
        else:
            st.header("Demo Video Preview")
            st.video(demo_video)
            if st.button('Process Demo Video'):
                process_video_realtime(demo_video, model, label_map, byte_tracker, box_annotator, line_counter, line_annotator)

if __name__ == "__main__":
    main()
