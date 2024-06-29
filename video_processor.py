from streamlit_webrtc import VideoProcessorBase

class VideoProcessor(VideoProcessorBase):
    def __init__(self, model, label_map, byte_tracker, box_annotator):
        self.model = model
        self.label_map = label_map
        self.byte_tracker = byte_tracker
        self.box_annotator = box_annotator
        self.frame = None

    def recv(self, frame):
        self.frame = frame.to_ndarray(format="bgr24")
        return self.frame
