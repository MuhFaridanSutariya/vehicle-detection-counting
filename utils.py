from supervision.detection.core import Detections
from supervision.utils.video import VideoInfo, get_video_frames_generator
import streamlit as st

def process_image(image, model, label_map, byte_tracker, box_annotator):
    results = model(image)[0]
    detections = Detections.from_ultralytics(results)
    detections = byte_tracker.update_with_detections(detections=detections)
    labels = [f"{label_map[class_id]} {confidence:0.2f} -track_id:{tracker_id}" for _, _, confidence, class_id, tracker_id in detections]
    annotated_image = box_annotator.annotate(scene=image, detections=detections, labels=labels)
    
    object_counts = {}
    for detection in detections:
        class_id = detection[3]
        class_name = label_map[class_id]
        if class_name in object_counts:
            object_counts[class_name] += 1
        else:
            object_counts[class_name] = 1

    return annotated_image, object_counts

def process_video_realtime(input_video_path, model, label_map, byte_tracker, box_annotator, line_counter, line_annotator):
    video_info = VideoInfo.from_video_path(input_video_path)
    generator = get_video_frames_generator(input_video_path)
    stframe = st.empty()

    for frame in generator:
        results = model(frame)[0]
        detections = Detections.from_ultralytics(results)
        detections = byte_tracker.update_with_detections(detections=detections)
        labels = [f"{label_map[class_id]} {confidence:0.2f} -track_id:{tracker_id}" for _, _, confidence, class_id, tracker_id in detections]
        line_counter.trigger(detections=detections)
        annotated_frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
        line_annotator.annotate(frame=annotated_frame, line_counter=line_counter)
        stframe.image(annotated_frame, channels="BGR")
