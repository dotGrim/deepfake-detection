import cv2
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from facenet_pytorch import MTCNN


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=False, device=device)


def crop_video(video_path: str, seq_length: int = 16, face_size: int = 224, stride: int = 1, use_center_crop: bool = False, transform=None) -> torch.Tensor:

    # Given a video path & hyperparameters(seq_length, face_size, stride),
    #   crop around located face OR center of image (via use_center_crop)
    #   & return tensor

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {video_path}")
    
    face_frames = []
    frame_count = 0
    
    try:
        while len(face_frames) < seq_length:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % stride == 0:
                cropped_face = None

                # Try MTCNN face detection
                if not use_center_crop:
                    try:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        boxes, _ = mtcnn.detect(frame_rgb)

                        if boxes is not None and len(boxes) > 0:
                            x1, y1, x2, y2 = [int(b) for b in boxes[0]]
                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(frame.shape[1], x2)
                            y2 = min(frame.shape[0], y2)
                            face_crop = frame[y1:y2, x1:x2]
                            if face_crop.size > 0:
                                cropped_face = cv2.resize(face_crop, (face_size, face_size))
                    except Exception as e:
                        print(f"Warning: MTCNN failed ({type(e).__name__}): {e}")

                # Fallback: center crop
                if cropped_face is None:
                    h, w = frame.shape[:2]
                    center_y, center_x = h // 2, w // 2
                    crop_h, crop_w = min(h, w), min(h, w)
                    y1 = max(0, center_y - crop_h // 2)
                    x1 = max(0, center_x - crop_w // 2)
                    y2 = min(h, y1 + crop_h)
                    x2 = min(w, x1 + crop_w)
                    cropped_face = frame[y1:y2, x1:x2]
                    cropped_face = cv2.resize(cropped_face, (face_size, face_size))

                # Note: crop from BGR frame, convert to RGB here for transform
                cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                cropped_face = transform(cropped_face)
                face_frames.append(cropped_face)

    finally:
        cap.release()

    if len(face_frames) < seq_length:
        padding_needed = seq_length - len(face_frames)
        zero_tensor = torch.zeros(3, face_size, face_size)
        face_frames.extend([zero_tensor] * padding_needed)

    video_tensor = torch.stack(face_frames[:seq_length])
    return video_tensor