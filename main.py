from src.object_detector import VideoObjectDetector

# Show options:
BOX = "box"
NOTHING = "nothing"

def detect_objects_on_camera(what_to_show: str = BOX):
    # Detect objects on camera
    vfd = VideoObjectDetector(0)
    if what_to_show == BOX:
        vfd.stream(vfd.highlight_object)
    elif what_to_show == NOTHING:
        vfd.stream()

def detect_objects_on_video(path_to_video: str, what_to_show: str = BOX):
    # Detect objects on video
    vfd = VideoObjectDetector(path_to_video)
    if what_to_show == BOX:
        vfd.stream(vfd.highlight_object)
    elif what_to_show == NOTHING:
        vfd.stream()

if __name__ == '__main__':
    # detect_objects_on_camera()
    detect_objects_on_video("data/ball.mp4", what_to_show=NOTHING)

