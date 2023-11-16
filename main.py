from src.object_detector import VideoObjectDetector

# Show options:
BOX = "box"
CONTOURS = "contours"
NOTHING = "nothing"

def vod_stream(vod: VideoObjectDetector, what_to_show: str = BOX):
    if what_to_show == BOX:
        vod.stream(vod.highlight_object_box)
    elif what_to_show == CONTOURS:
        vod.stream(vod.highlight_contours)
    elif what_to_show == NOTHING:
        vod.stream()

def vod_show_image(vod: VideoObjectDetector, what_to_show: str = BOX):
    if what_to_show == BOX:
        vod.show_image(vod.highlight_object_box)
    elif what_to_show == CONTOURS:
        vod.show_image(vod.highlight_contours)
    elif what_to_show == NOTHING:
        vod.show_image()


def detect_objects_on_camera(what_to_show: str = BOX):
    # Detect objects on camera
    vod = VideoObjectDetector(0)
    vod_stream(vod, what_to_show)

def detect_objects_on_video(path_to_video: str, what_to_show: str = BOX):
    # Detect objects on video
    vod = VideoObjectDetector(path_to_video)
    vod_stream(vod, what_to_show)

def detect_objects_on_image(path_to_image: str, what_to_show: str = BOX):
    # Detect objects on video
    vod = VideoObjectDetector(path_to_image)
    vod_show_image(vod, what_to_show)


if __name__ == '__main__':
    # detect_objects_on_camera(what_to_show=BOX)
    detect_objects_on_video("data/ball.mp4", what_to_show=BOX)
    # detect_objects_on_image("data/box.jpeg", what_to_show=BOX)

