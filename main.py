from src.object_detector import ObjectDetector
from typing import Union

# Show options:
BOX = "box"
CONTOURS = "contours"
NOTHING = "nothing"

# Modes
VIDEO_MODE = "VIDEO"
IMAGE_MODE = "IMAGE"


def vod_show_video(object_detector: ObjectDetector, what_to_show: str = BOX):
    if what_to_show == BOX:
        object_detector.stream(object_detector.highlight_object_box)
    elif what_to_show == CONTOURS:
        object_detector.stream(object_detector.highlight_contours)
    elif what_to_show == NOTHING:
        object_detector.stream()


def vod_show_image(object_detector: ObjectDetector, what_to_show: str = BOX):
    if what_to_show == BOX:
        object_detector.show_image(object_detector.highlight_object_box)
    elif what_to_show == CONTOURS:
        object_detector.show_image(object_detector.highlight_contours)
    elif what_to_show == NOTHING:
        object_detector.show_image()


def detect_objects(mode: str, input_: Union[int, str], what_to_show: str = BOX):
    object_detector = ObjectDetector(input_)
    if mode == VIDEO_MODE:
        vod_show_video(object_detector, what_to_show)
    elif mode == IMAGE_MODE:
        vod_show_image(object_detector, what_to_show)


def detect_colored_objects(mode: str, input_: Union[int, str], color_to_detect: Union[str, tuple], hue_koef: float):
    if isinstance(color_to_detect, str):
        if color_to_detect in ObjectDetector.colors:
            color_to_detect = ObjectDetector.colors[color_to_detect]
        else:
            raise ValueError(f"Current colors: {list(ObjectDetector.colors.keys())}")

    object_detector = ObjectDetector(input_)
    if mode == VIDEO_MODE:
        object_detector.stream(object_detector.highlight_colored_objects,
                               color_to_detect=color_to_detect,
                               hue_koef=hue_koef)
    elif mode == IMAGE_MODE:
        object_detector.show_image(object_detector.highlight_colored_objects,
                                   color_to_detect=color_to_detect,
                                   hue_koef=hue_koef)

if __name__ == '__main__':

    # detect_objects(mode=VIDEO_MODE, input_=0, what_to_show=BOX)
    # detect_objects(mode=VIDEO_MODE, input_="data/ball.mp4", what_to_show=CONTOURS)
    # detect_objects(mode=IMAGE_MODE, input_="data/box.jpeg", what_to_show=BOX)

    # detect_colored_objects(mode=VIDEO_MODE, input_="data/ball.mp4", color_to_detect='green', hue_koef=2.5)
    # detect_colored_objects(mode=VIDEO_MODE, input_=0, color_to_detect='blue', hue_koef=2)
    detect_colored_objects(mode=IMAGE_MODE, input_="data/box.jpeg",  color_to_detect="red", hue_koef=1)


