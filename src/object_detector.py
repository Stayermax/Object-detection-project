import cv2
import numpy as np

class VideoObjectDetector:
    font = cv2.FONT_HERSHEY_DUPLEX
    colors = {
        # BGR colors
        "red": (0,0,255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
        "white": (255, 255, 255)
    }

    def __init__(self, input_source=0, image_compression=4):
        """
        Input source can be port of camera or path to video
        :param input_source:
        """
        self.cap = cv2.VideoCapture(input_source)
        ret, image = self.cap.read()
        if not ret:
            raise Exception(f'Input source "{input_source}" is not accessible')
        self.height = image.shape[0]
        self.width = image.shape[1]
        self.image_compression = image_compression

    def stream(self, image_edit_function = lambda x: x):
        while True:
            ret, image = self.cap.read()
            cv2.imshow("Test", image_edit_function(image))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

    @staticmethod
    def color_by_name(color: str) -> tuple:
        return VideoObjectDetector.colors.get(color, (255, 255, 255))


    @staticmethod
    def draw_rectangle(image: np.ndarray, object_cords: list, bgr_color: tuple):
        (top, right, bottom, left) = object_cords
        image = cv2.rectangle(image, (left, top), (right, bottom), bgr_color, 5)
        return cv2.putText(image, "Object", (left + 6, bottom - 6), VideoObjectDetector.font, 1.0, bgr_color, 2)

    @staticmethod
    def draw_contour(image: np.ndarray, cords_list: tuple, bgr_color: tuple):
        return cv2.polylines(img=image,
                             pts=np.int32([cords_list]),
                             isClosed=False,
                             color=bgr_color,
                             thickness=3)

    def detect_objects(self, image):
        rgb_small_frame = image[:, :, ::-1]
        face_locations = face_recognition.face_locations(
            cv2.resize(rgb_small_frame, (0, 0), fx=1/self.image_compression, fy=1/self.image_compression)
        )
        face_locations = tuple([el*self.image_compression for el in fl] for fl in face_locations)
        return face_locations

    def highlight_object(self, image):
        cords_list = self.detect_objects(image)
        for face_cords in cords_list:
            image = self.draw_rectangle(image, face_cords, self.colors['red'])
        return image
