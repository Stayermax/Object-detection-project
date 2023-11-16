from typing import Union
import cv2
import numpy as np
from colorir import sRGB


class ObjectDetector:
    font = cv2.FONT_HERSHEY_DUPLEX
    colors = {
        # BGR colors
        "blue": (255, 0, 0),
        "green": (0, 255, 0),
        "red": (0, 0, 255),
        "yellow": (0, 0, 255),
        "white": (255, 255, 255),
        "black": (0, 0, 0)
    }

    def __init__(self, input_: Union[int, str], image_compression=4):
        """
        Input source can be port of camera or path to video or image
        :param input_:
        """
        self.input_source = input_
        self.cap = cv2.VideoCapture(self.input_source)
        ret, image = self.cap.read()
        if not ret:
            raise Exception(f'Input source "{self.input_source}" is not accessible')
        self.height = image.shape[0]
        self.width = image.shape[1]
        self.image_compression = image_compression

    def stream(self, image_edit_function = lambda x, *args, **kwargs: x, *args, **kwargs):
        while True:
            ret, image = self.cap.read()
            cv2.imshow("Test", image_edit_function(image, *args, **kwargs))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

    def show_image(self, image_edit_function = lambda x, *args, **kwargs: x, *args, **kwargs):
        while True:
            image = cv2.imread(self.input_source)
            cv2.imshow("Test", image_edit_function(image, *args, **kwargs))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    @staticmethod
    def color_by_name(color: str) -> tuple:
        return ObjectDetector.colors.get(color, (255, 255, 255))


    @staticmethod
    def draw_rectangle(image: np.ndarray, object_cords: list, bgr_color: tuple):
        (top, right, bottom, left) = object_cords
        image = cv2.rectangle(image, (left, top), (right, bottom), bgr_color, 5)
        return cv2.putText(image, "Object", (left + 6, bottom - 6), ObjectDetector.font, 1.0, bgr_color, 2)


    @staticmethod
    # calculate distance between two contours
    def calculate_contour_distance(contour1, contour2):
        x1, y1, w1, h1 = cv2.boundingRect(contour1)
        c_x1 = x1 + w1 / 2
        c_y1 = y1 + h1 / 2

        x2, y2, w2, h2 = cv2.boundingRect(contour2)
        c_x2 = x2 + w2 / 2
        c_y2 = y2 + h2 / 2

        return max(abs(c_x1 - c_x2) - (w1 + w2) / 2, abs(c_y1 - c_y2) - (h1 + h2) / 2)


    @staticmethod
    def take_biggest_contours(contours, max_number=20):
        sorted_contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        return sorted_contours[:max_number]

    @staticmethod
    def agglomerative_cluster(contours, threshold_distance=40.0):
        current_contours = contours
        while len(current_contours) > 1:
            min_distance = None
            min_coordinate = None

            for x in range(len(current_contours) - 1):
                for y in range(x + 1, len(current_contours)):
                    distance = ObjectDetector.calculate_contour_distance(current_contours[x], current_contours[y])
                    if min_distance is None:
                        min_distance = distance
                        min_coordinate = (x, y)
                    elif distance < min_distance:
                        min_distance = distance
                        min_coordinate = (x, y)

            if min_distance < threshold_distance:
                # merge closest two contours
                index1, index2 = min_coordinate
                current_contours[index1] = np.concatenate((current_contours[index1], current_contours[index2]),
                                                             axis=0)
                del current_contours[index2]
            else:
                break
        return current_contours

    def get_contours_from_updated_frame(self, updated_frame):
        contours, hierarchy = cv2.findContours(updated_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        print("contours after findContours: %s" % len(contours))

        contours = self.take_biggest_contours(contours)
        print("contours after take_biggest_contours: %s" % len(contours))

        contours = self.agglomerative_cluster(contours)
        print("contours after agglomerative_cluster: %s" % len(contours))

        objects = []
        for c in contours:
            rect = cv2.boundingRect(c)
            x, y, w, h = rect

            # ignore small contours
            if w < 20 and h < 20:
                print("dropping rect due to small size", rect)
                continue
            objects.append([x, y, w, h])
        return objects

    def prepare_frame_for_object_search(self, frame):
        updated_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # updated_image = cv2.equalizeHist(updated_image)
        # updated_image = cv2.GaussianBlur(updated_image, (9, 9), 0)
        updated_image = cv2.Canny(updated_image, 90, 180)
        # ret, updated_image = cv2.threshold(updated_image, 127, 255, 0)
        return updated_image

    def highlight_object_box(self, frame):
        updated_image = self.prepare_frame_for_object_search(frame)

        objects_crds = self.get_contours_from_updated_frame(updated_image)
        for x, y, w, h in objects_crds:
            frame = cv2.rectangle(frame, (x, y), (x+w, y+h), self.colors['red'], 5)
        return frame

    def highlight_contours(self, frame):
        search_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, search_frame = cv2.threshold(search_frame, 127, 255, 0)
        contours, hierarchy = cv2.findContours(search_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
        return frame

    def highlight_colored_objects(self, frame, color_to_detect: list, hue_koef: float = 1):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        b_, g_, r_ = color_to_detect
        rgb = sRGB(r_, g_, b_, max_rgb=255)
        h_, s_, v_ = rgb.hsv(max_h=179, max_sva=255)
        # print(f"HSV: {[h_, s_, v_]}")

        lower_limit = np.array([max(0, int(h_ - hue_koef * 10)), 100, 100])
        upper_limit = np.array([min(179, int(h_ + hue_koef * 10)),255, 255])

        mask = cv2.inRange(hsv_frame, lower_limit, upper_limit)
        bbox = cv2.boundingRect(mask)

        # if we get a bounding box, use it to draw a rectangle on the image
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), color_to_detect, 2)
            center = (int((2*x+w)/2), int((2*y+h)/2))
            # print(f"Object center is on cords: {center}")
            frame = cv2.circle(frame, center, radius=4, color=(255-b_, 255-g_, 255-r_), thickness=-1)
        else:
            print("Object not detected")

        return frame