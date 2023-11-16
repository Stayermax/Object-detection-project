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

    def __init__(self, input_=0, image_compression=4):
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

    def stream(self, image_edit_function = lambda x: x):
        while True:
            ret, image = self.cap.read()
            cv2.imshow("Test", image_edit_function(image))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

    def show_image(self, image_edit_function = lambda x: x):
        while True:
            image = cv2.imread(self.input_source)
            cv2.imshow("Test", image_edit_function(image))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    @staticmethod
    def color_by_name(color: str) -> tuple:
        return VideoObjectDetector.colors.get(color, (255, 255, 255))


    @staticmethod
    def draw_rectangle(image: np.ndarray, object_cords: list, bgr_color: tuple):
        (top, right, bottom, left) = object_cords
        image = cv2.rectangle(image, (left, top), (right, bottom), bgr_color, 5)
        return cv2.putText(image, "Object", (left + 6, bottom - 6), VideoObjectDetector.font, 1.0, bgr_color, 2)
    #
    # @staticmethod
    # def find_object_cords_from_contour(contour: list) -> list:
    #     x, y, w, h = cv2.boundingRect(contour)
    #     return [x, y, x+h, y+w]
    #     # top, right, bottom, left = -1, 10**8, 10**8, -1
    #     # for point in contour:
    #     #     x = point[0][0]
    #     #     y = point[0][1]
    #     #     if top < x:
    #     #         top = x
    #     #     if right > x:
    #     #         right = x
    #     #     if bottom > x:
    #     #         bottom = x
    #     #     if left < x:
    #     #         left = x
    #     # return [top, right, bottom, left]

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
                    distance = VideoObjectDetector.calculate_contour_distance(current_contours[x], current_contours[y])
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

        # for object_crds in objects_crds:
        #     frame = self.draw_rectangle(frame, object_crds, self.colors['red'])

        # for face_cords in cords_list:
        #     image = self.draw_rectangle(image, face_cords, self.colors['red'])
        return frame

    def highlight_contours(self, frame):
        search_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, search_frame = cv2.threshold(search_frame, 127, 255, 0)
        contours, hierarchy = cv2.findContours(search_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
        return frame