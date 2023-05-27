import numpy as np
import cv2 as cv
from typing import Tuple


class Calibration:
    """Obtains required parameters for color thresholding from sliders. Saves and loads parameters into/from a config file."""

    def __init__(self, cam_id: int) -> None:
        self._config_fn = f'Cam{cam_id}_config'

        self.upper_th = np.array([255, 255, 255])
        self.lower_th = np.array([0, 0, 0])
        self._loadconfig()

        self._window_name = f'Cam{cam_id} Calibration'
        cv.namedWindow(self._window_name)

        self._create_upper_th_trackbar('upper Hue', 0)
        self._create_upper_th_trackbar('upper Saturation', 1)
        self._create_upper_th_trackbar('upper Value', 2)

        self._create_lower_th_trackbar('lower Hue', 0)
        self._create_lower_th_trackbar('lower Saturation', 1)
        self._create_lower_th_trackbar('lower Value', 2)

    def _create_upper_th_trackbar(self, tb_name: str, idx: int):
        def cb(val: int):
            self.upper_th[idx] = val
            self._save_config()

        cv.createTrackbar(tb_name, self._window_name, 0, 255, cb)
        cv.setTrackbarPos(tb_name, self._window_name, self.upper_th[idx])

    def _create_lower_th_trackbar(self, tb_name: str, idx: int):
        def cb(val: int):
            self.lower_th[idx] = val
            self._save_config()

        cv.createTrackbar(tb_name, self._window_name, 0, 255, cb)
        cv.setTrackbarPos(tb_name, self._window_name, self.lower_th[idx])

    def _save_config(self):
        with open(self._config_fn, 'w') as file:
            file.writelines(str(val) + '\n' for val in self.lower_th)
            file.writelines(str(val) + '\n' for val in self.upper_th)

    def _loadconfig(self):
        try:
            with open(self._config_fn, 'r') as file:
                lines = file.readlines()

            for i in range(6):
                val = int(lines[i].strip())
                assert 0 <= val <= 255
                if i < 3:
                    self.lower_th[i] = val
                else:
                    self.upper_th[i-3] = val
        except Exception as e:
            print('Could not load config:')
            print(e)


class ObjectTracker:
    """Tracks an object from a camera stream."""

    def __init__(self, cam_id: int) -> None:
        self.cam_id = cam_id

        self._calibration = Calibration(self.cam_id)

        self._cap = cv.VideoCapture(self.cam_id)
        if not self._cap.isOpened():
            print(f"Cannot open camera with id {self.cam_id}")

        self.process()

    @property
    def lower_th(self) -> np.array:
        return self._calibration.lower_th

    @property
    def upper_th(self) -> np.array:
        return self._calibration.upper_th

    def _read_new_frame(self) -> np.array:
        ret, frame = self._cap.read()
        if not ret:
            print(f'Could not read frame for camera with id {self.cam_id}')
        return frame

    def _threshold(self, image: np.array) -> np.array:
        hsv_img = cv.cvtColor(image, cv.COLOR_RGB2HSV)

        mask = cv.inRange(hsv_img, self.lower_th, self.upper_th)
        mask = cv.erode(mask, None, iterations=3)
        mask = cv.dilate(mask, None, iterations=3)      
  
        res = cv.bitwise_and(image, image, mask=mask)
        return res

    def _get_contour(self, image: np.array) -> np.array:
        if np.any(image):
            imgray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            contours, hierarchy = cv.findContours(imgray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                return max(contours, key = cv.contourArea)

    def _get_moment(self, contour: np.array) -> Tuple[int, int]:
        if contour is not None:
            M = cv.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return cX, cY

    def _draw_contour(self, image: np.array, contour: np.array) -> None:
        cv.drawContours(image, contour, -1, (0,255,0), 3)     

    def _draw_moment(self, image: np.array, moment: Tuple[int, int]) -> None:
        cv.circle(image, moment, 5, (0, 0, 255), -1)

    def _get_proccessed_image(self) -> np.array:
        img = self.thresholded.copy()
        self._draw_contour(img, self.contour)
        self._draw_moment(img, self.moment)
        return img

    def process(self) -> None:
        self.original = self._read_new_frame()
        self.thresholded = self._threshold(self.original)
        self.contour = self._get_contour(self.thresholded)
        self.moment = self._get_moment(self.contour)
        self.processed_image = self._get_proccessed_image()

    def show(self) -> None:
        cv.imshow(f'Cam{self.cam_id} original', self.original)
        cv.imshow(f'Cam{self.cam_id} processed', self.processed_image)

    def release_cam(self) -> None:
        self._cap.release()


if __name__ == '__main__':
    ot1 = ObjectTracker(2)
    ##ot2 = ObjectTracker(2)

    while True:
        ot1.process()
        #ot2.process()
        ot1.show()
        #ot2.show()

        if cv.waitKey(1) == ord('q'):
            break

    cv.destroyAllWindows()
