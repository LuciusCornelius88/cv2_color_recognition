import cv2
import cvzone
import numpy as np

from config import *


def create_trackbars():
    cv2.createTrackbar(H, WIND_NAME, 180, 180, do_nothing)
    cv2.createTrackbar(S, WIND_NAME, 255, 255, do_nothing)
    cv2.createTrackbar(V, WIND_NAME, 255, 255, do_nothing)
    cv2.createTrackbar(HL, WIND_NAME, 150, 180, do_nothing)
    cv2.createTrackbar(SL, WIND_NAME, 100, 255, do_nothing)
    cv2.createTrackbar(VL, WIND_NAME, 100, 255, do_nothing)


def detect_color(frame, contours, color):
    for x in range(len(contours)):
        if cv2.contourArea(contours[x]) > 500:
            x, y, w, h = cv2.boundingRect(contours[x])
            if w >= 150 and h >= 150:
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                frame = cv2.rectangle(frame, (x, y), (x + 60, y - 25), (0, 0, 0), -1)
                cv2.putText(frame, color, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return frame


def handle_color(frame, hsv, color, hl, sl, vl, h, s, v):
    kernel = np.ones((3, 3), np.uint8)

    lower = np.array([hl, sl, vl])
    upper = np.array([h, s, v])

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, h = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    frame = detect_color(frame, contours, color)

    return frame, mask


def main():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow(WIND_NAME, cv2.WINDOW_NORMAL)

    create_trackbars()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.bilateralFilter(frame, 10, 100, 100)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        h = cv2.getTrackbarPos(H, WIND_NAME)
        s = cv2.getTrackbarPos(S, WIND_NAME)
        v = cv2.getTrackbarPos(V, WIND_NAME)
        hl = cv2.getTrackbarPos(HL, WIND_NAME)
        sl = cv2.getTrackbarPos(SL, WIND_NAME)
        vl = cv2.getTrackbarPos(VL, WIND_NAME)

        frame, mask_red = handle_color(frame, hsv, 'Red', hl=150, sl=100, vl=100, h=180, s=255, v=255)
        frame, mask_blue = handle_color(frame, hsv, 'Blue', hl=50, sl=70, vl=70, h=140, s=185, v=185)
        frame, mask_green = handle_color(frame, hsv, 'Green', hl=60, sl=35, vl=35, h=140, s=255, v=255)

        # frame, mask_green = handle_color(frame, hsv, 'Green', hl, sl, vl, h, s, v)

        image_stacked = cvzone.stackImages([frame, mask_red, mask_blue, mask_green], 2, 1)

        cv2.imshow('frame', image_stacked)

        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
