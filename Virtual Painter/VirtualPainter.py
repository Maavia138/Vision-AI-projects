import cv2
import numpy as np
import os
import HandTrackingModule as htm

# ================= SETTINGS =================
brushThickness = 25
eraserThickness = 100

# Color zones as fractions of frame width (works at any resolution)
COLOR_ZONES = [
    (0.195, 0.352, (255, 0, 255)),   # purple
    (0.430, 0.586, (255, 0, 0)),     # blue
    (0.625, 0.742, (0, 255, 0)),     # green
    (0.820, 0.938, (0, 0, 0)),       # eraser
]

# ================= HEADER IMAGES =================
folderPath = "Header"
myList = sorted(os.listdir(folderPath))

overlayList = []
for imPath in myList:
    image = cv2.imread(os.path.join(folderPath, imPath))
    if image is not None:
        overlayList.append(image)

if len(overlayList) < 4:
    raise FileNotFoundError(
        "Need 4 images in the Header folder (1.png–4.png). "
        "Download them from the Virtual Painter tutorial assets."
    )

drawColor = COLOR_ZONES[0][2]
headerIndex = 0

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# ================= HAND DETECTOR =================
detector = htm.handDetector(detectionCon=0.65, maxHands=1)

xp, yp = 0, 0
imgCanvas = None
header = None
headerH = 125


def pickColor(x, frameW):
    global drawColor, headerIndex, header
    for i, (xMin, xMax, color) in enumerate(COLOR_ZONES):
        if int(xMin * frameW) < x < int(xMax * frameW):
            drawColor = color
            headerIndex = i
            header = cv2.resize(overlayList[i], (frameW, headerH))
            return True
    return False

# ================= MAIN LOOP =================
while True:
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)
    h, w, _ = img.shape

    if imgCanvas is None:
        imgCanvas = np.zeros((h, w, 3), np.uint8)
        headerH = max(1, int(125 * h / 720))
        header = cv2.resize(overlayList[headerIndex], (w, headerH))

    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)

    if len(lmList) != 0:

        x1, y1 = lmList[8][1:]   # index finger tip
        x2, y2 = lmList[12][1:]  # middle finger tip

        fingers = detector.fingersUp()
        inHeader = y1 < headerH

        # ================= SELECTION MODE (top bar) =================
        if len(fingers) != 0 and inHeader and fingers[1]:
            xp, yp = 0, 0
            pickColor(x1, w)

            if fingers[2]:
                cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)
            else:
                cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)

        # ================= DRAW MODE (below header) =================
        elif len(fingers) != 0 and fingers[1] and not fingers[2]:

            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

    # ================= MERGE CANVAS =================
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # ================= HEADER =================
    img[0:headerH, 0:w] = header

    # ================= DISPLAY =================
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)

    cv2.waitKey(1)