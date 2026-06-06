import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv2.VideoCapture('Pose_videos/2.mp4')

pTime = 0

while True:

    success, img = cap.read()

    # Prevent crash when video ends
    if not success:
        print("Video Ended")
        break

    # Resize video so full body fits properly
    img = cv2.resize(img, (800, 600))

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = pose.process(imgRGB)

    if results.pose_landmarks:

        mpDraw.draw_landmarks(
            img,
            results.pose_landmarks,
            mpPose.POSE_CONNECTIONS
        )

        for id, lm in enumerate(results.pose_landmarks.landmark):

            h, w, c = img.shape

            cx, cy = int(lm.x * w), int(lm.y * h)

            print(id, cx, cy)

            cv2.circle(
                img,
                (cx, cy),
                5,
                (255, 0, 0),
                cv2.FILLED
            )

    cTime = time.time()

    fps = 1 / (cTime - pTime)

    pTime = cTime

    cv2.putText(
        img,
        str(int(fps)),
        (70, 50),
        cv2.FONT_HERSHEY_PLAIN,
        3,
        (255, 0, 0),
        3
    )

    cv2.imshow("Pose Estimation", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()