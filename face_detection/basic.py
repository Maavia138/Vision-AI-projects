import cv2
import mediapipe as mp
import time

# Video Capture
cap = cv2.VideoCapture("Pose_videos/5.mp4")

pTime = 0

# MediaPipe Setup
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh

faceMesh = mpFaceMesh.FaceMesh(
    max_num_faces=2
)

# Drawing Specifications
drawSpec = mpDraw.DrawingSpec(
    color=(0, 0, 0),   # BLACK points
    thickness=4,
    circle_radius=7
)

# Window setup
cv2.namedWindow("Face Mesh", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Face Mesh", 1200, 700)

while True:

    # Read Frame
    success, img = cap.read()

    if not success:
        break

    # Convert BGR to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process image
    results = faceMesh.process(imgRGB)

    # Check faces
    if results.multi_face_landmarks:

        for faceLms in results.multi_face_landmarks:

            # Draw mesh
            mpDraw.draw_landmarks(
                img,
                faceLms,
                mpFaceMesh.FACEMESH_CONTOURS,
                drawSpec,
                drawSpec
            )

            # Get landmarks
            for id, lm in enumerate(faceLms.landmark):

                ih, iw, ic = img.shape

                x = int(lm.x * iw)
                y = int(lm.y * ih)
                z = lm.z

                # Print x y z
                print(f"ID: {id}  X: {x}  Y: {y}  Z: {z:.4f}")

                # Draw ID on face
                cv2.putText(
                    img,
                    str(id),
                    (x, y),
                    cv2.FONT_HERSHEY_PLAIN,
                    0.5,
                    (255, 0, 255),
                    1
                )

    # FPS calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(
        img,
        f'FPS: {int(fps)}',
        (20, 70),
        cv2.FONT_HERSHEY_PLAIN,
        3,
        (0, 0, 0),
        3
    )

    # Show output
    cv2.imshow("Face Mesh", img)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()