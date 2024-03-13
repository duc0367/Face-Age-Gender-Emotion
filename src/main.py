import cv2
from facenet_pytorch import MTCNN
import torch

from utils import convert_tl_br_to_tl_wh

vid = cv2.VideoCapture(0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(keep_all=True, device=device)

prev_frame = None
diff = 0
threshold = 20
boxes = None

while True:
    # Capture the video frame by frame
    ret, frame = vid.read()

    similarity = 1
    if prev_frame is None:
        prev_frame = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_diff = cv2.absdiff(gray, prev_gray)
    _, threshold_diff = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
    similarity = cv2.countNonZero(threshold_diff) / (gray.shape[0] * gray.shape[1])

    if similarity > 0.1:
        # The current frame is really different from the previous one
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        prev_frame = frame.copy()

    if boxes is not None:
        for box in boxes:
            tl = (int(box[0]), int(box[1]))
            br = (int(box[2]), int(box[3]))
            frame = cv2.rectangle(frame, tl, br, (0, 255, 0), 2)
            (x, y), (w, h) = convert_tl_br_to_tl_wh(tl, br)

            face = frame[y:y+h, x: x+w]

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
