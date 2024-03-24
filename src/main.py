import cv2
from facenet_pytorch import MTCNN
import torch
import numpy as np

from utils import convert_tl_br_to_tl_wh

from model.emotion.emotions import create_model

vid = cv2.VideoCapture(0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(keep_all=True, device=device)

prev_frame = None
diff = 0
threshold = 20
boxes = None
ages = None
genders = None
gens = None
emotions = None

# Categories of distribution
la = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
      '(25-32)', '(38-43)', '(48-53)', '(60-100)']
lg = ['Male', 'Female']

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


age1 = "../model/age/age_deploy.prototxt"
age2 = "../model/age/age_net.caffemodel"
gen1 = "../model/gender/gender_deploy.prototxt"
gen2 = "../model/gender/gender_net.caffemodel"
emotion_path = '../model/emotion/model.h5'

age_model = cv2.dnn.readNet(age2, age1)
gen_model = cv2.dnn.readNet(gen2, gen1)
emotion_model = create_model()
emotion_model.load_weights(emotion_path)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

while True:
    # Capture the video frame by frame
    ret, frame = vid.read()

    height = frame.shape[0]
    width = frame.shape[1]

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
        pred_boxes, _ = mtcnn.detect(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        prev_frame = frame.copy()
        if pred_boxes is not None:
            boxes = []
            ages = []
            gens = []
            emotions = []
            for box in pred_boxes:
                tl = (int(box[0]), int(box[1]))
                br = (int(box[2]), int(box[3]))
                boxes.append((tl, br))

                (x, y), (w, h) = convert_tl_br_to_tl_wh(tl, br)

                face = frame[max(0, y - 15):min(y + h + 15, height - 1), max(0, x - 15): min(x + w + 15, width - 1)]
                face = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, False)
                age_model.setInput(face)
                age_predictions = age_model.forward()
                age = la[age_predictions.argmax()]
                ages.append(age)

                gen_model.setInput(face)
                gen_predictions = gen_model.forward()
                gen = lg[gen_predictions.argmax()]
                gens.append(gen)
                roi_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                prediction = emotion_model.predict(cropped_img)
                max_index = int(np.argmax(prediction))
                emotions.append(emotion_dict[max_index])

    if boxes is not None:
        for idx, box in enumerate(boxes):
            (tl, br) = box
            (x, y) = tl
            frame = cv2.rectangle(frame, tl, br, (0, 255, 0), 2)
            age = ages[idx]
            gender = gens[idx]
            emotion = emotions[idx]

            cv2.putText(frame,
                        f'gender: {gender}, age: {age}, emotion: {emotion}',
                        (x - 10, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 217),
                        4,
                        cv2.LINE_AA)

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
