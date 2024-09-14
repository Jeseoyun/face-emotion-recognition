import sys
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model, model_from_json
from tensorflow.compat.v1.keras.backend import set_session

from facial_analysis import FacialImageProcessing


def main(img_path):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    set_session(sess)

    imgProcessing = FacialImageProcessing(False)

    model = load_model('models/affectnet_emotions/mobilenet_7.h5')

    INPUT_SIZE = (224, 224)

    idx_to_class = {
        0: 'Anger',
        1: 'Anger',     # 원본 label: Disgust
        2: 'Anger',     # 원본 label: Fear
        3: 'Happiness',
        4: 'Neutral',
        5: 'Sadness',
        6: 'Surprise'
    }

    frame_bgr = cv2.imread(img_path)
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # 입력 이미지에서 얼굴 위치 찾기
    # 여러 명의 얼굴도 인식 가능 : 프로젝트 요구사항에 따라 예외 처리 필요
    bounding_boxes, points = imgProcessing.detect_faces(frame)
    points = points.T

    for bbox, p in zip(bounding_boxes, points):
        # 입력 이미지 preprocessing
        box = bbox.astype(np.int32)
        x1, y1, x2, y2 = box[0:4]
        face_img = frame[y1:y2, x1:x2, :]

        face_img = cv2.resize(face_img, INPUT_SIZE)
        inp = face_img.astype(np.float32)
        inp[..., 0] -= 103.939
        inp[..., 1] -= 116.779
        inp[..., 2] -= 123.68
        inp = np.expand_dims(inp, axis=0)

        # 이미지로부터 표정 예측
        scores = model.predict(inp)[0]
        result = idx_to_class[np.argmax(scores)]
        print(result)

        # 결과 저장
        with open("results/res.txt", "w") as f:
            f.write(result)


if __name__ == "__main__":
    main(
        img_path=sys.argv[1]  # s3 url
    )
