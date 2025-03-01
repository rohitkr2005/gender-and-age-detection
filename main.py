import cv2

def faceBox(faceNet, frame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bbox = []
    
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)
            bbox.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    
    return frame, bbox  # Return the frame first, then bbox

# Model paths
faceModel = "opencv_face_detector_uint8.pb"
faceProto = "opencv_face_detector.pbtxt"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

# Load networks
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)  # Fixed: was using ageModel instead of genderModel

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-3)', '(4-7)', '(8-12)', '(13-18)', '(19-24)', '(25-32)', '(33-40)', '(41-50)', '(51-80)']
genderList = ['Male', 'Female']

video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        break
        
    frame, bboxs = faceBox(faceNet, frame)
    
    for bbox in bboxs:
        try:
            face = frame[max(0, bbox[1]):bbox[3], max(0, bbox[0]):bbox[2]]
            if face.size == 0:
                continue
                
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            
            genderNet.setInput(blob)
            genderPred = genderNet.forward()
            gender = genderList[genderPred[0].argmax()]

            ageNet.setInput(blob)
            agePred = ageNet.forward()
            age = ageList[agePred[0].argmax()]

            label = f"{gender}, {age}"  # Fixed string formatting
            cv2.putText(frame, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255), 2)
        
        except Exception as e:
            print(f"Error processing face: {e}")
            continue
    
    cv2.imshow("age-gender", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()