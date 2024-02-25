import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregando o modelo de detecção de objetos
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
modelPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(modelPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Carregando a lista de classes
classesPath = 'coco.names'
with open(classesPath, 'r') as f:
    classesList = f.read().splitlines()

# Inicializando o vídeo
videoPath = 0
video = cv2.VideoCapture(videoPath)

while True:
    check, img = video.read()
    if not check:
        print("Erro ao ler o frame do vídeo")
        break

    img = cv2.resize(img, (1270, 720))

    # Detecção e desenho de bounding boxes
    classIds, confs, bbox = net.detect(img, confThreshold=0.5)

    if len(classIds) > 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            label = classesList[classId - 1] if classId - 1 < len(classesList) else 'Desconhecido'

            x, y, w, h = box
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)
            cv2.putText(img, f'{label} {round(float(confidence), 2)}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Exibindo a imagem usando Matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title("Imagem")
    plt.show()

    key = cv2.waitKey(1)
    if key == 27:  # Se a tecla 'Esc' for pressionada, saia do loop
        break

cv2.destroyAllWindows()
video.release()
