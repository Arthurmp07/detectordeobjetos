import cv2
import numpy as np

# Carrega o modelo pré-treinado MobileNet SSD
modelo = cv2.dnn.readNetFromTensorflow('pre-trained-models/ssdlite_mobilenet_v3_large_320x320_coco.config')

# Definindo cores para os objetos detectados
cores = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

# Nomes das etiquetas das classes COCO
labels = ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
          'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
          'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
          'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
          'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
          'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
          'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
          'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
          'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
          'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
          'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# Função para realizar a detecção de objetos
def detectar_objetos(frame):
    altura, largura, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    
    modelo.setInput(blob)
    deteccoes = modelo.forward()

    for i in range(deteccoes.shape[2]):
        confianca = deteccoes[0, 0, i, 2]
        if confianca > 0.5:
            classe_id = int(deteccoes[0, 0, i, 1])
            caixa = deteccoes[0, 0, i, 3:7] * np.array([largura, altura, largura, altura])
            x, y, w, h = caixa.astype("int")
            
            cor = cores[classe_id]
            cv2.rectangle(frame, (x, y), (w, h), cor, 2)
            texto = "{}: {:.4f}".format(labels[classe_id], confianca)
            cv2.putText(frame, texto, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)

    return frame

captura = cv2.VideoCapture(0)

while True:
    _, frame = captura.read()
    
    frame_detectado = detectar_objetos(frame)
    
    cv2.imshow("Detecção de Objetos", frame_detectado)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

captura.release()
cv2.destroyAllWindows()
