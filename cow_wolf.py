import cv2
import numpy as np
import argparse
from module_net import MobileNetDetector

class DangerDetector(MobileNetDetector):  # Agora herdamos de MobileNetDetector
    def __init__(self, CONFIDENCE=0.7):
        # Inicialize a classe Pai
        super().__init__(CONFIDENCE=CONFIDENCE)
        # Ajuste o valor de confiança para o valor que você achar melhor
        self.CONFIDENCE = CONFIDENCE

    def separar_caixa_entre_animais(self, img: np.ndarray, resultados: list) -> (np.ndarray, dict):
        
        """Não mude ou renomeie esta função."""
        img = img.copy()
        animais = {'vaca': [], 'lobo': []}
        return img, animais

    def calcula_iou(self, boxA: list, boxB: list) -> float:
        """Calcula a intersecção sobre a união (Intersection over Union - IoU) entre duas caixas."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def checar_perigo(self, image: np.ndarray, animais: dict) -> np.ndarray:
        """Não mude ou renomeie esta função."""
        return image

def main():
    import time
    bgr = cv2.imread("img/cow_wolf05.png")

    Detector = DangerDetector()
    image, results = Detector.detect(bgr)
    image, animais = Detector.separar_caixa_entre_animais(image, results)
    print(animais)
    image = Detector.checar_perigo(image, animais)

    cv2.imshow("Result_MobileNet", image)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
