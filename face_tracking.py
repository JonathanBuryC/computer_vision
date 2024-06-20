import cv2
import mediapipe as mp
import time

# Initialisation de la capture vidéo depuis la webcam
cap = cv2.VideoCapture(0)  # ("Videos/test.mp4")
pTime = 0

# Initialisation de Mediapipe pour le maillage facial
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=2, circle_radius=2)

# Configuration de la taille de la fenêtre de capture
frameWidth = 1920
frameHeight = 1080
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)

while True:
    success, img = cap.read()
    if not success:
        print("Échec de la capture de l'image")
        continue  # Passe à l'itération suivante si la capture échoue
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)  # Traite l'image pour le maillage facial
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:  # Pour chaque maillage facial détecté
            # Dessine le maillage sur le visage sans utiliser FACE_CONNECTIONS
            mpDraw.draw_landmarks(
                image=img,
                landmark_list=faceLms,
                connections=mpFaceMesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawSpec,
                connection_drawing_spec=drawSpec)
    
    cTime = time.time()  # Temps actuel en secondes (epoch)
    fps = 1 / (cTime - pTime)  # Calcul des FPS
    pTime = cTime  # Met à jour le temps précédent
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)  # Affiche les FPS
    cv2.imshow("Image", img)  # Affiche la vidéo
    if cv2.waitKey(1) == 27:
        break  # Quitte la boucle si la touche Échap est pressée

# Libération des ressources
cap.release()
cv2.destroyAllWindows()
