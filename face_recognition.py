import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from reface_attack import reface_attack
from adversarial_detector import detect_adversarial

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialisation MTCNN et modèle
mtcnn = MTCNN(keep_all=True, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Chargement embeddings connus
known_embeddings = {}
names = ['youssef', 'hisham']
for name in names:
    emb = torch.load(f'embeddings/{name}.pt').to(device)
    known_embeddings[name.capitalize()] = emb

def preprocess_face(face_img):
    face_pil = Image.fromarray(face_img).resize((160,160))
    face_tensor = torch.tensor(np.array(face_pil)).permute(2,0,1).float() / 255.
    face_tensor = (face_tensor - 0.5) / 0.5
    return face_tensor.unsqueeze(0).to(device)

def recognize(embedding, known_embeddings):
    best_name = "Inconnu"
    best_score = -1
    for name, emb in known_embeddings.items():
        sim = torch.nn.functional.cosine_similarity(embedding, emb).item()
        if sim > best_score:
            best_score = sim
            best_name = name
    return best_name, best_score

cap = cv2.VideoCapture(0)

print("Appuyez sur 'c' pour lancer attaque Reface")
print("Appuyez sur 'q' pour quitter")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(img_rgb)

    key = cv2.waitKey(1) & 0xFF

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = img_rgb[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face_tensor = preprocess_face(face)

            with torch.no_grad():
                emb = model(face_tensor)

            name, score = recognize(emb, known_embeddings)
            color = (0, 255, 0) if score > 0.7 else (0, 0, 255)
            label = f"{name} ({score:.2f})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if key == ord('c'):
                print("Lancement attaque Reface...")
                target_emb = known_embeddings['Hisham']  # cible de l'attaque
                perturbed = reface_attack(model, face_tensor, target_emb, num_steps=100, lr=0.05)

                # Reconnaissance post-attaque
                with torch.no_grad():
                    emb_adv = model(perturbed)
                name_adv, score_adv = recognize(emb_adv, known_embeddings)
                print(f"Reconnaissance après attaque : {name_adv} ({score_adv:.2f})")

                # Détection adversariale simple
                if detect_adversarial(face_tensor, perturbed, threshold=0.1):
                    print("!!! Attaque adversariale détectée !!!")

                # Affichage visage attaqué
                img_np = perturbed.squeeze(0).permute(1,2,0).cpu().numpy()
                img_np = (img_np * 0.5 + 0.5) * 255
                img_np = img_np.clip(0,255).astype(np.uint8)
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                cv2.imshow("Visage attaqué", img_np)

    cv2.imshow("Reconnaissance Faciale", frame)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
