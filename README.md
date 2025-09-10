# Face Recognition System with Adversarial Attack using Reface
This project implements a real-time face recognition system using deep learning, with capabilities to simulate and detect adversarial attacks via the Reface method. It demonstrates vulnerabilities in face recognition models and proposes basic defenses. Developed as part of the Master WISD program at Université Sidi Mohamed Ben Abdellah, Faculté des Sciences Dhar El Mahraz.
# Authors

Youssef Eljaouhary
Hisham Bamouh

# Supervisor

Jamal Rifii

# Description
The system uses MTCNN for face detection and InceptionResnetV1 for generating facial embeddings. It supports real-time recognition from a webcam and includes modules for adversarial attacks (impersonation via face substitution) and detection (based on L2 norm comparisons). The Reface attack perturbs input images to mislead the model while keeping changes visually imperceptible.
Key objectives:

# Features

Face Detection and Recognition: Detects faces in real-time and matches them against known embeddings using cosine similarity.
Adversarial Attack Simulation: Uses gradient-based optimization to generate perturbed images that mimic a target identity.
Adversarial Detection: Compares L2 norms of original and perturbed embeddings to flag potential attacks.
Embedding Generation: Precomputes facial embeddings for known individuals.

# Requirements

Python
PyTorch
OpenCV
Pillow (PIL)
MTCNN (via facenet-pytorch)
InceptionResnetV1 model (pre-trained from FaceNet)
Other dependencies: NumPy, Torchvision

Install dependencies:
textpip install torch torchvision opencv-python pillow facenet-pytorch
Project Structure

data/: Directory for input images (e.g., Youssef.jpg, Hisham.jpg).
embeddings/: Directory to store generated embedding files (e.g., Youssef.pt).
generate_embeddings.py: Generates embeddings for known faces.
face_recognition.py: Runs real-time face recognition with optional Reface attack (press 'c' to trigger).
reface_attack.py: Implements the adversarial attack to manipulate embeddings.
adversarial_detector.py: Detects adversarial perturbations via L2 norm.

# Usage

# Generate Embeddings:
textpython generate_embeddings.py
This processes images in data/ and saves embeddings in embeddings/.
Run Face Recognition:
textpython face_recognition.py

# Opens webcam for real-time detection.

Simulate Reface Attack (Standalone):
Use reface_attack.py to perturb a specific image toward a target embedding.
Detect Adversarial Attack:
Use adversarial_detector.py to compare tensors:
pythondetect_adversarial(original_tensor, perturbed_tensor, threshold=0.1)

if you want more detail , you can check rapport that's provides in this rep.

# Conclusion
This project highlights the effectiveness of adversarial attacks on face recognition systems and the need for robust defenses. Future work could include advanced detection methods.
References

Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014). Explaining and Harnessing Adversarial Examples.
Sandler, M., et al. (2018). FaceNet: A Unified Embedding for Face Recognition and Clustering.
PyTorch and FaceNet-PyTorch documentation.

License
This project is for educational purposes. No license specified; contact authors for usage rights.35,8s
