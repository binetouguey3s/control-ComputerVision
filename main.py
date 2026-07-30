import io
import os
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from PIL import Image
import torch
from transformers import AutoImageProcessor, ResNetForImageClassification

# Variables d'environnement 
MODEL_NAME = os.getenv("MODEL_NAME", "microsoft/resnet-50")
HF_TOKEN   = os.getenv("HF_TOKEN", None)

# Initialisation de l'application 
app = FastAPI(
    title="Atelier 35 - Contrôle Qualité Vision",
    description="Micro-service d'analyse d'images de produits via ResNet-50 pour la détection de défauts.",
    version="1.0.0",
)

# Chargement du modèle au démarrage (une seule fois en mémoire) 
print(f"Chargement du modèle : {MODEL_NAME}")
try:
    model = ResNetForImageClassification.from_pretrained(  # ResNetForImageClassification est utilisé pour la classification d'images et from_pretrained charge le modèle pré-entraîné depuis Hugging Face.
        MODEL_NAME,
        token=HF_TOKEN,
        ignore_mismatched_sizes=True, # permet de charger le modèle même si la taille des poids ne correspond pas exactement à celle attendue par la classe ResNetForImageClassification. Cela peut se produire si le modèle a été modifié ou si une version différente est utilisée.
    )
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, token=HF_TOKEN) # AutoImageProcessor est utilisé pour prétraiter les images avant de les passer au modèle. Il adapte automatiquement le prétraitement en fonction du modèle chargé.
    model.eval() # met le modèle en mode évaluation (inférence) plutôt qu'en mode entraînement. Cela désactive certaines fonctionnalités comme le dropout et la normalisation par lot, qui ne sont pas nécessaires lors de l'inférence.
except Exception as e:
    print(f"Erreur au chargement du modèle : {e}")
    raise RuntimeError(f"Impossible de charger le modèle '{MODEL_NAME}' : {e}")

# Constantes de validation 
MAX_FILE_SIZE      = 5 * 1024 * 1024  # 5 Mo
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}

# Mots-clés ImageNet qui indiquent une anomalie / défaut potentiel.
# ResNet-50 est entraîné sur ImageNet (1000 classes génériques) et non sur des défauts industriels. On utilise donc un seuil de confiance bas comme proxy :
# si le modèle n'est pas sûr (confidence < seuil), on considère le produit comme potentiellement défectueux.
CONFIDENCE_THRESHOLD = 0.70

# Fonction utilitaire pour interpréter la prédiction
def interpret_prediction(label: str, confidence: float) -> str:
    """
    Traduit la prédiction ImageNet en diagnostic qualité.
    Un score de confiance faible signale une anomalie visuelle.
    """
    if confidence >= CONFIDENCE_THRESHOLD:
        return "conforme"
    return "defaut"


#  Endpoint principal
@app.post(
    "/inspect-image",
    status_code=status.HTTP_200_OK,
    summary="Inspecter une image de produit",
    response_description="Diagnostic qualité : conforme ou défaut",
)
# sert à charger le modèle en mémoire
async def inspect_image(file: UploadFile = File(...)): 
    """
    Reçoit une image de pièce industrielle en multipart/form-data.
    Retourne un diagnostic structuré JSON avec le statut et le score de confiance.
    """
    # Validation 1 : type MIME
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Format non supporté ({file.content_type}). "
                "Formats acceptés : JPEG, PNG, WebP."
            ),
        )

    # Lecture en mémoire
    image_bytes = await file.read()

    # Validation 2 : taille du fichier
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"Fichier trop lourd ({len(image_bytes) / (1024 * 1024):.2f} Mo). " 
                "Limite : 5 Mo."
            ),
        )

    # Validation 3 : image valide (Pillow)
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB") # convertir en format RGB pour que le modèle puisse traiter les images , io.BytesIO est utilisé pour passer le fichier en mémoire
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, 
            detail="Le fichier est corrompu ou n'est pas une image valide.",
        )

    # Inférence 
    try:
        inputs = processor(image, return_tensors="pt") # prétraitement des images avant de les passer au modèle

        with torch.no_grad(): # empêche le modèle de modifier les poids
            logits = model(**inputs).logits  # shape [1, num_classes] , num_classes est le nombre de classes du modèle

        # Classe la plus probable
        predicted_idx   = logits.argmax(-1).item() 
        probabilities   = torch.nn.functional.softmax(logits, dim=-1)
        confidence      = round(probabilities[0][predicted_idx].item(), 4) # sert à afficher le score de confiance
        predicted_label = model.config.id2label[predicted_idx] # sert à afficher le nom de la classe
        diagnostic      = interpret_prediction(predicted_label, confidence) # sert à afficher le statut

        return {
            "status":     diagnostic,          # "conforme" ou "defaut"
            "confidence": confidence,           # score entre 0 et 1
            "prediction": predicted_label,      # classe ImageNet brute
            "filename":   file.filename,
            "model_used": MODEL_NAME,
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Erreur durant l'analyse : {str(e)}",
        )
