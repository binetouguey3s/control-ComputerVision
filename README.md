# Atelier 35 – Contrôle Qualité Automatisé (Vision par Ordinateur)

Micro-service FastAPI qui analyse des images de produits manufacturés via un modèle de vision
Hugging Face (ResNet-50) et retourne un diagnostic qualité : **conforme** ou **défaut**.

## Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/binetouguey3s/control-ComputerVision.git
cd atelier35-control-ComputerVision

# Éditez .env et renseignez votre HF_TOKEN si nécessaire

# 3. Installer les dépendances
uv sync
```

---

## Lancer le serveur

```bash
uv run uvicorn main:app --reload
```

Le serveur démarre sur `http://127.0.0.1:8000`.  
La documentation interactive est disponible sur `http://127.0.0.1:8000/docs`.

---

## Variables d'environnement

| Variable     | Description                                      | Défaut                  |
|--------------|--------------------------------------------------|-------------------------|
| `MODEL_NAME` | Identifiant du modèle Hugging Face à utiliser    | `microsoft/resnet-50`   |
| `HF_TOKEN`   | Token d'accès Hugging Face (modèles privés/gated) | *(optionnel)*           |

Voir `.env.example` pour un modèle de configuration.

---

## Utilisation de l'API

### `POST /inspect-image`

Envoie une image de produit et reçoit un diagnostic qualité.

**Paramètres**

| Champ  | Type   | Description                        |
|--------|--------|------------------------------------|
| `file` | fichier | Image au format JPEG, PNG ou WebP  |

**Exemple avec curl**

```bash
curl -X POST http://127.0.0.1:8000/inspect-image \
  -F "file=@photo_produit.jpg"
```

**Réponse**

```json
{
  "status": "conforme",
  "confidence": 0.8921,
  "prediction": "carton",
  "filename": "photo_produit.jpg",
  "model_used": "microsoft/resnet-50"
}
```

Le champ `status` vaut `"conforme"` si le score de confiance est ≥ 0.70, sinon `"defaut"`.

## Contraintes de validation

| Règle                  | Valeur              |
|------------------------|---------------------|
| Formats acceptés       | JPEG, PNG, WebP     |
| Taille maximale        | 5 Mo                |


## Structure du projet


atelier35-control-ComputerVision/
├── main.py                 # Code source de l'API FastAPI
├── pyproject.toml          # Dépendances du projet (uv)
├── .env.example            # Modèle de configuration des variables d'env
├── .env                    # Variables d'environnement locales (non versionné)
├── models_evaluation.md    # Comparaison des modèles candidats
└── README.md               # Ce fichier

## Choix du modèle

Voir models_evaluation.md pour la comparaison détaillée des deux modèles candidats et la justification du choix final.
