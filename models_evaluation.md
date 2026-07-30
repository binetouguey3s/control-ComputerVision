# Évaluation des modèles candidats – Atelier 35

## Contexte de sélection

L'objectif est de déployer un micro-service de contrôle qualité sur une ligne de production.
Le modèle doit être capable de traiter des images rapidement, idéalement sur **CPU** (pas de GPU
garanti en production), et doit être utilisable dans un contexte **commercial** (licence permissive).

---

## Modèle 1 : `google/vit-base-patch16-224`

| Critère        | Valeur                                                                 |
|----------------|------------------------------------------------------------------------|
| **Hub**        | [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) |
| **Architecture** | Vision Transformer (ViT) – attention globale sur patches 16×16      |
| **Paramètres** | ~86 millions                                                           |
| **Taille**     | ~330 Mo                                                                |
| **Précision**  | ~81% top-1 sur ImageNet                                                |
| **Licence**    | Apache 2.0 ✅ (usage commercial autorisé)                              |

**Points forts :** L'attention globale permet de détecter des anomalies subtiles (rayures fines,
irrégularités de surface) que les CNN peuvent manquer. Très bonne précision.

**Points faibles :** Lourd (86M params, ~330 Mo), lent sur CPU (latence élevée par image),
consommation mémoire importante — risque de saturation sur un serveur léger.

---

## Modèle 2 : `microsoft/resnet-50`

| Critère        | Valeur                                                                 |
|----------------|------------------------------------------------------------------------|
| **Hub**        | [microsoft/resnet-50](https://huggingface.co/microsoft/resnet-50)      |
| **Architecture** | Réseau convolutif résiduel (CNN) – 50 couches                        |
| **Paramètres** | ~25 millions                                                           |
| **Taille**     | ~100 Mo                                                                |
| **Précision**  | ~76% top-1 sur ImageNet                                                |
| **Licence**    | Apache 2.0 ✅ (usage commercial autorisé)                              |

**Points forts :** Architecture légère et éprouvée, excellente vitesse d'inférence sur CPU,
faible empreinte mémoire — idéal pour un micro-service embarqué ou un serveur sans GPU.

**Points faibles :** Précision légèrement inférieure au ViT. Moins performant sur des textures
très fines ou des défauts microscopiques.

---

## Tableau comparatif

| Critère               | ViT Base (google) | ResNet-50 (microsoft) |
|-----------------------|:-----------------:|:---------------------:|
| Paramètres            | 86M               | 25M                   |
| Taille du modèle      | ~330 Mo           | ~100 Mo               |
| Précision ImageNet    | ~81%              | ~76%                  |
| Vitesse CPU           | Lente             | Rapide                |
| Licence commerciale   | ✅ Apache 2.0     | ✅ Apache 2.0         |
| Adapté sans GPU       | ⚠️ Difficile      | ✅ Oui                |

---

## Choix final : `microsoft/resnet-50`

**ResNet-50** a été retenu pour les raisons suivantes :

1. **Ressource matérielle** : le micro-service cible un déploiement sur CPU standard. ResNet-50
   est 3× plus léger que ViT et offre une latence d'inférence bien inférieure, évitant toute
   saturation mémoire du serveur FastAPI.

2. **Robustesse** : architecture CNN résiduelle ultra-stable, comportement prévisible en
   production.

3. **Licence** : Apache 2.0, identique au ViT, donc compatible avec un usage commercial sans
   restriction.

4. **Evolutivité** : le code de l'API est paramétré via la variable d'environnement `MODEL_NAME`,
   ce qui permet de basculer vers `google/vit-base-patch16-224` (ou tout autre modèle compatible
   `AutoImageProcessor`) sans modifier le code source.
