# Evaluation des modeles candidats

## Modèle 1 : google/vit-base-patch16-224
- Architecture : Vision Transformer (ViT) complet.
- Lien: https://huggingface.co/google/vit-base-patch16-224?inference_provider=hf-inference
- Taille/ Parametres : 86,6 millions de paramètres
- Licence : Apache 2.0
- Justification : C'est le modèle ViT de référence absolue sur Hugging Face. Grâce à son mécanisme d'attention globale, il est capable de détecter de subtiles anomalies géométriques, des rayures fines ou des casses sur la surface d'un produit.

## Modele 2 : microsoft/resnet-50
- Architecture : Réseau de neurones convolutif classique (CNN).
- Lien: https://huggingface.co/microsoft/resnet-50
- Taille/ Parametres : 25.6M de paramètres
- Licence: Apache 2.0
- Justification :  C'est une architecture ultra-légère et robuste. Contrairement aux ViT, ResNet traite les images extrêmement vite sur un simple CPU sans saturer la mémoire de notre serveur FastAPI.

## Modele final : microsoft/resnet-50
- Modele leger , ideal pour le CPU( tres faible empreinte mémoire)