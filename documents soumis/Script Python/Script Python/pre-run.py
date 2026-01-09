"""
Script à exécuter AVANT mode.py pour précharger
"""
import subprocess
import sys
import time

print("🔄 Préparation de l'environnement SmartGraph...")

# 1. Installer packages manquants
print("📦 Installation des packages critiques...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "protobuf", "huggingface_hub[hf_xet]", "hf_xet", "sentencepiece"])

# 2. Pré-télécharger les modèles (sans les charger en mémoire)
print("⬇️  Pré-téléchargement des modèles (peut prendre 10-15min)...")

models = [
    "Jean-Baptiste/camembert-ner",
    "dslim/bert-base-NER",
    "intfloat/multilingual-e5-large",
    "facebook/bart-large-mnli"
]

for model in models:
    print(f"  - {model}")
    try:
        subprocess.run([
            sys.executable, "-c", 
            f"from huggingface_hub import snapshot_download; snapshot_download(repo_id='{model}', local_dir=f'./models/{model.split('/')[-1]}')"
        ], timeout=300)
    except:
        print(f"    ⚠️  Timeout pour {model}, continuons...")

print("✅ Préparation terminée!")
print("🎯 Maintenant exécutez: python mode.py")