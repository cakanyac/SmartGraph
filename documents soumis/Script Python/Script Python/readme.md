

# SmartGraph Sémantique v3.0 

## Présentation

SmartGraph Sémantique est une architecture avancée de machine learning qui intègre un raisonnement ontologique et une interopérabilité complète pour optimiser le traitement et l'interoperabilité des données dans des applications agricoles et robotiques. Elle exploite des modèles modernes de machine learning et des technologies du web sémantique pour améliorer la prise de décision et l'automatisation dans divers secteurs, tels que l'agriculture.

### Fonctionnalités principales

* **Machine Learning (ML)** : Intégration de la reconnaissance d'entités nommées (NER), des embeddings de graphes de connaissances (KGE) et du traitement du langage naturel (NLP).
* **Raisonnement basé sur les ontologies** : Utilisation de RDF, OWL et SPARQL pour les requêtes sémantiques et l'inférence ontologique.
* **Graph Neural Networks (GNN)** : Pour la prédiction de relations et l'amélioration du raisonnement basé sur les graphes.
* **Traitement des données** : Capacité à gérer des ensembles de données divers et volumineux grâce au traitement par lot et à un cache intelligent.
* **Intégration** : Support de plusieurs ontologies, dont AGROVOC, SSN, SAREF et GeoNames.

### Prérequis

* Python 3.10+
* NVIDIA CUDA pour l'accélération GPU
* Frameworks de machine learning : PyTorch, Transformers, SpaCy, etc.
* Neo4j pour l'intégration de la base de données graphique

### Installation

1. **Clonez le repository :**

```bash
git clone https://github.com/yourusername/smartgraph.git
cd smartgraph
```

2. **Créez et activez un environnement virtuel :**

```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Installez les dépendances nécessaires :**

```bash
pip install -r requirements.txt
```

4. **Configurez la connexion à Neo4j :**

Assurez-vous que votre **instance Neo4j** est en cours d'exécution et mettez à jour les informations de connexion :

* **Identifiants de la base de données** :
  Dans le fichier `config.py`, vous devez mettre à jour les informations de connexion à Neo4j dans la section suivante (autour des **lignes 83 à 85**) :

  ```python
  # Identifiants de la base de données Neo4j
  neo4j_uri: str = "neo4j+s://your-neo4j-uri"  # Mettez à jour avec votre URI Neo4j
  neo4j_user: str = "neo4j"                    # Mettez à jour avec votre nom d'utilisateur Neo4j
  neo4j_password: str = "your-password"        # Mettez à jour avec votre mot de passe Neo4j
  ```

5. **Ajoutez les données de test :**

Les **données de test** sont présentes dans la fonction `run_examples()` dans `main.py`. Cette section (autour de la **ligne 2166**) contient une variable `test_data` où vous pouvez modifier ou ajouter de nouvelles données :

```python
test_data = [  # Ligne 2166
    # Exemple de données pour un capteur de sol
    {
        "sensor_id": "SOIL_PROBE_001",
        "timestamp": "2024-03-15T08:30:00Z",
        "soil_moisture": 34.5,
        "soil_temperature": 18.2,
        "ph_level": 6.8,
        "nitrogen": 45,
        "phosphorus": 28,
        "potassium": 180,
        "parcel": "PARCEL_ALPHA",
        "farm": "FERME_DU_VAL"
    },
    # Ajoutez d'autres enregistrements ici
]
```

Il vous suffit de remplacer ou d'ajouter vos données de test selon vos besoins.

6. **Exécutez l'application :**

Une fois que vous avez mis à jour `run.py` avec vos identifiants Neo4j et vos données de test, vous pouvez exécuter l'application :

```bash
python pre-run.py
python main.py
```

