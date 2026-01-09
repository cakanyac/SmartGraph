SmartGraph

SmartGraph est une application qui permet de gérer et d’intégrer des données agricoles et robotiques en utilisant des graphes de connaissances et des ontologies sémantiques. Elle collecte des données provenant de capteurs IoT, drones et robots agricoles et les rend interrogeables pour faciliter la prise de décision.

          Table des Matières
          
          Présentation du projet
          
          Architecture
          
          Installation
          
          Backend
          
          Frontend
          
          Tests
          
          Déploiement

          Contributions

Présentation du projet

SmartGraph est une plateforme permettant d’intégrer des données agricoles provenant de diverses sources (capteurs IoT, robots, drones). Le projet utilise des graphes de connaissances pour rendre ces données exploitables et interrogeables.

Architecture

Le projet se compose de deux modules principaux :

Backend (SmartGraph-Backend) : API RESTful construite avec Quarkus, qui interagit avec la base de données Neo4j.

Frontend (SmartGraphFrontend) : Application web construite avec Angular pour la visualisation et l'interaction avec l'API backend.

Installation
1. Cloner le repository
git clone https://github.com/votre-utilisateur/smartgraph.git
cd smartgraph

2. Backend

Pour installer les dépendances et démarrer l'API :

cd smartgraph-backend
./mvnw clean install
./mvnw quarkus:dev

3. Frontend

Pour installer les dépendances et démarrer le serveur de développement Angular :

cd smartgraph-frontend
npm install
ng serve


Le frontend sera accessible à l'adresse http://localhost:4200.

Backend

L'API backend expose des points d'API pour gérer les données des parcelles, capteurs, robots, et observations. Elle est construite avec Quarkus et utilise Neo4j pour stocker les données en graphe.

Frontend

L'application frontend permet de visualiser et gérer les données via une interface construite avec Angular. Elle se connecte à l'API backend pour récupérer les données des capteurs, robots et parcelles.

Tests
1. Tests unitaires

Pour exécuter les tests unitaires du backend :

./mvnw test


Pour exécuter les tests du frontend :

ng test

2. Tests de bout en bout

Pour exécuter des tests de bout en bout pour le frontend :

ng e2e

Déploiement
1. Déployer le Backend

Pour générer le JAR du backend pour la production :

./mvnw package
java -jar target/quarkus-run.jar

2. Déployer le Frontend

Pour créer la version de production du frontend :

ng build --prod


Les fichiers sont générés dans le dossier dist/smartgraph-frontend/.

Contributions

Les contributions sont les bienvenues. Pour contribuer, vous pouvez :

          Forker le repository.
          
          Créer une nouvelle branche.
          
          Soumettre une pull request.
