SmartGraph-Backend - Description et Guide d'Utilisation

Ce projet fait partie du projet SmartGraph et utilise Quarkus, un framework Java ultra-rapide, conçu pour les applications microservices et les environnements Cloud natifs. Si tu souhaites en savoir plus sur Quarkus, tu peux visiter son site officiel : https://quarkus.io/
.

Exécution de l'application en mode développement

Dans le cadre du projet SmartGraph, tu peux lancer l’application en mode développement, ce qui permet de faire du live coding. Pour cela, utilise la commande suivante dans ton terminal :

./mvnw quarkus:dev


Remarque: Quarkus propose une interface de développement accessible uniquement en mode dev à l’adresse http://localhost:8080/q/dev/
.

Emballage et exécution de l’application

L’application peut être emballée (packagée) en utilisant la commande suivante :

./mvnw package


Cela génère un fichier quarkus-run.jar dans le répertoire target/quarkus-app/. Note que ce fichier n'est pas un "uber-jar" : les dépendances sont placées dans un dossier séparé, target/quarkus-app/lib/. Une fois l’application packagée, tu peux l’exécuter avec la commande suivante :

java -jar target/quarkus-app/quarkus-run.jar


Si tu veux créer un uber-jar (un fichier JAR autonome avec toutes les dépendances intégrées), utilise cette commande :

./mvnw package -Dquarkus.package.jar.type=uber-jar


Le fichier uber-jar généré peut ensuite être lancé avec :

java -jar target/*-runner.jar


Création d’un exécutable natif

Pour SmartGraph, tu peux créer un exécutable natif en utilisant la commande suivante :

./mvnw package -Dnative


Si tu n’as pas GraalVM installé, tu peux utiliser un conteneur Docker pour créer l’exécutable natif avec cette commande :

./mvnw package -Dnative -Dquarkus.native.container-build=true


Une fois l'exécutable natif créé, tu peux le lancer avec la commande suivante :

./target/smartgraph-backend-1.0.0-SNAPSHOT-runner


Pour plus d’informations sur la création d’exécutables natifs, consulte ce guide : https://quarkus.io/guides/maven-tooling
.

Guides associés

Client Neo4j (guide
) : Connecter l’application SmartGraph à une base de données graphique Neo4j.

REST Jackson (guide
) : Prise en charge de la sérialisation JSON avec Jackson pour les services REST de Quarkus. Cette extension n’est pas compatible avec l’extension quarkus-resteasy ou les extensions qui en dépendent.

Code fourni

Services REST

L’application SmartGraph inclut des services REST pour simplifier la création d’API Web. Pour démarrer facilement avec ces services, consulte la section dédiée du guide Quarkus sur le JAX-RS réactif : Getting started with Reactive JAX-RS.