SmartGraphFrontend - Description et Guide d'Utilisation

Ce projet fait partie du projet SmartGraph et utilise Angular, un framework JavaScript pour la construction d'applications web modernes et dynamiques. Développé avec Angular CLI (version 21.0.2), ce projet permet de gérer les fonctionnalités front-end du projet SmartGraph, en facilitant la création de composants, la gestion du serveur de développement et les tests.

Serveur de développement

Pour démarrer le serveur local de développement, exécute cette commande :

ng serve  


Une fois le serveur lancé, ouvre ton navigateur et va à l'adresse suivante :
http://localhost:4200/
L'application se recharge automatiquement chaque fois que tu modifies un fichier source.

Génération de code

Angular CLI inclut des outils puissants pour la génération automatique de code. Par exemple, pour générer un nouveau composant, utilise la commande :

ng generate component component-name  


Pour obtenir une liste complète des schémas disponibles (comme composants, directives ou pipes), utilise cette commande :

ng generate --help  

Compilation du projet

Pour compiler ton projet et obtenir le résultat final, utilise la commande suivante :

ng build  


Cela génère les artefacts du build dans le répertoire dist/. Par défaut, la compilation en mode production optimise l'application pour la performance et la vitesse.

Exécution des tests unitaires

Pour exécuter les tests unitaires avec Vitest (le test runner par défaut dans ce projet), utilise cette commande :

ng test  

Exécution des tests de bout en bout (e2e)

Pour effectuer des tests de bout en bout, utilise la commande suivante :

ng e2e  


Angular CLI ne propose pas de framework de test end-to-end par défaut, tu peux donc choisir celui qui correspond à tes besoins.

Ressources supplémentaires

Pour plus d'informations sur l'utilisation de Angular CLI, y compris une référence détaillée des commandes, consulte la page suivante :
Angular CLI Overview and Command Reference.