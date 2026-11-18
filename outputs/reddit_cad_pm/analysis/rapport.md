# Analyse de reddit_cad_pm — 4 novembre 2026

Le modèle apprend un signal d'abus, mais la politique automatise très peu de décisions aux coûts et exigences choisis. Cette expérience ne permet pas d'attribuer le problème uniquement à l'encodeur. Les représentations figées, l'apprentissage par régression des récompenses, la construction du contexte, la calibration et les contraintes de décision sont à distinguer expérimentalement.

## Résultats vérifiés

Les événements ont été rapprochés du fichier CAD local : identifiants, indices de messages, labels et appartenance aux partitions officielles concordent. Les conversations de validation et de test sont disjointes. Les coûts et faux négatifs ont été recalculés à partir des événements.

| Mesure sur test | Message | Contexte |
|---|---:|---:|
| Messages | 5 307 | 5 307 |
| Positifs | 906 (17,07 %) | 906 (17,07 %) |
| Average precision | 0,4405 | 0,3309 |
| ROC AUC | 0,7724 | 0,6989 |
| Conservations automatiques | 47 (0,89 %) | 13 (0,24 %) |
| Suppressions automatiques | 0 | 0 |
| Positifs conservés incorrectement | 1 | 1 |
| Révision humaine | 99,11 % | 99,76 % |
| Coût moyen | 0,20294 | 0,20422 |
| Coût moyen de la révision entièrement humaine | 0,20000 | 0,20000 |

Le rappel du pipeline de 99,89 % compte les décisions humaines comme parfaites. Le F1 de suppression nul vient de l'absence de suppressions automatiques ; il ne décrit pas à lui seul le classement appris. L'average precision a pour référence descriptive la prévalence de 17,07 %. Les comparaisons brutes avec l'ancienne expérience Reddit à 1,78 % de positifs ne sont pas directement interprétables.

La politique coûte ici 1,47 % et 2,11 % de plus que la révision entièrement humaine. Ces écarts reposent sur une seule conservation incorrecte par système : un bootstrap par conversation (2 000 réplications, graine 13) donne des intervalles à 95 % de [-0,00219 ; 0,01386] et [-0,00065 ; 0,01486] pour la différence de coût moyen. Le point estimé est défavorable ; le classement économique n'est pas établi avec précision.

## Pourquoi les seuils bloquent

Avec un coût humain de 0,2 et un faux négatif de 25, un groupe de conservations doit avoir au plus 0,8 % de positifs pour être au moins aussi rentable que la révision humaine. Une seule erreur coûte autant que 125 révisions. La validation de seuils contient 2 291 messages et 487 positifs. Le plus grand groupe de faibles scores satisfaisant cette contrainte contient 28 messages pour `message` et 3 pour `context`. Le premier positif arrive aux rangs 29 et 4. Aucun groupe plus grand ne redevient rentable dans les candidats examinés par le programme.

Les cibles de rappel 99 %, 98 %, 97 % et 95 % produisent exactement les mêmes seuils. Dans cette expérience, la contrainte économique bloque avant la contrainte de rappel. Abaisser seulement le rappel ne résout donc pas le problème.

Pour les suppressions, le code exige une probabilité d'au moins 0,5, un groupe d'au moins 100 messages, une précision empirique de 95 % et un coût au plus égal à celui des humains. Pour `message`, la meilleure précision d'un groupe admissible en taille et en probabilité n'est que 75,42 %. Pour `context`, seuls 21 messages atteignent une probabilité de 0,5 sur cette validation : le minimum de 100 est impossible. Le seuil `remove_min=2` désactive cette action.

Les seuils économiques théoriques, si les probabilités étaient exactes, sont p < 0,008 pour conserver et p > 0,8 pour supprimer. Le programme utilise toutefois des seuils empiriques de groupe, pas directement ces seuils théoriques. Aucun message de test n'a p < 0,008 ; le minimum prédit est d'environ 2,75 % pour `message`. Les probabilités des faibles scores ne permettent donc pas d'identifier clairement une grande zone sûre.

## Encodeur et apprentissage

Le checkpoint déclaré est `bert-base-uncased-reddit`. Son nom ne permet pas de déterminer s'il a été entraîné sur une tâche de toxicité ou seulement sur du texte Reddit. Il n'est pas disponible dans cet environnement au chemin du cluster, donc son historique d'entraînement n'a pas pu être vérifié.

Le code utilise `AutoModel`, `eval()` et `torch.no_grad()` : l'encodeur reste figé. Les représentations sont des moyennes des états des tokens non masqués. Seul le réseau de récompenses NeuralUCB est entraîné sur CAD. Les 10 époques et le learning rate de 0,001 concernent ce réseau, pas BERT. L'apprentissage minimise une erreur quadratique sur trois récompenses, avec un poids positif de 3. Un entraînement supervisé direct de la probabilité d'abus constitue une comparaison utile ; son avantage reste à mesurer.

## Contexte et calibration

La dégradation avec contexte existe aussi en validation de seuils : AP 0,5524 contre 0,3583 et AUC 0,7958 contre 0,6715. C'est un résultat sur cette configuration et cette graine, pas une démonstration que le contexte est inutile.

Le chargeur prend les trois messages précédents du fil par ordre chronologique ; il ne suit pas les liens de réponse. Le pooling mélange les tokens du message cible et de cet historique. Des messages d'une autre branche peuvent être inclus et l'information propre à la cible peut être diluée. Des pistes sont de suivre le parent et les ancêtres, d'encoder séparément cible et contexte, ou de conserver le contexte dans l'attention mais de calculer le pooling seulement sur les tokens cibles.

Les deux moitiés de validation diffèrent fortement : 14,01 % de positifs pour la calibration (313/2 234), 21,26 % pour la sélection des seuils (487/2 291), contre 17,07 % au test. Les probabilités moyennes sur la validation de seuils sont 16,39 % et 16,83 %, en dessous des 21,26 % observés. Cela justifie une validation plus stable par groupes de conversations, en conservant les partitions officielles et en évitant de réutiliser les mêmes observations pour toutes les étapes de sélection.

## Comparaisons conseillées

1. Garder `message` comme référence de travail et comparer le poids positif 1 au poids 3, sans changer les coûts. CAD contient 17,76 % de positifs en entraînement, contre environ 1,8 % dans l'ancienne expérience : le poids 3 n'est pas automatiquement adapté.
2. Comparer LinUCB et une régression logistique sur les mêmes représentations, aux mêmes partitions et aux mêmes exigences de décision. Cela aide à distinguer un problème de tête d'apprentissage d'un problème de représentations.
3. Fine-tuner l'encodeur actuel sur l'abus CAD, uniquement à partir du train, avec une séparation interne au train pour le choix du checkpoint. Point de départ expérimental : 2e-5, batch effectif 16 ou 32, 3 époques, puis sélection sur validation interne. Ce réglage n'est pas applicable au réseau actuel sans ajouter le fine-tuning de l'encodeur.
4. Comparer ensuite un encodeur DeBERTa-v3-base fine-tuné de la même façon, ou un modèle spécialisé en toxicité comme référence externe. Aucun gain sur CAD n'est présumé.
5. Revenir au contexte avec une construction suivant les réponses et une représentation distinguant la cible de son historique.

Choisir sur validation selon la couverture à contraintes fixes, le coût, l'average precision et le risque des faibles scores. Évaluer la variabilité sur plusieurs graines. Les changements motivés par l'analyse du test nécessitent ensuite une évaluation indépendante pour confirmer les gains.

Un modèle spécialisé chargé par `AutoModel` fournit son encodeur ; sa tête de classification n'est pas utilisée par ce programme. Pour comparer ses scores de toxicité natifs, il faut préserver sa tête, puis calibrer ses sorties sur CAD.

## Sources

- [CAD, article et taxonomie](https://aclanthology.org/2021.naacl-main.182/) : annotations contextuelles, catégories, partitions et baselines. Les scores multiétiquettes de l'article ne se comparent pas directement à cette cible binaire.
- [DeBERTa-v3-base, fiche officielle](https://huggingface.co/microsoft/deberta-v3-base) : encodeur candidat à fine-tuner.
- [Detoxify, dépôt officiel](https://github.com/unitaryai/detoxify) : modèles de toxicité Jigsaw, à distinguer de la définition CAD de l'abus.

Fichiers complémentaires : [graphiques](diagnostics.png), [mesures détaillées](diagnostics.json), [balayage des seuils sur validation](validation_tradeoffs_extended.csv).
