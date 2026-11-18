**Analyse de `reddit_calibrated_pm` — 21 octobre 2026**

Le contexte améliore modestement la discrimination et augmente la couverture automatique, mais les résultats ne montrent pas encore une réduction importante de la charge humaine. Plus de 93 % des messages passent chez les humains. Aucun message n'est supprimé automatiquement. Le rappel du pipeline sur test est inférieur à la cible de 98 % pour les deux systèmes, particulièrement pour `context`.

L'expérience utilise 2 082 785 messages, dont 1 463 754 pour l'apprentissage. La validation est divisée en calibration (152 734 messages) et sélection des seuils (154 773). Le test contient 311 524 messages, répartis entre 78 530 conversations, dont 5 549 messages étiquetés à supprimer, soit 1,781 %. Les deux fichiers d'événements portent sur les mêmes messages et labels. Les décomptes d'actions, erreurs et coûts ont été recalculés à partir des événements et concordent avec `summary.json`.

| Mesure sur test | `message` | `context` |
|---|---:|---:|
| Conservation automatique | 17 134 (5,500 %) | 21 172 (6,796 %) |
| Révision humaine | 294 390 (94,500 %) | 290 352 (93,204 %) |
| Suppression automatique | 0 | 0 |
| Messages à supprimer conservés automatiquement | 113 | 140 |
| Erreurs parmi les conservations automatiques | 0,660 % | 0,661 % |
| Rappel du pipeline, humains parfaits | 97,964 % | 97,477 % |
| Coût moyen | 0,198068 | 0,197643 |
| Coût total | 61 703,0 | 61 570,4 |
| Économie contre révision entièrement humaine | 0,966 % | 1,179 % |
| Average precision | 0,04596 | 0,05053 |
| ROC AUC, recalculée | 0,67495 | 0,69293 |

La couverture est ici entièrement de la conservation automatique. Les valeurs nulles de précision et F1 de suppression sont des conventions du code en l'absence de suppression. La précision de suppression n'est donc pas estimable dans cette expérience. L'accuracy de 99,34 % ne porte que sur les décisions automatiques et se lit avec la faible couverture. Le rappel du pipeline compte comme correctement traités tous les positifs envoyés à l'humain : il suppose des humains parfaits et ne mesure pas le rappel d'un modèle de suppression automatique.

**Le gain économique reste faible et son avantage entre systèmes est incertain.**

Les coûts sont 0,2 par révision humaine, 1 par suppression incorrecte et 25 par conservation incorrecte. Le coût d'une révision entièrement humaine serait de 62 304,8. Ainsi :

- `message` évite 3 426,8 unités de révision, mais ajoute 2 825 unités d'erreurs : économie nette de 601,8.
- `context` évite 4 234,4 unités de révision, mais ajoute 3 500 unités d'erreurs : économie nette de 734,4.
- Passer de `message` à `context` évite 4 038 révisions, soit 807,6 unités, et ajoute 27 erreurs, soit 675 unités. Le gain net supplémentaire est seulement de 132,6 unités, ou 0,213 % du coût entièrement humain.

La couverture augmente de 1,296 point, soit 23,6 % relativement à celle de `message`. La baisse de révision humaine est de 1,296 point. Les coûts d'encodage BERT et l'éventuel surcoût de calcul du contexte ne sont pas inclus dans cette fonction de coût.

Un bootstrap apparié par conversation, avec 2 000 réplications et la graine 13, donne un intervalle percentile à 95 % de [-0,00248 ; +0,00173] pour le coût moyen de `context` moins celui de `message`. Cet intervalle inclut zéro : le classement économique entre les deux n'est pas établi par ce test. Ces intervalles décrivent l'incertitude d'échantillonnage avec les modèles et seuils fixés, sans inclure la variabilité d'entraînement ou de choix d'hyperparamètres.

**La cible de rappel est une contrainte empirique de validation.**

Sur la partie de validation utilisée pour les seuils, les deux systèmes conservent incorrectement 59 positifs sur 2 994 et atteignent 98,029 % de rappel, juste au-dessus de la cible. Sur test, ils passent respectivement à 97,964 % et 97,477 %. Les écarts à la cible sont de -0,036 et -0,523 point.

Les intervalles bootstrap par conversation sont [97,570 % ; 98,331 %] pour `message` et [97,004 % ; 97,921 %] pour `context`. Le recul du second ne semble donc pas être seulement un effet d'arrondi autour de 98 %. Le réglage `alpha=0` utilise directement les probabilités et n'applique aucune marge d'incertitude. La recherche des seuils n'impose pas non plus de borne statistique de rappel.

**Les premiers messages constituent le principal point faible.**

| Sous-groupe du test | Conservation `message` | Conservation `context` | Erreurs `message` | Erreurs `context` | Rappel `message` | Rappel `context` |
|---|---:|---:|---:|---:|---:|---:|
| Premier message (`message_index=0`) | 4 137 | 6 955 | 34 | 63 | 97,826 % | 95,972 % |
| Messages suivants | 12 997 | 14 217 | 79 | 77 | 98,018 % | 98,068 % |

Le système `context` ajoute 29 erreurs sur les premiers messages et en évite 2 sur les suivants : cela explique les 27 erreurs supplémentaires au total. Au premier message, aucun historique n'est disponible ; les différences entre systèmes proviennent alors de leurs modèles de récompense et calibrateurs ajustés séparément.

Cette faiblesse était déjà visible en validation : le rappel de `context` y est de 95,862 % sur les premiers messages, contre 98,917 % sur les suivants. Son groupe de premiers messages conservés a un taux d'erreur de 1,018 % en validation et 0,906 % sur test, supérieur au seuil de rentabilité de 0,8 % (= 0,2 / 25). Son automatisation des premiers messages perd donc 192,8 unités en validation et 184,0 sur test par rapport à leur révision humaine. La contrainte de coût actuelle est globale à l'ensemble des conservations et laisse ce sous-groupe être compensé par les autres messages.

Cela suggère d'étudier sur validation une calibration ou des seuils distincts selon la présence d'historique, avec contraintes de rappel et de coût par groupe. Cela constitue une piste à valider ; cette analyse ne démontre pas la performance future d'une telle politique.

**La calibration moyenne est correcte, la séparation des classes reste limitée.**

L'average precision progresse de 0,04596 à 0,05053 avec le contexte, soit environ 10 % relativement. La prévalence de référence est de 0,01781. Les AUC sont de 0,675 et 0,693. Il existe donc un signal utile, mais insuffisant pour une automatisation importante aux exigences retenues.

Les probabilités moyennes prédites sont proches de 1,864 % pour les deux systèmes, contre 1,781 % de positifs observés. La calibration par déciles est assez proche de la diagonale. Les Brier scores de 0,017322 et 0,017286 ne gagnent cependant qu'environ 1,0 % et 1,2 % sur une prédiction constante égale à la prévalence de test (0,017495, référence descriptive). Une calibration moyenne correcte peut masquer les défauts de calibration par sous-groupe et aux probabilités extrêmes.

Sur la validation servant aux seuils, les probabilités supérieures ou égales à 0,5 concernent seulement 7 messages pour `message` et 21 pour `context`, avec 14,3 % de positifs dans chaque groupe. Cela ne fournit aucun groupe admissible de 100 suppressions à 95 % de précision. Le seuil `remove_min=2` désactive explicitement cette action. Sur test, les 100 probabilités les plus élevées donnent seulement 16 % et 21 % de précision : le problème dépasse le seul choix d'un seuil de suppression.

**Diagnostics complémentaires calculés exclusivement sur validation.**

Les CSV d'origine ne contiennent qu'une cible de rappel, 0,98. Le fichier `validation_tradeoffs_extended.csv` ajoute un balayage des cibles et des objectifs `coverage` / `cost`, sans réentraîner BERT, sans changer la politique d'origine et sans appliquer ces nouveaux seuils au test.

| Cible de rappel sur validation | Couverture `message` | Couverture `context` |
|---|---:|---:|
| 99 % | 2,72 % | 3,53 % |
| 98 % | 5,46 % | 6,84 % |
| 97 % | 8,66 % | 9,96 % |
| 95 % | 14,33 % | 15,42 % |

Ces lignes utilisent l'objectif de couverture. Abaisser la cible augmente l'automatisation en acceptant davantage d'erreurs. Ce n'est pas une amélioration à exigence de rappel constante. L'objectif `cost` peut aussi choisir d'autres seuils, mais il ne résout pas à lui seul le déficit de discrimination.

Les priorités expérimentales sont de traiter le sous-groupe sans historique, de sélectionner les seuils avec une marge de rappel validée par conversation, puis d'améliorer la discrimination du modèle avec des comparaisons contrôlées sur validation. Une nouvelle évaluation indépendante sera nécessaire après avoir choisi une modification à partir de cette analyse du test. Ces résultats concernent un seul split et une seule graine.

Les graphiques cumulatifs d'origine décrivent l'accumulation de décisions d'une politique fixe : le code ne met pas à jour le modèle pendant le test. Leur évolution n'est donc pas une courbe d'apprentissage en ligne.

Fichiers complémentaires : [graphiques](diagnostics.png), [diagnostics et intervalles bootstrap](diagnostics.json), [résultats par sous-groupe](subgroup_diagnostics.csv), [balayage sur validation](validation_tradeoffs_extended.csv), [calibration message](reliability_message.csv), [calibration contexte](reliability_context.csv).
