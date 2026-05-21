# Projet ML : Prédiction du Diabète avec Random Forest

---

## Vue d'ensemble

Ce projet porte sur l'implémentation, l'optimisation et la comparaison d'un modèle **Random Forest** pour la prédiction du diabète. Il s'appuie sur le dataset **Pima Indians Diabetes**, un problème de classification binaire visant à prédire la présence (1) ou l'absence (0) du diabète.

| Caractéristique | Détail |
|---|---|
| Dataset | Pima Indians Diabetes |
| Observations | 768 |
| Variables d'entrée | 8 variables médicales |
| Variable cible | `Outcome` (0 = non diabétique, 1 = diabétique) |
| Type de problème | Classification binaire supervisée |

---

## Variables du dataset

Les 8 variables explicatives sont toutes numériques et représentent des mesures médicales réelles :

- **Pregnancies** — Nombre de grossesses
- **Glucose** — Taux de glucose dans le sang
- **BloodPressure** — Pression artérielle
- **SkinThickness** — Épaisseur de la peau
- **Insulin** — Taux d'insuline
- **BMI** — Indice de masse corporelle
- **DiabetesPedigreeFunction** — Score génétique de risque
- **Age** — Âge du patient

---

## Architecture du projet

Le projet est structuré en 7 sections principales :

### Section 0 — Présentation du modèle Random Forest

Introduction théorique au modèle Random Forest, couvrant :

- **Fondations** : Arbre de décision CART, Bootstrap, Bagging, et comparaison des algorithmes ID3 / C4.5 / CART
- **Architecture** : Fonctionnement de la forêt aléatoire, Feature Randomness, dé-corrélation des arbres
- **Évaluation** : Erreur OOB (Out-Of-Bag), Feature Importances
- **Hyperparamètres clés** : `n_estimators`, `max_depth`, `max_features`, `oob_score`

La Random Forest repose sur deux idées fondamentales : le **bootstrap** (rééchantillonnage avec remise) et le **bagging** (Bootstrap Aggregating). Chaque arbre est construit selon l'algorithme CART, minimisant l'indice de Gini. Une randomisation supplémentaire lors de la sélection des variables à chaque split permet de réduire la corrélation entre les arbres.

---

### Section 1 — Chargement et Prétraitement des données

**Source des données** : Dataset chargé directement depuis le dépôt Plotly/GitHub.

**Étapes de prétraitement** :

1. **Détection des valeurs invalides** : Les variables `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin` et `BMI` ne peuvent pas être égales à 0 chez une personne vivante. Ces zéros sont traités comme des données manquantes.

2. **Imputation** :
   - Variables à faible taux de manquants (`Glucose`, `BloodPressure`, `BMI` < 5%) → **imputation par la médiane** (robuste aux outliers)
   - `SkinThickness` → **KNN Imputer** (k=5) pour préserver la structure des données
   - `Insulin` (~50% de valeurs manquantes) → traitée avec prudence

3. **Standardisation** : `StandardScaler` appliqué (moyenne = 0, écart-type = 1), nécessaire pour la Régression Logistique mais pas pour Random Forest et les arbres de décision.

4. **Split train/test** : 75% / 25%, avec stratification pour conserver les proportions de classes.

5. **Analyse de corrélation** :
   - `Glucose` est la variable la plus corrélée avec `Outcome` (r = 0.51)
   - Corrélation notable entre `Glucose` ↔ `Insulin` (0.64) et `SkinThickness` ↔ `BMI` (0.63), indiquant une redondance partielle

---

### Section 2 — Implémentation de base

Entraînement d'un premier modèle Random Forest avec les paramètres par défaut (`n_estimators=100`, `oob_score=True`) pour établir une baseline.

**Exploration de l'impact des variables** : Trois variantes du modèle sont testées pour évaluer l'effet de la redondance :
- Modèle avec toutes les features
- Modèle sans `Insulin`
- Modèle sans `Insulin` et `SkinThickness`

**Résultat** : La suppression simultanée d'`Insulin` et de `SkinThickness` améliore le rappel, suggérant que leur combinaison introduit du bruit ou une redondance avec `BMI` et `Glucose`.

---

### Section 3 — Feature Importance

Analyse de l'importance des variables dans le modèle final (sans `Insulin` et `SkinThickness`) :

| Rang | Variable | Importance |
|---|---|---|
| 1 | **Glucose** | La plus élevée |
| 2 | **BMI** | Élevée |
| 3 | **Age** | Modérée |
| 4 | **DiabetesPedigreeFunction** | Modérée |
| 5 | Autres | Plus faible |

`Glucose` est le facteur le plus influent dans la prédiction du diabète.

---

### Section 4 — Optimisation de `n_estimators`

Évaluation de l'erreur OOB pour `n_estimators` allant de 10 à 500 arbres.

**Observations** :
- L'erreur diminue fortement entre 10 et ~100 arbres
- À partir de 100 arbres, l'erreur se stabilise autour de 0.225
- **Choix optimal : 100 arbres** — meilleur compromis performance/coût de calcul

---

### Section 5 — Optimisation de `max_depth` et `max_features`

**`max_depth`** : Profondeurs testées : 3, 5, 10, None (illimité)
- Une profondeur trop grande → overfitting (train recall → 1.0, test recall stagne)
- **Choix optimal : `max_depth=3`** — écart minimal entre train et test (meilleur compromis biais/variance)

**`max_features`** : Valeurs testées : `'sqrt'`, `'log2'`, 1, 2, 3, 4, nombre total
- **Choix optimal : `max_features=3`** — meilleur recall (0.4925) grâce à un bon équilibre entre diversité des arbres et richesse d'information

**`class_weight`** : L'utilisation de `class_weight='balanced'` améliore significativement le recall, qui passe de **0.4925 à 0.7164**, en rééquilibrant les poids entre classes pour mieux détecter la classe minoritaire (diabétiques).

---

### Section 6 — Comparaison des modèles

Quatre modèles évalués sur le même jeu de test :

| Modèle | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression | 0.7188 | **0.6182** | 0.5075 | 0.5574 |
| Naive Bayes | 0.7292 | 0.6154 | 0.5970 | 0.6061 |
| **Random Forest** | **0.7292** | 0.5926 | **0.7164** | **0.6486** |

**Analyse** :
- **Logistic Regression** : modèle prudent, précision élevée mais recall faible (rate beaucoup de vrais positifs)
- **Naive Bayes** : bon compromis global, modèle simple mais efficace
- **Random Forest** : meilleur recall et F1-score, détecte le mieux les cas diabétiques — priorité dans un contexte médical

---

### Section 7 — Analyse finale

**Conclusion** : La Random Forest est le modèle le plus adapté au problème étudié. Dans un contexte médical, le **recall** (capacité à détecter les vrais positifs) est la métrique la plus critique pour éviter les faux négatifs (patients diabétiques non détectés). La Random Forest optimisée atteint un recall de **0.7164**, nettement supérieur aux modèles concurrents.

---

## Métriques d'évaluation utilisées

| Métrique | Description |
|---|---|
| **Accuracy** | Proportion de prédictions correctes |
| **Precision** | Proportion de prédictions positives correctes |
| **Recall** | Capacité à détecter les vrais positifs |
| **F1-score** | Moyenne harmonique entre précision et rappel |
| **OOB Score** | Estimation interne de performance (propre à Random Forest) |

---

## Hyperparamètres finaux retenus

```python
RandomForestClassifier(
    n_estimators  = 100,
    max_depth     = 3,
    max_features  = 3,
    class_weight  = 'balanced',
    oob_score     = True,
    random_state  = 42
)
```

**Features utilisées** (6 sur 8, après suppression d'`Insulin` et `SkinThickness`) :
`Pregnancies`, `Glucose`, `BloodPressure`, `BMI`, `DiabetesPedigreeFunction`, `Age`

---

## Stack technique

- **Python** avec `scikit-learn`, `pandas`, `numpy`
- **Visualisation** : `matplotlib`, `seaborn`
- **Modèles** : `RandomForestClassifier`, `DecisionTreeClassifier`, `LogisticRegression`, `GaussianNB`
- **Prétraitement** : `StandardScaler`, `KNNImputer`, `train_test_split`
