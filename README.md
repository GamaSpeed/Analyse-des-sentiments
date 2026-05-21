# Analyse des commentaires d'hôtels — Ottawa

Ce projet réalise une analyse textuelle complète des commentaires clients d'hôtels de la ville d'Ottawa. Il couvre la détection de langue, le prétraitement du texte, l'analyse de fréquences (unigrammes, bigrammes, trigrammes), la modélisation thématique (LDA) et une analyse de sentiment bilingue différenciée par langue.

---

## Données

- **Fichier source :** `all_comments_ottawa.csv`
- **Volume :** 5 946 commentaires uniques (aucun doublon, zéro valeur nulle)
- **Langue :** Bilingue — 5 035 commentaires en anglais (84,7 %), 910 en français (15,3 %), 1 exclu
- **Colonne :** `comments`

---

## Structure du notebook

### 1. Chargement et exploration des données
Chargement du CSV avec `pandas`, affichage des premières lignes via `df.head()`, inspection des types et comptage des doublons avec `df.duplicated().sum()`. Le DataFrame contient une seule colonne `comments` de type `object`, sans valeur nulle.

### 2. Détection de la langue
Détection automatique de la langue de chaque commentaire avec `langdetect` **avant tout prétraitement** (sur le texte brut). Les commentaires sont classés en `fr`, `en` ou `other`. Les résultats (84,7 % anglais, 15,3 % français) sont visualisés dans un barplot coloré par langue.

### 3. Prétraitement du texte
Le pipeline de nettoyage applique les étapes suivantes à chaque commentaire :
- Mise en minuscules
- Suppression des caractères spéciaux (conservation des lettres accentuées via regex `[^a-zà-ÿ\s]`)
- Normalisation des espaces
- Tokenisation avec `nltk.word_tokenize`
- Suppression des stopwords français **et** anglais (via `nltk.corpus.stopwords`), ainsi que des tokens de moins de 3 caractères
- Lemmatisation **par langue** avec `spaCy` (`fr_core_news_sm` pour le français, `en_core_web_sm` pour l'anglais)

### 4. Analyse de fréquences
- Calcul des **10 mots** les plus fréquents et visualisation en barplot (`seaborn`) — dominés par `hotel`, `room`, `stay`, `staff`, `location`, `clean`, `chambre`, `service`, `great`, `breakfast`
- Extraction des **10 bigrammes** les plus fréquents via `CountVectorizer(ngram_range=(2,2))` — ex. `hotel promenade`, `great location`, `room small`, `byward market`, `parliament hill`, `rideau canal`
- Extraction des **10 trigrammes** les plus fréquents via `CountVectorizer(ngram_range=(3,3))` — ex. `stay hotel promenade`, `view parliament building`, `first impression hotel`

### 5. Modélisation thématique (LDA)
- Vectorisation TF-IDF (`max_features=1000`, `max_df=0.95`, `min_df=2`)
- Entraînement d'un modèle **Latent Dirichlet Allocation** avec **5 thèmes** (`sklearn.decomposition.LatentDirichletAllocation`, `random_state=42`)
- Affichage des 10 mots les plus représentatifs par thème
- Le modèle est réutilisé ensuite par segment (positifs EN, négatifs EN, positifs FR, négatifs FR) via une fonction `run_lda` réutilisable

### 6. Analyse de sentiment bilingue
L'analyse est segmentée par langue pour éviter les biais introduits par les modèles monolingues.

**Anglais — TextBlob**
- Score de polarité calculé avec `TextBlob`
- Seuils : Positif > 0.05, Négatif < -0.05, Neutre sinon
- Résultats : **4 441 positifs (88,2 %)**, 372 neutres (7,4 %), 222 négatifs (4,4 %)

**Français — BERT Multilingual (nlptown)**
- Modèle `nlptown/bert-base-multilingual-uncased-sentiment` via `transformers.pipeline`
- Résultat en nombre d'étoiles (1–5) converti en Positif (≥ 4), Neutre (3) ou Négatif (≤ 2)
- Fallback sur `TextBlob` + `textblob_fr` si HuggingFace n'est pas disponible
- Résultats : **518 positifs (56,9 %)**, 201 neutres (21,0 %), 191 négatifs (22,1 %)

> **Signal clé :** L'écart de satisfaction entre anglophones (88,2 % positifs) et francophones (56,9 % positifs) est de **31,3 points**. Le taux d'insatisfaction francophone (22,1 %) est 5× supérieur à celui des anglophones (4,4 %).

### 7. Insights par segment
Pour chaque sous-groupe (avis positifs EN, négatifs EN, positifs FR, négatifs FR), trois visualisations réutilisables sont générées :
- **Top 20 mots** les plus fréquents (barplot)
- **Nuage de mots** (120 termes max, `WordCloud`)
- **Top 10 bigrammes** (barplot)
- Thèmes LDA spécifiques au segment (`run_lda`)

### 8. Nuage de mots global
Génération d'un **Word Cloud** des 100 termes les plus fréquents sur l'ensemble des commentaires prétraités.

---

## Prérequis

**Python 3.12+**

Installez les dépendances avec :

```bash
pip install pandas numpy nltk scikit-learn textblob textblob-fr wordcloud matplotlib seaborn langdetect transformers torch sentencepiece spacy
```

Téléchargez les ressources NLTK nécessaires :

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
```

Téléchargez les modèles spaCy :

```bash
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
```

---

## Utilisation

1. Placez le fichier `all_comments_ottawa.csv` dans le même répertoire que le notebook.
2. Lancez Jupyter Notebook :
   ```bash
   jupyter notebook notebook_nlp.ipynb
   ```
3. Exécutez les cellules dans l'ordre.

---

## Dépendances

| Bibliothèque | Usage |
|---|---|
| `pandas` | Manipulation des données |
| `numpy` | Calculs numériques |
| `nltk` | Tokenisation, stopwords, lemmatisation (fallback) |
| `spacy` | Lemmatisation par langue (EN / FR) |
| `langdetect` | Détection automatique de la langue |
| `scikit-learn` | TF-IDF, CountVectorizer, LDA |
| `textblob` | Analyse de sentiment anglais |
| `textblob-fr` | Fallback analyse de sentiment français |
| `transformers` | BERT Multilingual pour le sentiment français |
| `wordcloud` | Nuage de mots |
| `matplotlib` | Visualisations |
| `seaborn` | Visualisations statistiques |
