# Analyse des commentaires d'hôtels — Ottawa

Ce projet réalise une analyse textuelle complète des commentaires clients d'hôtels de la ville d'Ottawa. Il couvre le prétraitement du texte, l'analyse de fréquences, la modélisation thématique (LDA) et l'analyse de sentiment.

---

## Données

- **Fichier source :** `all_comments_ottawa.csv`
- **Volume :** 5 946 commentaires uniques (aucun doublon)
- **Langue :** Bilingue (français et anglais)
- **Colonne :** `comments`

---

## Structure du notebook

### 1. Chargement et exploration des données
Chargement du CSV avec `pandas`, affichage des premières lignes, inspection du type de données et vérification des doublons.

### 2. Prétraitement du texte
Le pipeline de nettoyage applique les étapes suivantes à chaque commentaire :
- Mise en minuscules
- Suppression des caractères spéciaux (conservation des lettres accentuées)
- Normalisation des espaces
- Tokenisation avec `nltk.word_tokenize`
- Suppression des stopwords français **et** anglais (via `nltk.corpus.stopwords`)
- Lemmatisation avec `WordNetLemmatizer`

### 3. Analyse de fréquences
- Calcul des 10 mots les plus fréquents et visualisation en barplot (`seaborn`)
- Extraction des **10 bigrammes** les plus fréquents via `CountVectorizer(ngram_range=(2,2))`
- Extraction des **10 trigrammes** les plus fréquents via `CountVectorizer(ngram_range=(3,3))`

Exemples de bigrammes dominants : `hotel promenade`, `byward market`, `parliament hill`, `rideau canal`.

### 4. Modélisation thématique (LDA)
- Vectorisation TF-IDF (`max_features=1000`, `max_df=0.95`, `min_df=2`)
- Entraînement d'un modèle **Latent Dirichlet Allocation** avec **5 thèmes** (`sklearn.decomposition.LatentDirichletAllocation`)
- Affichage des 10 mots les plus représentatifs par thème

### 5. Analyse de sentiment
- Calcul du score de polarité via `TextBlob`
- Classification en trois catégories :
  - **Positive** : score > 0.1
  - **Negative** : score < -0.1
  - **Neutral** : score entre -0.1 et 0.1
- Visualisation de la distribution des sentiments

### 6. Nuage de mots
Génération d'un **Word Cloud** des 100 termes les plus fréquents après prétraitement.

---

## Prérequis

**Python 3.12+**

Installez les dépendances avec :

```bash
pip install pandas numpy nltk scikit-learn textblob wordcloud matplotlib seaborn
```

Puis téléchargez les ressources NLTK nécessaires :

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

---

## Utilisation

1. Placez le fichier `all_comments_ottawa.csv` dans le même répertoire que le notebook.
2. Lancez Jupyter Notebook :
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. Exécutez les cellules dans l'ordre.

---

## Dépendances

| Bibliothèque | Usage |
|---|---|
| `pandas` | Manipulation des données |
| `numpy` | Calculs numériques |
| `nltk` | Tokenisation, stopwords, lemmatisation |
| `scikit-learn` | TF-IDF, CountVectorizer, LDA |
| `textblob` | Analyse de sentiment |
| `wordcloud` | Nuage de mots |
| `matplotlib` | Visualisations |
| `seaborn` | Visualisations statistiques |
