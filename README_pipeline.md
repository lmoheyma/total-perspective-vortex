# Pipeline de Preprocessing EEG avec Scikit-Learn

Cette pipeline complète permet de traiter des données EEG pour des tâches de classification BCI (Brain-Computer Interface).

## 🧠 Vue d'ensemble

La pipeline transforme des signaux EEG bruts en caractéristiques exploitables pour l'apprentissage automatique, en suivant les meilleures pratiques du domaine.

## 📋 Étapes de la Pipeline

### 1. **Preprocessing de Base (`EEGPreprocessor`)**
```python
# Filtrage des signaux
- Filtre haute-passe (1Hz) : Enlève les dérives lentes et DC offset
- Filtre basse-passe (50Hz) : Enlève le bruit haute fréquence  
- Filtre notch (50Hz) : Enlève l'interférence du réseau électrique
- Re-échantillonnage (optionnel) : Réduit la taille des données
```

**Pourquoi c'est important :**
- Les dérives lentes peuvent masquer l'activité neurale
- Le bruit haute fréquence pollue l'analyse
- L'interférence électrique (50/60Hz) est omniprésente

### 2. **Suppression d'Artefacts (`ArtifactRemover`)**
```python
# ICA (Independent Component Analysis)
- Décompose le signal en composantes indépendantes
- Identifie automatiquement les artefacts (clignements, mouvements)
- Supprime les composantes d'artefacts
```

**Pourquoi c'est important :**
- Les clignements d'yeux créent des artefacts massifs
- Les mouvements musculaires polluent le signal
- ICA sépare l'activité cérébrale des artefacts

### 3. **Extraction de Caractéristiques (`FeatureExtractor`)**

#### Caractéristiques Temporelles :
- **Moyenne** : Niveau d'activité général
- **Écart-type** : Variabilité du signal
- **Asymétrie (Skewness)** : Distribution des amplitudes
- **Kurtosis** : Présence de pics dans le signal

#### Caractéristiques Spectrales :
- **Delta (0.5-4Hz)** : Sommeil profond, méditation
- **Theta (4-8Hz)** : Relaxation, créativité
- **Alpha (8-13Hz)** : Relaxation éveillée, yeux fermés
- **Beta (13-30Hz)** : **Activité motrice, concentration**
- **Gamma (30-50Hz)** : Cognition élevée

#### Ratios de Puissance :
- **Alpha/Beta** : Balance relaxation/activation
- **Theta/Alpha** : États méditatifs vs éveillés

**Pourquoi ces caractéristiques :**
- Pour BCI motrice, Beta est cruciale (canaux C3/C4)
- Les ratios capturent l'état mental global
- Le fenêtrage (2s) équilibre résolution temporelle/stabilité

### 4. **Normalisation (`RobustScaler`)**
```python
# Standardisation robuste aux outliers
- Centre chaque caractéristique sur la médiane
- Mise à l'échelle par l'écart interquartile
- Résiste mieux aux artefacts résiduels que StandardScaler
```

### 5. **Sélection de Caractéristiques (`SelectKBest`)**
```python
# Test F pour classification
- Sélectionne les caractéristiques les plus discriminantes
- Réduit la malédiction de la dimensionnalité
- Améliore la généralisation
```

### 6. **Réduction de Dimensionnalité (`PCA`)**
```python
# Analyse en Composantes Principales
- Projette dans un espace de moindre dimension
- Garde 95% de la variance
- Débruite et compresse les données
```

## 🚀 Utilisation

### Installation des dépendances :
```bash
pip install mne scikit-learn numpy scipy matplotlib seaborn
```

### Utilisation basique :
```python
from preprocessing_pipeline import create_eeg_pipeline

# Créer la pipeline
pipeline = create_eeg_pipeline(include_ica=True, n_features=100)

# Appliquer sur vos données EEG (objets MNE Raw)
X = pipeline.fit_transform(raw_data, labels)

# Utiliser avec n'importe quel classifieur scikit-learn
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, labels)
```

### Démonstration complète :
```bash
python pipeline_demo.py
```

## 📊 Performance Attendue

Pour des tâches BCI standard (motor execution vs imagery) :
- **Précision typique** : 70-85%
- **Validation croisée** : Recommandée (5-fold)
- **Caractéristiques importantes** : Puissance Beta dans C3/C4

## 🔧 Paramètres Personnalisables

### Preprocessing :
- `l_freq`, `h_freq` : Fréquences de coupure des filtres
- `notch_freq` : Fréquence du filtre notch (50Hz Europe, 60Hz USA)
- `resample_freq` : Nouvelle fréquence d'échantillonnage

### Extraction de caractéristiques :
- `feature_types` : Types de caractéristiques à extraire
- `freq_bands` : Bandes de fréquences personnalisées
- `window_length` : Taille des fenêtres temporelles

### ICA :
- `n_components` : Nombre de composantes ICA
- `method` : Algorithme ICA ('fastica', 'infomax', 'picard')

## 🧪 Types d'Analyses Supportées

1. **Classification BCI** : Motor execution vs imagery
2. **Détection d'états** : Yeux ouverts vs fermés
3. **Analyse spectrale** : Puissance par bandes de fréquences
4. **Détection d'artefacts** : Identification automatique

## ⚠️ Considérations Importantes

### Qualité des données :
- Vérifiez l'impédance des électrodes
- Surveillez les artefacts pendant l'acquisition
- Calibrez régulièrement l'équipement

### Validation :
- Utilisez toujours une validation croisée
- Attention au sur-apprentissage avec peu de données
- Testez sur de nouveaux sujets pour la généralisation

### Optimisation :
- Ajustez les paramètres selon votre application
- L'ICA peut nécessiter plus de données pour être efficace
- Les bandes de fréquences peuvent être spécifiques à la tâche

## 📚 Références

- **MNE-Python** : https://mne.tools/
- **BCI2000** : http://www.bci2000.org
- **Physiological data dataset** : https://physionet.org/
- **Scikit-learn** : https://scikit-learn.org/

## 🤝 Contribution

Cette pipeline est optimisée pour les données BCI PhysioNet, mais peut être adaptée à d'autres types de données EEG. N'hésitez pas à modifier les paramètres selon vos besoins spécifiques.
