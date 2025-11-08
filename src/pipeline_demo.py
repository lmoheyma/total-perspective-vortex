import mne
import numpy as np
import os
from preprocessing_pipeline import create_eeg_pipeline, EEGPreprocessor, FeatureExtractor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_eeg_data(base_directory, subjects=['S001'], tasks=['R03', 'R04']):
    """
    Charge les données EEG pour les sujets et tâches spécifiés
    
    Paramètres:
    - base_directory: Répertoire contenant les dossiers patients
    - subjects: Liste des sujets à charger
    - tasks: Liste des tâches à charger
    
    Retourne:
    - raw_data: Liste des objets MNE Raw
    - labels: Labels correspondants
    - file_info: Informations sur les fichiers
    """
    
    # Noms de canaux standardisés
    channel_names_64 = [
        'Fp1', 'Fpz', 'Fp2',
        'AF7', 'AF3', 'AFz', 'AF4', 'AF8',
        'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
        'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
        'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
        'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
        'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
        'PO7', 'PO3', 'POz', 'PO4', 'PO8',
        'O1', 'Oz', 'O2', 'Iz', 'TP10', 'T9'
    ]
    
    raw_data = []
    labels = []
    file_info = []
    
    print("Chargement des données EEG...")
    
    for subject in subjects:
        subject_dir = os.path.join(base_directory, subject)
        
        if not os.path.exists(subject_dir):
            print(f"Attention: {subject_dir} n'existe pas")
            continue
            
        for task in tasks:
            # Chercher les fichiers correspondants à cette tâche
            task_files = []
            for file in os.listdir(subject_dir):
                if file.endswith('.edf') and task in file:
                    task_files.append(file)
            
            for file in task_files:
                filepath = os.path.join(subject_dir, file)
                
                try:
                    # Charger le fichier EDF
                    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
                    
                    # Renommer les canaux si nécessaire
                    if len(raw.ch_names) == len(channel_names_64):
                        rename_dict = dict(zip(raw.ch_names, channel_names_64))
                        raw.rename_channels(rename_dict)
                        raw.set_montage("standard_1005", verbose=False)
                    
                    raw_data.append(raw)
                    
                    # Créer le label basé sur la tâche
                    if 'R03' in file or 'R07' in file or 'R11' in file:
                        labels.append(0)  # Motor execution
                        task_name = "Motor Execution"
                    elif 'R04' in file or 'R08' in file or 'R12' in file:
                        labels.append(1)  # Motor imagery
                        task_name = "Motor Imagery"
                    else:
                        labels.append(-1)  # Autre
                        task_name = "Other"
                    
                    file_info.append({
                        'subject': subject,
                        'file': file,
                        'task': task_name,
                        'duration': raw.times[-1],
                        'sfreq': raw.info['sfreq'],
                        'n_channels': len(raw.ch_names)
                    })
                    
                    print(f"  ✓ {subject}/{file} - {task_name}")
                    
                except Exception as e:
                    print(f"  ✗ Erreur avec {subject}/{file}: {e}")
    
    print(f"\nTotal: {len(raw_data)} fichiers chargés")
    print(f"Labels: {np.bincount(labels)} (0=Motor Execution, 1=Motor Imagery)")
    
    return raw_data, np.array(labels), file_info

def analyze_pipeline_performance(X_train, X_test, y_train, y_test):
    """
    Analyse les performances de la pipeline avec différents classifieurs
    """
    print("\n=== ANALYSE DES PERFORMANCES ===")
    
    # Classifieurs à tester
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42),
        'SVM Linear': SVC(kernel='linear', random_state=42)
    }
    
    results = {}
    
    for name, clf in classifiers.items():
        print(f"\n{name}:")
        
        # Entraînement
        clf.fit(X_train, y_train)
        
        # Prédictions
        y_pred = clf.predict(X_test)
        
        # Scores
        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)
        
        print(f"  Score d'entraînement: {train_score:.3f}")
        print(f"  Score de test: {test_score:.3f}")
        
        # Validation croisée
        cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
        print(f"  Validation croisée: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        results[name] = {
            'train_score': train_score,
            'test_score': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred
        }
    
    return results

def visualize_features(X, y, feature_names=None, n_features=10):
    """
    Visualise les caractéristiques les plus importantes
    """
    from sklearn.feature_selection import SelectKBest, f_classif
    
    # Sélectionner les meilleures caractéristiques
    selector = SelectKBest(f_classif, k=n_features)
    selector.fit(X, y)
    
    # Obtenir les scores
    scores = selector.scores_
    
    # Créer les noms de caractéristiques si pas fournis
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(len(scores))]
    
    # Trier par score
    indices = np.argsort(scores)[::-1][:n_features]
    
    # Graphique
    plt.figure(figsize=(12, 6))
    plt.bar(range(n_features), scores[indices])
    plt.title(f'Top {n_features} caractéristiques les plus discriminantes')
    plt.xlabel('Caractéristiques')
    plt.ylabel('Score F')
    plt.xticks(range(n_features), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    """
    Fonction principale qui démontre l'utilisation complète de la pipeline
    """
    print("=== DÉMONSTRATION DE LA PIPELINE EEG ===\n")
    
    # 1. Charger les données
    base_directory = "/mnt/c/Users/louis/Downloads/archive/files/"
    
    # Charger quelques sujets et tâches pour la démo
    raw_data, labels, file_info = load_eeg_data(
        base_directory, 
        subjects=['S001', 'S002'],  # Premiers sujets
        tasks=['R03', 'R04']        # Motor execution vs imagery
    )
    
    if len(raw_data) == 0:
        print("Aucune donnée chargée. Vérifiez le chemin du répertoire.")
        return
    
    # 2. Créer la pipeline
    print("\nCréation de la pipeline de preprocessing...")
    pipeline = create_eeg_pipeline(include_ica=False, n_features=50)  # ICA désactivé pour la démo
    
    print("Étapes de la pipeline:")
    for i, (name, step) in enumerate(pipeline.steps, 1):
        print(f"  {i}. {name}: {type(step).__name__}")
    
    # 3. Appliquer la pipeline
    print("\nApplication de la pipeline...")
    
    try:
        # Transform les données (sans labels pour les étapes intermédiaires)
        # Appliquer preprocessing et extraction de caractéristiques
        preprocessor = pipeline.named_steps['preprocessing']
        feature_extractor = pipeline.named_steps['feature_extraction']
        scaler = pipeline.named_steps['scaler']
        feature_selector = pipeline.named_steps['feature_selection']
        
        # 1. Preprocessing
        print("  - Preprocessing (filtrage)...")
        preprocessed_data = preprocessor.fit_transform(raw_data)
        
        # 2. Extraction de caractéristiques
        print("  - Extraction de caractéristiques...")
        X_features = feature_extractor.fit_transform(preprocessed_data)
        print(f"    Forme après extraction: {X_features.shape}")
        
        # 3. Créer les labels pour chaque fenêtre
        print("  - Création des labels pour les fenêtres...")
        expanded_labels = []
        for i, raw in enumerate(raw_data):
            # Calculer le nombre de fenêtres pour ce fichier
            duration = raw.times[-1]
            window_length = 2.0
            n_windows = int(duration / window_length)
            
            # Répéter le label pour chaque fenêtre
            file_label = labels[i]
            expanded_labels.extend([file_label] * n_windows)
        
        expanded_labels = np.array(expanded_labels)
        
        # Ajuster si nécessaire
        if len(expanded_labels) != X_features.shape[0]:
            print(f"    Ajustement des labels: {len(expanded_labels)} -> {X_features.shape[0]}")
            if len(expanded_labels) > X_features.shape[0]:
                expanded_labels = expanded_labels[:X_features.shape[0]]
            else:
                diff = X_features.shape[0] - len(expanded_labels)
                last_label = expanded_labels[-1] if len(expanded_labels) > 0 else 0
                expanded_labels = np.concatenate([expanded_labels, [last_label] * diff])
        
        print(f"    Labels expanded: {np.bincount(expanded_labels)}")
        
        # 4. Normalisation
        print("  - Normalisation...")
        X_scaled = scaler.fit_transform(X_features)
        
        # 5. Sélection de caractéristiques
        print("  - Sélection de caractéristiques...")
        X = feature_selector.fit_transform(X_scaled, expanded_labels)
        
        print(f"Forme des données après preprocessing: {X.shape}")
        print(f"Nombre d'échantillons: {X.shape[0]}")
        print(f"Nombre de caractéristiques: {X.shape[1]}")
        
        # 4. Créer les labels pour chaque fenêtre
        # Chaque fichier EEG produit plusieurs fenêtres, on doit répéter les labels
        expanded_labels = []
        for i, raw in enumerate(raw_data):
            # Calculer le nombre de fenêtres pour ce fichier
            duration = raw.times[-1]
            window_length = 2.0
            n_windows = int(duration / window_length)
            
            # Répéter le label pour chaque fenêtre
            file_label = labels[i]
            expanded_labels.extend([file_label] * n_windows)
        
        expanded_labels = np.array(expanded_labels)
        
        
        # Vérifier que les dimensions correspondent
        if len(expanded_labels) != X.shape[0]:
            print(f"ERREUR: Dimensions incompatibles - Labels: {len(expanded_labels)}, Données: {X.shape[0]}")
            return
        
        # 5. Division train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, expanded_labels, test_size=0.3, random_state=42, stratify=expanded_labels
        )
        
        print(f"\nDivision des données:")
        print(f"  Entraînement: {X_train.shape[0]} échantillons")
        print(f"  Test: {X_test.shape[0]} échantillons")
        
        # 6. Analyser les performances
        results = analyze_pipeline_performance(X_train, X_test, y_train, y_test)
        
        print(f"\n=== RÉSUMÉ ===")
        best_model = max(results.keys(), key=lambda k: results[k]['test_score'])
        print(f"Meilleur modèle: {best_model}")
        print(f"Précision: {results[best_model]['test_score']:.3f}")
        print(f"Nombre de caractéristiques finales: {X.shape[1]}")
        print(f"Nombre total d'échantillons (fenêtres): {X.shape[0]}")
        print(f"Nombre de fichiers originaux: {len(raw_data)}")
        
    except Exception as e:
        print(f"Erreur lors de l'application de la pipeline: {e}")
        import traceback
        traceback.print_exc()
        print("Cela peut être dû à des données insuffisantes ou à des problèmes de format.")

if __name__ == "__main__":
    main()
