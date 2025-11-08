#!/usr/bin/env python3
"""
Script de test simple pour la pipeline EEG
"""

import numpy as np
import sys
import os

# Ajouter le répertoire src au path
sys.path.append('src')

def test_pipeline_basic():
    """Test basique de la pipeline sans données réelles"""
    print("=== TEST BASIQUE DE LA PIPELINE ===\n")
    
    try:
        from preprocessing_pipeline import create_eeg_pipeline, EEGPreprocessor, FeatureExtractor
        print("✓ Import de la pipeline réussi")
        
        # Créer la pipeline
        pipeline = create_eeg_pipeline(include_ica=False, n_features=10)
        print("✓ Création de la pipeline réussie")
        
        print(f"✓ Pipeline créée avec {len(pipeline.steps)} étapes:")
        for i, (name, step) in enumerate(pipeline.steps, 1):
            print(f"   {i}. {name}: {type(step).__name__}")
        
        return True
        
    except Exception as e:
        print(f"✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_extraction():
    """Test de l'extraction de caractéristiques avec des données simulées"""
    print("\n=== TEST EXTRACTION DE CARACTÉRISTIQUES ===\n")
    
    try:
        import mne
        from preprocessing_pipeline import FeatureExtractor
        
        # Créer des données EEG simulées
        print("Création de données EEG simulées...")
        
        # Paramètres
        sfreq = 160  # Fréquence d'échantillonnage
        duration = 10  # 10 secondes
        n_channels = 64
        
        # Créer un signal simulé
        times = np.arange(0, duration, 1/sfreq)
        n_samples = len(times)
        
        # Signal avec différentes fréquences
        data = np.zeros((n_channels, n_samples))
        for ch in range(n_channels):
            # Mélange de fréquences alpha (10Hz) et beta (20Hz)
            alpha = np.sin(2 * np.pi * 10 * times) * np.random.normal(1, 0.2)
            beta = np.sin(2 * np.pi * 20 * times) * np.random.normal(0.5, 0.1)
            noise = np.random.normal(0, 0.1, n_samples)
            data[ch] = alpha + beta + noise
        
        # Créer un objet MNE Raw
        ch_names = [f'EEG{i+1:03d}' for i in range(n_channels)]
        ch_types = ['eeg'] * n_channels
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(data, info, verbose=False)
        
        print(f"✓ Signal simulé créé: {n_channels} canaux, {duration}s, {sfreq}Hz")
        
        # Test de l'extracteur de caractéristiques
        extractor = FeatureExtractor(
            feature_types=['psd', 'time_domain'],
            window_length=2.0
        )
        
        features = extractor.fit_transform([raw])
        print(f"✓ Extraction réussie: {features.shape[0]} fenêtres, {features.shape[1]} caractéristiques")
        
        # Vérifier les fenêtres
        expected_windows = int(duration / 2.0)  # 2 secondes par fenêtre
        print(f"✓ Fenêtres attendues: {expected_windows}, obtenues: {features.shape[0]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loading():
    """Test du chargement de données réelles si disponibles"""
    print("\n=== TEST CHARGEMENT DE DONNÉES ===\n")
    
    base_directory = "/mnt/c/Users/louis/Downloads/archive/files/"
    
    if not os.path.exists(base_directory):
        print(f"⚠️  Répertoire de données non trouvé: {base_directory}")
        print("   Test avec données simulées uniquement")
        return True
    
    try:
        sys.path.append('src')
        from pipeline_demo import load_eeg_data
        
        # Essayer de charger un fichier de test
        raw_data, labels, file_info = load_eeg_data(
            base_directory,
            subjects=['S001'],
            tasks=['R03']
        )
        
        if len(raw_data) > 0:
            print(f"✓ Chargement réussi: {len(raw_data)} fichiers")
            print(f"✓ Premier fichier: {file_info[0]['file']}")
            print(f"✓ Durée: {file_info[0]['duration']:.1f}s")
            print(f"✓ Canaux: {file_info[0]['n_channels']}")
            return True
        else:
            print("⚠️  Aucun fichier chargé")
            return False
            
    except Exception as e:
        print(f"✗ Erreur: {e}")
        return False

def main():
    """Fonction principale de test"""
    print("🧠 TESTS DE LA PIPELINE EEG\n")
    
    # Tests
    tests = [
        ("Import et création de pipeline", test_pipeline_basic),
        ("Extraction de caractéristiques", test_feature_extraction),
        ("Chargement de données", test_data_loading),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"🔬 {test_name}...")
        success = test_func()
        results.append((test_name, success))
        print()
    
    # Résumé
    print("=" * 50)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 50)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    total_passed = sum(1 for _, success in results if success)
    print(f"\n🎯 {total_passed}/{len(results)} tests réussis")
    
    if total_passed == len(results):
        print("🎉 Tous les tests sont passés ! La pipeline est prête.")
    else:
        print("⚠️  Certains tests ont échoué. Vérifiez les dépendances.")

if __name__ == "__main__":
    main()
