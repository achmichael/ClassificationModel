import pandas as pd
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time

def train_models(X_train, y_train):
    """
    Fungsi untuk melatih 4 model machine learning:
    1. Multinomial Naive Bayes
    2. K-Nearest Neighbor
    3. Support Vector Machine (Linear)
    4. Random Forest
    
    Parameters:
    -----------
    X_train : sparse matrix
        Fitur training dalam bentuk TF-IDF matrix
    y_train : pandas Series atau array
        Label training
    
    Returns:
    --------
    dict : Dictionary berisi {nama_model: model_terlatih}
    """
    
    models = {}
    
    print("\n" + "="*80)
    print("MELATIH MODEL MACHINE LEARNING")
    print("="*80)
    
    # 1. Multinomial Naive Bayes
    print("\n[1/4] Training Multinomial Naive Bayes...")
    start_time = time.time()
    mnb = MultinomialNB(alpha=1.0)
    mnb.fit(X_train, y_train)
    training_time = time.time() - start_time
    models['Multinomial Naive Bayes'] = mnb
    print(f"✓ Multinomial Naive Bayes trained in {training_time:.2f} seconds")
    
    # 2. K-Nearest Neighbor
    print("\n[2/4] Training K-Nearest Neighbor...")
    start_time = time.time()
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(X_train, y_train)
    training_time = time.time() - start_time
    models['K-Nearest Neighbor'] = knn
    print(f"✓ K-Nearest Neighbor trained in {training_time:.2f} seconds")
    
    # 3. Support Vector Machine (Linear)
    print("\n[3/4] Training Support Vector Machine (Linear)...")
    start_time = time.time()
    svm = LinearSVC(C=1.0, max_iter=1000, random_state=42)
    svm.fit(X_train, y_train)
    training_time = time.time() - start_time
    models['Support Vector Machine'] = svm
    print(f"✓ Support Vector Machine trained in {training_time:.2f} seconds")
    
    # 4. Random Forest
    print("\n[4/4] Training Random Forest...")
    start_time = time.time()
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    training_time = time.time() - start_time
    models['Random Forest'] = rf
    print(f"✓ Random Forest trained in {training_time:.2f} seconds")
    
    print("\n" + "="*80)
    print(f"✓ Semua {len(models)} model berhasil dilatih!")
    print("="*80)
    
    return models


def evaluate_models(models, X_train, y_train, X_test, y_test):
    """
    Fungsi untuk evaluasi semua model pada data training dan testing
    
    Parameters:
    -----------
    models : dict
        Dictionary berisi {nama_model: model_terlatih}
    X_train : sparse matrix
        Fitur training
    y_train : array
        Label training
    X_test : sparse matrix
        Fitur testing
    y_test : array
        Label testing
    
    Returns:
    --------
    dict : Dictionary berisi hasil evaluasi setiap model
    """
    
    results = {}
    
    print("\n" + "="*80)
    print("EVALUASI MODEL")
    print("="*80)
    
    for model_name, model in models.items():
        print(f"\n{'='*80}")
        print(f"{model_name}")
        print(f"{'='*80}")
        
        # Prediksi pada training set
        y_train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        # Prediksi pada testing set
        y_test_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        print(f"\nTraining Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        print(f"Testing Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        print(f"\nClassification Report (Testing Set):")
        print(classification_report(y_test, y_test_pred, target_names=['halal', 'haram']))
        
        print(f"\nConfusion Matrix (Testing Set):")
        cm = confusion_matrix(y_test, y_test_pred)
        print(f"              Predicted")
        print(f"              halal  haram")
        print(f"Actual halal  {cm[0][0]:5d}  {cm[0][1]:5d}")
        print(f"       haram  {cm[1][0]:5d}  {cm[1][1]:5d}")
        
        # Simpan hasil
        results[model_name] = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'classification_report': classification_report(y_test, y_test_pred, output_dict=True),
            'confusion_matrix': cm
        }
    
    return results


def save_models(models, filepath='models/trained_models.pkl'):
    """
    Menyimpan semua model terlatih ke file pickle
    
    Parameters:
    -----------
    models : dict
        Dictionary berisi {nama_model: model_terlatih}
    filepath : str
        Path untuk menyimpan file
    """
    with open(filepath, 'wb') as f:
        pickle.dump(models, f)
    print(f"\n✓ Semua model disimpan ke: {filepath}")


def main():
    """
    Fungsi utama untuk menjalankan training dan evaluasi model
    """
    
    # Load data train-test yang sudah di-split
    print("Loading train-test data...")
    with open('models/train_test_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_train_tfidf = data['X_train_tfidf']
    X_test_tfidf = data['X_test_tfidf']
    y_train = data['y_train']
    y_test = data['y_test']
    
    print(f"✓ Data loaded successfully")
    print(f"  Training samples: {X_train_tfidf.shape[0]}")
    print(f"  Testing samples: {X_test_tfidf.shape[0]}")
    print(f"  Number of features: {X_train_tfidf.shape[1]}")
    
    # Train models
    models = train_models(X_train_tfidf, y_train)
    
    # Evaluate models
    results = evaluate_models(models, X_train_tfidf, y_train, X_test_tfidf, y_test)
    
    # Save models
    save_models(models)
    
    # Ringkasan performa semua model
    print("\n" + "="*80)
    print("RINGKASAN PERFORMA MODEL")
    print("="*80)
    print(f"\n{'Model':<30} {'Train Acc':<12} {'Test Acc':<12} {'Difference':<12}")
    print("-"*80)
    
    for model_name, result in results.items():
        train_acc = result['train_accuracy']
        test_acc = result['test_accuracy']
        diff = train_acc - test_acc
        print(f"{model_name:<30} {train_acc:.4f}       {test_acc:.4f}       {diff:+.4f}")
    
    # Tentukan model terbaik berdasarkan test accuracy
    best_model_name = max(results, key=lambda x: results[x]['test_accuracy'])
    best_accuracy = results[best_model_name]['test_accuracy']
    
    print("\n" + "="*80)
    print(f"🏆 MODEL TERBAIK: {best_model_name}")
    print(f"   Test Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print("="*80)
    
    # Simpan hasil evaluasi
    with open('models/evaluation_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"\n✓ Hasil evaluasi disimpan ke: models/evaluation_results.pkl")
    

if __name__ == "__main__":
    main()
