# obesity_prediction.py

import joblib
import pandas as pd

def predict_obesity(data):
    """
    Make obesity level prediction using the trained decision tree model.

    Parameters:
    -----------
    data : dict
        Dictionary of input features for prediction. Example fields:
        - Gender: 'Male' or 'Female'
        - Age: float or int
        - Height: float
        - Weight: float
        - family_history_with_overweight: 'yes' or 'no'
        - FAVC, FCVC, NCP, CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS: sesuai dataset
        (semua field harus sama dengan yang digunakan saat training)

    Returns:
    --------
    dict
        - prediction: encoded class
        - prediction_label: decoded class label
        - probability: float (confidence)
    """

    # Load model components
    components = joblib.load('obesity_prediction_components.joblib')
    
    model = components['model']
    encoding_maps = components['encoding_maps']
    feature_names = components['feature_names']
    
    # Convert to DataFrame
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data.copy()

    # Apply encoding
    for col in df.columns:
        if col in encoding_maps:
            df[col] = df[col].map(encoding_maps[col])
    
    # Filter only the features used by the model
    df = df[feature_names]
    
    # Make prediction
    prediction = model.predict(df)[0]
    probabilities = model.predict_proba(df)[0]

    # Decode prediction label
    target_map_inverse = {v: k for k, v in encoding_maps['NObeyesdad'].items()}
    
    return {
        'prediction': int(prediction),
        'prediction_label': target_map_inverse[prediction],
        'probability': float(probabilities[prediction])
    }

# Example usage
if __name__ == "__main__":
    example = {
        'Gender': 'Male',
        'Age': 22,
        'Height': 1.75,
        'Weight': 85,
        'family_history_with_overweight': 'yes',
        'FAVC': 'yes',
        'FCVC': 2.0,
        'NCP': 3.0,
        'CAEC': 'Sometimes',
        'SMOKE': 'no',
        'CH2O': 2.0,
        'SCC': 'no',
        'FAF': 1.0,
        'TUE': 1.0,
        'CALC': 'Sometimes',
        'MTRANS': 'Public_Transportation'
    }

    result = predict_obesity(example)
    print("Prediction result:")
    print(f"Class: {result['prediction_label']}")
    print(f"Probability: {result['probability']:.4f}")
