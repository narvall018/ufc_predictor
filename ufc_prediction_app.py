import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import os
import pickle
import joblib
import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="UFC Fight Predictor",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
        color: #FF0000;
        margin-bottom: 20px;
    }
    .sub-title {
        text-align: center;
        font-size: 1.5rem;
        margin-bottom: 30px;
    }
    .prediction-box {
        background-color: rgba(248, 249, 250, 0.8);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .red-fighter {
        color: #FF0000;
        font-weight: bold;
    }
    .blue-fighter {
        color: #0000FF;
        font-weight: bold;
    }
    .winner {
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    /* Adapter pour thème sombre */
    [data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 10px;
        border-radius: 5px;
    }
    
    [data-testid="stMetric"] > div {
        width: 100%;
    }
    
    [data-testid="stMetricLabel"] {
        display: flex;
        justify-content: center;
    }
    
    [data-testid="stMetricValue"] {
        display: flex;
        justify-content: center;
    }
    
    [data-testid="stMetricDelta"] {
        display: flex;
        justify-content: center;
    }
    
    /* Compatibilité thème sombre pour les cartes */
    .stTabs [data-baseweb="tab-panel"] {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 15px;
    }
    
    /* Badge pour le modèle ML */
    .ml-badge {
        background-color: #4CAF50;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
        margin-left: 10px;
    }
    
    /* Badge pour le modèle classique */
    .classic-badge {
        background-color: #2196F3;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
        margin-left: 10px;
    }
    
    /* Style pour le sélecteur de méthode de prédiction */
    .prediction-method {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    
    /* Note d'information */
    .info-box {
        background-color: rgba(255, 193, 7, 0.1);
        padding: 10px;
        border-radius: 10px;
        margin-top: 10px;
        margin-bottom: 15px;
        border-left: 3px solid #FFC107;
    }
    
    /* Divider pour les sections */
    .divider {
        border-top: 1px solid rgba(200, 200, 200, 0.3);
        margin: 20px 0;
    }
    
    /* Style pour les tableaux de paris */
    .betting-card {
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    
    .betting-card-red {
        background-color: rgba(255, 235, 238, 0.7);
    }
    
    .betting-card-blue {
        background-color: rgba(227, 242, 253, 0.7);
    }
    
    .favorable {
        color: green;
        font-weight: bold;
    }
    
    .neutral {
        color: orange;
        font-weight: bold;
    }
    
    .unfavorable {
        color: red;
        font-weight: bold;
    }
    
    /* Style pour les boutons d'action */
    .delete-button {
        background-color: #f44336;
        color: white;
        border: none;
        padding: 5px 10px;
        border-radius: 5px;
        cursor: pointer;
    }
    
    .update-button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 5px 10px;
        border-radius: 5px;
        cursor: pointer;
    }
    
    /* Style pour le conteneur de gestion des paris */
    .bet-management-container {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    
    /* Style pour les options de mise */
    .stake-options {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    
    /* Tableau de Kelly */
    .kelly-table {
        width: 100%;
        border-collapse: collapse;
    }
    .kelly-table th, .kelly-table td {
        padding: 8px;
        text-align: left;
        border-bottom: 1px solid rgba(255, 255, 255, 0.2);
    }
    .kelly-table tr:hover {
        background-color: rgba(255, 255, 255, 0.1);
    }
    .kelly-highlight {
        background-color: rgba(76, 175, 80, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# FONCTIONS POUR LE MODÈLE ML

@st.cache_resource
def load_ml_model():
    """
    Charge le modèle ML entraîné, le scaler et les feature names
    """
    try:
        # Essayer de charger avec joblib (plus rapide pour les gros modèles)
        if os.path.exists("ufc_prediction_model.joblib"):
            model_data = joblib.load("ufc_prediction_model.joblib")
            print("Modèle chargé depuis joblib")
        # Sinon, essayer avec pickle
        elif os.path.exists("ufc_prediction_model.pkl"):
            with open("ufc_prediction_model.pkl", 'rb') as file:
                model_data = pickle.load(file)
            print("Modèle chargé depuis pickle")
        else:
            print("Aucun modèle trouvé")
            return None, None, None
        
        model = model_data.get('model')
        scaler = model_data.get('scaler')
        feature_names = model_data.get('feature_names')
        
        return model, scaler, feature_names
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        return None, None, None

def get_float_value(stats_dict, key, default=0.0):
    """
    Récupère une valeur du dictionnaire et la convertit en float.
    """
    if key not in stats_dict:
        return default
        
    value = stats_dict[key]
    
    if isinstance(value, (int, float)):
        return float(value)
    
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def create_ml_features(r_stats, b_stats):
    """
    Crée les features nécessaires pour le modèle ML
    """
    features = {}
    
    # Liste des statistiques numériques que nous utiliserons
    numeric_stats = ['wins', 'losses', 'height', 'weight', 'reach', 'age', 
                     'SLpM', 'sig_str_acc', 'SApM', 'str_def', 
                     'td_avg', 'td_acc', 'td_def', 'sub_avg']
    
    # Extraire et convertir les statistiques numériques
    for stat in numeric_stats:
        r_value = get_float_value(r_stats, stat, 0.0)
        b_value = get_float_value(b_stats, stat, 0.0)
        
        features[f'r_{stat}'] = r_value
        features[f'b_{stat}'] = b_value
        features[f'diff_{stat}'] = r_value - b_value
        
        if b_value != 0:
            features[f'ratio_{stat}'] = r_value / b_value
        else:
            features[f'ratio_{stat}'] = 0.0
    
    # Features avancées
    
    # 1. Win ratio et expérience
    r_wins = get_float_value(r_stats, 'wins', 0)
    r_losses = get_float_value(r_stats, 'losses', 0)
    b_wins = get_float_value(b_stats, 'wins', 0)
    b_losses = get_float_value(b_stats, 'losses', 0)
    
    # Nombre total de combats (expérience)
    features['r_total_fights'] = r_wins + r_losses
    features['b_total_fights'] = b_wins + b_losses
    features['diff_total_fights'] = features['r_total_fights'] - features['b_total_fights']
    
    # Win ratio
    if r_wins + r_losses > 0:
        features['r_win_ratio'] = r_wins / (r_wins + r_losses)
    else:
        features['r_win_ratio'] = 0
    
    if b_wins + b_losses > 0:
        features['b_win_ratio'] = b_wins / (b_wins + b_losses)
    else:
        features['b_win_ratio'] = 0
        
    features['diff_win_ratio'] = features['r_win_ratio'] - features['b_win_ratio']
    
    # 2. Striking efficiency
    r_slpm = get_float_value(r_stats, 'SLpM', 0)
    r_sapm = get_float_value(r_stats, 'SApM', 0)
    b_slpm = get_float_value(b_stats, 'SLpM', 0)
    b_sapm = get_float_value(b_stats, 'SApM', 0)
    
    # Efficacité de frappe
    features['r_striking_efficiency'] = r_slpm - r_sapm
    features['b_striking_efficiency'] = b_slpm - b_sapm
    features['diff_striking_efficiency'] = features['r_striking_efficiency'] - features['b_striking_efficiency']
    
    # Ratio frappe/défense
    if r_sapm > 0:
        features['r_strike_defense_ratio'] = r_slpm / r_sapm
    else:
        features['r_strike_defense_ratio'] = r_slpm if r_slpm > 0 else 1.0
        
    if b_sapm > 0:
        features['b_strike_defense_ratio'] = b_slpm / b_sapm
    else:
        features['b_strike_defense_ratio'] = b_slpm if b_slpm > 0 else 1.0
        
    features['diff_strike_defense_ratio'] = features['r_strike_defense_ratio'] - features['b_strike_defense_ratio']
    
    # 3. Différences physiques 
    r_height = get_float_value(r_stats, 'height', 0)
    r_weight = get_float_value(r_stats, 'weight', 0)
    r_reach = get_float_value(r_stats, 'reach', 0)
    b_height = get_float_value(b_stats, 'height', 0)
    b_weight = get_float_value(b_stats, 'weight', 0)
    b_reach = get_float_value(b_stats, 'reach', 0)
    
    # Rapport taille/poids
    if r_weight > 0:
        features['r_height_weight_ratio'] = r_height / r_weight
    else:
        features['r_height_weight_ratio'] = 0
        
    if b_weight > 0:
        features['b_height_weight_ratio'] = b_height / b_weight
    else:
        features['b_height_weight_ratio'] = 0
        
    features['diff_height_weight_ratio'] = features['r_height_weight_ratio'] - features['b_height_weight_ratio']
    
    # Avantage d'allonge normalisé par la taille
    if r_height > 0:
        features['r_reach_height_ratio'] = r_reach / r_height
    else:
        features['r_reach_height_ratio'] = 0
        
    if b_height > 0:
        features['b_reach_height_ratio'] = b_reach / b_height
    else:
        features['b_reach_height_ratio'] = 0
        
    features['diff_reach_height_ratio'] = features['r_reach_height_ratio'] - features['b_reach_height_ratio']
    
    # 4. Indicateurs de style de combat
    r_td_avg = get_float_value(r_stats, 'td_avg', 0)
    r_sub_avg = get_float_value(r_stats, 'sub_avg', 0)
    r_str_def = get_float_value(r_stats, 'str_def', 0)
    r_td_def = get_float_value(r_stats, 'td_def', 0)
    b_td_avg = get_float_value(b_stats, 'td_avg', 0)
    b_sub_avg = get_float_value(b_stats, 'sub_avg', 0)
    b_str_def = get_float_value(b_stats, 'str_def', 0)
    b_td_def = get_float_value(b_stats, 'td_def', 0)
    
    # Spécialiste de striking vs grappling
    if r_td_avg > 0:
        features['r_striking_grappling_ratio'] = r_slpm / r_td_avg
    else:
        features['r_striking_grappling_ratio'] = r_slpm if r_slpm > 0 else 0
        
    if b_td_avg > 0:
        features['b_striking_grappling_ratio'] = b_slpm / b_td_avg
    else:
        features['b_striking_grappling_ratio'] = b_slpm if b_slpm > 0 else 0
        
    # Offensive vs défensive (plus le ratio est élevé, plus le combattant est offensif)
    features['r_offensive_rating'] = r_slpm * r_td_avg * (1 + r_sub_avg)
    features['b_offensive_rating'] = b_slpm * b_td_avg * (1 + b_sub_avg)
    features['diff_offensive_rating'] = features['r_offensive_rating'] - features['b_offensive_rating']
    
    features['r_defensive_rating'] = r_str_def * r_td_def
    features['b_defensive_rating'] = b_str_def * b_td_def
    features['diff_defensive_rating'] = features['r_defensive_rating'] - features['b_defensive_rating']
    
    # 5. Variables composites
    # Performance globale = win_ratio * offensive_rating * defensive_rating
    features['r_overall_performance'] = features['r_win_ratio'] * features['r_offensive_rating'] * features['r_defensive_rating']
    features['b_overall_performance'] = features['b_win_ratio'] * features['b_offensive_rating'] * features['b_defensive_rating']
    features['diff_overall_performance'] = features['r_overall_performance'] - features['b_overall_performance']
    
    # Avantage physique combiné (taille, poids, allonge)
    features['r_physical_advantage'] = features['r_reach_height_ratio'] * features['r_height_weight_ratio']
    features['b_physical_advantage'] = features['b_reach_height_ratio'] * features['b_height_weight_ratio']
    features['diff_physical_advantage'] = features['r_physical_advantage'] - features['b_physical_advantage']
    
    # 6. Style matching
    if 'stance' in r_stats and 'stance' in b_stats:
        features['same_stance'] = 1 if r_stats['stance'] == b_stats['stance'] else 0
        
        # One-hot encoding des stances
        r_stance = r_stats.get('stance', '').lower()
        b_stance = b_stats.get('stance', '').lower()
        
        stances = ['orthodox', 'southpaw', 'switch', 'open stance']
        for stance in stances:
            features[f'r_stance_{stance}'] = 1 if r_stance == stance else 0
            features[f'b_stance_{stance}'] = 1 if b_stance == stance else 0
    
    return features

def predict_with_ml(r_stats, b_stats, model, scaler, feature_names):
    """
    Prédit l'issue d'un combat avec le modèle ML
    """
    # Si le modèle n'est pas chargé, retourner None
    if model is None or scaler is None or feature_names is None:
        return None
    
    try:
        # Créer les features
        features = create_ml_features(r_stats, b_stats)
        
        # Convertir en DataFrame
        features_df = pd.DataFrame([features])
        
        # S'assurer que toutes les colonnes nécessaires sont présentes
        for col in feature_names:
            if col not in features_df.columns:
                features_df[col] = 0
        
        # Ne garder que les colonnes utilisées par le modèle
        features_df = features_df[feature_names]
        
        # Remplacer les valeurs infinies et NaN
        features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        features_df.fillna(0, inplace=True)
        
        # Normaliser
        X_scaled = scaler.transform(features_df)
        
        # Prédire
        if hasattr(model, "predict_proba"):
            red_prob = model.predict_proba(X_scaled)[0][1]
        else:
            red_prob = float(model.predict(X_scaled)[0])
        
        blue_prob = 1 - red_prob
        
        # Créer le résultat
        result = {
            'prediction': 'Red' if red_prob > blue_prob else 'Blue',
            'red_probability': red_prob,
            'blue_probability': blue_prob,
            'confidence': 'Élevé' if abs(red_prob - blue_prob) > 0.2 else 'Modéré'
        }
        
        return result
    except Exception as e:
        st.error(f"Erreur lors de la prédiction ML: {e}")
        return None

# FONCTIONS ORIGINALES DE L'APP

def load_fighters_stats(file_path):
    """
    Charge les statistiques des combattants depuis un fichier texte
    """
    fighters = []
    current_fighter = {}
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:  # Ligne vide, séparateur entre combattants
                    if current_fighter and 'name' in current_fighter:
                        fighters.append(current_fighter)
                        current_fighter = {}
                elif ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Gestion des valeurs manquantes
                    if value.lower() in ['none', 'nan', '', 'null']:
                        if key in ['wins', 'losses', 'age']:
                            value = 0
                        elif key in ['height', 'weight', 'reach', 'SLpM', 'sig_str_acc', 
                                   'SApM', 'str_def', 'td_avg', 'td_acc', 'td_def', 'sub_avg']:
                            value = 0.0
                        else:
                            value = 'Unknown'
                    else:
                        # Conversion des types
                        try:
                            if key in ['wins', 'losses', 'age']:
                                value = int(value)
                            elif key in ['height', 'weight', 'reach', 'SLpM', 'sig_str_acc', 
                                       'SApM', 'str_def', 'td_avg', 'td_acc', 'td_def', 'sub_avg']:
                                value = float(value)
                        except ValueError:
                            if key in ['wins', 'losses', 'age']:
                                value = 0
                            elif key in ['height', 'weight', 'reach', 'SLpM', 'sig_str_acc', 
                                       'SApM', 'str_def', 'td_avg', 'td_acc', 'td_def', 'sub_avg']:
                                value = 0.0
                    
                    current_fighter[key] = value
        
        # Ajouter le dernier combattant
        if current_fighter and 'name' in current_fighter:
            fighters.append(current_fighter)
            
        return fighters
        
    except Exception as e:
        st.error(f"Erreur lors du chargement: {e}")
        return []

# Éliminer les doublons en gardant le meilleur combattant de chaque nom
def deduplicate_fighters(fighters_list):
    """
    Supprime les doublons de noms en gardant le combattant le plus performant
    """
    fighters_by_name = {}
    
    for fighter in fighters_list:
        name = fighter['name']
        
        # Calculer un score de performance
        wins = fighter.get('wins', 0)
        losses = fighter.get('losses', 0)
        win_ratio = wins / max(wins + losses, 1)
        
        # Score combiné (ratio de victoires + âge inverse + nombre de combats)
        performance = win_ratio + (1/max(fighter.get('age', 30), 1)) + (wins + losses)/10
        
        # Garder seulement le combattant avec le meilleur score
        if name not in fighters_by_name or performance > fighters_by_name[name]['performance']:
            fighters_by_name[name] = {
                'fighter': fighter,
                'performance': performance
            }
    
    # Extraire les meilleurs combattants uniques
    unique_fighters = [info['fighter'] for info in fighters_by_name.values()]
    
    return unique_fighters

# Fonction pour prédire l'issue d'un combat (méthode originale basée sur les statistiques)
def predict_fight_classic(fighter_a, fighter_b, odds_a=0, odds_b=0):
    """
    Prédit l'issue d'un combat avec analyse de paris (méthode classique)
    """
    # Calcul des scores de base
    a_score = (
        fighter_a['wins'] / max(fighter_a['wins'] + fighter_a['losses'], 1) * 2 + 
        fighter_a['SLpM'] * fighter_a['sig_str_acc'] - 
        fighter_a['SApM'] * (1 - fighter_a['str_def']) + 
        fighter_a['td_avg'] * fighter_a['td_acc'] + 
        fighter_a['sub_avg']
    )
    
    b_score = (
        fighter_b['wins'] / max(fighter_b['wins'] + fighter_b['losses'], 1) * 2 + 
        fighter_b['SLpM'] * fighter_b['sig_str_acc'] - 
        fighter_b['SApM'] * (1 - fighter_b['str_def']) + 
        fighter_b['td_avg'] * fighter_b['td_acc'] + 
        fighter_b['sub_avg']
    )
    
    # Normaliser pour obtenir des probabilités
    total = a_score + b_score
    red_prob = a_score / total if total > 0 else 0.5
    blue_prob = b_score / total if total > 0 else 0.5
    
    # Résultat de base
    result = {
        'prediction': 'Red' if red_prob > blue_prob else 'Blue',
        'winner_name': fighter_a['name'] if red_prob > blue_prob else fighter_b['name'],
        'loser_name': fighter_b['name'] if red_prob > blue_prob else fighter_a['name'],
        'red_probability': red_prob,
        'blue_probability': blue_prob,
        'confidence': 'Élevé' if abs(red_prob - blue_prob) > 0.2 else 'Modéré'
    }
    
    # Ajouter l'analyse des paris si des cotes sont fournies
    if odds_a > 0 and odds_b > 0:
        # Probabilité implicite selon les bookmakers
        implied_prob_a = 1 / odds_a
        implied_prob_b = 1 / odds_b
        
        # Normaliser pour éliminer la marge du bookmaker
        total_implied = implied_prob_a + implied_prob_b
        implied_prob_a_norm = implied_prob_a / total_implied
        implied_prob_b_norm = implied_prob_b / total_implied
        
        # Valeur espérée (Expected Value)
        ev_a = (red_prob * odds_a) - 1
        ev_b = (blue_prob * odds_b) - 1
        
        # Recommandation de pari
        bet_recommendation_a = "Favorable" if ev_a > 0.1 else "Neutre" if ev_a > -0.1 else "Défavorable"
        bet_recommendation_b = "Favorable" if ev_b > 0.1 else "Neutre" if ev_b > -0.1 else "Défavorable"
        
        result['betting'] = {
            'odds_red': odds_a,
            'odds_blue': odds_b,
            'implied_prob_red': implied_prob_a_norm,
            'implied_prob_blue': implied_prob_b_norm,
            'ev_red': ev_a,
            'ev_blue': ev_b,
            'recommendation_red': bet_recommendation_a,
            'recommendation_blue': bet_recommendation_b,
            'edge_red': red_prob - implied_prob_a_norm,
            'edge_blue': blue_prob - implied_prob_b_norm
        }
    
    return result

# Fonction de prédiction améliorée qui retourne les résultats des deux méthodes
def predict_both_methods(fighter_a, fighter_b, odds_a=0, odds_b=0):
    """
    Prédit l'issue d'un combat en utilisant les deux méthodes (ML et classique)
    et retourne les deux prédictions
    """
    # Prédiction avec la méthode classique
    classic_prediction = predict_fight_classic(fighter_a, fighter_b, odds_a, odds_b)
    classic_prediction['method'] = 'classic'
    
    # Prédiction avec ML si disponible
    model, scaler, feature_names = load_ml_model()
    ml_prediction = None
    
    if model is not None:
        ml_result = predict_with_ml(fighter_a, fighter_b, model, scaler, feature_names)
        
        if ml_result is not None:
            ml_prediction = ml_result
            ml_prediction['winner_name'] = fighter_a['name'] if ml_prediction['prediction'] == 'Red' else fighter_b['name']
            ml_prediction['loser_name'] = fighter_b['name'] if ml_prediction['prediction'] == 'Red' else fighter_a['name']
            ml_prediction['method'] = 'ml'
            
            # Ajouter l'analyse des paris si des cotes sont fournies
            if odds_a > 0 and odds_b > 0:
                red_prob = ml_prediction['red_probability']
                blue_prob = ml_prediction['blue_probability']
                
                # Probabilité implicite selon les bookmakers
                implied_prob_a = 1 / odds_a
                implied_prob_b = 1 / odds_b
                
                # Normaliser pour éliminer la marge du bookmaker
                total_implied = implied_prob_a + implied_prob_b
                implied_prob_a_norm = implied_prob_a / total_implied
                implied_prob_b_norm = implied_prob_b / total_implied
                
                # Valeur espérée (Expected Value)
                ev_a = (red_prob * odds_a) - 1
                ev_b = (blue_prob * odds_b) - 1
                
                # Recommandation de pari
                bet_recommendation_a = "Favorable" if ev_a > 0.1 else "Neutre" if ev_a > -0.1 else "Défavorable"
                bet_recommendation_b = "Favorable" if ev_b > 0.1 else "Neutre" if ev_b > -0.1 else "Défavorable"
                
                ml_prediction['betting'] = {
                    'odds_red': odds_a,
                    'odds_blue': odds_b,
                    'implied_prob_red': implied_prob_a_norm,
                    'implied_prob_blue': implied_prob_b_norm,
                    'ev_red': ev_a,
                    'ev_blue': ev_b,
                    'recommendation_red': bet_recommendation_a,
                    'recommendation_blue': bet_recommendation_b,
                    'edge_red': red_prob - implied_prob_a_norm,
                    'edge_blue': blue_prob - implied_prob_b_norm
                }
    
    return classic_prediction, ml_prediction

# FONCTIONS DE VISUALISATION

def create_radar_chart(fighter_a, fighter_b):
    """Crée un graphique radar comparant les attributs des combattants"""
    categories = ['Win Ratio', 'Striking', 'Defense', 'Ground', 'Experience']
    
    # Calculer les valeurs pour chaque catégorie
    a_values = [
        fighter_a['wins']/(fighter_a['wins']+fighter_a['losses']) if fighter_a['wins']+fighter_a['losses'] > 0 else 0,
        fighter_a['SLpM'] * fighter_a['sig_str_acc'],
        fighter_a['str_def'],
        fighter_a['td_avg'] * fighter_a['td_acc'] + fighter_a['sub_avg'],
        fighter_a['wins'] + fighter_a['losses']
    ]
    
    b_values = [
        fighter_b['wins']/(fighter_b['wins']+fighter_b['losses']) if fighter_b['wins']+fighter_b['losses'] > 0 else 0,
        fighter_b['SLpM'] * fighter_b['sig_str_acc'],
        fighter_b['str_def'],
        fighter_b['td_avg'] * fighter_b['td_acc'] + fighter_b['sub_avg'],
        fighter_b['wins'] + fighter_b['losses']
    ]
    
    # Normaliser les valeurs
    max_values = [max(a, b) for a, b in zip(a_values, b_values)]
    a_values_norm = [a/m if m > 0 else 0 for a, m in zip(a_values, max_values)]
    b_values_norm = [b/m if m > 0 else 0 for b, m in zip(b_values, max_values)]
    
    # Utiliser Plotly pour le graphique radar
    fig = go.Figure()
    
    # Ajouter les traces pour chaque combattant
    fig.add_trace(go.Scatterpolar(
        r=a_values_norm + [a_values_norm[0]],  # Fermer le polygone
        theta=categories + [categories[0]],
        fill='toself',
        name=fighter_a['name'],
        line_color='red',
        fillcolor='rgba(255, 0, 0, 0.2)'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=b_values_norm + [b_values_norm[0]],  # Fermer le polygone
        theta=categories + [categories[0]],
        fill='toself',
        name=fighter_b['name'],
        line_color='blue',
        fillcolor='rgba(0, 0, 255, 0.2)'
    ))
    
    # Configurer la mise en page
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1.2]
            )
        ),
        title={
            'text': "Comparaison des attributs",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        height=500,
        margin=dict(l=80, r=80, t=100, b=80)
    )
    
    return fig

def create_strengths_weaknesses_chart(fighter_a, fighter_b):
    """Crée un graphique des forces et faiblesses des combattants"""
    attributes = ['Striking', 'Ground Game', 'Defense', 'Endurance', 'Experience']
    
    # Calcul des scores pour chaque attribut
    a_striking = fighter_a['SLpM'] * fighter_a['sig_str_acc']
    a_ground = fighter_a['td_avg'] * fighter_a['td_acc'] + fighter_a['sub_avg']
    a_defense = fighter_a['str_def'] * 0.7 + fighter_a['td_def'] * 0.3
    a_endurance = 1 / (fighter_a['SApM'] + 0.1)  # Inverse des coups reçus
    a_experience = fighter_a['wins'] + fighter_a['losses']
    
    b_striking = fighter_b['SLpM'] * fighter_b['sig_str_acc']
    b_ground = fighter_b['td_avg'] * fighter_b['td_acc'] + fighter_b['sub_avg']
    b_defense = fighter_b['str_def'] * 0.7 + fighter_b['td_def'] * 0.3
    b_endurance = 1 / (fighter_b['SApM'] + 0.1)
    b_experience = fighter_b['wins'] + fighter_b['losses']
    
    a_scores = [a_striking, a_ground, a_defense, a_endurance, a_experience]
    b_scores = [b_striking, b_ground, b_defense, b_endurance, b_experience]
    
    # Normalisation
    max_scores = [max(a, b) for a, b in zip(a_scores, b_scores)]
    a_norm = [a/m if m > 0 else 0 for a, m in zip(a_scores, max_scores)]
    b_norm = [b/m if m > 0 else 0 for b, m in zip(b_scores, max_scores)]
    
    # Créer le dataframe pour Plotly
    df = pd.DataFrame({
        'Attribute': attributes + attributes,
        'Value': a_norm + b_norm,
        'Fighter': [fighter_a['name']] * 5 + [fighter_b['name']] * 5
    })
    
    # Créer le graphique avec Plotly
    fig = px.bar(
        df, 
        x='Attribute', 
        y='Value', 
        color='Fighter',
        barmode='group',
        color_discrete_map={fighter_a['name']: 'red', fighter_b['name']: 'blue'},
        title="Forces et faiblesses comparatives"
    )
    
    fig.update_layout(
        yaxis_title="Score normalisé",
        xaxis_title="",
        legend_title="Combattant",
        height=500
    )
    
    return fig

def create_style_analysis_chart(fighter_a, fighter_b):
    """Crée un graphique d'analyse des styles de combat"""
    # Calculer des indicateurs de style
    a_striking = fighter_a['SLpM'] * fighter_a['sig_str_acc']
    a_ground = fighter_a['td_avg'] * fighter_a['td_acc'] + fighter_a['sub_avg']
    a_striker_grappler = a_striking / (a_ground + 0.1)  # > 1 = striker, < 1 = grappler
    a_aggressive_defensive = fighter_a['SLpM'] / (fighter_a['str_def'] + 0.1)  # > 1 = agressif
    
    b_striking = fighter_b['SLpM'] * fighter_b['sig_str_acc']
    b_ground = fighter_b['td_avg'] * fighter_b['td_acc'] + fighter_b['sub_avg']
    b_striker_grappler = b_striking / (b_ground + 0.1)
    b_aggressive_defensive = fighter_b['SLpM'] / (fighter_b['str_def'] + 0.1)
    
    # Normalisation pour le graphique
    max_sg = max(a_striker_grappler, b_striker_grappler)
    max_ad = max(a_aggressive_defensive, b_aggressive_defensive)
    
    a_sg_norm = a_striker_grappler / max_sg if max_sg > 0 else 0
    a_ad_norm = a_aggressive_defensive / max_ad if max_ad > 0 else 0
    
    b_sg_norm = b_striker_grappler / max_sg if max_sg > 0 else 0
    b_ad_norm = b_aggressive_defensive / max_ad if max_ad > 0 else 0
    
    # Créer le graphique avec Plotly
    fig = go.Figure()
    
    # Ajouter une grille de fond et des lignes de quadrant
    fig.add_shape(
        type="rect", xref="paper", yref="paper",
        x0=0, y0=0, x1=1, y1=1,
        line=dict(color="gray", width=1, dash="dash"),
    )
    
    fig.add_shape(
        type="line", xref="paper", yref="paper",
        x0=0.5, y0=0, x1=0.5, y1=1,
        line=dict(color="gray", width=1, dash="dash"),
    )
    
    fig.add_shape(
        type="line", xref="paper", yref="paper",
        x0=0, y0=0.5, x1=1, y1=0.5,
        line=dict(color="gray", width=1, dash="dash"),
    )
    
    # Ajouter les points pour chaque combattant
    fig.add_trace(go.Scatter(
        x=[a_sg_norm],
        y=[a_ad_norm],
        mode='markers',
        marker=dict(size=15, color='red'),
        name=fighter_a['name']
    ))
    
    fig.add_trace(go.Scatter(
        x=[b_sg_norm],
        y=[b_ad_norm],
        mode='markers',
        marker=dict(size=15, color='blue'),
        name=fighter_b['name']
    ))
    
    # Ajouter des annotations pour les quadrants
    fig.add_annotation(x=0.25, y=0.75, text="Grappler Agressif", showarrow=False)
    fig.add_annotation(x=0.75, y=0.75, text="Striker Agressif", showarrow=False)
    fig.add_annotation(x=0.25, y=0.25, text="Grappler Défensif", showarrow=False)
    fig.add_annotation(x=0.75, y=0.25, text="Striker Défensif", showarrow=False)
    
    # Configurer la mise en page
    fig.update_layout(
        title="Analyse de style de combat",
        xaxis_title="Style de combat (Grappler ← → Striker)",
        yaxis_title="Approche (Défensif ← → Agressif)",
        xaxis=dict(range=[0, 1.1]),
        yaxis=dict(range=[0, 1.1]),
        height=600
    )
    
    return fig

# Fonction pour créer un DataFrame des statistiques comparatives
def create_stats_comparison_df(fighter_a, fighter_b):
    stats_to_compare = [
        ('Victoires', 'wins', False), 
        ('Défaites', 'losses', True),  # True = une valeur plus basse est meilleure
        ('Ratio V/D', lambda f: f['wins']/(f['wins']+f['losses']) if f['wins']+f['losses'] > 0 else 0, False),
        ('Âge', 'age', True),
        ('Taille (cm)', 'height', False),
        ('Poids (kg)', 'weight', False),
        ('Allonge (cm)', 'reach', False),
        ('Frappes/min', 'SLpM', False),
        ('Précision frappes', 'sig_str_acc', False),
        ('Frappes reçues/min', 'SApM', True),
        ('Défense frappes', 'str_def', False),
        ('Takedowns/combat', 'td_avg', False),
        ('Précision takedowns', 'td_acc', False),
        ('Défense takedowns', 'td_def', False),
        ('Soumissions/combat', 'sub_avg', False)
    ]
    
    data = []
    
    for stat_name, stat_key, lower_better in stats_to_compare:
        if callable(stat_key):
            a_value = stat_key(fighter_a)
            b_value = stat_key(fighter_b)
        else:
            a_value = fighter_a.get(stat_key, 0)
            b_value = fighter_b.get(stat_key, 0)
        
        # Déterminer qui a l'avantage
        if isinstance(a_value, (int, float)) and isinstance(b_value, (int, float)):
            if lower_better:
                advantage = fighter_a['name'] if a_value < b_value else fighter_b['name'] if b_value < a_value else "Égal"
            else:
                advantage = fighter_a['name'] if a_value > b_value else fighter_b['name'] if b_value > a_value else "Égal"
        else:
            advantage = "Égal"
        
        # Formatage pour l'affichage
        if isinstance(a_value, float) and isinstance(b_value, float):
            a_display = f"{a_value:.2f}"
            b_display = f"{b_value:.2f}"
        else:
            a_display = str(a_value)
            b_display = str(b_value)
        
        data.append({
            'Statistique': stat_name,
            fighter_a['name']: a_display,
            fighter_b['name']: b_display,
            'Avantage': advantage
        })
    
    return pd.DataFrame(data)

# NOUVELLES FONCTIONS POUR LA GESTION DE BANKROLL ET PARIS

def init_bankroll():
    """
    Initialise ou charge la bankroll depuis le fichier
    """
    bets_dir = "bets"
    bankroll_file = os.path.join(bets_dir, "bankroll.csv")
    
    # Créer le dossier s'il n'existe pas
    if not os.path.exists(bets_dir):
        os.makedirs(bets_dir)
    
    # Charger la bankroll si le fichier existe
    if os.path.exists(bankroll_file):
        bankroll_df = pd.read_csv(bankroll_file)
        if not bankroll_df.empty:
            return bankroll_df.iloc[-1]["amount"]
    
    # Sinon, initialiser le fichier avec une valeur par défaut
    bankroll_df = pd.DataFrame({
        "date": [datetime.datetime.now().strftime("%Y-%m-%d")],
        "amount": [1000],  # Valeur par défaut
        "action": ["initial"],
        "note": ["Bankroll initiale"]
    })
    
    # Sauvegarder le fichier
    bankroll_df.to_csv(bankroll_file, index=False)
    return 1000

def init_bets_file():
    """
    Initialise le fichier de paris s'il n'existe pas
    """
    bets_dir = "bets"
    bets_file = os.path.join(bets_dir, "bets.csv")
    
    # Créer le dossier s'il n'existe pas
    if not os.path.exists(bets_dir):
        os.makedirs(bets_dir)
    
    # Créer le fichier s'il n'existe pas
    if not os.path.exists(bets_file):
        columns = ["bet_id", "date_placed", "event_name", "event_date", 
                  "fighter_red", "fighter_blue", "pick", "odds", 
                  "stake", "kelly_fraction", "model_probability", 
                  "status", "result", "profit", "roi"]
        
        empty_df = pd.DataFrame(columns=columns)
        empty_df.to_csv(bets_file, index=False)

def calculate_kelly(prob, odds, bankroll, fraction=1):
    """
    Calcule la mise optimale selon le critère de Kelly fractionné
    
    Args:
        prob: probabilité de gain selon le modèle ML
        odds: cote décimale (européenne)
        bankroll: montant total disponible
        fraction: diviseur Kelly (ex: 4 pour Kelly/4)
    
    Returns:
        Montant recommandé à parier
    """
    b = odds - 1  # gain net par unité misée
    q = 1 - prob  # probabilité de perte
    
    # Formule de Kelly: (p*b - q) / b
    kelly_percentage = (prob * b - q) / b
    
    # Si Kelly est négatif, ne pas parier
    if kelly_percentage <= 0:
        return 0
    
    # Appliquer la fraction Kelly
    fractional_kelly = kelly_percentage / fraction
    
    # Calculer la mise recommandée
    recommended_stake = bankroll * fractional_kelly
    
    return round(recommended_stake, 2)

def update_bet_result(bet_id, result, current_bankroll):
    """
    Met à jour le résultat d'un pari existant et ajuste la bankroll
    
    Args:
        bet_id: Identifiant du pari à mettre à jour
        result: Résultat du pari ('win', 'loss', 'void')
        current_bankroll: Bankroll actuelle
        
    Returns:
        Nouveau solde de la bankroll après la mise à jour
    """
    bets_dir = "bets"
    bets_file = os.path.join(bets_dir, "bets.csv")
    bankroll_file = os.path.join(bets_dir, "bankroll.csv")
    
    if not os.path.exists(bets_file):
        st.error("Fichier de paris introuvable.")
        return current_bankroll
    
    # Charger les fichiers
    bets_df = pd.read_csv(bets_file)
    bankroll_df = pd.read_csv(bankroll_file)
    
    # Vérifier si le pari existe
    if bet_id not in bets_df["bet_id"].values:
        st.error(f"Pari #{bet_id} introuvable.")
        return current_bankroll
    
    # Récupérer les informations du pari
    bet_row = bets_df[bets_df["bet_id"] == bet_id].iloc[0]
    stake = float(bet_row["stake"])
    odds = float(bet_row["odds"])
    
    # Calculer le profit
    if result == "win":
        profit = stake * (odds - 1)
        roi = (profit / stake) * 100
    elif result == "loss":
        profit = -stake
        roi = -100
    else:  # void
        profit = 0
        roi = 0
    
    # Mettre à jour le pari
    bets_df.loc[bets_df["bet_id"] == bet_id, "status"] = "closed"
    bets_df.loc[bets_df["bet_id"] == bet_id, "result"] = result
    bets_df.loc[bets_df["bet_id"] == bet_id, "profit"] = profit
    bets_df.loc[bets_df["bet_id"] == bet_id, "roi"] = roi
    bets_df.to_csv(bets_file, index=False)
    
    # Mettre à jour la bankroll
    new_bankroll = current_bankroll + profit
    new_entry = pd.DataFrame({
        "date": [datetime.datetime.now().strftime("%Y-%m-%d")],
        "amount": [new_bankroll],
        "action": ["update"],
        "note": [f"Résultat pari #{bet_id}: {result}"]
    })
    bankroll_df = pd.concat([bankroll_df, new_entry], ignore_index=True)
    bankroll_df.to_csv(bankroll_file, index=False)
    
    return new_bankroll

def delete_bet(bet_id):
    """
    Supprime un pari du fichier historique
    
    Args:
        bet_id: Identifiant du pari à supprimer
    
    Returns:
        True si la suppression a réussi, False sinon
    """
    bets_dir = "bets"
    bets_file = os.path.join(bets_dir, "bets.csv")
    
    if not os.path.exists(bets_file):
        st.error("Fichier de paris introuvable.")
        return False
    
    # Charger le fichier
    bets_df = pd.read_csv(bets_file)
    
    # Vérifier si le pari existe
    if bet_id not in bets_df["bet_id"].values:
        st.error(f"Pari #{bet_id} introuvable.")
        return False
    
    # Vérifier si c'est un pari "fermé" (on ne peut pas supprimer des paris déjà réglés)
    bet_row = bets_df[bets_df["bet_id"] == bet_id].iloc[0]
    if bet_row["status"] == "closed":
        st.error("Impossible de supprimer un pari déjà réglé.")
        return False
    
    # Supprimer le pari
    bets_df = bets_df[bets_df["bet_id"] != bet_id]
    bets_df.to_csv(bets_file, index=False)
    
    return True

def add_manual_bet(event_name, event_date, fighter_red, fighter_blue, pick, odds, stake, model_probability=None, kelly_fraction=None):
    """
    Ajoute un pari manuellement à l'historique
    
    Args:
        event_name: Nom de l'événement
        event_date: Date de l'événement
        fighter_red: Nom du combattant rouge
        fighter_blue: Nom du combattant bleu
        pick: Combattant sur lequel le pari est placé
        odds: Cote du pari
        stake: Montant misé
        model_probability: Probabilité prédite par le modèle (optionnel)
        kelly_fraction: Fraction Kelly utilisée (optionnel)
        
    Returns:
        True si l'ajout a réussi, False sinon
    """
    bets_dir = "bets"
    bets_file = os.path.join(bets_dir, "bets.csv")
    
    # Créer le dossier et le fichier s'ils n'existent pas
    if not os.path.exists(bets_dir):
        os.makedirs(bets_dir)
    
    # Charger le fichier des paris ou en créer un nouveau
    if os.path.exists(bets_file):
        bets_df = pd.read_csv(bets_file)
        # Générer un nouveau bet_id
        new_id = 1 if bets_df.empty else bets_df["bet_id"].max() + 1
    else:
        columns = ["bet_id", "date_placed", "event_name", "event_date", 
                  "fighter_red", "fighter_blue", "pick", "odds", 
                  "stake", "kelly_fraction", "model_probability", 
                  "status", "result", "profit", "roi"]
        bets_df = pd.DataFrame(columns=columns)
        new_id = 1
    
    # Formater la date
    if isinstance(event_date, str):
        event_date_str = event_date
    else:
        event_date_str = event_date.strftime("%Y-%m-%d")
    
    # Créer le nouveau pari
    new_bet = pd.DataFrame({
        "bet_id": [new_id],
        "date_placed": [datetime.datetime.now().strftime("%Y-%m-%d")],
        "event_name": [event_name],
        "event_date": [event_date_str],
        "fighter_red": [fighter_red],
        "fighter_blue": [fighter_blue],
        "pick": [pick],
        "odds": [odds],
        "stake": [stake],
        "kelly_fraction": [kelly_fraction if kelly_fraction is not None else "N/A"],
        "model_probability": [model_probability if model_probability is not None else "N/A"],
        "status": ["open"],
        "result": [""],
        "profit": [0],
        "roi": [0]
    })
    
    # Ajouter le pari
    bets_df = pd.concat([bets_df, new_bet], ignore_index=True)
    bets_df.to_csv(bets_file, index=False)
    
    return True

def get_betting_summary(bets_df):
    """
    Génère un résumé des statistiques de paris
    
    Args:
        bets_df: DataFrame contenant l'historique des paris
        
    Returns:
        Un dictionnaire avec les statistiques résumées
    """
    if bets_df.empty:
        return {
            "total_bets": 0,
            "open_bets": 0,
            "closed_bets": 0,
            "wins": 0,
            "losses": 0,
            "voids": 0,
            "win_rate": 0,
            "total_staked": 0,
            "total_profit": 0,
            "roi": 0
        }
    
    # Filtrer les paris fermés
    closed_bets = bets_df[bets_df["status"] == "closed"]
    open_bets = bets_df[bets_df["status"] == "open"]
    
    # Nombre de paris
    total_bets = len(bets_df)
    open_bets_count = len(open_bets)
    closed_bets_count = len(closed_bets)
    
    # Résultats des paris fermés
    wins = len(closed_bets[closed_bets["result"] == "win"])
    losses = len(closed_bets[closed_bets["result"] == "loss"])
    voids = len(closed_bets[closed_bets["result"] == "void"])
    
    # Taux de réussite
    win_rate = wins / max(wins + losses, 1) * 100
    
    # Montants financiers
    total_staked = closed_bets["stake"].sum() + open_bets["stake"].sum()
    total_profit = closed_bets["profit"].sum()
    
    # ROI global
    roi = total_profit / max(closed_bets["stake"].sum(), 1) * 100
    
    return {
        "total_bets": total_bets,
        "open_bets": open_bets_count,
        "closed_bets": closed_bets_count,
        "wins": wins,
        "losses": losses,
        "voids": voids,
        "win_rate": win_rate,
        "total_staked": total_staked,
        "total_profit": total_profit,
        "roi": roi
    }

# Initialiser l'état de session pour éviter le rechargement de page
if 'kelly_strategy' not in st.session_state:
    st.session_state.kelly_strategy = "Kelly/4"
    
if 'use_manual_stake' not in st.session_state:
    st.session_state.use_manual_stake = False
    
if 'event_name' not in st.session_state:
    st.session_state.event_name = "UFC Fight Night"
    
if 'event_date' not in st.session_state:
    st.session_state.event_date = datetime.datetime.now()
    
if 'manual_stake' not in st.session_state:
    st.session_state.manual_stake = 0.0

# Fonctions de callback pour mettre à jour l'état de session
def update_kelly_strategy(value):
    st.session_state.kelly_strategy = value
    
def update_use_manual_stake(value):
    st.session_state.use_manual_stake = value
    
def update_event_name(value):
    st.session_state.event_name = value
    
def update_event_date(value):
    st.session_state.event_date = value
    
def update_manual_stake(value):
    st.session_state.manual_stake = value

# FONCTION PRINCIPALE

def main():
    # Titre principal
    st.markdown('<div class="main-title">🥊 Prédicteur de Combats UFC 🥊</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Analysez et prédisez l\'issue des affrontements</div>', unsafe_allow_html=True)
    
    # Initialiser/Charger la bankroll et les paris
    current_bankroll = init_bankroll()  
    init_bets_file()
    
    # Chargement des données
    fighter_stats_path = 'fighters_stats.txt'
    
    if os.path.exists(fighter_stats_path):
        fighters = load_fighters_stats(fighter_stats_path)
        # Ne pas afficher le nombre de combattants chargés
    else:
        # Utiliser les deux combattants d'exemple
        fighters = [
            {
                'name': 'Josh Emmett',
                'wins': 19,
                'losses': 5,
                'height': 167.64,
                'weight': 65.77,
                'reach': 177.8,
                'stance': 'Orthodox',
                'age': 40,
                'SLpM': 3.75,
                'sig_str_acc': 0.35,
                'SApM': 4.46,
                'str_def': 0.6,
                'td_avg': 1.09,
                'td_acc': 0.37,
                'td_def': 0.46,
                'sub_avg': 0.1
            },
            {
                'name': 'Lerone Murphy',
                'wins': 16,
                'losses': 0,
                'height': 175.26,
                'weight': 65.77,
                'reach': 185.42,
                'stance': 'Orthodox',
                'age': 33,
                'SLpM': 4.53,
                'sig_str_acc': 0.54,
                'SApM': 2.48,
                'str_def': 0.61,
                'td_avg': 1.45,
                'td_acc': 0.54,
                'td_def': 0.52,
                'sub_avg': 0.5
            }
        ]
        # Ne pas afficher d'info sur les exemples
    
    # Appliquer la déduplication
    fighters = deduplicate_fighters(fighters)
    # Ne pas afficher le nombre après déduplication
    
    # Créer un dictionnaire pour accéder rapidement aux statistiques des combattants
    fighters_dict = {fighter['name']: fighter for fighter in fighters}
    
    # Créer les onglets principaux
    tabs = st.tabs(["🎯 Prédictions", "💰 Gestion de Bankroll", "📊 Historique & Performance"])
    
    # Onglet de prédiction
    with tabs[0]:
        # Interface de sélection des combattants
        st.sidebar.markdown("## Sélection des combattants")
        
        # Message d'avertissement sur l'importance de l'ordre des combattants
        st.sidebar.markdown("""
        <div class="info-box">
            <b>⚠️ Important :</b> L'ordre des combattants (Rouge/Bleu) influence les prédictions. 
            Traditionnellement, le combattant mieux classé ou favori est placé dans le coin rouge.
        </div>
        """, unsafe_allow_html=True)
        
        # Liste des noms de combattants
        fighter_names = sorted([fighter['name'] for fighter in fighters])
        
        # Sélection des combattants avec barre de recherche intégrée
        st.sidebar.markdown("### 🔴 Combattant Rouge")
        fighter_a_name = st.sidebar.selectbox(
            "Sélectionner combattant rouge",
            options=fighter_names,
            key="fighter_a_selectbox"
        )
        
        # Sélection du combattant bleu (en excluant le combattant rouge)
        st.sidebar.markdown("### 🔵 Combattant Bleu")
        fighter_b_options = [name for name in fighter_names if name != fighter_a_name]
        fighter_b_name = st.sidebar.selectbox(
            "Sélectionner combattant bleu",
            options=fighter_b_options,
            key="fighter_b_selectbox"
        )
        
        # Options de paris
        st.sidebar.markdown("## 💰 Options de paris")
        odds_a = st.sidebar.number_input("Cote Rouge", min_value=1.01, value=2.0, step=0.05, format="%.2f", key="odds_a_input")
        odds_b = st.sidebar.number_input("Cote Bleu", min_value=1.01, value=1.8, step=0.05, format="%.2f", key="odds_b_input")
        
        # Section de gestion de bankroll
        st.sidebar.markdown("---")
        st.sidebar.markdown("## 💼 Ma Bankroll")
        displayed_bankroll = st.sidebar.number_input(
            "Bankroll actuelle (€)",
            min_value=0.0,
            value=float(current_bankroll),
            step=10.0,
            format="%.2f",
            key="bankroll_input"
        )

        # Option pour modifier la bankroll
        if st.sidebar.button("Mettre à jour la bankroll", key="update_bankroll_btn"):
            if displayed_bankroll != current_bankroll:
                bets_dir = "bets"
                bankroll_file = os.path.join(bets_dir, "bankroll.csv")
                
                # Charger le fichier existant
                if os.path.exists(bankroll_file):
                    bankroll_df = pd.read_csv(bankroll_file)
                else:
                    bankroll_df = pd.DataFrame(columns=["date", "amount", "action", "note"])
                
                # Ajouter la nouvelle entrée
                new_entry = pd.DataFrame({
                    "date": [datetime.datetime.now().strftime("%Y-%m-%d")],
                    "amount": [displayed_bankroll],
                    "action": ["update"],
                    "note": ["Mise à jour manuelle"]
                })
                
                bankroll_df = pd.concat([bankroll_df, new_entry], ignore_index=True)
                bankroll_df.to_csv(bankroll_file, index=False)
                
                st.sidebar.success(f"Bankroll mise à jour : {displayed_bankroll:.2f} €")
                current_bankroll = displayed_bankroll
        
        # Bouton de prédiction
        predict_btn = st.sidebar.button("🥊 Prédire le combat", type="primary", key="predict_btn")
        
        # Récupérer les statistiques des combattants sélectionnés
        fighter_a = fighters_dict.get(fighter_a_name)
        fighter_b = fighters_dict.get(fighter_b_name)
        
        # Vérifier si on peut faire une prédiction
        if predict_btn and fighter_a and fighter_b:
            if fighter_a_name == fighter_b_name:
                st.error("Veuillez sélectionner deux combattants différents.")
            else:
                # Faire les prédictions avec les deux méthodes
                classic_prediction, ml_prediction = predict_both_methods(
                    fighter_a, 
                    fighter_b,
                    odds_a=odds_a,
                    odds_b=odds_b
                )
                
                # Afficher les résultats des deux prédictions
                st.markdown("""
                <div style="text-align:center;">
                    <h2>🔮 Prédictions du combat 🔮</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Créer le graphique comparatif des probabilités pour les deux méthodes en un seul
                if ml_prediction:
                    # Créer un DataFrame pour le graphique comparatif
                    proba_data = pd.DataFrame({
                        'Combattant': [fighter_a_name, fighter_b_name],
                        'Statistique': [classic_prediction['red_probability'], classic_prediction['blue_probability']],
                        'Machine Learning': [ml_prediction['red_probability'], ml_prediction['blue_probability']]
                    })
                    
                    # Créer un graphique qui montre les deux probabilités côte à côte
                    fig = go.Figure()
                    
                    # Ajouter les barres pour chaque méthode
                    fig.add_trace(go.Bar(
                        x=proba_data['Combattant'],
                        y=proba_data['Statistique'],
                        name='Prédiction Statistique',
                        marker_color='#2196F3',
                        text=[f"{proba:.2f}" for proba in proba_data['Statistique']],
                        textposition='auto'
                    ))
                    
                    fig.add_trace(go.Bar(
                        x=proba_data['Combattant'],
                        y=proba_data['Machine Learning'],
                        name='Prédiction ML',
                        marker_color='#4CAF50',
                        text=[f"{proba:.2f}" for proba in proba_data['Machine Learning']],
                        textposition='auto'
                    ))
                    
                    # Configurer la mise en page
                    fig.update_layout(
                        title="Probabilités de victoire selon les deux méthodes",
                        xaxis_title="",
                        yaxis_title="Probabilité",
                        yaxis=dict(range=[0, 1]),
                        legend_title="Méthode",
                        height=400,
                        barmode='group'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Si seulement la méthode statistique est disponible
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=[fighter_a_name, fighter_b_name],
                        y=[classic_prediction['red_probability'], classic_prediction['blue_probability']],
                        marker_color=['red', 'blue'],
                        text=[f"{classic_prediction['red_probability']:.2f}", f"{classic_prediction['blue_probability']:.2f}"],
                        textposition='auto'
                    ))
                    
                    fig.update_layout(
                        title="Probabilités de victoire (Méthode Statistique)",
                        xaxis_title="",
                        yaxis_title="Probabilité",
                        yaxis=dict(range=[0, 1]),
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Créer deux colonnes pour les deux prédictions
                pred_cols = st.columns(2 if ml_prediction else 1)
                
                # Afficher la prédiction statistique
                with pred_cols[0]:
                    winner_color = "red" if classic_prediction['prediction'] == 'Red' else "blue"
                    winner_name = classic_prediction['winner_name']
                    
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h3 style="text-align:center;"><span class="classic-badge">Prédiction statistique</span></h3>
                        <h3 style="text-align:center; color:{winner_color};" class="winner">
                            🏆 {winner_name} 🏆
                        </h3>
                        <p style="text-align:center; font-size:1.2em;">
                            Probabilité: <span class="red-fighter">{classic_prediction['red_probability']:.2f}</span> pour {fighter_a_name}, 
                            <span class="blue-fighter">{classic_prediction['blue_probability']:.2f}</span> pour {fighter_b_name}
                        </p>
                        <p style="text-align:center;">Niveau de confiance: <b>{classic_prediction['confidence']}</b></p>
                    </div>
                    """, unsafe_allow_html=True)

                # Afficher la prédiction ML si disponible
                if ml_prediction:
                    with pred_cols[1]:
                        winner_color_ml = "red" if ml_prediction['prediction'] == 'Red' else "blue"
                        winner_name_ml = ml_prediction['winner_name']
                        
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h3 style="text-align:center;"><span class="ml-badge">Prédiction Machine Learning</span></h3>
                            <h3 style="text-align:center; color:{winner_color_ml};" class="winner">
                                🏆 {winner_name_ml} 🏆
                            </h3>
                            <p style="text-align:center; font-size:1.2em;">
                                Probabilité: <span class="red-fighter">{ml_prediction['red_probability']:.2f}</span> pour {fighter_a_name}, 
                                <span class="blue-fighter">{ml_prediction['blue_probability']:.2f}</span> pour {fighter_b_name}
                            </p>
                            <p style="text-align:center;">Niveau de confiance: <b>{ml_prediction['confidence']}</b></p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Message de convergence/divergence si les deux méthodes sont disponibles
                if ml_prediction:
                    same_prediction = classic_prediction['prediction'] == ml_prediction['prediction']
                    agreement_message = "✅ Les deux méthodes prédisent le même vainqueur!" if same_prediction else "⚠️ Les méthodes prédisent des vainqueurs différents!"
                    agreement_color = "green" if same_prediction else "orange"
                    
                    st.markdown(f"""
                    <div style="text-align:center; margin-top:10px; margin-bottom:20px;">
                        <h3 style="color:{agreement_color};">{agreement_message}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Analyse des paris (utiliser les deux méthodes si disponibles)
                if 'betting' in classic_prediction:
                    betting_classic = classic_prediction['betting']
                    betting_ml = ml_prediction.get('betting') if ml_prediction else None
                    
                    st.markdown("""
                    <div class="divider"></div>
                    <div style="text-align:center;">
                        <h2>💰 Analyse des paris 💰</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Analyse des paris pour les deux combattants avec Streamlit natif (pas d'HTML)
                    col1, col2 = st.columns(2)
                    
                    # Combattant Rouge - Carte de paris
                    with col1:
                        st.subheader(f"🔴 {fighter_a_name}")
                        
                        # Créer un DataFrame pour une présentation simple
                        red_data = {
                            "Métrique": [
                                "Cote", 
                                "Probabilité implicite", 
                                "Probabilité statistique",
                                "Probabilité ML" if betting_ml else None,
                                "Avantage (stat.)",
                                "Valeur espérée (stat.)",
                                "Recommandation stat.",
                                "Recommandation ML" if betting_ml else None
                            ],
                            "Valeur": [
                                f"{betting_classic['odds_red']:.2f}",
                                f"{betting_classic['implied_prob_red']:.2f}",
                                f"{classic_prediction['red_probability']:.2f}",
                                f"{ml_prediction['red_probability']:.2f}" if betting_ml else None,
                                f"{betting_classic['edge_red']*100:.1f}%",
                                f"{betting_classic['ev_red']*100:.1f}%",
                                betting_classic['recommendation_red'],
                                betting_ml['recommendation_red'] if betting_ml else None
                            ]
                        }
                        
                        # Filtrer les lignes None
                        red_df = pd.DataFrame(red_data)
                        red_df = red_df.dropna()
                        
                        # Afficher le DataFrame stylisé
                        st.dataframe(
                            red_df,
                            column_config={
                                "Métrique": st.column_config.TextColumn("Métrique"),
                                "Valeur": st.column_config.TextColumn("Valeur")
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                        
                        # Affichage manuel des recommandations
                        st.markdown("**Recommandation statistique:**")
                        rec_class = "favorable" if betting_classic['recommendation_red'] == "Favorable" else "neutral" if betting_classic['recommendation_red'] == "Neutre" else "unfavorable"
                        st.markdown(f"<span class='{rec_class}'>{betting_classic['recommendation_red']}</span>", unsafe_allow_html=True)
                        
                        if betting_ml:
                            st.markdown("**Recommandation ML:**")
                            rec_ml_class = "favorable" if betting_ml['recommendation_red'] == "Favorable" else "neutral" if betting_ml['recommendation_red'] == "Neutre" else "unfavorable"
                            st.markdown(f"<span class='{rec_ml_class}'>{betting_ml['recommendation_red']}</span>", unsafe_allow_html=True)
                    
                    # Combattant Bleu - Carte de paris
                    with col2:
                        st.subheader(f"🔵 {fighter_b_name}")
                        
                        # Créer un DataFrame pour une présentation simple
                        blue_data = {
                            "Métrique": [
                                "Cote", 
                                "Probabilité implicite", 
                                "Probabilité statistique",
                                "Probabilité ML" if betting_ml else None,
                                "Avantage (stat.)",
                                "Valeur espérée (stat.)",
                                "Recommandation stat.",
                                "Recommandation ML" if betting_ml else None
                            ],
                            "Valeur": [
                                f"{betting_classic['odds_blue']:.2f}",
                                f"{betting_classic['implied_prob_blue']:.2f}",
                                f"{classic_prediction['blue_probability']:.2f}",
                                f"{ml_prediction['blue_probability']:.2f}" if betting_ml else None,
                                f"{betting_classic['edge_blue']*100:.1f}%",
                                f"{betting_classic['ev_blue']*100:.1f}%",
                                betting_classic['recommendation_blue'],
                                betting_ml['recommendation_blue'] if betting_ml else None
                            ]
                        }
                        
                        # Filtrer les lignes None
                        blue_df = pd.DataFrame(blue_data)
                        blue_df = blue_df.dropna()
                        
                        # Afficher le DataFrame stylisé
                        st.dataframe(
                            blue_df,
                            column_config={
                                "Métrique": st.column_config.TextColumn("Métrique"),
                                "Valeur": st.column_config.TextColumn("Valeur")
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                        
                        # Affichage manuel des recommandations
                        st.markdown("**Recommandation statistique:**")
                        rec_class = "favorable" if betting_classic['recommendation_blue'] == "Favorable" else "neutral" if betting_classic['recommendation_blue'] == "Neutre" else "unfavorable"
                        st.markdown(f"<span class='{rec_class}'>{betting_classic['recommendation_blue']}</span>", unsafe_allow_html=True)
                        
                        if betting_ml:
                            st.markdown("**Recommandation ML:**")
                            rec_ml_class = "favorable" if betting_ml['recommendation_blue'] == "Favorable" else "neutral" if betting_ml['recommendation_blue'] == "Neutre" else "unfavorable"
                            st.markdown(f"<span class='{rec_ml_class}'>{betting_ml['recommendation_blue']}</span>", unsafe_allow_html=True)
                
                # Afficher les statistiques comparatives
                st.markdown("""
                <div class="divider"></div>
                <div style="text-align:center;">
                    <h2>📊 Statistiques comparatives 📊</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Création du DataFrame des statistiques comparatives
                stats_df = create_stats_comparison_df(fighter_a, fighter_b)
                
                # Appliquer un style conditionnel pour mettre en évidence les avantages
                def highlight_advantage(row):
                    styles = [''] * len(row)
                    advantage = row['Avantage']
                    
                    if advantage == fighter_a_name:
                        styles[1] = 'background-color: rgba(255, 0, 0, 0.2); font-weight: bold;'
                    elif advantage == fighter_b_name:
                        styles[2] = 'background-color: rgba(0, 0, 255, 0.2); font-weight: bold;'
                    
                    return styles
                
                # Appliquer le style et afficher
                styled_df = stats_df.style.apply(highlight_advantage, axis=1)
                st.dataframe(styled_df, use_container_width=True, height=600)
                
                # Visualisations
                st.markdown("""
                <div class="divider"></div>
                <div style="text-align:center; margin-top:30px;">
                    <h2>📈 Visualisations des performances 📈</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Disposer les graphiques en deux colonnes
                col1, col2 = st.columns(2)
                
                with col1:
                    # Graphique radar
                    radar_fig = create_radar_chart(fighter_a, fighter_b)
                    st.plotly_chart(radar_fig, use_container_width=True)
                
                with col2:
                    # Graphique des forces et faiblesses
                    strengths_fig = create_strengths_weaknesses_chart(fighter_a, fighter_b)
                    st.plotly_chart(strengths_fig, use_container_width=True)
                
                # Style de combat
                style_fig = create_style_analysis_chart(fighter_a, fighter_b)
                st.plotly_chart(style_fig, use_container_width=True)
                
                # Calcul Kelly et enregistrement des paris
                st.markdown("""
                <div class="divider"></div>
                <div style="text-align:center;">
                    <h2>💰 Optimisation des mises 💰</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Si la bankroll n'est pas initialisée correctement, afficher un message d'erreur
                if current_bankroll <= 0:
                    st.error("⚠️ Vous devez initialiser votre bankroll avant de pouvoir utiliser les recommandations de mise. Utilisez le champ 'Bankroll actuelle (€)' dans la barre latérale.")
                else:
                    # Déterminer le favori selon le modèle ML
                    if ml_prediction:
                        if ml_prediction['prediction'] == 'Red':
                            favorite_name = fighter_a_name
                            favorite_odds = odds_a
                            favorite_prob = ml_prediction['red_probability']
                        else:
                            favorite_name = fighter_b_name
                            favorite_odds = odds_b
                            favorite_prob = ml_prediction['blue_probability']
                        
                        # Afficher le favori ML et sa probabilité
                        st.subheader(f"Recommandations pour le favori ML: {favorite_name}")
                        st.info(f"Probabilité ML: {favorite_prob:.2f} | Cote: {favorite_odds:.2f}")
                        
                        # Calculer les mises Kelly pour différentes fractions
                        kelly_options = {
                            "Kelly pur": 1,
                            "Kelly/2": 2,
                            "Kelly/3": 3,
                            "Kelly/4": 4,
                            "Kelly/5": 5,
                            "Kelly/10": 10
                        }
                        
                        # Calculer les mises recommandées
                        kelly_stakes = {
                            name: calculate_kelly(favorite_prob, favorite_odds, current_bankroll, fraction) 
                            for name, fraction in kelly_options.items()
                        }
                        
                        # Afficher toutes les options Kelly dans un tableau
                        st.markdown("""
                        <p>Le tableau ci-dessous présente les montants à miser selon différentes stratégies Kelly, basés sur la probabilité ML et votre bankroll actuelle:</p>
                        """, unsafe_allow_html=True)
                        
                        # Créer un tableau avec les mises Kelly
                        kelly_data = {
                            "Stratégie": list(kelly_options.keys()),
                            "Mise (€)": [kelly_stakes[name] for name in kelly_options.keys()],
                            "% de la bankroll": [f"{kelly_stakes[name]/current_bankroll*100:.1f}%" for name in kelly_options.keys()],
                            "Gain potentiel (€)": [f"{kelly_stakes[name] * (favorite_odds-1):.2f}" for name in kelly_options.keys()]
                        }
                        
                        df_kelly = pd.DataFrame(kelly_data)
                        st.dataframe(df_kelly, use_container_width=True)
                        

                    else:
                        st.warning("Le modèle ML n'est pas disponible. Les recommandations Kelly ne peuvent pas être calculées.")
        
        else:
            # Message d'accueil - CORRIGÉ
            st.markdown("""
            <div style="background-color:rgba(240, 242, 246, 0.7); padding:20px; border-radius:10px; text-align:center;">
                <h2>Bienvenue sur le Prédicteur de Combats UFC!</h2>            
                <p style="font-size:1.2em;">Sélectionnez deux combattants dans le menu latéral et cliquez sur "Prédire le combat" pour obtenir une analyse complète.</p>
                <p>Vous pouvez également entrer les cotes proposées par les bookmakers pour recevoir des recommandations de paris.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Note sur l'importance de l'ordre des combattants
            st.markdown("""
            <div class="info-box">
                <h3>⚠️ L'ordre des combattants est important!</h3>
                <p>La position des combattants (coin Rouge vs Bleu) peut influencer significativement les prédictions, particulièrement avec le modèle ML.</p>
                <p>Traditionnellement, le combattant favori ou mieux classé est placé dans le coin rouge. Pour obtenir les résultats les plus précis, suivez cette convention.</p>
                <p>Si vous inversez les positions, les probabilités de victoire peuvent changer considérablement.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Nouvelle fonctionnalité
            st.markdown("""
            <div style="background-color:rgba(76, 175, 80, 0.1); padding:15px; border-radius:10px; margin-top:20px;">
                <h3>🔄 Nouvelle fonctionnalité: Prédictions comparatives!</h3>
                <p>L'application affiche maintenant simultanément les prédictions des deux méthodes:</p>
                <ul>
                    <li><b>🤖 Machine Learning:</b> Prédiction basée sur un modèle entraîné sur des milliers de combats</li>
                    <li><b>📊 Calcul statistique:</b> Prédiction basée sur une formule utilisant les statistiques des combattants</li>
                </ul>
                <p>Cette double prédiction vous permet de comparer les résultats et d'avoir une vision plus complète des chances de victoire.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Explication des fonctionnalités
            st.markdown("""
            ### Comment utiliser l'application:
            
            1. **Sélectionnez vos combattants**: Utilisez les menus déroulants pour choisir les combattants rouge et bleu que vous souhaitez comparer.
            
            2. **Respectez les positions**: Pour des prédictions plus précises, placez le combattant favori ou mieux classé dans le coin rouge.
            
            3. **Entrez les cotes** (optionnel): Si vous souhaitez analyser les opportunités de paris, entrez les cotes proposées par les bookmakers.
            
            4. **Lancez la prédiction**: Cliquez sur le bouton "Prédire le combat" pour obtenir l'analyse complète avec les deux méthodes de prédiction.
            
            5. **Comparez les résultats**: Analysez les différences entre les prédictions ML et statistiques pour une meilleure compréhension.
            
            6. **Explorez les visualisations**: Consultez les graphiques et tableaux pour comprendre les forces et faiblesses de chaque combattant.
            """)

            # Afficher les informations sur le modèle ML
            model, _, _ = load_ml_model()
            ml_available = model is not None
            
            if ml_available:
                st.markdown("""
                <div style="background-color:rgba(76, 175, 80, 0.1); padding:15px; border-radius:10px; margin-top:20px;">
                    <h3>✅ Modèle ML détecté!</h3>
                    <p>Le modèle de machine learning a été correctement chargé et est prêt à être utilisé pour des prédictions.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background-color:rgba(244, 67, 54, 0.1); padding:15px; border-radius:10px; margin-top:20px;">
                    <h3>⚠️ Modèle ML non détecté</h3>
                    <p>Le modèle de machine learning n'a pas été trouvé. Assurez-vous que les fichiers suivants sont présents dans le même répertoire que cette application:</p>
                    <ul>
                        <li><code>ufc_prediction_model.joblib</code> ou <code>ufc_prediction_model.pkl</code></li>
                    </ul>
                    <p>Seule la méthode de prédiction statistique classique sera disponible.</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Onglet de gestion de bankroll
    with tabs[1]:
        st.markdown("""
        <div style="text-align:center;">
            <h2>💰 Gestion de bankroll et paris 💰</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Afficher la bankroll actuelle
        st.metric("Bankroll actuelle", f"{current_bankroll:.2f} €", delta=None)
        st.markdown("---")
        
        # Interface pour ajuster la bankroll
        st.subheader("Ajuster la bankroll")
        adjust_cols = st.columns(3)
        
        with adjust_cols[0]:
            adjustment_amount = st.number_input(
                "Montant (€)",
                min_value=0.0,
                step=10.0,
                format="%.2f",
                key="bankroll_adjustment_amount"
            )
        
        with adjust_cols[1]:
            adjustment_type = st.selectbox(
                "Type d'opération",
                options=["Dépôt", "Retrait", "Définir montant exact"],
                key="bankroll_adjustment_type"
            )
        
        with adjust_cols[2]:
            adjustment_note = st.text_input(
                "Note (optionnel)",
                value="",
                key="bankroll_adjustment_note"
            )
        
        if st.button("Valider l'ajustement", type="primary", key="validate_bankroll_adjust"):
            # Calculer la nouvelle bankroll
            if adjustment_type == "Dépôt":
                new_bankroll = current_bankroll + adjustment_amount
                action = "deposit"
                if not adjustment_note:
                    adjustment_note = "Dépôt"
            elif adjustment_type == "Retrait":
                if adjustment_amount > current_bankroll:
                    st.error(f"Montant du retrait ({adjustment_amount:.2f} €) supérieur à la bankroll actuelle ({current_bankroll:.2f} €)")
                    new_bankroll = current_bankroll
                    action = None
                else:
                    new_bankroll = current_bankroll - adjustment_amount
                    action = "withdraw"
                    if not adjustment_note:
                        adjustment_note = "Retrait"
            else:  # "Définir montant exact"
                new_bankroll = adjustment_amount
                action = "update"
                if not adjustment_note:
                    adjustment_note = "Mise à jour manuelle"
            
            # Mettre à jour la bankroll si nécessaire
            if action and new_bankroll != current_bankroll:
                bets_dir = "bets"
                bankroll_file = os.path.join(bets_dir, "bankroll.csv")
                
                # Charger le fichier existant
                if os.path.exists(bankroll_file):
                    bankroll_df = pd.read_csv(bankroll_file)
                else:
                    bankroll_df = pd.DataFrame(columns=["date", "amount", "action", "note"])
                
                # Ajouter la nouvelle entrée
                new_entry = pd.DataFrame({
                    "date": [datetime.datetime.now().strftime("%Y-%m-%d")],
                    "amount": [new_bankroll],
                    "action": [action],
                    "note": [adjustment_note]
                })
                
                bankroll_df = pd.concat([bankroll_df, new_entry], ignore_index=True)
                bankroll_df.to_csv(bankroll_file, index=False)
                
                st.success(f"Bankroll mise à jour: {new_bankroll:.2f} €")
                st.experimental_rerun()
        
        st.markdown("---")
        
        # Section pour ajouter un pari manuellement
        st.subheader("Ajouter un pari manuellement")
        
        # Interface d'ajout de pari
        # Ligne 1: Informations sur l'événement
        event_cols = st.columns(2)
        with event_cols[0]:
            manual_event_name = st.text_input("Nom de l'événement", value="UFC Fight Night", key="manual_event_name")
        with event_cols[1]:
            manual_event_date = st.date_input("Date de l'événement", value=datetime.datetime.now(), key="manual_event_date")
        
        # Ligne 2: Informations sur les combattants
        fighter_cols = st.columns(2)
        with fighter_cols[0]:
            manual_fighter_red = st.selectbox("Combattant rouge", options=fighter_names, key="manual_fighter_red")
        with fighter_cols[1]:
            # Exclure le combattant rouge des options
            manual_blue_options = [name for name in fighter_names if name != manual_fighter_red]
            manual_fighter_blue = st.selectbox("Combattant bleu", options=manual_blue_options, key="manual_fighter_blue")
        
        # Ligne 3: Informations sur le pari
        bet_cols = st.columns(3)
        with bet_cols[0]:
            manual_pick = st.selectbox("Pari sur", options=[manual_fighter_red, manual_fighter_blue], key="manual_pick")
        with bet_cols[1]:
            manual_odds = st.number_input("Cote", min_value=1.01, value=2.0, step=0.05, format="%.2f", key="manual_odds")
        with bet_cols[2]:
            manual_stake = st.number_input(
                "Mise (€)",
                min_value=0.0, 
                max_value=float(current_bankroll),
                value=min(50.0, float(current_bankroll)),
                step=5.0,
                format="%.2f",
                key="manual_stake"
            )
        
        # Afficher le gain potentiel
        potential_profit = manual_stake * (manual_odds - 1)
        st.info(f"Mise: {manual_stake:.2f} € @ {manual_odds:.2f} | Gain potentiel: {potential_profit:.2f} € | % de bankroll: {manual_stake/current_bankroll*100:.1f}%")
        
        # Bouton pour enregistrer le pari
        if st.button("Enregistrer le pari", type="primary", key="save_manual_bet_btn"):
            if manual_stake > current_bankroll:
                st.error(f"Mise ({manual_stake:.2f} €) supérieure à la bankroll actuelle ({current_bankroll:.2f} €)")
            elif manual_stake <= 0:
                st.error("La mise doit être supérieure à 0 €")
            else:
                # Enregistrer le pari
                if add_manual_bet(
                    event_name=manual_event_name,
                    event_date=manual_event_date,
                    fighter_red=manual_fighter_red,
                    fighter_blue=manual_fighter_blue,
                    pick=manual_pick,
                    odds=manual_odds,
                    stake=manual_stake,
                    model_probability=None,  # Pas de probabilité ML pour les paris manuels
                    kelly_fraction=None      # Pas de fraction Kelly pour les paris manuels
                ):
                    st.success(f"Pari enregistré avec succès! Mise de {manual_stake:.2f} € sur {manual_pick} @ {manual_odds:.2f}")
                    # Recharger la page
                    st.experimental_rerun()
                else:
                    st.error("Erreur lors de l'enregistrement du pari.")
    
    # Onglet historique et performance
    with tabs[2]:
        st.markdown("""
        <div style="text-align:center;">
            <h2>📊 Historique des paris et performances 📊</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Vérifier si les fichiers existent
        bets_file = os.path.join("bets", "bets.csv")
        bankroll_file = os.path.join("bets", "bankroll.csv")
        has_bets = os.path.exists(bets_file)
        has_bankroll = os.path.exists(bankroll_file)
        
        if has_bets and has_bankroll:
            bets_df = pd.read_csv(bets_file)
            bankroll_df = pd.read_csv(bankroll_file)
            
            # Graphique d'évolution de la bankroll
            if not bankroll_df.empty:
                st.subheader("Évolution de la bankroll")
                fig = px.line(bankroll_df, x="date", y="amount", 
                            title="Évolution de la bankroll dans le temps",
                            labels={"amount": "Bankroll (€)", "date": "Date"})
                st.plotly_chart(fig, use_container_width=True)
            
            # Résumé des performances
            if not bets_df.empty:
                # Obtenir les statistiques
                betting_stats = get_betting_summary(bets_df)
                
                # Afficher les métriques principales
                st.subheader("Résumé des performances")
                metrics_cols = st.columns(4)
                
                with metrics_cols[0]:
                    st.metric("Total des paris", f"{betting_stats['total_bets']}")
                with metrics_cols[1]:
                    st.metric("Paris en cours", f"{betting_stats['open_bets']}")
                with metrics_cols[2]:
                    st.metric("Victoires/Défaites", f"{betting_stats['wins']}/{betting_stats['losses']}")
                with metrics_cols[3]:
                    st.metric("Taux de réussite", f"{betting_stats['win_rate']:.1f}%")
                
                # Deuxième ligne de métriques
                metrics_row2 = st.columns(4)
                with metrics_row2[0]:
                    st.metric("Total misé", f"{betting_stats['total_staked']:.2f} €")
                with metrics_row2[1]:
                    st.metric("Profit total", f"{betting_stats['total_profit']:.2f} €")
                with metrics_row2[2]:
                    st.metric("ROI", f"{betting_stats['roi']:.1f}%")
                with metrics_row2[3]:
                    avg_stake = betting_stats['total_staked'] / max(betting_stats['total_bets'], 1)
                    st.metric("Mise moyenne", f"{avg_stake:.2f} €")
                
                # Afficher les paris en cours dans un tableau interactif
                st.markdown("---")
                
                # Utiliser des onglets pour organiser les paris
                bet_subtabs = st.tabs(["Paris en cours", "Historique des paris", "Modifier/Supprimer"])
                
                # Section des paris en cours
                with bet_subtabs[0]:
                    st.subheader("Paris en cours")
                    open_bets = bets_df[bets_df["status"] == "open"]
                    if not open_bets.empty:
                        # Formater le DataFrame pour l'affichage
                        display_open_bets = open_bets.copy()
                        display_open_bets['gain_potentiel'] = display_open_bets.apply(lambda row: row['stake'] * (row['odds'] - 1), axis=1)
                        
                        # Sélectionner et renommer les colonnes
                        display_open_bets = display_open_bets[["bet_id", "event_name", "event_date", "fighter_red", "fighter_blue", "pick", "odds", "stake", "gain_potentiel"]]
                        display_open_bets.columns = ["ID", "Événement", "Date", "Rouge", "Bleu", "Pari sur", "Cote", "Mise (€)", "Gain potentiel (€)"]
                        
                        # Afficher le tableau
                        st.dataframe(
                            display_open_bets,
                            use_container_width=True,
                            column_config={
                                "ID": st.column_config.NumberColumn("ID", format="%d"),
                                "Événement": st.column_config.TextColumn("Événement"),
                                "Date": st.column_config.DateColumn("Date", format="DD/MM/YYYY"),
                                "Rouge": st.column_config.TextColumn("Rouge"),
                                "Bleu": st.column_config.TextColumn("Bleu"),
                                "Pari sur": st.column_config.TextColumn("Pari sur"),
                                "Cote": st.column_config.NumberColumn("Cote", format="%.2f"),
                                "Mise (€)": st.column_config.NumberColumn("Mise (€)", format="%.2f"),
                                "Gain potentiel (€)": st.column_config.NumberColumn("Gain potentiel (€)", format="%.2f")
                            }
                        )
                    else:
                        st.info("Aucun pari en cours.")
                
                # Section historique des paris
                with bet_subtabs[1]:
                    st.subheader("Historique des paris")
                    closed_bets = bets_df[bets_df["status"] == "closed"]
                    if not closed_bets.empty:
                        # Formater le DataFrame pour l'affichage
                        display_closed_bets = closed_bets.copy()
                        
                        # Sélectionner et renommer les colonnes
                        display_closed_bets = display_closed_bets[["bet_id", "event_name", "event_date", "fighter_red", "fighter_blue", "pick", "odds", "stake", "result", "profit", "roi"]]
                        display_closed_bets.columns = ["ID", "Événement", "Date", "Rouge", "Bleu", "Pari sur", "Cote", "Mise (€)", "Résultat", "Profit (€)", "ROI (%)"]
                        
                        # Afficher le tableau
                        st.dataframe(
                            display_closed_bets,
                            use_container_width=True,
                            column_config={
                                "ID": st.column_config.NumberColumn("ID", format="%d"),
                                "Événement": st.column_config.TextColumn("Événement"),
                                "Date": st.column_config.DateColumn("Date", format="DD/MM/YYYY"),
                                "Rouge": st.column_config.TextColumn("Rouge"),
                                "Bleu": st.column_config.TextColumn("Bleu"),
                                "Pari sur": st.column_config.TextColumn("Pari sur"),
                                "Cote": st.column_config.NumberColumn("Cote", format="%.2f"),
                                "Mise (€)": st.column_config.NumberColumn("Mise (€)", format="%.2f"),
                                "Résultat": st.column_config.TextColumn("Résultat"),
                                "Profit (€)": st.column_config.NumberColumn("Profit (€)", format="%.2f"),
                                "ROI (%)": st.column_config.NumberColumn("ROI (%)", format="%.1f")
                            }
                        )
                    else:
                        st.info("Aucun pari dans l'historique.")
                
                # Section de gestion des paris
                with bet_subtabs[2]:
                    st.subheader("Gérer les paris")
                    
                    # Créer deux colonnes
                    manage_columns = st.columns(2)
                    
                    # Colonne pour mettre à jour les paris
                    with manage_columns[0]:
                        st.markdown("#### Mettre à jour un pari")
                        
                        # Sélectionner un pari à mettre à jour
                        open_bets = bets_df[bets_df["status"] == "open"]
                        open_bet_ids = open_bets["bet_id"].tolist() if not open_bets.empty else []
                        
                        if open_bet_ids:
                            update_bet_id = st.selectbox(
                                "Choisir un pari à mettre à jour:",
                                options=open_bet_ids,
                                format_func=lambda x: f"#{x} - {open_bets[open_bets['bet_id'] == x]['event_name'].values[0]} ({open_bets[open_bets['bet_id'] == x]['pick'].values[0]})",
                                key="update_bet_select"
                            )
                            
                            # Récupérer les informations du pari
                            selected_bet = open_bets[open_bets["bet_id"] == update_bet_id].iloc[0]
                            
                            st.info(f"Pari #{update_bet_id}: {selected_bet['pick']} @ {selected_bet['odds']} (Mise: {selected_bet['stake']}€)")
                            
                            # Sélectionner le résultat
                            result = st.radio(
                                "Résultat du pari:",
                                options=["win", "loss", "void"],
                                horizontal=True,
                                key="update_result_radio"
                            )
                            
                            # Expliquer les options
                            st.caption("win = gagné, loss = perdu, void = annulé/remboursé")
                            
                            # Bouton pour mettre à jour
                            if st.button("Mettre à jour le pari", key="update_bet_btn"):
                                # Mettre à jour le pari
                                new_bankroll = update_bet_result(update_bet_id, result, current_bankroll)
                                
                                st.success(f"Pari mis à jour avec succès! Nouvelle bankroll: {new_bankroll:.2f} €")
                                # Recharger la page
                                st.experimental_rerun()
                        else:
                            st.info("Aucun pari en cours à mettre à jour.")
                    
                    # Colonne pour supprimer les paris
                    with manage_columns[1]:
                        st.markdown("#### Supprimer un pari")
                        
                        # Sélectionner un pari à supprimer (seulement les paris ouverts)
                        open_bets = bets_df[bets_df["status"] == "open"]
                        open_bet_ids = open_bets["bet_id"].tolist() if not open_bets.empty else []
                        
                        if open_bet_ids:
                            delete_bet_id = st.selectbox(
                                "Choisir un pari à supprimer:",
                                options=open_bet_ids,
                                format_func=lambda x: f"#{x} - {open_bets[open_bets['bet_id'] == x]['event_name'].values[0]} ({open_bets[open_bets['bet_id'] == x]['pick'].values[0]})",
                                key="delete_bet_select"
                            )
                            
                            # Récupérer les informations du pari
                            selected_bet = open_bets[open_bets["bet_id"] == delete_bet_id].iloc[0]
                            
                            st.info(f"Pari #{delete_bet_id}: {selected_bet['pick']} @ {selected_bet['odds']} (Mise: {selected_bet['stake']}€)")
                            
                            # Confirmation
                            st.warning("⚠️ La suppression est définitive et ne peut pas être annulée.")
                            
                            # Bouton pour supprimer
                            if st.button("Supprimer le pari", key="delete_bet_btn"):
                                # Supprimer le pari
                                if delete_bet(delete_bet_id):
                                    st.success(f"Pari #{delete_bet_id} supprimé avec succès!")
                                    # Recharger la page
                                    st.experimental_rerun()
                                else:
                                    st.error("Erreur lors de la suppression du pari.")
                        else:
                            st.info("Aucun pari à supprimer.")
                
                # Graphique de distribution des résultats
                if not closed_bets.empty:
                    st.markdown("---")
                    st.subheader("Analyse des performances")
                    
                    # Distribution des résultats
                    results_count = closed_bets['result'].value_counts().reset_index()
                    results_count.columns = ['Résultat', 'Nombre']
                    
                    # Créer le graphique avec Plotly
                    if not results_count.empty:
                        fig = px.pie(
                            results_count, 
                            values='Nombre', 
                            names='Résultat',
                            title="Distribution des résultats",
                            color_discrete_map={'win': 'green', 'loss': 'red', 'void': 'gray'}
                        )
                        st.plotly_chart(fig)
                    
                    # Graphique évolution du profit
                    if len(closed_bets) > 1:
                        # Trier par date
                        closed_bets_sorted = closed_bets.sort_values(by='event_date')
                        
                        # Calculer le profit cumulatif
                        closed_bets_sorted['profit_cumul'] = closed_bets_sorted['profit'].cumsum()
                        
                        # Créer le graphique
                        fig = px.line(
                            closed_bets_sorted,
                            x='event_date',
                            y='profit_cumul',
                            title="Évolution du profit dans le temps",
                            labels={'event_date': 'Date', 'profit_cumul': 'Profit cumulé (€)'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Aucun pari enregistré. Commencez à parier pour voir vos statistiques.")
        else:
            st.info("Aucune donnée d'historique disponible. Placez votre premier pari pour commencer à suivre vos performances.")
        
        # Téléchargement des données
        if has_bets and has_bankroll and not bets_df.empty:
            st.markdown("---")
            st.subheader("Exporter les données")
            
            download_cols = st.columns(2)
            
            with download_cols[0]:
                st.download_button(
                    label="Télécharger les paris (CSV)",
                    data=bets_df.to_csv(index=False).encode('utf-8'),
                    file_name='ufc_bets_history.csv',
                    mime='text/csv',
                    key="download_bets_btn"
                )
            
            with download_cols[1]:
                st.download_button(
                    label="Télécharger l'historique bankroll (CSV)",
                    data=bankroll_df.to_csv(index=False).encode('utf-8'),
                    file_name='ufc_bankroll_history.csv',
                    mime='text/csv',
                    key="download_bankroll_btn"
                )

if __name__ == "__main__":
    main()
                
