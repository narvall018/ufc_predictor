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
import requests
from bs4 import BeautifulSoup
import re
import time
import random
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
    
    /* Style pour les événements à venir */
    .upcoming-event {
        background-color: rgba(240, 242, 246, 0.7);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    
    .upcoming-fight {
        background-color: rgba(248, 249, 250, 0.5);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 4px solid #FF9800;
    }
    
    .loading-spinner {
        text-align: center;
        margin: 20px 0;
    }
    
    .fight-card {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 15px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    
    .fight-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 15px;
    }
    
    /* Nouveaux styles pour les sections d'événements */
    .event-section {
        background-color: rgba(38, 39, 48, 0.8);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 30px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .event-title {
        background-color: rgba(255, 69, 0, 0.8);
        color: white;
        padding: 10px 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    /* Style pour les cartes de combat améliorées */
    .fight-card-improved {
        background-color: rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .fighters-banner {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    
    .fighter-name-red {
        color: #ff4d4d;
        font-weight: bold;
        font-size: 1.2rem;
        flex: 1;
        text-align: left;
    }
    
    .fighter-name-blue {
        color: #4d79ff;
        font-weight: bold;
        font-size: 1.2rem;
        flex: 1;
        text-align: right;
    }
    
    .vs-badge {
        background-color: rgba(0, 0, 0, 0.7);
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        margin: 0 10px;
    }
    
    /* Barres de probabilité */
    .probability-container {
        margin-top: 10px;
        margin-bottom: 15px;
    }
    
    .probability-bar {
        height: 25px;
        width: 100%;
        background-color: #e0e0e0;
        border-radius: 5px;
        position: relative;
        overflow: hidden;
    }
    
    .probability-bar-red {
        height: 100%;
        background-color: #ff4d4d;
        float: left;
        text-align: center;
        line-height: 25px;
        color: white;
        font-weight: bold;
    }
    
    .probability-bar-blue {
        height: 100%;
        background-color: #4d79ff;
        float: right;
        text-align: center;
        line-height: 25px;
        color: white;
        font-weight: bold;
    }
    
    .prediction-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
        margin-top: 5px;
    }
    
    .prediction-badge-red {
        background-color: #ff4d4d;
        color: white;
    }
    
    .prediction-badge-blue {
        background-color: #4d79ff;
        color: white;
    }
    
    .prediction-summary {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 5px;
    }
    
    .prediction-method {
        font-size: 0.9rem;
        color: #aaa;
    }
    
    .confidence-badge {
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 0.85rem;
    }
    
    .confidence-high {
        background-color: rgba(76, 175, 80, 0.3);
        color: #2e7d32;
    }
    
    .confidence-moderate {
        background-color: rgba(255, 193, 7, 0.3);
        color: #ff8f00;
    }
    
    /* Style pour la page d'accueil */
    .welcome-header {
        text-align: center;
        padding: 40px 0;
        background: linear-gradient(135deg, rgba(255, 0, 0, 0.8) 0%, rgba(0, 0, 255, 0.8) 100%);
        border-radius: 15px;
        margin-bottom: 30px;
        color: white;
    }
    
    .welcome-title {
        font-size: 3.5rem;
        font-weight: bold;
        margin-bottom: 15px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }
    
    .welcome-subtitle {
        font-size: 1.8rem;
        margin-bottom: 20px;
    }
    
    .home-card {
        background-color: rgba(255, 255, 255, 0.08);
        border-radius: 15px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 15px;
        text-align: center;
    }
    
    .feature-title {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 10px;
        text-align: center;
    }
    
    .feature-description {
        text-align: center;
        font-size: 1.1rem;
    }
    
    /* Style pour le tableau Kelly */
    .kelly-box {
        background-color: rgba(25, 135, 84, 0.1);
        border-radius: 10px;
        padding: 15px;
        margin: 15px 0;
        border-left: 3px solid #198754;
    }
    
    .kelly-title {
        color: #198754;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    .kelly-table {
        width: 100%;
        border-collapse: collapse;
    }
    
    .kelly-table th, .kelly-table td {
        padding: 8px;
        text-align: left;
        border-bottom: 1px solid rgba(200, 200, 200, 0.3);
    }
    
    .kelly-table tr:hover {
        background-color: rgba(255, 255, 255, 0.05);
    }
    
    .bet-placement-box {
        background-color: rgba(13, 110, 253, 0.1);
        border-radius: 10px;
        padding: 15px;
        margin: 15px 0;
        border-left: 3px solid #0d6efd;
    }
    
    .bet-placement-title {
        color: #0d6efd;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
/* Style pour la section stratégie de paris */
.betting-strategy-box {
    background-color: rgba(76, 175, 80, 0.1);
    border-radius: 10px;
    padding: 20px;
    margin: 15px 0;
    border-left: 3px solid #4CAF50;
}

.strategy-title {
    color: #4CAF50;
    font-weight: bold;
    margin-bottom: 15px;
}

.strategy-summary {
    background-color: rgba(76, 175, 80, 0.1);
    padding: 15px;
    border-radius: 10px;
    margin-top: 15px;
}

.value-betting-positive {
    color: #4CAF50;
    font-weight: bold;
}

.value-betting-negative {
    color: #f44336;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# CONFIGURATION POUR LES REQUÊTES WEB
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9,fr;q=0.8'
}

# Cache pour les requêtes
request_cache = {}

@st.cache_resource(ttl=3600*24)
def load_app_data():
    """Version optimisée du chargement des données"""
    data = {
        "ml_model": None,
        "scaler": None,
        "feature_names": None,
        "fighters": [],
        "fighters_dict": {},
        "fighter_names": [],
        "current_bankroll": 1000
    }
    
    # Charger le modèle ML avec gestion d'erreur améliorée
    try:
        model_files = ["ufc_prediction_model.joblib", "ufc_prediction_model.pkl"]
        for model_file in model_files:
            if os.path.exists(model_file):
                if model_file.endswith('.joblib'):
                    model_data = joblib.load(model_file)
                else:
                    with open(model_file, 'rb') as file:
                        model_data = pickle.load(file)
                
                if model_data:
                    data["ml_model"] = model_data.get('model')
                    data["scaler"] = model_data.get('scaler')
                    data["feature_names"] = model_data.get('feature_names')
                    break
    except Exception as e:
        print(f"Erreur lors du chargement du modèle ML: {e}")
    
    # Optimisation du chargement des stats des combattants
    fighter_stats_path = 'fighters_stats.txt'
    if os.path.exists(fighter_stats_path):
        fighters = load_fighters_stats(fighter_stats_path)
        fighters = deduplicate_fighters(fighters)
        data["fighters"] = fighters
        
        # Création optimisée du dictionnaire
        data["fighters_dict"] = {fighter['name']: fighter for fighter in fighters}
        data["fighter_names"] = sorted([fighter['name'] for fighter in fighters])
    
    # Initialiser/Charger la bankroll
    data["current_bankroll"] = init_bankroll()
    init_bets_file()
    
    return data

@st.cache_data(ttl=86400, show_spinner=False)
def make_request(url, max_retries=3, delay_range=(0.5, 1.5)):
    """Requête HTTP avec cache intelligent"""
    # Vérifier d'abord dans le cache
    if url in request_cache:
        return request_cache[url]
    
    # Stratégie de backoff exponentiel
    for attempt in range(max_retries):
        try:
            # Ajouter un délai pour éviter de surcharger les serveurs
            backoff_time = delay_range[0] * (2 ** attempt) if attempt > 0 else delay_range[0]
            actual_delay = min(backoff_time, delay_range[1] * 3)
            time.sleep(actual_delay)
            
            response = requests.get(url, headers=HEADERS, timeout=20)
            
            if response.status_code == 200:
                request_cache[url] = response
                return response
            
            # Délai spécial pour les erreurs de limitation
            if response.status_code in [403, 429]:
                time.sleep(5 * (attempt + 1))
        except Exception:
            pass
    
    return None

def split_fighter_names(text):
    """Sépare automatiquement les noms concaténés des combattants"""
    if not text or "vs" in text:
        return None, None
    
    # Nettoyer le texte d'abord
    text = text.strip()
    
    # Méthode 1: Rechercher une lettre minuscule suivie d'une majuscule
    matches = list(re.finditer(r'([a-z])([A-Z])', text))
    
    if matches:
        # Trouver la dernière occurrence
        match = matches[-1]
        split_pos = match.start() + 1
        
        # Séparer les noms
        first_fighter = text[:split_pos].strip()
        second_fighter = text[split_pos:].strip()
        
        return first_fighter, second_fighter
    
    # Méthode 2: Si un seul espace est présent, séparer à cet espace
    if text.count(' ') == 1:
        parts = text.split(' ')
        return parts[0], parts[1]
    
    # Si aucune séparation claire n'est trouvée
    return None, None

@st.cache_data(ttl=86400, show_spinner=False)  # Augmentation du TTL à 24h
def get_upcoming_events(max_events=3):
    """Récupère les événements UFC à venir"""
    # URL pour les événements à venir
    urls = [
        "http://ufcstats.com/statistics/events/upcoming",
        "http://ufcstats.com/statistics/events/completed"  # Fallback pour les événements récents
    ]
    
    events = []
    
    for url in urls:
        response = make_request(url)
        if not response:
            continue
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Méthode 1: Rechercher dans la table des événements
        events_table = soup.find('table', class_='b-statistics__table-events')
        
        if events_table:
            rows = events_table.find_all('tr')[1:]  # Ignorer l'en-tête
            
            for row in rows[:max_events]:  # Prendre les N premiers événements
                cells = row.find_all('td')
                if len(cells) >= 1:
                    event_link = cells[0].find('a')
                    if event_link:
                        event_url = event_link.get('href')
                        event_name = event_link.text.strip()
                        
                        events.append({
                            'name': event_name,
                            'url': event_url
                        })
        
        # Méthode 2: Chercher les liens directement
        if len(events) < max_events:
            event_links = soup.find_all('a', href=lambda href: href and 'event-details' in href)
            
            for link in event_links[:max_events]:
                event_url = link.get('href')
                event_name = link.text.strip()
                
                if event_url and event_name:
                    # Vérifier si cet événement est déjà dans la liste
                    if not any(e['url'] == event_url for e in events):
                        events.append({
                            'name': event_name,
                            'url': event_url
                        })
                        
                        # Si on a assez d'événements, on s'arrête
                        if len(events) >= max_events:
                            break
        
        # Si on a trouvé assez d'événements, on peut passer à l'URL suivante
        if len(events) >= max_events:
            break
    
    return events[:max_events]

@st.cache_data(ttl=86400, show_spinner=False)  # Augmentation du TTL à 24h
def extract_upcoming_fights(event_url):
    resp = make_request(event_url)
    if not resp:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", class_="b-fight-details__table")
    fights = []

    if table:
        rows = table.select("tbody > tr")[0:]   # on saute la ligne d'en-tête
        for row in rows:
            links = row.select("td:nth-child(2) a")  # les 2 balises <a> avec les noms
            if len(links) >= 2:
                fights.append({
                    "red_fighter": links[0].text.strip(),
                    "blue_fighter": links[1].text.strip()
                })

    return fights


def find_best_match(name, fighters_dict):
    """Recherche le meilleur match pour un nom de combattant dans les stats"""
    if not name:
        return None
    
    # Nettoyage du nom
    name = name.strip()
    
    # Recherche exacte
    if name in fighters_dict:
        return name
    
    # Recherche insensible à la casse
    name_lower = name.lower()
    for fighter_name in fighters_dict:
        if fighter_name.lower() == name_lower:
            return fighter_name
    
    # Recherche partielle
    best_match = None
    best_score = 0
    
    for fighter_name in fighters_dict:
        # Calculer un score de correspondance simple
        score = 0
        fighter_lower = fighter_name.lower()
        
        # Si l'un contient l'autre complètement
        if name_lower in fighter_lower or fighter_lower in name_lower:
            score += 5
        
        # Correspondance partielle de mots
        name_words = name_lower.split()
        fighter_words = fighter_lower.split()
        
        for word in name_words:
            if word in fighter_words:
                score += 2
            for fighter_word in fighter_words:
                if word in fighter_word or fighter_word in word:
                    score += 1
        
        if score > best_score:
            best_score = score
            best_match = fighter_name
    
    # Seulement retourner s'il y a une correspondance raisonnable
    if best_score >= 3:
        return best_match
    
    return None

# FONCTIONS POUR LE MODÈLE ML

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

# OPTIMISATION 3: MISE EN CACHE STRATÉGIQUE
@st.cache_data(ttl=3600, show_spinner=False)
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

@st.cache_data(ttl=86400, show_spinner=False)  # Augmentation du TTL à 24h
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
def predict_both_methods(fighter_a, fighter_b, odds_a=0, odds_b=0, model=None, scaler=None, feature_names=None):
    """
    Prédit l'issue d'un combat en utilisant les deux méthodes (ML et classique)
    et retourne les deux prédictions
    """
    # Prédiction avec la méthode classique
    classic_prediction = predict_fight_classic(fighter_a, fighter_b, odds_a, odds_b)
    classic_prediction['method'] = 'classic'
    
    # Prédiction avec ML si disponible
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
@st.cache_data(ttl=3600)
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
@st.cache_data(ttl=3600)
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
@st.cache_data(ttl=3600)
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

# FONCTIONS POUR LA GESTION DE BANKROLL ET PARIS

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
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
    
if 'upcoming_events' not in st.session_state:
    st.session_state.upcoming_events = None
    
if 'upcoming_events_timestamp' not in st.session_state:
    st.session_state.upcoming_events_timestamp = None
    
if 'upcoming_fights' not in st.session_state:
    st.session_state.upcoming_fights = {}
    
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Accueil"

if 'kelly_strategy' not in st.session_state:
    st.session_state.kelly_strategy = "Kelly/4"
# Au début du script, après les autres initialisations de session_state
if 'saved_bet_events' not in st.session_state:
    st.session_state.saved_bet_events = {}
    
if 'betting_recommendations' not in st.session_state:
    st.session_state.betting_recommendations = {}

# Charger les données une seule fois au démarrage
app_data = load_app_data()



# FONCTION PRINCIPALE

def main():
    # Titre principal
    st.markdown('<div class="main-title">🥊 Prédicteur de Combats UFC 🥊</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Analysez et prédisez l\'issue des affrontements</div>', unsafe_allow_html=True)
    
    # Créer les onglets principaux
    tabs = st.tabs(["🏠 Accueil", "🎯 Prédiction", "🗓️ Événements à venir", "💰 Gestion de Bankroll", "📊 Historique & Performance"])
    
    # Onglet d'accueil
    with tabs[0]:
        show_welcome_page()
    
    # Onglet de prédiction
    with tabs[1]:
        show_prediction_page()
    
    # Onglet des événements à venir
    with tabs[2]:
        show_upcoming_events_page()
    
    # Onglet de gestion de bankroll
    with tabs[3]:
        show_bankroll_page()
    
    # Onglet historique et performance
    with tabs[4]:
        show_history_page()

def show_welcome_page():
    # En-tête de bienvenue
    st.markdown("""
    <div class="welcome-header">
        <h1 class="welcome-title">🥊 UFC Fight Predictor 🥊</h1>
        <p class="welcome-subtitle">Prédisez les résultats des combats UFC avec intelligence artificielle</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    L'UFC Fight Predictor est un outil avancé qui combine analyse statistique et machine learning 
    pour prédire les résultats des combats de l'UFC. Que vous soyez un fan passionné cherchant 
    à anticiper les résultats ou un parieur à la recherche d'un avantage analytique, cette application 
    vous fournit des prédictions détaillées basées sur l'historique et les statistiques des combattants.
    """)
    
    # Fonctionnalités principales
    st.markdown("## Principales fonctionnalités")
    
    # Afficher les fonctionnalités dans une mise en page à trois colonnes
    cols = st.columns(3)
    
    with cols[0]:
        st.markdown("""
        <div class="home-card">
            <div class="feature-icon">🎯</div>
            <h3 class="feature-title">Prédictions précises</h3>
            <p class="feature-description">Obtenez des prédictions basées sur deux méthodes complémentaires: analyse statistique classique et modèle de machine learning avancé.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown("""
        <div class="home-card">
            <div class="feature-icon">📊</div>
            <h3 class="feature-title">Analyses détaillées</h3>
            <p class="feature-description">Visualisez les forces et faiblesses de chaque combattant avec des graphiques comparatifs et des statistiques pertinentes.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[2]:
        st.markdown("""
        <div class="home-card">
            <div class="feature-icon">💰</div>
            <h3 class="feature-title">Conseils de paris</h3>
            <p class="feature-description">Recevez des recommandations de paris basées sur l'analyse des cotes et la gestion optimale de votre bankroll avec la méthode Kelly.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Instructions d'utilisation
    st.markdown("## Comment utiliser l'application")
    
    st.markdown("""
    1. **Onglet Prédiction**: Sélectionnez deux combattants pour obtenir une analyse complète et une prédiction
    2. **Onglet Événements à venir**: Consultez les prochains combats UFC avec des prédictions automatiques
    3. **Onglet Gestion de Bankroll**: Suivez vos paris et gérez votre bankroll
    4. **Onglet Historique & Performance**: Analysez vos performances de paris passés
    """)
    
    # Disclaimer
    st.markdown("""
    <div style="background-color:rgba(255, 193, 7, 0.1); padding:15px; border-radius:10px; margin-top:40px;">
        <h3>⚠️ Avertissement</h3>
        <p>Les prédictions fournies par cette application sont basées sur des modèles statistiques et d'apprentissage automatique, mais ne garantissent pas les résultats des combats. Les paris sportifs comportent des risques, et cette application ne doit être utilisée qu'à titre informatif. Pariez de manière responsable.</p>
    </div>
    """, unsafe_allow_html=True)

def show_prediction_page():
    # Interface de sélection des combattants
    st.sidebar.markdown("## Sélection des combattants")
    
    # Message d'avertissement sur l'importance de l'ordre des combattants
    st.sidebar.markdown("""
    <div class="info-box">
        <b>⚠️ Important :</b> L'ordre des combattants (Rouge/Bleu) influence les prédictions. 
        Traditionnellement, le combattant mieux classé ou favori est placé dans le coin rouge.
    </div>
    """, unsafe_allow_html=True)
    
    # Sélection du combattant rouge
    st.sidebar.markdown("### 🔴 Combattant Rouge")
    fighter_a_name = st.sidebar.selectbox(
        "Sélectionner combattant rouge",
        options=app_data["fighter_names"],
        key="fighter_a_selectbox"
    )
    
    # Sélection du combattant bleu (en excluant le combattant rouge)
    st.sidebar.markdown("### 🔵 Combattant Bleu")
    fighter_b_options = [name for name in app_data["fighter_names"] if name != fighter_a_name]
    fighter_b_name = st.sidebar.selectbox(
        "Sélectionner combattant bleu",
        options=fighter_b_options,
        key="fighter_b_selectbox"
    )
    
    # Options de paris
    st.sidebar.markdown("## 💰 Options de paris")
    odds_a = st.sidebar.number_input("Cote Rouge", min_value=1.01, value=2.0, step=0.05, format="%.2f", key="odds_a_input")
    odds_b = st.sidebar.number_input("Cote Bleu", min_value=1.01, value=1.8, step=0.05, format="%.2f", key="odds_b_input")
    
    # Stratégie Kelly
    st.sidebar.markdown("## 📈 Critères Kelly")
    kelly_strategy = st.sidebar.selectbox(
        "Stratégie Kelly",
        options=["Kelly pur", "Kelly/2", "Kelly/3", "Kelly/4", "Kelly/5", "Kelly/10"],
        index=3,  # Kelly/4 par défaut
        key="kelly_strategy_select"
    )
    st.session_state.kelly_strategy = kelly_strategy
    
    # Section de gestion de bankroll
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 💼 Ma Bankroll")
    displayed_bankroll = st.sidebar.number_input(
        "Bankroll actuelle (€)",
        min_value=0.0,
        value=float(app_data["current_bankroll"]),
        step=10.0,
        format="%.2f",
        key="bankroll_input"
    )

    # Option pour modifier la bankroll
    if st.sidebar.button("Mettre à jour la bankroll", key="update_bankroll_btn"):
        if displayed_bankroll != app_data["current_bankroll"]:
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
            app_data["current_bankroll"] = displayed_bankroll
    
    # Bouton de prédiction
    predict_btn = st.sidebar.button("🥊 Prédire le combat", type="primary", key="predict_btn")
    
    # Récupérer les statistiques des combattants sélectionnés
    fighter_a = app_data["fighters_dict"].get(fighter_a_name)
    fighter_b = app_data["fighters_dict"].get(fighter_b_name)
    
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
                odds_b=odds_b,
                model=app_data["ml_model"],
                scaler=app_data["scaler"],
                feature_names=app_data["feature_names"]
            )
            
            # Stocker les résultats dans la session
            st.session_state.prediction_result = {
                'fighter_a': fighter_a,
                'fighter_b': fighter_b,
                'classic_prediction': classic_prediction,
                'ml_prediction': ml_prediction,
                'odds_a': odds_a,
                'odds_b': odds_b
            }
    
    # Afficher les résultats de prédiction
    if st.session_state.prediction_result:
        result = st.session_state.prediction_result
        fighter_a = result['fighter_a']
        fighter_b = result['fighter_b']
        classic_prediction = result['classic_prediction']
        ml_prediction = result['ml_prediction']
        odds_a = result['odds_a']
        odds_b = result['odds_b']
        
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
                'Combattant': [fighter_a['name'], fighter_b['name']],
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
                x=[fighter_a['name'], fighter_b['name']],
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
                    Probabilité: <span class="red-fighter">{classic_prediction['red_probability']:.2f}</span> pour {fighter_a['name']}, 
                    <span class="blue-fighter">{classic_prediction['blue_probability']:.2f}</span> pour {fighter_b['name']}
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
                        Probabilité: <span class="red-fighter">{ml_prediction['red_probability']:.2f}</span> pour {fighter_a['name']}, 
                        <span class="blue-fighter">{ml_prediction['blue_probability']:.2f}</span> pour {fighter_b['name']}
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
        
        # Analyse Kelly et recommandations de paris
        if ml_prediction:
            st.markdown("""
            <div class="divider"></div>
            <div style="text-align:center;">
                <h2>📊 Analyse Kelly et recommandations de paris 📊</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Obtenir la fraction Kelly sélectionnée
            kelly_fractions = {
                "Kelly pur": 1,
                "Kelly/2": 2,
                "Kelly/3": 3, 
                "Kelly/4": 4,
                "Kelly/5": 5,
                "Kelly/10": 10
            }
            selected_fraction = kelly_fractions[st.session_state.kelly_strategy]
            
            # Détermine le combattant qui a la plus forte valeur attendue
            if ml_prediction['prediction'] == 'Red':
                best_fighter = fighter_a['name']
                best_odds = odds_a
                best_prob = ml_prediction['red_probability']
            else:
                best_fighter = fighter_b['name']
                best_odds = odds_b
                best_prob = ml_prediction['blue_probability']
            
            # Calculer les recommandations Kelly pour le combattant favori selon le ML
            kelly_amount = calculate_kelly(best_prob, best_odds, app_data["current_bankroll"], selected_fraction)
            
            # Afficher la section Kelly
            st.markdown(f"""
            <div class="kelly-box">
                <h3 class="kelly-title">Recommandation de mise avec la méthode {st.session_state.kelly_strategy}</h3>
                <p>Pour maximiser votre ROI sur le long terme, la méthode Kelly recommande:</p>
                <table class="kelly-table">
                    <tr>
                        <th>Combattant</th>
                        <th>Probabilité ML</th>
                        <th>Cote</th>
                        <th>Mise recommandée</th>
                        <th>% de bankroll</th>
                        <th>Gain potentiel</th>
                    </tr>
                    <tr>
                        <td><b>{best_fighter}</b></td>
                        <td>{best_prob:.2f}</td>
                        <td>{best_odds:.2f}</td>
                        <td><b>{kelly_amount:.2f} €</b></td>
                        <td>{(kelly_amount/app_data["current_bankroll"]*100):.1f}%</td>
                        <td>{kelly_amount * (best_odds-1):.2f} €</td>
                    </tr>
                </table>
                <p style="margin-top:10px;"><i>Le critère de Kelly détermine la mise optimale en fonction de votre avantage et de votre bankroll totale.</i></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Afficher la section pour placer un pari
            st.markdown(f"""
            <div class="bet-placement-box">
                <h3 class="bet-placement-title">Placer un pari sur {best_fighter}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Colonnes pour les informations du pari
            bet_cols = st.columns(2)
            
            with bet_cols[0]:
                # Nom de l'événement
                event_name = st.text_input("Nom de l'événement", value="UFC Fight Night", key="event_name_input")
                
                # Date de l'événement
                event_date = st.date_input("Date de l'événement", value=datetime.datetime.now(), key="event_date_input")
            
            with bet_cols[1]:
                # Montant à miser
                bet_amount = st.number_input(
                    "Montant à miser (€)",
                    min_value=0.0,
                    max_value=float(app_data["current_bankroll"]),
                    value=float(kelly_amount),
                    step=5.0,
                    format="%.2f",
                    key="bet_amount_input"
                )
                
                # Utiliser la mise Kelly recommandée
                use_kelly = st.checkbox("Utiliser la mise Kelly recommandée", value=True, key="use_kelly_checkbox")
                if use_kelly:
                    bet_amount = kelly_amount
            
            # Afficher les détails du pari
            st.info(f"Pari: {bet_amount:.2f}€ sur {best_fighter} @ {best_odds:.2f} | Gain potentiel: {bet_amount * (best_odds-1):.2f}€")
            
            # Bouton pour placer le pari
            if st.button("💰 Placer ce pari", type="primary", key="place_bet_btn"):
                if bet_amount > app_data["current_bankroll"]:
                    st.error(f"Montant du pari ({bet_amount:.2f}€) supérieur à votre bankroll actuelle ({app_data['current_bankroll']:.2f}€)")
                elif bet_amount <= 0:
                    st.error("Le montant du pari doit être supérieur à 0€")
                else:
                    # Ajouter le pari à l'historique
                    if add_manual_bet(
                        event_name=event_name,
                        event_date=event_date,
                        fighter_red=fighter_a['name'],
                        fighter_blue=fighter_b['name'],
                        pick=best_fighter,
                        odds=best_odds,
                        stake=bet_amount,
                        model_probability=best_prob,
                        kelly_fraction=selected_fraction
                    ):
                        st.success(f"Pari enregistré avec succès! {bet_amount:.2f}€ sur {best_fighter} @ {best_odds:.2f}")
                    else:
                        st.error("Erreur lors de l'enregistrement du pari.")
        
        # Analyse des paris (utiliser les deux méthodes si disponibles)
        if 'betting' in classic_prediction:
            st.markdown("""
            <div class="divider"></div>
            <div style="text-align:center;">
                <h2>💰 Analyse des paris 💰</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Analyse des paris pour les deux combattants avec Streamlit natif
            col1, col2 = st.columns(2)
            
            # Combattant Rouge - Carte de paris
            with col1:
                st.subheader(f"🔴 {fighter_a['name']}")
                
                # Créer un DataFrame pour une présentation simple
                betting_classic = classic_prediction['betting']
                betting_ml = ml_prediction.get('betting') if ml_prediction else None
                
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
                
                # Ajouter un bouton pour parier sur le combattant rouge
                if st.button(f"Parier sur {fighter_a['name']}", key="bet_on_red_btn"):
                    # Calculer le montant Kelly pour ce combattant
                    red_kelly = calculate_kelly(
                        ml_prediction['red_probability'] if ml_prediction else classic_prediction['red_probability'],
                        odds_a,
                        app_data["current_bankroll"],
                        kelly_fractions[st.session_state.kelly_strategy]
                    )
                    
                    # Stocker dans la session pour précharger le formulaire
                    st.session_state.temp_bet = {
                        "fighter": fighter_a['name'],
                        "odds": odds_a,
                        "kelly_amount": red_kelly,
                        "probability": ml_prediction['red_probability'] if ml_prediction else classic_prediction['red_probability']
                    }
                    
                    # Afficher le formulaire pour parier
                    show_bet_form(
                        fighter_a['name'], 
                        fighter_b['name'], 
                        fighter_a['name'], 
                        odds_a, 
                        red_kelly,
                        ml_prediction['red_probability'] if ml_prediction else classic_prediction['red_probability'],
                        kelly_fractions[st.session_state.kelly_strategy]
                    )
            
            # Combattant Bleu - Carte de paris
            with col2:
                st.subheader(f"🔵 {fighter_b['name']}")
                
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
                
                # Ajouter un bouton pour parier sur le combattant bleu
                if st.button(f"Parier sur {fighter_b['name']}", key="bet_on_blue_btn"):
                    # Calculer le montant Kelly pour ce combattant
                    blue_kelly = calculate_kelly(
                        ml_prediction['blue_probability'] if ml_prediction else classic_prediction['blue_probability'],
                        odds_b,
                        app_data["current_bankroll"],
                        kelly_fractions[st.session_state.kelly_strategy]
                    )
                    
                    # Stocker dans la session pour précharger le formulaire
                    st.session_state.temp_bet = {
                        "fighter": fighter_b['name'],
                        "odds": odds_b,
                        "kelly_amount": blue_kelly,
                        "probability": ml_prediction['blue_probability'] if ml_prediction else classic_prediction['blue_probability']
                    }
                    
                    # Afficher le formulaire pour parier
                    show_bet_form(
                        fighter_a['name'], 
                        fighter_b['name'], 
                        fighter_b['name'], 
                        odds_b, 
                        blue_kelly,
                        ml_prediction['blue_probability'] if ml_prediction else classic_prediction['blue_probability'],
                        kelly_fractions[st.session_state.kelly_strategy]
                    )
        
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
            
            if advantage == fighter_a['name']:
                styles[1] = 'background-color: rgba(255, 0, 0, 0.2); font-weight: bold;'
            elif advantage == fighter_b['name']:
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
    else:
        # Message d'accueil
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
        ml_available = app_data["ml_model"] is not None
        
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

def show_bet_form(fighter_red, fighter_blue, pick, odds, kelly_amount, probability, kelly_fraction):
    """Affiche un formulaire pour placer un pari"""
    st.markdown(f"""
    <div class="bet-placement-box">
        <h3 class="bet-placement-title">Placer un pari sur {pick}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Colonnes pour les informations du pari
    bet_cols = st.columns(2)
    
    with bet_cols[0]:
        # Nom de l'événement
        event_name = st.text_input("Nom de l'événement", value="UFC Fight Night", key="bet_event_name")
        
        # Date de l'événement
        event_date = st.date_input("Date de l'événement", value=datetime.datetime.now(), key="bet_event_date")
    
    with bet_cols[1]:
        # Montant à miser
        bet_amount = st.number_input(
            "Montant à miser (€)",
            min_value=0.0,
            max_value=float(app_data["current_bankroll"]),
            value=float(kelly_amount),
            step=5.0,
            format="%.2f",
            key="place_bet_amount"
        )
        
        # Utiliser la mise Kelly recommandée
        use_kelly = st.checkbox("Utiliser la mise Kelly recommandée", value=True, key="place_use_kelly")
        if use_kelly:
            bet_amount = kelly_amount
    
    # Afficher les détails du pari
    st.info(f"Pari: {bet_amount:.2f}€ sur {pick} @ {odds:.2f} | Gain potentiel: {bet_amount * (odds-1):.2f}€")
    
    # Bouton pour placer le pari
    if st.button("💰 Confirmer ce pari", type="primary", key="confirm_bet_btn"):
        if bet_amount > app_data["current_bankroll"]:
            st.error(f"Montant du pari ({bet_amount:.2f}€) supérieur à votre bankroll actuelle ({app_data['current_bankroll']:.2f}€)")
        elif bet_amount <= 0:
            st.error("Le montant du pari doit être supérieur à 0€")
        else:
            # Ajouter le pari à l'historique
            if add_manual_bet(
                event_name=event_name,
                event_date=event_date,
                fighter_red=fighter_red,
                fighter_blue=fighter_blue,
                pick=pick,
                odds=odds,
                stake=bet_amount,
                model_probability=probability,
                kelly_fraction=kelly_fraction
            ):
                st.success(f"Pari enregistré avec succès! {bet_amount:.2f}€ sur {pick} @ {odds:.2f}")
            else:
                st.error("Erreur lors de l'enregistrement du pari.")

def show_betting_strategy_section(event_url, event_name, fights, predictions_data, current_bankroll=300):
    """Affiche la section de stratégie de paris basée sur les prédictions existantes"""
    
    # Vérifier si on a déjà fait des recommandations
    event_key = f"recommendations_{event_url}"
    has_existing_recommendations = event_key in st.session_state.betting_recommendations
    
    st.markdown("""
    <div class="divider"></div>
    <div style="text-align:center;">
        <h2>💰 Stratégie de paris optimisée 💰</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Si on a des paris sauvegardés pour cet événement, afficher un message
    if event_url in st.session_state.saved_bet_events:
        st.success(f"Vos paris pour cet événement ont été enregistrés avec succès! ({st.session_state.saved_bet_events[event_url]} paris)")
    
    # Paramètres de la stratégie
    st.markdown("### Configurez votre stratégie de paris")
    
    col1, col2 = st.columns(2)
    
    with col1:
        total_budget = st.number_input(
            "Budget total (€)",
            min_value=10.0,
            max_value=float(current_bankroll),
            value=min(300.0, float(current_bankroll)),
            step=10.0,
            key=f"total_budget_{event_url}"
        )
    
    with col2:
        kelly_strategy = st.selectbox(
            "Stratégie Kelly",
            options=["Kelly complet", "Demi-Kelly", "Quart-Kelly"],
            index=1,  # Demi-Kelly par défaut (plus prudent)
            key=f"kelly_strategy_{event_url}"
        )
        
        # Déterminer le diviseur Kelly en fonction de la stratégie
        if kelly_strategy == "Kelly complet":
            kelly_divisor = 1
        elif kelly_strategy == "Demi-Kelly":
            kelly_divisor = 2
        else:  # "Quart-Kelly"
            kelly_divisor = 4
    
    st.markdown("### Entrez les cotes proposées par les bookmakers")
    
    # Créer un dictionnaire pour stocker les cotes entrées
    if f"odds_dict_{event_url}" not in st.session_state:
        st.session_state[f"odds_dict_{event_url}"] = {}
    
    # Pour chaque combat avec prédiction
    bettable_fights = []
    for fight in fights:
        red_fighter_name = fight['red_fighter']
        blue_fighter_name = fight['blue_fighter']
        fight_key = f"{red_fighter_name}_vs_{blue_fighter_name}"
        
        # Vérifier si une prédiction existe pour ce combat
        prediction_data = predictions_data.get(fight_key, None)
        if not prediction_data or prediction_data.get('status') != 'success':
            continue
        
        # Récupérer le résultat ML ou classique
        ml_result = prediction_data.get('ml_result', None)
        classic_result = prediction_data.get('classic_result', None)
        
        # Préférer le résultat ML s'il existe
        result = ml_result if ml_result else classic_result
        if not result:
            continue
        
        # Créer deux colonnes
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write(f"**{red_fighter_name}** vs **{blue_fighter_name}**")
            
            # Afficher la prédiction
            winner = "Red" if result['red_probability'] > result['blue_probability'] else "Blue"
            winner_name = red_fighter_name if winner == "Red" else blue_fighter_name
            winner_prob = max(result['red_probability'], result['blue_probability'])
            
            st.write(f"Vainqueur prédit: **{winner_name}** (Probabilité: {winner_prob:.2f})")
        
        with col2:
            # Champ pour entrer la cote du bookmaker
            odds_key = f"odds_{fight_key}"
            
            # Initialiser la valeur si première utilisation
            if odds_key not in st.session_state[f"odds_dict_{event_url}"]:
                st.session_state[f"odds_dict_{event_url}"][odds_key] = 2.0
            
            # Champ de saisie de la cote
            odds = st.number_input(
                "Cote",
                min_value=1.01,
                value=st.session_state[f"odds_dict_{event_url}"][odds_key],
                step=0.05,
                format="%.2f",
                key=f"input_{odds_key}"
            )
            
            # Sauvegarder la valeur dans la session
            st.session_state[f"odds_dict_{event_url}"][odds_key] = odds
        
        # Ajouter aux combats sur lesquels parier si approprié
        bettable_fights.append({
            'fight_key': fight_key,
            'red_fighter': red_fighter_name,
            'blue_fighter': blue_fighter_name,
            'winner': winner,
            'winner_name': winner_name,
            'probability': winner_prob,
            'odds': odds
        })
    
    # Récupérer les recommandations existantes ou générer de nouvelles
    if has_existing_recommendations and not st.button("📊 Recalculer la stratégie de paris", key=f"recalculate_strategy_{event_url}"):
        filtered_fights = st.session_state.betting_recommendations[event_key]
        st.markdown("### 💰 Recommandations de paris")
        
        # Afficher les recommandations sauvegardées
        if filtered_fights:
            recommendation_data = []
            for fight in filtered_fights:
                recommendation_data.append({
                    "Combat": f"{fight['red_fighter']} vs {fight['blue_fighter']}",
                    "Pari sur": fight['winner_name'],
                    "Probabilité": f"{fight['probability']:.2f}",
                    "Cote": f"{fight['odds']:.2f}",
                    "Value": f"{fight['edge']*100:.1f}%",
                    "Rendement": f"{fight['value']:.2f}",  
                    "Montant": f"{fight['stake']:.2f} €",
                    "Gain potentiel": f"{fight['stake'] * (fight['odds']-1):.2f} €"
                })
                
            df = pd.DataFrame(recommendation_data)
            st.dataframe(df, use_container_width=True)
            
            # Résumé de la stratégie
            total_stake = sum(fight['stake'] for fight in filtered_fights)
            total_potential_profit = sum(fight['stake'] * (fight['odds']-1) for fight in filtered_fights)
            
            st.markdown(f"""
            <div style="background-color:rgba(76, 175, 80, 0.1); padding:15px; border-radius:10px; margin-top:15px;">
                <h4>Résumé de la stratégie</h4>
                <ul>
                    <li>Budget total: <b>{total_budget:.2f} €</b></li>
                    <li>Montant total misé: <b>{total_stake:.2f} €</b> ({total_stake/total_budget*100:.1f}% du budget)</li>
                    <li>Gain potentiel maximal: <b>{total_potential_profit:.2f} €</b> (ROI: {total_potential_profit/total_stake*100:.1f}%)</li>
                    <li>Stratégie Kelly utilisée: <b>{kelly_strategy}</b></li>
                    <li>Nombre de paris recommandés: <b>{len(filtered_fights)}</b></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Option pour enregistrer les paris seulement si pas déjà sauvegardés
            if event_url not in st.session_state.saved_bet_events:
                if st.button("💾 Enregistrer ces paris dans mon suivi", key=f"save_all_bets_{event_url}"):
                    # Cette partie s'exécute quand on clique sur le bouton
                    try:
                        successful_bets = 0
                        for fight in filtered_fights:
                            # Ajouter le pari à l'historique
                            result = add_manual_bet(
                                event_name=event_name,
                                event_date=datetime.datetime.now(),
                                fighter_red=fight['red_fighter'],
                                fighter_blue=fight['blue_fighter'],
                                pick=fight['winner_name'],
                                odds=fight['odds'],
                                stake=fight['stake'],
                                model_probability=fight['probability'],
                                kelly_fraction=kelly_divisor
                            )
                            if result:
                                successful_bets += 1
                        
                        # Sauvegarder l'état dans la session pour le prochain chargement
                        st.session_state.saved_bet_events[event_url] = successful_bets
                        
                        # Forcer un message qui apparaîtra au prochain rechargement
                        if successful_bets == len(filtered_fights):
                            st.success(f"Tous les paris ({successful_bets}) ont été enregistrés avec succès!")
                        elif successful_bets > 0:
                            st.warning(f"{successful_bets}/{len(filtered_fights)} paris ont été enregistrés. Certains paris n'ont pas pu être enregistrés.")
                        else:
                            st.error("Aucun pari n'a pu être enregistré.")
                    except Exception as e:
                        st.error(f"Erreur lors de l'enregistrement des paris: {e}")
        else:
            st.warning("Aucun combat ne correspond aux critères de value betting (confiance ≥ 65% et value positive).")
    else:
        # Bouton pour générer/régénérer la stratégie
        generate_btn = st.button("📊 Générer la stratégie de paris", key=f"generate_strategy_{event_url}")
        
        if generate_btn or (has_existing_recommendations and "recalculate" in st.session_state):
            # Filtrer les combats intéressants
            filtered_fights = []
            for fight in bettable_fights:
                # Vérifier la confiance du modèle
                if fight['probability'] < 0.65:
                    continue
                    
                # Vérifier le value betting
                implicit_prob = 1 / fight['odds']
                if implicit_prob >= fight['probability']:
                    continue
                
                value = fight['probability'] * fight['odds']
                if value < 1.15:
                    continue
                    
                # Calculer la fraction Kelly
                p = fight['probability']
                q = 1 - p
                b = fight['odds'] - 1
                kelly = (p * b - q) / b
                
                # Appliquer le diviseur Kelly
                fractional_kelly = kelly / kelly_divisor
                
                # Ajouter aux paris recommandés
                if fractional_kelly > 0:
                    filtered_fights.append({
                        **fight,
                        'kelly': kelly,
                        'fractional_kelly': fractional_kelly,
                        'edge': p - implicit_prob,
                        'value': value
                    })
            
            # Sauvegarder les recommandations dans la session state
            st.session_state.betting_recommendations[event_key] = filtered_fights
            
            # Afficher les résultats
            if not filtered_fights:
                st.warning("Aucun combat ne correspond aux critères de value betting (confiance ≥ 65% et value positive).")
            else:
                # Calculer la somme totale des fractions Kelly
                total_kelly = sum(fight['fractional_kelly'] for fight in filtered_fights)
                
                # Calculer les montants à miser
                for fight in filtered_fights:
                    if total_kelly > 0:
                        # Répartir le budget proportionnellement
                        fight['stake'] = total_budget * (fight['fractional_kelly'] / total_kelly)
                    else:
                        fight['stake'] = 0
                
                # Afficher les recommandations
                st.markdown("### 💰 Recommandations de paris")
                
                # Tableau des combats recommandés
                recommendation_data = []
                for fight in filtered_fights:
                    recommendation_data.append({
                        "Combat": f"{fight['red_fighter']} vs {fight['blue_fighter']}",
                        "Pari sur": fight['winner_name'],
                        "Probabilité": f"{fight['probability']:.2f}",
                        "Cote": f"{fight['odds']:.2f}",
                        "Value": f"{fight['edge']*100:.1f}%",
                        "Rendement": f"{fight['value']:.2f}",  
                        "Montant": f"{fight['stake']:.2f} €",
                        "Gain potentiel": f"{fight['stake'] * (fight['odds']-1):.2f} €"
                    })
                
                if recommendation_data:
                    df = pd.DataFrame(recommendation_data)
                    st.dataframe(df, use_container_width=True)
                    
                    # Résumé de la stratégie
                    total_stake = sum(fight['stake'] for fight in filtered_fights)
                    total_potential_profit = sum(fight['stake'] * (fight['odds']-1) for fight in filtered_fights)
                    
                    st.markdown(f"""
                    <div style="background-color:rgba(76, 175, 80, 0.1); padding:15px; border-radius:10px; margin-top:15px;">
                        <h4>Résumé de la stratégie</h4>
                        <ul>
                            <li>Budget total: <b>{total_budget:.2f} €</b></li>
                            <li>Montant total misé: <b>{total_stake:.2f} €</b> ({total_stake/total_budget*100:.1f}% du budget)</li>
                            <li>Gain potentiel maximal: <b>{total_potential_profit:.2f} €</b> (ROI: {total_potential_profit/total_stake*100:.1f}%)</li>
                            <li>Stratégie Kelly utilisée: <b>{kelly_strategy}</b></li>
                            <li>Nombre de paris recommandés: <b>{len(filtered_fights)}</b></li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Option pour enregistrer les paris seulement si pas déjà sauvegardés
                    if event_url not in st.session_state.saved_bet_events:
                        if st.button("💾 Enregistrer ces paris dans mon suivi", key=f"save_all_bets_{event_url}"):
                            try:
                                successful_bets = 0
                                for fight in filtered_fights:
                                    # Ajouter le pari à l'historique
                                    result = add_manual_bet(
                                        event_name=event_name,
                                        event_date=datetime.datetime.now(),
                                        fighter_red=fight['red_fighter'],
                                        fighter_blue=fight['blue_fighter'],
                                        pick=fight['winner_name'],
                                        odds=fight['odds'],
                                        stake=fight['stake'],
                                        model_probability=fight['probability'],
                                        kelly_fraction=kelly_divisor
                                    )
                                    if result:
                                        successful_bets += 1
                                
                                # Sauvegarder l'état dans la session pour le prochain chargement
                                st.session_state.saved_bet_events[event_url] = successful_bets
                                
                                # Afficher un message de confirmation
                                if successful_bets == len(filtered_fights):
                                    st.success(f"Tous les paris ({successful_bets}) ont été enregistrés avec succès!")
                                elif successful_bets > 0:
                                    st.warning(f"{successful_bets}/{len(filtered_fights)} paris ont été enregistrés. Certains paris n'ont pas pu être enregistrés.")
                                else:
                                    st.error("Aucun pari n'a pu être enregistré.")
                            except Exception as e:
                                st.error(f"Erreur lors de l'enregistrement des paris: {e}")

def show_upcoming_events_page():
    st.markdown("""
    <div style="text-align:center;">
        <h2>🗓️ Événements UFC à venir 🗓️</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Bouton pour récupérer UNIQUEMENT les noms des événements à venir
    if st.button("🔍 Récupérer les noms des événements à venir", key="load_events_btn", type="primary"):
        with st.spinner("Récupération des événements à venir..."):
            st.session_state.upcoming_events = get_upcoming_events(max_events=3)
            st.session_state.upcoming_events_timestamp = datetime.datetime.now()
            
            # Initialiser le dictionnaire des combats s'il n'existe pas déjà
            if 'upcoming_fights' not in st.session_state:
                st.session_state.upcoming_fights = {}
            
            # Initialiser les prédictions s'il n'existe pas déjà
            if 'fight_predictions' not in st.session_state:
                st.session_state.fight_predictions = {}
            
            st.success(f"Événements récupérés avec succès! {len(st.session_state.upcoming_events)} événements trouvés.")
    
    # Afficher les événements s'ils existent
    if st.session_state.get('upcoming_events'):
        # Bouton pour rafraîchir la liste des événements
        refresh_col, _ = st.columns([1, 3])
        with refresh_col:
            if st.button("🔄 Rafraîchir la liste des événements", key="refresh_events_btn"):
                with st.spinner("Mise à jour des événements..."):
                    st.session_state.upcoming_events = get_upcoming_events(max_events=3)
                    st.session_state.upcoming_events_timestamp = datetime.datetime.now()
                st.success("Liste des événements mise à jour!")
        
        # Créer des onglets pour chaque événement
        event_names = [event['name'] for event in st.session_state.upcoming_events]
        event_tabs = st.tabs(event_names)
        
        # Afficher chaque événement dans son propre onglet
        for i, (event, event_tab) in enumerate(zip(st.session_state.upcoming_events, event_tabs)):
            event_name = event['name']
            event_url = event['url']
            
            with event_tab:
                # Titre de l'événement
                st.header(f"🥊 {event_name}")
                st.markdown("---")
                
                # Vérifier si les combats pour cet événement sont déjà chargés
                fights = st.session_state.upcoming_fights.get(event_url, [])
                
                # Bouton pour charger les combats de cet événement spécifique
                if not fights:
                    if st.button(f"🔍 Charger les combats pour {event_name}", key=f"load_fights_btn_{i}"):
                        with st.spinner(f"Récupération des combats pour {event_name}..."):
                            fights = extract_upcoming_fights(event_url)
                            st.session_state.upcoming_fights[event_url] = fights
                            if fights and len(fights) > 0:
                                st.success(f"{len(fights)} combats chargés avec succès!")
                            else:
                                st.warning(f"Aucun combat trouvé pour {event_name}. L'événement n'est peut-être pas encore finalisé.")
                
                if not fights:
                    st.info(f"Cliquez sur le bouton 'Charger les combats pour {event_name}' pour voir les combats.")
                else:
                    # Afficher le nombre de combats chargés
                    st.write(f"**{len(fights)} combats chargés**")
                    
                    # Vérifier si les prédictions ont déjà été générées
                    predictions_generated = event_url in st.session_state.fight_predictions
                    
                    # Bouton pour générer les prédictions
                    if not predictions_generated:
                        if st.button(f"🔮 Générer les prédictions pour {event_name}", key=f"predict_fights_btn_{i}"):
                            with st.spinner(f"Génération des prédictions pour {len(fights)} combats..."):
                                # Initialiser le dictionnaire pour cet événement
                                st.session_state.fight_predictions[event_url] = {}
                                
                                # Générer les prédictions pour chaque combat
                                for fight in fights:
                                    red_fighter_name = fight['red_fighter']
                                    blue_fighter_name = fight['blue_fighter']
                                    
                                    # Trouver la correspondance dans la base de données
                                    red_match = find_best_match(red_fighter_name, app_data["fighters_dict"])
                                    blue_match = find_best_match(blue_fighter_name, app_data["fighters_dict"])
                                    
                                    fight_key = f"{red_fighter_name}_vs_{blue_fighter_name}"
                                    
                                    if not red_match or not blue_match:
                                        # Pas de prédiction si un combattant n'est pas reconnu
                                        st.session_state.fight_predictions[event_url][fight_key] = {
                                            'status': 'error',
                                            'message': "Données insuffisantes pour faire une prédiction"
                                        }
                                        continue
                                    
                                    # Récupérer les statistiques des combattants
                                    red_stats = app_data["fighters_dict"][red_match]
                                    blue_stats = app_data["fighters_dict"][blue_match]
                                    
                                    # Faire les prédictions
                                    classic_result = predict_fight_classic(red_stats, blue_stats)
                                    ml_result = None
                                    
                                    if app_data["ml_model"] is not None:
                                        ml_result = predict_with_ml(red_stats, blue_stats, app_data["ml_model"], app_data["scaler"], app_data["feature_names"])
                                        if ml_result is not None:
                                            ml_result['winner_name'] = red_match if ml_result['prediction'] == 'Red' else blue_match
                                    
                                    # Stocker les résultats
                                    st.session_state.fight_predictions[event_url][fight_key] = {
                                        'status': 'success',
                                        'red_match': red_match,
                                        'blue_match': blue_match,
                                        'red_stats': red_stats,
                                        'blue_stats': blue_stats,
                                        'classic_result': classic_result,
                                        'ml_result': ml_result
                                    }
                                
                                st.success(f"Prédictions générées pour {len(fights)} combats!")
                                st.session_state[f"show_strategy_{event_url}"] = True

                    
                    # Afficher les combats et leurs prédictions
                    for j, fight in enumerate(fights):
                        red_fighter_name = fight['red_fighter']
                        blue_fighter_name = fight['blue_fighter']
                        fight_key = f"{red_fighter_name}_vs_{blue_fighter_name}"
                        
                        st.markdown(f"### Combat {j+1}")
                        
                        # Vérifier si les prédictions ont été générées pour ce combat
                        prediction_data = st.session_state.fight_predictions.get(event_url, {}).get(fight_key, None)
                        
                        if not prediction_data:
                            # Afficher juste les noms des combattants sans prédiction
                            st.write(f"**🔴 {red_fighter_name}** vs **🔵 {blue_fighter_name}**")
                            st.info("Cliquez sur 'Générer les prédictions' pour obtenir l'analyse de ce combat.")
                            st.markdown("---")
                            continue
                        
                        if prediction_data['status'] == 'error':
                            # Afficher le message d'erreur
                            st.write(f"**🔴 {red_fighter_name}** vs **🔵 {blue_fighter_name}**")
                            st.info(prediction_data['message'])
                            st.markdown("---")
                            continue
                        
                        # Extraire les données de prédiction
                        red_match = prediction_data['red_match']
                        blue_match = prediction_data['blue_match']
                        red_stats = prediction_data['red_stats']
                        blue_stats = prediction_data['blue_stats']
                        classic_result = prediction_data['classic_result']
                        ml_result = prediction_data['ml_result']
                        
                        # Calculer les valeurs pour l'affichage
                        # Résultat classique
                        red_prob_classic = classic_result['red_probability']
                        blue_prob_classic = classic_result['blue_probability']
                        winner_classic = "Red" if red_prob_classic > blue_prob_classic else "Blue"
                        
                        # Résultat ML (si disponible)
                        if ml_result:
                            red_prob_ml = ml_result['red_probability']
                            blue_prob_ml = ml_result['blue_probability']
                            winner_ml = "Red" if red_prob_ml > blue_prob_ml else "Blue"
                            
                            # Consensus?
                            consensus = winner_classic == winner_ml
                            
                            # Utiliser le ML pour l'affichage principal
                            winner_color = "red" if winner_ml == "Red" else "blue"
                            winner_name = red_match if winner_ml == "Red" else blue_match
                            red_prob = red_prob_ml
                            blue_prob = blue_prob_ml
                            confidence = ml_result['confidence']
                            method = "Machine Learning"
                        else:
                            # Utiliser la méthode classique si ML n'est pas disponible
                            winner_color = "red" if winner_classic == "Red" else "blue"
                            winner_name = red_match if winner_classic == "Red" else blue_match
                            red_prob = red_prob_classic
                            blue_prob = blue_prob_classic
                            confidence = classic_result['confidence']
                            method = "Statistique"
                            consensus = True  # Pas de comparaison possible
                        
                        # Afficher les noms des combattants avec couleurs
                        fighters_col1, fighters_col2, fighters_col3 = st.columns([2, 1, 2])
                        
                        with fighters_col1:
                            st.markdown(f"<h4 style='color: #ff4d4d; text-align: right;'>{red_match}</h4>", unsafe_allow_html=True)
                        
                        with fighters_col2:
                            st.markdown("<h4 style='text-align: center;'>VS</h4>", unsafe_allow_html=True)
                        
                        with fighters_col3:
                            st.markdown(f"<h4 style='color: #4d79ff; text-align: left;'>{blue_match}</h4>", unsafe_allow_html=True)
                        
                        # Créer une barre de probabilités avec des colonnes Streamlit
                        red_pct = int(red_prob * 100)
                        blue_pct = int(blue_prob * 100)
                        
                        # S'assurer que chaque barre a au moins 1% pour qu'elle soit visible
                        if red_pct == 0: red_pct = 1
                        if blue_pct == 0: blue_pct = 1
                        
                        # S'assurer que la somme est exactement 100
                        total = red_pct + blue_pct
                        if total != 100:
                            # Ajuster proportionnellement
                            red_pct = int((red_pct / total) * 100)
                            blue_pct = 100 - red_pct
                        
                        prob_cols = st.columns([red_pct, blue_pct])
                        
                        with prob_cols[0]:
                            st.markdown(f"""
                            <div style="background-color: #ff4d4d; padding: 8px; color: white; text-align: center; border-radius: 5px 0 0 5px;">
                            {red_prob:.0%}
                            </div>
                            """, unsafe_allow_html=True)
                            
                        with prob_cols[1]:
                            st.markdown(f"""
                            <div style="background-color: #4d79ff; padding: 8px; color: white; text-align: center; border-radius: 0 5px 5px 0;">
                            {blue_prob:.0%}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Afficher le vainqueur prédit
                        pred_cols = st.columns([1, 1])
                        
                        with pred_cols[0]:
                            # Badge du gagnant
                            winner_bg = "#ff4d4d" if winner_color == "red" else "#4d79ff"
                            st.markdown(f"""
                            <div style="background-color: {winner_bg}; color: white; display: inline-block; 
                                 padding: 5px 10px; border-radius: 5px; margin-top: 5px;">
                                Vainqueur prédit: {winner_name}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with pred_cols[1]:
                            # Info sur la méthode et la confiance
                            conf_bg = "#4CAF50" if confidence == "Élevé" else "#FFC107"
                            conf_color = "#fff" if confidence == "Élevé" else "#000"
                            
                            st.markdown(f"""
                            <div style="text-align: right;">
                                <span style="color: #888; font-size: 0.9rem;">Méthode: {method}</span>
                                <span style="background-color: {conf_bg}; color: {conf_color}; 
                                     padding: 2px 8px; border-radius: 3px; margin-left: 5px; font-size: 0.8rem;">
                                    {confidence}
                                </span>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Si ML est disponible, afficher l'info sur le consensus
                        if ml_result:
                            consensus_text = "✅ Les deux méthodes prédisent le même vainqueur" if consensus else "⚠️ Les méthodes prédisent des vainqueurs différents"
                            consensus_color = "green" if consensus else "orange"
                            
                            st.markdown(f"""
                            <div style="text-align: center; margin-top: 5px; margin-bottom: 10px;">
                                <span style="color: {consensus_color};">{consensus_text}</span>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Créer un expander pour les détails du combat
                        with st.expander("Voir détails et statistiques"):
                            # Créer deux colonnes pour les prédictions
                            detail_cols = st.columns(2 if ml_result else 1)
                            
                            # Afficher la prédiction statistique
                            with detail_cols[0]:
                                st.markdown("### Prédiction Statistique")
                                winner_color_classic = "red" if classic_result['prediction'] == 'Red' else "blue"
                                winner_name_classic = classic_result['winner_name']
                                
                                st.markdown(f"**Vainqueur prédit:** <span style='color:{winner_color_classic};'>{winner_name_classic}</span>", unsafe_allow_html=True)
                                st.markdown(f"**Probabilités:** {classic_result['red_probability']:.2f} (Rouge) vs {classic_result['blue_probability']:.2f} (Bleu)")
                                st.markdown(f"**Confiance:** {classic_result['confidence']}")
                            
                            # Afficher la prédiction ML si disponible
                            if ml_result:
                                with detail_cols[1]:
                                    st.markdown("### Prédiction Machine Learning")
                                    winner_color_ml = "red" if ml_result['prediction'] == 'Red' else "blue"
                                    winner_name_ml = ml_result['winner_name']
                                    
                                    st.markdown(f"**Vainqueur prédit:** <span style='color:{winner_color_ml};'>{winner_name_ml}</span>", unsafe_allow_html=True)
                                    st.markdown(f"**Probabilités:** {ml_result['red_probability']:.2f} (Rouge) vs {ml_result['blue_probability']:.2f} (Bleu)")
                                    st.markdown(f"**Confiance:** {ml_result['confidence']}")
                            
                            # Afficher les statistiques comparatives
                            st.markdown("### Statistiques comparatives")
                            stats_df = create_stats_comparison_df(red_stats, blue_stats)
                            
                            # Appliquer un style conditionnel pour mettre en évidence les avantages
                            def highlight_advantage(row):
                                styles = [''] * len(row)
                                advantage = row['Avantage']
                                
                                if advantage == red_match:
                                    styles[1] = 'background-color: rgba(255, 0, 0, 0.2); font-weight: bold;'
                                elif advantage == blue_match:
                                    styles[2] = 'background-color: rgba(0, 0, 255, 0.2); font-weight: bold;'
                                
                                return styles
                            
                            # Appliquer le style et afficher
                            styled_df = stats_df.style.apply(highlight_advantage, axis=1)
                            st.dataframe(styled_df, use_container_width=True)
                            
                            # Visualisation radar
                            st.markdown("### Visualisation comparative")
                            radar_fig = create_radar_chart(red_stats, blue_stats)
                            st.plotly_chart(radar_fig, use_container_width=True)
                        
                        # Séparateur entre combats
                        st.markdown("---")
                    
                    # # AJOUT DE LA NOUVELLE FONCTIONNALITÉ: Stratégie de paris
                    # if predictions_generated:
                    #     # Afficher la section de stratégie de paris
                    #     show_betting_strategy_section(
                    #         event_url=event_url,
                    #         event_name=event_name,
                    #         fights=fights,
                    #         predictions_data=st.session_state.fight_predictions[event_url],
                    #         current_bankroll=app_data["current_bankroll"]
                    #     )
                        
                    if predictions_generated or st.session_state.get(f"show_strategy_{event_url}", False):
                        # Afficher la section de stratégie de paris
                        show_betting_strategy_section(
                            event_url=event_url,
                            event_name=event_name,
                            fights=fights,
                            predictions_data=st.session_state.fight_predictions[event_url],
                            current_bankroll=app_data["current_bankroll"]
                        )


                        
                        
    else:
        st.info("Cliquez sur le bouton 'Récupérer les noms des événements à venir' pour charger les prochains événements UFC.")

def show_bankroll_page():
    st.markdown("""
    <div style="text-align:center;">
        <h2>💰 Gestion de bankroll et paris 💰</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Afficher la bankroll actuelle
    st.metric("Bankroll actuelle", f"{app_data['current_bankroll']:.2f} €", delta=None)
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
            new_bankroll = app_data['current_bankroll'] + adjustment_amount
            action = "deposit"
            if not adjustment_note:
                adjustment_note = "Dépôt"
        elif adjustment_type == "Retrait":
            if adjustment_amount > app_data['current_bankroll']:
                st.error(f"Montant du retrait ({adjustment_amount:.2f} €) supérieur à la bankroll actuelle ({app_data['current_bankroll']:.2f} €)")
                new_bankroll = app_data['current_bankroll']
                action = None
            else:
                new_bankroll = app_data['current_bankroll'] - adjustment_amount
                action = "withdraw"
                if not adjustment_note:
                    adjustment_note = "Retrait"
        else:  # "Définir montant exact"
            new_bankroll = adjustment_amount
            action = "update"
            if not adjustment_note:
                adjustment_note = "Mise à jour manuelle"
        
        # Mettre à jour la bankroll si nécessaire
        if action and new_bankroll != app_data['current_bankroll']:
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
            
            # Mettre à jour app_data
            app_data['current_bankroll'] = new_bankroll
            
            st.success(f"Bankroll mise à jour: {new_bankroll:.2f} €")
            # Afficher la nouvelle bankroll sans rechargement complet
            st.metric("Bankroll actuelle", f"{new_bankroll:.2f} €", delta=None)
    
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
        manual_fighter_red = st.selectbox("Combattant rouge", options=app_data["fighter_names"], key="manual_fighter_red")
    with fighter_cols[1]:
        # Exclure le combattant rouge des options
        manual_blue_options = [name for name in app_data["fighter_names"] if name != manual_fighter_red]
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
            max_value=float(app_data['current_bankroll']),
            value=min(50.0, float(app_data['current_bankroll'])),
            step=5.0,
            format="%.2f",
            key="manual_stake"
        )
    
    # Afficher le gain potentiel
    potential_profit = manual_stake * (manual_odds - 1)
    st.info(f"Mise: {manual_stake:.2f} € @ {manual_odds:.2f} | Gain potentiel: {potential_profit:.2f} € | % de bankroll: {manual_stake/app_data['current_bankroll']*100:.1f}%")
    
    # Bouton pour enregistrer le pari
    if st.button("Enregistrer le pari", type="primary", key="save_manual_bet_btn"):
        if manual_stake > app_data['current_bankroll']:
            st.error(f"Mise ({manual_stake:.2f} €) supérieure à la bankroll actuelle ({app_data['current_bankroll']:.2f} €)")
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
                stake=manual_stake
            ):
                st.success(f"Pari enregistré avec succès! Mise de {manual_stake:.2f} € sur {manual_pick} @ {manual_odds:.2f}")
            else:
                st.error("Erreur lors de l'enregistrement du pari.")

def show_history_page():
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
                            new_bankroll = update_bet_result(update_bet_id, result, app_data['current_bankroll'])
                            
                            # Mettre à jour la bankroll dans app_data
                            app_data['current_bankroll'] = new_bankroll
                            
                            st.success(f"Pari mis à jour avec succès! Nouvelle bankroll: {new_bankroll:.2f} €")
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
                        
                        # stake
                        st.warning("⚠️ La suppression est définitive et ne peut pas être annulée.")
                        
                        # Bouton pour supprimer
                        if st.button("Supprimer le pari", key="delete_bet_btn"):
                            # Supprimer le pari
                            if delete_bet(delete_bet_id):
                                st.success(f"Pari #{delete_bet_id} supprimé avec succès!")
                            else:
                                st.error("Erreur lors de la suppression du pari.")
                    else:
                        st.info("Aucun pari à supprimer.")
        else:
            st.info("Aucun pari enregistré. Commencez à parier pour voir vos statistiques.")
    else:
        st.info("Aucune donnée d'historique disponible. Placez votre premier pari pour commencer à suivre vos performances.")
    
    # Téléchargement des données
    if has_bets and has_bankroll and os.path.exists(bets_file) and os.path.exists(bankroll_file):
        st.markdown("---")
        st.subheader("Exporter les données")
        
        download_cols = st.columns(2)
        
        with download_cols[0]:
            if os.path.exists(bets_file):
                with open(bets_file, 'rb') as f:
                    st.download_button(
                        label="Télécharger les paris (CSV)",
                        data=f,
                        file_name='ufc_bets_history.csv',
                        mime='text/csv',
                        key="download_bets_btn"
                    )
        
        with download_cols[1]:
            if os.path.exists(bankroll_file):
                with open(bankroll_file, 'rb') as f:
                    st.download_button(
                        label="Télécharger l'historique bankroll (CSV)",
                        data=f,
                        file_name='ufc_bankroll_history.csv',
                        mime='text/csv',
                        key="download_bankroll_btn"
                    )

# Lancer l'application
if __name__ == "__main__":
    main()
