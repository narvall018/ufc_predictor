# PARTIE 1 

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
import base64
import json
warnings.filterwarnings('ignore')

# AJOUT GITHUB: Configuration GitHub
GITHUB_API_URL = "https://api.github.com"
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", "")
GITHUB_REPO = st.secrets.get("GITHUB_REPO", "")  # Format: "username/repo-name"

def github_request(method, endpoint, data=None):
    """Fonction utilitaire pour les requêtes GitHub API"""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        st.error("Configuration GitHub manquante dans secrets.toml")
        return None
        
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json"
    }
    
    url = f"{GITHUB_API_URL}/repos/{GITHUB_REPO}/contents/{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "PUT":
            response = requests.put(url, headers=headers, json=data)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers, json=data)
        
        if response.status_code in [200, 201]:
            return response.json()
        else:
            st.error(f"Erreur GitHub API: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Erreur de connexion GitHub: {e}")
        return None

def read_github_file(file_path):
    """Lit un fichier depuis GitHub"""
    response = github_request("GET", file_path)
    if response and 'content' in response:
        content = base64.b64decode(response['content']).decode('utf-8')
        return content, response['sha']
    return None, None

def write_github_file(file_path, content, sha=None, commit_message="Update file"):
    """Écrit un fichier sur GitHub"""
    encoded_content = base64.b64encode(content.encode('utf-8')).decode('utf-8')
    
    data = {
        "message": commit_message,
        "content": encoded_content
    }
    
    if sha:
        data["sha"] = sha
    
    return github_request("PUT", file_path, data)

def sync_to_github(file_path, df, commit_message="Update data"):
    """Synchronise un DataFrame avec GitHub"""
    try:
        # Convertir le DataFrame en CSV
        csv_content = df.to_csv(index=False)
        
        # Lire le fichier existant pour obtenir le SHA
        existing_content, sha = read_github_file(file_path)
        
        # Écrire le fichier mis à jour
        result = write_github_file(file_path, csv_content, sha, commit_message)
        
        if result:
            return True
        else:
            return False
    except Exception as e:
        st.error(f"Erreur lors de la synchronisation avec GitHub: {e}")
        return False

def create_probability_bar(red_prob, blue_prob, red_name, blue_name):
    """
    Crée une barre de progression HTML bicolore pour afficher les probabilités
    
    Args:
        red_prob: Probabilité du combattant rouge (0-1)
        blue_prob: Probabilité du combattant bleu (0-1)
        red_name: Nom du combattant rouge
        blue_name: Nom du combattant bleu
    
    Returns:
        HTML string de la barre de progression
    """
    # S'assurer que les probabilités sont entre 0 et 100
    red_pct = max(0, min(100, red_prob * 100))
    blue_pct = max(0, min(100, blue_prob * 100))
    
    # Normaliser pour que le total soit 100%
    total = red_pct + blue_pct
    if total > 0:
        red_pct = (red_pct / total) * 100
        blue_pct = (blue_pct / total) * 100
    else:
        red_pct = 50
        blue_pct = 50
    
    html = f"""
    <div class="probability-container">
        <div class="probability-bar">
            <div class="probability-bar-red" style="width: {red_pct}%;">
                {red_pct:.0f}%
            </div>
            <div class="probability-bar-blue" style="width: {blue_pct}%;">
                {blue_pct:.0f}%
            </div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 5px; font-size: 0.9rem;">
            <span style="color: #E53935;">🔴 {red_name}</span>
            <span style="color: #1E88E5;">🔵 {blue_name}</span>
        </div>
    </div>
    """
    return html

# Configuration de la page avec un thème plus moderne
st.set_page_config(
    page_title="UFC Fight Predictor",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé amélioré avec une palette de couleurs cohérente
# AMÉLIORATION UI: Palette de couleurs harmonisée et variables CSS pour faciliter la modification
st.markdown("""
<style>
    /* Variables CSS pour une palette de couleurs cohérente */
    :root {
        --primary-red: #E53935;
        --primary-blue: #1E88E5;
        --primary-dark: #212121;
        --primary-light: #F5F5F5;
        --primary-accent: #FF9800;
        --success-color: #4CAF50;
        --warning-color: #FFC107;
        --error-color: #F44336;
        --card-bg: rgba(255, 255, 255, 0.08);
        --card-border: rgba(255, 255, 255, 0.1);
        --elevation-1: 0 2px 5px rgba(0, 0, 0, 0.1);
        --elevation-2: 0 4px 8px rgba(0, 0, 0, 0.2);
    }

    /* Styles globaux et typographie améliorés */
    body {
        font-family: 'Inter', 'Segoe UI', -apple-system, sans-serif;
        line-height: 1.6;
    }
    
    h1, h2, h3, h4, h5 {
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    
    /* Composants de base */
    .main-title {
        text-align: center;
        font-size: 2.8rem;
        font-weight: 800;
        color: var(--primary-red);
        margin-bottom: 10px;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.2);
    }
    
    .sub-title {
        text-align: center;
        font-size: 1.3rem;
        margin-bottom: 30px;
        color: #C0C0C0;
    }
    
    /* AMÉLIORATION UI: Design système de cartes */
    .card {
        background-color: var(--card-bg);
        border-radius: 12px;
        border: 1px solid var(--card-border);
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: var(--elevation-1);
        transition: all 0.2s ease;
    }
    
    .card:hover {
        box-shadow: var(--elevation-2);
        border-color: rgba(255, 255, 255, 0.2);
    }
    
    .card-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 15px;
        color: white;
    }
    
    /* Prédiction */
    .prediction-box {
        background-color: var(--card-bg);
        padding: 25px;
        border-radius: 12px;
        box-shadow: var(--elevation-1);
        margin-bottom: 20px;
        border: 1px solid var(--card-border);
    }
    
    .red-fighter {
        color: var(--primary-red);
        font-weight: bold;
    }
    
    .blue-fighter {
        color: var(--primary-blue);
        font-weight: bold;
    }
    
    .winner {
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    /* AMÉLIORATION UI: Métriques et KPIs */
    [data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 16px;
        border-radius: 10px;
        transition: all 0.2s ease;
    }
    
    [data-testid="stMetric"]:hover {
        background-color: rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stMetric"] > div {
        width: 100%;
    }
    
    [data-testid="stMetricLabel"] {
        display: flex;
        justify-content: center;
        font-weight: 500;
        font-size: 0.9rem;
        color: gray !important;
    }
    
    [data-testid="stMetricValue"] {
        display: flex;
        justify-content: center;
        font-weight: 700;
        font-size: 1.5rem;
    }
    
    [data-testid="stMetricDelta"] {
        display: flex;
        justify-content: center;
    }
    
    /* Compatibilité thème sombre pour les cartes */
    .stTabs [data-baseweb="tab-panel"] {
        background-color: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* AMÉLIORATION UI: Badges et labels plus modernes */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    
    .ml-badge {
        background-color: var(--success-color);
        color: white;
        padding: 5px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.75rem;
        margin-left: 10px;
    }
    
    .classic-badge {
        background-color: var(--primary-blue);
        color: white;
        padding: 5px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.75rem;
        margin-left: 10px;
    }
    
    /* AMÉLIORATION UI: Boîtes d'information */
    .info-box {
        background-color: rgba(255, 193, 7, 0.1);
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
        border-left: 3px solid var(--warning-color);
    }
    
    /* Divider pour les sections */
    .divider {
        border-top: 1px solid rgba(200, 200, 200, 0.15);
        margin: 25px 0;
    }
    
    /* AMÉLIORATION UI: Cartes de paris */
    .betting-card {
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        transition: transform 0.2s;
    }
    
    .betting-card:hover {
        transform: translateY(-2px);
    }
    
    .betting-card-red {
        background: linear-gradient(135deg, rgba(229, 57, 53, 0.1) 0%, rgba(229, 57, 53, 0.05) 100%);
        border: 1px solid rgba(229, 57, 53, 0.2);
    }
    
    .betting-card-blue {
        background: linear-gradient(135deg, rgba(30, 136, 229, 0.1) 0%, rgba(30, 136, 229, 0.05) 100%);
        border: 1px solid rgba(30, 136, 229, 0.2);
    }
    
    /* AMÉLIORATION UI: Statuts et indicateurs */
    .favorable {
        color: var(--success-color);
        font-weight: bold;
    }
    
    .neutral {
        color: var(--warning-color);
        font-weight: bold;
    }
    
    .unfavorable {
        color: var(--error-color);
        font-weight: bold;
    }
    
    /* AMÉLIORATION UI: Boutons d'action modernisés */
    .action-button {
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
        border: none;
        cursor: pointer;
        transition: all 0.2s;
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }
    
    .action-button:hover {
        transform: translateY(-1px);
        box-shadow: 0 3px 6px rgba(0, 0, 0, 0.1);
    }
    
    .delete-button {
        background-color: var(--error-color);
        color: white;
    }
    
    .update-button {
        background-color: var(--success-color);
        color: white;
    }

    /* AMÉLIORATION UI: Gestion de paris améliorée */
    .bet-management-container {
        background-color: rgba(255, 255, 255, 0.03);
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 25px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    /* Kelly Table */
    .kelly-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 8px;
        overflow: hidden;
    }
    
    .kelly-table th {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 12px;
        text-align: left;
        font-weight: 600;
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.9);
    }
    
    .kelly-table td {
        padding: 12px;
        text-align: left;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        font-size: 0.9rem;
    }
    
    .kelly-table tr:last-child td {
        border-bottom: none;
    }
    
    .kelly-table tr:hover {
        background-color: rgba(255, 255, 255, 0.05);
    }
    
    .kelly-highlight {
        background-color: rgba(76, 175, 80, 0.1);
    }
    
    /* Style pour les événements à venir */
    .upcoming-event {
        background: linear-gradient(145deg, rgba(40, 42, 54, 0.8) 0%, rgba(30, 31, 38, 0.8) 100%);
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.07);
        box-shadow: var(--elevation-1);
    }
    
    .upcoming-fight {
        background-color: rgba(248, 249, 250, 0.03);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 15px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* AMÉLIORATION UI: Loading state */
    .loading-spinner {
        text-align: center;
        margin: 25px 0;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 10px;
    }
    
    .loading-spinner::after {
        content: "";
        width: 30px;
        height: 30px;
        border: 3px solid rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        border-top-color: white;
        animation: spinner 1s ease-in-out infinite;
    }
    
    @keyframes spinner {
        to { transform: rotate(360deg); }
    }
    
    /* AMÉLIORATION UI: Combat cards plus modernes */
    .fight-card {
        background: linear-gradient(145deg, rgba(48, 51, 66, 0.7) 0%, rgba(33, 36, 46, 0.7) 100%);
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: var(--elevation-1);
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: all 0.3s ease;
    }
    
    .fight-card:hover {
        box-shadow: var(--elevation-2);
        border-color: rgba(255, 255, 255, 0.1);
        transform: translateY(-2px);
    }
    
    .fight-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
    }
    
    /* AMÉLIORATION UI: Sections d'événements */
    .event-section {
        background-color: rgba(38, 39, 48, 0.8);
        border-radius: 15px;
        padding: 25px;
        margin-bottom: 30px;
        box-shadow: var(--elevation-2);
        border: 1px solid rgba(255, 255, 255, 0.07);
    }
    
    .event-title {
        background: linear-gradient(135deg, var(--primary-red) 0%, #E63946 100%);
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 20px;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 2px 10px rgba(230, 57, 70, 0.3);
    }
    
    /* AMÉLIORATION UI: Combat cards modernisées */
    .fight-card-improved {
        background-color: rgba(200, 200, 200, 0.15);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: var(--elevation-1);
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: all 0.2s ease;
    }
    
    .fight-card-improved:hover {
        transform: translateY(-2px);
        box-shadow: var(--elevation-2);
        border-color: rgba(255, 255, 255, 0.1);
    }
    
    .fighters-banner {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
    }
    
    .fighter-name-red {
        color: var(--primary-red);
        font-weight: bold;
        font-size: 1.3rem;
        flex: 1;
        text-align: left;
    }
    
    .fighter-name-blue {
        color: var(--primary-blue);
        font-weight: bold;
        font-size: 1.3rem;
        flex: 1;
        text-align: right;
    }
    
    .vs-badge {
        background-color: rgba(0, 0, 0, 0.6);
        color: white;
        padding: 6px 12px;
        border-radius: 20px;
        margin: 0 10px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Suite du CSS amélioré
st.markdown("""
<style>
    /* AMÉLIORATION UI: Barres de probabilité plus intuitives */
    .probability-container {
        margin: 15px 0;
    }
    
    .probability-bar {
        height: 30px;
        width: 100%;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        position: relative;
        overflow: hidden;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.2);
    }
    
    .probability-bar-red {
        height: 100%;
        background: linear-gradient(90deg, #E53935 0%, #F44336 100%);
        float: left;
        text-align: center;
        line-height: 30px;
        color: white;
        font-weight: bold;
        border-radius: 15px 0 0 15px;
    }
    
    .probability-bar-blue {
        height: 100%;
        background: linear-gradient(90deg, #1E88E5 0%, #2196F3 100%);
        float: right;
        text-align: center;
        line-height: 30px;
        color: white;
        font-weight: bold;
        border-radius: 0 15px 15px 0;
    }
    
    /* AMÉLIORATION UI: Badges et étiquettes */
    .prediction-badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 10px;
    }
    
    .prediction-badge-red {
        background-color: var(--primary-red);
        color: white;
    }
    
    .prediction-badge-blue {
        background-color: var(--primary-blue);
        color: white;
    }
    
    .prediction-summary {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 10px;
    }
    
    .prediction-method {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.6);
    }
    
    .confidence-badge {
        padding: 4px 8px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .confidence-high {
        background-color: rgba(76, 175, 80, 0.2);
        color: #81c784;
        border: 1px solid rgba(76, 175, 80, 0.3);
    }
    
    .confidence-moderate {
        background-color: rgba(255, 193, 7, 0.2);
        color: #ffd54f;
        border: 1px solid rgba(255, 193, 7, 0.3);
    }
    
    /* AMÉLIORATION UI: Section d'accueil totalement repensée */
    .welcome-header {
        text-align: center;
        padding: 50px 0;
        background: linear-gradient(135deg, rgba(229, 57, 53, 0.8) 0%, rgba(30, 136, 229, 0.8) 100%);
        border-radius: 15px;
        margin-bottom: 40px;
        color: white;
        box-shadow: var(--elevation-2);
    }
    
    .welcome-title {
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 15px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        letter-spacing: -0.02em;
    }
    
    .welcome-subtitle {
        font-size: 1.5rem;
        margin-bottom: 20px;
        max-width: 700px;
        margin-left: auto;
        margin-right: auto;
        color: rgba(255, 255, 255, 0.9);
    }
    
    /* AMÉLIORATION UI: Feature cards plus visuelles */
    .home-card {
        background: linear-gradient(145deg, rgba(48, 51, 66, 0.7) 0%, rgba(33, 36, 46, 0.7) 100%);
        border-radius: 12px;
        padding: 30px;
        margin-bottom: 25px;
        box-shadow: var(--elevation-1);
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: all 0.3s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    
    .home-card:hover {
        transform: translateY(-5px);
        box-shadow: var(--elevation-2);
        border-color: rgba(255, 255, 255, 0.1);
    }
    
    .feature-icon {
        font-size: 2.8rem;
        margin-bottom: 20px;
        text-align: center;
        color: var(--primary-accent);
    }
    
    .feature-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 15px;
        text-align: center;
        color: white;
    }
    
    .feature-description {
        text-align: center;
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.8);
        flex-grow: 1;
    }
    
    /* AMÉLIORATION UI: Stratégie kelly */
    .kelly-box {
        background: linear-gradient(145deg, rgba(40, 124, 70, 0.1) 0%, rgba(30, 100, 60, 0.1) 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
        border-left: 3px solid var(--success-color);
        box-shadow: var(--elevation-1);
    }
    
    .kelly-title {
        color: var(--success-color);
        font-weight: bold;
        margin-bottom: 15px;
        font-size: 1.2rem;
    }
    
    /* AMÉLIORATION UI: Placement de paris */
    .bet-placement-box {
        background: linear-gradient(145deg, rgba(25, 70, 186, 0.1) 0%, rgba(20, 60, 150, 0.1) 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
        border-left: 3px solid var(--primary-blue);
        box-shadow: var(--elevation-1);
    }
    
    .bet-placement-title {
        color: var(--primary-blue);
        font-weight: bold;
        margin-bottom: 15px;
        font-size: 1.2rem;
    }
    
    /* AMÉLIORATION UI: Stratégie de paris */
    .betting-strategy-box {
        background: linear-gradient(145deg, rgba(76, 175, 80, 0.1) 0%, rgba(56, 142, 60, 0.1) 100%);
        border-radius: 12px;
        padding: 25px;
        margin: 20px 0;
        border-left: 3px solid var(--success-color);
        box-shadow: var(--elevation-1);
    }
    
    .strategy-title {
        color: var(--success-color);
        font-weight: bold;
        margin-bottom: 15px;
        font-size: 1.3rem;
    }
    
    .strategy-summary {
        background-color: rgba(76, 175, 80, 0.05);
        padding: 15px;
        border-radius: 10px;
        margin-top: 15px;
        border: 1px solid rgba(76, 175, 80, 0.1);
    }
    
    .value-betting-positive {
        color: var(--success-color);
        font-weight: bold;
    }
    
    .value-betting-negative {
        color: var(--error-color);
        font-weight: bold;
    }

    /* NOUVELLE SECTION UI: Responsive design */
    @media (max-width: 768px) {
        .welcome-title {
            font-size: 2.5rem;
        }
        
        .welcome-subtitle {
            font-size: 1.2rem;
        }
        
        .feature-icon {
            font-size: 2.2rem;
        }
        
        .main-title {
            font-size: 2.2rem;
        }
        
        .fight-card, .prediction-box, .kelly-box, .bet-placement-box {
            padding: 15px;
        }
        
        .fighter-name-red, .fighter-name-blue {
            font-size: 1.1rem;
        }
        
        /* Adaptation des tableaux */
        .kelly-table th, .kelly-table td {
            padding: 8px 5px;
            font-size: 0.8rem;
        }
    }

    /* NOUVELLE SECTION UI: Animations subtiles */
    .section-fade-in {
        animation: fadeIn 0.5s ease-in-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
/* NOUVELLE SECTION UI: Améliorations des formulaires */
input[type="number"], input[type="text"], input[type="date"], select {
    background-color: rgba(255, 255, 255, 0.08) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 8px !important;
    /* color: supprimé pour laisser Streamlit gérer les couleurs selon le thème */
    transition: all 0.2s !important;
}

    
    input[type="number"]:focus, input[type="text"]:focus, 
    input[type="date"]:focus, select:focus {
        border-color: var(--primary-blue) !important;
        box-shadow: 0 0 0 1px var(--primary-blue) !important;
    }
    
    /* Adaptation des contrôles Streamlit */
    div.stButton > button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5em 1em;
        transition: all 0.2s;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 3px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Correction des styles pour recommandations */
    .recommendation-box {
        margin-top: 15px;
    }
    
    .recommendation-label {
        margin-bottom: 5px;
        font-weight: 500;
    }
    
    .recommendation-value {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .recommendation-favorable {
        background-color: rgba(76, 175, 80, 0.2);
        color: #4CAF50;
        border: 1px solid rgba(76, 175, 80, 0.5);
    }
    
    .recommendation-neutral {
        background-color: rgba(255, 193, 7, 0.2);
        color: #FFC107;
        border: 1px solid rgba(255, 193, 7, 0.5);
    }
    
    .recommendation-unfavorable {
        background-color: rgba(244, 67, 54, 0.2);
        color: #F44336;
        border: 1px solid rgba(244, 67, 54, 0.5);
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
    """Version optimisée du chargement des données avec retour visuel amélioré"""
    data = {
        "ml_model": None,
        "scaler": None,
        "feature_names": None,
        "fighters": [],
        "fighters_dict": {},
        "fighter_names": [],
        "current_bankroll": 1000
    }
    
    # AMÉLIORATION UI: Utilisation d'un placeholder pour afficher les étapes de chargement
    loading_status = st.empty()
    
    # Charger le modèle ML avec gestion d'erreur améliorée
    loading_status.info("Chargement du modèle ML...")
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
        loading_status.error(f"Erreur lors du chargement du modèle ML: {e}")
    
    # Optimisation du chargement des stats des combattants
    loading_status.info("Chargement des données des combattants...")
    fighter_stats_path = 'fighters_stats.txt'
    if os.path.exists(fighter_stats_path):
        fighters = load_fighters_stats(fighter_stats_path)
        fighters = deduplicate_fighters(fighters)
        data["fighters"] = fighters
        
        # Création optimisée du dictionnaire
        data["fighters_dict"] = {fighter['name']: fighter for fighter in fighters}
        data["fighter_names"] = sorted([fighter['name'] for fighter in fighters])
    
    # Initialiser/Charger la bankroll
    loading_status.info("Initialisation de la bankroll...")
    data["current_bankroll"] = init_bankroll()
    init_bets_file()
    
    # Effacer les messages de statut après le chargement
    loading_status.empty()
    
    return data

@st.cache_data(ttl=86400, show_spinner=False)
def make_request(url, max_retries=3, delay_range=(0.5, 1.5)):
    """Requête HTTP avec cache intelligent et gestion des erreurs améliorée"""
    # Vérifier d'abord dans le cache
    if url in request_cache:
        return request_cache[url]
    
    # AMÉLIORATION: Affichage d'une erreur plus informative en cas d'échec
    error_message = None
    
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
                error_message = f"Erreur d'accès: Limite de requêtes atteinte (Code {response.status_code})"
                time.sleep(5 * (attempt + 1))
            else:
                error_message = f"Erreur serveur: Code {response.status_code}"
        except requests.RequestException as e:
            error_message = f"Erreur de connexion: {str(e)}"
        except Exception as e:
            error_message = f"Erreur inattendue: {str(e)}"
    
    # Si toutes les tentatives échouent, retourner None avec une erreur informative
    if error_message:
        st.warning(f"Échec de la requête à {url}: {error_message}")
        
    return None

# PARTIE 2 

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
    """Récupère les événements UFC à venir avec une UI améliorée pour les échecs"""
    # AMÉLIORATION UI: Placeholder pour les messages de progression
    progress_message = st.empty()
    progress_message.info("Recherche des événements UFC à venir...")
    
    # URL pour les événements à venir
    urls = [
        "http://ufcstats.com/statistics/events/upcoming",
        "http://ufcstats.com/statistics/events/completed"  # Fallback pour les événements récents
    ]
    
    events = []
    
    for url in urls:
        progress_message.info(f"Tentative de récupération depuis {url}...")
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
    
    # Effacer le message de progression
    progress_message.empty()
    
    # AMÉLIORATION UI: Retourner aussi un statut pour une meilleure expérience utilisateur
    result = {
        'events': events[:max_events],
        'status': 'success' if events else 'error',
        'message': f"{len(events)} événements trouvés" if events else "Aucun événement trouvé"
    }
    
    return result

@st.cache_data(ttl=86400, show_spinner=False)  # Augmentation du TTL à 24h
def extract_upcoming_fights(event_url):
    """Récupère les combats à venir avec une meilleure gestion des erreurs"""
    # AMÉLIORATION UI: Placeholder pour message de progression
    progress_message = st.empty()
    progress_message.info(f"Récupération des combats depuis {event_url}...")
    
    resp = make_request(event_url)
    if not resp:
        progress_message.empty()
        return {'fights': [], 'status': 'error', 'message': 'Impossible de récupérer les données'}

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

    # Effacer le message de progression
    progress_message.empty()
    
    # AMÉLIORATION UI: Retourner aussi un statut
    result = {
        'fights': fights,
        'status': 'success' if fights else 'warning',
        'message': f"{len(fights)} combats trouvés" if fights else "Aucun combat trouvé pour cet événement"
    }
    
    return result

def find_best_match(name, fighters_dict):
    """Recherche le meilleur match pour un nom de combattant dans les stats avec feedback amélioré"""
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
    
    # AMÉLIORATION: Recherche floue plus robuste
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
            # Considérer les matchs partiels (ex: "Mcgregor" vs "McGregor")
            for fighter_word in fighter_words:
                # Match de préfixe (plus fort)
                if fighter_word.startswith(word) or word.startswith(fighter_word):
                    score += 1.5
                # Match partiel
                elif word in fighter_word or fighter_word in word:
                    score += 1
        
        # Bonus pour les noms de longueur similaire
        length_diff = abs(len(name) - len(fighter_name))
        if length_diff <= 3:
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
    Prédit l'issue d'un combat avec le modèle ML avec une meilleure gestion des erreurs
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

# PARTIE 3 

# FONCTIONS DE VISUALISATION AMÉLIORÉES
@st.cache_data(ttl=3600)
def create_radar_chart(fighter_a, fighter_b):
    """Crée un graphique radar comparant les attributs des combattants avec une meilleure esthétique"""
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
    
    # AMÉLIORATION UI: Utiliser Plotly avec des couleurs et un design moderne
    fig = go.Figure()
    
    # Ajouter les traces pour chaque combattant avec des couleurs améliorées et plus de transparence
    fig.add_trace(go.Scatterpolar(
        r=a_values_norm + [a_values_norm[0]],  # Fermer le polygone
        theta=categories + [categories[0]],
        fill='toself',
        name=fighter_a['name'],
        line_color='#E53935',
        fillcolor='rgba(229, 57, 53, 0.3)',
        line=dict(width=2)
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=b_values_norm + [b_values_norm[0]],  # Fermer le polygone
        theta=categories + [categories[0]],
        fill='toself',
        name=fighter_b['name'],
        line_color='#1E88E5',
        fillcolor='rgba(30, 136, 229, 0.3)',
        line=dict(width=2)
    ))
    
    # Configurer la mise en page moderne
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1.2],
                showticklabels=False,
                gridcolor='rgba(255, 255, 255, 0.1)',
                linecolor='rgba(255, 255, 255, 0.1)'
            ),
            angularaxis=dict(
                gridcolor='rgba(255, 255, 255, 0.1)',
                linecolor='rgba(255, 255, 255, 0.1)'
            ),
            bgcolor='rgba(0, 0, 0, 0)'
        ),
        title={
            'text': "Comparaison des attributs",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=18, color='white')
        },
        height=500,
        margin=dict(l=80, r=80, t=100, b=80),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='gray'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

@st.cache_data(ttl=3600)
def create_strengths_weaknesses_chart(fighter_a, fighter_b):
    """Crée un graphique des forces et faiblesses des combattants avec un design amélioré"""
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
    
    # AMÉLIORATION UI: Graphique avec des couleurs et styles modernes
    fig = px.bar(
        df, 
        x='Attribute', 
        y='Value', 
        color='Fighter',
        barmode='group',
        color_discrete_map={fighter_a['name']: '#E53935', fighter_b['name']: '#1E88E5'},
        title="Forces et faiblesses comparatives"
    )
    
    fig.update_layout(
        yaxis_title="Score normalisé",
        xaxis_title="",
        legend_title="Combattant",
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)',
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)',
            zeroline=False,
            range=[0, 1.05]
        ),
        bargap=0.2,
        bargroupgap=0.1,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    # Style moderne pour les barres
    fig.update_traces(
        marker_line_width=0, 
        opacity=0.8,
        hovertemplate='<b>%{x}</b><br>Score: %{y:.2f}<extra></extra>'
    )
    
    return fig

@st.cache_data(ttl=3600)
def create_style_analysis_chart(fighter_a, fighter_b):
    """Crée un graphique d'analyse des styles de combat avec un design moderne"""
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
    
    # AMÉLIORATION UI: Graphique modernisé
    fig = go.Figure()
    
    # Ajouter un arrière-plan de quadrant avec contour gris
    fig.add_shape(
        type="rect", xref="paper", yref="paper",
        x0=0, y0=0, x1=1, y1=1,
        line=dict(color="rgba(128, 128, 128, 0.3)", width=1)
    )
    
    # Ajouter des lignes de quadrant avec style amélioré en gris
    fig.add_shape(
        type="line", xref="paper", yref="paper",
        x0=0.5, y0=0, x1=0.5, y1=1,
        line=dict(color="rgba(128, 128, 128, 0.4)", width=1, dash="dot")
    )
    
    fig.add_shape(
        type="line", xref="paper", yref="paper",
        x0=0, y0=0.5, x1=1, y1=0.5,
        line=dict(color="rgba(128, 128, 128, 0.4)", width=1, dash="dot")
    )
    
    # Ajouter les points pour chaque combattant avec style amélioré
    fig.add_trace(go.Scatter(
        x=[a_sg_norm],
        y=[a_ad_norm],
        mode='markers+text',
        marker=dict(
            size=20, 
            color='#E53935',
            line=dict(width=2, color='white')
        ),
        text=fighter_a['name'],
        textposition="top center",
        name=fighter_a['name'],
        hovertemplate=f'<b>{fighter_a["name"]}</b><br>Style: %{{x:.2f}}<br>Approche: %{{y:.2f}}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=[b_sg_norm],
        y=[b_ad_norm],
        mode='markers+text',
        marker=dict(
            size=20, 
            color='#1E88E5',
            line=dict(width=2, color='white')
        ),
        text=fighter_b['name'],
        textposition="top center",
        name=fighter_b['name'],
        hovertemplate=f'<b>{fighter_b["name"]}</b><br>Style: %{{x:.2f}}<br>Approche: %{{y:.2f}}<extra></extra>'
    ))
    
    # Ajouter des annotations pour les quadrants avec style amélioré
    fig.add_annotation(
        x=0.25, y=0.75, 
        text="Grappler Agressif",
        showarrow=False,
        font=dict(color="rgba(128, 128, 128, 0.7)", size=12)
    )
    fig.add_annotation(
        x=0.75, y=0.75, 
        text="Striker Agressif",
        showarrow=False,
        font=dict(color="rgba(128, 128, 128, 0.7)", size=12)
    )
    fig.add_annotation(
        x=0.25, y=0.25, 
        text="Grappler Défensif",
        showarrow=False,
        font=dict(color="rgba(128, 128, 128, 0.7)", size=12)
    )
    fig.add_annotation(
        x=0.75, y=0.25, 
        text="Striker Défensif",
        showarrow=False,
        font=dict(color="rgba(128, 128, 128, 0.7)", size=12)
    )

    
    # Configurer la mise en page moderne
    fig.update_layout(
        title={
            'text': "Analyse de style de combat",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=18, color='white')
        },
        xaxis_title="Style de combat (Grappler ← → Striker)",
        yaxis_title="Approche (Défensif ← → Agressif)",
        xaxis=dict(
            range=[0, 1.1],
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        yaxis=dict(
            range=[0, 1.1],
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=50, r=50, t=100, b=50)
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

# FONCTIONS POUR LA GESTION DE BANKROLL ET PARIS - AVEC SYNCHRONISATION GITHUB

def init_bankroll():
    """
    Initialise ou charge la bankroll depuis le fichier avec synchronisation GitHub
    """
    bets_dir = "bets"
    bankroll_file = os.path.join(bets_dir, "bankroll.csv")
    
    try:
        # Créer le dossier s'il n'existe pas
        if not os.path.exists(bets_dir):
            os.makedirs(bets_dir)
        
        # Charger la bankroll si le fichier existe
        if os.path.exists(bankroll_file):
            bankroll_df = pd.read_csv(bankroll_file)
            if not bankroll_df.empty:
                return bankroll_df.iloc[-1]["amount"]
        
        # AJOUT GITHUB: Essayer de charger depuis GitHub si le fichier local n'existe pas
        github_content, sha = read_github_file("bets/bankroll.csv")
        if github_content:
            # Écrire le fichier localement
            with open(bankroll_file, 'w') as f:
                f.write(github_content)
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
        
        # Sauvegarder le fichier localement
        bankroll_df.to_csv(bankroll_file, index=False)
        
        # AJOUT GITHUB: Synchroniser avec GitHub
        sync_to_github("bets/bankroll.csv", bankroll_df, "Initialize bankroll")
        
        st.success("Bankroll initialisée avec succès à 1000€")
        return 1000
    
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation de la bankroll: {e}")
        # Valeur par défaut en cas d'erreur
        return 1000

def init_bets_file():
    """
    Initialise le fichier de paris s'il n'existe pas avec synchronisation GitHub
    """
    bets_dir = "bets"
    bets_file = os.path.join(bets_dir, "bets.csv")
    
    try:
        # Créer le dossier s'il n'existe pas
        if not os.path.exists(bets_dir):
            os.makedirs(bets_dir)
        
        # Créer le fichier s'il n'existe pas
        if not os.path.exists(bets_file):
            # AJOUT GITHUB: Essayer de charger depuis GitHub d'abord
            github_content, sha = read_github_file("bets/bets.csv")
            if github_content:
                # Écrire le fichier localement
                with open(bets_file, 'w') as f:
                    f.write(github_content)
                st.success("Fichier de paris chargé depuis GitHub")
            else:
                # Créer un nouveau fichier
                columns = ["bet_id", "date_placed", "event_name", "event_date", 
                        "fighter_red", "fighter_blue", "pick", "odds", 
                        "stake", "kelly_fraction", "model_probability", 
                        "status", "result", "profit", "roi"]
                
                empty_df = pd.DataFrame(columns=columns)
                empty_df.to_csv(bets_file, index=False)
                
                # AJOUT GITHUB: Synchroniser avec GitHub
                sync_to_github("bets/bets.csv", empty_df, "Initialize bets file")
                
                st.success("Fichier de paris initialisé avec succès")
    
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation du fichier de paris: {e}")

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
    Met à jour le résultat d'un pari existant et ajuste la bankroll avec synchronisation GitHub
    
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
    
    try:
        if not os.path.exists(bets_file):
            st.error("Fichier de paris introuvable.")
            return current_bankroll
        
        # Charger les fichiers
        bets_df = pd.read_csv(bets_file)
        if os.path.exists(bankroll_file):
            bankroll_df = pd.read_csv(bankroll_file)
        else:
            # Créer un nouveau fichier bankroll si nécessaire
            bankroll_df = pd.DataFrame(columns=["date", "amount", "action", "note"])
        
        # Vérifier si le pari existe
        if bet_id not in bets_df["bet_id"].values:
            st.error(f"Pari #{bet_id} introuvable.")
            return current_bankroll
        
        # Récupérer les informations du pari
        bet_row = bets_df[bets_df["bet_id"] == bet_id].iloc[0]
        stake = float(bet_row["stake"])
        odds = float(bet_row["odds"])
        
        # Vérifier si le pari est déjà fermé
        if bet_row["status"] == "closed":
            st.warning(f"Le pari #{bet_id} est déjà fermé avec le résultat: {bet_row['result']}")
            return current_bankroll
        
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
        
        # AJOUT GITHUB: Synchroniser le fichier des paris avec GitHub
        sync_to_github("bets/bets.csv", bets_df, f"Update bet #{bet_id} result: {result}")
        
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
        
        # AJOUT GITHUB: Synchroniser le fichier de bankroll avec GitHub
        sync_to_github("bets/bankroll.csv", bankroll_df, f"Update bankroll after bet #{bet_id}")
        
        # Feedback à l'utilisateur
        if result == "win":
            st.success(f"Victoire! Gain de {profit:.2f}€ - Nouvelle bankroll: {new_bankroll:.2f}€")
        elif result == "loss":
            st.warning(f"Perte de {stake:.2f}€ - Nouvelle bankroll: {new_bankroll:.2f}€")
        else:
            st.info(f"Pari annulé - Bankroll inchangée: {new_bankroll:.2f}€")
        
        return new_bankroll
    
    except Exception as e:
        st.error(f"Erreur lors de la mise à jour du pari: {e}")
        return current_bankroll

def delete_bet(bet_id):
    """
    Supprime un pari du fichier historique avec synchronisation GitHub
    
    Args:
        bet_id: Identifiant du pari à supprimer
    
    Returns:
        True si la suppression a réussi, False sinon
    """
    bets_dir = "bets"
    bets_file = os.path.join(bets_dir, "bets.csv")
    
    try:
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
        
        # Récupérer les infos du pari pour le message de confirmation
        bet_info = f"{bet_row['pick']} @ {bet_row['odds']} (Mise: {bet_row['stake']}€)"
        
        # Supprimer le pari
        bets_df = bets_df[bets_df["bet_id"] != bet_id]
        bets_df.to_csv(bets_file, index=False)
        
        # AJOUT GITHUB: Synchroniser avec GitHub
        sync_to_github("bets/bets.csv", bets_df, f"Delete bet #{bet_id}")
        
        st.success(f"Pari #{bet_id} ({bet_info}) supprimé avec succès.")
        return True
    
    except Exception as e:
        st.error(f"Erreur lors de la suppression du pari: {e}")
        return False

def add_manual_bet(event_name, event_date, fighter_red, fighter_blue, pick, odds, stake, model_probability=None, kelly_fraction=None):
    """
    Ajoute un pari manuellement à l'historique avec synchronisation GitHub
    
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
    
    try:
        # Créer le dossier et le fichier s'ils n'existent pas
        if not os.path.exists(bets_dir):
            os.makedirs(bets_dir)
        
        # Validation des données
        if not event_name or not fighter_red or not fighter_blue or not pick:
            st.error("Tous les champs texte doivent être remplis.")
            return False
        
        if odds < 1.01:
            st.error("La cote doit être supérieure à 1.01.")
            return False
        
        if stake <= 0:
            st.error("La mise doit être supérieure à 0.")
            return False
        
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
        
        # AJOUT GITHUB: Synchroniser avec GitHub
        sync_to_github("bets/bets.csv", bets_df, f"Add new bet #{new_id}: {pick} @ {odds}")
        
        return True
    
    except Exception as e:
        st.error(f"Erreur lors de l'ajout du pari: {e}")
        return False

def get_betting_summary(bets_df):
    """
    Génère un résumé des statistiques de paris avec des métriques supplémentaires
    
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
            "roi": 0,
            "avg_odds": 0,
            "avg_stake": 0,
            "biggest_win": 0,
            "biggest_loss": 0,
            "current_streak": 0,
            "longest_win_streak": 0,
            "current_streak_type": "none"
        }
    
    # Filtrer les paris fermés
    closed_bets = bets_df[bets_df["status"] == "closed"]
    open_bets = bets_df[bets_df["status"] == "open"]
    
    # Nombre de paris
    total_bets = len(bets_df)
    open_bets_count = len(open_bets)
    closed_bets_count = len(closed_bets)
    
    # AMÉLIORATION: Vérifier si nous avons des paris fermés avant de calculer les métriques
    if closed_bets_count == 0:
        return {
            "total_bets": total_bets,
            "open_bets": open_bets_count,
            "closed_bets": 0,
            "wins": 0,
            "losses": 0,
            "voids": 0,
            "win_rate": 0,
            "total_staked": open_bets["stake"].sum() if not open_bets.empty else 0,
            "total_profit": 0,
            "roi": 0,
            "avg_odds": open_bets["odds"].mean() if not open_bets.empty else 0,
            "avg_stake": open_bets["stake"].mean() if not open_bets.empty else 0,
            "biggest_win": 0,
            "biggest_loss": 0,
            "current_streak": 0,
            "longest_win_streak": 0,
            "current_streak_type": "none"
        }
    
    # Résultats des paris fermés
    wins = len(closed_bets[closed_bets["result"] == "win"])
    losses = len(closed_bets[closed_bets["result"] == "loss"])
    voids = len(closed_bets[closed_bets["result"] == "void"])
    
    # Taux de réussite
    win_rate = wins / max(wins + losses, 1) * 100
    
    # Montants financiers
    total_staked_closed = closed_bets["stake"].sum()
    total_staked_open = open_bets["stake"].sum() if not open_bets.empty else 0
    total_staked = total_staked_closed + total_staked_open
    total_profit = closed_bets["profit"].sum()
    
    # ROI global
    roi = total_profit / max(total_staked_closed, 1) * 100
    
    # AMÉLIORATION: Nouvelles métriques
    avg_odds = closed_bets["odds"].mean()
    avg_stake = closed_bets["stake"].mean()
    
    # Plus gros gain et perte
    biggest_win = closed_bets["profit"].max() if not closed_bets.empty else 0
    biggest_loss = closed_bets["profit"].min() if not closed_bets.empty else 0
    
    # AMÉLIORATION: Série de victoires/défaites actuelles
    if not closed_bets.empty:
        # Trier par date pour obtenir les paris dans l'ordre chronologique
        sorted_bets = closed_bets.sort_values(by=["date_placed", "bet_id"])
        results = sorted_bets[sorted_bets["result"].isin(["win", "loss"])]["result"].tolist()
        
        current_streak = 0
        longest_win_streak = 0
        current_streak_type = "none"
        
        if results:
            current_type = results[-1]
            current_streak_type = current_type
            
            # Calculer la série actuelle
            for result in reversed(results):
                if result == current_type:
                    current_streak += 1
                else:
                    break
            
            # Calculer la plus longue série de victoires
            temp_streak = 0
            for result in results:
                if result == "win":
                    temp_streak += 1
                    longest_win_streak = max(longest_win_streak, temp_streak)
                else:
                    temp_streak = 0
    else:
        current_streak = 0
        longest_win_streak = 0
        current_streak_type = "none"
    
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
        "roi": roi,
        "avg_odds": avg_odds,
        "avg_stake": avg_stake,
        "biggest_win": biggest_win,
        "biggest_loss": biggest_loss,
        "current_streak": current_streak,
        "longest_win_streak": longest_win_streak,
        "current_streak_type": current_streak_type
    }

# PARTIE 4 

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

# AMÉLIORATION UI: Session state pour l'interface utilisateur
if 'theme_mode' not in st.session_state:
    st.session_state.theme_mode = "dark"

if 'show_loading_welcome' not in st.session_state:
    st.session_state.show_loading_welcome = True

# Au début du script, après les autres initialisations de session_state
if 'odds_dicts' not in st.session_state:
    st.session_state.odds_dicts = {}

if 'saved_bet_events' not in st.session_state:
    st.session_state.saved_bet_events = {}
    
if 'betting_recommendations' not in st.session_state:
    st.session_state.betting_recommendations = {}

# Charger les données une seule fois au démarrage
# AMÉLIORATION UI: Fonction de chargement avec indicateur de progression
with st.spinner("Chargement des données de l'application..."):
    app_data = load_app_data()
    if st.session_state.show_loading_welcome:
        # st.balloons()
        st.session_state.show_loading_welcome = False

# FONCTION PRINCIPALE

def main():
    # AMÉLIORATION UI: Titre principal avec effet shadow et animation
    st.markdown('<div class="main-title section-fade-in">🥊 Prédicteur de Combats UFC 🥊</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Analysez et prédisez l\'issue des affrontements avec intelligence</div>', unsafe_allow_html=True)
    
    # AMÉLIORATION UI: Tabs modernisés avec icônes
    tabs = st.tabs([
        "🏠 Accueil", 
        "🎯 Prédiction", 
        "🗓️ Événements à venir", 
        "💰 Gestion de Bankroll", 
        "📊 Historique & Performance"
    ])
    
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

# PARTIE 5 

def show_welcome_page():
    """Affiche la page d'accueil avec un design moderne et attrayant"""
    
    # AMÉLIORATION UI: En-tête de bienvenue animé et moderne
    st.markdown("""
    <div class="welcome-header section-fade-in">
        <h1 class="welcome-title">🥊 UFC Fight Predictor 🥊</h1>
        <p class="welcome-subtitle">Prédisez les résultats des combats UFC avec intelligence artificielle</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Introduction avec un style amélioré
    st.write("""
    L'UFC Fight Predictor est un outil avancé qui combine l'analyse statistique et le machine learning 
    pour prédire les résultats des combats de l'UFC. Que vous soyez un fan passionné cherchant 
    à anticiper les résultats ou un parieur à la recherche d'un avantage analytique, cette application 
    vous fournit des prédictions détaillées basées sur l'historique et les statistiques des combattants.
    """)
    
    # AMÉLIORATION UI: Fonctionnalités principales avec des cards animées
    st.markdown("## Principales fonctionnalités")
    
    # Afficher les fonctionnalités dans une mise en page à trois colonnes modernisée
    cols = st.columns(3)
    
    with cols[0]:
        st.markdown("""
        <div class="home-card section-fade-in">
            <div class="feature-icon">🎯</div>
            <h3 class="feature-title">Prédictions précises</h3>
            <p class="feature-description">Obtenez des prédictions basées sur deux méthodes complémentaires: analyse statistique classique et modèle de machine learning avancé.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown("""
        <div class="home-card section-fade-in">
            <div class="feature-icon">📊</div>
            <h3 class="feature-title">Analyses détaillées</h3>
            <p class="feature-description">Visualisez les forces et faiblesses de chaque combattant avec des graphiques comparatifs et des statistiques pertinentes.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[2]:
        st.markdown("""
        <div class="home-card section-fade-in">
            <div class="feature-icon">💰</div>
            <h3 class="feature-title">Conseils de paris</h3>
            <p class="feature-description">Recevez des recommandations de paris basées sur l'analyse des cotes et la gestion optimale de votre bankroll avec la méthode Kelly.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # AMÉLIORATION UI: Instructions d'utilisation modernisées
    st.markdown("## Comment utiliser l'application")
    
    # AMÉLIORATION UI: Utiliser des colonnes pour une meilleure organisation
    how_to_cols = st.columns(2)
    
    with how_to_cols[0]:
        st.markdown("""
        ### Étapes essentielles
        
        1. **Onglet Prédiction**: Sélectionnez deux combattants pour obtenir une analyse complète et une prédiction du résultat du combat
        
        2. **Onglet Événements à venir**: Consultez les prochains combats UFC avec des prédictions automatiques
        
        3. **Onglet Gestion de Bankroll**: Suivez vos paris et gérez votre bankroll pour optimiser vos gains
        
        4. **Onglet Historique & Performance**: Analysez vos performances de paris passés
        """)
        
    with how_to_cols[1]:
        # AMÉLIORATION UI: Statut du modèle ML
        ml_available = app_data["ml_model"] is not None
        
        if ml_available:
            st.markdown("""
            <div style="background: linear-gradient(145deg, rgba(76, 175, 80, 0.1) 0%, rgba(56, 142, 60, 0.1) 100%); 
                        padding: 20px; border-radius: 12px; margin: 15px 0; border-left: 3px solid #4CAF50;">
                <h3 style="color: #4CAF50; margin-top: 0;">✅ Modèle ML opérationnel</h3>
                <p>Le modèle de machine learning a été correctement chargé et est prêt à être utilisé pour des prédictions de haute précision.</p>
                <p><i>Les prédictions ML sont généralement plus précises que les prédictions statistiques classiques.</i></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: linear-gradient(145deg, rgba(244, 67, 54, 0.1) 0%, rgba(211, 47, 47, 0.1) 100%); 
                        padding: 20px; border-radius: 12px; margin: 15px 0; border-left: 3px solid #F44336;">
                <h3 style="color: #F44336; margin-top: 0;">⚠️ Modèle ML non détecté</h3>
                <p>Le modèle de machine learning n'a pas été trouvé. L'application fonctionnera avec la méthode de prédiction statistique uniquement.</p>
                <p>Pour activer les prédictions par machine learning, assurez-vous que <code>ufc_prediction_model.joblib</code> ou <code>ufc_prediction_model.pkl</code> est présent dans le répertoire de l'application.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # AMÉLIORATION UI: Disclaimer avec un style moderne
    st.markdown("""
    <div style="background: linear-gradient(145deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 160, 0, 0.1) 100%);
                padding: 20px; border-radius: 12px; margin-top: 40px; border-left: 3px solid #FFC107;">
        <h3 style="color: #FFC107; margin-top: 0;">⚠️ Avertissement</h3>
        <p>Les prédictions fournies par cette application sont basées sur des modèles statistiques et d'apprentissage automatique, mais ne garantissent pas les résultats des combats. Les paris sportifs comportent des risques, et cette application ne doit être utilisée qu'à titre informatif. Pariez de manière responsable.</p>
    </div>
    """, unsafe_allow_html=True)

    # AMÉLIORATION UI: Section des dernières mises à jour
    st.markdown("""
    <div class="divider"></div>
    <h3 style="margin-top: 30px;">🆕 Dernières mises à jour</h3>
    """, unsafe_allow_html=True)
    
    updates = [
        {"date": "Mai 2025", "title": "Interface utilisateur repensée", "desc": "Design moderne, navigation améliorée et meilleure visualisation des données"},
        {"date": "Avril 2025", "title": "Stratégie de paris Kelly optimisée", "desc": "Recommandations de paris intelligentes basées sur la méthode Kelly"},
        {"date": "Mars 2025", "title": "Suivi des performances", "desc": "Nouvelles métriques et graphiques pour analyser vos résultats de paris"}
    ]
    
    for update in updates:
        st.markdown(f"""
        <div style="display: flex; margin-bottom: 15px; align-items: flex-start;">
            <div style="min-width: 100px; font-weight: 500; color: #888;">{update['date']}</div>
            <div>
                <div style="font-weight: 600; margin-bottom: 5px;">{update['title']}</div>
                <div style="color: #aaa; font-size: 0.9rem;">{update['desc']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# PARTIE 6

def show_prediction_page():
    """Interface de prédiction améliorée avec une meilleure organisation"""
    # Section titre 
    st.title("🎯 Prédicteur de Combat")
    st.write("Sélectionnez deux combattants et obtenez des prédictions précises")
    
    # Layout à deux colonnes
    main_cols = st.columns([1, 3])
    
    with main_cols[0]:
        # Sélection des combattants
        st.subheader("Sélection des combattants")
        
        # Message d'avertissement sur l'importance de l'ordre des combattants
        st.warning("⚠️ Important : L'ordre des combattants (Rouge/Bleu) influence les prédictions. Traditionnellement, le combattant mieux classé ou favori est placé dans le coin rouge.")
        
        # Sélection du combattant rouge
        st.subheader("🔴 Combattant Rouge")
        fighter_a_name = st.selectbox(
            "Sélectionner combattant rouge",
            options=app_data["fighter_names"],
            key="fighter_a_selectbox"
        )
        
        # Sélection du combattant bleu (en excluant le combattant rouge)
        st.subheader("🔵 Combattant Bleu")
        fighter_b_options = [name for name in app_data["fighter_names"] if name != fighter_a_name]
        fighter_b_name = st.selectbox(
            "Sélectionner combattant bleu",
            options=fighter_b_options,
            key="fighter_b_selectbox"
        )
        
        # Options de paris
        st.subheader("Options de paris")

        # Mode de saisie des cotes (manuel ou slider)
        cote_input_mode = st.radio(
            "Mode de saisie des cotes",
            options=["Manuel", "Slider"],
            index=0,  # Manuel par défaut
            key="cote_input_mode"
        )
        
        if cote_input_mode == "Manuel":
            odds_a = st.number_input("Cote Rouge", min_value=1.01, value=2.0, step=0.01, format="%.2f", key="odds_a_input_manual")
            odds_b = st.number_input("Cote Bleu", min_value=1.01, value=1.8, step=0.01, format="%.2f", key="odds_b_input_manual")
        else:
            odds_a = st.slider("Cote Rouge", min_value=1.01, max_value=10.0, value=2.0, step=0.05, format="%.2f", key="odds_a_input_slider")
            odds_b = st.slider("Cote Bleu", min_value=1.01, max_value=10.0, value=1.8, step=0.05, format="%.2f", key="odds_b_input_slider")
        
        # Stratégie Kelly
        st.subheader("📈 Critères Kelly")
        kelly_strategy = st.selectbox(
            "Stratégie Kelly",
            options=["Kelly pur", "Kelly/2", "Kelly/3", "Kelly/4", "Kelly/5", "Kelly/10"],
            index=3,  # Kelly/4 par défaut
            key="kelly_strategy_select"
        )
        st.session_state.kelly_strategy = kelly_strategy
        
        # Bankroll actuelle
        st.subheader("💼 Bankroll actuelle")
        st.metric(
            "",
            f"{app_data['current_bankroll']:.2f} €", 
            delta=None
        )
        
        # Bouton de prédiction
        predict_btn = st.button(
            "🔮 Prédire le combat", 
            type="primary", 
            key="predict_btn", 
            use_container_width=True
        )
    
    with main_cols[1]:
        # Récupérer les statistiques des combattants sélectionnés
        fighter_a = app_data["fighters_dict"].get(fighter_a_name)
        fighter_b = app_data["fighters_dict"].get(fighter_b_name)
        
        # Vérifier si on peut faire une prédiction
        if predict_btn and fighter_a and fighter_b:
            if fighter_a_name == fighter_b_name:
                st.error("Veuillez sélectionner deux combattants différents.")
            else:
                # Afficher un spinner pendant le calcul
                with st.spinner("Analyse en cours..."):
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
            
            # Afficher une vue en tête-à-tête des combattants
            st.subheader("Combat")
            col1, col2, col3 = st.columns([2, 1, 2])
            with col1:
                st.write(f"### 🔴 {fighter_a['name']}")
                st.write(f"Record: {fighter_a['wins']}-{fighter_a['losses']}")
            with col2:
                st.write("## VS")
            with col3:
                st.write(f"### 🔵 {fighter_b['name']}")
                st.write(f"Record: {fighter_b['wins']}-{fighter_b['losses']}")
            
            # Afficher les résultats des deux prédictions
            st.subheader("🔮 Prédictions du combat")
            
            # Créer le graphique comparatif des probabilités pour les deux méthodes en un seul
            if ml_prediction:
                # Créer un DataFrame pour le graphique comparatif
                proba_data = pd.DataFrame({
                    'Combattant': [fighter_a['name'], fighter_b['name']],
                    'Statistique': [classic_prediction['red_probability'], classic_prediction['blue_probability']],
                    'Machine Learning': [ml_prediction['red_probability'], ml_prediction['blue_probability']]
                })
                
                # Graphique modernisé
                fig = go.Figure()
                
                # Ajouter les barres pour chaque méthode avec un style amélioré
                fig.add_trace(go.Bar(
                    x=proba_data['Combattant'],
                    y=proba_data['Statistique'],
                    name='Prédiction Statistique',
                    marker_color='#2196F3',
                    text=[f"{proba:.0%}" for proba in proba_data['Statistique']],
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>Probabilité: %{y:.1%}<extra></extra>'
                ))
                
                fig.add_trace(go.Bar(
                    x=proba_data['Combattant'],
                    y=proba_data['Machine Learning'],
                    name='Prédiction ML',
                    marker_color='#4CAF50',
                    text=[f"{proba:.0%}" for proba in proba_data['Machine Learning']],
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>Probabilité: %{y:.1%}<extra></extra>'
                ))
                
                # Configurer la mise en page - FIXED: Remove the undefined legend entry
                fig.update_layout(
                    xaxis_title=None,
                    yaxis_title="Probabilité de victoire",
                    yaxis=dict(
                        range=[0, 1],
                        tickformat='.0%',
                        gridcolor='rgba(255,255,255,0.1)',
                        showgrid=True
                    ),
                    xaxis=dict(
                        gridcolor='rgba(255,255,255,0.1)',
                    ),
                    legend_title=None,
                    height=400,
                    barmode='group',
                    bargap=0.30,
                    bargroupgap=0.1,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5
                    ),
                    margin=dict(t=50, b=50),
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Si seulement la méthode statistique est disponible
                # Graphique modernisé pour la méthode unique
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[fighter_a['name'], fighter_b['name']],
                    y=[classic_prediction['red_probability'], classic_prediction['blue_probability']],
                    marker_color=['#E53935', '#1E88E5'],
                    text=[f"{classic_prediction['red_probability']:.0%}", f"{classic_prediction['blue_probability']:.0%}"],
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>Probabilité: %{y:.1%}<extra></extra>',
                    showlegend=False
                ))
                
                fig.update_layout(
                    xaxis_title=None,
                    yaxis_title="Probabilité de victoire",
                    yaxis=dict(
                        range=[0, 1],
                        tickformat='.0%',
                        gridcolor='rgba(255,255,255,0.1)',
                        showgrid=True
                    ),
                    xaxis=dict(
                        gridcolor='rgba(255,255,255,0.1)',
                    ),
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    margin=dict(t=50, b=50)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # NOUVELLE SECTION: Affichage amélioré des prédictions
            st.subheader("📊 Résultats des prédictions")
            
            # Utiliser un layout alternatif pour éviter le nidification de colonnes
            pred_cols = st.columns(2 if ml_prediction else 1)
            
            # Prédiction Statistique (première colonne)
            with pred_cols[0]:
                st.subheader("Prédiction Statistique")
                
                # Vainqueur prédit avec mise en évidence améliorée
                winner_name = classic_prediction['winner_name']
                is_red_winner = classic_prediction['prediction'] == 'Red'
                
                # AMÉLIORATION: Vainqueur prédit en TRÈS GRAND
                winner_color = "#E53935" if is_red_winner else "#1E88E5"
                st.markdown("### Vainqueur prédit:")
                
                # Utiliser Markdown pour afficher le vainqueur en très grand et en couleur
                st.markdown(f"<h1 style='color: {winner_color}; font-size: 36px; text-align: center;'>{'🔴' if is_red_winner else '🔵'} {winner_name}</h1>", unsafe_allow_html=True)
                
                # Les probabilités pour chaque combattant
                red_prob = classic_prediction['red_probability']
                blue_prob = classic_prediction['blue_probability']
                
                # Afficher les métriques avec les valeurs originales
                st.metric(f"🔴 {fighter_a['name']}", f"{red_prob:.0%}")
                st.metric(f"🔵 {fighter_b['name']}", f"{blue_prob:.0%}")
                
                # AMÉLIORATION: Toujours afficher une barre de progression significative
                if red_prob < 0 or red_prob > 1 or blue_prob < 0 or blue_prob > 1:
                    # Message expliquant l'adaptation mais plus concis
                    st.caption(f"Note: Probabilités originales: {red_prob:.0%} vs {blue_prob:.0%}")
                    
                    # Adaptation de la visualisation pour probabilités extrêmes
                    if is_red_winner:
                        # Vainqueur rouge - montrer une barre à 80% (dominance claire mais pas totale)
                        progress_value = 0.8
                    else:
                        # Vainqueur bleu - montrer une barre à 20% (dominance claire du bleu)
                        progress_value = 0.2
                else:
                    # Probabilités normales, utilisation directe
                    progress_value = red_prob
                
                # Toujours afficher la barre de progression
                # Toujours afficher la barre de progression bicolore
                progress_html = create_probability_bar(
                    red_prob, 
                    blue_prob, 
                    fighter_a['name'], 
                    fighter_b['name']
                )
                st.markdown(progress_html, unsafe_allow_html=True)
                
                # Afficher la confiance
                confidence = classic_prediction['confidence']
                if confidence == "Élevé":
                    st.success(f"Confiance: {confidence}")
                else:
                    st.warning(f"Confiance: {confidence}")
            
            # Prédiction ML (deuxième colonne, si disponible)
            if ml_prediction:
                with pred_cols[1]:
                    st.subheader("Prédiction Machine Learning")
                    
                    # Vainqueur prédit avec mise en évidence améliorée
                    winner_name_ml = ml_prediction['winner_name']
                    is_red_winner_ml = ml_prediction['prediction'] == 'Red'
                    
                    # AMÉLIORATION: Vainqueur prédit en TRÈS GRAND
                    winner_color_ml = "#E53935" if is_red_winner_ml else "#1E88E5"
                    st.markdown("### Vainqueur prédit:")
                    
                    # Utiliser Markdown pour afficher le vainqueur en très grand et en couleur
                    st.markdown(f"<h1 style='color: {winner_color_ml}; font-size: 36px; text-align: center;'>{'🔴' if is_red_winner_ml else '🔵'} {winner_name_ml}</h1>", unsafe_allow_html=True)
                    
                    # Les probabilités pour chaque combattant
                    red_prob_ml = ml_prediction['red_probability']
                    blue_prob_ml = ml_prediction['blue_probability']
                    
                    # Afficher les métriques avec les valeurs originales
                    st.metric(f"🔴 {fighter_a['name']}", f"{red_prob_ml:.0%}")
                    st.metric(f"🔵 {fighter_b['name']}", f"{blue_prob_ml:.0%}")
                    
                    # AMÉLIORATION: Toujours afficher une barre de progression significative
                    if red_prob_ml < 0 or red_prob_ml > 1 or blue_prob_ml < 0 or blue_prob_ml > 1:
                        # Message expliquant l'adaptation mais plus concis
                        st.caption(f"Note: Probabilités originales: {red_prob_ml:.0%} vs {blue_prob_ml:.0%}")
                        
                        # Adaptation de la visualisation pour probabilités extrêmes
                        if is_red_winner_ml:
                            # Vainqueur rouge - montrer une barre à 80% (dominance claire mais pas totale)
                            progress_value_ml = 0.8
                        else:
                            # Vainqueur bleu - montrer une barre à 20% (dominance claire du bleu)
                            progress_value_ml = 0.2
                    else:
                        # Probabilités normales, utilisation directe
                        progress_value_ml = red_prob_ml
                    
                    # Toujours afficher la barre de progression
                    # Toujours afficher la barre de progression bicolore
                    progress_html_ml = create_probability_bar(
                        red_prob_ml, 
                        blue_prob_ml, 
                        fighter_a['name'], 
                        fighter_b['name']
                    )
                    st.markdown(progress_html_ml, unsafe_allow_html=True)

                    
                    # Afficher la confiance
                    confidence_ml = ml_prediction['confidence']
                    if confidence_ml == "Élevé":
                        st.success(f"Confiance: {confidence_ml}")
                    else:
                        st.warning(f"Confiance: {confidence_ml}")
            
            # Message de convergence/divergence si les deux méthodes sont disponibles
            if ml_prediction:
                same_prediction = classic_prediction['prediction'] == ml_prediction['prediction']
                if same_prediction:
                    st.success("✅ Les deux méthodes prédisent le même vainqueur!")
                else:
                    st.warning("⚠️ Les méthodes prédisent des vainqueurs différents!")
                
            # PARTIE 7: Analyse Kelly et recommandations de paris
            if ml_prediction:
                st.divider()
                st.subheader("📊 Analyse Kelly et recommandations de paris")
                
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
                
                # Ajustement pour éviter les erreurs si probabilité hors limites
                best_prob = max(0.01, min(0.99, best_prob))
                
                # Calculer les recommandations Kelly pour le combattant favori selon le ML
                kelly_amount = calculate_kelly(best_prob, best_odds, app_data["current_bankroll"], selected_fraction)
                
                # Section Kelly modernisée avec composants Streamlit natifs
                st.write("### Recommandation de mise avec la méthode " + st.session_state.kelly_strategy)
                st.write("Pour maximiser votre ROI sur le long terme, la méthode Kelly recommande:")
                
                # Créer un DataFrame au lieu d'une table HTML
                kelly_data = pd.DataFrame({
                    "Combattant": [best_fighter],
                    "Probabilité ML": [f"{best_prob:.0%}"],
                    "Cote": [f"{best_odds:.2f}"],
                    "Mise recommandée": [f"{kelly_amount:.2f} €"],
                    "% de bankroll": [f"{(kelly_amount/app_data['current_bankroll']*100):.1f}%"],
                    "Gain potentiel": [f"{kelly_amount * (best_odds-1):.2f} €"]
                })
                
                # Afficher le DataFrame avec style
                st.dataframe(kelly_data, use_container_width=True, hide_index=True)
                
                st.caption("Le critère de Kelly détermine la mise optimale en fonction de votre avantage et de votre bankroll totale.")
                
                # Section pour placer un pari modernisée
                st.subheader(f"Placer un pari sur {best_fighter}")
                
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
                
                # Afficher les détails du pari avec un design attractif
                pot_gain = bet_amount * (best_odds-1)
                roi_pct = (pot_gain / bet_amount) * 100 if bet_amount > 0 else 0
                
                # Créer 3 colonnes pour les métriques
                bet_metrics_cols = st.columns(3)
                with bet_metrics_cols[0]:
                    st.metric("Mise", f"{bet_amount:.2f}€")
                with bet_metrics_cols[1]:
                    st.metric("Gain potentiel", f"{pot_gain:.2f}€")
                with bet_metrics_cols[2]:
                    st.metric("ROI", f"{roi_pct:.1f}%")
                
                # Bouton pour placer le pari
                if st.button("💰 Placer ce pari", type="primary", key="place_bet_btn", use_container_width=True):
                    if bet_amount > app_data["current_bankroll"]:
                        st.error(f"Montant du pari ({bet_amount:.2f}€) supérieur à votre bankroll actuelle ({app_data['current_bankroll']:.2f}€)")
                    elif bet_amount <= 0:
                        st.error("Le montant du pari doit être supérieur à 0€")
                    else:
                        # Animation de chargement
                        with st.spinner("Enregistrement du pari..."):
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
                                # Message de succès avec détails
                                st.success(f"Pari enregistré avec succès! {bet_amount:.2f}€ sur {best_fighter} @ {best_odds:.2f}")
                                
                                # Ajouter un petit délai pour l'animation
                                time.sleep(0.5)
                                
                                # Afficher une confirmation
                                st.info(f"Vous avez parié {bet_amount:.2f}€ sur {best_fighter}. Gain potentiel: {pot_gain:.2f}€ (ROI: {roi_pct:.1f}%)")
                                st.write("Vous pouvez suivre ce pari dans l'onglet 'Gestion de Bankroll'")
                            else:
                                st.error("Erreur lors de l'enregistrement du pari.")
            
            # Analyse des paris (utiliser les deux méthodes si disponibles)
            if 'betting' in classic_prediction:
                st.divider()
                st.subheader("💰 Analyse des paris")
                st.write("Comparaison des cotes du marché avec nos probabilités prédites")
                
                # Analyse des paris pour les deux combattants avec un design modernisé
                col1, col2 = st.columns(2)
                
                # Combattant Rouge
                with col1:
                    st.write(f"### 🔴 {fighter_a['name']}")
                    
                    # Données de paris
                    betting_classic = classic_prediction['betting']
                    betting_ml = ml_prediction.get('betting') if ml_prediction else None
                    
                    # Créer une table pour les données du combattant rouge
                    st.write("**Données de paris:**")
                    red_data = [
                        ["Cote du marché", f"{betting_classic['odds_red']:.2f}"],
                        ["Probabilité implicite", f"{betting_classic['implied_prob_red']:.0%}"],
                        ["Probabilité statistique", f"{classic_prediction['red_probability']:.0%}"]
                    ]
                    
                    if betting_ml:
                        red_data.append(["Probabilité ML", f"{ml_prediction['red_probability']:.0%}"])
                    
                    red_data.extend([
                        ["Avantage statistique", f"{betting_classic['edge_red']*100:.1f}%"],
                        ["Valeur espérée", f"{betting_classic['ev_red']*100:.1f}%"]
                    ])
                    
                    # Afficher les données sous forme de tableau
                    red_df = pd.DataFrame(red_data, columns=["Métrique", "Valeur"])
                    st.dataframe(red_df, hide_index=True, use_container_width=True)
                    
                    # Afficher les recommandations avec des composants Streamlit natifs
                    st.write("**Recommandation statistique:**")
                    if betting_classic['recommendation_red'] == "Favorable":
                        st.success("Favorable")
                    elif betting_classic['recommendation_red'] == "Neutre":
                        st.info("Neutre")
                    else:
                        st.error("Défavorable")
                    
                    if betting_ml:
                        st.write("**Recommandation ML:**")
                        if betting_ml['recommendation_red'] == "Favorable":
                            st.success("Favorable")
                        elif betting_ml['recommendation_red'] == "Neutre":
                            st.info("Neutre")
                        else:
                            st.error("Défavorable")
                    
                    # Bouton pour parier sur le combattant rouge
                    if st.button(f"Parier sur {fighter_a['name']}", key="bet_on_red_btn", use_container_width=True):
                        # Calculer le montant Kelly pour ce combattant
                        red_prob_for_kelly = max(0.01, min(0.99, ml_prediction['red_probability'] if ml_prediction else classic_prediction['red_probability']))
                        
                        red_kelly = calculate_kelly(
                            red_prob_for_kelly,
                            odds_a,
                            app_data["current_bankroll"],
                            kelly_fractions[st.session_state.kelly_strategy]
                        )
                        
                        # Stocker dans la session pour précharger le formulaire
                        st.session_state.temp_bet = {
                            "fighter": fighter_a['name'],
                            "odds": odds_a,
                            "kelly_amount": red_kelly,
                            "probability": red_prob_for_kelly
                        }
                        
                        # Afficher le formulaire pour parier
                        show_bet_form(
                            fighter_a['name'], 
                            fighter_b['name'], 
                            fighter_a['name'], 
                            odds_a, 
                            red_kelly,
                            red_prob_for_kelly,
                            kelly_fractions[st.session_state.kelly_strategy]
                        )
                
                # Combattant Bleu
                with col2:
                    st.write(f"### 🔵 {fighter_b['name']}")
                    
                    # Créer une table pour les données du combattant bleu
                    st.write("**Données de paris:**")
                    blue_data = [
                        ["Cote du marché", f"{betting_classic['odds_blue']:.2f}"],
                        ["Probabilité implicite", f"{betting_classic['implied_prob_blue']:.0%}"],
                        ["Probabilité statistique", f"{classic_prediction['blue_probability']:.0%}"]
                    ]
                    
                    if betting_ml:
                        blue_data.append(["Probabilité ML", f"{ml_prediction['blue_probability']:.0%}"])
                    
                    blue_data.extend([
                        ["Avantage statistique", f"{betting_classic['edge_blue']*100:.1f}%"],
                        ["Valeur espérée", f"{betting_classic['ev_blue']*100:.1f}%"]
                    ])
                    
                    # Afficher les données sous forme de tableau
                    blue_df = pd.DataFrame(blue_data, columns=["Métrique", "Valeur"])
                    st.dataframe(blue_df, hide_index=True, use_container_width=True)
                    
                    # Afficher les recommandations avec des composants Streamlit natifs
                    st.write("**Recommandation statistique:**")
                    if betting_classic['recommendation_blue'] == "Favorable":
                        st.success("Favorable")
                    elif betting_classic['recommendation_blue'] == "Neutre":
                        st.info("Neutre")
                    else:
                        st.error("Défavorable")
                    
                    if betting_ml:
                        st.write("**Recommandation ML:**")
                        if betting_ml['recommendation_blue'] == "Favorable":
                            st.success("Favorable")
                        elif betting_ml['recommendation_blue'] == "Neutre":
                            st.info("Neutre")
                        else:
                            st.error("Défavorable")
                    
                    # Bouton pour parier sur le combattant bleu
                    if st.button(f"Parier sur {fighter_b['name']}", key="bet_on_blue_btn", use_container_width=True):
                        # Calculer le montant Kelly pour ce combattant
                        blue_prob_for_kelly = max(0.01, min(0.99, ml_prediction['blue_probability'] if ml_prediction else classic_prediction['blue_probability']))
                        
                        blue_kelly = calculate_kelly(
                            blue_prob_for_kelly,
                            odds_b,
                            app_data["current_bankroll"],
                            kelly_fractions[st.session_state.kelly_strategy]
                        )
                        
                        # Stocker dans la session pour précharger le formulaire
                        st.session_state.temp_bet = {
                            "fighter": fighter_b['name'],
                            "odds": odds_b,
                            "kelly_amount": blue_kelly,
                            "probability": blue_prob_for_kelly
                        }
                        
                        # Afficher le formulaire pour parier
                        show_bet_form(
                            fighter_a['name'], 
                            fighter_b['name'], 
                            fighter_b['name'], 
                            odds_b, 
                            blue_kelly,
                            blue_prob_for_kelly,
                            kelly_fractions[st.session_state.kelly_strategy]
                        )
                        
            # PARTIE 8: Nouvel onglet avec les statistiques et graphiques
            stats_tabs = st.tabs(["🔍 Statistiques", "📊 Graphiques", "📝 Notes"])
            
            # Onglet des statistiques
            with stats_tabs[0]:
                # Afficher les statistiques comparatives
                st.subheader("📊 Statistiques comparatives")
                
                # Création du DataFrame des statistiques comparatives
                stats_df = create_stats_comparison_df(fighter_a, fighter_b)
                
                # Appliquer un style conditionnel pour mettre en évidence les avantages
                def highlight_advantage(row):
                    styles = [''] * len(row)
                    advantage = row['Avantage']
                    
                    if advantage == fighter_a['name']:
                        styles[1] = 'background-color: rgba(229, 57, 53, 0.2); font-weight: bold;'
                    elif advantage == fighter_b['name']:
                        styles[2] = 'background-color: rgba(30, 136, 229, 0.2); font-weight: bold;'
                    
                    return styles
                
                # Appliquer le style et afficher avec un design plus moderne
                styled_df = stats_df.style.apply(highlight_advantage, axis=1)
                st.dataframe(
                    styled_df, 
                    use_container_width=True, 
                    height=500,
                    hide_index=True,
                )
            
            # Onglet des visualisations
            with stats_tabs[1]:
                st.subheader("📈 Visualisations des performances")
                
                # Disposer les graphiques en deux colonnes
                col1, col2 = st.columns(2)
                
                with col1:
                    # Graphique radar
                    radar_fig = create_radar_chart(fighter_a, fighter_b)
                    st.plotly_chart(radar_fig, use_container_width=True, height=400)
                
                with col2:
                    # Graphique des forces et faiblesses
                    strengths_fig = create_strengths_weaknesses_chart(fighter_a, fighter_b)
                    st.plotly_chart(strengths_fig, use_container_width=True, height=400)
                
                # Style de combat
                style_fig = create_style_analysis_chart(fighter_a, fighter_b)
                st.plotly_chart(style_fig, use_container_width=True)
            
            # Onglet des notes
            with stats_tabs[2]:
                st.subheader("📝 Notes d'analyse")
                
                # Analyse textuelle générée
                # Déterminer les styles de combat
                a_striking = fighter_a['SLpM'] * fighter_a['sig_str_acc']
                a_ground = fighter_a['td_avg'] * fighter_a['td_acc'] + fighter_a['sub_avg']
                a_style = "striker" if a_striking > a_ground * 1.5 else "grappler" if a_ground > a_striking * 1.5 else "équilibré"
                
                b_striking = fighter_b['SLpM'] * fighter_b['sig_str_acc']
                b_ground = fighter_b['td_avg'] * fighter_b['td_acc'] + fighter_b['sub_avg']
                b_style = "striker" if b_striking > b_ground * 1.5 else "grappler" if b_ground > b_striking * 1.5 else "équilibré"
                
                # Expérience
                a_exp = fighter_a['wins'] + fighter_a['losses']
                b_exp = fighter_b['wins'] + fighter_b['losses']
                exp_diff = abs(a_exp - b_exp)
                exp_advantage = f"{fighter_a['name']} a {exp_diff} combats de plus" if a_exp > b_exp else f"{fighter_b['name']} a {exp_diff} combats de plus" if b_exp > a_exp else "Les deux combattants ont le même niveau d'expérience"
                
                # Forme récente (à calculer à partir du record)
                a_winrate = fighter_a['wins'] / max(a_exp, 1)
                b_winrate = fighter_b['wins'] / max(b_exp, 1)
                
                # Stats physiques
                height_diff = abs(fighter_a['height'] - fighter_b['height'])
                reach_diff = abs(fighter_a['reach'] - fighter_b['reach'])
                
                physical_advantage = ""
                if fighter_a['height'] > fighter_b['height'] and fighter_a['reach'] > fighter_b['reach']:
                    physical_advantage = f"{fighter_a['name']} a un avantage de taille ({height_diff:.1f} cm) et d'allonge ({reach_diff:.1f} cm)"
                elif fighter_b['height'] > fighter_a['height'] and fighter_b['reach'] > fighter_a['reach']:
                    physical_advantage = f"{fighter_b['name']} a un avantage de taille ({height_diff:.1f} cm) et d'allonge ({reach_diff:.1f} cm)"
                else:
                    physical_advantage = "Les avantages physiques sont partagés entre les deux combattants"
                
                # Profil des combattants
                st.write("#### Profil des combattants")
                st.write(f"**{fighter_a['name']}** est un combattant de style **{a_style}** avec un taux de victoires de **{a_winrate:.0%}** sur {a_exp} combats.")
                st.write(f"**{fighter_b['name']}** est un combattant de style **{b_style}** avec un taux de victoires de **{b_winrate:.0%}** sur {b_exp} combats.")
                
                # Facteurs clés
                st.write("#### Facteurs clés du combat")
                st.write(f"* **Expérience:** {exp_advantage}.")
                st.write(f"* **Avantage physique:** {physical_advantage}.")
                st.write(f"* **Dynamique du combat:** {fighter_a['name']} donne {fighter_a['SLpM']:.1f} coups par minute contre {fighter_b['SLpM']:.1f} pour {fighter_b['name']}.")
                st.write(f"* **Facteur sol:** {fighter_a['name']} tente {fighter_a['td_avg']:.1f} takedowns par combat contre {fighter_b['td_avg']:.1f} pour {fighter_b['name']}.")
                
                # Points à surveiller
                st.write("#### Points à surveiller")
                st.write(f"Ce combat présente un affrontement de styles {a_style if a_style != b_style else 'similaires'}, où {fighter_a['name'] if a_winrate > b_winrate else fighter_b['name']} a l'avantage en termes d'historique de victoires.")
                
                if a_style != b_style:
                    st.write(f"Le vainqueur sera probablement celui qui pourra imposer sa stratégie préférée: {fighter_a['name']} voudra maintenir le combat {a_style}, tandis que {fighter_b['name']} cherchera à l'amener vers une dynamique {b_style}.")
                else:
                    st.write("Les deux combattants auront des approches similaires, donc la technique et les adaptations en cours de combat seront déterminantes.")
        else:
            # Message d'accueil
            st.info("Bienvenue sur le Prédicteur de Combats UFC! Sélectionnez deux combattants et cliquez sur 'Prédire le combat' pour obtenir une analyse complète.")
            
            # Message d'information
            st.warning("⚠️ L'ordre des combattants est important! La position des combattants (coin Rouge vs Bleu) peut influencer significativement les prédictions. Traditionnellement, le combattant favori ou mieux classé est placé dans le coin rouge.")

def show_bet_form(fighter_red, fighter_blue, pick, odds, kelly_amount, probability, kelly_fraction):
    """Affiche un formulaire modernisé pour placer un pari"""
    # AMÉLIORATION UI: Box de placement de paris améliorée
    st.markdown(f"""
    <div class="bet-placement-box section-fade-in">
        <h3 class="bet-placement-title">Placer un pari sur {pick}</h3>
        <p>Complétez les informations ci-dessous pour enregistrer votre pari</p>
    </div>
    """, unsafe_allow_html=True)
    
    # AMÉLIORATION UI: Organisation en deux colonnes
    bet_cols = st.columns(2)
    
    with bet_cols[0]:
        # Nom de l'événement
        event_name = st.text_input("Nom de l'événement", value="UFC Fight Night", key="bet_event_name")
        
        # Date de l'événement
        event_date = st.date_input("Date de l'événement", value=datetime.datetime.now(), key="bet_event_date")
    
    with bet_cols[1]:
        # Mode de saisie pour le montant à miser
        bet_input_mode = st.radio(
            "Mode de saisie du montant",
            options=["Manuel", "Slider"],
            index=0,  # Manuel par défaut
            key="bet_input_mode"
        )
        
        if bet_input_mode == "Manuel":
            # Montant à miser - saisie manuelle
            bet_amount = st.number_input(
                "Montant à miser (€)",
                min_value=0.0,
                max_value=float(app_data["current_bankroll"]),
                value=float(kelly_amount),
                step=5.0,
                format="%.2f",
                key="place_bet_amount_manual"
            )
        else:
            # Montant à miser avec slider pour une meilleure UX
            bet_amount = st.slider(
                "Montant à miser (€)",
                min_value=0.0,
                max_value=float(app_data["current_bankroll"]),
                value=float(kelly_amount),
                step=5.0,
                format="%.2f",
                key="place_bet_amount_slider"
            )
        
        # Utiliser la mise Kelly recommandée
        use_kelly = st.checkbox("Utiliser la mise Kelly recommandée", value=True, key="place_use_kelly")
        if use_kelly:
            bet_amount = kelly_amount
    
    # AMÉLIORATION UI: Détails du pari avec des métriques visuelles
    pot_gain = bet_amount * (odds-1)
    roi_pct = (pot_gain / bet_amount) * 100 if bet_amount > 0 else 0
    
    # Afficher les métriques en 3 colonnes
    metrics_cols = st.columns(3)
    with metrics_cols[0]:
        st.metric("Mise", f"{bet_amount:.2f}€")
    with metrics_cols[1]:
        st.metric("Gain potentiel", f"{pot_gain:.2f}€")
    with metrics_cols[2]:
        st.metric("ROI", f"{roi_pct:.1f}%")
    
    # AMÉLIORATION UI: Résumé du pari dans un card
    st.markdown(f"""
    <div class="card" style="margin: 15px 0; background: linear-gradient(145deg, rgba(13, 110, 253, 0.05) 0%, rgba(13, 110, 253, 0.1) 100%); border-left: 3px solid #0d6efd;">
        <div style="text-align: center; margin-bottom: 10px;">
            <h4 style="margin: 0; color: #0d6efd;">Résumé du pari</h4>
        </div>
        <div style="display: flex; justify-content: space-between; margin: 10px 0;">
            <div>Combat</div>
            <div style="font-weight: 600;">{fighter_red} vs {fighter_blue}</div>
        </div>
        <div style="display: flex; justify-content: space-between; margin: 10px 0;">
            <div>Pari sur</div>
            <div style="font-weight: 600; color: {('#E53935' if pick == fighter_red else '#1E88E5')};">{pick}</div>
        </div>
        <div style="display: flex; justify-content: space-between; margin: 10px 0;">
            <div>Cote</div>
            <div style="font-weight: 600;">{odds:.2f}</div>
        </div>
        <div style="display: flex; justify-content: space-between; margin: 10px 0;">
            <div>Probabilité estimée</div>
            <div style="font-weight: 600;">{probability:.0%}</div>
        </div>
        <div style="display: flex; justify-content: space-between; margin: 10px 0;">
            <div>Critère Kelly</div>
            <div style="font-weight: 600;">Kelly/{kelly_fraction}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Bouton pour placer le pari avec style amélioré
    if st.button("💰 Confirmer ce pari", type="primary", key="confirm_bet_btn", use_container_width=True):
        if bet_amount > app_data["current_bankroll"]:
            # AMÉLIORATION UI: Message d'erreur amélioré
            st.error(f"⚠️ Montant du pari ({bet_amount:.2f}€) supérieur à votre bankroll actuelle ({app_data['current_bankroll']:.2f}€)")
        elif bet_amount <= 0:
            st.error("⚠️ Le montant du pari doit être supérieur à 0€")
        else:
            # Animation de chargement
            with st.spinner("Enregistrement du pari..."):
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
                    # Message de succès avec animation
                    st.success(f"✅ Pari enregistré avec succès! {bet_amount:.2f}€ sur {pick} @ {odds:.2f}")
                    
                    # AMÉLIORATION UI: Afficher un récapitulatif attrayant
                    st.markdown(f"""
                    <div class="card section-fade-in" style="background: linear-gradient(145deg, rgba(76, 175, 80, 0.1) 0%, rgba(56, 142, 60, 0.1) 100%); 
                                                             border: 1px solid rgba(76, 175, 80, 0.3);">
                        <div style="text-align: center;">
                            <div style="font-size: 3rem; margin-bottom: 10px;">✅</div>
                            <h3 style="margin-bottom: 15px; color: #4CAF50;">Pari enregistré avec succès</h3>
                            <div style="font-weight: 600; font-size: 1.2rem; margin-bottom: 15px;">
                                {event_name} - {event_date}
                            </div>
                            <div style="font-weight: 600; font-size: 1.1rem;">{pick} @ {odds:.2f}</div>
                            <div style="display: flex; justify-content: space-between; margin: 15px 0; color: rgba(255,255,255,0.8);">
                                <div>Mise: <b>{bet_amount:.2f}€</b></div>
                                <div>Gain potentiel: <b>{pot_gain:.2f}€</b></div>
                                <div>ROI: <b>{roi_pct:.1f}%</b></div>
                            </div>
                            <div style="margin-top: 10px; font-size: 0.9rem; color: rgba(255,255,255,0.6);">
                                Vous pouvez suivre ce pari dans l'onglet "Gestion de Bankroll"
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("❌ Erreur lors de l'enregistrement du pari.")

# PARTIE 9 

def show_betting_strategy_section(event_url, event_name, fights, predictions_data, current_bankroll=300):
    """Affiche la section de stratégie de paris basée sur les prédictions existantes avec UI améliorée"""
    
    # Vérifier si on a déjà fait des recommandations
    event_key = f"recommendations_{event_url}"
    has_existing_recommendations = event_key in st.session_state.betting_recommendations
    
    # Vérifier si on a un dictionnaire global des cotes
    if 'odds_dicts' not in st.session_state:
        st.session_state.odds_dicts = {}

    # Récupérer ou créer le dictionnaire des cotes pour cet événement
    if event_url not in st.session_state.odds_dicts:
        st.session_state.odds_dicts[event_url] = {}
    
    # AMÉLIORATION UI: Titre de section plus attrayant
    st.markdown("""
    <div class="divider"></div>
    <div class="section-fade-in" style="text-align:center; margin: 25px 0;">
        <h2>💰 Stratégie de paris optimisée</h2>
        <p style="color: #aaa;">Utilisez l'intelligence artificielle pour maximiser vos gains</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Si on a des paris sauvegardés pour cet événement, afficher un message
    if event_url in st.session_state.saved_bet_events:
        # AMÉLIORATION UI: Message de succès plus visuel
        st.markdown(f"""
        <div class="card" style="background: linear-gradient(145deg, rgba(76, 175, 80, 0.1) 0%, rgba(56, 142, 60, 0.1) 100%);
                             border: 1px solid rgba(76, 175, 80, 0.3); margin-bottom: 20px; text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 10px;">✅</div>
            <h3 style="color: #4CAF50; margin-bottom: 10px;">Paris enregistrés avec succès</h3>
            <p style="margin-bottom: 0;">Vos {st.session_state.saved_bet_events[event_url]} paris pour cet événement ont été ajoutés à votre suivi.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # AMÉLIORATION UI: Interface de stratégie modernisée
    st.markdown("""
    <div class="card">
        <h3 style="margin-top: 0;">Configurez votre stratégie de paris</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    # Utiliser 100% de la bankroll par défaut pour le budget total affichable
    with col1:
        total_budget = st.number_input(
            "Budget total (€)",
            min_value=10.0,
            max_value=float(current_bankroll),
            value=float(current_bankroll),  # 100% de la bankroll par défaut
            step=10.0,
            format="%.2f",
            key=f"total_budget_{event_url}"
        )
    
    with col2:
        # MODIFICATION: Ajout de la stratégie REALISTIC
        strategy_options = [
            "REALISTIC",
            "ULTRA-SECURE", 
            "OPTIMAL STRATEGY", 
            "AGGRESSIVE", 
            "Kelly complet", 
            "Demi-Kelly", 
            "Quart-Kelly", 
            "Kelly/5", 
            "Kelly/6", 
            "Kelly/8", 
            "Kelly/10", 
            "Kelly/12", 
            "Kelly/14", 
            "Kelly/16"
        ]
        
        kelly_strategy = st.selectbox(
            "Stratégie de paris",
            options=strategy_options,
            index=0,  # REALISTIC par défaut
            key=f"kelly_strategy_{event_url}"
        )
        
        # NOUVEAU: Définition des paramètres pour chaque stratégie
        strategy_params = {
            "REALISTIC": {
                "kelly_fraction": 4.9,
                "min_confidence": 0.578,
                "min_value": 1.15,
                "max_bet_fraction": 0.017,
                "min_edge": 0.094,
                "min_bet_pct": 0.01,
                "description": "Stratégie réaliste équilibrée avec des critères modérés et mise plafonnée à 1.7%"
            },
            "ULTRA-SECURE": {
                "kelly_fraction": 17.3,
                "min_confidence": 0.671,
                "min_value": 1.139,
                "max_bet_fraction": 0.0057,
                "min_edge": 0.045,
                "min_bet_pct": 0.001,
                "description": "Stratégie ultra-conservatrice avec mise maximale de 0.57% par pari"
            },
            "OPTIMAL STRATEGY": {
                "kelly_fraction": 10.84413949349914,
                "min_confidence": 0.6868474607949324,
                "min_value": 1.1443872756387072,
                "max_bet_fraction": 0.04157348795229382,
                "min_edge": 0.0579010170275563,
                "min_bet_pct": 0.0156550016861687,
                "description": "Stratégie optimisée mathématiquement pour un équilibre rendement/risque"
            },
            "AGGRESSIVE": {
                "kelly_fraction": 54.57895572533219,
                "min_confidence": 0.8804031702586482,
                "min_value": 1.0127078432620351,
                "max_bet_fraction": 0.16561756998989577,
                "min_edge": 0.003714553031616009,
                "min_bet_pct": 0.03731285201070915,
                "description": "Stratégie agressive avec critères stricts et mises importantes"
            },
            "Kelly complet": {
                "kelly_fraction": 1,
                "min_confidence": 0.65,
                "min_value": 1.15,
                "max_bet_fraction": 0.05,
                "min_edge": 0.05,
                "min_bet_pct": 0.01,
                "description": "Kelly pur avec plafond de sécurité à 5%"
            },
            "Demi-Kelly": {
                "kelly_fraction": 2,
                "min_confidence": 0.65,
                "min_value": 1.15,
                "max_bet_fraction": 0.05,
                "min_edge": 0.05,
                "min_bet_pct": 0.01,
                "description": "Kelly divisé par 2 pour réduire la variance"
            },
            "Quart-Kelly": {
                "kelly_fraction": 4,
                "min_confidence": 0.65,
                "min_value": 1.15,
                "max_bet_fraction": 0.05,
                "min_edge": 0.05,
                "min_bet_pct": 0.01,
                "description": "Kelly divisé par 4 - approche conservatrice"
            },
            "Kelly/5": {
                "kelly_fraction": 5,
                "min_confidence": 0.65,
                "min_value": 1.15,
                "max_bet_fraction": 0.05,
                "min_edge": 0.05,
                "min_bet_pct": 0.01,
                "description": "Kelly divisé par 5 - très conservateur"
            },
            "Kelly/6": {
                "kelly_fraction": 6,
                "min_confidence": 0.65,
                "min_value": 1.15,
                "max_bet_fraction": 0.05,
                "min_edge": 0.05,
                "min_bet_pct": 0.01,
                "description": "Kelly divisé par 6 - ultra conservateur"
            },
            "Kelly/8": {
                "kelly_fraction": 8,
                "min_confidence": 0.65,
                "min_value": 1.15,
                "max_bet_fraction": 0.05,
                "min_edge": 0.05,
                "min_bet_pct": 0.01,
                "description": "Kelly divisé par 8 - très prudent"
            },
            "Kelly/10": {
                "kelly_fraction": 10,
                "min_confidence": 0.65,
                "min_value": 1.15,
                "max_bet_fraction": 0.05,
                "min_edge": 0.05,
                "min_bet_pct": 0.01,
                "description": "Kelly divisé par 10 - extrêmement prudent"
            },
            "Kelly/12": {
                "kelly_fraction": 12,
                "min_confidence": 0.65,
                "min_value": 1.15,
                "max_bet_fraction": 0.05,
                "min_edge": 0.05,
                "min_bet_pct": 0.01,
                "description": "Kelly divisé par 12 - très minimal"
            },
            "Kelly/14": {
                "kelly_fraction": 14,
                "min_confidence": 0.65,
                "min_value": 1.15,
                "max_bet_fraction": 0.05,
                "min_edge": 0.05,
                "min_bet_pct": 0.01,
                "description": "Kelly divisé par 14 - minimal"
            },
            "Kelly/16": {
                "kelly_fraction": 16,
                "min_confidence": 0.65,
                "min_value": 1.15,
                "max_bet_fraction": 0.05,
                "min_edge": 0.05,
                "min_bet_pct": 0.01,
                "description": "Kelly divisé par 16 - ultra minimal"
            }
        }
        
        # Récupérer les paramètres de la stratégie sélectionnée
        current_strategy = strategy_params[kelly_strategy]
        kelly_divisor = current_strategy["kelly_fraction"]
        min_confidence = current_strategy["min_confidence"]
        min_value = current_strategy["min_value"]
        max_kelly_pct = current_strategy["max_bet_fraction"]
        edge_minimum_fraction = current_strategy["min_edge"]
        min_bet_pct = current_strategy["min_bet_pct"]
        
    # NOUVEAU: Affichage des informations de la stratégie sélectionnée
    st.markdown(f"""
    <div class="card" style="background: linear-gradient(145deg, rgba(30, 136, 229, 0.1) 0%, rgba(21, 101, 192, 0.1) 100%);
                         border-left: 3px solid #1E88E5; margin: 15px 0;">
        <h4 style="color: #1E88E5; margin-top: 0;">📋 Informations sur la stratégie "{kelly_strategy}"</h4>
        <p style="margin-bottom: 15px; font-style: italic;">{current_strategy["description"]}</p>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; font-size: 0.9rem;">
            <div>
                <div style="color: #aaa;">Diviseur Kelly:</div>
                <div style="font-weight: 600;">{kelly_divisor:.2f}</div>
            </div>
            <div>
                <div style="color: #aaa;">Confiance minimum:</div>
                <div style="font-weight: 600;">{min_confidence*100:.1f}%</div>
            </div>
            <div>
                <div style="color: #aaa;">Value minimum:</div>
                <div style="font-weight: 600;">{min_value:.3f}</div>
            </div>
            <div>
                <div style="color: #aaa;">Mise max par pari:</div>
                <div style="font-weight: 600;">{max_kelly_pct*100:.2f}%</div>
            </div>
            <div>
                <div style="color: #aaa;">Edge minimum:</div>
                <div style="font-weight: 600;">{edge_minimum_fraction*100:.1f}%</div>
            </div>
            <div>
                <div style="color: #aaa;">Mise minimum:</div>
                <div style="font-weight: 600;">{min_bet_pct*100:.2f}%</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # AMÉLIORATION UI: Section pour les cotes améliorée
    st.markdown("""
    <div class="card" style="margin-top: 15px;">
        <h3 style="margin-top: 0;">Entrez les cotes proposées par les bookmakers</h3>
        <p style="color: #aaa; margin-bottom: 15px;">Ces cotes seront utilisées pour calculer la valeur de chaque pari</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Option pour ajuster l'edge minimum (avec valeur par défaut de la stratégie)
    edge_minimum = st.slider(
        "Edge minimum (%) - Différence minimum entre probabilité modèle et probabilité implicite",
        min_value=0.1,
        max_value=15.0,
        value=edge_minimum_fraction * 100,
        step=0.1,
        key=f"min_edge_{event_url}"
    )
    
    # Convertir le pourcentage en fraction
    edge_minimum_fraction = edge_minimum / 100.0
    
    # Initialiser ou récupérer le dictionnaire pour stocker les cotes entrées
    if f"odds_dict_{event_url}" not in st.session_state:
        st.session_state[f"odds_dict_{event_url}"] = st.session_state.odds_dicts.get(event_url, {})
    
    # Calculer le nombre de combats bettables
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
            
        # Créer une ligne pour chaque combat bettable
        winner = "Red" if result['red_probability'] > result['blue_probability'] else "Blue"
        winner_name = red_fighter_name if winner == "Red" else blue_fighter_name
        winner_prob = max(result['red_probability'], result['blue_probability'])
        
        # Ajouter aux combats pour parier
        bettable_fights.append({
            'fight_key': fight_key,
            'red_fighter': red_fighter_name,
            'blue_fighter': blue_fighter_name,
            'winner': winner,
            'winner_name': winner_name,
            'probability': winner_prob
        })
    
    # AMÉLIORATION UI: Mode de saisie des cotes (manuel vs slider)
    odds_input_mode = st.radio(
        "Mode de saisie des cotes",
        options=["Manuel", "Slider"],
        index=0,  # Manuel par défaut
        key=f"odds_input_mode_{event_url}"
    )
    
    # AMÉLIORATION UI: Afficher les combats en grille responsive
    if bettable_fights:
        # Créer des rangées de 2 combats chacune
        for i in range(0, len(bettable_fights), 2):
            row_fights = bettable_fights[i:i+2]
            cols = st.columns(len(row_fights))
            
            for j, fight in enumerate(row_fights):
                with cols[j]:
                    fight_key = fight['fight_key']
                    
                    # NOUVEAU: Déterminer la couleur du pourcentage selon la confiance minimum
                    probability_color = "#4CAF50" if fight['probability'] >= min_confidence else "#F44336"
                    probability_bg = "rgba(76, 175, 80, 0.1)" if fight['probability'] >= min_confidence else "rgba(244, 67, 54, 0.1)"
                    
                    # AMÉLIORATION UI: Card de combat modernisée avec pourcentage coloré
                    st.markdown(f"""
                    <div class="fight-card-improved">
                        <div class="fighters-banner">
                            <div class="fighter-name-red">{fight['red_fighter']}</div>
                            <div class="vs-badge">VS</div>
                            <div class="fighter-name-blue">{fight['blue_fighter']}</div>
                        </div>
                        <div style="text-align: center; margin-bottom: 10px;">
                            <span>Vainqueur prédit: </span>
                            <span style="font-weight: 600; color: {('#E53935' if fight['winner'] == 'Red' else '#1E88E5')};">
                                {fight['winner_name']}
                            </span>
                            <span style="background: {probability_bg}; color: {probability_color}; padding: 2px 6px; border-radius: 12px; font-weight: 600; margin-left: 5px;">
                                ({fight['probability']:.0%})
                            </span>
                        </div>
                        <div style="text-align: center; font-size: 0.8rem; color: #aaa; margin-bottom: 10px;">
                            Confiance min: {min_confidence*100:.1f}% 
                            {'✅' if fight['probability'] >= min_confidence else '❌'}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Champ pour entrer la cote du bookmaker
                    odds_key = f"odds_{fight_key}"
                    
                    # Initialiser la valeur si première utilisation - MODIFICATION: cote par défaut à 1.01
                    if odds_key not in st.session_state[f"odds_dict_{event_url}"]:
                        st.session_state[f"odds_dict_{event_url}"][odds_key] = 1.01
                    
                    # Selon le mode de saisie choisi
                    if odds_input_mode == "Manuel":
                        # Saisie manuelle de la cote
                        odds = st.number_input(
                            "Cote",
                            min_value=1.01,
                            value=st.session_state[f"odds_dict_{event_url}"][odds_key],
                            step=0.01,
                            format="%.2f",
                            key=f"manual_{odds_key}"
                        )
                    else:
                        # Saisie avec slider
                        odds = st.slider(
                            "Cote",
                            min_value=1.01,
                            max_value=10.0,
                            value=st.session_state[f"odds_dict_{event_url}"][odds_key],
                            step=0.05,
                            format="%.2f",
                            key=f"slider_{odds_key}"
                        )
                    
                    # CORRECTION: Mettre à jour à la fois dans le combat et dans le dictionnaire
                    fight['odds'] = odds
                    st.session_state[f"odds_dict_{event_url}"][odds_key] = odds
                    
                    # Sauvegarder aussi dans le dictionnaire global
                    st.session_state.odds_dicts[event_url] = st.session_state[f"odds_dict_{event_url}"]
                    
                    # Fermeture de la div
                    st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Aucun combat avec prédiction n'est disponible. Veuillez d'abord générer des prédictions.")
    
    # CORRECTION: Recalcul de la stratégie avec un flag d'état
    recalculate_btn = st.button("📊 Recalculer la stratégie de paris", key=f"recalculate_strategy_{event_url}")
    
    # Quand le bouton de recalcul est cliqué, définir un flag
    if recalculate_btn:
        st.session_state[f"recalculate_{event_url}"] = True
        generate_strategy = True
    else:
        generate_strategy = False

    # Récupérer les recommandations existantes ou générer de nouvelles
    if has_existing_recommendations and not (generate_strategy or st.session_state.get(f"recalculate_{event_url}", False)):
        filtered_fights = st.session_state.betting_recommendations[event_key]
        # AMÉLIORATION UI: Titre de section amélioré
        st.markdown("""
        <div class="divider"></div>
        <div class="section-fade-in" style="text-align:center; margin: 25px 0;">
            <h2>💰 Recommandations de paris</h2>
            <p style="color: #aaa;">Combats identifiés comme offrant une valeur positive</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Afficher les recommandations sauvegardées
        if filtered_fights:
            # AMÉLIORATION UI: Tableau des recommandations plus moderne
            recommendation_data = []
            for fight in filtered_fights:
                # Vérifier si les nouveaux attributs existent
                if 'original_kelly_pct' in fight:
                    kelly_pct = fight['original_kelly_pct'] * 100
                    is_capped = fight.get('is_capped', False)
                    
                    # Identifier quelle valeur afficher dans la colonne Kelly
                    display_kelly = f"{kelly_pct:.1f}%"
                    if is_capped:
                        display_kelly += f" → {fight.get('capped_kelly_pct', 0.0057)*100:.2f}%"  # Ajouter l'information de plafonnement
                else:
                    # Compatibilité avec l'ancien format
                    kelly_pct = fight.get('fractional_kelly', 0) * 100
                    display_kelly = f"{kelly_pct:.1f}%"
                
                recommendation_data.append({
                    "Combat": f"{fight['red_fighter']} vs {fight['blue_fighter']}",
                    "Pari sur": fight['winner_name'],
                    "Probabilité": f"{fight['probability']:.0%}",
                    "Cote": f"{fight['odds']:.2f}",
                    "Value": f"{fight['edge']*100:.1f}%",
                    "Kelly": display_kelly,
                    "Montant": f"{fight['stake']:.2f} €",
                    "Gain potentiel": f"{fight['stake'] * (fight['odds']-1):.2f} €"
                })
                
            df = pd.DataFrame(recommendation_data)
            
            # AMÉLIORATION UI: Dataframe avec formatage amélioré
            st.dataframe(
                df,
                use_container_width=True,
                column_config={
                    "Combat": st.column_config.TextColumn("Combat"),
                    "Pari sur": st.column_config.TextColumn("Pari sur"),
                    "Probabilité": st.column_config.TextColumn("Probabilité"),
                    "Cote": st.column_config.TextColumn("Cote"),
                    "Value": st.column_config.TextColumn("Value"),
                    "Kelly": st.column_config.TextColumn("Kelly"),
                    "Montant": st.column_config.TextColumn("Montant"),
                    "Gain potentiel": st.column_config.TextColumn("Gain potentiel")
                },
                hide_index=True
            )
            
            # AMÉLIORATION UI: Résumé de la stratégie dans une card moderne
            total_stake = sum(fight['stake'] for fight in filtered_fights)
            total_potential_profit = sum(fight['stake'] * (fight['odds']-1) for fight in filtered_fights)
            
            # Afficher les métriques en 3 colonnes
            summary_cols = st.columns(3)
            with summary_cols[0]:
                st.metric("Bankroll", f"{current_bankroll:.2f}€")
            with summary_cols[1]:
                st.metric("Montant misé", f"{total_stake:.2f}€", f"{total_stake/current_bankroll*100:.1f}%")
            with summary_cols[2]:
                st.metric("Gain potentiel", f"{total_potential_profit:.2f}€", f"{total_potential_profit/total_stake*100:.1f}%")
            
            # Résumé détaillé
            strategy_name = f"{kelly_strategy} avec plafond à {max_kelly_pct*100:.2f}%"
            
            st.markdown(f"""
            <div class="strategy-summary">
                <h4 style="margin-top: 0;">Résumé de la stratégie</h4>
                <ul>
                    <li>Stratégie: <b>{strategy_name}</b></li>
                    <li>Diviseur Kelly: <b>{kelly_divisor:.2f}</b></li>
                    <li>Confiance minimum: <b>{min_confidence*100:.1f}%</b></li>
                    <li>Value minimum: <b>{min_value:.3f}</b></li>
                    <li>Edge minimum: <b>{edge_minimum:.1f}%</b></li>
                    <li>Nombre de paris recommandés: <b>{len(filtered_fights)}</b></li>
                    <li>Exposition totale: <b>{total_stake/current_bankroll*100:.1f}%</b> de la bankroll</li>
                    <li>ROI potentiel: <b>{total_potential_profit/total_stake*100:.1f}%</b></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Option pour enregistrer les paris seulement si pas déjà sauvegardés
            if event_url not in st.session_state.saved_bet_events:
                if st.button("💾 Enregistrer ces paris dans mon suivi", type="primary", key=f"save_all_bets_{event_url}", use_container_width=True):
                    # Animation de chargement
                    with st.spinner("Enregistrement des paris..."):
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
                                # Message de succès avec animation
                                st.success(f"✅ Tous les paris ({successful_bets}) ont été enregistrés avec succès!")
                                
                                # AMÉLIORATION UI: Afficher un récapitulatif attrayant
                                st.markdown(f"""
                                <div class="card section-fade-in" style="background: linear-gradient(145deg, rgba(76, 175, 80, 0.1) 0%, rgba(56, 142, 60, 0.1) 100%); 
                                                                         border: 1px solid rgba(76, 175, 80, 0.3); text-align: center;">
                                    <div style="font-size: 3rem; margin-bottom: 10px;">✅</div>
                                    <h3 style="margin-bottom: 15px; color: #4CAF50;">Tous les paris enregistrés</h3>
                                    <div style="font-weight: 600; font-size: 1.2rem; margin-bottom: 15px;">
                                        {successful_bets} paris ajoutés pour {event_name}
                                    </div>
                                    <div style="display: flex; justify-content: space-between; margin: 15px 0; color: rgba(255,255,255,0.8);">
                                        <div>Mise totale: <b>{total_stake:.2f}€</b></div>
                                        <div>Gain potentiel: <b>{total_potential_profit:.2f}€</b></div>
                                        <div>ROI: <b>{total_potential_profit/total_stake*100:.1f}%</b></div>
                                    </div>
                                    <div style="margin-top: 10px; font-size: 0.9rem; color: rgba(255,255,255,0.6);">
                                        Vous pouvez suivre ces paris dans l'onglet "Gestion de Bankroll"
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                            elif successful_bets > 0:
                                st.warning(f"⚠️ {successful_bets}/{len(filtered_fights)} paris ont été enregistrés. Certains paris n'ont pas pu être enregistrés.")
                            else:
                                st.error("❌ Aucun pari n'a pu être enregistré.")
                        except Exception as e:
                            st.error(f"❌ Erreur lors de l'enregistrement des paris: {e}")
        else:
            # AMÉLIORATION UI: Message d'avertissement plus visuel
            st.markdown(f"""
            <div class="card" style="background: linear-gradient(145deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 160, 0, 0.1) 100%);
                                     border: 1px solid rgba(255, 193, 7, 0.3); text-align: center; padding: 20px;">
                <div style="font-size: 2rem; margin-bottom: 10px;">⚠️</div>
                <h3 style="color: #FFC107; margin-bottom: 10px;">Aucun combat intéressant</h3>
                <p style="margin-bottom: 0;">Aucun combat ne correspond aux critères de value betting (confiance ≥ {min_confidence*100:.1f}%, edge ≥ {edge_minimum:.1f}%, value ≥ {min_value:.3f}).</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        # CORRECTION: Si on a cliqué sur recalculer, effacer le flag pour le prochain chargement
        if st.session_state.get(f"recalculate_{event_url}", False):
            st.session_state[f"recalculate_{event_url}"] = False
            
        # Bouton pour générer/régénérer la stratégie
        generate_btn = st.button("📊 Générer la stratégie de paris", type="primary", key=f"generate_strategy_{event_url}", use_container_width=True)
        
        if generate_btn or generate_strategy:
            # Animation de chargement
            with st.spinner("Analyse des opportunités de paris..."):
                # Filtrer les combats intéressants
                filtered_fights = []
                
                # CORRECTION: S'assurer que les cotes sont correctement appliquées aux combats
                for fight in bettable_fights:
                    fight_key = fight['fight_key']
                    odds_key = f"odds_{fight_key}"
                    
                    # Récupérer la cote depuis le dictionnaire de session
                    if odds_key in st.session_state[f"odds_dict_{event_url}"]:
                        fight['odds'] = st.session_state[f"odds_dict_{event_url}"][odds_key]
                    else:
                        # Valeur par défaut si non définie - MODIFICATION: cote par défaut à 1.01
                        fight['odds'] = 1.01
                        st.session_state[f"odds_dict_{event_url}"][odds_key] = 1.01
                        
                    # Vérifier la confiance du modèle selon la stratégie
                    if fight['probability'] < min_confidence:
                        continue
                        
                    # Vérifier le value betting
                    implicit_prob = 1 / fight['odds']
                    
                    # NOUVEAU: Calculer l'edge et vérifier qu'il est supérieur au minimum défini
                    edge = fight['probability'] - implicit_prob
                    if edge < edge_minimum_fraction:  # Edge minimum (exprimé en fraction décimale)
                        continue
                    
                    # Vérifier la value selon la stratégie
                    value = fight['probability'] * fight['odds']
                    if value < min_value:
                        continue
                        
                    # Calculer la fraction Kelly pure
                    p = fight['probability']
                    q = 1 - p
                    b = fight['odds'] - 1
                    kelly = (p * b - q) / b
                    
                    # Appliquer le diviseur Kelly
                    fractional_kelly = kelly / kelly_divisor
                    
                    # Vérifier que la mise est au-dessus du minimum
                    if fractional_kelly < min_bet_pct:
                        continue
                    
                    # Ajouter aux paris recommandés
                    if fractional_kelly > 0:
                        filtered_fights.append({
                            **fight,
                            'kelly': kelly,
                            'fractional_kelly': fractional_kelly,
                            'edge': edge,
                            'value': value
                        })
                
                # Sauvegarder les recommandations dans la session state
                st.session_state.betting_recommendations[event_key] = filtered_fights
            
            # AMÉLIORATION UI: Titre de section amélioré
            st.markdown("""
            <div class="divider"></div>
            <div class="section-fade-in" style="text-align:center; margin: 25px 0;">
                <h2>💰 Recommandations de paris</h2>
                <p style="color: #aaa;">Combats identifiés comme offrant une valeur positive</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Afficher les résultats
            if not filtered_fights:
                # AMÉLIORATION UI: Message d'avertissement plus visuel
                st.markdown(f"""
                <div class="card" style="background: linear-gradient(145deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 160, 0, 0.1) 100%);
                                         border: 1px solid rgba(255, 193, 7, 0.3); text-align: center; padding: 20px;">
                    <div style="font-size: 2rem; margin-bottom: 10px;">⚠️</div>
                    <h3 style="color: #FFC107; margin-bottom: 10px;">Aucun combat intéressant</h3>
                    <p style="margin-bottom: 0;">Aucun combat ne correspond aux critères de value betting (confiance ≥ {min_confidence*100:.1f}%, edge ≥ {edge_minimum:.1f}%, value ≥ {min_value:.3f}).</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Afficher des informations de débogage sur les combats évalués
                debug_info = st.expander("🔍 Détails des combats évalués", expanded=False)
                with debug_info:
                    for fight in bettable_fights:
                        fight_key = fight['fight_key']
                        odds_key = f"odds_{fight_key}"
                        odds_value = st.session_state[f"odds_dict_{event_url}"].get(odds_key, "Non définie")
                        implicit_prob = 1 / float(odds_value) if isinstance(odds_value, (int, float)) and odds_value > 0 else "N/A"
                        edge = fight['probability'] - implicit_prob if isinstance(implicit_prob, float) else "N/A"
                        
                        # Affichage des raisons adaptées à la nouvelle stratégie
                        if isinstance(edge, float):
                            value = fight['probability'] * float(odds_value)
                            reason = f"Confiance < {min_confidence*100:.1f}%" if fight['probability'] < min_confidence else \
                                    f"Edge insuffisant ({edge*100:.1f}% < {edge_minimum:.1f}%)" if edge < edge_minimum_fraction else \
                                    f"Value insuffisante ({value:.3f} < {min_value:.3f})" if value < min_value else \
                                    "Raison inconnue"
                        else:
                            reason = "Erreur dans les données de cote"
                        
                        st.markdown(f"""
                        <div style="margin-bottom: 10px; padding: 10px; background: rgba(255,255,255,0.05); border-radius: 8px;">
                            <div><b>{fight['red_fighter']} vs {fight['blue_fighter']}</b></div>
                            <div>Vainqueur prédit: <b>{fight['winner_name']}</b> ({fight['probability']:.0%})</div>
                            <div>Cote: <b>{odds_value}</b> (Probabilité implicite: {implicit_prob if isinstance(implicit_prob, float) else implicit_prob})</div>
                            <div>Edge: <b>{edge*100:.1f}%</b> si edge est un nombre à virgule flottante else edge</div>
                            <div>Raison: {reason}</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                # Application de la méthode Kelly avec plafond adapté selon la stratégie
                # Appliquer le Kelly fixe avec plafond absolu pour chaque combat
                for fight in filtered_fights:
                    # Utiliser directement le pourcentage Kelly calculé (déjà divisé par kelly_divisor)
                    kelly_pct = fight['fractional_kelly']
                    
                    # Plafonner selon la stratégie
                    capped_kelly = min(kelly_pct, max_kelly_pct)
                    
                    # Calculer la mise finale en euros
                    fight['stake'] = current_bankroll * capped_kelly
                    
                    # Stocker les deux valeurs pour l'affichage (original et plafonné)
                    fight['original_kelly_pct'] = kelly_pct
                    fight['capped_kelly_pct'] = capped_kelly
                    fight['is_capped'] = kelly_pct > max_kelly_pct
                    
                    # Arrondir pour plus de clarté
                    fight['stake'] = round(fight['stake'], 2)
                
                # AMÉLIORATION UI: Afficher les recommandations dans un tableau moderne
                recommendation_data = []
                for fight in filtered_fights:
                    kelly_pct = fight['original_kelly_pct'] * 100  # Convertir en pourcentage pour l'affichage
                    
                    # Identifier quelle valeur afficher dans la colonne Kelly
                    display_kelly = f"{kelly_pct:.1f}%"
                    if fight['is_capped']:
                        cap_pct = max_kelly_pct * 100
                        display_kelly += f" → {cap_pct:.2f}%"  # Ajouter l'information de plafonnement
                    
                    recommendation_data.append({
                        "Combat": f"{fight['red_fighter']} vs {fight['blue_fighter']}",
                        "Pari sur": fight['winner_name'],
                        "Probabilité": f"{fight['probability']:.0%}",
                        "Cote": f"{fight['odds']:.2f}",
                        "Value": f"{fight['edge']*100:.1f}%",
                        "Kelly": display_kelly,
                        "Montant": f"{fight['stake']:.2f} €",
                        "Gain potentiel": f"{fight['stake'] * (fight['odds']-1):.2f} €"
                    })
                
                if recommendation_data:
                    df = pd.DataFrame(recommendation_data)
                    
                    # AMÉLIORATION UI: Dataframe avec formatage amélioré
                    st.dataframe(
                        df,
                        use_container_width=True,
                        column_config={
                            "Combat": st.column_config.TextColumn("Combat"),
                            "Pari sur": st.column_config.TextColumn("Pari sur"),
                            "Probabilité": st.column_config.TextColumn("Probabilité"),
                            "Cote": st.column_config.TextColumn("Cote"),
                            "Value": st.column_config.TextColumn("Value"),
                            "Kelly": st.column_config.TextColumn("Kelly"),
                            "Montant": st.column_config.TextColumn("Montant"),
                            "Gain potentiel": st.column_config.TextColumn("Gain potentiel")
                        },
                        hide_index=True
                    )
                    
                    # Message pour les paris plafonnés adapté à la stratégie
                    capped_fights = [f for f in filtered_fights if f.get('is_capped', False)]
                    if capped_fights:
                        cap_pct = max_kelly_pct * 100
                        st.markdown(f"""
                        <div class="card" style="background: linear-gradient(145deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 160, 0, 0.1) 100%);
                                     border-left: 3px solid #FFC107; margin-top: 15px;">
                            <h4 style="color: #FFC107; margin-top: 0;">⚠️ Information importante</h4>
                            <p>Certaines mises ont été plafonnées à {cap_pct:.1f}% de votre bankroll (règle de gestion du risque {kelly_strategy}):</p>
                            <ul>
                        """, unsafe_allow_html=True)
                        
                        for fight in capped_fights:
                            original_stake = current_bankroll * fight['original_kelly_pct']
                            st.markdown(f"""
                                <li>{fight['winner_name']} : Kelly={fight['original_kelly_pct']*100:.1f}% → {fight['capped_kelly_pct']*100:.1f}% (plafonné à {cap_pct:.1f}%)</li>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("""
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # AMÉLIORATION UI: Résumé de la stratégie avec des métriques
                    total_stake = sum(fight['stake'] for fight in filtered_fights)
                    total_potential_profit = sum(fight['stake'] * (fight['odds']-1) for fight in filtered_fights)
                    
                    # Afficher les métriques en 3 colonnes
                    summary_cols = st.columns(3)
                    with summary_cols[0]:
                        st.metric("Bankroll", f"{current_bankroll:.2f}€")
                    with summary_cols[1]:
                        st.metric("Montant misé", f"{total_stake:.2f}€", f"{total_stake/current_bankroll*100:.1f}%")
                    with summary_cols[2]:
                        st.metric("Gain potentiel", f"{total_potential_profit:.2f}€", f"{total_potential_profit/total_stake*100:.1f}%")
                    
                    # Résumé détaillé avec paramètres spécifiques à la stratégie
                    strategy_name = f"{kelly_strategy} avec plafond à {max_kelly_pct*100:.1f}%"
                    
                    st.markdown(f"""
                    <div class="strategy-summary">
                        <h4 style="margin-top: 0;">Résumé de la stratégie</h4>
                        <ul>
                            <li>Stratégie: <b>{strategy_name}</b></li>
                            <li>Diviseur Kelly: <b>{kelly_divisor:.1f}</b></li>
                            <li>Confiance minimum: <b>{min_confidence*100:.1f}%</b></li>
                            <li>Value minimum: <b>{min_value:.3f}</b></li>
                            <li>Edge minimum: <b>{edge_minimum:.1f}%</b></li>
                            <li>Mise minimum: <b>{min_bet_pct*100:.1f}%</b></li>
                            <li>Nombre de paris recommandés: <b>{len(filtered_fights)}</b></li>
                            <li>Exposition totale: <b>{total_stake/current_bankroll*100:.1f}%</b> de la bankroll</li>
                            <li>ROI potentiel: <b>{total_potential_profit/total_stake*100:.1f}%</b></li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Option pour enregistrer les paris
                    if event_url not in st.session_state.saved_bet_events:
                        if st.button("💾 Enregistrer ces paris dans mon suivi", type="primary", key=f"save_all_bets_{event_url}", use_container_width=True):
                            # Animation de chargement
                            with st.spinner("Enregistrement des paris..."):
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
                                        # Message de succès avec animation
                                        st.success(f"✅ Tous les paris ({successful_bets}) ont été enregistrés avec succès!")
                                        
                                        # AMÉLIORATION UI: Afficher un récapitulatif attrayant
                                        st.markdown(f"""
                                        <div class="card section-fade-in" style="background: linear-gradient(145deg, rgba(76, 175, 80, 0.1) 0%, rgba(56, 142, 60, 0.1) 100%); 
                                                                                 border: 1px solid rgba(76, 175, 80, 0.3); text-align: center;">
                                            <div style="font-size: 3rem; margin-bottom: 10px;">✅</div>
                                            <h3 style="margin-bottom: 15px; color: #4CAF50;">Tous les paris enregistrés</h3>
                                            <div style="font-weight: 600; font-size: 1.2rem; margin-bottom: 15px;">
                                                {successful_bets} paris ajoutés pour {event_name}
                                            </div>
                                            <div style="display: flex; justify-content: space-between; margin: 15px 0; color: rgba(255,255,255,0.8);">
                                                <div>Mise totale: <b>{total_stake:.2f}€</b></div>
                                                <div>Gain potentiel: <b>{total_potential_profit:.2f}€</b></div>
                                                <div>ROI: <b>{total_potential_profit/total_stake*100:.1f}%</b></div>
                                            </div>
                                            <div style="margin-top: 10px; font-size: 0.9rem; color: rgba(255,255,255,0.6);">
                                                Vous pouvez suivre ces paris dans l'onglet "Gestion de Bankroll"
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    elif successful_bets > 0:
                                        st.warning(f"⚠️ {successful_bets}/{len(filtered_fights)} paris ont été enregistrés. Certains paris n'ont pas pu être enregistrés.")
                                    else:
                                        st.error("❌ Aucun pari n'a pu être enregistré.")
                                except Exception as e:
                                    st.error(f"❌ Erreur lors de l'enregistrement des paris: {e}")

def show_upcoming_events_page():
    """Affiche la page des événements à venir avec UI améliorée"""
    # AMÉLIORATION UI: Titre de page simple
    st.title("🗓️ Événements UFC à venir")
    st.write("Consultez et analysez les prochains combats de l'UFC")
    
    # AMÉLIORATION UI: Bouton de récupération plus visible et explicite
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("🔍 Récupérer les événements", key="load_events_btn", type="primary", use_container_width=True):
            # AMÉLIORATION UI: Animation de chargement
            with st.spinner("Récupération des événements en cours..."):
                events_result = get_upcoming_events(max_events=8)
                st.session_state.upcoming_events = events_result['events'] if events_result['status'] == 'success' else []
                st.session_state.upcoming_events_timestamp = datetime.datetime.now()
                
                # Initialiser le dictionnaire des combats s'il n'existe pas déjà
                if 'upcoming_fights' not in st.session_state:
                    st.session_state.upcoming_fights = {}
                
                # Initialiser les prédictions s'il n'existe pas déjà
                if 'fight_predictions' not in st.session_state:
                    st.session_state.fight_predictions = {}
                
                # Message de réussite ou d'erreur selon le résultat
                if events_result['status'] == 'success':
                    # Message de succès
                    st.success(f"✅ {len(st.session_state.upcoming_events)} événements récupérés avec succès!")
                else:
                    # Message d'erreur plus informatif
                    st.error(f"❌ Impossible de récupérer les événements: {events_result['message']}")
    
    with col2:
        # Afficher la date de dernière mise à jour
        if st.session_state.get('upcoming_events_timestamp'):
            last_update = st.session_state.upcoming_events_timestamp
            time_diff = (datetime.datetime.now() - last_update).total_seconds() / 60
            
            # Formater le message selon le temps écoulé
            if time_diff < 60:
                time_msg = f"il y a {int(time_diff)} minutes"
            else:
                time_msg = f"il y a {int(time_diff/60)} heures"
            
            st.caption(f"Dernière mise à jour: {time_msg}")
    
    # Afficher les événements s'ils existent
    if st.session_state.get('upcoming_events'):
        # Bouton de rafraîchissement
        refresh_col, _ = st.columns([1, 3])
        with refresh_col:
            if st.button("🔄 Rafraîchir les événements", key="refresh_events_btn", use_container_width=True):
                with st.spinner("Mise à jour des événements..."):
                    events_result = get_upcoming_events(max_events=3)
                    st.session_state.upcoming_events = events_result['events'] if events_result['status'] == 'success' else []
                    st.session_state.upcoming_events_timestamp = datetime.datetime.now()
                st.success("✅ Liste des événements mise à jour!")
        
        # Onglets avec noms d'événements
        event_names = [event['name'] for event in st.session_state.upcoming_events]
        event_tabs = st.tabs(event_names)
        
        # Afficher chaque événement dans son propre onglet
        for i, (event, event_tab) in enumerate(zip(st.session_state.upcoming_events, event_tabs)):
            event_name = event['name']
            event_url = event['url']
            
            with event_tab:
                # Titre de l'événement
                st.subheader(f"🥊 {event_name}")
                
                # Vérifier si les combats pour cet événement sont déjà chargés
                fights = st.session_state.upcoming_fights.get(event_url, [])
                
                # Bouton de chargement pour les combats spécifiques
                if not fights:
                    if st.button(f"🔍 Charger les combats pour {event_name}", key=f"load_fights_btn_{i}", use_container_width=True):
                        with st.spinner(f"Récupération des combats pour {event_name}..."):
                            fights_result = extract_upcoming_fights(event_url)
                            fights = fights_result['fights']
                            st.session_state.upcoming_fights[event_url] = fights
                            
                            # Message selon le résultat
                            if fights_result['status'] == 'success' and fights:
                                st.success(f"✅ {len(fights)} combats chargés avec succès!")
                            elif fights_result['status'] == 'warning':
                                st.warning(f"⚠️ {fights_result['message']}")
                            else:
                                st.error(f"❌ {fights_result['message']}")
                
                if not fights:
                    # Message d'information simple
                    st.info(f"Aucun combat chargé. Cliquez sur le bouton 'Charger les combats pour {event_name}' pour voir les affrontements prévus.")
                else:
                    # Afficher le nombre de combats chargés
                    st.success(f"{len(fights)} combats chargés")
                    
                    # Vérifier si les prédictions ont déjà été générées
                    predictions_generated = event_url in st.session_state.fight_predictions
                    
                    # Bouton pour générer les prédictions
                    if not predictions_generated:
                        if st.button(f"🔮 Générer les prédictions", key=f"predict_fights_btn_{i}", type="primary", use_container_width=True):
                            # Animation de chargement
                            with st.spinner(f"Génération des prédictions pour {len(fights)} combats..."):
                                # Initialiser le dictionnaire pour cet événement
                                st.session_state.fight_predictions[event_url] = {}
                                
                                # Barre de progression
                                progress_bar = st.progress(0)
                                
                                # Générer les prédictions pour chaque combat
                                for idx, fight in enumerate(fights):
                                    # Mettre à jour la barre de progression
                                    progress_bar.progress((idx + 1) / len(fights))
                                    
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
                                
                                # Supprimer la barre de progression
                                progress_bar.empty()
                                
                                # Message de succès
                                st.success(f"✅ Prédictions générées pour {len(fights)} combats!")
                                st.session_state[f"show_strategy_{event_url}"] = True
                    
                    # Afficher les combats
                    st.subheader("🔮 Carte des combats avec prédictions")
                    
                    # Disposer les combats en grille responsive de 2 colonnes
                    for j in range(0, len(fights), 2):
                        cols = st.columns(2 if j + 1 < len(fights) else 1)
                        
                        for k, col in enumerate(cols):
                            if j + k < len(fights):
                                fight = fights[j + k]
                                red_fighter_name = fight['red_fighter']
                                blue_fighter_name = fight['blue_fighter']
                                fight_key = f"{red_fighter_name}_vs_{blue_fighter_name}"
                                
                                with col:
                                    # Créer un cadre pour chaque combat
                                    st.write("---")
                                    # Afficher les noms des combattants
                                    st.write(f"**🔴 {red_fighter_name}** VS **🔵 {blue_fighter_name}**")
                                    
                                    # Vérifier si les prédictions ont été générées pour ce combat
                                    prediction_data = st.session_state.fight_predictions.get(event_url, {}).get(fight_key, None)
                                    
                                    if not prediction_data:
                                        # Affichage simplifié sans prédiction
                                        st.warning("En attente de prédiction")
                                        continue
                                    
                                    if prediction_data['status'] == 'error':
                                        # Message d'erreur
                                        st.error(prediction_data['message'])
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
                                    
                                    # Affichage des probabilités avec une barre de progression Streamlit
                                    st.write("**Probabilités de victoire:**")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.write(f"🔴 {red_match}: {red_prob:.0%}")
                                    with col2:
                                        st.write(f"🔵 {blue_match}: {blue_prob:.0%}")
                                    
                                    # Utiliser la barre de progression pour montrer les probabilités
                                    # Utiliser la barre de progression bicolore pour montrer les probabilités
                                    progress_html = create_probability_bar(
                                        red_prob, 
                                        blue_prob, 
                                        red_match, 
                                        blue_match
                                    )
                                    st.markdown(progress_html, unsafe_allow_html=True)
                                    
                                    # Afficher le vainqueur prédit
                                    st.write("**Vainqueur prédit:**")
                                    winner_emoji = "🔴" if winner_ml == "Red" else "🔵"
                                    winner_hex_color = "#E53935" if winner_ml == "Red" else "#1E88E5"
                                    st.markdown(f"<h3 style='color: {winner_hex_color};'>{winner_emoji} {winner_name}</h3>", unsafe_allow_html=True)
                                    
                                    # Afficher la méthode et la confiance
                                    st.write(f"**Méthode:** {method}")
                                    confidence_color = "green" if confidence == "Élevé" else "orange"
                                    st.markdown(f"**Confiance:** :{confidence_color}[{confidence}]")
                                    
                                    # Si ML est disponible, afficher l'info sur le consensus
                                    if ml_result and not consensus:
                                        st.warning("⚠️ Méthodes en désaccord")
                                    elif ml_result:
                                        st.success("✅ Méthodes en accord")
                                    
                                    # Ajouter un expander pour les détails du combat
                                    with st.expander("Voir les détails"):
                                        # Créer deux colonnes pour les prédictions
                                        detail_cols = st.columns(2 if ml_result else 1)
                                        
                                        # Afficher la prédiction statistique
                                        with detail_cols[0]:
                                            st.write("### Prédiction Statistique")
                                            winner_name_classic = classic_result['winner_name']
                                            winner_classic_emoji = "🔴" if winner_classic == "Red" else "🔵"
                                            winner_classic_color = "#E53935" if winner_classic == "Red" else "#1E88E5"
                                            st.markdown(f"**Vainqueur prédit:** <span style='color: {winner_classic_color};'>{winner_classic_emoji} {winner_name_classic}</span>", unsafe_allow_html=True)
                                            st.write(f"**Probabilités:** {classic_result['red_probability']:.0%} (Rouge) vs {classic_result['blue_probability']:.0%} (Bleu)")
                                            st.write(f"**Confiance:** {classic_result['confidence']}")
                                        
                                        # Afficher la prédiction ML si disponible
                                        if ml_result:
                                            with detail_cols[1]:
                                                st.write("### Prédiction Machine Learning")
                                                winner_name_ml = ml_result['winner_name']
                                                winner_ml_emoji = "🔴" if winner_ml == "Red" else "🔵"
                                                winner_ml_color = "#E53935" if winner_ml == "Red" else "#1E88E5"
                                                st.markdown(f"**Vainqueur prédit:** <span style='color: {winner_ml_color};'>{winner_ml_emoji} {winner_name_ml}</span>", unsafe_allow_html=True)
                                                st.write(f"**Probabilités:** {ml_result['red_probability']:.0%} (Rouge) vs {ml_result['blue_probability']:.0%} (Bleu)")
                                                st.write(f"**Confiance:** {ml_result['confidence']}")
                                        
                                        # Statistiques principales seulement
                                        st.write("### Statistiques principales")
                                        
                                        # Extraire les stats les plus importantes
                                        key_stats = [
                                            ('Record', f"{red_stats['wins']}-{red_stats['losses']}", f"{blue_stats['wins']}-{blue_stats['losses']}"),
                                            ('Frappes/min', f"{red_stats['SLpM']:.1f}", f"{blue_stats['SLpM']:.1f}"),
                                            ('Précision frappes', f"{red_stats['sig_str_acc']:.0%}", f"{blue_stats['sig_str_acc']:.0%}"),
                                            ('Takedowns/match', f"{red_stats['td_avg']:.1f}", f"{blue_stats['td_avg']:.1f}"),
                                            ('Défense takedowns', f"{red_stats['td_def']:.0%}", f"{blue_stats['td_def']:.0%}")
                                        ]
                                        
                                        # Créer un DataFrame pour affichage
                                        stats_df = pd.DataFrame(key_stats, columns=['Statistique', red_match, blue_match])
                                        st.dataframe(stats_df, use_container_width=True, hide_index=True)
                                        
                                        # Lien pour plus de détails
                                        st.caption("Utilisez l'onglet Prédiction pour une analyse complète")
                    
                    # Ajouter la section de stratégie de paris si les prédictions sont générées
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
        # État vide
        st.info("Aucun événement chargé. Utilisez le bouton ci-dessus pour récupérer les prochains événements UFC.")

# PARTIE 10

def show_bankroll_page():
    """Affiche la page de gestion de bankroll avec une interface améliorée et synchronisation GitHub"""
    # AMÉLIORATION UI: Titre de page avec animation
    st.markdown("""
    <div class="section-fade-in" style="text-align:center; margin-bottom: 25px;">
        <h2>💰 Gestion de bankroll et paris</h2>
        <p style="color: #aaa;">Suivez vos performances et gérez vos paris</p>
    </div>
    """, unsafe_allow_html=True)
    
    # AMÉLIORATION UI: Affichage de la bankroll actuelle dans une card moderne
    st.markdown(f"""
    <div class="card" style="text-align: center; margin-bottom: 30px;">
        <div style="color: #aaa; margin-bottom: 5px;">Bankroll actuelle</div>
        <div style="font-size: 2.5rem; font-weight: 700;">{app_data['current_bankroll']:.2f} €</div>
    </div>
    """, unsafe_allow_html=True)
    
    # AMÉLIORATION UI: Tabs modernes pour organiser les fonctionnalités
    bankroll_tabs = st.tabs(["💼 Ajuster la bankroll", "➕ Ajouter un pari", "⚙️ Paramètres"])
    
    # Tab 1: Ajuster la bankroll
    with bankroll_tabs[0]:
        st.subheader("Ajuster la bankroll")
        
        # AMÉLIORATION UI: Layout à deux colonnes
        adjust_cols = st.columns([2, 1, 1])
        
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
        
        # AMÉLIORATION UI: Informations contextuelles sur l'ajustement
        if adjustment_type == "Dépôt":
            new_amount = app_data['current_bankroll'] + adjustment_amount
            operation = "+"
            color = "#4CAF50"
        elif adjustment_type == "Retrait":
            new_amount = max(0, app_data['current_bankroll'] - adjustment_amount)
            operation = "-" if adjustment_amount <= app_data['current_bankroll'] else "!"
            color = "#F44336" if adjustment_amount > app_data['current_bankroll'] else "#FFC107"
        else:  # "Définir montant exact"
            new_amount = adjustment_amount
            operation = "="
            color = "#1E88E5"
        
        # AMÉLIORATION UI: Résumé de l'opération avant validation
        if adjustment_amount > 0:
            st.markdown(f"""
            <div class="card" style="background: linear-gradient(145deg, rgba({
                '76, 175, 80' if operation == '+' else
                '244, 67, 54' if operation == '!' else
                '255, 193, 7' if operation == '-' else
                '30, 136, 229'
            }, 0.1) 0%, rgba({
                '56, 142, 60' if operation == '+' else
                '211, 47, 47' if operation == '!' else
                '255, 160, 0' if operation == '-' else
                '21, 101, 192'
            }, 0.1) 100%); margin: 15px 0;">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div>
                        <div style="font-weight: 500;">Opération</div>
                        <div style="font-size: 1.2rem; font-weight: 600; color: {color};">
                            {adjustment_type}
                            {f" ({operation} {adjustment_amount:.2f} €)" if operation != "=" else f" → {adjustment_amount:.2f} €"}
                        </div>
                    </div>
                    <div>
                        <div style="font-weight: 500;">Nouvelle bankroll</div>
                        <div style="font-size: 1.2rem; font-weight: 600;">{new_amount:.2f} €</div>
                    </div>
                </div>
                {f'<div style="margin-top: 10px; color: #F44336; font-weight: 500;">⚠️ Montant de retrait supérieur à la bankroll!</div>' if operation == '!' else ''}
            </div>
            """, unsafe_allow_html=True)
        
        # Bouton de validation
        if st.button("Valider l'ajustement", type="primary", key="validate_bankroll_adjust", use_container_width=True):
            # Calculer la nouvelle bankroll
            if adjustment_type == "Dépôt":
                new_bankroll = app_data['current_bankroll'] + adjustment_amount
                action = "deposit"
                if not adjustment_note:
                    adjustment_note = "Dépôt"
            elif adjustment_type == "Retrait":
                if adjustment_amount > app_data['current_bankroll']:
                    st.error(f"⚠️ Montant du retrait ({adjustment_amount:.2f} €) supérieur à la bankroll actuelle ({app_data['current_bankroll']:.2f} €)")
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
                # Animation de chargement
                with st.spinner("Mise à jour de la bankroll..."):
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
                    
                    # AJOUT GITHUB: Synchroniser avec GitHub
                    sync_success = sync_to_github("bets/bankroll.csv", bankroll_df, f"Update bankroll: {action} {adjustment_amount}€")
                    
                    # Mettre à jour app_data
                    app_data['current_bankroll'] = new_bankroll
                
                # AMÉLIORATION UI: Confirmation améliorée avec info sur la synchro GitHub
                if sync_success:
                    st.success(f"✅ Bankroll mise à jour: {new_bankroll:.2f} € (synchronisé avec GitHub)")
                else:
                    st.warning(f"✅ Bankroll mise à jour: {new_bankroll:.2f} € (synchronisation GitHub échouée)")
                
                # AMÉLIORATION UI: Card mise à jour
                st.markdown(f"""
                <div class="card section-fade-in" style="text-align: center; margin: 20px 0; background: linear-gradient(145deg, rgba(76, 175, 80, 0.1) 0%, rgba(56, 142, 60, 0.1) 100%); 
                                                                         border: 1px solid rgba(76, 175, 80, 0.3);">
                    <div style="font-size: 3rem; margin-bottom: 10px;">✅</div>
                    <h3 style="margin-bottom: 15px; color: #4CAF50;">Bankroll mise à jour</h3>
                    <div style="font-size: 2rem; font-weight: 700;">{new_bankroll:.2f} €</div>
                    <div style="margin-top: 10px; color: rgba(255,255,255,0.6);">
                        {adjustment_note}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Tab 2: Ajouter un pari
    with bankroll_tabs[1]:
        st.subheader("Ajouter un pari manuellement")
        
        # AMÉLIORATION UI: Interface d'ajout de pari réorganisée
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
        
        # Mode de saisie de la cote (manuel vs slider)
        with bet_cols[1]:
            manual_odds_mode = st.radio(
                "Mode de saisie de la cote",
                options=["Manuel", "Slider"],
                index=0,  # Manuel par défaut
                key="manual_odds_mode"
            )
            
            if manual_odds_mode == "Manuel":
                manual_odds = st.number_input("Cote", min_value=1.01, value=2.0, step=0.01, format="%.2f", key="manual_odds_manual")
            else:
                manual_odds = st.slider("Cote", min_value=1.01, max_value=10.0, value=2.0, step=0.05, format="%.2f", key="manual_odds_slider")
        
        with bet_cols[2]:
            # Mode de saisie de la mise (manuel vs slider)
            manual_stake_mode = st.radio(
                "Mode de saisie de la mise",
                options=["Manuel", "Slider"],
                index=0,  # Manuel par défaut
                key="manual_stake_mode"
            )
            
            if manual_stake_mode == "Manuel":
                manual_stake = st.number_input(
                    "Mise (€)",
                    min_value=0.0, 
                    max_value=float(app_data['current_bankroll']),
                    value=min(50.0, float(app_data['current_bankroll'])),
                    step=5.0,
                    key="manual_stake_manual"
                )
            else:
                # AMÉLIORATION UI: Utiliser un slider pour la mise
                manual_stake = st.slider(
                    "Mise (€)",
                    min_value=0.0, 
                    max_value=float(app_data['current_bankroll']),
                    value=min(50.0, float(app_data['current_bankroll'])),
                    step=5.0,
                    key="manual_stake_slider"
                )
        
        # AMÉLIORATION UI: Calculer et afficher le gain potentiel
        potential_profit = manual_stake * (manual_odds - 1)
        roi_pct = (potential_profit / manual_stake) * 100 if manual_stake > 0 else 0
        
        # Afficher les métriques en 3 colonnes
        metrics_cols = st.columns(3)
        with metrics_cols[0]:
            st.metric("Mise", f"{manual_stake:.2f}€")
        with metrics_cols[1]:
            st.metric("Gain potentiel", f"{potential_profit:.2f}€")
        with metrics_cols[2]:
            st.metric("ROI", f"{roi_pct:.1f}%")
        
        # AMÉLIORATION UI: Information sur la bankroll
        st.markdown(f"""
        <div style="margin: 10px 0; font-size: 0.9rem; color: #aaa; text-align: right;">
            <i>% de bankroll: {manual_stake/app_data['current_bankroll']*100:.1f}%</i>
        </div>
        """, unsafe_allow_html=True)
        
        # Bouton pour enregistrer le pari
        if st.button("💾 Enregistrer le pari", type="primary", key="save_manual_bet_btn", use_container_width=True):
            if manual_stake > app_data['current_bankroll']:
                st.error(f"⚠️ Mise ({manual_stake:.2f} €) supérieure à la bankroll actuelle ({app_data['current_bankroll']:.2f} €)")
            elif manual_stake <= 0:
                st.error("⚠️ La mise doit être supérieure à 0 €")
            else:
                # Animation de chargement
                with st.spinner("Enregistrement du pari..."):
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
                        # AMÉLIORATION UI: Confirmation plus visuelle
                        st.success(f"✅ Pari enregistré avec succès! Mise de {manual_stake:.2f} € sur {manual_pick} @ {manual_odds:.2f}")
                        
                        # AMÉLIORATION UI: Carte de confirmation
                        st.markdown(f"""
                        <div class="card section-fade-in" style="background: linear-gradient(145deg, rgba(76, 175, 80, 0.1) 0%, rgba(56, 142, 60, 0.1) 100%); 
                                                                 border: 1px solid rgba(76, 175, 80, 0.3); text-align: center;">
                            <div style="font-size: 3rem; margin-bottom: 10px;">✅</div>
                            <h3 style="margin-bottom: 15px; color: #4CAF50;">Pari enregistré avec succès</h3>
                            <div style="font-weight: 600; font-size: 1.2rem; margin-bottom: 15px;">
                                {manual_event_name} - {manual_event_date}
                            </div>
                            <div style="font-weight: 600; font-size: 1.1rem;">{manual_pick} @ {manual_odds:.2f}</div>
                            <div style="display: flex; justify-content: space-between; margin: 15px 0; color: rgba(255,255,255,0.8);">
                                <div>Mise: <b>{manual_stake:.2f}€</b></div>
                                <div>Gain potentiel: <b>{potential_profit:.2f}€</b></div>
                                <div>ROI: <b>{roi_pct:.1f}%</b></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("❌ Erreur lors de l'enregistrement du pari.")
    
    # Tab 3: Paramètres
    with bankroll_tabs[2]:
        st.subheader("Paramètres de gestion de bankroll")
        
        # AMÉLIORATION UI: Paramètres dans une card moderne
        st.markdown("""
        <div class="card">
            <h4 style="margin-top: 0;">Stratégie Kelly par défaut</h4>
            <p>Choisissez la fraction Kelly à utiliser par défaut pour les recommandations de paris</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sélection de la stratégie Kelly par défaut
        kelly_strategy = st.selectbox(
            "Stratégie Kelly par défaut",
            options=["Kelly pur", "Kelly/2", "Kelly/3", "Kelly/4", "Kelly/5", "Kelly/10"],
            index=3,  # Kelly/4 par défaut
            key="default_kelly_strategy"
        )
        st.session_state.kelly_strategy = kelly_strategy
        
        # AMÉLIORATION UI: Information sur les stratégies Kelly
        st.markdown("""
        <div class="info-box">
            <h4 style="margin-top: 0;">🧮 À propos des stratégies Kelly</h4>
            <p><b>Kelly pur</b>: Mise optimale théorique, mais peut être risquée</p>
            <p><b>Kelly fractionné</b>: Version plus conservatrice (Kelly/2, Kelly/4...) qui réduit la variance</p>
            <p>Pour la plupart des parieurs, une stratégie Kelly/4 ou Kelly/5 offre un bon équilibre entre rendement et gestion du risque.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Paramètre de bankroll minimum
        st.markdown("""
        <div class="card" style="margin-top: 20px;">
            <h4 style="margin-top: 0;">Alerte de bankroll minimum</h4>
            <p>Définissez une limite en dessous de laquelle vous recevrez des alertes</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sélection de la bankroll minimum
        min_bankroll = st.number_input(
            "Bankroll minimum (€)",
            min_value=0.0,
            max_value=float(app_data['current_bankroll']),
            value=float(app_data['current_bankroll']) * 0.5,  # 50% par défaut
            step=10.0,
            format="%.2f",
            key="min_bankroll"
        )
        
        # AMÉLIORATION UI: Affichage du statut actuel
        if app_data['current_bankroll'] <= min_bankroll:
            st.markdown("""
            <div style="background-color: rgba(244, 67, 54, 0.1); padding: 10px; border-radius: 8px; margin-top: 10px; border-left: 3px solid #F44336;">
                <b style="color: #F44336;">⚠️ Attention:</b> Votre bankroll actuelle est en dessous ou égale à votre limite d'alerte.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background-color: rgba(76, 175, 80, 0.1); padding: 10px; border-radius: 8px; margin-top: 10px; border-left: 3px solid #4CAF50;">
                <b style="color: #4CAF50;">✅ Statut:</b> Votre bankroll actuelle est au-dessus de votre limite d'alerte.
            </div>
            """, unsafe_allow_html=True)

def show_history_page():
    """Affiche la page d'historique des paris avec une interface modernisée et synchronisation GitHub"""
    # AMÉLIORATION UI: Titre de page avec animation
    st.markdown("""
    <div class="section-fade-in" style="text-align:center; margin-bottom: 25px;">
        <h2>📊 Historique des paris et performances</h2>
        <p style="color: #aaa;">Analysez vos résultats et optimisez votre stratégie</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Vérifier si les fichiers existent
    bets_file = os.path.join("bets", "bets.csv")
    bankroll_file = os.path.join("bets", "bankroll.csv")
    has_bets = os.path.exists(bets_file)
    has_bankroll = os.path.exists(bankroll_file)
    
    # AJOUT GITHUB: Essayer de charger depuis GitHub si les fichiers locaux n'existent pas
    if not has_bets or not has_bankroll:
        with st.spinner("Vérification des données sur GitHub..."):
            if not has_bets:
                # Essayer de charger les paris depuis GitHub
                github_content, sha = read_github_file("bets/bets.csv")
                if github_content:
                    if not os.path.exists("bets"):
                        os.makedirs("bets")
                    with open(bets_file, 'w') as f:
                        f.write(github_content)
                    has_bets = True
                    st.info("📥 Données de paris chargées depuis GitHub")
            
            if not has_bankroll:
                # Essayer de charger la bankroll depuis GitHub
                github_content, sha = read_github_file("bets/bankroll.csv")
                if github_content:
                    if not os.path.exists("bets"):
                        os.makedirs("bets")
                    with open(bankroll_file, 'w') as f:
                        f.write(github_content)
                    has_bankroll = True
                    st.info("📥 Données de bankroll chargées depuis GitHub")
    
    if has_bets and has_bankroll:
        # Animation de chargement
        with st.spinner("Chargement des données d'historique..."):
            bets_df = pd.read_csv(bets_file)
            bankroll_df = pd.read_csv(bankroll_file)
        
        # AMÉLIORATION UI: Graph d'évolution de la bankroll modernisé
        if not bankroll_df.empty:
            st.subheader("Évolution de la bankroll")
            
            # AMÉLIORATION UI: Graphique modernisé
            bankroll_df['date'] = pd.to_datetime(bankroll_df['date'])
            
            fig = px.line(
                bankroll_df, 
                x="date", 
                y="amount",
                title=None,
                labels={"amount": "Bankroll (€)", "date": "Date"}
            )
            
            # Amélioration du style du graphique
            fig.update_traces(
                line=dict(width=3, color='#4CAF50'),
                mode='lines+markers',
                marker=dict(size=8, color='#4CAF50')
            )
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(
                    title="Date",
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)',
                    tickformat='%d %b %Y'
                ),
                yaxis=dict(
                    title="Bankroll (€)",
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)',
                ),
                margin=dict(l=40, r=40, t=40, b=40),
                hovermode='x unified'
            )
            
            # Ajouter des annotations pour les points importants
            if len(bankroll_df) > 1:
                # Point initial
                fig.add_annotation(
                    x=bankroll_df['date'].iloc[0],
                    y=bankroll_df['amount'].iloc[0],
                    text="Initial",
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=-40,
                    font=dict(color="#90EE90", size=16, family="Arial Black") 
                )
                
                # Point final
                fig.add_annotation(
                    x=bankroll_df['date'].iloc[-1],
                    y=bankroll_df['amount'].iloc[-1],
                    text="Actuel",
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=-40,
                    font=dict(color="#90EE90", size=16, family="Arial Black") 

                )
                
                # Point haut
                max_idx = bankroll_df['amount'].idxmax()
                if max_idx != 0 and max_idx != len(bankroll_df) - 1:
                    fig.add_annotation(
                        x=bankroll_df['date'].iloc[max_idx],
                        y=bankroll_df['amount'].iloc[max_idx],
                        text="Max",
                        showarrow=True,
                        arrowhead=1,
                        ax=0,
                        ay=-40,
                        font=dict(color="#90EE90", size=16, family="Arial Black") 

                    )
                
                # Point bas (après le début)
                if len(bankroll_df) > 2:
                    min_data = bankroll_df.iloc[1:] 
                    min_idx = min_data['amount'].idxmin()
                    if min_idx != len(bankroll_df) - 1:
                        fig.add_annotation(
                            x=bankroll_df['date'].iloc[min_idx],
                            y=bankroll_df['amount'].iloc[min_idx],
                            text="Min",
                            showarrow=True,
                            arrowhead=1,
                            ax=0,
                            ay=40,
                            font=dict(color="#90EE90", size=16, family="Arial Black") 
                        )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Résumé des performances
        if not bets_df.empty:
            # Obtenir les statistiques
            betting_stats = get_betting_summary(bets_df)
            
            # AMÉLIORATION UI: Dashboard de métriques principales
            st.subheader("Tableau de bord des performances")
            
            # Top row: 4 main metrics
            metrics_row1 = st.columns(4)
            with metrics_row1[0]:
                st.metric("Total des paris", f"{betting_stats['total_bets']}")
            with metrics_row1[1]:
                st.metric("Paris en cours", f"{betting_stats['open_bets']}")
            with metrics_row1[2]:
                st.metric("Victoires/Défaites", f"{betting_stats['wins']}/{betting_stats['losses']}")
            with metrics_row1[3]:
                st.metric(
                    "Taux de réussite", 
                    f"{betting_stats['win_rate']:.1f}%",
                    delta=None
                )
            
            # Second row: Financial metrics
            metrics_row2 = st.columns(4)
            with metrics_row2[0]:
                st.metric(
                    "Total misé", 
                    f"{betting_stats['total_staked']:.2f} €",
                    delta=None
                )
            with metrics_row2[1]:
                st.metric(
                    "Profit total", 
                    f"{betting_stats['total_profit']:.2f} €",
                    delta=f"{betting_stats['roi']:.1f}%" if betting_stats['total_profit'] != 0 else None,
                    delta_color="normal"
                )
            with metrics_row2[2]:
                avg_stake = betting_stats['total_staked'] / max(betting_stats['total_bets'], 1)
                st.metric("Mise moyenne", f"{avg_stake:.2f} €")
            with metrics_row2[3]:
                avg_odds = betting_stats['avg_odds']
                st.metric("Cote moyenne", f"{avg_odds:.2f}")
            
            # AMÉLIORATION UI: Carte de streaks
            streak_cols = st.columns(2)
            with streak_cols[0]:
                # Streak actuelle
                streak_type = betting_stats['current_streak_type']
                streak_count = betting_stats['current_streak']
                
                if streak_count > 0:
                    streak_color = "#4CAF50" if streak_type == "win" else "#F44336"
                    streak_text = "victoires" if streak_type == "win" else "défaites"
                    
                    st.markdown(f"""
                    <div class="card" style="background: linear-gradient(145deg, rgba({
                        '76, 175, 80' if streak_type == 'win' else '244, 67, 54'
                    }, 0.1) 0%, rgba({
                        '56, 142, 60' if streak_type == 'win' else '211, 47, 47'
                    }, 0.1) 100%); text-align: center;">
                        <div style="color: #aaa; margin-bottom: 5px;">Série actuelle</div>
                        <div style="font-size: 2rem; font-weight: 700; color: {streak_color};">{streak_count} {streak_text}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="card" style="text-align: center;">
                        <div style="color: #aaa; margin-bottom: 5px;">Série actuelle</div>
                        <div style="font-size: 1.2rem; font-weight: 500;">Aucune série en cours</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with streak_cols[1]:
                # Meilleure série
                best_streak = betting_stats['longest_win_streak']
                
                if best_streak > 0:
                    st.markdown(f"""
                    <div class="card" style="background: linear-gradient(145deg, rgba(76, 175, 80, 0.1) 0%, rgba(56, 142, 60, 0.1) 100%); text-align: center;">
                        <div style="color: #aaa; margin-bottom: 5px;">Meilleure série</div>
                        <div style="font-size: 2rem; font-weight: 700; color: #4CAF50;">{best_streak} victoires</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="card" style="text-align: center;">
                        <div style="color: #aaa; margin-bottom: 5px;">Meilleure série</div>
                        <div style="font-size: 1.2rem; font-weight: 500;">Aucune série enregistrée</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # AMÉLIORATION UI: Utiliser des onglets pour organiser les paris
            bet_subtabs = st.tabs(["🎲 Paris en cours", "📜 Historique des paris", "✏️ Modifier/Supprimer"])
            
            # Section des paris en cours
            with bet_subtabs[0]:
                st.subheader("Paris en cours")
                open_bets = bets_df[bets_df["status"] == "open"]
                if not open_bets.empty:
                    # AMÉLIORATION UI: Formater le DataFrame pour un affichage modernisé
                    display_open_bets = open_bets.copy()
                    display_open_bets['gain_potentiel'] = display_open_bets.apply(lambda row: row['stake'] * (row['odds'] - 1), axis=1)
                    
                    # Sélectionner et renommer les colonnes
                    display_open_bets = display_open_bets[["bet_id", "event_name", "event_date", "fighter_red", "fighter_blue", "pick", "odds", "stake", "gain_potentiel"]]
                    display_open_bets.columns = ["ID", "Événement", "Date", "Rouge", "Bleu", "Pari sur", "Cote", "Mise (€)", "Gain potentiel (€)"]
                    
                    # AMÉLIORATION UI: Dataframe avec formatage amélioré
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
                        },
                        hide_index=True
                    )
                    
                    # AMÉLIORATION UI: Résumé des paris en cours
                    total_stake = display_open_bets["Mise (€)"].sum()
                    total_potential = display_open_bets["Gain potentiel (€)"].sum()
                    avg_odds = display_open_bets["Cote"].mean()
                    
                    st.markdown(f"""
                    <div class="card" style="margin-top: 15px; background: linear-gradient(145deg, rgba(30, 136, 229, 0.1) 0%, rgba(21, 101, 192, 0.1) 100%);
                                         border-left: 3px solid #1E88E5;">
                        <h4 style="margin-top: 0; color: #1E88E5;">Résumé des paris en cours</h4>
                        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">
                            <div>
                                <div style="color: #aaa; font-size: 0.9rem;">Total misé</div>
                                <div style="font-size: 1.1rem; font-weight: 600;">{total_stake:.2f} €</div>
                            </div>
                            <div>
                                <div style="color: #aaa; font-size: 0.9rem;">Gain potentiel</div>
                                <div style="font-size: 1.1rem; font-weight: 600;">{total_potential:.2f} €</div>
                            </div>
                            <div>
                                <div style="color: #aaa; font-size: 0.9rem;">Cote moyenne</div>
                                <div style="font-size: 1.1rem; font-weight: 600;">{avg_odds:.2f}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # AMÉLIORATION UI: Message d'état vide stylisé
                    st.markdown("""
                    <div class="card" style="text-align: center; padding: 30px 20px;">
                        <div style="font-size: 3rem; margin-bottom: 15px;">🎲</div>
                        <h3 style="margin-bottom: 15px;">Aucun pari en cours</h3>
                        <p>Utilisez l'onglet "Prédiction" ou "Événements à venir" pour placer de nouveaux paris.</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Section historique des paris
            with bet_subtabs[1]:
                st.subheader("Historique des paris")
                closed_bets = bets_df[bets_df["status"] == "closed"]
                if not closed_bets.empty:
                    # AMÉLIORATION UI: Formater le DataFrame pour l'affichage avec des icônes
                    display_closed_bets = closed_bets.copy()
                    
                    # Ajouter une colonne pour formater le résultat avec des icônes
                    def format_result(result):
                        if result == "win":
                            return "✅ Victoire"
                        elif result == "loss":
                            return "❌ Défaite"
                        elif result == "void":
                            return "⚪ Annulé"
                        else:
                            return result
                    
                    display_closed_bets['formatted_result'] = display_closed_bets['result'].apply(format_result)
                    
                    # Sélectionner et renommer les colonnes
                    display_closed_bets = display_closed_bets[["bet_id", "event_name", "event_date", "fighter_red", "fighter_blue", "pick", "odds", "stake", "formatted_result", "profit", "roi"]]
                    display_closed_bets.columns = ["ID", "Événement", "Date", "Rouge", "Bleu", "Pari sur", "Cote", "Mise (€)", "Résultat", "Profit (€)", "ROI (%)"]
                    
                    # AMÉLIORATION UI: Option de filtrage
                    filter_col1, filter_col2 = st.columns(2)
                    
                    with filter_col1:
                        result_filter = st.multiselect(
                            "Filtrer par résultat",
                            options=["✅ Victoire", "❌ Défaite", "⚪ Annulé"],
                            default=["✅ Victoire", "❌ Défaite", "⚪ Annulé"],
                            key="result_filter"
                        )
                    
                    with filter_col2:
                        sort_by = st.selectbox(
                            "Trier par",
                            options=["Date (récent → ancien)", "Date (ancien → récent)", "ROI (élevé → bas)", "Profit (élevé → bas)"],
                            index=0,
                            key="sort_by"
                        )
                    
                    # Appliquer les filtres
                    if result_filter:
                        display_closed_bets = display_closed_bets[display_closed_bets["Résultat"].isin(result_filter)]
                    
                    # Appliquer le tri
                    if sort_by == "Date (récent → ancien)":
                        display_closed_bets = display_closed_bets.sort_values("Date", ascending=False)
                    elif sort_by == "Date (ancien → récent)":
                        display_closed_bets = display_closed_bets.sort_values("Date", ascending=True)
                    elif sort_by == "ROI (élevé → bas)":
                        display_closed_bets = display_closed_bets.sort_values("ROI (%)", ascending=False)
                    elif sort_by == "Profit (élevé → bas)":
                        display_closed_bets = display_closed_bets.sort_values("Profit (€)", ascending=False)
                    
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
                        },
                        hide_index=True
                    )
                    
                    # AMÉLIORATION UI: Graphiques d'analyse
                    st.subheader("Analyse des résultats")
                    
                    analysis_tabs = st.tabs(["📊 Répartition", "📈 Performance", "🔍 Détails"])
                    
                    # Onglet Répartition
                    with analysis_tabs[0]:
                        chart_cols = st.columns(2)
                        
                        with chart_cols[0]:
                            # Répartition des résultats en pie chart
                            result_counts = closed_bets["result"].value_counts().reset_index()
                            result_counts.columns = ["result", "count"]
                            
                            # Remplacer les valeurs pour l'affichage
                            result_counts["result"] = result_counts["result"].replace({
                                "win": "Victoire",
                                "loss": "Défaite",
                                "void": "Annulé"
                            })
                            
                            fig_results = px.pie(
                                result_counts, 
                                values="count", 
                                names="result",
                                title="Répartition des résultats",
                                color="result",
                                color_discrete_map={
                                    "Victoire": "#4CAF50",
                                    "Défaite": "#F44336",
                                    "Annulé": "#9E9E9E"
                                }
                            )
                            
                            fig_results.update_layout(
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='white'),
                                margin=dict(l=20, r=20, t=30, b=20),
                                legend=dict(orientation="h", y=-0.1)
                            )
                            
                            st.plotly_chart(fig_results, use_container_width=True)
                        
                        with chart_cols[1]:
                            # Répartition des mises et profits
                            # Créer un DataFrame résumé
                            summary_df = pd.DataFrame({
                                "Catégorie": ["Mises", "Profits", "Pertes"],
                                "Montant": [
                                    closed_bets["stake"].sum(),
                                    closed_bets[closed_bets["profit"] > 0]["profit"].sum(),
                                    abs(closed_bets[closed_bets["profit"] < 0]["profit"].sum())
                                ]
                            })
                            
                            fig_finances = px.bar(
                                summary_df,
                                x="Catégorie",
                                y="Montant",
                                title="Répartition financière",
                                color="Catégorie",
                                color_discrete_map={
                                    "Mises": "#1E88E5",
                                    "Profits": "#4CAF50",
                                    "Pertes": "#F44336"
                                }
                            )
                            
                            fig_finances.update_layout(
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='white'),
                                margin=dict(l=20, r=20, t=30, b=20),
                                showlegend=False,
                                xaxis=dict(
                                    title=None,
                                    showgrid=False
                                ),
                                yaxis=dict(
                                    title="Montant (€)",
                                    showgrid=True,
                                    gridcolor='rgba(255,255,255,0.1)'
                                )
                            )
                            
                            st.plotly_chart(fig_finances, use_container_width=True)
                    
                    # Onglet Performance
                    with analysis_tabs[1]:
                        # Convertir la date pour le graphique
                        perf_df = display_closed_bets.copy()
                        perf_df["Date"] = pd.to_datetime(perf_df["Date"])
                        perf_df = perf_df.sort_values("Date")
                        
                        # Calculer le profit cumulatif
                        perf_df["profit_cumul"] = perf_df["Profit (€)"].cumsum()
                        
                        # Créer le graphique d'évolution du profit
                        fig_profit = px.line(
                            perf_df,
                            x="Date",
                            y="profit_cumul",
                            title="Évolution du profit cumulé",
                            markers=True
                        )
                        
                        fig_profit.update_traces(
                            line=dict(width=3, color='#4CAF50'),
                            marker=dict(size=8, color='#4CAF50')
                        )
                        
                        fig_profit.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'),
                            margin=dict(l=20, r=20, t=30, b=20),
                            xaxis=dict(
                                title="Date",
                                showgrid=True,
                                gridcolor='rgba(255,255,255,0.1)'
                            ),
                            yaxis=dict(
                                title="Profit cumulé (€)",
                                showgrid=True,
                                gridcolor='rgba(255,255,255,0.1)'
                            )
                        )
                        
                        # Ajouter une ligne horizontale à zéro
                        fig_profit.add_shape(
                            type="line",
                            x0=perf_df["Date"].min(),
                            x1=perf_df["Date"].max(),
                            y0=0,
                            y1=0,
                            line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dash")
                        )
                        
                        st.plotly_chart(fig_profit, use_container_width=True)
                        
                        # Graphique ROI par cote
                        roi_by_odds = px.scatter(
                            display_closed_bets,
                            x="Cote",
                            y="ROI (%)",
                            title="ROI par rapport à la cote",
                            color="Résultat",
                            color_discrete_map={
                                "✅ Victoire": "#4CAF50",
                                "❌ Défaite": "#F44336",
                                "⚪ Annulé": "#9E9E9E"
                            },
                            hover_data=["Événement", "Pari sur", "Mise (€)"]
                        )
                        
                        roi_by_odds.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'),
                            margin=dict(l=20, r=20, t=30, b=20),
                            xaxis=dict(
                                title="Cote",
                                showgrid=True,
                                gridcolor='rgba(255,255,255,0.1)'
                            ),
                            yaxis=dict(
                                title="ROI (%)",
                                showgrid=True,
                                gridcolor='rgba(255,255,255,0.1)'
                            )
                        )
                        
                        # Ajouter une ligne horizontale à zéro
                        roi_by_odds.add_shape(
                            type="line",
                            x0=display_closed_bets["Cote"].min(),
                            x1=display_closed_bets["Cote"].max(),
                            y0=0,
                            y1=0,
                            line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dash")
                        )
                        
                        st.plotly_chart(roi_by_odds, use_container_width=True)
                    
                    # Onglet Détails
                    with analysis_tabs[2]:
                        # Métrique par mois
                        if not display_closed_bets.empty and "Date" in display_closed_bets.columns:
                            # Convertir la date au format datetime si ce n'est pas déjà fait
                            display_closed_bets["Date"] = pd.to_datetime(display_closed_bets["Date"])
                            display_closed_bets["Mois"] = display_closed_bets["Date"].dt.strftime("%Y-%m")
                            
                            monthly_stats = display_closed_bets.groupby("Mois").agg({
                                "ID": "count",
                                "Profit (€)": "sum",
                                "Mise (€)": "sum"
                            }).reset_index()
                            
                            # Calculer le ROI mensuel
                            monthly_stats["ROI (%)"] = monthly_stats["Profit (€)"] / monthly_stats["Mise (€)"] * 100
                            monthly_stats.columns = ["Mois", "Nombre de paris", "Profit (€)", "Total misé (€)", "ROI (%)"]
                            
                            st.subheader("Performance par mois")
                            
                            st.dataframe(
                                monthly_stats,
                                use_container_width=True,
                                column_config={
                                    "Mois": st.column_config.TextColumn("Mois"),
                                    "Nombre de paris": st.column_config.NumberColumn("Nombre de paris", format="%d"),
                                    "Profit (€)": st.column_config.NumberColumn("Profit (€)", format="%.2f"),
                                    "Total misé (€)": st.column_config.NumberColumn("Total misé (€)", format="%.2f"),
                                    "ROI (%)": st.column_config.NumberColumn("ROI (%)", format="%.1f")
                                },
                                hide_index=True
                            )
                            
                            # Graphique d'évolution mensuelle
                            fig_monthly = px.bar(
                                monthly_stats,
                                x="Mois",
                                y="Profit (€)",
                                title="Profit mensuel",
                                color="ROI (%)",
                                color_continuous_scale=["#F44336", "#FFEB3B", "#4CAF50"],
                                text="ROI (%)"
                            )
                            
                            fig_monthly.update_layout(
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='white'),
                                margin=dict(l=20, r=20, t=30, b=20),
                                xaxis=dict(
                                    title=None,
                                    showgrid=False
                                ),
                                yaxis=dict(
                                    title="Profit (€)",
                                    showgrid=True,
                                    gridcolor='rgba(255,255,255,0.1)'
                                )
                            )
                            
                            fig_monthly.update_traces(
                                texttemplate="%{text:.1f}%",
                                textposition="outside"
                            )
                            
                            st.plotly_chart(fig_monthly, use_container_width=True)
                else:
                    # AMÉLIORATION UI: Message d'état vide stylisé
                    st.markdown("""
                    <div class="card" style="text-align: center; padding: 30px 20px;">
                        <div style="font-size: 3rem; margin-bottom: 15px;">📜</div>
                        <h3 style="margin-bottom: 15px;">Aucun pari dans l'historique</h3>
                        <p>Commencez à parier pour voir apparaître votre historique ici.</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Section de gestion des paris
            with bet_subtabs[2]:
                st.subheader("Gérer les paris")
                
                # AMÉLIORATION UI: Interface à deux colonnes
                manage_columns = st.columns(2)
                
                # Colonne pour mettre à jour les paris
                with manage_columns[0]:
                    st.markdown("""
                    <div class="card">
                        <h3 style="margin-top: 0;">Mettre à jour un pari</h3>
                        <p>Entrez le résultat d'un pari en cours</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
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
                        
                        # AMÉLIORATION UI: Affichage amélioré des détails du pari
                        st.markdown(f"""
                        <div class="card" style="background: linear-gradient(145deg, rgba(30, 136, 229, 0.1) 0%, rgba(21, 101, 192, 0.1) 100%);">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                                <div style="font-weight: 500;">Pari #{update_bet_id}</div>
                                <div style="font-weight: 500;">{selected_bet['event_name']}</div>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                                <div>Pick</div>
                                <div style="font-weight: 600;">{selected_bet['pick']}</div>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                                <div>Cote</div>
                                <div style="font-weight: 600;">{selected_bet['odds']}</div>
                            </div>
                            <div style="display: flex; justify-content: space-between;">
                                <div>Mise</div>
                                <div style="font-weight: 600;">{selected_bet['stake']}€</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Sélectionner le résultat avec une interface visuelle améliorée
                        st.markdown("### Sélectionner le résultat")
                        result_cols = st.columns(3)
                        
                        with result_cols[0]:
                            win_clicked = st.button(
                                "✅ Victoire", 
                                key="win_result_btn",
                                use_container_width=True,
                                type="primary"
                            )
                        
                        with result_cols[1]:
                            loss_clicked = st.button(
                                "❌ Défaite", 
                                key="loss_result_btn",
                                use_container_width=True
                            )
                        
                        with result_cols[2]:
                            void_clicked = st.button(
                                "⚪ Annulé", 
                                key="void_result_btn",
                                use_container_width=True
                            )
                        
                        # Traitement des clics
                        if win_clicked or loss_clicked or void_clicked:
                            result = "win" if win_clicked else "loss" if loss_clicked else "void"
                            
                            # Animation de chargement
                            with st.spinner(f"Mise à jour du pari #{update_bet_id}..."):
                                # Mettre à jour le pari
                                new_bankroll = update_bet_result(update_bet_id, result, app_data['current_bankroll'])
                                
                                # Mettre à jour la bankroll dans app_data
                                app_data['current_bankroll'] = new_bankroll
                                
                                # AMÉLIORATION UI: Message de confirmation dynamique
                                result_icon = "✅" if result == "win" else "❌" if result == "loss" else "⚪"
                                result_text = "Victoire" if result == "win" else "Défaite" if result == "loss" else "Annulé"
                                result_color = "#4CAF50" if result == "win" else "#F44336" if result == "loss" else "#9E9E9E"
                                
                                # Calcul du profit
                                if result == "win":
                                    profit = selected_bet['stake'] * (selected_bet['odds'] - 1)
                                    msg = f"Gain de {profit:.2f}€"
                                elif result == "loss":
                                    profit = -selected_bet['stake']
                                    msg = f"Perte de {abs(profit):.2f}€"
                                else:
                                    profit = 0
                                    msg = "Mise remboursée"
                                
                                # AMÉLIORATION UI: Carte de confirmation
                                st.markdown(f"""
                                <div class="card section-fade-in" style="background: linear-gradient(145deg, rgba({
                                    '76, 175, 80' if result == 'win' else
                                    '244, 67, 54' if result == 'loss' else
                                    '158, 158, 158'
                                }, 0.1) 0%, rgba({
                                    '56, 142, 60' if result == 'win' else
                                    '211, 47, 47' if result == 'loss' else
                                    '117, 117, 117'
                                }, 0.1) 100%); text-align: center;">
                                    <div style="font-size: 3rem; margin-bottom: 10px;">{result_icon}</div>
                                    <h3 style="margin-bottom: 15px; color: {result_color};">{result_text}</h3>
                                    <div style="font-weight: 600; font-size: 1.2rem; margin-bottom: 15px;">
                                        {selected_bet['pick']} @ {selected_bet['odds']}
                                    </div>
                                    <div style="font-weight: 500; font-size: 1.1rem;">
                                        {msg}
                                    </div>
                                    <div style="margin-top: 15px; font-size: 1rem;">
                                        Nouvelle bankroll: <b>{new_bankroll:.2f}€</b>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.info("Aucun pari en cours à mettre à jour.")
                
                # Colonne pour supprimer les paris
                with manage_columns[1]:
                    st.markdown("""
                    <div class="card">
                        <h3 style="margin-top: 0;">Supprimer un pari</h3>
                        <p>Supprimer un pari en cours (avant d'en connaître le résultat)</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
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
                        
                        # AMÉLIORATION UI: Affichage amélioré des détails du pari
                        st.markdown(f"""
                        <div class="card" style="background: linear-gradient(145deg, rgba(244, 67, 54, 0.1) 0%, rgba(211, 47, 47, 0.1) 100%);">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                                <div style="font-weight: 500;">Pari #{delete_bet_id}</div>
                                <div style="font-weight: 500;">{selected_bet['event_name']}</div>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                                <div>Pick</div>
                                <div style="font-weight: 600;">{selected_bet['pick']}</div>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                                <div>Cote</div>
                                <div style="font-weight: 600;">{selected_bet['odds']}</div>
                            </div>
                            <div style="display: flex; justify-content: space-between;">
                                <div>Mise</div>
                                <div style="font-weight: 600;">{selected_bet['stake']}€</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # AMÉLIORATION UI: Message d'avertissement
                        st.markdown("""
                        <div style="background-color: rgba(244, 67, 54, 0.1); padding: 10px; border-radius: 8px; margin: 15px 0; border-left: 3px solid #F44336;">
                            <b style="color: #F44336;">⚠️ Attention:</b> La suppression est définitive et ne peut pas être annulée.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Confirmation avant suppression
                        confirm_delete = st.checkbox("Je confirme vouloir supprimer ce pari", key="confirm_delete")
                        
                        if confirm_delete:
                            # Bouton pour supprimer
                            if st.button("🗑️ Supprimer le pari", type="primary", key="delete_bet_btn", use_container_width=True):
                                # Animation de chargement
                                with st.spinner(f"Suppression du pari #{delete_bet_id}..."):
                                    # Supprimer le pari
                                    if delete_bet(delete_bet_id):
                                        # AMÉLIORATION UI: Message de confirmation
                                        st.success(f"✅ Pari #{delete_bet_id} supprimé avec succès!")
                                        
                                        # AMÉLIORATION UI: Carte de confirmation
                                        st.markdown(f"""
                                        <div class="card section-fade-in" style="text-align: center; margin: 15px 0;">
                                            <div style="font-size: 3rem; margin-bottom: 10px;">🗑️</div>
                                            <h3 style="margin-bottom: 15px;">Pari supprimé</h3>
                                            <div style="color: #aaa;">
                                                Le pari a été définitivement retiré de votre historique.
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    else:
                                        st.error("❌ Erreur lors de la suppression du pari.")
                        else:
                            # Bouton désactivé
                            st.markdown("""
                            <div style="opacity: 0.5; pointer-events: none; margin-top: 15px;">
                                <button style="width: 100%; padding: 8px 0; background-color: #F44336; color: white; border: none; border-radius: 4px; cursor: not-allowed;">
                                    🗑️ Supprimer le pari
                                </button>
                            </div>
                            <div style="text-align: center; font-size: 0.9rem; color: #aaa; margin-top: 5px;">
                                Cochez la case de confirmation pour activer le bouton
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("Aucun pari à supprimer.")
        else:
            # AMÉLIORATION UI: Message d'information pour état vide
            st.markdown("""
            <div class="card" style="text-align: center; padding: 40px 20px;">
                <div style="font-size: 6rem; margin-bottom: 20px;">📊</div>
                <h2 style="margin-bottom: 15px;">Aucune donnée d'historique</h2>
                <p style="margin-bottom: 30px;">Placez votre premier pari pour commencer à suivre vos performances.</p>
                <div style="opacity: 0.6; font-style: italic;">
                    Utilisez les onglets <b>Prédiction</b> ou <b>Événements à venir</b> pour placer vos paris.
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Téléchargement des données
    if has_bets and has_bankroll and os.path.exists(bets_file) and os.path.exists(bankroll_file):
        st.markdown("---")
        st.subheader("Exporter les données")
        
        download_cols = st.columns(2)
        
        with download_cols[0]:
            if os.path.exists(bets_file):
                with open(bets_file, 'rb') as f:
                    st.download_button(
                        label="📥 Télécharger les paris (CSV)",
                        data=f,
                        file_name='ufc_bets_history.csv',
                        mime='text/csv',
                        key="download_bets_btn",
                        use_container_width=True
                    )
        
        with download_cols[1]:
            if os.path.exists(bankroll_file):
                with open(bankroll_file, 'rb') as f:
                    st.download_button(
                        label="📥 Télécharger l'historique bankroll (CSV)",
                        data=f,
                        file_name='ufc_bankroll_history.csv',
                        mime='text/csv',
                        key="download_bankroll_btn",
                        use_container_width=True
                    )

# Lancer l'application
if __name__ == "__main__":
    main()
        