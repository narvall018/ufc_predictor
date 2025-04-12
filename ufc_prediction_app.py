import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import os
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
</style>
""", unsafe_allow_html=True)

# Fonction pour charger le fichier de statistiques des combattants
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

# Fonction pour prédire l'issue d'un combat
def predict_fight(fighter_a, fighter_b, model_info=None, odds_a=0, odds_b=0):
    """
    Prédit l'issue d'un combat avec analyse de paris
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
    
    # Ajouter un peu de bruit pour simuler un modèle
    a_score += np.random.normal(0, 0.2)
    b_score += np.random.normal(0, 0.2)
    
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

def main():
    # Titre principal
    st.markdown('<div class="main-title">🥊 Prédicteur de Combats UFC 🥊</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Analysez et prédisez l\'issue des affrontements</div>', unsafe_allow_html=True)
    
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
    
    # Interface de sélection des combattants
    st.sidebar.markdown("## Sélection des combattants")
    
    # Liste des noms de combattants
    fighter_names = sorted([fighter['name'] for fighter in fighters])
    
    # Recherche et sélection des combattants
    st.sidebar.markdown("### 🔴 Combattant Rouge")
    fighter_a_search = st.sidebar.text_input("Rechercher Rouge", key="search_red")
    if fighter_a_search:
        filtered_names_a = [name for name in fighter_names if fighter_a_search.lower() in name.lower()]
        fighter_a_options = filtered_names_a if filtered_names_a else fighter_names
    else:
        fighter_a_options = fighter_names
    
    fighter_a_name = st.sidebar.selectbox("Sélectionner Rouge", fighter_a_options, index=0 if fighter_a_options else 0)
    
    st.sidebar.markdown("### 🔵 Combattant Bleu")
    fighter_b_search = st.sidebar.text_input("Rechercher Bleu", key="search_blue")
    if fighter_b_search:
        filtered_names_b = [name for name in fighter_names if fighter_b_search.lower() in name.lower()]
        fighter_b_options = filtered_names_b if filtered_names_b else fighter_names
    else:
        fighter_b_options = fighter_names
    
    # S'assurer qu'on n'a pas le même combattant par défaut
    default_index_b = min(1, len(fighter_b_options) - 1) if len(fighter_b_options) > 1 and fighter_b_options[0] == fighter_a_name else 0
    fighter_b_name = st.sidebar.selectbox("Sélectionner Bleu", fighter_b_options, index=default_index_b)
    
    # Options de paris
    st.sidebar.markdown("## 💰 Options de paris")
    odds_a = st.sidebar.number_input("Cote Rouge", min_value=1.01, value=2.0, step=0.05, format="%.2f")
    odds_b = st.sidebar.number_input("Cote Bleu", min_value=1.01, value=1.8, step=0.05, format="%.2f")
    
    # Bouton de prédiction
    predict_btn = st.sidebar.button("🥊 Prédire le combat", type="primary")
    
    # Récupérer les statistiques des combattants sélectionnés
    fighter_a = fighters_dict.get(fighter_a_name)
    fighter_b = fighters_dict.get(fighter_b_name)
    
    # Vérifier si on peut faire une prédiction
    if predict_btn and fighter_a and fighter_b:
        if fighter_a_name == fighter_b_name:
            st.error("Veuillez sélectionner deux combattants différents.")
        else:
            # Faire la prédiction
            prediction = predict_fight(
                fighter_a, 
                fighter_b, 
                None,
                odds_a=odds_a,
                odds_b=odds_b
            )
            
            # Afficher la prédiction principale
            winner_color = "red" if prediction['prediction'] == 'Red' else "blue"
            winner_name = prediction['winner_name']
            loser_name = prediction['loser_name']
            
            # Container pour l'affichage du résultat
            result_container = st.container()
            
            with result_container:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h2 style="text-align:center;">Prédiction du combat</h2>
                        <h3 style="text-align:center; color:{winner_color};" class="winner">
                            🏆 Vainqueur prédit: {winner_name} 🏆
                        </h3>
                        <p style="text-align:center; font-size:1.2em;">
                            Probabilité: <span class="red-fighter">{prediction['red_probability']:.2f}</span> pour {fighter_a_name}, 
                            <span class="blue-fighter">{prediction['blue_probability']:.2f}</span> pour {fighter_b_name}
                        </p>
                        <p style="text-align:center;">Niveau de confiance: <b>{prediction['confidence']}</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Graphique des probabilités
                    proba_fig = go.Figure()
                    proba_fig.add_trace(go.Bar(
                        x=[fighter_a_name, fighter_b_name],
                        y=[prediction['red_probability'], prediction['blue_probability']],
                        text=[f"{prediction['red_probability']:.2f}", f"{prediction['blue_probability']:.2f}"],
                        textposition='auto',
                        marker_color=['red', 'blue']
                    ))
                    proba_fig.update_layout(
                        title="Probabilités de victoire",
                        yaxis=dict(range=[0, 1]),
                        height=300
                    )
                    st.plotly_chart(proba_fig, use_container_width=True)
            
            # Analyse des paris si disponible
            if 'betting' in prediction:
                betting = prediction['betting']
                
                st.markdown("""
                <div style="text-align:center;">
                    <h2>💰 Analyse des paris 💰</h2>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Table d'analyse des paris pour le combattant rouge
                    rec_color_red = "green" if betting['recommendation_red'] == 'Favorable' else "orange" if betting['recommendation_red'] == 'Neutre' else "red"
                    st.markdown(f"""
                    <div style="background-color:rgba(255, 235, 238, 0.7); padding:15px; border-radius:10px; margin-bottom:20px;">
                        <h3 style="text-align:center; color:red;">{fighter_a_name}</h3>
                        <table style="width:100%;">
                            <tr style="background-color:rgba(255, 255, 255, 0.1);">
                                <th style="padding:8px; text-align:left;">Cote</th>
                                <td style="padding:8px; text-align:center;">{betting['odds_red']:.2f}</td>
                            </tr>
                            <tr>
                                <th style="padding:8px; text-align:left;">Probabilité implicite</th>
                                <td style="padding:8px; text-align:center;">{betting['implied_prob_red']:.2f}</td>
                            </tr>
                            <tr style="background-color:rgba(255, 255, 255, 0.1);">
                                <th style="padding:8px; text-align:left;">Notre probabilité</th>
                                <td style="padding:8px; text-align:center;">{prediction['red_probability']:.2f}</td>
                            </tr>
                            <tr>
                                <th style="padding:8px; text-align:left;">Avantage</th>
                                <td style="padding:8px; text-align:center;">{betting['edge_red']*100:.1f}%</td>
                            </tr>
                            <tr style="background-color:rgba(255, 255, 255, 0.1);">
                                <th style="padding:8px; text-align:left;">Valeur espérée</th>
                                <td style="padding:8px; text-align:center;">{betting['ev_red']*100:.1f}%</td>
                            </tr>
                            <tr>
                                <th style="padding:8px; text-align:left;">Recommandation</th>
                                <td style="padding:8px; text-align:center; color:{rec_color_red}; font-weight:bold;">
                                    {betting['recommendation_red']}
                                </td>
                            </tr>
                        </table>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Table d'analyse des paris pour le combattant bleu
                    rec_color_blue = "green" if betting['recommendation_blue'] == 'Favorable' else "orange" if betting['recommendation_blue'] == 'Neutre' else "red"
                    st.markdown(f"""
                    <div style="background-color:rgba(227, 242, 253, 0.7); padding:15px; border-radius:10px; margin-bottom:20px;">
                        <h3 style="text-align:center; color:blue;">{fighter_b_name}</h3>
                        <table style="width:100%;">
                            <tr style="background-color:rgba(255, 255, 255, 0.1);">
                                <th style="padding:8px; text-align:left;">Cote</th>
                                <td style="padding:8px; text-align:center;">{betting['odds_blue']:.2f}</td>
                            </tr>
                            <tr>
                                <th style="padding:8px; text-align:left;">Probabilité implicite</th>
                                <td style="padding:8px; text-align:center;">{betting['implied_prob_blue']:.2f}</td>
                            </tr>
                            <tr style="background-color:rgba(255, 255, 255, 0.1);">
                                <th style="padding:8px; text-align:left;">Notre probabilité</th>
                                <td style="padding:8px; text-align:center;">{prediction['blue_probability']:.2f}</td>
                            </tr>
                            <tr>
                                <th style="padding:8px; text-align:left;">Avantage</th>
                                <td style="padding:8px; text-align:center;">{betting['edge_blue']*100:.1f}%</td>
                            </tr>
                            <tr style="background-color:rgba(255, 255, 255, 0.1);">
                                <th style="padding:8px; text-align:left;">Valeur espérée</th>
                                <td style="padding:8px; text-align:center;">{betting['ev_blue']*100:.1f}%</td>
                            </tr>
                            <tr>
                                <th style="padding:8px; text-align:left;">Recommandation</th>
                                <td style="padding:8px; text-align:center; color:{rec_color_blue}; font-weight:bold;">
                                    {betting['recommendation_blue']}
                                </td>
                            </tr>
                        </table>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Afficher les statistiques comparatives
            st.markdown("""
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
        
        # Explication des fonctionnalités
        st.markdown("""
        ### Comment utiliser l'application:
        
        1. **Sélectionnez les combattants**: Utilisez les menus déroulants dans la barre latérale pour choisir les deux combattants que vous souhaitez comparer.
        
        2. **Entrez les cotes** (optionnel): Si vous souhaitez analyser les opportunités de paris, entrez les cotes proposées par les bookmakers.
        
        3. **Lancez la prédiction**: Cliquez sur le bouton "Prédire le combat" pour obtenir une analyse détaillée.
        
        4. **Explorez les résultats**: Consultez les différentes visualisations et tableaux pour comprendre les forces et faiblesses de chaque combattant.
        """)

if __name__ == "__main__":
    main()
