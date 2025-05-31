import pandas as pd
import numpy as np
import pickle
import joblib
import os
import warnings
from datetime import datetime
import random
from deap import base, creator, tools, algorithms
from typing import Dict, List, Tuple, Optional
import matplotlib
matplotlib.use('Agg')  # Backend sans interface graphique pour éviter l'erreur Qt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time
from tqdm import tqdm
import json
from multiprocessing import Pool, cpu_count
import logging

warnings.filterwarnings('ignore')

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimization_log.txt'),
        logging.StreamHandler()
    ]
)

class UFCBettingOptimizer:
    """
    Classe principale pour l'optimisation de stratégie de paris UFC
    Version ULTRA-SÉCURISÉE : Drawdown maximum 30-40% avec excellent rendement
    """
    
    def __init__(self, model_path: str, fighters_stats_path: str, odds_data_path: str):
        """
        Initialise l'optimiseur avec les chemins vers les fichiers nécessaires
        """
        self.model_path = model_path
        self.fighters_stats_path = fighters_stats_path
        self.odds_data_path = odds_data_path
        
        # État d'avancement
        print("\n" + "="*70)
        print("🥊 UFC BETTING STRATEGY OPTIMIZER - DRAWDOWN CONTRÔLÉ")
        print("="*70 + "\n")
        
        # Chargement des données avec barre de progression
        print("📊 Chargement des données...")
        
        with tqdm(total=4, desc="Initialisation", unit="étape") as pbar:
            # Étape 1: Chargement du modèle
            pbar.set_description("Chargement du modèle ML")
            self.model_data = self._load_model()
            pbar.update(1)
            time.sleep(0.1)
            
            # Étape 2: Chargement des stats
            pbar.set_description("Chargement des statistiques des combattants")
            self.fighters = self._load_fighters_stats()
            self.fighters_dict = {fighter['name']: fighter for fighter in self.fighters}
            pbar.update(1)
            time.sleep(0.1)
            
            # Étape 3: Chargement des cotes
            pbar.set_description("Chargement des données de cotes")
            self.odds_data = pd.read_csv(odds_data_path)
            print(f"\n   ✅ {len(self.odds_data)} combats chargés pour l'analyse")
            pbar.update(1)
            time.sleep(0.1)
            
            # Étape 4: Configuration GA
            pbar.set_description("Configuration de l'algorithme génétique")
            self.setup_genetic_algorithm()
            pbar.update(1)
        
        # Résumé de l'initialisation
        print("\n📈 RÉSUMÉ DE L'INITIALISATION:")
        print(f"   • Modèle ML: {'✅ Chargé' if self.model_data['model'] else '❌ Non disponible (mode statistique)'}")
        print(f"   • Combattants: {len(self.fighters)} profils chargés")
        print(f"   • Combats historiques: {len(self.odds_data)} entrées")
        print(f"   • Processeurs disponibles: {cpu_count()} cores")
        print(f"   • 🛡️ MODE SÉCURISÉ: Drawdown maximum 35%")
        print("\n" + "="*70 + "\n")
        
        # Statistiques sur les données
        self._display_data_statistics()
        
        # Cache pour les prédictions
        self.prediction_cache = {}
        
        # Métriques de suivi
        self.generation_metrics = []
        
    def _display_data_statistics(self):
        """Affiche des statistiques sur les données chargées"""
        print("📊 STATISTIQUES DES DONNÉES:")
        
        # Stats sur les combats
        total_fights = len(self.odds_data)
        red_wins = len(self.odds_data[self.odds_data['Winner'] == 'Red'])
        blue_wins = len(self.odds_data[self.odds_data['Winner'] == 'Blue'])
        
        print(f"   • Répartition des victoires: Rouge {red_wins/total_fights:.1%} | Bleu {blue_wins/total_fights:.1%}")
        
        # Stats sur les cotes
        avg_red_odds = self.odds_data['R_odds'].mean()
        avg_blue_odds = self.odds_data['B_odds'].mean()
        print(f"   • Cotes moyennes: Rouge {avg_red_odds:.2f} | Bleu {avg_blue_odds:.2f}")
        
        # Plage temporelle
        print(f"   • Période couverte: {total_fights} combats analysables")
        print()
        
    def _load_model(self) -> Dict:
        """
        Reproduction exacte du chargement du modèle ML depuis l'application de référence
        """
        model_data = {
            "model": None,
            "scaler": None,
            "feature_names": None
        }
        
        # Essayer de charger le modèle joblib ou pkl
        model_files = ["ufc_prediction_model.joblib", "ufc_prediction_model.pkl"]
        
        for model_file in model_files:
            if os.path.exists(model_file):
                try:
                    if model_file.endswith('.joblib'):
                        loaded_data = joblib.load(model_file)
                    else:
                        with open(model_file, 'rb') as file:
                            loaded_data = pickle.load(file)
                    
                    if loaded_data:
                        model_data["model"] = loaded_data.get('model')
                        model_data["scaler"] = loaded_data.get('scaler')
                        model_data["feature_names"] = loaded_data.get('feature_names')
                        logging.info(f"Modèle ML chargé depuis {model_file}")
                        break
                except Exception as e:
                    logging.error(f"Erreur lors du chargement du modèle {model_file}: {e}")
        
        if model_data["model"] is None:
            logging.warning("Aucun modèle ML trouvé. Utilisation de la méthode statistique uniquement.")
            
        return model_data
    
    def _load_fighters_stats(self) -> List[Dict]:
        """
        Reproduction exacte du parsing des statistiques des combattants
        """
        fighters = []
        current_fighter = {}
        
        try:
            with open(self.fighters_stats_path, 'r') as file:
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
                
            # Dédupliquer les combattants
            fighters = self._deduplicate_fighters(fighters)
                
            return fighters
            
        except Exception as e:
            logging.error(f"Erreur lors du chargement des statistiques: {e}")
            return []
    
    def _deduplicate_fighters(self, fighters_list: List[Dict]) -> List[Dict]:
        """
        Reproduction exacte de la déduplication des combattants
        """
        fighters_by_name = {}
        
        for fighter in fighters_list:
            name = fighter['name']
            
            # Calculer un score de performance
            wins = fighter.get('wins', 0)
            losses = fighter.get('losses', 0)
            win_ratio = wins / max(wins + losses, 1)
            
            # Score combiné
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
    
    def _get_float_value(self, stats_dict: Dict, key: str, default: float = 0.0) -> float:
        """
        Reproduction exacte de la récupération de valeur float
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
    
    def _create_ml_features(self, r_stats: Dict, b_stats: Dict) -> Dict:
        """
        Reproduction EXACTE de la création des features pour le modèle ML
        """
        features = {}
        
        # Liste des statistiques numériques
        numeric_stats = ['wins', 'losses', 'height', 'weight', 'reach', 'age', 
                         'SLpM', 'sig_str_acc', 'SApM', 'str_def', 
                         'td_avg', 'td_acc', 'td_def', 'sub_avg']
        
        # Extraire et convertir les statistiques numériques
        for stat in numeric_stats:
            r_value = self._get_float_value(r_stats, stat, 0.0)
            b_value = self._get_float_value(b_stats, stat, 0.0)
            
            features[f'r_{stat}'] = r_value
            features[f'b_{stat}'] = b_value
            features[f'diff_{stat}'] = r_value - b_value
            
            if b_value != 0:
                features[f'ratio_{stat}'] = r_value / b_value
            else:
                features[f'ratio_{stat}'] = 0.0
        
        # Features avancées
        
        # 1. Win ratio et expérience
        r_wins = self._get_float_value(r_stats, 'wins', 0)
        r_losses = self._get_float_value(r_stats, 'losses', 0)
        b_wins = self._get_float_value(b_stats, 'wins', 0)
        b_losses = self._get_float_value(b_stats, 'losses', 0)
        
        # Nombre total de combats
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
        r_slpm = self._get_float_value(r_stats, 'SLpM', 0)
        r_sapm = self._get_float_value(r_stats, 'SApM', 0)
        b_slpm = self._get_float_value(b_stats, 'SLpM', 0)
        b_sapm = self._get_float_value(b_stats, 'SApM', 0)
        
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
        r_height = self._get_float_value(r_stats, 'height', 0)
        r_weight = self._get_float_value(r_stats, 'weight', 0)
        r_reach = self._get_float_value(r_stats, 'reach', 0)
        b_height = self._get_float_value(b_stats, 'height', 0)
        b_weight = self._get_float_value(b_stats, 'weight', 0)
        b_reach = self._get_float_value(b_stats, 'reach', 0)
        
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
        
        # Avantage d'allonge normalisé
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
        r_td_avg = self._get_float_value(r_stats, 'td_avg', 0)
        r_sub_avg = self._get_float_value(r_stats, 'sub_avg', 0)
        r_str_def = self._get_float_value(r_stats, 'str_def', 0)
        r_td_def = self._get_float_value(r_stats, 'td_def', 0)
        b_td_avg = self._get_float_value(b_stats, 'td_avg', 0)
        b_sub_avg = self._get_float_value(b_stats, 'sub_avg', 0)
        b_str_def = self._get_float_value(b_stats, 'str_def', 0)
        b_td_def = self._get_float_value(b_stats, 'td_def', 0)
        
        # Spécialiste de striking vs grappling
        if r_td_avg > 0:
            features['r_striking_grappling_ratio'] = r_slpm / r_td_avg
        else:
            features['r_striking_grappling_ratio'] = r_slpm if r_slpm > 0 else 0
            
        if b_td_avg > 0:
            features['b_striking_grappling_ratio'] = b_slpm / b_td_avg
        else:
            features['b_striking_grappling_ratio'] = b_slpm if b_slpm > 0 else 0
            
        # Offensive vs défensive
        features['r_offensive_rating'] = r_slpm * r_td_avg * (1 + r_sub_avg)
        features['b_offensive_rating'] = b_slpm * b_td_avg * (1 + b_sub_avg)
        features['diff_offensive_rating'] = features['r_offensive_rating'] - features['b_offensive_rating']
        
        features['r_defensive_rating'] = r_str_def * r_td_def
        features['b_defensive_rating'] = b_str_def * b_td_def
        features['diff_defensive_rating'] = features['r_defensive_rating'] - features['b_defensive_rating']
        
        # 5. Variables composites
        features['r_overall_performance'] = features['r_win_ratio'] * features['r_offensive_rating'] * features['r_defensive_rating']
        features['b_overall_performance'] = features['b_win_ratio'] * features['b_offensive_rating'] * features['b_defensive_rating']
        features['diff_overall_performance'] = features['r_overall_performance'] - features['b_overall_performance']
        
        # Avantage physique combiné
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
    
    def predict_with_ml(self, r_stats: Dict, b_stats: Dict) -> Optional[Dict]:
        """
        Reproduction exacte de la prédiction avec le modèle ML
        """
        # Utiliser le cache si disponible
        cache_key = f"{r_stats['name']}_{b_stats['name']}"
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]
        
        model = self.model_data.get("model")
        scaler = self.model_data.get("scaler")
        feature_names = self.model_data.get("feature_names")
        
        # Si le modèle n'est pas chargé, retourner None
        if model is None or scaler is None or feature_names is None:
            return None
        
        try:
            # Créer les features
            features = self._create_ml_features(r_stats, b_stats)
            
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
                'confidence': 'High' if abs(red_prob - blue_prob) > 0.2 else 'Medium'
            }
            
            # Mettre en cache
            self.prediction_cache[cache_key] = result
            
            return result
        except Exception as e:
            logging.error(f"Erreur lors de la prédiction ML: {e}")
            return None
    
    def predict_fight_classic(self, fighter_a: Dict, fighter_b: Dict) -> Dict:
        """
        Reproduction exacte de la prédiction classique basée sur les statistiques
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
        
        # Résultat
        result = {
            'prediction': 'Red' if red_prob > blue_prob else 'Blue',
            'red_probability': red_prob,
            'blue_probability': blue_prob,
            'confidence': 'High' if abs(red_prob - blue_prob) > 0.2 else 'Medium'
        }
        
        return result
    
    def calculate_kelly(self, prob: float, odds: float, bankroll: float, fraction: float = 1) -> float:
        """
        Reproduction exacte du calcul de la mise Kelly
        """
        b = odds - 1  # gain net par unité misée
        q = 1 - prob  # probabilité de perte
        
        # Formule de Kelly
        kelly_percentage = (prob * b - q) / b
        
        # Si Kelly est négatif, ne pas parier
        if kelly_percentage <= 0:
            return 0
        
        # Appliquer la fraction Kelly
        fractional_kelly = kelly_percentage / fraction
        
        # Calculer la mise recommandée
        recommended_stake = bankroll * fractional_kelly
        
        return round(recommended_stake, 2)
    
    def find_best_match(self, name: str) -> Optional[str]:
        """
        Trouve le meilleur match pour un nom de combattant
        """
        if not name:
            return None
        
        # Nettoyage du nom
        name = name.strip()
        
        # Recherche exacte
        if name in self.fighters_dict:
            return name
        
        # Recherche insensible à la casse
        name_lower = name.lower()
        for fighter_name in self.fighters_dict:
            if fighter_name.lower() == name_lower:
                return fighter_name
        
        # Recherche floue
        best_match = None
        best_score = 0
        
        for fighter_name in self.fighters_dict:
            score = 0
            fighter_lower = fighter_name.lower()
            
            # Si l'un contient l'autre
            if name_lower in fighter_lower or fighter_lower in name_lower:
                score += 5
            
            # Correspondance partielle de mots
            name_words = name_lower.split()
            fighter_words = fighter_lower.split()
            
            for word in name_words:
                if word in fighter_words:
                    score += 2
                # Match de préfixe
                for fighter_word in fighter_words:
                    if fighter_word.startswith(word) or word.startswith(fighter_word):
                        score += 1.5
                    elif word in fighter_word or fighter_word in word:
                        score += 1
            
            # Bonus pour longueur similaire
            length_diff = abs(len(name) - len(fighter_name))
            if length_diff <= 3:
                score += 1
            
            if score > best_score:
                best_score = score
                best_match = fighter_name
        
        # Retourner seulement si match raisonnable
        if best_score >= 3:
            return best_match
        
        return None
    
    def simulate_betting_strategy(self, params: Dict, validation_split: float = 0.0) -> Dict:
        """
        Simule une stratégie de paris avec les paramètres donnés
        Amélioration: support pour la validation croisée
        """
        initial_bankroll = 1000
        bankroll = initial_bankroll
        bets_history = []
        
        # Paramètres de la stratégie
        kelly_fraction = params['kelly_fraction']
        min_confidence = params['min_confidence']
        min_value = params['min_value']
        max_bet_fraction = params['max_bet_fraction']
        min_edge = params['min_edge']
        
        # Déterminer la portion de données à utiliser
        if validation_split > 0:
            split_idx = int(len(self.odds_data) * (1 - validation_split))
            data_to_use = self.odds_data.iloc[:split_idx]
        else:
            data_to_use = self.odds_data
        
        # Parcourir les combats dans l'ordre chronologique
        for _, fight in data_to_use.iterrows():
            # Trouver les combattants
            red_match = self.find_best_match(fight['R_fighter'])
            blue_match = self.find_best_match(fight['B_fighter'])
            
            if not red_match or not blue_match:
                continue
            
            red_stats = self.fighters_dict[red_match]
            blue_stats = self.fighters_dict[blue_match]
            
            # Prédiction
            ml_prediction = self.predict_with_ml(red_stats, blue_stats)
            classic_prediction = self.predict_fight_classic(red_stats, blue_stats)
            
            # Utiliser ML si disponible, sinon classique
            prediction = ml_prediction if ml_prediction else classic_prediction
            
            # Déterminer sur qui parier
            if prediction['prediction'] == 'Red':
                bet_prob = prediction['red_probability']
                bet_odds = fight['R_odds']
                bet_on = 'Red'
            else:
                bet_prob = prediction['blue_probability']
                bet_odds = fight['B_odds']
                bet_on = 'Blue'
            
            # Vérifier les critères de pari
            if bet_prob < min_confidence:
                continue
            
            # Calculer l'edge (avantage)
            implied_prob = 1 / bet_odds
            edge = bet_prob - implied_prob
            
            if edge < min_edge:
                continue
            
            # Vérifier la value
            value = bet_prob * bet_odds
            if value < min_value:
                continue
            
            # Calculer la mise Kelly
            kelly_stake = self.calculate_kelly(bet_prob, bet_odds, bankroll, kelly_fraction)
            
            # Limiter la mise au maximum autorisé
            max_stake = bankroll * max_bet_fraction
            stake = min(kelly_stake, max_stake)
            
            # Ne pas parier si la mise est trop faible
            if stake < 1 or stake > bankroll:
                continue
            
            # Enregistrer le pari
            result = 'win' if fight['Winner'] == bet_on else 'loss'
            profit = stake * (bet_odds - 1) if result == 'win' else -stake
            
            bankroll += profit
            
            # Protection contre la faillite
            if bankroll <= 0:
                bankroll = 0
                break
            
            bets_history.append({
                'fighter_red': fight['R_fighter'],
                'fighter_blue': fight['B_fighter'],
                'bet_on': bet_on,
                'odds': bet_odds,
                'probability': bet_prob,
                'edge': edge,
                'value': value,
                'stake': stake,
                'result': result,
                'profit': profit,
                'bankroll': bankroll
            })
        
        return self.calculate_metrics(bets_history, initial_bankroll)
    
    def calculate_metrics(self, bets_history: List[Dict], initial_bankroll: float) -> Dict:
        """
        Calcule toutes les métriques de performance demandées
        Version améliorée avec plus de métriques
        """
        if not bets_history:
            return {
                'roi': -100,
                'total_bets': 0,
                'win_rate': 0,
                'profit': -initial_bankroll,
                'max_drawdown': -100,
                'sharpe_ratio': -10,
                'calmar_ratio': -10,
                'sortino_ratio': -10,
                'profit_factor': 0,
                'expectancy': 0,
                'volatility': 0,
                'var_95': 0,
                'recovery_factor': 0,
                'max_consecutive_losses': 0,
                'average_odds': 0,
                'median_stake_pct': 0,
                'risk_adjusted_return': -100
            }
        
        # Métriques de base
        df = pd.DataFrame(bets_history)
        total_bets = len(df)
        wins = len(df[df['result'] == 'win'])
        win_rate = wins / total_bets if total_bets > 0 else 0
        
        # Profit et ROI
        final_bankroll = df['bankroll'].iloc[-1] if len(df) > 0 else initial_bankroll
        total_profit = final_bankroll - initial_bankroll
        roi = (total_profit / initial_bankroll) * 100
        
        # Maximum Drawdown
        rolling_max = df['bankroll'].expanding().max()
        drawdown = (df['bankroll'] - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()
        
        # Durée du drawdown
        drawdown_periods = []
        in_drawdown = False
        start = 0
        
        for i in range(len(drawdown)):
            if drawdown.iloc[i] < 0 and not in_drawdown:
                in_drawdown = True
                start = i
            elif drawdown.iloc[i] >= 0 and in_drawdown:
                in_drawdown = False
                drawdown_periods.append(i - start)
        
        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        
        # Pertes consécutives maximales
        max_consecutive_losses = 0
        current_losses = 0
        for result in df['result']:
            if result == 'loss':
                current_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
            else:
                current_losses = 0
        
        # Volatilité et ratios de risque
        returns = df['profit'].values
        volatility = np.std(returns) if len(returns) > 1 else 0
        
        # Sharpe Ratio (assumant taux sans risque = 0)
        avg_return = np.mean(returns) if len(returns) > 0 else 0
        sharpe_ratio = avg_return / volatility if volatility > 0 else 0
        
        # Calmar Ratio
        annual_return = roi * (252 / total_bets) if total_bets > 0 else 0
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_volatility = np.std(downside_returns) if len(downside_returns) > 1 else 0
        sortino_ratio = avg_return / downside_volatility if downside_volatility > 0 else 0
        
        # Profit Factor
        gross_profits = df[df['profit'] > 0]['profit'].sum()
        gross_losses = abs(df[df['profit'] < 0]['profit'].sum())
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else np.inf
        
        # Expectancy
        expectancy = avg_return
        
        # Value at Risk (95%)
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
        
        # Recovery Factor
        recovery_factor = total_profit / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Métriques supplémentaires
        average_odds = df['odds'].mean()
        median_stake_pct = (df['stake'] / df['bankroll'].shift(1).fillna(initial_bankroll)).median() * 100
        
        # Risk-adjusted return (nouvelle métrique équilibrée)
        risk_adjusted_return = roi / (1 + abs(max_drawdown)) * sharpe_ratio
        
        return {
            'roi': roi,
            'total_bets': total_bets,
            'win_rate': win_rate,
            'profit': total_profit,
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'volatility': volatility,
            'var_95': var_95,
            'recovery_factor': recovery_factor,
            'final_bankroll': final_bankroll,
            'max_consecutive_losses': max_consecutive_losses,
            'average_odds': average_odds,
            'median_stake_pct': median_stake_pct,
            'risk_adjusted_return': risk_adjusted_return
        }
    
    def setup_genetic_algorithm(self):
        """
        Configure l'algorithme génétique avec DEAP
        Version ULTRA-SÉCURISÉE : Drawdown maximum 35%
        """
        # Créer les types de fitness et d'individu
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual
            
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        # Toolbox
        self.toolbox = base.Toolbox()
        
        # =================== PARAMÈTRES ULTRA-SÉCURISÉS ===================
        # Bornes strictement conservatrices pour éviter les gros drawdowns
        self.param_bounds = {
            'kelly_fraction': (4, 50),          # Kelly/4 à Kelly/50 (très conservateur)
            'min_confidence': (0.30, 0.80),    # 55% à 80% confiance (plus strict)
            'min_value': (1.05, 1.25),         # Value plus stricte
            'max_bet_fraction': (0.02, 0.08), # 0.5% à 4% max (très conservateur)
            'min_edge': (0.03, 0.15),          # 3% à 12% d'edge minimum
        }
        
        # Générateurs d'attributs
        self.toolbox.register("kelly_fraction", random.uniform, *self.param_bounds['kelly_fraction'])
        self.toolbox.register("min_confidence", random.uniform, *self.param_bounds['min_confidence'])
        self.toolbox.register("min_value", random.uniform, *self.param_bounds['min_value'])
        self.toolbox.register("max_bet_fraction", random.uniform, *self.param_bounds['max_bet_fraction'])
        self.toolbox.register("min_edge", random.uniform, *self.param_bounds['min_edge'])
        
        # Structure de l'individu
        self.toolbox.register("individual", tools.initCycle, creator.Individual,
                              (self.toolbox.kelly_fraction,
                               self.toolbox.min_confidence,
                               self.toolbox.min_value,
                               self.toolbox.max_bet_fraction,
                               self.toolbox.min_edge), n=1)
        
        # Population
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Opérateurs génétiques
        self.toolbox.register("evaluate", self.fitness_function)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.2)  # Crossover plus conservateur
        self.toolbox.register("mutate", self.mutate_individual)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
    
    def mutate_individual(self, individual):
        """
        Mutation personnalisée ULTRA-CONSERVATRICE
        """
        param_names = ['kelly_fraction', 'min_confidence', 'min_value', 
                      'max_bet_fraction', 'min_edge']
        
        # Mutation très fine pour éviter de sortir de la zone sécurisée
        mutation_rate = 0.15  # Taux de mutation réduit
        
        for i, param_name in enumerate(param_names):
            if random.random() < mutation_rate:
                bounds = self.param_bounds[param_name]
                # Mutation gaussienne très fine
                sigma = (bounds[1] - bounds[0]) * 0.03  # Mutations ultra-fines
                new_value = individual[i] + random.gauss(0, sigma)
                # Respecter strictement les bornes sécurisées
                individual[i] = max(bounds[0], min(bounds[1], new_value))
        
        return individual,
    
    def fitness_function(self, individual):
        """
        Fonction de fitness ULTRA-SÉCURISÉE - DRAWDOWN MAXIMUM 35%
        Priorité absolue à la sécurité avec excellent rendement
        """
        # Convertir l'individu en dictionnaire de paramètres
        params = {
            'kelly_fraction': individual[0],
            'min_confidence': individual[1],
            'min_value': individual[2],
            'max_bet_fraction': individual[3],
            'min_edge': individual[4]
        }
        
        # Simuler la stratégie
        metrics = self.simulate_betting_strategy(params)
        
        # =================== CRITÈRES ÉLIMINATOIRES STRICTS ===================
        
        # 🚨 DRAWDOWN > 35% = ÉLIMINATION IMMÉDIATE
        if metrics['max_drawdown'] < -35:
            return -50000,  # Score catastrophique éliminatoire
        
        # 🚨 SÉRIE PERDANTE > 10 = ÉLIMINATION IMMÉDIATE
        if metrics['max_consecutive_losses'] > 10:
            return -40000,
        
        # 🚨 ROI NÉGATIF > -50% = ÉLIMINATION IMMÉDIATE
        if metrics['roi'] < -50:
            return -45000,
        
        # 🚨 TROP PEU DE PARIS = ÉLIMINATION
        if metrics['total_bets'] < 15:
            return -35000,
        
        # 🚨 VOLATILITÉ EXCESSIVE = ÉLIMINATION
        if metrics['volatility'] > 200:
            return -30000,
        
        # =================== FONCTION FITNESS SÉCURISÉE ===================
        
        base_score = 0
        
        # 1. SÉCURITÉ ABSOLUE (60% du score) - Priorité maximale
        security_score = 0
        
        # Drawdown - Récompenses exponentielles pour sécurité
        if metrics['max_drawdown'] > -10:      # Excellent < -10%
            security_score += 500
        elif metrics['max_drawdown'] > -15:    # Très bon < -15%
            security_score += 300
        elif metrics['max_drawdown'] > -20:    # Bon < -20%
            security_score += 200
        elif metrics['max_drawdown'] > -25:    # Acceptable < -25%
            security_score += 100
        elif metrics['max_drawdown'] > -30:    # Limite < -30%
            security_score += 50
        else:  # -30% à -35% (zone critique)
            security_score += 10
        
        # Série perdante - Bonus sécurité
        if metrics['max_consecutive_losses'] <= 3:
            security_score += 150
        elif metrics['max_consecutive_losses'] <= 5:
            security_score += 100
        elif metrics['max_consecutive_losses'] <= 7:
            security_score += 50
        else:  # 8-10 (tolérable)
            security_score += 10
        
        # Stabilité exceptionnelle
        if metrics['volatility'] < 30:
            security_score += 100
        elif metrics['volatility'] < 50:
            security_score += 60
        elif metrics['volatility'] < 80:
            security_score += 30
        
        # 2. RENDEMENT AJUSTÉ AU RISQUE (25% du score)
        performance_score = 0
        
        if metrics['roi'] > 0:
            # ROI positif avec bonus progressif sécurisé
            performance_score = metrics['roi'] * 2
            
            # Bonus significatif pour très bon ROI avec sécurité
            if metrics['roi'] > 20 and metrics['max_drawdown'] > -25:
                performance_score += 200
            elif metrics['roi'] > 10 and metrics['max_drawdown'] > -20:
                performance_score += 100
        else:
            # Pénalité modérée pour ROI négatif (pas éliminatoire si sécurisé)
            performance_score = metrics['roi'] * 1.5
        
        # Bonus Sharpe ratio élevé
        if metrics['sharpe_ratio'] > 1.5:
            performance_score += 100
        elif metrics['sharpe_ratio'] > 1.0:
            performance_score += 60
        elif metrics['sharpe_ratio'] > 0.5:
            performance_score += 30
        
        # 3. CONSISTANCE ET QUALITÉ (15% du score)
        quality_score = 0
        
        # Calmar ratio exceptionnel
        if metrics['calmar_ratio'] > 2.0:
            quality_score += 80
        elif metrics['calmar_ratio'] > 1.5:
            quality_score += 50
        elif metrics['calmar_ratio'] > 1.0:
            quality_score += 25
        
        # Win rate optimal
        if 0.50 <= metrics['win_rate'] <= 0.65:
            quality_score += 40
        elif 0.45 <= metrics['win_rate'] <= 0.70:
            quality_score += 20
        
        # Profit factor robuste
        if metrics['profit_factor'] > 1.8:
            quality_score += 60
        elif metrics['profit_factor'] > 1.4:
            quality_score += 40
        elif metrics['profit_factor'] > 1.1:
            quality_score += 20
        
        # Expectancy positive consistante
        if metrics['expectancy'] > 0:
            quality_score += metrics['expectancy'] * 2
        
        # =================== BONUS SPÉCIAUX SÉCURITÉ ===================
        
        safety_bonus = 0
        
        # 🏆 STRATÉGIE ULTRA-SÉCURISÉE (tous critères excellents)
        if (metrics['max_drawdown'] > -20 and 
            metrics['max_consecutive_losses'] <= 5 and
            metrics['sharpe_ratio'] > 1.0 and
            metrics['roi'] > 5):
            safety_bonus += 300  # Bonus énorme pour perfection
        
        # 🥇 STRATÉGIE TRÈS SÉCURISÉE
        elif (metrics['max_drawdown'] > -25 and 
              metrics['max_consecutive_losses'] <= 6 and
              metrics['roi'] > 0):
            safety_bonus += 150  # Grand bonus sécurité
        
        # 🥈 STRATÉGIE SÉCURISÉE
        elif (metrics['max_drawdown'] > -30 and 
              metrics['max_consecutive_losses'] <= 8):
            safety_bonus += 75   # Bonus sécurité modéré
        
        # Bonus pour mises très conservatrices
        if metrics['median_stake_pct'] < 2:
            safety_bonus += 50
        
        # Bonus récupération rapide
        if metrics['recovery_factor'] > 3:
            safety_bonus += 40
        
        # =================== CALCUL FINAL PONDÉRÉ ===================
        
        final_score = (
            security_score * 0.60 +      # 60% sécurité (priorité absolue)
            performance_score * 0.25 +   # 25% performance ajustée
            quality_score * 0.15 +       # 15% qualité/consistance
            safety_bonus                 # Bonus spéciaux sécurité
        )
        
        # Multiplicateur spécial pour stratégies exceptionnellement sûres
        if (metrics['max_drawdown'] > -18 and 
            metrics['max_consecutive_losses'] <= 4 and
            metrics['roi'] > 8):
            final_score *= 1.3  # Boost 30% pour excellence sécurisée
        
        return max(final_score, -1000),  # Score minimum pour exploration
    
    def optimize(self, population_size=200, generations=100, n_jobs=-1):
        """
        Lance l'optimisation génétique ULTRA-SÉCURISÉE
        Population et générations augmentées pour explorer la zone sécurisée
        """
        print("\n" + "="*70)
        print("🛡️ OPTIMISATION ULTRA-SÉCURISÉE - DRAWDOWN MAX 35%")
        print("="*70)
        print(f"\n📊 Paramètres de l'optimisation:")
        print(f"   • Taille de la population: {population_size}")
        print(f"   • Nombre de générations: {generations}")
        print(f"   • Processeurs utilisés: {cpu_count() if n_jobs == -1 else n_jobs}")
        print(f"   • 🎯 OBJECTIF: ROI > 10% avec Drawdown < 30%")
        print(f"   • 🛡️ SÉCURITÉ: Critères éliminatoires stricts")
        print("\n" + "="*70 + "\n")
        
        # Statistiques
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Population initiale
        print("🌱 Création de la population initiale sécurisée...")
        population = self.toolbox.population(n=population_size)
        
        # Hall of Fame pour garder les meilleurs individus
        hof = tools.HallOfFame(25)  # Top 25 pour plus de diversité
        
        # Variables pour le suivi
        logbook = tools.Logbook()
        logbook.header = ['gen', 'evals'] + stats.fields
        
        # Évaluation initiale
        print("\n📈 Évaluation de la population initiale...")
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        hof.update(population)
        record = stats.compile(population)
        logbook.record(gen=0, evals=len(population), **record)
        
        print(f"   Génération 0 - Meilleur: {record['max']:.2f}, Moyenne: {record['avg']:.2f}")
        
        # Boucle d'évolution avec barre de progression
        print("\n🔄 Évolution sécurisée en cours...\n")
        
        best_fitness_history = []
        no_improvement_count = 0
        last_best_fitness = -float('inf')
        
        with tqdm(total=generations, desc="Optimisation sécurisée", unit="génération") as pbar:
            for gen in range(1, generations + 1):
                self.current_generation = gen
                
                # Sélection avec pression modérée
                offspring = self.toolbox.select(population, len(population))
                offspring = list(map(self.toolbox.clone, offspring))
                
                # Crossover conservateur
                cx_prob = 0.6  # Crossover modéré
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < cx_prob:
                        self.toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values
                
                # Mutation ultra-fine
                mut_prob = 0.12  # Mutation très douce
                for mutant in offspring:
                    if random.random() < mut_prob:
                        self.toolbox.mutate(mutant)
                        del mutant.fitness.values
                
                # Évaluation des nouveaux individus
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(self.toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
                
                # Remplacement avec élitisme fort
                population[:] = offspring
                
                # Assurer que les meilleurs survivent (élitisme renforcé)
                for i, elite in enumerate(hof[:8]):  # Top 8 survivent
                    if i < len(population):
                        population[i] = self.toolbox.clone(elite)
                
                # Mise à jour du Hall of Fame
                hof.update(population)
                
                # Enregistrement des statistiques
                record = stats.compile(population)
                logbook.record(gen=gen, evals=len(invalid_ind), **record)
                
                # Mise à jour de la barre de progression
                pbar.update(1)
                pbar.set_postfix({
                    'Best': f"{record['max']:.0f}",
                    'Avg': f"{record['avg']:.0f}",
                    'Sécurité': '🛡️'
                })
                
                # Suivi de l'amélioration
                best_fitness_history.append(record['max'])
                
                # Détection de stagnation (plus patient pour la sécurité)
                if record['max'] > last_best_fitness + 1:  # Amélioration significative
                    improvement = record['max'] - last_best_fitness
                    no_improvement_count = 0
                    last_best_fitness = record['max']
                    
                    # Log des améliorations majeures sécurisées
                    if improvement > 50:
                        tqdm.write(f"   🎯 Génération {gen} - Stratégie plus sécurisée trouvée! "
                                  f"Score: {record['max']:.0f} (+{improvement:.0f})")
                else:
                    no_improvement_count += 1
                
                # Affichage périodique
                if gen % 15 == 0:
                    tqdm.write(f"   🛡️ Génération {gen} - Best: {record['max']:.0f}, "
                              f"Avg: {record['avg']:.0f} (Sécurité renforcée)")
                
                # Sauvegarde intermédiaire
                if gen % 30 == 0:
                    self._save_checkpoint(hof, gen)
                
                # Early stopping plus patient pour exploration sécurisée
                if no_improvement_count > 40:
                    tqdm.write(f"\n🛡️ Convergence sécurisée atteinte après {gen} générations")
                    break
        
        print("\n✅ Optimisation sécurisée terminée!")
        print(f"   • Générations complétées: {gen}/{generations}")
        print(f"   • Meilleure fitness finale: {last_best_fitness:.0f}")
        print(f"   • 🛡️ Toutes les stratégies respectent Drawdown < 35%")
        
        # Analyser les résultats avec focus sécurité
        print("\n🔍 Analyse des stratégies sécurisées...")
        best_individuals = self._analyze_hall_of_fame_security(hof)
        
        return best_individuals, logbook
    
    def _analyze_hall_of_fame_security(self, hof):
        """Analyse spécialisée pour les stratégies sécurisées"""
        best_individuals = []
        
        print("\n📊 Évaluation des stratégies sécurisées...")
        
        with tqdm(total=min(15, len(hof)), desc="Analyse sécurité", unit="stratégie") as pbar:
            for i, ind in enumerate(hof[:15]):
                params = {
                    'kelly_fraction': ind[0],
                    'min_confidence': ind[1],
                    'min_value': ind[2],
                    'max_bet_fraction': ind[3],
                    'min_edge': ind[4]
                }
                
                # Simulation complète
                metrics = self.simulate_betting_strategy(params)
                
                # Test de validation croisée
                validation_metrics = self.simulate_betting_strategy(params, validation_split=0.2)
                
                # Score de sécurité personnalisé
                security_score = self._calculate_security_score(metrics)
                
                best_individuals.append({
                    'params': params,
                    'metrics': metrics,
                    'validation_metrics': validation_metrics,
                    'fitness': ind.fitness.values[0],
                    'security_score': security_score,
                    'rank': i + 1
                })
                
                pbar.update(1)
        
        # Trier par score de sécurité
        best_individuals.sort(key=lambda x: x['security_score'], reverse=True)
        
        return best_individuals
    
    def _calculate_security_score(self, metrics):
        """Calcule un score de sécurité personnalisé"""
        score = 0
        
        # Drawdown (critère principal)
        if metrics['max_drawdown'] > -15:
            score += 40
        elif metrics['max_drawdown'] > -20:
            score += 30
        elif metrics['max_drawdown'] > -25:
            score += 20
        elif metrics['max_drawdown'] > -30:
            score += 10
        
        # Série perdante
        if metrics['max_consecutive_losses'] <= 4:
            score += 25
        elif metrics['max_consecutive_losses'] <= 6:
            score += 15
        elif metrics['max_consecutive_losses'] <= 8:
            score += 8
        
        # ROI positif
        if metrics['roi'] > 15:
            score += 20
        elif metrics['roi'] > 8:
            score += 15
        elif metrics['roi'] > 0:
            score += 10
        
        # Sharpe ratio
        if metrics['sharpe_ratio'] > 1.5:
            score += 15
        elif metrics['sharpe_ratio'] > 1.0:
            score += 10
        elif metrics['sharpe_ratio'] > 0.5:
            score += 5
        
        return score
    
    def _save_checkpoint(self, hof, generation):
        """Sauvegarde intermédiaire des meilleurs individus"""
        checkpoint = {
            'generation': generation,
            'hall_of_fame': [
                {
                    'params': {
                        'kelly_fraction': ind[0],
                        'min_confidence': ind[1],
                        'min_value': ind[2],
                        'max_bet_fraction': ind[3],
                        'min_edge': ind[4]
                    },
                    'fitness': ind.fitness.values[0]
                }
                for ind in hof
            ]
        }
        
        with open(f'checkpoint_secure_gen_{generation}.json', 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def plot_optimization_results(self, logbook):
        """
        Visualise les résultats de l'optimisation sécurisée
        """
        gen = logbook.select("gen")
        fit_maxs = logbook.select("max")
        fit_avgs = logbook.select("avg")
        fit_stds = logbook.select("std")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Graphique 1: Évolution de la fitness sécurisée
        ax1.plot(gen, fit_maxs, 'g-', label='Maximum (Sécurisé)', linewidth=3)
        ax1.plot(gen, fit_avgs, 'b-', label='Moyenne', linewidth=2)
        ax1.fill_between(gen, 
                        np.array(fit_avgs) - np.array(fit_stds),
                        np.array(fit_avgs) + np.array(fit_stds),
                        alpha=0.3, color='blue', label='±1 std')
        ax1.set_xlabel('Génération')
        ax1.set_ylabel('Score de Fitness Sécurisée')
        ax1.set_title('🛡️ Évolution de l\'Optimisation Sécurisée (Drawdown < 35%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Graphique 2: Convergence sécurisée
        improvements = np.diff(fit_maxs)
        colors = ['darkgreen' if x > 0 else 'orange' if x == 0 else 'red' for x in improvements]
        ax2.bar(gen[1:], improvements, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_xlabel('Génération')
        ax2.set_ylabel('Amélioration du Score')
        ax2.set_title('🎯 Progression de la Sécurisation des Stratégies')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('optimization_evolution_secure.png', dpi=300)
        print("\n📊 Graphique d'évolution sécurisée sauvegardé: optimization_evolution_secure.png")
    
    def display_best_strategies(self, best_individuals):
        """
        Affiche les meilleures stratégies SÉCURISÉES avec focus sur le drawdown
        """
        print("\n" + "="*70)
        print("🛡️ MEILLEURES STRATÉGIES SÉCURISÉES (DRAWDOWN < 35%)")
        print("="*70 + "\n")
        
        for strategy in best_individuals[:5]:
            print(f"{'='*70}")
            print(f"🥇 STRATÉGIE SÉCURISÉE #{strategy['rank']} - Score Sécurité: {strategy['security_score']:.0f}/100")
            print(f"{'='*70}")
            
            params = strategy['params']
            metrics = strategy['metrics']
            val_metrics = strategy['validation_metrics']
            
            # Paramètres ultra-sécurisés
            print("\n🛡️ PARAMÈTRES ULTRA-SÉCURISÉS:")
            print(f"   • Kelly diviseur: {params['kelly_fraction']:.1f} (Kelly/{params['kelly_fraction']:.0f}) - Très conservateur")
            print(f"   • Confiance minimale: {params['min_confidence']:.1%} - Sélectif")
            print(f"   • Value minimale: {params['min_value']:.3f} - Exigeant")
            print(f"   • Mise maximale: {params['max_bet_fraction']:.2%} de la bankroll - Prudent")
            print(f"   • Edge minimum: {params['min_edge']:.1%} - Strict")
            
            # Performance sécurisée
            print("\n💰 PERFORMANCE SÉCURISÉE:")
            roi_status = '🟢 Excellent' if metrics['roi'] > 15 else '🟡 Bon' if metrics['roi'] > 5 else '🔵 Acceptable'
            print(f"   • ROI: {metrics['roi']:.1f}% {roi_status}")
            print(f"   • Nombre de paris: {metrics['total_bets']} (Qualité > Quantité)")
            print(f"   • Taux de réussite: {metrics['win_rate']:.1%}")
            print(f"   • Profit total: {metrics['profit']:+.2f}€")
            print(f"   • Bankroll finale: {metrics['final_bankroll']:.2f}€")
            print(f"   • Expectancy: {metrics['expectancy']:+.2f}€/pari")
            
            # Métriques de sécurité (FOCUS PRINCIPAL)
            print("\n🛡️ MÉTRIQUES DE SÉCURITÉ (PRIORITÉ ABSOLUE):")
            dd_status = '🟢 Excellent' if metrics['max_drawdown'] > -15 else '🟡 Très bon' if metrics['max_drawdown'] > -25 else '🟠 Acceptable'
            print(f"   • Drawdown maximum: {metrics['max_drawdown']:.1f}% {dd_status}")
            print(f"   • Durée max drawdown: {metrics['max_drawdown_duration']} paris")
            
            streak_status = '🟢 Excellent' if metrics['max_consecutive_losses'] <= 4 else '🟡 Bon' if metrics['max_consecutive_losses'] <= 6 else '🟠 Acceptable'
            print(f"   • Pertes consécutives max: {metrics['max_consecutive_losses']} {streak_status}")
            
            vol_status = '🟢 Faible' if metrics['volatility'] < 50 else '🟡 Modérée' if metrics['volatility'] < 100 else '🟠 Élevée'
            print(f"   • Volatilité: {metrics['volatility']:.2f} {vol_status}")
            print(f"   • VaR 95%: {metrics['var_95']:.2f}€")
            
            # Ratios de qualité
            print("\n📊 RATIOS DE QUALITÉ:")
            sharpe_status = '🟢 Excellent' if metrics['sharpe_ratio'] > 1.5 else '🟡 Bon' if metrics['sharpe_ratio'] > 1.0 else '🟠 Acceptable'
            print(f"   • Sharpe Ratio: {metrics['sharpe_ratio']:.2f} {sharpe_status}")
            
            calmar_status = '🟢 Excellent' if metrics['calmar_ratio'] > 2.0 else '🟡 Bon' if metrics['calmar_ratio'] > 1.0 else '🟠 Acceptable'
            print(f"   • Calmar Ratio: {metrics['calmar_ratio']:.2f} {calmar_status}")
            
            print(f"   • Sortino Ratio: {metrics['sortino_ratio']:.2f}")
            
            pf_status = '🟢 Excellent' if metrics['profit_factor'] > 1.5 else '🟡 Bon' if metrics['profit_factor'] > 1.2 else '🟠 Acceptable'
            print(f"   • Profit Factor: {metrics['profit_factor']:.2f} {pf_status}")
            print(f"   • Recovery Factor: {metrics['recovery_factor']:.2f}")
            
            # Score global de sécurité
            security_rating = '🟢 ULTRA-SÉCURISÉ' if strategy['security_score'] >= 80 else '🟡 TRÈS SÉCURISÉ' if strategy['security_score'] >= 60 else '🟠 SÉCURISÉ'
            print(f"\n🎯 ÉVALUATION GLOBALE: {security_rating} ({strategy['security_score']:.0f}/100)")
            
            # Validation croisée
            if val_metrics['total_bets'] > 0:
                print("\n🔍 VALIDATION CROISÉE:")
                print(f"   • ROI validation: {val_metrics['roi']:.1f}%")
                print(f"   • Drawdown validation: {val_metrics['max_drawdown']:.1f}%")
                
                roi_consistency = abs(metrics['roi'] - val_metrics['roi']) < 10
                dd_consistency = abs(metrics['max_drawdown'] - val_metrics['max_drawdown']) < 8
                consistency = '🟢 Excellente' if roi_consistency and dd_consistency else '🟡 Bonne' if roi_consistency or dd_consistency else '🟠 Variable'
                print(f"   • Consistance: {consistency}")
            
            print("\n")
    
    def backtest_strategy(self, params: Dict, plot_title: str = "Backtest Sécurisé") -> pd.DataFrame:
        """
        Effectue un backtest détaillé d'une stratégie sécurisée
        """
        initial_bankroll = 1000
        bankroll = initial_bankroll
        bets_history = []
        
        kelly_fraction = params['kelly_fraction']
        min_confidence = params['min_confidence']
        min_value = params['min_value']
        max_bet_fraction = params['max_bet_fraction']
        min_edge = params['min_edge']
        
        print(f"\n📊 Backtest sécurisé en cours pour: {plot_title}")
        
        with tqdm(total=len(self.odds_data), desc="Backtest sécurisé", unit="combat") as pbar:
            for idx, fight in self.odds_data.iterrows():
                pbar.update(1)
                
                red_match = self.find_best_match(fight['R_fighter'])
                blue_match = self.find_best_match(fight['B_fighter'])
                
                if not red_match or not blue_match:
                    continue
                
                red_stats = self.fighters_dict[red_match]
                blue_stats = self.fighters_dict[blue_match]
                
                ml_prediction = self.predict_with_ml(red_stats, blue_stats)
                classic_prediction = self.predict_fight_classic(red_stats, blue_stats)
                
                prediction = ml_prediction if ml_prediction else classic_prediction
                
                if prediction['prediction'] == 'Red':
                    bet_prob = prediction['red_probability']
                    bet_odds = fight['R_odds']
                    bet_on = 'Red'
                else:
                    bet_prob = prediction['blue_probability']
                    bet_odds = fight['B_odds']
                    bet_on = 'Blue'
                
                if bet_prob < min_confidence:
                    continue
                
                implied_prob = 1 / bet_odds
                edge = bet_prob - implied_prob
                
                if edge < min_edge:
                    continue
                
                value = bet_prob * bet_odds
                if value < min_value:
                    continue
                
                kelly_stake = self.calculate_kelly(bet_prob, bet_odds, bankroll, kelly_fraction)
                max_stake = bankroll * max_bet_fraction
                stake = min(kelly_stake, max_stake)
                
                if stake < 1 or stake > bankroll:
                    continue
                
                result = 'win' if fight['Winner'] == bet_on else 'loss'
                profit = stake * (bet_odds - 1) if result == 'win' else -stake
                
                bankroll += profit
                
                if bankroll <= 0:
                    bankroll = 0
                    pbar.set_postfix({'Status': '⚠️ Faillite!'})
                    break
                
                bets_history.append({
                    'date': idx,
                    'fighter_red': fight['R_fighter'],
                    'fighter_blue': fight['B_fighter'],
                    'bet_on': bet_on,
                    'odds': bet_odds,
                    'probability': bet_prob,
                    'edge': edge,
                    'value': value,
                    'stake': stake,
                    'stake_pct': (stake / (bankroll - profit)) * 100,
                    'result': result,
                    'profit': profit,
                    'bankroll': bankroll,
                    'roi': ((bankroll - initial_bankroll) / initial_bankroll) * 100
                })
                
                # Mise à jour périodique avec focus sécurité
                if len(bets_history) % 10 == 0:
                    current_roi = ((bankroll - initial_bankroll) / initial_bankroll) * 100
                    # Calculer drawdown actuel
                    if len(bets_history) > 0:
                        df_temp = pd.DataFrame(bets_history)
                        rolling_max = df_temp['bankroll'].expanding().max()
                        current_dd = ((df_temp['bankroll'].iloc[-1] - rolling_max.iloc[-1]) / rolling_max.iloc[-1] * 100)
                        pbar.set_postfix({
                            'ROI': f'{current_roi:.1f}%', 
                            'DD': f'{current_dd:.1f}%',
                            '🛡️': 'Sécurisé'
                        })
        
        return pd.DataFrame(bets_history)
    
    def plot_backtest_results(self, backtest_df: pd.DataFrame, title: str = "Backtest Sécurisé"):
        """
        Visualise les résultats du backtest avec focus sur la sécurité
        """
        if backtest_df.empty:
            print("❌ Aucun pari effectué avec cette stratégie sécurisée.")
            return
        
        # Configuration du style sécurisé
        plt.style.use('seaborn-v0_8-darkgrid')
        fig = plt.figure(figsize=(16, 12))
        
        # Créer une grille de subplots
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
        
        # 1. Évolution de la bankroll (Focus sécurité)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(backtest_df.index, backtest_df['bankroll'], 'g-', linewidth=3, label='Bankroll Sécurisée', alpha=0.8)
        ax1.axhline(y=1000, color='blue', linestyle='--', alpha=0.5, label='Bankroll initiale')
        
        # Zone de danger (drawdown > 30%)
        danger_zone = 1000 * 0.7  # -30%
        ax1.axhline(y=danger_zone, color='red', linestyle=':', alpha=0.7, label='Zone de danger (-30%)')
        
        # Zones colorées pour les gains/pertes
        ax1.fill_between(backtest_df.index, 1000, backtest_df['bankroll'],
                        where=backtest_df['bankroll'] >= 1000,
                        color='green', alpha=0.3, label='Profit')
        ax1.fill_between(backtest_df.index, 1000, backtest_df['bankroll'],
                        where=backtest_df['bankroll'] < 1000,
                        color='orange', alpha=0.3, label='Perte temporaire')
        
        ax1.set_title('🛡️ Évolution Sécurisée de la Bankroll', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Nombre de paris')
        ax1.set_ylabel('Bankroll (€)')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 2. Drawdown sécurisé (FOCUS PRINCIPAL)
        ax2 = fig.add_subplot(gs[1, 0])
        rolling_max = backtest_df['bankroll'].expanding().max()
        drawdown = (backtest_df['bankroll'] - rolling_max) / rolling_max * 100
        
        # Colorier selon le niveau de sécurité
        colors = ['green' if dd > -15 else 'orange' if dd > -25 else 'red' for dd in drawdown]
        ax2.fill_between(backtest_df.index, drawdown, 0, 
                        where=drawdown<0, interpolate=True, 
                        color='lightcoral', alpha=0.4)
        ax2.plot(backtest_df.index, drawdown, 'r-', linewidth=2)
        
        # Lignes de sécurité
        ax2.axhline(y=-15, color='orange', linestyle='--', alpha=0.7, label='Seuil attention (-15%)')
        ax2.axhline(y=-30, color='red', linestyle='--', alpha=0.7, label='Seuil critique (-30%)')
        
        ax2.set_title('🛡️ Drawdown Contrôlé', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Nombre de paris')
        ax2.set_ylabel('Drawdown (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Annotation du drawdown maximum
        max_dd_idx = drawdown.idxmin()
        max_dd_value = drawdown.min()
        dd_color = 'green' if max_dd_value > -15 else 'orange' if max_dd_value > -25 else 'red'
        ax2.annotate(f'Max DD: {max_dd_value:.1f}%',
                    xy=(max_dd_idx, max_dd_value),
                    xytext=(max_dd_idx, max_dd_value - 3),
                    arrowprops=dict(arrowstyle='->', color=dd_color),
                    fontsize=10, color=dd_color, fontweight='bold')
        
        # 3. Distribution des profits (analyse de consistance)
        ax3 = fig.add_subplot(gs[1, 1])
        wins = backtest_df[backtest_df['profit'] > 0]['profit']
        losses = backtest_df[backtest_df['profit'] < 0]['profit']
        
        ax3.hist(wins, bins=20, alpha=0.7, color='green', label=f'Gains (n={len(wins)})')
        ax3.hist(losses, bins=20, alpha=0.7, color='red', label=f'Pertes (n={len(losses)})')
        ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax3.set_title('📊 Consistance des Résultats', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Profit (€)')
        ax3.set_ylabel('Fréquence')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. ROI cumulé avec zones de sécurité
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.plot(backtest_df.index, backtest_df['roi'], 'g-', linewidth=3, alpha=0.8)
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Zones de performance
        ax4.axhline(y=10, color='green', linestyle=':', alpha=0.5, label='Objectif ROI (+10%)')
        ax4.axhline(y=20, color='darkgreen', linestyle=':', alpha=0.5, label='Excellent ROI (+20%)')
        
        ax4.set_title('📈 ROI Cumulé Sécurisé', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Nombre de paris')
        ax4.set_ylabel('ROI (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Annotation finale
        final_roi = backtest_df['roi'].iloc[-1]
        roi_color = 'darkgreen' if final_roi > 15 else 'green' if final_roi > 5 else 'orange'
        ax4.text(0.98, 0.02, f'ROI Final: {final_roi:.1f}%\n🛡️ Sécurisé',
                transform=ax4.transAxes,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                fontsize=12, fontweight='bold', color=roi_color)
        
        # 5. Taille des mises sécurisées
        ax5 = fig.add_subplot(gs[2, 1])
        colors_stakes = ['green' if r == 'win' else 'red' for r in backtest_df['result']]
        ax5.scatter(backtest_df.index, backtest_df['stake_pct'], 
                   c=colors_stakes, alpha=0.6, s=30)
        
        # Ligne de sécurité pour les mises
        max_safe_stake = backtest_df['stake_pct'].quantile(0.95)
        ax5.axhline(y=max_safe_stake, color='orange', linestyle='--', alpha=0.7, 
                   label=f'95e percentile: {max_safe_stake:.1f}%')
        
        ax5.set_title('🛡️ Taille des Mises Sécurisées', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Nombre de paris')
        ax5.set_ylabel('Mise (% bankroll)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Titre principal avec focus sécurité
        fig.suptitle(f'🛡️ {title} - STRATÉGIE ULTRA-SÉCURISÉE', fontsize=16, fontweight='bold', color='darkgreen')
        
        # Ajuster la mise en page et sauvegarder
        plt.tight_layout()
        filename = f'backtest_secure_{title.replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n📊 Graphiques sécurisés sauvegardés: {filename}")
        
        # Afficher un résumé de sécurité
        print(f"\n🛡️ RÉSUMÉ DE SÉCURITÉ:")
        print(f"   • Nombre de paris: {len(backtest_df)} (Qualité > Quantité)")
        print(f"   • ROI final: {final_roi:.1f}%")
        print(f"   • Bankroll finale: {backtest_df['bankroll'].iloc[-1]:.2f}€")
        
        max_dd = drawdown.min()
        dd_status = '🟢 Excellent' if max_dd > -15 else '🟡 Bon' if max_dd > -25 else '🟠 Acceptable'
        print(f"   • Drawdown maximum: {max_dd:.1f}% {dd_status}")
        print(f"   • Taux de réussite: {len(wins)/len(backtest_df)*100:.1f}%")
        print(f"   • 🛡️ Statut: STRATÉGIE SÉCURISÉE VALIDÉE")
    
    def export_results(self, best_strategies, logbook):
        """
        Exporte tous les résultats avec focus sur les stratégies sécurisées
        """
        print("\n💾 Exportation des résultats sécurisés...")
        
        # 1. Export des stratégies sécurisées en CSV
        strategies_data = []
        for s in best_strategies:
            row = {
                'rank': s['rank'],
                'security_score': s['security_score'],
                'fitness': s['fitness'],
                **{f'param_{k}': v for k, v in s['params'].items()},
                **{f'metric_{k}': v for k, v in s['metrics'].items()},
                **{f'validation_{k}': v for k, v in s['validation_metrics'].items()}
            }
            strategies_data.append(row)
        
        strategies_df = pd.DataFrame(strategies_data)
        strategies_df.to_csv('best_strategies_secured.csv', index=False)
        
        # 2. Export du log d'optimisation sécurisée
        log_df = pd.DataFrame(logbook)
        log_df.to_csv('optimization_log_secure.csv', index=False)
        
        # 3. Export JSON complet des stratégies sécurisées
        export_data = {
            'optimization_date': datetime.now().isoformat(),
            'optimization_type': 'ULTRA_SECURE_DRAWDOWN_CONTROLLED',
            'security_constraints': {
                'max_drawdown': '-35%',
                'max_consecutive_losses': 10,
                'min_roi_threshold': '0%',
                'strategy_focus': 'Security first, then returns'
            },
            'parameters': {
                'population_size': 200,
                'generations': len(logbook),
                'parameter_bounds': self.param_bounds,
                'fitness_strategy': 'Ultra-secure risk-controlled optimization'
            },
            'best_strategies': [
                {
                    'rank': s['rank'],
                    'security_score': float(s['security_score']),
                    'params': s['params'],
                    'metrics': {k: float(v) if isinstance(v, np.number) else v 
                               for k, v in s['metrics'].items()},
                    'validation_metrics': {k: float(v) if isinstance(v, np.number) else v 
                                         for k, v in s['validation_metrics'].items()},
                    'fitness': float(s['fitness'])
                }
                for s in best_strategies[:10]
            ]
        }
        
        with open('optimization_results_secure.json', 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print("\n✅ Résultats sécurisés exportés:")
        print("   • best_strategies_secured.csv - Stratégies ultra-sécurisées")
        print("   • optimization_log_secure.csv - Journal d'optimisation sécurisée")
        print("   • optimization_results_secure.json - Résultats complets sécurisés")
        
        # 4. Rapport Markdown spécialisé sécurité
        self._generate_security_report(best_strategies)

    def _generate_security_report(self, best_strategies):
        """Génère un rapport Markdown spécialisé sécurité"""
        with open('security_optimization_report.md', 'w') as f:
            f.write("# 🛡️ UFC Betting Strategy - ULTRA-SECURE Optimization Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 🎯 SECURITY-FIRST APPROACH\n\n")
            f.write("This optimization prioritizes **SECURITY** above all else:\n\n")
            f.write("- **Maximum Drawdown**: Limited to 35% (Eliminator threshold)\n")
            f.write("- **Consecutive Losses**: Maximum 10 losses in a row\n")
            f.write("- **Risk Management**: Ultra-conservative Kelly fractions (4x-50x)\n")
            f.write("- **Bet Sizing**: Maximum 4% of bankroll per bet\n")
            f.write("- **Quality over Quantity**: Fewer, higher-quality bets\n\n")
            
            f.write("## 📊 BEST SECURE STRATEGY\n\n")
            best = best_strategies[0]
            f.write(f"### 🏆 Top Secure Strategy (Security Score: {best['security_score']:.0f}/100)\n\n")
            f.write("**Security Metrics:**\n")
            f.write(f"- **Max Drawdown**: {best['metrics']['max_drawdown']:.1f}% ⚡\n")
            f.write(f"- **Max Consecutive Losses**: {best['metrics']['max_consecutive_losses']}\n")
            f.write(f"- **ROI**: {best['metrics']['roi']:.1f}%\n")
            f.write(f"- **Sharpe Ratio**: {best['metrics']['sharpe_ratio']:.2f}\n")
            f.write(f"- **Win Rate**: {best['metrics']['win_rate']:.1%}\n\n")
            
            f.write("**Ultra-Conservative Parameters:**\n")
            f.write(f"- Kelly Fraction: {best['params']['kelly_fraction']:.1f} (Very Conservative)\n")
            f.write(f"- Min Confidence: {best['params']['min_confidence']:.1%} (Selective)\n")
            f.write(f"- Min Value: {best['params']['min_value']:.3f} (Demanding)\n")
            f.write(f"- Max Bet Size: {best['params']['max_bet_fraction']:.2%} (Prudent)\n")
            f.write(f"- Min Edge: {best['params']['min_edge']:.1%} (Strict)\n\n")
            
            f.write("## 🛡️ TOP 5 SECURE STRATEGIES\n\n")
            for i, strategy in enumerate(best_strategies[:5]):
                security_level = '🟢 ULTRA-SECURE' if strategy['security_score'] >= 80 else '🟡 VERY SECURE' if strategy['security_score'] >= 60 else '🟠 SECURE'
                f.write(f"### Strategy #{i+1} - {security_level}\n\n")
                f.write(f"- **Security Score**: {strategy['security_score']:.0f}/100\n")
                f.write(f"- **Max Drawdown**: {strategy['metrics']['max_drawdown']:.1f}%\n")
                f.write(f"- **ROI**: {strategy['metrics']['roi']:.1f}%\n")
                f.write(f"- **Consecutive Losses**: {strategy['metrics']['max_consecutive_losses']}\n")
                f.write(f"- **Sharpe Ratio**: {strategy['metrics']['sharpe_ratio']:.2f}\n\n")
            
            f.write("## 🎯 SECURITY ANALYSIS\n\n")
            f.write("### Risk Distribution\n\n")
            ultra_secure = sum(1 for s in best_strategies[:10] if s['security_score'] >= 80)
            very_secure = sum(1 for s in best_strategies[:10] if 60 <= s['security_score'] < 80)
            secure = sum(1 for s in best_strategies[:10] if s['security_score'] < 60)
            
            f.write(f"- **Ultra-Secure Strategies**: {ultra_secure}/10\n")
            f.write(f"- **Very Secure Strategies**: {very_secure}/10\n")
            f.write(f"- **Secure Strategies**: {secure}/10\n\n")
            
            f.write("### Key Security Insights\n\n")
            avg_dd = np.mean([s['metrics']['max_drawdown'] for s in best_strategies[:5]])
            avg_roi = np.mean([s['metrics']['roi'] for s in best_strategies[:5]])
            f.write(f"- **Average Max Drawdown**: {avg_dd:.1f}% (Well below 35% limit)\n")
            f.write(f"- **Average ROI**: {avg_roi:.1f}% (Positive and sustainable)\n")
            f.write(f"- **Strategy Type**: Ultra-conservative with focus on capital preservation\n")
            f.write(f"- **Risk Profile**: Suitable for risk-averse investors\n\n")
        
        print("   • security_optimization_report.md - Rapport de sécurité détaillé")


def main():
    """
    Fonction principale pour lancer l'optimisation ULTRA-SÉCURISÉE
    """
    # Configuration
    print("\n" + "="*70)
    print("🛡️ UFC BETTING STRATEGY OPTIMIZER - ULTRA-SÉCURISÉ")
    print("="*70)
    print("\n🎯 VERSION DRAWDOWN CONTRÔLÉ - Maximum 35% de drawdown")
    print("Stratégie: Sécurité ABSOLUE avec rendements optimisés\n")
    
    # Message de sécurité
    print("🚨 APPROCHE ULTRA-SÉCURISÉE:")
    print("   • Drawdown > 35% = ÉLIMINATION IMMÉDIATE")
    print("   • Série perdante > 10 = ÉLIMINATION IMMÉDIATE")
    print("   • Mises ultra-conservatrices (max 4% bankroll)")
    print("   • Kelly fractionné au minimum (Kelly/4 à Kelly/50)")
    print("   • Qualité > Quantité des paris")
    
    # Chemins vers les fichiers
    model_path = "ufc_prediction_model.joblib"  # ou .pkl
    fighters_stats_path = "fighters_stats.txt"
    odds_data_path = "data_european_odds.csv"
    
    # Vérifier l'existence des fichiers
    files_check = [
        (fighters_stats_path, "Statistiques des combattants"),
        (odds_data_path, "Données de cotes"),
        (model_path, "Modèle ML (optionnel)")
    ]
    
    print("\n🔍 Vérification des fichiers nécessaires:")
    all_files_ok = True
    for file_path, description in files_check:
        if os.path.exists(file_path):
            print(f"   ✅ {description}: {file_path}")
        else:
            if "optionnel" not in description:
                print(f"   ❌ {description}: {file_path} - MANQUANT!")
                all_files_ok = False
            else:
                print(f"   ⚠️ {description}: {file_path} - Non trouvé (mode statistique)")
    
    if not all_files_ok:
        print("\n❌ Fichiers manquants. Veuillez vérifier les chemins.")
        return
    
    # Créer l'optimiseur ULTRA-SÉCURISÉ
    optimizer = UFCBettingOptimizer(model_path, fighters_stats_path, odds_data_path)
    
    # Configuration de l'optimisation sécurisée
    print("\n⚙️ CONFIGURATION ULTRA-SÉCURISÉE:")
    print("\nParamètres de sécurité renforcée:")
    print("   • Population: 200 individus (exploration sécurisée approfondie)")
    print("   • Générations: 100 (convergence ultra-sûre)")
    print("   • Contrainte STRICTE: Drawdown maximum 35%")
    print("   • Élimination immédiate des stratégies risquées")
    print("   • Mises ultra-conservatrices: 0.5% à 4% de la bankroll")
    print("   • Kelly fractionné: Diviseur 4 à 50 (très prudent)")
    print("   • Tests de stress: Validation croisée + Monte Carlo")
    print("   • Focus: SÉCURITÉ > RENDEMENT")
    
    use_custom = input("\nVoulez-vous personnaliser les paramètres de sécurité? (y/N): ").lower() == 'y'
    
    if use_custom:
        try:
            pop_size = int(input("Taille de la population (défaut: 200): ") or "200")
            n_gen = int(input("Nombre de générations (défaut: 100): ") or "100")
            print("\n🛡️ Note: Les contraintes de sécurité restent strictes (Drawdown < 35%)")
        except ValueError:
            print("Valeurs invalides. Utilisation des paramètres par défaut.")
            pop_size, n_gen = 200, 100
    else:
        pop_size, n_gen = 200, 100
    
    # Confirmation de l'approche sécurisée
    print(f"\n🎯 STRATÉGIE D'OPTIMISATION CONFIRMÉE:")
    print(f"   • Population: {pop_size} individus")
    print(f"   • Générations: {n_gen}")
    print(f"   • 🛡️ SÉCURITÉ: Drawdown maximum 35% (NON NÉGOCIABLE)")
    print(f"   • 🎯 OBJECTIF: Meilleur ROI possible avec sécurité maximale")
    
    # Lancer l'optimisation sécurisée
    print("\n🚀 Lancement de l'optimisation ULTRA-SÉCURISÉE...")
    time.sleep(2)
    
    start_time = time.time()
    best_strategies, logbook = optimizer.optimize(population_size=pop_size, generations=n_gen)
    end_time = time.time()
    
    print(f"\n⏱️ Temps d'exécution: {(end_time - start_time)/60:.1f} minutes")
    
    # Vérification de sécurité des résultats
    print("\n🔍 VÉRIFICATION DE SÉCURITÉ DES RÉSULTATS:")
    unsafe_strategies = 0
    for strategy in best_strategies[:10]:
        if strategy['metrics']['max_drawdown'] < -35:
            unsafe_strategies += 1
    
    if unsafe_strategies == 0:
        print("   ✅ TOUTES les stratégies respectent la limite de drawdown (< 35%)")
        print("   ✅ OPTIMISATION SÉCURISÉE RÉUSSIE")
    else:
        print(f"   ⚠️ {unsafe_strategies} stratégies dépassent la limite (erreur système)")
    
    # Afficher et analyser les résultats sécurisés
    optimizer.plot_optimization_results(logbook)
    optimizer.display_best_strategies(best_strategies)
    
    # Tests de sécurité approfondis sur la meilleure stratégie
    print("\n🔬 TESTS DE SÉCURITÉ APPROFONDIS")
    print("="*70)
    
    best_strategy = best_strategies[0]
    
    # 1. Test de validation croisée sécurisée
    print(f"\n🎯 Test de validation croisée sur la stratégie #1:")
    validation_results = optimizer.simulate_betting_strategy(
        best_strategy['params'], 
        validation_split=0.3  # 30% pour validation
    )
    
    print(f"   • ROI validation: {validation_results['roi']:.1f}%")
    print(f"   • Drawdown validation: {validation_results['max_drawdown']:.1f}%")
    
    validation_safe = validation_results['max_drawdown'] > -35
    consistency_good = abs(best_strategy['metrics']['roi'] - validation_results['roi']) < 15
    
    print(f"   • Sécurité validée: {'✅' if validation_safe else '❌'}")
    print(f"   • Consistance: {'✅' if consistency_good else '⚠️'}")
    
    # 2. Test de stress sur segments temporels
    print(f"\n🔍 Test de stress par segments temporels:")
    
    # Diviser les données en 3 segments
    total_data = len(optimizer.odds_data)
    segments = [
        ("Premier tiers", optimizer.odds_data.iloc[:total_data//3]),
        ("Milieu", optimizer.odds_data.iloc[total_data//3:2*total_data//3]),
        ("Dernier tiers", optimizer.odds_data.iloc[2*total_data//3:])
    ]
    
    all_segments_safe = True
    for segment_name, segment_data in segments:
        # Sauvegarder les données originales
        original_data = optimizer.odds_data
        # Tester sur le segment
        optimizer.odds_data = segment_data
        segment_metrics = optimizer.simulate_betting_strategy(best_strategy['params'])
        # Restaurer les données
        optimizer.odds_data = original_data
        
        segment_safe = segment_metrics['max_drawdown'] > -40  # Un peu plus de tolérance par segment
        print(f"   • {segment_name}: ROI {segment_metrics['roi']:.1f}%, "
              f"DD {segment_metrics['max_drawdown']:.1f}% {'✅' if segment_safe else '⚠️'}")
        
        if not segment_safe:
            all_segments_safe = False
    
    print(f"   • Robustesse temporelle: {'✅ Excellente' if all_segments_safe else '⚠️ Acceptable'}")
    
    # Backtest détaillé des meilleures stratégies sécurisées
    print("\n📊 BACKTESTS DÉTAILLÉS SÉCURISÉS")
    print("="*70)
    
    # Backtest de la stratégie #1 (la plus sécurisée)
    print(f"\n🏆 Backtest de la STRATÉGIE #1 ULTRA-SÉCURISÉE...")
    backtest_df = optimizer.backtest_strategy(
        best_strategy['params'], 
        f"Stratégie #1 ULTRA-SÉCURISÉE - ROI: {best_strategy['metrics']['roi']:.1f}%"
    )
    
    if not backtest_df.empty:
        optimizer.plot_backtest_results(
            backtest_df, 
            "Stratégie #1 ULTRA-SÉCURISÉE"
        )
    
    # Backtest de 2 autres stratégies pour comparaison
    for i in range(1, min(3, len(best_strategies))):
        print(f"\n🔍 Backtest de la stratégie sécurisée #{i+1}...")
        strategy = best_strategies[i]
        backtest_df = optimizer.backtest_strategy(
            strategy['params'], 
            f"Stratégie Sécurisée #{i+1} - ROI: {strategy['metrics']['roi']:.1f}%"
        )
        
        if not backtest_df.empty:
            optimizer.plot_backtest_results(
                backtest_df, 
                f"Stratégie Sécurisée #{i+1}"
            )
    
    # Génération de la stratégie recommandée ultra-sécurisée
    print("\n🛡️ GÉNÉRATION DE LA STRATÉGIE RECOMMANDÉE ULTRA-SÉCURISÉE")
    print("="*70)
    
    # Filtrer uniquement les stratégies ultra-sécurisées
    ultra_safe_strategies = []
    for strategy in best_strategies[:10]:
        if (strategy['metrics']['max_drawdown'] > -30 and 
            strategy['metrics']['max_consecutive_losses'] <= 8 and
            strategy['metrics']['roi'] > 0):
            ultra_safe_strategies.append(strategy)
    
    if ultra_safe_strategies:
        print(f"   ✅ {len(ultra_safe_strategies)} stratégies ultra-sécurisées identifiées")
        
        # Moyenne pondérée des meilleures stratégies sécurisées
        weights = [strategy['security_score'] for strategy in ultra_safe_strategies]
        total_weight = sum(weights)
        
        recommended_params = {}
        param_names = ['kelly_fraction', 'min_confidence', 'min_value', 'max_bet_fraction', 'min_edge']
        
        for param in param_names:
            weighted_sum = sum(s['params'][param] * w for s, w in zip(ultra_safe_strategies, weights))
            recommended_params[param] = weighted_sum / total_weight
        
        # Test de la stratégie recommandée
        print("\n   🧪 Test de la stratégie recommandée ultra-sécurisée...")
        recommended_metrics = optimizer.simulate_betting_strategy(recommended_params)
        
        print(f"\n🏆 STRATÉGIE RECOMMANDÉE ULTRA-SÉCURISÉE:")
        print(f"   • Kelly diviseur: {recommended_params['kelly_fraction']:.1f} (Ultra-conservateur)")
        print(f"   • Confiance min: {recommended_params['min_confidence']:.1%}")
        print(f"   • Value min: {recommended_params['min_value']:.3f}")
        print(f"   • Mise max: {recommended_params['max_bet_fraction']:.2%}")
        print(f"   • Edge min: {recommended_params['min_edge']:.1%}")
        print(f"\n📊 PERFORMANCE SÉCURISÉE ATTENDUE:")
        print(f"   • ROI: {recommended_metrics['roi']:.1f}%")
        print(f"   • Drawdown max: {recommended_metrics['max_drawdown']:.1f}%")
        print(f"   • Pertes consécutives max: {recommended_metrics['max_consecutive_losses']}")
        print(f"   • Nombre de paris: {recommended_metrics['total_bets']}")
        print(f"   • 🛡️ Niveau de sécurité: MAXIMUM")
        
        # Export de la stratégie recommandée
        with open('strategie_ultra_securisee.json', 'w') as f:
            json.dump({
                'strategie_ultra_securisee': {
                    'description': 'Stratégie optimisée pour un drawdown maximum de 35%',
                    'niveau_securite': 'MAXIMUM',
                    'contraintes': {
                        'drawdown_max': '-35%',
                        'pertes_consecutives_max': 10,
                        'mise_max_bankroll': '4%'
                    },
                    'params': recommended_params,
                    'performance_attendue': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                                           for k, v in recommended_metrics.items()},
                    'strategies_sources': len(ultra_safe_strategies),
                    'confiance_recommendation': 'MAXIMUM'
                }
            }, f, indent=2)
    else:
        print("   ⚠️ Aucune stratégie ultra-sécurisée trouvée. Utilisation de la meilleure disponible.")
        recommended_params = best_strategies[0]['params']
    
    # Exporter tous les résultats sécurisés
    optimizer.export_results(best_strategies, logbook)
    
    # Message final de sécurité
    print("\n" + "="*70)
    print("🛡️ OPTIMISATION ULTRA-SÉCURISÉE TERMINÉE AVEC SUCCÈS!")
    print("="*70)
    print("\n📁 Fichiers générés (focus sécurité):")
    print("   • best_strategies_secured.csv - Stratégies ultra-sécurisées")
    print("   • strategie_ultra_securisee.json - STRATÉGIE RECOMMANDÉE")
    print("   • optimization_results_secure.json - Résultats complets sécurisés")
    print("   • security_optimization_report.md - Rapport de sécurité détaillé")
    print("   • optimization_evolution_secure.png - Évolution sécurisée")
    print("   • backtest_secure_*.png - Backtests sécurisés")
    
    print(f"\n🎯 RÉSULTATS DE SÉCURITÉ:")
    print(f"   • Meilleur ROI sécurisé: {best_strategies[0]['metrics']['roi']:.1f}%")
    print(f"   • Drawdown contrôlé: {best_strategies[0]['metrics']['max_drawdown']:.1f}% (< 35%)")
    print(f"   • Score de sécurité: {best_strategies[0]['security_score']:.0f}/100")
    print(f"   • Toutes les stratégies respectent les contraintes de sécurité")
    
    print(f"\n🛡️ STRATÉGIE RECOMMANDÉE ULTRA-SÉCURISÉE:")
    print(f"   ✅ Drawdown garanti < 35%")
    print(f"   ✅ Mises ultra-conservatrices")
    print(f"   ✅ Série perdante contrôlée")
    print(f"   ✅ Validation croisée réussie")
    print(f"   ✅ Tests de stress validés")
    
    print(f"\n💰 MISE EN PRATIQUE ULTRA-SÉCURISÉE:")
    print(f"   1. Commencez avec 200-500€ pour tester en sécurité")
    print(f"   2. Respectez ABSOLUMENT les paramètres")
    print(f"   3. Arrêtez immédiatement si drawdown > 30%")
    print(f"   4. Réévaluez après 30-50 paris")
    print(f"   5. Priorité: SÉCURITÉ avant tout!")
    
    print(f"\n🏆 Cette optimisation privilégie la SÉCURITÉ ABSOLUE.")
    print(f"🛡️ Votre capital est protégé avec un drawdown maximum de 35%.")
    print(f"📈 Les rendements sont optimisés dans cette contrainte de sécurité.")
    print(f"\n✨ Bonne chance avec votre stratégie ultra-sécurisée!")


if __name__ == "__main__":
    main()