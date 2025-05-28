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
    Version améliorée avec état d'avancement détaillé
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
        print("🥊 UFC BETTING STRATEGY OPTIMIZER - INITIALISATION")
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
                'median_stake_pct': 0
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
            'median_stake_pct': median_stake_pct
        }
    
    def setup_genetic_algorithm(self):
        """
        Configure l'algorithme génétique avec DEAP
        Version améliorée avec paramètres optimisés
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
        
        # Définir les bornes des paramètres optimisées
        self.param_bounds = {
            'kelly_fraction': (2, 25),       # Kelly/2 à Kelly/25 (plus conservateur)
            'min_confidence': (0.58, 0.72),  # 58% à 72% (zone optimale)
            'min_value': (1.05, 1.25),       # Value minimale réduite
            'max_bet_fraction': (0.02, 0.08), # 2% à 8% de la bankroll
            'min_edge': (0.02, 0.12),        # 2% à 12% d'edge
            'volatility_penalty': (0.5, 3),   # Pénalité pour la volatilité
            'drawdown_penalty': (1, 5),       # Pénalité pour le drawdown
            'min_bets_penalty': (0.1, 1)     # Pénalité si pas assez de paris
        }
        
        # Générateurs d'attributs
        self.toolbox.register("kelly_fraction", random.uniform, *self.param_bounds['kelly_fraction'])
        self.toolbox.register("min_confidence", random.uniform, *self.param_bounds['min_confidence'])
        self.toolbox.register("min_value", random.uniform, *self.param_bounds['min_value'])
        self.toolbox.register("max_bet_fraction", random.uniform, *self.param_bounds['max_bet_fraction'])
        self.toolbox.register("min_edge", random.uniform, *self.param_bounds['min_edge'])
        self.toolbox.register("volatility_penalty", random.uniform, *self.param_bounds['volatility_penalty'])
        self.toolbox.register("drawdown_penalty", random.uniform, *self.param_bounds['drawdown_penalty'])
        self.toolbox.register("min_bets_penalty", random.uniform, *self.param_bounds['min_bets_penalty'])
        
        # Structure de l'individu
        self.toolbox.register("individual", tools.initCycle, creator.Individual,
                              (self.toolbox.kelly_fraction,
                               self.toolbox.min_confidence,
                               self.toolbox.min_value,
                               self.toolbox.max_bet_fraction,
                               self.toolbox.min_edge,
                               self.toolbox.volatility_penalty,
                               self.toolbox.drawdown_penalty,
                               self.toolbox.min_bets_penalty), n=1)
        
        # Population
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Opérateurs génétiques
        self.toolbox.register("evaluate", self.fitness_function)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.3)
        self.toolbox.register("mutate", self.mutate_individual)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
    
    def mutate_individual(self, individual):
        """
        Mutation personnalisée qui respecte les bornes des paramètres
        Version améliorée avec mutation adaptative
        """
        param_names = ['kelly_fraction', 'min_confidence', 'min_value', 
                      'max_bet_fraction', 'min_edge', 'volatility_penalty',
                      'drawdown_penalty', 'min_bets_penalty']
        
        # Mutation adaptative: plus forte au début, plus faible à la fin
        mutation_rate = 0.3 if not hasattr(self, 'current_generation') else max(0.1, 0.3 - self.current_generation * 0.005)
        
        for i, param_name in enumerate(param_names):
            if random.random() < mutation_rate:
                bounds = self.param_bounds[param_name]
                # Mutation gaussienne
                sigma = (bounds[1] - bounds[0]) * 0.1
                new_value = individual[i] + random.gauss(0, sigma)
                # Respecter les bornes
                individual[i] = max(bounds[0], min(bounds[1], new_value))
        
        return individual,
    
    def fitness_function(self, individual):
        """
        Fonction de fitness multi-objectifs améliorée
        Focus sur la minimisation du drawdown et la stabilité
        """
        # Convertir l'individu en dictionnaire de paramètres
        params = {
            'kelly_fraction': individual[0],
            'min_confidence': individual[1],
            'min_value': individual[2],
            'max_bet_fraction': individual[3],
            'min_edge': individual[4]
        }
        volatility_penalty_weight = individual[5]
        drawdown_penalty_weight = individual[6]
        min_bets_penalty_weight = individual[7]
        
        # Simuler la stratégie
        metrics = self.simulate_betting_strategy(params)
        
        # Si pas assez de paris, pénaliser
        min_bets_threshold = 20
        if metrics['total_bets'] < min_bets_threshold:
            bets_penalty = (min_bets_threshold - metrics['total_bets']) * min_bets_penalty_weight * 10
            return -1000 - bets_penalty,
        
        # Fonction de fitness multi-objectifs améliorée
        
        # 1. Composante de rendement
        roi_score = metrics['roi'] * 2  # Doubler l'importance du ROI
        
        # 2. Composante de risque
        drawdown_penalty = abs(metrics['max_drawdown']) * drawdown_penalty_weight
        volatility_penalty = metrics['volatility'] * volatility_penalty_weight
        consecutive_losses_penalty = metrics['max_consecutive_losses'] * 5
        
        # 3. Composante de qualité
        quality_score = 0
        if metrics['win_rate'] > 0.5:
            quality_score += (metrics['win_rate'] - 0.5) * 100
        
        # Bonus pour profit factor élevé
        if metrics['profit_factor'] > 1.5:
            quality_score += (metrics['profit_factor'] - 1.5) * 20
        
        # 4. Ratios de risque ajusté
        sharpe_score = max(0, metrics['sharpe_ratio'] * 15)
        calmar_score = max(0, metrics['calmar_ratio'] * 10)
        sortino_score = max(0, metrics['sortino_ratio'] * 8)
        
        # 5. Stabilité et consistance
        consistency_score = 0
        if metrics['max_drawdown'] > -20:  # Drawdown acceptable
            consistency_score += 20
        if metrics['max_consecutive_losses'] < 5:
            consistency_score += 15
        
        # Score final équilibré
        fitness = (
            roi_score 
            - drawdown_penalty 
            - volatility_penalty
            - consecutive_losses_penalty
            + quality_score
            + sharpe_score 
            + calmar_score
            + sortino_score
            + consistency_score
        )
        
        return fitness,
    
    def optimize(self, population_size=100, generations=50, n_jobs=-1):
        """
        Lance l'optimisation génétique avec état d'avancement détaillé
        """
        print("\n" + "="*70)
        print("🧬 DÉMARRAGE DE L'OPTIMISATION GÉNÉTIQUE")
        print("="*70)
        print(f"\n📊 Paramètres de l'optimisation:")
        print(f"   • Taille de la population: {population_size}")
        print(f"   • Nombre de générations: {generations}")
        print(f"   • Processeurs utilisés: {cpu_count() if n_jobs == -1 else n_jobs}")
        print(f"   • Espace de recherche: {len(self.param_bounds)} paramètres")
        print("\n" + "="*70 + "\n")
        
        # Statistiques
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Population initiale
        print("🌱 Création de la population initiale...")
        population = self.toolbox.population(n=population_size)
        
        # Hall of Fame pour garder les meilleurs individus
        hof = tools.HallOfFame(20)  # Augmenté à 20
        
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
        print("\n🔄 Évolution en cours...\n")
        
        best_fitness_history = []
        no_improvement_count = 0
        last_best_fitness = -float('inf')
        
        with tqdm(total=generations, desc="Optimisation", unit="génération") as pbar:
            for gen in range(1, generations + 1):
                self.current_generation = gen
                
                # Sélection
                offspring = self.toolbox.select(population, len(population))
                offspring = list(map(self.toolbox.clone, offspring))
                
                # Crossover
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < 0.7:  # Probabilité de crossover
                        self.toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values
                
                # Mutation
                for mutant in offspring:
                    if random.random() < 0.3:  # Probabilité de mutation
                        self.toolbox.mutate(mutant)
                        del mutant.fitness.values
                
                # Évaluation des nouveaux individus
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(self.toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
                
                # Remplacement
                population[:] = offspring
                
                # Mise à jour du Hall of Fame
                hof.update(population)
                
                # Enregistrement des statistiques
                record = stats.compile(population)
                logbook.record(gen=gen, evals=len(invalid_ind), **record)
                
                # Mise à jour de la barre de progression
                pbar.update(1)
                pbar.set_postfix({
                    'Best': f"{record['max']:.2f}",
                    'Avg': f"{record['avg']:.2f}",
                    'Std': f"{record['std']:.2f}"
                })
                
                # Suivi de l'amélioration
                best_fitness_history.append(record['max'])
                
                # Détection de stagnation
                if record['max'] > last_best_fitness:
                    improvement = record['max'] - last_best_fitness
                    no_improvement_count = 0
                    last_best_fitness = record['max']
                    
                    # Log des améliorations significatives
                    if improvement > 10:
                        tqdm.write(f"   🎯 Génération {gen} - Amélioration significative! "
                                  f"Nouveau best: {record['max']:.2f} (+{improvement:.2f})")
                else:
                    no_improvement_count += 1
                
                # Affichage périodique
                if gen % 10 == 0:
                    tqdm.write(f"   Génération {gen} - Best: {record['max']:.2f}, "
                              f"Avg: {record['avg']:.2f}, Std: {record['std']:.2f}")
                
                # Sauvegarde intermédiaire
                if gen % 20 == 0:
                    self._save_checkpoint(hof, gen)
                
                # Early stopping si pas d'amélioration
                if no_improvement_count > 15:
                    tqdm.write(f"\n⚠️ Arrêt anticipé: pas d'amélioration depuis {no_improvement_count} générations")
                    break
        
        print("\n✅ Optimisation terminée!")
        print(f"   • Générations complétées: {gen}/{generations}")
        print(f"   • Meilleure fitness finale: {last_best_fitness:.2f}")
        print(f"   • Amélioration totale: {last_best_fitness - best_fitness_history[0]:.2f}")
        
        # Analyser les résultats
        print("\n🔍 Analyse des résultats...")
        best_individuals = self._analyze_hall_of_fame(hof)
        
        return best_individuals, logbook
    
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
        
        with open(f'checkpoint_gen_{generation}.json', 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def _analyze_hall_of_fame(self, hof):
        """Analyse détaillée des meilleurs individus"""
        best_individuals = []
        
        print("\n📊 Évaluation détaillée des meilleures stratégies...")
        
        with tqdm(total=min(10, len(hof)), desc="Analyse", unit="stratégie") as pbar:
            for i, ind in enumerate(hof[:10]):
                params = {
                    'kelly_fraction': ind[0],
                    'min_confidence': ind[1],
                    'min_value': ind[2],
                    'max_bet_fraction': ind[3],
                    'min_edge': ind[4],
                    'volatility_penalty': ind[5],
                    'drawdown_penalty': ind[6],
                    'min_bets_penalty': ind[7]
                }
                
                # Simulation complète pour obtenir toutes les métriques
                metrics = self.simulate_betting_strategy(params)
                
                # Test de validation (derniers 20% des données)
                validation_metrics = self.simulate_betting_strategy(params, validation_split=0.2)
                
                best_individuals.append({
                    'params': params,
                    'metrics': metrics,
                    'validation_metrics': validation_metrics,
                    'fitness': ind.fitness.values[0],
                    'rank': i + 1
                })
                
                pbar.update(1)
        
        return best_individuals
    
    def plot_optimization_results(self, logbook):
        """
        Visualise les résultats de l'optimisation avec des graphiques améliorés
        """
        gen = logbook.select("gen")
        fit_maxs = logbook.select("max")
        fit_avgs = logbook.select("avg")
        fit_stds = logbook.select("std")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Graphique 1: Évolution de la fitness
        ax1.plot(gen, fit_maxs, 'b-', label='Maximum', linewidth=2)
        ax1.plot(gen, fit_avgs, 'g-', label='Moyenne', linewidth=2)
        ax1.fill_between(gen, 
                        np.array(fit_avgs) - np.array(fit_stds),
                        np.array(fit_avgs) + np.array(fit_stds),
                        alpha=0.3, color='green', label='±1 std')
        ax1.set_xlabel('Génération')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Évolution de la fitness au cours de l\'optimisation')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Graphique 2: Convergence
        improvements = np.diff(fit_maxs)
        ax2.bar(gen[1:], improvements, color=['green' if x > 0 else 'red' for x in improvements])
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_xlabel('Génération')
        ax2.set_ylabel('Amélioration')
        ax2.set_title('Amélioration de la meilleure fitness par génération')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('optimization_evolution.png', dpi=300)
        plt.show()
    
    def display_best_strategies(self, best_individuals):
        """
        Affiche les meilleures stratégies trouvées avec formatage amélioré
        """
        print("\n" + "="*70)
        print("🏆 MEILLEURES STRATÉGIES TROUVÉES")
        print("="*70 + "\n")
        
        for strategy in best_individuals[:5]:
            print(f"{'='*70}")
            print(f"🥇 STRATÉGIE #{strategy['rank']} - Fitness: {strategy['fitness']:.2f}")
            print(f"{'='*70}")
            
            params = strategy['params']
            metrics = strategy['metrics']
            val_metrics = strategy['validation_metrics']
            
            # Paramètres
            print("\n📐 PARAMÈTRES OPTIMAUX:")
            print(f"   • Kelly diviseur: {params['kelly_fraction']:.1f} (Kelly/{params['kelly_fraction']:.0f})")
            print(f"   • Confiance minimale: {params['min_confidence']:.1%}")
            print(f"   • Value minimale: {params['min_value']:.3f}")
            print(f"   • Mise maximale: {params['max_bet_fraction']:.1%} de la bankroll")
            print(f"   • Edge minimum: {params['min_edge']:.1%}")
            
            # Performance
            print("\n💰 PERFORMANCE:")
            print(f"   • ROI: {metrics['roi']:.1f}% {'✅' if metrics['roi'] > 0 else '❌'}")
            print(f"   • Nombre de paris: {metrics['total_bets']}")
            print(f"   • Taux de réussite: {metrics['win_rate']:.1%}")
            print(f"   • Profit total: {metrics['profit']:+.2f}€")
            print(f"   • Bankroll finale: {metrics['final_bankroll']:.2f}€")
            print(f"   • Expectancy: {metrics['expectancy']:+.2f}€/pari")
            
            # Risque
            print("\n⚠️ MÉTRIQUES DE RISQUE:")
            print(f"   • Drawdown maximum: {metrics['max_drawdown']:.1f}% {'✅' if metrics['max_drawdown'] > -20 else '⚠️' if metrics['max_drawdown'] > -30 else '❌'}")
            print(f"   • Durée max drawdown: {metrics['max_drawdown_duration']} paris")
            print(f"   • Pertes consécutives max: {metrics['max_consecutive_losses']}")
            print(f"   • Volatilité: {metrics['volatility']:.2f}")
            print(f"   • VaR 95%: {metrics['var_95']:.2f}€")
            
            # Ratios
            print("\n📊 RATIOS DE PERFORMANCE:")
            print(f"   • Sharpe Ratio: {metrics['sharpe_ratio']:.2f} {'✅' if metrics['sharpe_ratio'] > 1 else '⚠️' if metrics['sharpe_ratio'] > 0.5 else '❌'}")
            print(f"   • Calmar Ratio: {metrics['calmar_ratio']:.2f} {'✅' if metrics['calmar_ratio'] > 1 else '⚠️' if metrics['calmar_ratio'] > 0.5 else '❌'}")
            print(f"   • Sortino Ratio: {metrics['sortino_ratio']:.2f}")
            print(f"   • Profit Factor: {metrics['profit_factor']:.2f} {'✅' if metrics['profit_factor'] > 1.5 else '⚠️' if metrics['profit_factor'] > 1.2 else '❌'}")
            print(f"   • Recovery Factor: {metrics['recovery_factor']:.2f}")
            
            # Validation
            if val_metrics['total_bets'] > 0:
                print("\n🔍 VALIDATION (20% dernières données):")
                print(f"   • ROI validation: {val_metrics['roi']:.1f}%")
                print(f"   • Drawdown validation: {val_metrics['max_drawdown']:.1f}%")
                consistency = abs(metrics['roi'] - val_metrics['roi']) < 20
                print(f"   • Consistance: {'✅ Bonne' if consistency else '⚠️ Écart important'}")
            
            print("\n")
    
    def backtest_strategy(self, params: Dict, plot_title: str = "Backtest Results") -> pd.DataFrame:
        """
        Effectue un backtest détaillé d'une stratégie avec visualisation améliorée
        """
        initial_bankroll = 1000
        bankroll = initial_bankroll
        bets_history = []
        
        kelly_fraction = params['kelly_fraction']
        min_confidence = params['min_confidence']
        min_value = params['min_value']
        max_bet_fraction = params['max_bet_fraction']
        min_edge = params['min_edge']
        
        print(f"\n📊 Backtest en cours pour: {plot_title}")
        
        with tqdm(total=len(self.odds_data), desc="Backtest", unit="combat") as pbar:
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
                    pbar.set_postfix({'Status': 'Faillite!'})
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
                
                # Mise à jour périodique
                if len(bets_history) % 10 == 0:
                    current_roi = ((bankroll - initial_bankroll) / initial_bankroll) * 100
                    pbar.set_postfix({'ROI': f'{current_roi:.1f}%', 'Bankroll': f'{bankroll:.0f}€'})
        
        return pd.DataFrame(bets_history)
    
    def plot_backtest_results(self, backtest_df: pd.DataFrame, title: str = "Backtest Results"):
        """
        Visualise les résultats du backtest avec des graphiques améliorés
        """
        if backtest_df.empty:
            print("❌ Aucun pari effectué avec cette stratégie.")
            return
        
        # Configuration du style
        plt.style.use('seaborn-v0_8-darkgrid')
        fig = plt.figure(figsize=(16, 12))
        
        # Créer une grille de subplots
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
        
        # 1. Évolution de la bankroll
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(backtest_df.index, backtest_df['bankroll'], 'b-', linewidth=2, label='Bankroll')
        ax1.axhline(y=1000, color='r', linestyle='--', alpha=0.5, label='Bankroll initiale')
        
        # Ajouter des zones colorées pour les gains/pertes
        ax1.fill_between(backtest_df.index, 1000, backtest_df['bankroll'],
                        where=backtest_df['bankroll'] >= 1000,
                        color='green', alpha=0.2, label='Profit')
        ax1.fill_between(backtest_df.index, 1000, backtest_df['bankroll'],
                        where=backtest_df['bankroll'] < 1000,
                        color='red', alpha=0.2, label='Perte')
        
        ax1.set_title('Évolution de la Bankroll', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Nombre de paris')
        ax1.set_ylabel('Bankroll (€)')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 2. Drawdown
        ax2 = fig.add_subplot(gs[1, 0])
        rolling_max = backtest_df['bankroll'].expanding().max()
        drawdown = (backtest_df['bankroll'] - rolling_max) / rolling_max * 100
        ax2.fill_between(backtest_df.index, drawdown, 0, 
                        where=drawdown<0, interpolate=True, 
                        color='red', alpha=0.3)
        ax2.plot(backtest_df.index, drawdown, 'r-', linewidth=1)
        ax2.set_title('Drawdown', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Nombre de paris')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        # Ajouter le drawdown maximum
        max_dd_idx = drawdown.idxmin()
        ax2.annotate(f'Max: {drawdown.min():.1f}%',
                    xy=(max_dd_idx, drawdown.min()),
                    xytext=(max_dd_idx, drawdown.min() - 5),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')
        
        # 3. Distribution des profits
        ax3 = fig.add_subplot(gs[1, 1])
        wins = backtest_df[backtest_df['profit'] > 0]['profit']
        losses = backtest_df[backtest_df['profit'] < 0]['profit']
        
        ax3.hist(wins, bins=30, alpha=0.7, color='green', label=f'Gains (n={len(wins)})')
        ax3.hist(losses, bins=30, alpha=0.7, color='red', label=f'Pertes (n={len(losses)})')
        ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax3.set_title('Distribution des Profits/Pertes', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Profit (€)')
        ax3.set_ylabel('Fréquence')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. ROI cumulé
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.plot(backtest_df.index, backtest_df['roi'], 'g-', linewidth=2)
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.set_title('ROI Cumulé', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Nombre de paris')
        ax4.set_ylabel('ROI (%)')
        ax4.grid(True, alpha=0.3)
        
        # Ajouter des annotations pour les points clés
        final_roi = backtest_df['roi'].iloc[-1]
        ax4.text(0.98, 0.02, f'ROI Final: {final_roi:.1f}%',
                transform=ax4.transAxes,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=12, fontweight='bold')
        
        # 5. Taille des mises
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.scatter(backtest_df.index, backtest_df['stake_pct'], 
                   c=['green' if r == 'win' else 'red' for r in backtest_df['result']], 
                   alpha=0.6, s=30)
        ax5.set_title('Taille des mises (% de bankroll)', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Nombre de paris')
        ax5.set_ylabel('Mise (%)')
        ax5.grid(True, alpha=0.3)
        
        # Titre principal
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Ajuster la mise en page et sauvegarder
        plt.tight_layout()
        filename = f'backtest_{title.replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Afficher un résumé
        print(f"\n📊 RÉSUMÉ DU BACKTEST:")
        print(f"   • Nombre de paris: {len(backtest_df)}")
        print(f"   • ROI final: {final_roi:.1f}%")
        print(f"   • Bankroll finale: {backtest_df['bankroll'].iloc[-1]:.2f}€")
        print(f"   • Drawdown maximum: {drawdown.min():.1f}%")
        print(f"   • Taux de réussite: {len(wins)/len(backtest_df)*100:.1f}%")
        print(f"   • Graphiques sauvegardés dans: {filename}")
    
    def export_results(self, best_strategies, logbook):
        """
        Exporte tous les résultats dans différents formats
        """
        print("\n💾 Exportation des résultats...")
        
        # 1. Export des stratégies en CSV
        strategies_data = []
        for s in best_strategies:
            row = {
                'rank': s['rank'],
                'fitness': s['fitness'],
                **{f'param_{k}': v for k, v in s['params'].items()},
                **{f'metric_{k}': v for k, v in s['metrics'].items()},
                **{f'validation_{k}': v for k, v in s['validation_metrics'].items()}
            }
            strategies_data.append(row)
        
        strategies_df = pd.DataFrame(strategies_data)
        strategies_df.to_csv('best_strategies_optimized.csv', index=False)
        
        # 2. Export du log d'optimisation
        log_df = pd.DataFrame(logbook)
        log_df.to_csv('optimization_log.csv', index=False)
        
        # 3. Export JSON complet
        export_data = {
            'optimization_date': datetime.now().isoformat(),
            'parameters': {
                'population_size': 100,
                'generations': len(logbook),
                'parameter_bounds': self.param_bounds
            },
            'best_strategies': [
                {
                    'rank': s['rank'],
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
        
        with open('optimization_results.json', 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print("\n✅ Résultats exportés:")
        print("   • best_strategies_optimized.csv - Tableau des meilleures stratégies")
        print("   • optimization_log.csv - Historique de l'optimisation")
        print("   • optimization_results.json - Résultats complets en JSON")
        
        # 4. Rapport Markdown
        self._generate_markdown_report(best_strategies)

    def _generate_markdown_report(self, best_strategies):
        """Génère un rapport Markdown des résultats"""
        with open('optimization_report.md', 'w') as f:
            f.write("# UFC Betting Strategy Optimization Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            best = best_strategies[0]
            f.write(f"- **Best ROI**: {best['metrics']['roi']:.1f}%\n")
            f.write(f"- **Max Drawdown**: {best['metrics']['max_drawdown']:.1f}%\n")
            f.write(f"- **Sharpe Ratio**: {best['metrics']['sharpe_ratio']:.2f}\n")
            f.write(f"- **Total Bets**: {best['metrics']['total_bets']}\n\n")
            
            f.write("## Top 5 Strategies\n\n")
            for i, strategy in enumerate(best_strategies[:5]):
                f.write(f"### Strategy #{i+1}\n\n")
                f.write("**Parameters:**\n")
                for k, v in strategy['params'].items():
                    if k.endswith('_penalty'):
                        continue
                    f.write(f"- {k}: {v:.3f}\n")
                f.write("\n**Performance:**\n")
                f.write(f"- ROI: {strategy['metrics']['roi']:.1f}%\n")
                f.write(f"- Drawdown: {strategy['metrics']['max_drawdown']:.1f}%\n")
                f.write(f"- Win Rate: {strategy['metrics']['win_rate']:.1%}\n")
                f.write("\n---\n\n")


def main():
    """
    Fonction principale pour lancer l'optimisation
    """
    # Configuration
    print("\n" + "="*70)
    print("🥊 UFC BETTING STRATEGY OPTIMIZER")
    print("="*70)
    print("\nVersion 2.0 - Optimisation avancée avec focus sur le drawdown")
    print("Développé pour maximiser les profits tout en minimisant les risques\n")
    
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
    
    print("🔍 Vérification des fichiers nécessaires:")
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
    
    # Créer l'optimiseur
    optimizer = UFCBettingOptimizer(model_path, fighters_stats_path, odds_data_path)
    
    # Configuration de l'optimisation
    print("\n⚙️ CONFIGURATION DE L'OPTIMISATION:")
    print("\nParamètres par défaut recommandés:")
    print("   • Population: 100 individus")
    print("   • Générations: 50")
    print("   • Stratégie: Focus sur minimisation du drawdown")
    
    use_custom = input("\nVoulez-vous personnaliser les paramètres? (y/N): ").lower() == 'y'
    
    if use_custom:
        try:
            pop_size = int(input("Taille de la population (défaut: 100): ") or "100")
            n_gen = int(input("Nombre de générations (défaut: 50): ") or "50")
        except ValueError:
            print("Valeurs invalides. Utilisation des paramètres par défaut.")
            pop_size, n_gen = 100, 50
    else:
        pop_size, n_gen = 100, 50
    
    # Lancer l'optimisation
    print("\n🚀 Lancement de l'optimisation...")
    time.sleep(1)
    
    start_time = time.time()
    best_strategies, logbook = optimizer.optimize(population_size=pop_size, generations=n_gen)
    end_time = time.time()
    
    print(f"\n⏱️ Temps d'exécution: {(end_time - start_time)/60:.1f} minutes")
    
    # Afficher et analyser les résultats
    optimizer.plot_optimization_results(logbook)
    optimizer.display_best_strategies(best_strategies)
    
    # Backtest des meilleures stratégies
    print("\n📊 BACKTESTS DÉTAILLÉS")
    print("="*70)
    
    for i in range(min(3, len(best_strategies))):
        print(f"\n🔍 Backtest de la stratégie #{i+1}...")
        strategy = best_strategies[i]
        backtest_df = optimizer.backtest_strategy(
            strategy['params'], 
            f"Stratégie #{i+1} - ROI: {strategy['metrics']['roi']:.1f}%"
        )
        
        if not backtest_df.empty:
            optimizer.plot_backtest_results(
                backtest_df, 
                f"Stratégie #{i+1} - Backtest complet"
            )
    
    # Exporter tous les résultats
    optimizer.export_results(best_strategies, logbook)
    
    # Message final
    print("\n" + "="*70)
    print("✅ OPTIMISATION TERMINÉE AVEC SUCCÈS!")
    print("="*70)
    print("\n📁 Fichiers générés:")
    print("   • best_strategies_optimized.csv - Stratégies optimales")
    print("   • optimization_log.csv - Journal d'optimisation")
    print("   • optimization_results.json - Résultats complets")
    print("   • optimization_report.md - Rapport détaillé")
    print("   • optimization_evolution.png - Graphique d'évolution")
    print("   • backtest_*.png - Graphiques de backtest")
    print("\n💡 Recommandation: Utilisez la stratégie #1 pour commencer,")
    print("   mais surveillez régulièrement les performances et ajustez si nécessaire.")
    print("\n🎯 Bonne chance avec vos paris UFC! Pariez toujours de manière responsable.")


if __name__ == "__main__":
    main()