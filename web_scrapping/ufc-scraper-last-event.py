#!/usr/bin/env python
# coding: utf-8

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
import time
import random
import sys

# Configuration des headers pour éviter d'être bloqué
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9,fr;q=0.8'
}

# Paramètres de configuration
MAX_EVENTS_TO_CHECK = 2
MAX_RETRIES = 3
DELAY_RANGE = (0.5, 1.5)
VERBOSE = True

def log(message, is_debug=False):
    """Affiche un message de log, selon le niveau de verbosité"""
    if not is_debug or (is_debug and VERBOSE):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")

def clean_numeric_string(text):
    """Nettoie une chaîne de caractères pour extraire uniquement les chiffres"""
    if not text:
        return "0"
    
    # Pour le cas spécifique des pourcentages
    if isinstance(text, str) and '%' in text:
        try:
            # Extraire juste le nombre avant le %
            return text.replace('%', '').strip()
        except:
            return "0"
    
    # Pour les strings avec des unités comme "of X"
    if isinstance(text, str) and 'of' in text:
        try:
            # Extraire juste le premier nombre
            return text.split('of')[0].strip()
        except:
            return "0"
            
    # Supprimer tous les caractères non numériques sauf le point décimal
    if isinstance(text, str):
        cleaned = re.sub(r'[^\d.]', '', text.split()[0] if ' ' in text else text)
        return cleaned if cleaned else "0"
    
    return str(text) if text else "0"

def make_request(url, max_retries=MAX_RETRIES, delay_range=DELAY_RANGE):
    """Effectue une requête HTTP avec gestion des erreurs et des délais"""
    for attempt in range(max_retries):
        try:
            # Pause aléatoire pour éviter d'être bloqué
            time.sleep(random.uniform(delay_range[0], delay_range[1]))
            
            log(f"Requête: {url}", is_debug=True)
            response = requests.get(url, headers=HEADERS, timeout=20)
            
            status = response.status_code
            if status == 200:
                return response
            
            log(f"Tentative {attempt+1}/{max_retries}: code {status}")
            
            # Si le site renvoie 403 (Forbidden) ou 429 (Too Many Requests), augmenter le délai
            if status in [403, 429]:
                longer_delay = random.uniform(5, 15)
                log(f"Délai supplémentaire de {longer_delay:.2f}s suite au code {status}")
                time.sleep(longer_delay)
        except Exception as e:
            log(f"Erreur lors de la requête (tentative {attempt+1}/{max_retries}): {e}")
            time.sleep(random.uniform(2, 5))  # Délai plus long en cas d'erreur
    
    return None

def get_recent_events(max_events=MAX_EVENTS_TO_CHECK):
    """Récupère les événements UFC récents"""
    log(f"Récupération des {max_events} événements récents...")
    
    recent_events = []
    
    # Essayer d'abord la page "tous les événements"
    url = "http://ufcstats.com/statistics/events/completed?page=all"
    response = make_request(url)
    
    if not response:
        # Essayer la page principale
        url = "http://ufcstats.com/statistics/events/completed"
        response = make_request(url)
        
        if not response:
            log("Échec de la requête pour récupérer les événements récents.")
            return recent_events
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Rechercher les liens d'événements
    event_links = soup.find_all('a', href=lambda href: href and 'event-details' in href)
    
    # Afficher le HTML pour déboguer si aucun lien n'est trouvé
    if not event_links and VERBOSE:
        log("Aucun lien d'événement trouvé. Voici les 500 premiers caractères de la page HTML:")
        log(soup.prettify()[:500], is_debug=True)
        
        # Tentative alternative en cherchant dans toutes les balises <a>
        all_links = soup.find_all('a')
        log(f"Nombre total de liens sur la page: {len(all_links)}")
        event_links = [link for link in all_links if link.get('href') and 'event-details' in link.get('href')]
    
    for i, link in enumerate(event_links):
        if i >= max_events:
            break
            
        event_url = link.get('href')
        event_name = link.text.strip()
        
        # Récupérer également l'ID de l'événement à partir de l'URL
        event_id = event_url.split("/")[-1] if event_url else ""
        
        recent_events.append({
            'name': event_name,
            'url': event_url,
            'id': event_id
        })
        
        log(f"Événement trouvé: {event_name}")
    
    log(f"Total: {len(recent_events)} événements récents récupérés")
    return recent_events

def parse_fight_stats(stats_text, default_value=0):
    """
    Parse les statistiques de combat depuis un texte, en gérant les cas mal formatés
    
    Args:
        stats_text: Texte contenant les statistiques (ex: "48 of 110")
        default_value: Valeur par défaut si l'extraction échoue
        
    Returns:
        Tuple (landed, attempted, accuracy)
    """
    try:
        # Nettoyer le texte pour extraire seulement les chiffres
        if ' of ' in stats_text:
            parts = stats_text.split(' of ')
            if len(parts) >= 2:
                # Nettoyer chaque partie pour extraire uniquement les chiffres
                landed_str = clean_numeric_string(parts[0])
                attempted_str = clean_numeric_string(parts[1])
                
                landed = int(landed_str) if landed_str.isdigit() else default_value
                attempted = int(attempted_str) if attempted_str.isdigit() else default_value
                accuracy = round(landed / attempted, 2) if attempted > 0 else 0.0
                
                return landed, attempted, accuracy
    except Exception as e:
        log(f"Erreur lors du parsing des stats: {e} - Texte: '{stats_text}'", is_debug=True)
    
    # En cas d'échec, retourner les valeurs par défaut
    return default_value, default_value, 0.0

def clean_fighter_name(name):
    """Nettoie le nom d'un combattant en enlevant les préfixes inutiles"""
    # Enlever "win vs" et autres préfixes
    name = re.sub(r'^win\s+vs\s+', '', name)
    name = re.sub(r'^vs\s+', '', name)
    
    # Enlever les espaces supplémentaires
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name

def clean_weight_class(weight_class_text):
    """Nettoie et normalise le texte de la catégorie de poids"""
    if not weight_class_text:
        return "", False, "Men"
    
    weight_class_text = weight_class_text.strip()
    
    # Déterminer s'il s'agit d'un combat pour le titre
    is_title_bout = "Title" in weight_class_text or "Championship" in weight_class_text
    
    # Déterminer le genre
    gender = "Women" if "Women" in weight_class_text else "Men"
    
    # Nettoyer la catégorie de poids
    weight_class = weight_class_text
    weight_class = weight_class.replace("Women's", "")
    weight_class = weight_class.replace("Title", "")
    weight_class = weight_class.replace("Championship", "")
    weight_class = weight_class.replace("Bout", "")
    weight_class = re.sub(r'\s+', ' ', weight_class).strip()
    
    return weight_class, is_title_bout, gender

def extract_referee_from_fight(fight_url):
    """Extrait le nom de l'arbitre à partir d'une page de combat détaillée"""
    log(f"Extraction de l'arbitre depuis: {fight_url}", is_debug=True)
    
    response = make_request(fight_url)
    if not response:
        return ""
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Rechercher les détails du combat
    details = soup.find_all('i', class_='b-fight-details__text-item')
    
    for detail in details:
        text = detail.get_text(strip=True)
        if "Referee:" in text:
            referee = text.replace("Referee:", "").strip()
            log(f"Arbitre trouvé: {referee}", is_debug=True)
            return referee
    
    return ""

def determine_winner_from_method(method_text, red_fighter, blue_fighter):
    """Détermine le vainqueur en analysant le texte de la méthode"""
    method_text = method_text.lower()
    
    # Vérifier si c'est un match nul
    if "draw" in method_text:
        return "Draw"
    
    # Normaliser les noms pour la comparaison
    red_name_lower = red_fighter.lower()
    blue_name_lower = blue_fighter.lower()
    
    # Extraire le premier et le dernier nom de chaque combattant
    red_parts = red_name_lower.split() if " " in red_name_lower else [red_name_lower]
    blue_parts = blue_name_lower.split() if " " in blue_name_lower else [blue_name_lower]
    
    # Rechercher les noms dans le texte de la méthode
    for part in red_parts:
        if len(part) > 3 and part in method_text:  # Éviter les prépositions/articles courts
            return "Red"
    
    for part in blue_parts:
        if len(part) > 3 and part in method_text:  # Éviter les prépositions/articles courts
            return "Blue"
    
    return "No Decision"

def extract_fight_data_from_individual_pages(fight_urls, event_name):
    """Extrait les données de combat à partir des pages individuelles de combat"""
    log("Extraction des données à partir des pages individuelles de combat...")
    
    fights_data = []
    
    for i, fight_url in enumerate(fight_urls):
        try:
            log(f"Analyse du combat {i+1}: {fight_url}")
            fight_response = make_request(fight_url)
            if not fight_response:
                continue
            
            fight_soup = BeautifulSoup(fight_response.text, 'html.parser')
            
            # Extraire les noms des combattants
            fighter_elems = fight_soup.select('div.b-fight-details__person')
            if len(fighter_elems) < 2:
                log(f"Combat {i+1}: Impossible de trouver les noms des combattants")
                continue
            
            red_fighter_elem = fighter_elems[0]
            blue_fighter_elem = fighter_elems[1]
            
            red_fighter_name_elem = red_fighter_elem.select_one('h3.b-fight-details__person-name a')
            blue_fighter_name_elem = blue_fighter_elem.select_one('h3.b-fight-details__person-name a')
            
            if not red_fighter_name_elem or not blue_fighter_name_elem:
                log(f"Combat {i+1}: Noms des combattants incomplets")
                continue
            
            red_fighter = red_fighter_name_elem.text.strip()
            blue_fighter = blue_fighter_name_elem.text.strip()
            
            # Nettoyer les noms des combattants
            red_fighter = clean_fighter_name(red_fighter)
            blue_fighter = clean_fighter_name(blue_fighter)
            
            # Déterminer le vainqueur en regardant la classe CSS ou le texte
            winner = "No Decision"
            red_status = red_fighter_elem.select_one('i.b-fight-details__person-status')
            blue_status = blue_fighter_elem.select_one('i.b-fight-details__person-status')
            
            if red_status and 'W' in red_status.text:
                winner = "Red"
            elif blue_status and 'W' in blue_status.text:
                winner = "Blue"
            else:
                # Vérifier s'il s'agit d'un match nul
                draw_indicators = ['draw', 'no contest']
                result_elem = fight_soup.select_one('div.b-fight-details__content')
                if result_elem:
                    result_text = result_elem.text.strip().lower()
                    if any(indicator in result_text for indicator in draw_indicators):
                        winner = "Draw"
            
            # Sauter les combats "No Decision"
            if winner == "No Decision":
                log(f"Combat {i+1}: Pas de décision, ignoré")
                continue
            
            # Extraire la méthode et la classe de poids
            method = ""
            weight_class = ""
            is_title_bout = 0.0
            gender = "Men"
            finish_round = 0.0
            time_str = ""
            
            # Extraire tous les éléments de détail
            detail_items = fight_soup.select('i.b-fight-details__text-item')
            
            for item in detail_items:
                text = item.text.strip()
                if "Method:" in text:
                    method = text.replace("Method:", "").strip()
                elif "Weight class:" in text:
                    weight_class_text = text.replace("Weight class:", "").strip()
                    weight_class, is_title_flag, gender_value = clean_weight_class(weight_class_text)
                    is_title_bout = 1.0 if is_title_flag else 0.0
                    gender = gender_value
                elif "Round:" in text:
                    round_text = text.replace("Round:", "").strip()
                    finish_round = float(round_text) if round_text.isdigit() else 0.0
                elif "Time:" in text:
                    time_str = text.replace("Time:", "").strip()
            
            # Normaliser la méthode
            if "KO/TKO" in method:
                method = "KO/TKO"
            elif "SUB" in method:
                method = "Submission"
            elif "U-DEC" in method or "Decision - Unanimous" in method:
                method = "Decision - Unanimous"
            elif "S-DEC" in method or "Decision - Split" in method:
                method = "Decision - Split"
            elif "M-DEC" in method or "Decision - Majority" in method:
                method = "Decision - Majority"
            
            # Calculer le temps en secondes
            time_sec = 0.0
            if ":" in time_str:
                try:
                    minutes, seconds = time_str.split(':')
                    round_time_sec = int(minutes) * 60 + int(seconds)
                    prev_rounds_sec = (int(finish_round) - 1) * 300 if finish_round > 0 else 0
                    time_sec = prev_rounds_sec + round_time_sec
                except Exception as e:
                    log(f"Erreur lors du parsing du temps: {e}", is_debug=True)
            
            # Nombre total de rounds (5 pour les combats de titre, 3 sinon)
            total_rounds = 5.0 if is_title_bout == 1.0 else 3.0
            
            # Extraire l'arbitre
            referee = ""
            for item in detail_items:
                text = item.text.strip()
                if "Referee:" in text:
                    referee = text.replace("Referee:", "").strip()
                    break
            
            # Initialiser les statistiques de combat
            stats = {
                'r_kd': 0, 'b_kd': 0,
                'r_sig_str': 0, 'r_sig_str_att': 0, 'r_sig_str_acc': 0.0,
                'b_sig_str': 0, 'b_sig_str_att': 0, 'b_sig_str_acc': 0.0,
                'r_str': 0, 'r_str_att': 0, 'r_str_acc': 0.0,
                'b_str': 0, 'b_str_att': 0, 'b_str_acc': 0.0,
                'r_td': 0, 'r_td_att': 0, 'r_td_acc': 0.0,
                'b_td': 0, 'b_td_att': 0, 'b_td_acc': 0.0,
                'r_sub_att': 0, 'b_sub_att': 0,
                'r_rev': 0, 'b_rev': 0,
                'r_ctrl_sec': 0, 'b_ctrl_sec': 0
            }
            
            # Extraire les statistiques depuis la table "Totals"
            totals_table = fight_soup.select_one('table.b-fight-details__table.js-fight-table')
            if totals_table:
                rows = totals_table.select('tr.b-fight-details__table-row')
                for row in rows[1:]:  # Ignorer l'en-tête
                    header_cell = row.select_one('th')
                    cols = row.select('td.b-fight-details__table-col')
                    
                    if not header_cell or len(cols) < 8:
                        continue
                        
                    header_text = header_cell.text.strip().lower()
                    
                    if "totals" in header_text:
                        # KD
                        stats['r_kd'] = int(clean_numeric_string(cols[0].text.strip()))
                        stats['b_kd'] = int(clean_numeric_string(cols[1].text.strip()))
                        
                        # Sig Strikes
                        r_str_text = cols[2].text.strip()
                        b_str_text = cols[3].text.strip()
                        
                        r_sig_str, r_sig_str_att, r_sig_str_acc = parse_fight_stats(r_str_text)
                        stats['r_sig_str'] = r_sig_str
                        stats['r_sig_str_att'] = r_sig_str_att
                        stats['r_sig_str_acc'] = r_sig_str_acc
                        
                        b_sig_str, b_sig_str_att, b_sig_str_acc = parse_fight_stats(b_str_text)
                        stats['b_sig_str'] = b_sig_str
                        stats['b_sig_str_att'] = b_sig_str_att
                        stats['b_sig_str_acc'] = b_sig_str_acc
                        
                        # Takedowns
                        r_td_text = cols[4].text.strip()
                        b_td_text = cols[5].text.strip()
                        
                        r_td, r_td_att, r_td_acc = parse_fight_stats(r_td_text)
                        stats['r_td'] = r_td
                        stats['r_td_att'] = r_td_att
                        stats['r_td_acc'] = r_td_acc
                        
                        b_td, b_td_att, b_td_acc = parse_fight_stats(b_td_text)
                        stats['b_td'] = b_td
                        stats['b_td_att'] = b_td_att
                        stats['b_td_acc'] = b_td_acc
                    
                    # Rechercher les soumissions et le temps de contrôle dans d'autres lignes
                    elif "submission" in header_text:
                        stats['r_sub_att'] = int(clean_numeric_string(cols[0].text.strip()))
                        stats['b_sub_att'] = int(clean_numeric_string(cols[1].text.strip()))
                    
                    elif "reversal" in header_text:
                        stats['r_rev'] = int(clean_numeric_string(cols[0].text.strip()))
                        stats['b_rev'] = int(clean_numeric_string(cols[1].text.strip()))
                    
                    elif "control" in header_text:
                        # Convertir le temps de contrôle (format MM:SS) en secondes
                        r_ctrl_text = cols[0].text.strip()
                        b_ctrl_text = cols[1].text.strip()
                        
                        if ":" in r_ctrl_text:
                            mins, secs = r_ctrl_text.split(":")
                            stats['r_ctrl_sec'] = int(clean_numeric_string(mins)) * 60 + int(clean_numeric_string(secs))
                        
                        if ":" in b_ctrl_text:
                            mins, secs = b_ctrl_text.split(":")
                            stats['b_ctrl_sec'] = int(clean_numeric_string(mins)) * 60 + int(clean_numeric_string(secs))
            
            # Ensure we have values for all stats
            for key in stats:
                if stats[key] is None:
                    stats[key] = 0 if key.endswith('acc') else 0.0
            
            # Copying significant strikes to total strikes if not available
            stats['r_str'] = stats['r_sig_str'] 
            stats['r_str_att'] = stats['r_sig_str_att']
            stats['r_str_acc'] = stats['r_sig_str_acc']
            stats['b_str'] = stats['b_sig_str']
            stats['b_str_att'] = stats['b_sig_str_att']
            stats['b_str_acc'] = stats['b_sig_str_acc']
            
            # Calculate accuracy percentages
            if stats['r_sig_str_att'] > 0:
                stats['r_sig_str_acc'] = round(stats['r_sig_str'] / stats['r_sig_str_att'], 2)
            if stats['b_sig_str_att'] > 0:
                stats['b_sig_str_acc'] = round(stats['b_sig_str'] / stats['b_sig_str_att'], 2)
            if stats['r_td_att'] > 0:
                stats['r_td_acc'] = round(stats['r_td'] / stats['r_td_att'], 2)
            if stats['b_td_att'] > 0:
                stats['b_td_acc'] = round(stats['b_td'] / stats['b_td_att'], 2)
            
            # Create the fight dictionary with all values
            fight_dict = {
                'event_name': event_name,
                'r_fighter': red_fighter,
                'b_fighter': blue_fighter,
                'winner': winner,
                'weight_class': weight_class,
                'is_title_bout': is_title_bout,
                'gender': gender,
                'method': method,
                'finish_round': finish_round,
                'total_rounds': total_rounds,
                'time_sec': time_sec,
                'referee': referee,
                'r_kd': stats['r_kd'],
                'r_sig_str': stats['r_sig_str'],
                'r_sig_str_att': stats['r_sig_str_att'],
                'r_sig_str_acc': stats['r_sig_str_acc'],
                'r_str': stats['r_str'],
                'r_str_att': stats['r_str_att'],
                'r_str_acc': stats['r_str_acc'],
                'r_td': stats['r_td'],
                'r_td_att': stats['r_td_att'],
                'r_td_acc': stats['r_td_acc'],
                'r_sub_att': stats['r_sub_att'],
                'r_rev': stats['r_rev'],
                'r_ctrl_sec': stats['r_ctrl_sec'],
                'b_kd': stats['b_kd'],
                'b_sig_str': stats['b_sig_str'],
                'b_sig_str_att': stats['b_sig_str_att'],
                'b_sig_str_acc': stats['b_sig_str_acc'],
                'b_str': stats['b_str'],
                'b_str_att': stats['b_str_att'],
                'b_str_acc': stats['b_str_acc'],
                'b_td': stats['b_td'],
                'b_td_att': stats['b_td_att'],
                'b_td_acc': stats['b_td_acc'],
                'b_sub_att': stats['b_sub_att'],
                'b_rev': stats['b_rev'],
                'b_ctrl_sec': stats['b_ctrl_sec']
            }
            
            fights_data.append(fight_dict)
            log(f"Combat {i+1}: {red_fighter} vs {blue_fighter} - Vainqueur: {winner}")
        
        except Exception as e:
            log(f"Erreur lors du traitement du combat {i+1} ({fight_url}): {e}")
            # Continuer avec le combat suivant
            continue
    
    return fights_data

def extract_fight_data_from_table(soup, event_name):
    """Extrait les données de combat à partir du tableau de l'événement"""
    log("Extraction des données à partir du tableau de l'événement...")
    
    # Tableau des combats
    table = soup.find('table', class_='b-fight-details__table')
    if not table:
        log("Aucun tableau de combats trouvé.")
        return []
    
    # Récupérer les liens vers les détails des combats pour les arbitres
    fight_links = soup.find_all('a', href=lambda href: href and 'fight-details' in href)
    fight_urls = [link.get('href') for link in fight_links]
    
    # Récupérer les combats
    rows = table.find_all('tr', class_='b-fight-details__table-row')
    
    log(f"Nombre de lignes dans le tableau: {len(rows)}")
    
    if len(rows) <= 1:
        log("Pas assez de lignes dans le tableau de combats.")
        return []
    
    fights_data = []
    fight_index = 0
    
    # Ignorer la première ligne (en-tête)
    for row in rows[1:]:
        if 'b-fight-details__table-header' in row.get('class', []):
            continue  # Ignorer les en-têtes supplémentaires
            
        try:
            cells = row.find_all('td')
            if len(cells) < 8:
                log(f"Ligne ignorée: seulement {len(cells)} cellules trouvées", is_debug=True)
                continue
            
            # Analyser correctement le HTML pour extraire les noms des combattants
            fighter_names = []
            
            # Méthode 1: Analyser les liens des combattants
            fighter_links = row.select('a.b-link.b-fight-details__person-link')
            if fighter_links and len(fighter_links) >= 2:
                red_fighter = fighter_links[0].text.strip()
                blue_fighter = fighter_links[1].text.strip()
                fighter_names = [red_fighter, blue_fighter]
                log(f"Méthode 1: Combattants trouvés: {red_fighter} vs {blue_fighter}", is_debug=True)
            else:
                # Méthode 2: Analyser les cellules directement
                if len(cells) >= 2:
                    # Extraire le texte des deux premières cellules qui devraient contenir les noms
                    red_cell = cells[0].get_text(strip=True)
                    blue_cell = cells[1].get_text(strip=True)
                    
                    # Nettoyer tout texte supplémentaire comme "win vs"
                    red_fighter = clean_fighter_name(red_cell)
                    blue_fighter = clean_fighter_name(blue_cell)
                    
                    fighter_names = [red_fighter, blue_fighter]
                    log(f"Méthode 2: Combattants trouvés: {red_fighter} vs {blue_fighter}", is_debug=True)
                else:
                    # Méthode 3: Chercher les textes dans les paragraphes
                    p_texts = row.select('p.b-fight-details__table-text')
                    if len(p_texts) >= 2:
                        red_fighter = clean_fighter_name(p_texts[0].text.strip())
                        blue_fighter = clean_fighter_name(p_texts[1].text.strip())
                        fighter_names = [red_fighter, blue_fighter]
                        log(f"Méthode 3: Combattants trouvés: {red_fighter} vs {blue_fighter}", is_debug=True)
            
            if len(fighter_names) < 2:
                log("Combat ignoré: noms des combattants non trouvés", is_debug=True)
                continue
            
            red_fighter = fighter_names[0]
            blue_fighter = fighter_names[1]
            
            # Déterminer le vainqueur correctement
            winner = "No Decision"
            
            # Méthode 1: Regarder les icônes de statut
            status_icons = row.select('i.b-fight-details__person-status')
            if status_icons and len(status_icons) >= 2:
                red_status = status_icons[0].text.strip()
                blue_status = status_icons[1].text.strip()
                
                log(f"Statuts: Red={red_status}, Blue={blue_status}", is_debug=True)
                
                if red_status == 'W':
                    winner = "Red"
                elif blue_status == 'W':
                    winner = "Blue"
                elif red_status == 'D' or blue_status == 'D':
                    winner = "Draw"
            
            # Méthode 2: Si la méthode 1 échoue, chercher des indications dans le texte de la méthode
            if winner == "No Decision":
                method_cells = row.select('p.b-fight-details__table-text')
                if len(method_cells) >= 13:
                    method_text = method_cells[12].text.strip()
                    winner = determine_winner_from_method(method_text, red_fighter, blue_fighter)
            
            # Sauter les combats "No Decision"
            if winner == "No Decision":
                log(f"Combat {fight_index+1}: Pas de décision, ignoré")
                fight_index += 1
                continue
            
            # Récupérer toutes les statistiques via les paragraphes de la ligne
            all_texts = row.select('p.b-fight-details__table-text')
            
            # Initialiser les données du combat
            fight_data = {
                'event_name': event_name,
                'r_fighter': red_fighter,
                'b_fighter': blue_fighter,
                'winner': winner,
                'weight_class': "",
                'is_title_bout': 0.0,
                'gender': "Men",
                'method': "",
                'finish_round': 0.0,
                'total_rounds': 3.0,
                'time_sec': 0.0,
                'referee': "",
                'r_kd': 0,
                'r_sig_str': 0,
                'r_sig_str_att': 0,
                'r_sig_str_acc': 0.0,
                'r_str': 0,
                'r_str_att': 0,
                'r_str_acc': 0.0,
                'r_td': 0,
                'r_td_att': 0,
                'r_td_acc': 0.0,
                'r_sub_att': 0,
                'r_rev': 0,
                'r_ctrl_sec': 0,
                'b_kd': 0,
                'b_sig_str': 0,
                'b_sig_str_att': 0,
                'b_sig_str_acc': 0.0,
                'b_str': 0,
                'b_str_att': 0,
                'b_str_acc': 0.0,
                'b_td': 0,
                'b_td_att': 0,
                'b_td_acc': 0.0,
                'b_sub_att': 0,
                'b_rev': 0,
                'b_ctrl_sec': 0
            }
            
            # Extraire les statistiques disponibles
            if len(all_texts) >= 16:
                # KD (Knockdowns)
                if all_texts[3].text.strip().isdigit():
                    fight_data['r_kd'] = int(all_texts[3].text.strip())
                if all_texts[4].text.strip().isdigit():
                    fight_data['b_kd'] = int(all_texts[4].text.strip())
                
                # Sig Strikes - avec gestion des erreurs de format
                r_str_text = all_texts[5].text.strip()
                r_sig_str, r_sig_str_att, r_sig_str_acc = parse_fight_stats(r_str_text)
                fight_data['r_sig_str'] = r_sig_str
                fight_data['r_sig_str_att'] = r_sig_str_att
                fight_data['r_sig_str_acc'] = r_sig_str_acc
                fight_data['r_str'] = r_sig_str
                fight_data['r_str_att'] = r_sig_str_att
                fight_data['r_str_acc'] = r_sig_str_acc
                
                b_str_text = all_texts[6].text.strip()
                b_sig_str, b_sig_str_att, b_sig_str_acc = parse_fight_stats(b_str_text)
                fight_data['b_sig_str'] = b_sig_str
                fight_data['b_sig_str_att'] = b_sig_str_att
                fight_data['b_sig_str_acc'] = b_sig_str_acc
                fight_data['b_str'] = b_sig_str
                fight_data['b_str_att'] = b_sig_str_att
                fight_data['b_str_acc'] = b_sig_str_acc
                
                # Takedowns - avec gestion des erreurs de format
                r_td_text = all_texts[7].text.strip()
                r_td, r_td_att, r_td_acc = parse_fight_stats(r_td_text)
                fight_data['r_td'] = r_td
                fight_data['r_td_att'] = r_td_att
                fight_data['r_td_acc'] = r_td_acc
                
                b_td_text = all_texts[8].text.strip()
                b_td, b_td_att, b_td_acc = parse_fight_stats(b_td_text)
                fight_data['b_td'] = b_td
                fight_data['b_td_att'] = b_td_att
                fight_data['b_td_acc'] = b_td_acc
                
                # Sub attempts
                if all_texts[9].text.strip().isdigit():
                    fight_data['r_sub_att'] = int(all_texts[9].text.strip())
                if all_texts[10].text.strip().isdigit():
                    fight_data['b_sub_att'] = int(all_texts[10].text.strip())
                
                # Weight class
                weight_class_text = all_texts[11].text.strip()
                weight_class, is_title_bout, gender = clean_weight_class(weight_class_text)
                fight_data['weight_class'] = weight_class
                fight_data['is_title_bout'] = 1.0 if is_title_bout else 0.0
                fight_data['gender'] = gender
                
                # Method
                method = all_texts[12].text.strip()
                if "KO/TKO" in method:
                    fight_data['method'] = "KO/TKO"
                elif "SUB" in method:
                    fight_data['method'] = "Submission"
                elif "U-DEC" in method:
                    fight_data['method'] = "Decision - Unanimous"
                elif "S-DEC" in method:
                    fight_data['method'] = "Decision - Split"
                elif "M-DEC" in method:
                    fight_data['method'] = "Decision - Majority"
                else:
                    fight_data['method'] = method
                
                # Round
                if len(all_texts) > 14 and all_texts[14].text.strip().isdigit():
                    fight_data['finish_round'] = float(all_texts[14].text.strip())
                
                # Time
                if len(all_texts) > 15 and ":" in all_texts[15].text.strip():
                    time_str = all_texts[15].text.strip()
                    try:
                        minutes, seconds = time_str.split(':')
                        round_time_sec = int(minutes) * 60 + int(seconds)
                        prev_rounds_sec = (int(fight_data['finish_round']) - 1) * 300 if fight_data['finish_round'] > 0 else 0
                        fight_data['time_sec'] = prev_rounds_sec + round_time_sec
                    except Exception as e:
                        log(f"Erreur lors du parsing du temps: {e}", is_debug=True)
                
                # Total rounds (5 for title bouts, 3 otherwise)
                fight_data['total_rounds'] = 5.0 if fight_data['is_title_bout'] == 1.0 else 3.0
            
            # Récupérer l'arbitre si disponible
            if fight_index < len(fight_urls):
                referee = extract_referee_from_fight(fight_urls[fight_index])
                fight_data['referee'] = referee
            
            fights_data.append(fight_data)
            log(f"Combat {fight_index+1}: {red_fighter} vs {blue_fighter} - Vainqueur: {winner}")
            
            fight_index += 1
        except Exception as e:
            log(f"Erreur lors du traitement d'une ligne de combat: {e}")
            # Continuer avec la ligne suivante
            fight_index += 1
            continue
    
    return fights_data

def extract_fight_data(event_url, event_name):
    """Extrait les données complètes des combats d'un événement UFC"""
    log(f"Extraction des combats de l'événement: {event_name}")
    
    response = make_request(event_url)
    if not response:
        log("Échec de l'accès à l'événement.")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Récupérer les liens vers les détails des combats
    fight_links = soup.find_all('a', href=lambda href: href and 'fight-details' in href)
    fight_urls = [link.get('href') for link in fight_links]
    
    log(f"Nombre de liens de combat trouvés: {len(fight_urls)}")
    
    # Stratégie 1: Essayer d'extraire les données depuis les pages individuelles des combats
    fights_data = extract_fight_data_from_individual_pages(fight_urls, event_name)
    
    # Si aucun combat n'a été extrait, essayer la méthode alternative
    if not fights_data:
        log("Aucun combat extrait via les pages individuelles. Tentative avec le tableau de l'événement...")
        fights_data = extract_fight_data_from_table(soup, event_name)
    
    return fights_data

def load_fighter_stats(filename='content/fighters_stats.txt'):
    """Charge les statistiques des combattants à partir du fichier"""
    log(f"Chargement des statistiques des combattants depuis {filename}...")
    
    fighter_stats = {}
    
    if not os.path.exists(filename):
        log(f"Fichier {filename} non trouvé.")
        return fighter_stats
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            current_fighter = {}
            for line in f:
                line = line.strip()
                if line and ':' in line:
                    key, value = line.split(':', 1)
                    current_fighter[key.strip()] = value.strip()
                elif not line and current_fighter and 'name' in current_fighter:
                    fighter_stats[current_fighter['name']] = current_fighter
                    current_fighter = {}
            
            # Ajouter le dernier combattant si nécessaire
            if current_fighter and 'name' in current_fighter:
                fighter_stats[current_fighter['name']] = current_fighter
        
        log(f"Chargé {len(fighter_stats)} statistiques de combattants")
    except Exception as e:
        log(f"Erreur lors du chargement des statistiques: {e}")
    
    return fighter_stats

def extract_fighter_urls(event_url):
    """Extrait les URLs des profils de tous les combattants d'un événement UFC"""
    log(f"Extraction des URLs des combattants depuis: {event_url}")
    
    response = make_request(event_url)
    if not response:
        log("Échec de l'accès à l'événement.")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    fighter_urls = set()
    
    # Méthode 1: Liens directs dans la page de l'événement
    fighter_links = soup.find_all('a', href=lambda href: href and 'fighter-details' in href)
    for link in fighter_links:
        url = link.get('href')
        if url:
            fighter_urls.add(url)
            log(f"URL de combattant trouvée: {url}", is_debug=True)
    
    # Méthode 2: Parcourir les liens de combat
    if len(fighter_urls) < 2:  # Si peu de combattants trouvés
        fight_links = soup.find_all('a', href=lambda href: href and 'fight-details' in href)
        
        for fight_link in fight_links:
            fight_url = fight_link.get('href')
            
            # Récupérer les détails du combat
            fight_response = make_request(fight_url)
            if not fight_response:
                continue
            
            fight_soup = BeautifulSoup(fight_response.text, 'html.parser')
            
            # Extraire les liens des combattants
            for link in fight_soup.find_all('a', href=lambda href: href and 'fighter-details' in href):
                url = link.get('href')
                if url:
                    fighter_urls.add(url)
                    log(f"URL de combattant trouvée (via combat): {url}", is_debug=True)
    
    log(f"Total: {len(fighter_urls)} URLs de combattants extraites")
    return list(fighter_urls)

def get_fighter_stats(fighter_url):
    """Extrait les statistiques d'un combattant"""
    log(f"Récupération des stats pour: {fighter_url}", is_debug=True)
    
    response = make_request(fighter_url)
    if not response:
        log("Échec de l'accès au profil du combattant.")
        return None
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Nom du combattant
    name_elem = soup.find('span', class_='b-content__title-highlight')
    if not name_elem:
        log("Nom du combattant non trouvé.")
        return None
    
    fighter_name = name_elem.text.strip()
    log(f"Extraction des stats de: {fighter_name}")
    
    # Record (W-L-D)
    record_elem = soup.find('span', class_='b-content__title-record')
    if not record_elem:
        log("Record du combattant non trouvé.")
        return None
    
    record_text = record_elem.text.replace('Record:', '').strip()
    wins, losses, draws = 0, 0, 0
    
    record_match = re.match(r'(\d+)-(\d+)(?:-(\d+))?', record_text)
    if record_match:
        wins = int(record_match.group(1))
        losses = int(record_match.group(2))
        draws = int(record_match.group(3)) if record_match.group(3) else 0
    
    # Statistiques de base
    stats = {
        'name': fighter_name,
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'height': None,
        'weight': None,
        'reach': None,
        'stance': 'Unknown',
        'age': None,
        'SLpM': 0.0,
        'sig_str_acc': 0.0,
        'SApM': 0.0,
        'str_def': 0.0,
        'td_avg': 0.0,
        'td_acc': 0.0,
        'td_def': 0.0,
        'sub_avg': 0.0
    }
    
    # Extraire toutes les stats disponibles
    stats_elems = soup.find_all('li', class_='b-list__box-list-item b-list__box-list-item_type_block')
    
    for elem in stats_elems:
        text = elem.get_text(strip=True)
        
        # Hauteur
        if 'Height:' in text:
            height_match = re.search(r'Height:\s*(\d+)\'\s*(\d+)"', text)
            if height_match:
                feet, inches = int(height_match.group(1)), int(height_match.group(2))
                stats['height'] = round((feet * 30.48) + (inches * 2.54), 2)
        
        # Poids
        elif 'Weight:' in text:
            weight_match = re.search(r'Weight:\s*(\d+)\s*lbs', text)
            if weight_match:
                stats['weight'] = round(int(weight_match.group(1)) * 0.453592, 2)
        
        # Allonge
        elif 'Reach:' in text:
            reach_match = re.search(r'Reach:\s*(\d+\.?\d*)"', text)
            if reach_match:
                stats['reach'] = round(float(reach_match.group(1)) * 2.54, 2)
        
        # Stance
        elif 'STANCE:' in text:
            stance = text.replace('STANCE:', '').strip()
            stats['stance'] = stance if stance != '--' else 'Unknown'
        
        # Date de naissance / Âge
        elif 'DOB:' in text:
            dob_text = text.replace('DOB:', '').strip()
            if dob_text != '--':
                try:
                    # Essayer différents formats de date
                    for date_format in ["%b %d, %Y", "%B %d, %Y"]:
                        try:
                            dob = datetime.strptime(dob_text, date_format)
                            current_date = datetime.now()
                            stats['age'] = current_date.year - dob.year - ((current_date.month, current_date.day) < (dob.month, dob.day))
                            break
                        except ValueError:
                            continue
                except Exception as e:
                    log(f"Erreur lors du parsing de la date de naissance: {e}", is_debug=True)
        
        # SLpM (Significant Strikes Landed per Minute)
        elif 'SLpM:' in text:
            slpm_match = re.search(r'SLpM:\s*([\d\.]+)', text)
            if slpm_match and slpm_match.group(1) != '--':
                try:
                    stats['SLpM'] = float(slpm_match.group(1))
                except:
                    pass
        
        # Striking Accuracy
        elif 'Str. Acc.:' in text:
            acc_match = re.search(r'Str. Acc.:\s*([\d\.]+)%', text)
            if acc_match and acc_match.group(1) != '--':
                try:
                    stats['sig_str_acc'] = float(acc_match.group(1)) / 100
                except:
                    pass
        
        # SApM (Significant Strikes Absorbed per Minute)
        elif 'SApM:' in text:
            sapm_match = re.search(r'SApM:\s*([\d\.]+)', text)
            if sapm_match and sapm_match.group(1) != '--':
                try:
                    stats['SApM'] = float(sapm_match.group(1))
                except:
                    pass
        
        # Striking Defense
        elif 'Str. Def:' in text:
            def_match = re.search(r'Str. Def:\s*([\d\.]+)%', text)
            if def_match and def_match.group(1) != '--':
                try:
                    stats['str_def'] = float(def_match.group(1)) / 100
                except:
                    pass
        
        # Takedown Average
        elif 'TD Avg.:' in text:
            td_match = re.search(r'TD Avg.:\s*([\d\.]+)', text)
            if td_match and td_match.group(1) != '--':
                try:
                    stats['td_avg'] = float(td_match.group(1))
                except:
                    pass
        
        # Takedown Accuracy
        elif 'TD Acc.:' in text:
            td_acc_match = re.search(r'TD Acc.:\s*([\d\.]+)%', text)
            if td_acc_match and td_acc_match.group(1) != '--':
                try:
                    stats['td_acc'] = float(td_acc_match.group(1)) / 100
                except:
                    pass
        
        # Takedown Defense
        elif 'TD Def.:' in text:
            td_def_match = re.search(r'TD Def.:\s*([\d\.]+)%', text)
            if td_def_match and td_def_match.group(1) != '--':
                try:
                    stats['td_def'] = float(td_def_match.group(1)) / 100
                except:
                    pass
        
        # Submission Average
        elif 'Sub. Avg.:' in text:
            sub_match = re.search(r'Sub. Avg.:\s*([\d\.]+)', text)
            if sub_match and sub_match.group(1) != '--':
                try:
                    stats['sub_avg'] = float(sub_match.group(1))
                except:
                    pass
    
    return stats

def update_fighters_stats_file(fighter_urls, filename='content/fighters_stats.txt'):
    """Met à jour le fichier fighters_stats.txt avec les nouveaux combattants ou les mises à jour"""
    log(f"Mise à jour du fichier {filename}...")
    
    # Créer le répertoire si nécessaire
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Lire les combattants existants
    existing_fighters = {}
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            current_fighter = {}
            for line in f:
                line = line.strip()
                if line and ':' in line:
                    key, value = line.split(':', 1)
                    current_fighter[key.strip()] = value.strip()
                elif not line and current_fighter and 'name' in current_fighter:
                    existing_fighters[current_fighter['name']] = current_fighter
                    current_fighter = {}
            
            # Ajouter le dernier combattant si nécessaire
            if current_fighter and 'name' in current_fighter:
                existing_fighters[current_fighter['name']] = current_fighter
    
    log(f"Combattants existants: {len(existing_fighters)}")
    
    # Récupérer les données des combattants
    new_fighters = []
    updated_fighters = []
    all_fighter_stats = {}
    
    for fighter_url in fighter_urls:
        fighter_data = get_fighter_stats(fighter_url)
        if not fighter_data:
            continue
        
        fighter_name = fighter_data['name']
        all_fighter_stats[fighter_name] = fighter_data
        
        if fighter_name not in existing_fighters:
            new_fighters.append(fighter_data)
            log(f"Nouveau combattant: {fighter_name}")
        else:
            # Vérifier si les victoires/défaites ont changé
            old_fighter = existing_fighters[fighter_name]
            old_wins = int(old_fighter.get('wins', 0))
            old_losses = int(old_fighter.get('losses', 0))
            
            if fighter_data['wins'] > old_wins or fighter_data['losses'] > old_losses:
                updated_fighters.append(fighter_data)
                log(f"Mise à jour: {fighter_name} ({old_wins}-{old_losses} -> {fighter_data['wins']}-{fighter_data['losses']})")
            else:
                # Utiliser les stats existantes
                all_fighter_stats[fighter_name] = old_fighter
    
    log(f"Nouveaux combattants: {len(new_fighters)}, Mises à jour: {len(updated_fighters)}")
    
    # Si aucun changement, terminer
    if not new_fighters and not updated_fighters:
        log("Aucune mise à jour nécessaire pour les combattants.")
        return all_fighter_stats
    
    # Créer une sauvegarde du fichier existant
    if os.path.exists(filename):
        backup_name = f"{filename}.{int(time.time())}.backup"
        try:
            import shutil
            shutil.copy2(filename, backup_name)
            log(f"Sauvegarde créée: {backup_name}")
        except Exception as e:
            log(f"Erreur lors de la création de la sauvegarde: {e}")
    
    # Mise à jour des combattants existants
    for fighter in updated_fighters:
        existing_fighters[fighter['name']] = fighter
    
    # Ajouter les nouveaux combattants
    for fighter in new_fighters:
        existing_fighters[fighter['name']] = fighter
    
    # Écrire tous les combattants dans le fichier
    with open(filename, 'w', encoding='utf-8') as f:
        for fighter_data in existing_fighters.values():
            for key, value in fighter_data.items():
                if value is not None:
                    f.write(f"{key}: {value}\n")
            f.write('\n')
    
    log(f"Fichier {filename} mis à jour avec succès.")
    
    # Fusionner les stats existantes avec les nouvelles
    all_fighter_stats.update(existing_fighters)
    return all_fighter_stats

def enrich_fights_with_fighter_stats(fights_data, fighter_stats):
    """Enrichit les données de combat avec les statistiques globales des combattants"""
    log("Enrichissement des données de combat avec les statistiques des combattants...")
    
    enriched_fights = []
    
    for fight in fights_data:
        enriched_fight = fight.copy()
        
        # Récupérer les statistiques des combattants rouge et bleu
        r_fighter = fight['r_fighter']
        b_fighter = fight['b_fighter']
        
        # Vérifier que les données de base du combat sont présentes
        for key in ['r_kd', 'r_sig_str', 'r_sig_str_att', 'r_sig_str_acc', 
                    'r_str', 'r_str_att', 'r_str_acc', 
                    'r_td', 'r_td_att', 'r_td_acc',
                    'r_sub_att', 'r_rev', 'r_ctrl_sec',
                    'b_kd', 'b_sig_str', 'b_sig_str_att', 'b_sig_str_acc',
                    'b_str', 'b_str_att', 'b_str_acc',
                    'b_td', 'b_td_att', 'b_td_acc',
                    'b_sub_att', 'b_rev', 'b_ctrl_sec']:
            if key not in enriched_fight or enriched_fight[key] is None:
                if key.endswith('acc'):
                    enriched_fight[key] = 0.0
                else:
                    enriched_fight[key] = 0
        
        # Ajouter les statistiques du combattant rouge
        if r_fighter in fighter_stats:
            r_stats = fighter_stats[r_fighter]
            enriched_fight['r_wins_total'] = float(r_stats.get('wins', 0))
            enriched_fight['r_losses_total'] = float(r_stats.get('losses', 0))
            enriched_fight['r_age'] = float(r_stats.get('age', 0)) if r_stats.get('age') else 0.0
            enriched_fight['r_height'] = float(r_stats.get('height', 0)) if r_stats.get('height') else 0.0
            enriched_fight['r_weight'] = float(r_stats.get('weight', 0)) if r_stats.get('weight') else 0.0
            enriched_fight['r_reach'] = float(r_stats.get('reach', 0)) if r_stats.get('reach') else 0.0
            enriched_fight['r_stance'] = r_stats.get('stance', 'Unknown')
            enriched_fight['r_SLpM_total'] = float(r_stats.get('SLpM', 0))
            enriched_fight['r_SApM_total'] = float(r_stats.get('SApM', 0))
            enriched_fight['r_sig_str_acc_total'] = float(r_stats.get('sig_str_acc', 0))
            enriched_fight['r_td_acc_total'] = float(r_stats.get('td_acc', 0))
            enriched_fight['r_str_def_total'] = float(r_stats.get('str_def', 0))
            enriched_fight['r_td_def_total'] = float(r_stats.get('td_def', 0))
            enriched_fight['r_sub_avg'] = float(r_stats.get('sub_avg', 0))
            enriched_fight['r_td_avg'] = float(r_stats.get('td_avg', 0))
        else:
            # Valeurs par défaut si le combattant n'est pas trouvé
            for key in ['r_wins_total', 'r_losses_total', 'r_age', 'r_height', 'r_weight', 'r_reach',
                        'r_SLpM_total', 'r_SApM_total', 'r_sig_str_acc_total', 'r_td_acc_total',
                        'r_str_def_total', 'r_td_def_total', 'r_sub_avg', 'r_td_avg']:
                enriched_fight[key] = 0.0
            enriched_fight['r_stance'] = 'Unknown'
        
        # Ajouter les statistiques du combattant bleu
        if b_fighter in fighter_stats:
            b_stats = fighter_stats[b_fighter]
            enriched_fight['b_wins_total'] = float(b_stats.get('wins', 0))
            enriched_fight['b_losses_total'] = float(b_stats.get('losses', 0))
            enriched_fight['b_age'] = float(b_stats.get('age', 0)) if b_stats.get('age') else 0.0
            enriched_fight['b_height'] = float(b_stats.get('height', 0)) if b_stats.get('height') else 0.0
            enriched_fight['b_weight'] = float(b_stats.get('weight', 0)) if b_stats.get('weight') else 0.0
            enriched_fight['b_reach'] = float(b_stats.get('reach', 0)) if b_stats.get('reach') else 0.0
            enriched_fight['b_stance'] = b_stats.get('stance', 'Unknown')
            enriched_fight['b_SLpM_total'] = float(b_stats.get('SLpM', 0))
            enriched_fight['b_SApM_total'] = float(b_stats.get('SApM', 0))
            enriched_fight['b_sig_str_acc_total'] = float(b_stats.get('sig_str_acc', 0))
            enriched_fight['b_td_acc_total'] = float(b_stats.get('td_acc', 0))
            enriched_fight['b_str_def_total'] = float(b_stats.get('str_def', 0))
            enriched_fight['b_td_def_total'] = float(b_stats.get('td_def', 0))
            enriched_fight['b_sub_avg'] = float(b_stats.get('sub_avg', 0))
            enriched_fight['b_td_avg'] = float(b_stats.get('td_avg', 0))
        else:
            # Valeurs par défaut si le combattant n'est pas trouvé
            for key in ['b_wins_total', 'b_losses_total', 'b_age', 'b_height', 'b_weight', 'b_reach',
                        'b_SLpM_total', 'b_SApM_total', 'b_sig_str_acc_total', 'b_td_acc_total',
                        'b_str_def_total', 'b_td_def_total', 'b_sub_avg', 'b_td_avg']:
                enriched_fight[key] = 0.0
            enriched_fight['b_stance'] = 'Unknown'
        
        # Calculer les différences entre les combattants
        enriched_fight['kd_diff'] = enriched_fight.get('r_kd', 0) - enriched_fight.get('b_kd', 0)
        enriched_fight['sig_str_diff'] = enriched_fight.get('r_sig_str', 0) - enriched_fight.get('b_sig_str', 0)
        enriched_fight['sig_str_att_diff'] = enriched_fight.get('r_sig_str_att', 0) - enriched_fight.get('b_sig_str_att', 0)
        enriched_fight['sig_str_acc_diff'] = enriched_fight.get('r_sig_str_acc', 0) - enriched_fight.get('b_sig_str_acc', 0)
        enriched_fight['str_diff'] = enriched_fight.get('r_str', 0) - enriched_fight.get('b_str', 0)
        enriched_fight['str_att_diff'] = enriched_fight.get('r_str_att', 0) - enriched_fight.get('b_str_att', 0)
        enriched_fight['str_acc_diff'] = enriched_fight.get('r_str_acc', 0) - enriched_fight.get('b_str_acc', 0)
        enriched_fight['td_diff'] = enriched_fight.get('r_td', 0) - enriched_fight.get('b_td', 0)
        enriched_fight['td_att_diff'] = enriched_fight.get('r_td_att', 0) - enriched_fight.get('b_td_att', 0)
        enriched_fight['td_acc_diff'] = enriched_fight.get('r_td_acc', 0) - enriched_fight.get('b_td_acc', 0)
        enriched_fight['sub_att_diff'] = enriched_fight.get('r_sub_att', 0) - enriched_fight.get('b_sub_att', 0)
        enriched_fight['rev_diff'] = enriched_fight.get('r_rev', 0) - enriched_fight.get('b_rev', 0)
        enriched_fight['ctrl_sec_diff'] = enriched_fight.get('r_ctrl_sec', 0) - enriched_fight.get('b_ctrl_sec', 0)
        
        # Différences dans les statistiques globales
        enriched_fight['wins_total_diff'] = enriched_fight.get('r_wins_total', 0) - enriched_fight.get('b_wins_total', 0)
        enriched_fight['losses_total_diff'] = enriched_fight.get('r_losses_total', 0) - enriched_fight.get('b_losses_total', 0)
        enriched_fight['age_diff'] = enriched_fight.get('r_age', 0) - enriched_fight.get('b_age', 0)
        enriched_fight['height_diff'] = enriched_fight.get('r_height', 0) - enriched_fight.get('b_height', 0)
        enriched_fight['weight_diff'] = enriched_fight.get('r_weight', 0) - enriched_fight.get('b_weight', 0)
        enriched_fight['reach_diff'] = enriched_fight.get('r_reach', 0) - enriched_fight.get('b_reach', 0)
        enriched_fight['SLpM_total_diff'] = enriched_fight.get('r_SLpM_total', 0) - enriched_fight.get('b_SLpM_total', 0)
        enriched_fight['SApM_total_diff'] = enriched_fight.get('r_SApM_total', 0) - enriched_fight.get('b_SApM_total', 0)
        enriched_fight['sig_str_acc_total_diff'] = enriched_fight.get('r_sig_str_acc_total', 0) - enriched_fight.get('b_sig_str_acc_total', 0)
        enriched_fight['td_acc_total_diff'] = enriched_fight.get('r_td_acc_total', 0) - enriched_fight.get('b_td_acc_total', 0)
        enriched_fight['str_def_total_diff'] = enriched_fight.get('r_str_def_total', 0) - enriched_fight.get('b_str_def_total', 0)
        enriched_fight['td_def_total_diff'] = enriched_fight.get('r_td_def_total', 0) - enriched_fight.get('b_td_def_total', 0)
        enriched_fight['sub_avg_diff'] = enriched_fight.get('r_sub_avg', 0) - enriched_fight.get('b_sub_avg', 0)
        enriched_fight['td_avg_diff'] = enriched_fight.get('r_td_avg', 0) - enriched_fight.get('b_td_avg', 0)
        
        enriched_fights.append(enriched_fight)
    
    log(f"Enrichissement terminé pour {len(enriched_fights)} combats")
    return enriched_fights

def check_event_exists(event_name, csv_file='ufc_data.csv'):
    """Vérifie si un événement existe déjà dans le fichier de données"""
    if not os.path.exists(csv_file):
        return False
    
    try:
        df = pd.read_csv(csv_file)
        
        # Vérifier différentes colonnes qui pourraient contenir le nom de l'événement
        event_cols = ['event_name', 'event']
        
        for col in event_cols:
            if col in df.columns:
                if df[col].astype(str).str.contains(event_name, regex=False).any():
                    return True
    except Exception as e:
        log(f"Erreur lors de la vérification de l'existence de l'événement: {e}")
    
    return False

def update_ufc_data_file(fights_data, filename='ufc_data.csv'):
    """Met à jour le fichier ufc_data.csv en ajoutant les nouveaux combats"""
    log(f"Mise à jour du fichier {filename}...")
    
    if not fights_data:
        log("Aucun combat à ajouter.")
        return False
    
    # Créer un DataFrame avec les nouvelles données
    new_df = pd.DataFrame(fights_data)
    log(f"Nouveau DataFrame créé avec {len(new_df)} combats et {len(new_df.columns)} colonnes")
    
    # Vérifier si le fichier existe
    if os.path.exists(filename):
        # Créer une sauvegarde
        backup_filename = f"{filename}.{int(time.time())}.backup"
        try:
            import shutil
            shutil.copy2(filename, backup_filename)
            log(f"Sauvegarde créée: {backup_filename}")
        except Exception as e:
            log(f"Erreur lors de la création de la sauvegarde: {e}")
        
        # Charger le fichier existant
        existing_df = pd.read_csv(filename)
        log(f"Fichier existant chargé avec {len(existing_df)} combats et {len(existing_df.columns)} colonnes")
        
        # Vérifier si l'événement existe déjà
        event_name = fights_data[0]['event_name']
        
        # Vérifier différentes colonnes qui pourraient contenir le nom de l'événement
        event_cols = ['event_name', 'event']
        event_exists = False
        
        for col in event_cols:
            if col in existing_df.columns:
                matches = existing_df[col].astype(str).str.contains(event_name, regex=False)
                if matches.any():
                    event_exists = True
                    log(f"Événement '{event_name}' déjà présent dans la colonne '{col}'")
                    break
        
        if not event_exists:
            log(f"L'événement {event_name} n'existe pas encore dans {filename}, ajout en cours...")
            
            # Harmoniser les colonnes
            all_columns = set(existing_df.columns) | set(new_df.columns)
            log(f"Nombre total de colonnes à harmoniser: {len(all_columns)}")
            
            # Ajouter les colonnes manquantes
            for col in all_columns:
                if col not in existing_df.columns:
                    existing_df[col] = np.nan if col in new_df.columns and pd.api.types.is_numeric_dtype(new_df[col]) else None
                    log(f"Ajout de la colonne '{col}' au DataFrame existant", is_debug=True)
                
                if col not in new_df.columns:
                    new_df[col] = np.nan if col in existing_df.columns and pd.api.types.is_numeric_dtype(existing_df[col]) else None
                    log(f"Ajout de la colonne '{col}' au nouveau DataFrame", is_debug=True)
            
            # Vérifier les types de données
            for col in all_columns:
                if col in existing_df.columns and col in new_df.columns:
                    # Convertir les colonnes numériques
                    if pd.api.types.is_numeric_dtype(existing_df[col]) or pd.api.types.is_numeric_dtype(new_df[col]):
                        try:
                            existing_df[col] = pd.to_numeric(existing_df[col], errors='coerce')
                            new_df[col] = pd.to_numeric(new_df[col], errors='coerce')
                        except Exception as e:
                            log(f"Erreur lors de la conversion numérique de la colonne '{col}': {e}", is_debug=True)
            
            # Réorganiser les colonnes dans le même ordre
            existing_cols = list(existing_df.columns)
            new_df = new_df.reindex(columns=existing_cols)
            
            # Concaténer les DataFrames (nouveaux combats au début)
            try:
                combined_df = pd.concat([new_df, existing_df], ignore_index=True)
                combined_df.to_csv(filename, index=False)
                log(f"Ajouté {len(new_df)} combats au fichier {filename}.")
                return True
            except Exception as e:
                log(f"Erreur lors de la concaténation des DataFrames: {e}")
                
                # Sauvegarder seulement les nouveaux combats dans un fichier séparé
                new_filename = f"{filename}.new.{int(time.time())}.csv"
                new_df.to_csv(new_filename, index=False)
                log(f"Nouveaux combats sauvegardés dans {new_filename}")
                return True
        else:
            log(f"L'événement {event_name} existe déjà dans {filename}, aucune mise à jour nécessaire.")
            return False
    else:
        # Créer un nouveau fichier
        try:
            new_df.to_csv(filename, index=False)
            log(f"Nouveau fichier {filename} créé avec {len(new_df)} combats")
            return True
        except Exception as e:
            log(f"Erreur lors de la création du fichier: {e}")
            return False

def main():
    """Fonction principale"""
    log("=== Démarrage du script de mise à jour automatique UFC ===")
    log(f"Date et heure actuelles: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Récupérer les événements récents
    recent_events = get_recent_events(MAX_EVENTS_TO_CHECK)
    
    if not recent_events:
        log("Aucun événement récent trouvé. Fin du script.")
        return
    
    # 2. Charger les stats des combattants existants
    all_fighter_stats = load_fighter_stats()
    updated = False
    
    # 3. Vérifier chaque événement récent
    for event in recent_events:
        event_name = event['name']
        event_url = event['url']
        
        log(f"Analyse de l'événement: {event_name}")
        
        # Vérifier si l'événement existe déjà dans le fichier
        if check_event_exists(event_name):
            log(f"L'événement {event_name} existe déjà dans ufc_data.csv, passage au suivant.")
            continue
        
        # Extraire les données des combats
        fights_data = extract_fight_data(event_url, event_name)
        
        if not fights_data:
            log(f"Aucun combat trouvé pour l'événement {event_name}, passage au suivant.")
            continue
        
        log(f"Extraction réussie de {len(fights_data)} combats pour {event_name}")
        
        # Extraire les URLs des combattants
        fighter_urls = extract_fighter_urls(event_url)
        
        if fighter_urls:
            log(f"Extraction réussie de {len(fighter_urls)} URLs de combattants")
            
            # Mettre à jour le fichier fighters_stats.txt
            fighter_stats = update_fighters_stats_file(fighter_urls)
            
            # Fusionner avec les stats existantes
            all_fighter_stats.update(fighter_stats)
        
        # Enrichir les données de combat avec les statistiques des combattants
        enriched_fights = enrich_fights_with_fighter_stats(fights_data, all_fighter_stats)
        
        # Mettre à jour le fichier ufc_data.csv
        update_success = update_ufc_data_file(enriched_fights)
        
        if update_success:
            log(f"Mise à jour réussie pour l'événement {event_name}")
            updated = True
        else:
            log(f"Pas de mise à jour pour l'événement {event_name}")
    
    if updated:
        log("=== Mise à jour terminée avec succès! ===")
    else:
        log("=== Aucun nouvel événement à ajouter. ===")

if __name__ == "__main__":
    # Configurer le mode verbeux depuis la ligne de commande
    if len(sys.argv) > 1 and sys.argv[1] in ['-v', '--verbose']:
        VERBOSE = True
    
    main()
