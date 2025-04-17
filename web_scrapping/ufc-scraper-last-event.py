#!/usr/bin/env python
# coding: utf-8

import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import re
from datetime import datetime
import time
import random

# Configuration des headers pour éviter d'être bloqué
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Nombre d'événements à vérifier 
MAX_EVENTS_TO_CHECK = 30

def get_completed_event_url():
    """Récupère l'URL du dernier événement UFC complété (qui a eu lieu)"""
    print("Recherche du dernier événement UFC complété...")
    
    url = "http://ufcstats.com/statistics/events/completed?page=all"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code != 200:
        print(f"Échec de la récupération des événements. Code: {response.status_code}")
        return None, None
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Récupérer les liens vers tous les événements (triés du plus récent au plus ancien)
    event_links = soup.find_all('a', href=lambda href: href and 'event-details' in href)
    
    if not event_links:
        print("Aucun lien d'événement trouvé.")
        return None, None
    
    print(f"Vérification des {min(MAX_EVENTS_TO_CHECK, len(event_links))} événements les plus récents...")
    
    # Parcourir les événements pour trouver le premier qui a eu lieu
    for i, link in enumerate(event_links[:MAX_EVENTS_TO_CHECK]):
        if i >= MAX_EVENTS_TO_CHECK:
            break
            
        event_url = link.get('href')
        event_name = link.text.strip()
        
        print(f"Vérification de l'événement {i+1}: {event_name}")
        
        # Vérifier si l'événement a des résultats
        event_response = requests.get(event_url, headers=HEADERS)
        if event_response.status_code != 200:
            continue
            
        event_soup = BeautifulSoup(event_response.text, 'html.parser')
        
        # Méthode 1: Chercher les scores/résultats dans le tableau des combats
        fight_table = event_soup.find('table', class_='b-fight-details__table')
        completed_fights = False
        
        if fight_table:
            # Chercher des colonnes de données numériques complètes (indiquant que les combats ont eu lieu)
            rows = fight_table.find_all('tr', class_='b-fight-details__table-row')
            for row in rows:
                if 'b-fight-details__table-head' in row.get('class', []):
                    continue  # Ignorer les en-têtes
                
                cells = row.find_all('p', class_='b-fight-details__table-text')
                if len(cells) >= 16 and cells[3].text.strip() != '':
                    # Si les données comme les knockdowns sont remplies, le combat a eu lieu
                    completed_fights = True
                    break
        
        # Méthode 2: Vérifier les résultats W/L/D
        if not completed_fights:
            result_elements = event_soup.find_all('i', class_='b-fight-details__person-status')
            for result in result_elements:
                if result.text.strip() in ['W', 'L', 'D']:
                    completed_fights = True
                    break
        
        # Méthode 3: Vérifier si l'événement est dans le passé
        event_date = None
        date_elements = event_soup.find_all('li', class_='b-list__box-list-item')
        for date_elem in date_elements:
            text = date_elem.get_text(strip=True)
            if "Date:" in text:
                date_text = text.replace("Date:", "").strip()
                try:
                    # Essayer différents formats de date
                    try:
                        event_date = datetime.strptime(date_text, "%B %d, %Y")
                    except ValueError:
                        try:
                            event_date = datetime.strptime(date_text, "%b %d, %Y")
                        except ValueError:
                            match = re.search(r'([A-Za-z]+)\s+(\d+),?\s+(\d{4})', date_text)
                            if match:
                                month, day, year = match.groups()
                                event_date = datetime.strptime(f"{month} {day} {year}", "%B %d %Y")
                except:
                    pass
                
                break
        
        if event_date and event_date < datetime.now():
            completed_fights = True
        
        # Méthode 4: Vérifier si des détails de combat sont présents
        if not completed_fights:
            # Chercher des liens vers les détails de combat
            fight_links = event_soup.find_all('a', href=lambda href: href and 'fight-details' in href)
            if fight_links:
                # Vérifier si les liens contiennent des résultats
                for fight_link in fight_links[:1]:  # Vérifier seulement le premier lien
                    fight_url = fight_link.get('href')
                    fight_response = requests.get(fight_url, headers=HEADERS)
                    if fight_response.status_code == 200:
                        fight_soup = BeautifulSoup(fight_response.text, 'html.parser')
                        fight_result = fight_soup.find('i', class_='b-fight-details__text-item')
                        if fight_result and "win" in fight_result.text.lower():
                            completed_fights = True
                            break
        
        # Si l'événement a des combats complétés, le retourner
        if completed_fights:
            print(f"Événement complété trouvé: {event_name}")
            
            # Vérifier que des données de combat existent
            fight_data = extract_event_data(event_url)
            if fight_data and len(fight_data) > 0:
                print(f"Événement avec {len(fight_data)} combats complets trouvé!")
                return event_url, event_name
            else:
                print(f"Événement sans données de combat complètes, continuant la recherche...")
        
        # Pause pour éviter d'être bloqué
        time.sleep(random.uniform(0.5, 1.0))
    
    print("Aucun événement complété récent trouvé avec des données.")
    return None, None

def extract_event_data(event_url):
    """Extrait les données des combats d'un événement UFC"""
    print(f"Extraction des données de: {event_url}")
    
    response = requests.get(event_url, headers=HEADERS)
    if response.status_code != 200:
        print(f"Échec de l'accès à l'événement. Code: {response.status_code}")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Récupérer le nom de l'événement
    event_name_elem = soup.find('span', class_='b-content__title-highlight')
    event_name = event_name_elem.text.strip() if event_name_elem else "Unknown Event"
    
    # Récupérer les métadonnées (date, lieu)
    meta_data = []
    meta_info = soup.find_all('li', class_='b-list__box-list-item')
    for meta in meta_info:
        meta_text = meta.get_text(strip=True)
        meta_text = meta_text.replace('Date:', '').replace('Location:', '').strip()
        meta_data.append(meta_text)
    
    date = meta_data[0] if len(meta_data) > 0 else ""
    location = meta_data[1] if len(meta_data) > 1 else ""
    
    # Récupérer les détails des combats
    fights_data = []
    fight_table = soup.find('table', class_='b-fight-details__table')
    
    if fight_table:
        rows = fight_table.find_all('tr', class_='b-fight-details__table-row')
        
        for row in rows:
            # Ignorer les en-têtes
            if 'b-fight-details__table-head' in row.get('class', []):
                continue
                
            cells = row.find_all('p', class_='b-fight-details__table-text')
            
            if len(cells) >= 16:
                # Récupérer le résultat du combat (vainqueur)
                fighter_status_elements = row.find_all('i', class_='b-fight-details__person-status')
                winner = "Draw"  # Par défaut, match nul
                
                if len(fighter_status_elements) >= 2:
                    r_status = fighter_status_elements[0].text.strip() if len(fighter_status_elements) > 0 else ""
                    b_status = fighter_status_elements[1].text.strip() if len(fighter_status_elements) > 1 else ""
                    
                    if r_status == "W":
                        winner = cells[1].text.strip()  # Red fighter a gagné
                    elif b_status == "W":
                        winner = cells[2].text.strip()  # Blue fighter a gagné
                
                fight_data = {
                    'event': event_name,
                    'date': date,
                    'location': location,
                    'r_fighter': cells[1].text.strip(),
                    'b_fighter': cells[2].text.strip(),
                    'r_kd': cells[3].text.strip(),
                    'b_kd': cells[4].text.strip(),
                    'r_str': cells[5].text.strip(),
                    'b_str': cells[6].text.strip(),
                    'r_td': cells[7].text.strip(),
                    'b_td': cells[8].text.strip(),
                    'r_sub': cells[9].text.strip(),
                    'b_sub': cells[10].text.strip(),
                    'weight_class': cells[11].text.strip(),
                    'method': cells[12].text.strip(),
                    'round': cells[14].text.strip(),
                    'time': cells[15].text.strip(),
                    'winner': winner  # Ajout du vainqueur
                }
                fights_data.append(fight_data)
                print(f"Combat trouvé: {fight_data['r_fighter']} vs {fight_data['b_fighter']} - Vainqueur: {winner}")
    
    # Si aucun combat trouvé avec la méthode standard, essayer une méthode alternative
    if not fights_data:
        print("Aucun combat trouvé avec la méthode standard, tentative avec des méthodes alternatives...")
        # Chercher dans les sections de détails
        sections = soup.find_all('section', class_='b-statistics__section_details')
        for section in sections:
            rows = section.find_all('tr', class_='b-fight-details__table-row')
            
            for row in rows:
                if 'b-fight-details__table-head' in row.get('class', []):
                    continue
                    
                cells = row.find_all('p', class_='b-fight-details__table-text')
                
                if len(cells) >= 16:
                    # Récupérer le résultat du combat (vainqueur) - méthode alternative
                    fighter_status_elements = row.find_all('i', class_='b-fight-details__person-status')
                    winner = "Draw"  # Par défaut, match nul
                    
                    if len(fighter_status_elements) >= 2:
                        r_status = fighter_status_elements[0].text.strip() if len(fighter_status_elements) > 0 else ""
                        b_status = fighter_status_elements[1].text.strip() if len(fighter_status_elements) > 1 else ""
                        
                        if r_status == "W":
                            winner = cells[1].text.strip()  # Red fighter a gagné
                        elif b_status == "W":
                            winner = cells[2].text.strip()  # Blue fighter a gagné
                    
                    fight_data = {
                        'event': event_name,
                        'date': date,
                        'location': location,
                        'r_fighter': cells[1].text.strip(),
                        'b_fighter': cells[2].text.strip(),
                        'r_kd': cells[3].text.strip(),
                        'b_kd': cells[4].text.strip(),
                        'r_str': cells[5].text.strip(),
                        'b_str': cells[6].text.strip(),
                        'r_td': cells[7].text.strip(),
                        'b_td': cells[8].text.strip(),
                        'r_sub': cells[9].text.strip(),
                        'b_sub': cells[10].text.strip(),
                        'weight_class': cells[11].text.strip(),
                        'method': cells[12].text.strip(),
                        'round': cells[14].text.strip(),
                        'time': cells[15].text.strip(),
                        'winner': winner  # Ajout du vainqueur
                    }
                    fights_data.append(fight_data)
                    print(f"Combat trouvé (section détails): {fight_data['r_fighter']} vs {fight_data['b_fighter']} - Vainqueur: {winner}")
    
    # Vérifier si les combats ont des données valides (knockdowns, strikes, etc.)
    valid_fights = []
    for fight in fights_data:
        # Tous les combats sont considérés comme valides maintenant
        valid_fights.append(fight)
    
    # Si aucun combat valide, essayer d'autres méthodes
    if not valid_fights:
        print("Aucun combat avec données valides trouvé, tentative avec d'autres méthodes...")
        # Si aucun combat valide, essayer une dernière méthode
        all_rows = soup.find_all('tr', class_='b-fight-details__table-row')
        if all_rows:
            for row in all_rows:
                cells = row.find_all('p', class_='b-fight-details__table-text')
                if len(cells) >= 3:  # Au moins les noms des combattants
                    # Récupérer le résultat du combat (vainqueur) - dernière méthode
                    fighter_status_elements = row.find_all('i', class_='b-fight-details__person-status')
                    winner = "Unknown"  # Par défaut
                    
                    if len(fighter_status_elements) >= 2:
                        r_status = fighter_status_elements[0].text.strip() if len(fighter_status_elements) > 0 else ""
                        b_status = fighter_status_elements[1].text.strip() if len(fighter_status_elements) > 1 else ""
                        
                        if r_status == "W":
                            winner = cells[1].text.strip() if len(cells) > 1 else "Unknown"
                        elif b_status == "W":
                            winner = cells[2].text.strip() if len(cells) > 2 else "Unknown"
                    
                    fight_data = {
                        'event': event_name,
                        'date': date,
                        'location': location,
                        'r_fighter': cells[1].text.strip() if len(cells) > 1 else "",
                        'b_fighter': cells[2].text.strip() if len(cells) > 2 else "",
                        'weight_class': cells[11].text.strip() if len(cells) > 11 else "",
                        'method': cells[12].text.strip() if len(cells) > 12 else "",
                        'round': cells[14].text.strip() if len(cells) > 14 else "",
                        'time': cells[15].text.strip() if len(cells) > 15 else "",
                        'winner': winner  # Ajout du vainqueur
                    }
                    valid_fights.append(fight_data)
                    print(f"Combat basique trouvé: {fight_data['r_fighter']} vs {fight_data['b_fighter']} - Vainqueur: {winner}")
    
    # Dernière tentative: utiliser directement les liens des combattants
    if not valid_fights:
        print("Dernière tentative: utilisation des liens des combattants...")
        fighter_links = soup.find_all('a', href=lambda href: href and 'fighter-details' in href)
        fighter_names = [link.text.strip() for link in fighter_links]
        
        if len(fighter_names) >= 2 and len(fighter_names) % 2 == 0:
            for i in range(0, len(fighter_names), 2):
                if i+1 < len(fighter_names):
                    valid_fights.append({
                        'event': event_name,
                        'date': date,
                        'location': location,
                        'r_fighter': fighter_names[i],
                        'b_fighter': fighter_names[i+1],
                        'weight_class': "Unknown",
                        'method': "Unknown",
                        'round': "0",
                        'time': "0:00",
                        'winner': "Unknown"  # Ajout du vainqueur (inconnu dans ce cas)
                    })
                    print(f"Paire de combattants trouvée: {fighter_names[i]} vs {fighter_names[i+1]}")
    
    return valid_fights

def get_fight_links(event_url):
    """Récupère les liens vers les détails des combats"""
    print(f"Récupération des liens de combats pour: {event_url}")
    
    response = requests.get(event_url, headers=HEADERS)
    if response.status_code != 200:
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Chercher les liens vers les détails des combats
    fight_links = soup.find_all('a', href=lambda href: href and 'fight-details' in href)
    fight_urls = [link.get('href') for link in fight_links]
    
    print(f"Trouvé {len(fight_urls)} liens de combat")
    return fight_urls

def update_ufc_data_file(fights_data, filename='ufc_data.csv'):
    """Met à jour le fichier ufc_data.csv en ajoutant les nouveaux combats au début"""
    print(f"Mise à jour du fichier {filename}...")
    
    if not fights_data:
        print("Aucun combat à ajouter.")
        return False
    
    # Créer un DataFrame avec les nouvelles données
    new_df = pd.DataFrame(fights_data)
    print(f"Nombre de combats récupérés: {len(new_df)}")
    print(f"Colonnes du DataFrame: {new_df.columns.tolist()}")
    
    # Afficher les informations du premier combat pour le débogage
    if not new_df.empty:
        print("Exemple du premier combat:")
        for key, value in new_df.iloc[0].items():
            print(f"  {key}: {value}")
    
    # Si le fichier existe, ajouter les données au début
    if os.path.exists(filename):
        try:
            existing_df = pd.read_csv(filename)
            print(f"Fichier existant chargé. Nombre de combats existants: {len(existing_df)}")
            
            # Vérifier si l'événement existe déjà de manière plus détaillée
            event_name = fights_data[0]['event']
            print(f"Vérification de l'existence de l'événement: {event_name}")
            
            # Vérifier le nom exact de l'événement
            event_exists = (existing_df['event'] == event_name).any()
            
            # Si l'événement n'existe pas déjà, l'ajouter
            if not event_exists:
                print(f"L'événement {event_name} n'existe pas encore dans {filename}, ajout en cours...")
                
                # S'assurer que les nouvelles données sont compatibles avec le fichier existant
                # Obtenir l'ensemble de toutes les colonnes nécessaires
                all_columns = set(existing_df.columns).union(set(new_df.columns))
                
                # Ajouter les colonnes manquantes aux deux DataFrames
                for col in all_columns:
                    if col not in existing_df:
                        existing_df[col] = None
                    if col not in new_df:
                        new_df[col] = None
                
                # Concaténer les DataFrames (nouvelles données au début)
                combined_df = pd.concat([new_df, existing_df], ignore_index=True)
                
                # Sauvegarder le fichier
                combined_df.to_csv(filename, index=False)
                print(f"Ajouté {len(new_df)} combats au début de {filename}")
                return True
            else:
                print(f"L'événement {event_name} existe déjà dans {filename}, aucune mise à jour nécessaire.")
                return False
        except Exception as e:
            print(f"Erreur lors de la mise à jour du fichier: {e}")
            print("Tentative de création d'un nouveau fichier...")
            new_df.to_csv(filename, index=False)
            print(f"Nouveau fichier {filename} créé avec {len(new_df)} combats")
            return True
    else:
        # Créer un nouveau fichier
        new_df.to_csv(filename, index=False)
        print(f"Nouveau fichier {filename} créé avec {len(new_df)} combats")
        return True

def extract_fighter_urls(fight_urls):
    """Extrait les URLs des combattants à partir des URLs de combats"""
    fighter_urls = []
    
    for fight_url in fight_urls:
        try:
            response = requests.get(fight_url, headers=HEADERS)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Rechercher les liens des combattants
                fighter_links = soup.find_all('a', class_='b-link b-fight-details__person-link')
                if not fighter_links:
                    fighter_links = soup.find_all('a', href=lambda href: href and '/fighter-details/' in href)
                
                for link in fighter_links:
                    url = link.get('href')
                    if url and url not in fighter_urls:
                        fighter_urls.append(url)
                        print(f"Lien de combattant trouvé: {url}")
            
            # Pause entre les requêtes
            time.sleep(random.uniform(0.3, 0.7))
        except Exception as e:
            print(f"Erreur lors de l'extraction des URLs de combattants: {e}")
    
    return fighter_urls

def get_fighter_data(fighter_url):
    """Extrait les statistiques d'un combattant"""
    try:
        response = requests.get(fighter_url, headers=HEADERS)
        if response.status_code != 200:
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Nom du combattant
        name_elem = soup.find('span', class_='b-content__title-highlight')
        if not name_elem:
            return None
        
        fighter_name = name_elem.text.strip()
        print(f"Extraction des stats de: {fighter_name}")
        
        # Record
        record_elem = soup.find('span', class_='b-content__title-record')
        if not record_elem:
            return None
        
        record = record_elem.text.replace('Record:', '').strip()
        try:
            record_parts = record.split('-')
            wins = int(record_parts[0])
            losses = int(record_parts[1])
        except:
            wins = 0
            losses = 0
        
        # Statistiques
        stats_elems = soup.find_all('li', class_='b-list__box-list-item b-list__box-list-item_type_block')
        stats_text = [stat.get_text(strip=True) for stat in stats_elems]
        
        # Statistiques de base
        stats_dict = {
            'name': fighter_name,
            'wins': wins,
            'losses': losses,
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
        
        # Parcourir les statistiques
        for stat in stats_text:
            # Hauteur
            if 'Height:' in stat:
                height_match = re.match(r'Height:(\d+)\' (\d+)"', stat)
                if height_match:
                    feet, inches = map(int, height_match.groups())
                    stats_dict['height'] = round((feet * 30.48) + (inches * 2.54), 2)
            
            # Poids
            elif 'Weight:' in stat:
                weight_match = re.match(r'Weight:(\d+) lbs\.', stat)
                if weight_match:
                    stats_dict['weight'] = round(int(weight_match.group(1)) * 0.453592, 2)
            
            # Allonge
            elif 'Reach:' in stat:
                reach_text = stat.replace('Reach:', '').strip()
                if reach_text != '--':
                    try:
                        reach_inches = reach_text.replace('"', '').strip()
                        stats_dict['reach'] = round(float(reach_inches) * 2.54, 2)
                    except:
                        pass
            
            # Stance
            elif 'STANCE:' in stat:
                stance = stat.replace('STANCE:', '').strip()
                stats_dict['stance'] = stance if stance != '--' else 'Unknown'
            
            # Date de naissance
            elif 'DOB:' in stat:
                dob_text = stat.replace('DOB:', '').strip()
                if dob_text != '--':
                    try:
                        dob = datetime.strptime(dob_text, '%b %d, %Y')
                        current_date = datetime.now()
                        stats_dict['age'] = current_date.year - dob.year - ((current_date.month, current_date.day) < (dob.month, dob.day))
                    except:
                        pass
            
            # SLpM
            elif 'SLpM:' in stat:
                slpm_text = stat.replace('SLpM:', '').strip()
                if slpm_text != '--':
                    stats_dict['SLpM'] = float(slpm_text)
            
            # Striking Accuracy
            elif 'Str. Acc.:' in stat:
                acc_text = stat.replace('Str. Acc.:', '').replace('%', '').strip()
                if acc_text != '--':
                    stats_dict['sig_str_acc'] = float(acc_text) / 100
            
            # SApM
            elif 'SApM:' in stat:
                sapm_text = stat.replace('SApM:', '').strip()
                if sapm_text != '--':
                    stats_dict['SApM'] = float(sapm_text)
            
            # Striking Defense
            elif 'Str. Def:' in stat:
                def_text = stat.replace('Str. Def:', '').replace('%', '').strip()
                if def_text != '--':
                    stats_dict['str_def'] = float(def_text) / 100
            
            # Takedown Average
            elif 'TD Avg.:' in stat:
                td_text = stat.replace('TD Avg.:', '').strip()
                if td_text != '--':
                    stats_dict['td_avg'] = float(td_text)
            
            # Takedown Accuracy
            elif 'TD Acc.:' in stat:
                td_acc_text = stat.replace('TD Acc.:', '').replace('%', '').strip()
                if td_acc_text != '--':
                    stats_dict['td_acc'] = float(td_acc_text) / 100
            
            # Takedown Defense
            elif 'TD Def.:' in stat:
                td_def_text = stat.replace('TD Def.:', '').replace('%', '').strip()
                if td_def_text != '--':
                    stats_dict['td_def'] = float(td_def_text) / 100
            
            # Submission Average
            elif 'Sub. Avg.:' in stat:
                sub_text = stat.replace('Sub. Avg.:', '').strip()
                if sub_text != '--':
                    stats_dict['sub_avg'] = float(sub_text)
        
        return stats_dict
    except Exception as e:
        print(f"Erreur lors de l'extraction des données de {fighter_url}: {e}")
        return None

def update_fighters_stats_file(fighter_urls, filename='content/fighters_stats.txt'):
    """Met à jour le fichier fighters_stats.txt avec les nouveaux combattants"""
    print(f"Mise à jour du fichier {filename}...")
    
    # Créer le répertoire si nécessaire
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Lire les combattants existants
    existing_fighters = set()
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            current_fighter = None
            for line in f:
                line = line.strip()
                if line.startswith('name:'):
                    current_fighter = line.split(':', 1)[1].strip()
                elif line == '' and current_fighter:
                    existing_fighters.add(current_fighter)
                    current_fighter = None
    
    print(f"Combattants existants: {len(existing_fighters)}")
    
    # Récupérer les données des combattants
    new_fighters = []
    updated_fighters = []
    
    for fighter_url in fighter_urls:
        # Pause entre chaque requête pour éviter d'être bloqué
        time.sleep(random.uniform(0.3, 0.7))
        
        fighter_data = get_fighter_data(fighter_url)
        if not fighter_data:
            continue
        
        fighter_name = fighter_data['name']
        
        if fighter_name not in existing_fighters:
            new_fighters.append(fighter_data)
            print(f"Nouveau combattant: {fighter_name}")
        else:
            # Lire les stats existantes pour vérifier si mise à jour nécessaire
            old_stats = {}
            with open(filename, 'r', encoding='utf-8') as f:
                reading_fighter = False
                for line in f:
                    line = line.strip()
                    if line.startswith('name:'):
                        if line.split(':', 1)[1].strip() == fighter_name:
                            reading_fighter = True
                            old_stats = {'name': fighter_name}
                        else:
                            reading_fighter = False
                    elif reading_fighter and ':' in line:
                        key, value = line.split(':', 1)
                        old_stats[key.strip()] = value.strip()
                    elif line == '' and reading_fighter:
                        reading_fighter = False
            
            # Vérifier si les victoires/défaites ont changé
            old_wins = int(old_stats.get('wins', 0))
            old_losses = int(old_stats.get('losses', 0))
            
            if fighter_data['wins'] > old_wins or fighter_data['losses'] > old_losses:
                updated_fighters.append(fighter_data)
                print(f"Mise à jour: {fighter_name} ({old_wins}-{old_losses} -> {fighter_data['wins']}-{fighter_data['losses']})")
    
    print(f"Nouveaux combattants: {len(new_fighters)}, Mises à jour: {len(updated_fighters)}")
    
    # Si aucun changement, terminer
    if not new_fighters and not updated_fighters:
        print("Aucune mise à jour nécessaire pour les combattants.")
        return
    
    # Extraire tous les combattants existants
    all_fighters = []
    current_fighter = {}
    
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith('name:'):
                    current_fighter = {'name': line.split(':', 1)[1].strip()}
                    i += 1
                    # Lire toutes les lignes jusqu'à la ligne vide
                    while i < len(lines) and lines[i].strip() != '':
                        line = lines[i].strip()
                        if ':' in line:
                            key, value = line.split(':', 1)
                            current_fighter[key.strip()] = value.strip()
                        i += 1
                    # Ajouter le combattant si son nom n'est pas dans les mises à jour
                    if current_fighter['name'] not in [f['name'] for f in updated_fighters]:
                        all_fighters.append(current_fighter)
                    current_fighter = {}
                else:
                    i += 1
    
    # Mise à jour du fichier
    with open(filename, 'w', encoding='utf-8') as f:
        # Écrire d'abord les nouveaux combattants et les mises à jour
        for fighter in new_fighters + updated_fighters:
            for key, value in fighter.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
        
        # Puis les combattants existants non modifiés
        for fighter in all_fighters:
            for key, value in fighter.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
    
    print(f"Fichier {filename} mis à jour avec succès.")

def main():
    """Fonction principale"""
    print("Démarrage du script de mise à jour UFC...")
    
    # 1. Trouver le dernier événement complété
    event_url, event_name = get_completed_event_url()
    if not event_url:
        print("Impossible de trouver un événement complété. Fin du script.")
        return
    
    # 2. Extraire les données des combats
    fights_data = extract_event_data(event_url)
    if not fights_data:
        print("Aucun combat trouvé pour cet événement. Fin du script.")
        return
    
    # 3. Mettre à jour le fichier ufc_data.csv
    update_success = update_ufc_data_file(fights_data)
    if not update_success:
        print("Mise à jour de ufc_data.csv échouée ou non nécessaire.")
    
    # 4. Récupérer les liens vers les détails des combats
    fight_links = get_fight_links(event_url)
    
    # 5. Extraire les URLs des combattants
    fighter_urls = extract_fighter_urls(fight_links)
    if not fighter_urls:
        print("Aucune URL de combattant trouvée. Fin du script.")
        return
    
    # 6. Mettre à jour le fichier fighters_stats.txt
    update_fighters_stats_file(fighter_urls)
    
    print("Mise à jour terminée avec succès!")

if __name__ == "__main__":
    main()
