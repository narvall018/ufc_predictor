#!/usr/bin/env python
# coding: utf-8

import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import re
from datetime import datetime
import math

def get_latest_event_direct():
    """
    Récupère directement l'URL du dernier événement UFC
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    
    url = "http://ufcstats.com/statistics/events/completed"
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        # Rechercher directement tous les liens d'événements
        event_links = soup.find_all('a', href=lambda href: href and 'event-details' in href)
        
        if event_links:
            latest_event_url = event_links[0].get('href')
            print(f"URL du dernier événement: {latest_event_url}")
            return latest_event_url
    
    return None

def update_fighters_stats_from_event(event_url):
    """
    Fonction simplifiée qui récupère et met à jour les statistiques des combattants
    """
    if not event_url:
        print("URL d'événement non fournie.")
        return False
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    # 1. Récupérer les liens de combat de l'événement
    print(f"Récupération des combats de l'événement: {event_url}")
    response = requests.get(event_url, headers=headers)
    
    if response.status_code != 200:
        print(f"Échec de l'accès à l'événement. Code: {response.status_code}")
        return False
    
    soup = BeautifulSoup(response.text, 'html.parser')
    fight_links = soup.find_all('a', href=lambda href: href and 'fight-details' in href)
    
    fight_urls = [link.get('href') for link in fight_links]
    print(f"{len(fight_urls)} URLs de combat trouvées")
    
    # 2. Extraire les liens des combattants de chaque combat
    fighter_urls = []
    for fight_url in fight_urls:
        fight_response = requests.get(fight_url, headers=headers)
        if fight_response.status_code == 200:
            fight_soup = BeautifulSoup(fight_response.text, 'html.parser')
            fighter_links = fight_soup.find_all('a', class_='b-link b-fight-details__person-link')
            
            for link in fighter_links:
                fighter_url = link.get('href')
                if fighter_url and fighter_url not in fighter_urls:
                    fighter_urls.append(fighter_url)
    
    print(f"{len(fighter_urls)} URLs de combattants extraites")
    
    # 3. Récupérer les données des combattants
    fighter_stats = []
    
    for fighter_url in fighter_urls:
        try:
            fighter_response = requests.get(fighter_url, headers=headers)
            if fighter_response.status_code == 200:
                fighter_soup = BeautifulSoup(fighter_response.text, 'html.parser')
                
                # Nom du combattant
                name_elem = fighter_soup.find('span', class_='b-content__title-highlight')
                if not name_elem:
                    continue
                
                fighter_name = name_elem.text.strip()
                print(f"Traitement de {fighter_name}")
                
                # Record
                record_elem = fighter_soup.find('span', class_='b-content__title-record')
                if not record_elem:
                    continue
                
                record = record_elem.text.replace('Record:', '').strip()
                try:
                    record_parts = record.split('-')
                    wins = int(record_parts[0])
                    losses = int(record_parts[1])
                except:
                    wins = 0
                    losses = 0
                
                # Autres statistiques
                stats_elems = fighter_soup.find_all('li', class_='b-list__box-list-item b-list__box-list-item_type_block')
                stats_text = [stat.get_text(strip=True) for stat in stats_elems]
                
                # Hauteur
                height = None
                if len(stats_text) > 0:
                    height_text = stats_text[0]
                    if height_text != '--' and 'Height:' in height_text:
                        height_match = re.match(r'Height:(\d+)\' (\d+)"', height_text)
                        if height_match:
                            feet, inches = map(int, height_match.groups())
                            height = round((feet * 30.48) + (inches * 2.54), 2)
                
                # Poids
                weight = None
                if len(stats_text) > 1:
                    weight_text = stats_text[1]
                    if weight_text != '--' and 'Weight:' in weight_text:
                        weight_match = re.match(r'Weight:(\d+) lbs\.', weight_text)
                        if weight_match:
                            weight = round(int(weight_match.group(1)) * 0.453592, 2)
                
                # Allonge
                reach = None
                if len(stats_text) > 2:
                    reach_text = stats_text[2].replace('Reach:', '').strip()
                    if reach_text != '--':
                        try:
                            reach_inches = reach_text.replace('"', '').strip()
                            reach = round(float(reach_inches) * 2.54, 2)
                        except:
                            pass
                
                # Stance
                stance = 'Unknown'
                if len(stats_text) > 3:
                    stance = stats_text[3].replace('STANCE:', '').strip()
                    if stance == '--':
                        stance = 'Unknown'
                
                # Âge
                age = None
                if len(stats_text) > 4:
                    dob_text = stats_text[4].replace('DOB:', '').strip()
                    if dob_text != '--':
                        try:
                            dob = datetime.strptime(dob_text, '%b %d, %Y')
                            current_date = datetime.now()
                            age = current_date.year - dob.year - ((current_date.month, current_date.day) < (dob.month, dob.day))
                        except:
                            pass
                
                # Statistiques de combat
                stats_dict = {
                    'name': fighter_name,
                    'wins': wins,
                    'losses': losses,
                    'height': height,
                    'weight': weight,
                    'reach': reach,
                    'stance': stance,
                    'age': age
                }
                
                # Extraire les statistiques de combat si disponibles
                if len(stats_text) > 5:
                    stats_dict['SLpM'] = float(stats_text[5].replace('SLpM:', '').strip()) if stats_text[5].replace('SLpM:', '').strip() != '--' else 0.0
                if len(stats_text) > 6:
                    stats_dict['sig_str_acc'] = float(stats_text[6].replace('Str. Acc.:', '').rstrip('%')) / 100 if stats_text[6].replace('Str. Acc.:', '').rstrip('%') != '--' else 0.0
                if len(stats_text) > 7:
                    stats_dict['SApM'] = float(stats_text[7].replace('SApM:', '').strip()) if stats_text[7].replace('SApM:', '').strip() != '--' else 0.0
                if len(stats_text) > 8:
                    stats_dict['str_def'] = float(stats_text[8].replace('Str. Def:', '').rstrip('%')) / 100 if stats_text[8].replace('Str. Def:', '').rstrip('%') != '--' else 0.0
                if len(stats_text) > 10:
                    stats_dict['td_avg'] = float(stats_text[10].replace('TD Avg.:', '').strip()) if stats_text[10].replace('TD Avg.:', '').strip() != '--' else 0.0
                if len(stats_text) > 11:
                    stats_dict['td_acc'] = float(stats_text[11].replace('TD Acc.:', '').rstrip('%')) / 100 if stats_text[11].replace('TD Acc.:', '').rstrip('%') != '--' else 0.0
                if len(stats_text) > 12:
                    stats_dict['td_def'] = float(stats_text[12].replace('TD Def.:', '').rstrip('%')) / 100 if stats_text[12].replace('TD Def.:', '').rstrip('%') != '--' else 0.0
                if len(stats_text) > 13:
                    stats_dict['sub_avg'] = float(stats_text[13].replace('Sub. Avg.:', '').strip()) if stats_text[13].replace('Sub. Avg.:', '').strip() != '--' else 0.0
                
                fighter_stats.append(stats_dict)
                
        except Exception as e:
            print(f"Erreur lors du traitement du combattant {fighter_url}: {e}")
    
    # 4. Mettre à jour le fichier fighters_stats.txt
    os.makedirs('content', exist_ok=True)
    
    # Lire les combattants existants pour éviter les doublons
    existing_fighters = set()
    if os.path.exists('content/fighters_stats.txt'):
        with open('content/fighters_stats.txt', 'r', encoding='utf-8') as f:
            current_fighter = None
            for line in f:
                line = line.strip()
                if line.startswith('name:'):
                    current_fighter = line.split(':', 1)[1].strip()
                elif line == '' and current_fighter:
                    existing_fighters.add(current_fighter)
                    current_fighter = None
    
    # Ajouter les nouveaux combattants
    with open('content/fighters_stats.txt', 'a', encoding='utf-8') as f:
        for fighter in fighter_stats:
            if fighter['name'] not in existing_fighters:
                for key, value in fighter.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
                print(f"Ajouté {fighter['name']} au fichier")
    
    # 5. Récupérer les détails de l'événement pour ufc_data.csv
    response = requests.get(event_url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        
        event_name = soup.find('span', class_='b-content__title-highlight')
        event_name = event_name.text.strip() if event_name else "Unknown Event"
        
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
        for fight_row in soup.find_all('tr', class_='b-fight-details__table-row'):
            cells = fight_row.find_all('p', class_='b-fight-details__table-text')
            
            if len(cells) >= 16:
                fight_data = {
                    'event': event_name,
                    'date': date,
                    'location': location,
                    'r_fighter': cells[1].text.strip(),
                    'b_fighter': cells[2].text.strip(),
                    'weight_class': cells[11].text.strip(),
                    'method': cells[12].text.strip(),
                    'round': cells[14].text.strip(),
                    'time': cells[15].text.strip()
                }
                fights_data.append(fight_data)
        
        # Mettre à jour ufc_data.csv
        new_df = pd.DataFrame(fights_data)
        
        if os.path.exists('ufc_data.csv'):
            existing_df = pd.read_csv('ufc_data.csv')
            # Vérifier si l'événement existe déjà
            if event_name in existing_df['event'].values:
                print(f"L'événement {event_name} existe déjà dans ufc_data.csv")
            else:
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df.to_csv('ufc_data.csv', index=False)
                print(f"ufc_data.csv mis à jour avec {len(new_df)} combats")
        else:
            new_df.to_csv('ufc_data.csv', index=False)
            print(f"ufc_data.csv créé avec {len(new_df)} combats")
    
    print("Mise à jour terminée.")
    return True

# Exécution du script
if __name__ == "__main__":
    latest_event_url = get_latest_event_direct()
    if latest_event_url:
        update_fighters_stats_from_event(latest_event_url)
    else:
        print("Impossible de trouver l'URL du dernier événement.")
