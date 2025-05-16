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
    
    with col1:
        total_budget = st.number_input(
            "Budget total (€)",
            min_value=10.0,
            max_value=float(current_bankroll),
            value=min(300.0, float(current_bankroll)),
            step=10.0,
            format="%.2f",
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
    
    # AMÉLIORATION UI: Section pour les cotes améliorée
    st.markdown("""
    <div class="card" style="margin-top: 15px;">
        <h3 style="margin-top: 0;">Entrez les cotes proposées par les bookmakers</h3>
        <p style="color: #aaa; margin-bottom: 15px;">Ces cotes seront utilisées pour calculer la valeur de chaque pari</p>
    </div>
    """, unsafe_allow_html=True)
    
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
                    # AMÉLIORATION UI: Card de combat modernisée
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
                            <span> ({fight['probability']:.0%})</span>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Champ pour entrer la cote du bookmaker
                    odds_key = f"odds_{fight_key}"
                    
                    # Initialiser la valeur si première utilisation
                    if odds_key not in st.session_state[f"odds_dict_{event_url}"]:
                        st.session_state[f"odds_dict_{event_url}"][odds_key] = 2.0
                    
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
                recommendation_data.append({
                    "Combat": f"{fight['red_fighter']} vs {fight['blue_fighter']}",
                    "Pari sur": fight['winner_name'],
                    "Probabilité": f"{fight['probability']:.0%}",
                    "Cote": f"{fight['odds']:.2f}",
                    "Value": f"{fight['edge']*100:.1f}%",
                    "Rendement": f"{fight['value']:.2f}",  
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
                    "Rendement": st.column_config.TextColumn("Rendement"),
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
                st.metric("Budget total", f"{total_budget:.2f}€")
            with summary_cols[1]:
                st.metric("Montant misé", f"{total_stake:.2f}€", f"{total_stake/total_budget*100:.1f}%")
            with summary_cols[2]:
                st.metric("Gain potentiel", f"{total_potential_profit:.2f}€", f"{total_potential_profit/total_stake*100:.1f}%")
            
            # Résumé détaillé
            st.markdown(f"""
            <div class="strategy-summary">
                <h4 style="margin-top: 0;">Résumé de la stratégie</h4>
                <ul>
                    <li>Stratégie Kelly utilisée: <b>{kelly_strategy}</b></li>
                    <li>Nombre de paris recommandés: <b>{len(filtered_fights)}</b></li>
                    <li>Utilisation du budget: <b>{total_stake/total_budget*100:.1f}%</b></li>
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
            st.markdown("""
            <div class="card" style="background: linear-gradient(145deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 160, 0, 0.1) 100%);
                                     border: 1px solid rgba(255, 193, 7, 0.3); text-align: center; padding: 20px;">
                <div style="font-size: 2rem; margin-bottom: 10px;">⚠️</div>
                <h3 style="color: #FFC107; margin-bottom: 10px;">Aucun combat intéressant</h3>
                <p style="margin-bottom: 0;">Aucun combat ne correspond aux critères de value betting (confiance ≥ 65% et value positive).</p>
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
                        # Valeur par défaut si non définie (ne devrait pas arriver)
                        fight['odds'] = 2.0
                        st.session_state[f"odds_dict_{event_url}"][odds_key] = 2.0
                        
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
                st.markdown("""
                <div class="card" style="background: linear-gradient(145deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 160, 0, 0.1) 100%);
                                         border: 1px solid rgba(255, 193, 7, 0.3); text-align: center; padding: 20px;">
                    <div style="font-size: 2rem; margin-bottom: 10px;">⚠️</div>
                    <h3 style="color: #FFC107; margin-bottom: 10px;">Aucun combat intéressant</h3>
                    <p style="margin-bottom: 0;">Aucun combat ne correspond aux critères de value betting (confiance ≥ 65% et value positive).</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Afficher des informations de débogage sur les combats évalués
                debug_info = st.expander("🔍 Détails des combats évalués", expanded=False)
                with debug_info:
                    for fight in bettable_fights:
                        odds_key = f"odds_{fight['fight_key']}"
                        odds_value = st.session_state[f"odds_dict_{event_url}"].get(odds_key, "Non définie")
                        implicit_prob = 1 / float(odds_value) if isinstance(odds_value, (int, float)) and odds_value > 0 else "N/A"
                        edge = fight['probability'] - implicit_prob if isinstance(implicit_prob, float) else "N/A"
                        
                        st.markdown(f"""
                        <div style="margin-bottom: 10px; padding: 10px; background: rgba(255,255,255,0.05); border-radius: 8px;">
                            <div><b>{fight['red_fighter']} vs {fight['blue_fighter']}</b></div>
                            <div>Vainqueur prédit: <b>{fight['winner_name']}</b> ({fight['probability']:.0%})</div>
                            <div>Cote: <b>{odds_value}</b> (Probabilité implicite: {implicit_prob if isinstance(implicit_prob, float) else implicit_prob})</div>
                            <div>Edge: <b>{edge if isinstance(edge, float) else edge}</b></div>
                            <div>Raison: {
                                "Confiance < 65%" if fight['probability'] < 0.65 
                                else "Cote trop basse (probabilité implicite trop élevée)" if isinstance(implicit_prob, float) and implicit_prob >= fight['probability']
                                else "Value insuffisante" if isinstance(implicit_prob, float) and fight['probability'] * float(odds_value) < 1.15
                                else "Raison inconnue"
                            }</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                # Calculer la somme totale des fractions Kelly
                total_kelly = sum(fight['fractional_kelly'] for fight in filtered_fights)
                
                # CORRECTION: Ajouter une vérification pour éviter la division par zéro
                if total_kelly <= 0:
                    st.warning("Impossible de calculer les mises : la somme des fractions Kelly est nulle ou négative.")
                    for fight in filtered_fights:
                        fight['stake'] = 0
                else:
                    # Calculer les montants à miser
                    for fight in filtered_fights:
                        # Répartir le budget proportionnellement
                        fight['stake'] = total_budget * (fight['fractional_kelly'] / total_kelly)
                        
                        # CORRECTION: Arrondir les mises pour plus de clarté
                        fight['stake'] = round(fight['stake'], 2)
                
                # AMÉLIORATION UI: Afficher les recommandations dans un tableau moderne
                recommendation_data = []
                for fight in filtered_fights:
                    recommendation_data.append({
                        "Combat": f"{fight['red_fighter']} vs {fight['blue_fighter']}",
                        "Pari sur": fight['winner_name'],
                        "Probabilité": f"{fight['probability']:.0%}",
                        "Cote": f"{fight['odds']:.2f}",
                        "Value": f"{fight['edge']*100:.1f}%",
                        "Rendement": f"{fight['value']:.2f}",  
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
                            "Rendement": st.column_config.TextColumn("Rendement"),
                            "Montant": st.column_config.TextColumn("Montant"),
                            "Gain potentiel": st.column_config.TextColumn("Gain potentiel")
                        },
                        hide_index=True
                    )
                    
                    # AMÉLIORATION UI: Résumé de la stratégie avec des métriques
                    total_stake = sum(fight['stake'] for fight in filtered_fights)
                    total_potential_profit = sum(fight['stake'] * (fight['odds']-1) for fight in filtered_fights)
                    
                    # Afficher les métriques en 3 colonnes
                    summary_cols = st.columns(3)
                    with summary_cols[0]:
                        st.metric("Budget total", f"{total_budget:.2f}€")
                    with summary_cols[1]:
                        st.metric("Montant misé", f"{total_stake:.2f}€", f"{total_stake/total_budget*100:.1f}%")
                    with summary_cols[2]:
                        st.metric("Gain potentiel", f"{total_potential_profit:.2f}€", f"{total_potential_profit/total_stake*100:.1f}%")
                    
                    # Résumé détaillé
                    st.markdown(f"""
                    <div class="strategy-summary">
                        <h4 style="margin-top: 0;">Résumé de la stratégie</h4>
                        <ul>
                            <li>Stratégie Kelly utilisée: <b>{kelly_strategy}</b></li>
                            <li>Nombre de paris recommandés: <b>{len(filtered_fights)}</b></li>
                            <li>Utilisation du budget: <b>{total_stake/total_budget*100:.1f}%</b></li>
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
    
    # Fonction de débogage
    def debug_betting_strategy(event_url, bettable_fights, filtered_fights):
        """Fonction de débogage pour la stratégie de paris"""
        
        debug_info = st.expander("📝 Informations de débogage (développeur)", expanded=False)
        
        with debug_info:
            st.write("### État du dictionnaire des cotes")
            st.write(st.session_state.get(f"odds_dict_{event_url}", {}))
            
            st.write("### Combats bettables")
            for fight in bettable_fights:
                fight_key = fight['fight_key']
                odds_key = f"odds_{fight_key}"
                odds_value = st.session_state.get(f"odds_dict_{event_url}", {}).get(odds_key, "Non définie")
                
                st.write(f"- {fight['red_fighter']} vs {fight['blue_fighter']}: Prob={fight['probability']:.2f}, Cote={odds_value}")
            
            st.write("### Combats filtrés pour paris")
            for fight in filtered_fights:
                st.write(f"- {fight['red_fighter']} vs {fight['blue_fighter']}: Prob={fight['probability']:.2f}, Cote={fight['odds']}, Kelly={fight.get('fractional_kelly', 0):.4f}, Mise={fight.get('stake', 0):.2f}€")
    
    # Ajouter le débogage pour les développeurs
    if st.checkbox("Afficher le débogage (développeur)", value=False, key=f"debug_{event_url}"):
        debug_betting_strategy(event_url, bettable_fights, filtered_fights)
