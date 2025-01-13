import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from fuzzywuzzy import fuzz, process
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from st_on_hover_tabs import on_hover_tabs

st.set_page_config(page_title="L'Antre des Cinéphiles" ,layout="wide")

css = '''
<style>
    html, body, [class*="css"] {
        color: #cccccc !important;
    }
    
    h1, h2, h3, h4, h5, h6, p, li, span, div {
        color: #cccccc !important;
    }

    section[data-testid='stSidebar'] {
        background-color: #111;
        min-width: unset !important;
        width: unset !important;
        flex-shrink: unset !important;
    }

    button[kind="header"] {
        background-color: transparent;
        color: rgb(180, 167, 141);
    }

    @media(hover) {
        header[data-testid="stHeader"] {
            display: none;
        }

        section[data-testid='stSidebar'] > div {
            height: 100%;
            width: 95px;
            position: relative;
            z-index: 1;
            top: 0;
            left: 0;
            background-color: #111;
            overflow-x: hidden;
            transition: 0.5s ease;
            padding-top: 60px;
            white-space: nowrap;
        }

        section[data-testid='stSidebar'] > div:hover {
            width: 450px;
        }

        button[kind="header"] {
            display: none;
        }
    }

    @media(max-width: 272px) {
        section[data-testid='stSidebar'] > div {
            width: 15rem;
        }
    }

</style>
'''
st.markdown(css, unsafe_allow_html=True)

# Configuration initiale
SIMILARITY_THRESHOLD = 80  # Seuil de similarité pour déterminer si un titre correspond
TEXT_WEIGHT = 0.95  # Poids donné à la similarité textuelle par rapport aux attributs numériques
NUM_RECOMMENDATIONS = 5  # Nombre de recommandations à afficher

# Téléchargement des ressources NLTK pour la tokenisation et le traitement linguistique
@st.cache_resource
def download_nltk_resources():
    nltk.download(['stopwords', 'wordnet', 'punkt'])  # Téléchargement des stopwords, lemmatizer et tokenizer

# Chargement des données depuis une URL
@st.cache_data
def load_data(url):
    return pd.read_csv(url)  # Charge les données au format CSV et les met en cache

# Fonction pour nettoyer le texte
def clean_text(text, lemmatizer, stop_words):
    if not isinstance(text, str):
        return ""  # Retourne une chaîne vide si le texte n'est pas une chaîne valide
    text = re.sub(r'[^a-z\\s]', ' ', text.lower())  # Convertit en minuscules et supprime les caractères spéciaux
    tokens = nltk.word_tokenize(text)  # Tokenise le texte
    return ' '.join(lemmatizer.lemmatize(w) for w in tokens if w not in stop_words)  # Lemmatisation et suppression des stopwords

# Préparation du dataframe en ajoutant des colonnes traitées
def prepare_dataframe(df):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    df = df.copy()
    list_columns = ['genres', 'Actors', 'Directors', 'Writers', 'production_countries']
    
    # Transforme les colonnes de type liste en listes Python
    for col in list_columns:
        df[col] = df[col].apply(lambda x: [item.strip() for item in str(x).split(',')] if pd.notna(x) else [])

    df['year'] = pd.to_numeric(df['year'], errors='coerce')  # Convertit les années en format numérique
    df['keywords'] = df['overview'].apply(lambda x: clean_text(x, lemmatizer, stop_words))  # Ajoute une colonne 'keywords' nettoyée
    
    return df

# Création de la "soupe de fonctionnalités" pour chaque film
def create_feature_soup(row):
    elements = []
    elements.extend(row['genres'] * 2)  # Double poids pour les genres
    elements.extend(row['Actors'])  # Poids normal pour les acteurs
    elements.extend(row['Directors'] * 2)  # Double poids pour les réalisateurs
    elements.extend(row['production_countries'])  # Poids normal pour les pays de production
    elements.append(str(row['keywords']))  # Ajout des mots-clés nettoyés
    return ' '.join(map(str, elements))  # Combine tous les éléments en une seule chaîne

def create_feature_soup_shorts(row):
    elements = []
    elements.extend(row['genres'] * 2)  # Double poids pour les genres
    elements.extend(row['primaryName'] * 2)  # Double poids pour les réalisateurs
    elements.extend(row['production_countries'])  # Poids normal pour les pays de production
    elements.append(str(row['keywords']))  # Ajout des mots-clés nettoyés
    return ' '.join(map(str, elements))  # Combine tous les éléments en une seule chaîne

# Calcul de la matrice de similarité en combinant similarité textuelle et numérique
def compute_similarity_matrix(df, text_weight):
    df['text_features'] = df.apply(create_feature_soup, axis=1)  # Création des colonnes textuelles combinées
    vectorizer = CountVectorizer(stop_words='english')  # Vectorisation avec suppression des stopwords
    text_matrix = vectorizer.fit_transform(df['text_features'])  # Création d'une matrice textuelle

    scaler = MinMaxScaler()  # Mise à l'échelle des attributs numériques
    numeric_features = scaler.fit_transform(df[['year']].copy())  # Normalisation de l'année

    # Combinaison pondérée des similarités textuelle et numérique
    text_similarity = cosine_similarity(text_matrix)
    numeric_similarity = cosine_similarity(numeric_features)

    similarity_matrix = (text_weight * text_similarity + 
                         (1 - text_weight) * numeric_similarity)
    return similarity_matrix

def shorts_reco(df, df_shorts, movie_title):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    df = df.copy()
    df_shorts = df_shorts.copy()
    list_columns_df = ['genres', 'Directors', 'production_countries']
    list_columns_shorts = ['genres', 'primaryName', 'production_countries']
    # Transforme les colonnes de type liste en listes Python
    for col in list_columns_df:
        df[col] = df[col].apply(lambda x: [item.strip() for item in str(x).split(',')] if pd.notna(x) else [])
    for col in list_columns_shorts:
        # Si la colonne contient des chaînes qui ressemblent à des listes
        df_shorts[col] = df_shorts[col].apply(lambda x: 
            # Supprimer les crochets et diviser par virgule
            [item.strip(" '[]\"") for item in str(x).strip("[]").split(",")]
            if pd.notna(x) else [])
    df['keywords'] = df['overview'].apply(lambda x: clean_text(x, lemmatizer, stop_words))
    df_shorts['keywords'] = df_shorts['overview'].apply(lambda x: clean_text(x, lemmatizer, stop_words))

    df['text_features'] = df.apply(create_feature_soup, axis=1)  # Création des colonnes textuelles combinées
    df_shorts['text_features'] = df_shorts.apply(create_feature_soup_shorts, axis=1)  # Création des colonnes textuelles combinées

    vectorizer = CountVectorizer(stop_words='english')
    film_vectors = vectorizer.fit_transform(df['text_features'])
    short_vectors = vectorizer.transform(df_shorts['text_features'])
    similarity_matrix = cosine_similarity(film_vectors, short_vectors)
    closest_title, match_score = find_movie(movie_title, df['localised_title'].tolist(), SIMILARITY_THRESHOLD)

    if not closest_title:
        return None

    idx = df[df['localised_title'] == closest_title].index[0]
    sim_scores = sorted(list(enumerate(similarity_matrix[idx])), 
                       key=lambda x: x[1], reverse=True)[:3]
    
    # Récupérer les indices des courts métrages recommandés
    short_indices = [i[0] for i in sim_scores]
    
    # Adapter les noms de colonnes à ceux présents dans df_shorts
    columns_to_select = []
    for col in ['akas_title', 'startYear', 'genres', 'overview_fr', 'primaryName', 'poster_path', 'runtimeMinutes']:
        if col in df_shorts.columns:
            columns_to_select.append(col)
    
    recommendations = df_shorts.iloc[short_indices][columns_to_select].copy()
    recommendations['similarity_score'] = [i[1] for i in sim_scores]

    return (closest_title, match_score, recommendations)  # Retourne explicitement un tuple de 3 éléments
    
def find_movie(query, titles, threshold):
    def calculate_score(query, title):
        # Donne plus d'importance aux mots qui commencent pareil
        query_words = query.lower().split()             # découpe la recherche et le titre en mots séparés en minuscule
        title_words = title.lower().split()
        
        # Score pour les mots qui commencent pareil
        start_match = any(tw.startswith(qw) or qw.startswith(tw)            # any renvoie True si l'une des conditions est vraie pour au moins une paire de mots
                         for qw in query_words              # pour chaque mot de la recherche
                         for tw in title_words)             # on vérifie si l'un commence par l'autre et vice versa
        
        # Scores de base
        token_score = fuzz.token_set_ratio(query, title)        # compare la présence de mots dans les 2 chaines en ignorant leur ordre et les doublons
        position_score = 100 if start_match else 0              # bonus de 100 points si l'un des mots commence par un autre
        
        return (token_score + position_score) / 2               # score moyen

    best_match = None
    highest_score = 0
    
    for title in titles:
        score = calculate_score(query, title)
        if score > highest_score:
            highest_score = score
            best_match = title
    
    return (best_match, highest_score) if highest_score >= threshold else (None, 0)

# Obtenir des recommandations en fonction d'un film donné
def get_recommendations(df, similarity_matrix, movie_title, n):
    closest_title, match_score = find_movie(movie_title, df['localised_title'].tolist(), SIMILARITY_THRESHOLD)

    if not closest_title:
        return None  # Aucun titre correspondant trouvé

    idx = df[df['localised_title'] == closest_title].index[0]  # Index du film trouvé
    sim_scores = sorted(list(enumerate(similarity_matrix[idx])),                    # Récupère la ligne de la matrice de similarité qui correspond au film trouvé et associe chaque score à son index.    
                        key=lambda x: x[1], reverse=True)[1:n+1]  # Trie les films par score de similarité (de plus élevé à plus faible). x[1] fait référence au score de similarité.
                                # [1:n+1] : Sélectionne les n meilleurs films après le film d’origine (on commence à l'indice 1 pour exclure le film lui-même, qui a un score de similarité maximal avec lui-même).
    movie_indices = [i[0] for i in sim_scores]  # Récupération des indices des films recommandés
    recommendations = df.iloc[movie_indices][
        ['localised_title', 'year', 'genres', 'Directors', 'Actors', 'overview_fr', 'poster_path']
    ].copy()
    recommendations['similarity_score'] = [i[1] for i in sim_scores]  # Ajout des scores de similarité

    return closest_title, match_score, recommendations

def get_unique_sorted_values(df, column):
    """Extrait les valeurs uniques d'une colonne contenant des listes"""
    values = set()
    for items in df[column]:
        if isinstance(items, list):
            values.update(items)
    return sorted(list(values))

def filter_movies(df, selected_genres=None, selected_countries=None, 
                 selected_directors=None, selected_actors=None):
    """Filtre les films selon les critères sélectionnés"""
    filtered_df = df.copy()
    
    if selected_genres:
        filtered_df = filtered_df[filtered_df['genres'].apply(
            lambda x: any(genre in x for genre in selected_genres))]
    
    if selected_countries:
        filtered_df = filtered_df[filtered_df['production_countries'].apply(
            lambda x: any(country in x for country in selected_countries))]
    
    if selected_directors:
        filtered_df = filtered_df[filtered_df['Directors'].apply(
            lambda x: any(director in x for director in selected_directors))]
    
    if selected_actors:
        filtered_df = filtered_df[filtered_df['Actors'].apply(
            lambda x: any(actor in x for actor in selected_actors))]
    
    return filtered_df

# Fonction pour afficher 3 affiches dans la page d'accueil
def afficher_6_affiches(df):
    # Sélectionne aléatoirement 6 films
    random_movies = df[['localised_title', 'poster_path']].sample(n=6)
    rows = [random_movies.iloc[i:i+3] for i in range(0, len(random_movies), 3)]  # Divise en groupes de 3 films
    
    for row in rows:  # Parcourt chaque groupe (rangée)
        cols = st.columns(3)  # Crée 3 colonnes pour chaque rangée
        for i, (_, movie) in enumerate(row.iterrows()):
            poster_url = f"https://image.tmdb.org/t/p/w200{movie['poster_path']}" if pd.notna(movie['poster_path']) else None
            with cols[i]:  # Place les éléments dans les colonnes correspondantes
                if poster_url:
                    st.image(poster_url, use_container_width=True)
                else:
                    st.write("Aucune image disponible")

# Fonction pour afficher un film aléatoire avec détails
def afficher_film_aleatoire(df):
    random_movie = df[['localised_title', 'year', 'averageRating', 'overview_fr', 'poster_path']].sample(n=1)
    
    col1, col2 = st.columns([1, 2])  # 1 pour l'image, 2 pour les informations
    with col1:
        if pd.notna(random_movie['poster_path'].values[0]):
            poster_url = f"https://image.tmdb.org/t/p/w200{random_movie['poster_path'].values[0]}"
            st.image(poster_url, caption=random_movie['localised_title'].values[0], use_container_width=True)
        else:
            st.write("Aucune image disponible.")
    
    with col2:
        st.markdown(f"**Film :** {random_movie['localised_title'].values[0]}")
        st.markdown(f"**Année :** {random_movie['year'].values[0]}")
        st.markdown(f"**Note moyenne :** {random_movie['averageRating'].values[0]}")
        
        if not random_movie['overview_fr'].isnull().values[0]:
            st.markdown("**Résumé :**")
            st.markdown(f"*{random_movie['overview_fr'].values[0]}*")
        else:
            st.markdown("**Résumé :** Aucune information disponible.")
    
    return random_movie


def main():
    def afficher_contenu(selection):
        if selection == "Accueil":
            st.markdown("<h2 style='font-size: 30px; text-align: center;'>🎥 Bienvenue dans l'Antre des Cinéphiles 🎥</h2>", unsafe_allow_html=True)            
            st.markdown("<h3 style='font-size: 20px; text-align: center;'>Explorez les classiques du cinéma des années 60 et 70 à travers notre interface interactive.</h3>", unsafe_allow_html=True)            
            df = load_data('https://raw.githubusercontent.com/SeaJayEm/projet_2/refs/heads/main/df_list.csv')
            afficher_6_affiches(df)

            
        elif selection == "La base de données":
            st.markdown("<h2 style='font-size: 24px; text-align: center;'>Statistiques de la Base de Données 📊</h2>", unsafe_allow_html=True)            
            # Chargement des données
            df = load_data('https://raw.githubusercontent.com/SeaJayEm/projet_2/refs/heads/main/df_list.csv')
            director_counts = []
            for directors in df['Directors']:
                if isinstance(directors, str):
                    director_list = [d.strip() for d in directors.split(',')]
                    director_counts.extend(director_list)

            # Statistiques globales
            st.markdown("<h3 style='font-size: 20px;'>Statistiques globales</h1>", unsafe_allow_html=True)            

            col3, col4, col5 = st.columns(3)
            
            with col3:
                total_movies = len(df)
                st.metric("Nombre total de films", total_movies)
            
            with col4:
                total_directors = len(set(director_counts))
                st.metric("Nombre total de réalisateurs", total_directors)
            
            with col5:
                year_range = f"{df['year'].min()} - {df['year'].max()}"
                st.metric("Période couverte", year_range)

            st.write("")
            st.write("")

            col1, col_space, col2 = st.columns([4, 1, 4])
            
            with col1:
                # Distribution des genres
                st.markdown("<h3 style='font-size: 18px;'>Distribution des genres</h3>", unsafe_allow_html=True)
                genre_counts = []
                for genres in df['genres']:
                    if isinstance(genres, str):
                        genre_list = [g.strip() for g in genres.split(',')]
                        genre_counts.extend(genre_list)
                
                genre_data = pd.Series(genre_counts).value_counts().head(10)
                st.bar_chart(genre_data)
                
                st.write("")
                st.write("")
                
                # Top 10 des réalisateurs
                st.markdown("<h3 style='font-size: 18px;'>Top 10 des réalisateurs</h3>", unsafe_allow_html=True)                        
                director_data = pd.Series(director_counts).value_counts().head(10)
                st.bar_chart(director_data)

            with col_space:
                st.write("")
            
            with col2:
                # Distribution des pays de production
                st.markdown("<h3 style='font-size: 18px;'>Top 10 des pays de production</h3>", unsafe_allow_html=True)
                country_counts = []
                for countries in df['production_countries']:
                    if isinstance(countries, str):
                        country_list = [c.strip() for c in countries.split(',')]
                        country_counts.extend(country_list)
                
                country_data = pd.Series(country_counts).value_counts().head(10)
                st.bar_chart(country_data)
                
                st.write("")
                st.write("")

                # Distribution par année
                st.markdown("<h3 style='font-size: 18px;'>Nombre de films par année</h3>", unsafe_allow_html=True)
                year_data = df['year'].value_counts().sort_index()
                st.line_chart(year_data)
                           
        elif selection == "Recherche de films":
            st.markdown("<h2 style='font-size: 24px; text-align: center;'>Moteur de recommandations de films 🎥</h2>", unsafe_allow_html=True)             # Chargement et préparation des données
            
            # Chargement et préparation des données
            download_nltk_resources()
            df = load_data('https://raw.githubusercontent.com/SeaJayEm/projet_2/refs/heads/main/df_list.csv')
            df_prepared = prepare_dataframe(df)
            
            # Création des listes de valeurs uniques
            genres = get_unique_sorted_values(df_prepared, 'genres')
            countries = get_unique_sorted_values(df_prepared, 'production_countries')
            directors = get_unique_sorted_values(df_prepared, 'Directors')
            actors = get_unique_sorted_values(df_prepared, 'Actors')
            
            # Interface utilisateur avec deux modes
            st.markdown(
                """
                <style>
                    .radio-label {
                        margin-bottom: -60px; 
                    }
                </style>
                <h3 class="radio-label" style="font-size: 20px;">Choisissez votre mode de recherche :</h3>
                """,
                unsafe_allow_html=True,
            )            
            mode = st.radio(
                "",
                ["Recherche par titre", "Recherche par filtres", "Surprends moi !"]
            )
            
            if mode == "Recherche par titre":
                movie_title = st.text_input("Nom du film", placeholder="Entrez le titre d'un film")
                
                if st.button("Recommander") or movie_title:
                    if movie_title:
                        similarity_matrix = compute_similarity_matrix(df_prepared, TEXT_WEIGHT)
                        result = get_recommendations(df_prepared, similarity_matrix, movie_title, NUM_RECOMMENDATIONS)
                        
                        if result is None:
                            st.error(f"Aucun film trouvé correspondant à '{movie_title}'")  # Message d'erreur si aucun film
                        else:
                            closest_title, match_score, recommendations = result

                            if closest_title != movie_title:
                                st.info(f"Film trouvé : '{closest_title}' (score de similarité : {match_score:.1f}%)")  # Correspondance approximative
                                st.write(f"Si vous avez aimé **{closest_title}**, vous pourriez aimer :")
                                for _, row in recommendations.iterrows():
                                    genres = ', '.join(row['genres'])
                                    directors = ', '.join(row['Directors'])
                                    actors = ', '.join(row['Actors'])
                                    poster_url = f"https://image.tmdb.org/t/p/w200{row['poster_path']}"  # URL complète du poster

                                    # Création de deux colonnes
                                    col1, col2 = st.columns([1, 2])  # Largeur personnalisée : 1 pour le poster, 2 pour les infos
                                    with col1:
                                            if pd.notna(row['poster_path']):
                                                st.image(poster_url, use_container_width=True)
                                            else:
                                                st.write("Aucune image disponible.")

                                    with col2:
                                        st.write(f"- **{row['localised_title']}** ({row['year']})  \n"
                                                f"  \n**Genres** : {genres}  \n"
                                                f"  \n**Réalisateur(s)** : {directors}  \n"
                                                f"  \n**Acteur(s)** : {actors}  \n"
                                                f"  \n**Résumé** : {row['overview_fr']}")
            
            elif mode == "Surprends moi !":
                if st.button("Reset"):
                    st.session_state.random_movie = afficher_film_aleatoire(df)
                elif 'random_movie' not in st.session_state:
                    st.session_state.random_movie = afficher_film_aleatoire(df)            
            
            else:  
                with st.expander("Filtres de recherche", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        selected_genres = st.multiselect("Genres", genres)
                        selected_countries = st.multiselect("Pays de production", countries)
                    
                    with col2:
                        selected_directors = st.multiselect("Réalisateurs", directors)
                        selected_actors = st.multiselect("Acteurs", actors)
                
                if st.button("Rechercher des films"):
                    filtered_df = filter_movies(
                        df_prepared,
                        selected_genres,
                        selected_countries,
                        selected_directors,
                        selected_actors
                    )
                    
                    if len(filtered_df) == 0:
                        st.warning("Aucun film ne correspond à ces critères.")
                    else:
                        st.success(f"{len(filtered_df)} films trouvés !")
                        
                        # Affichage des films filtrés
                        for _, row in filtered_df.head(NUM_RECOMMENDATIONS).iterrows():
                            genres = ', '.join(row['genres'])
                            directors = ', '.join(row['Directors'])
                            actors = ', '.join(row['Actors'])
                            poster_url = f"https://image.tmdb.org/t/p/w200{row['poster_path']}"
                            
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                if pd.notna(row['poster_path']):
                                    st.image(poster_url, use_container_width=True)
                                else:
                                    st.write("Aucune image disponible.")
                                    
                            with col2:
                                st.write(f"- **{row['localised_title']}** ({row['year']})  \n"
                                        f"  \n**Genres** : {genres}  \n"
                                        f"  \n**Réalisateur(s)** : {directors}  \n"
                                        f"  \n**Acteur(s)** : {actors}  \n"
                                        f"  \n**Résumé** : {row['overview_fr']}")
                                
        elif selection == "Recherche de courts métrages":
            st.markdown("<h2 style='font-size: 24px; text-align: center;'>Moteur de recommandations de courts métrages 🎥</h2>", unsafe_allow_html=True)             # Chargement et préparation des données
            download_nltk_resources()
            df = load_data('https://raw.githubusercontent.com/SeaJayEm/projet_2/refs/heads/main/df_list.csv')
            df_shorts = load_data('https://raw.githubusercontent.com/SeaJayEm/projet_2/refs/heads/main/shorts_translated.csv')
            movie_title = st.text_input("Nom du film", placeholder="Entrez le titre d'un film")
            genres = get_unique_sorted_values(df_shorts, 'genres')
            countries = get_unique_sorted_values(df_shorts, 'production_countries')
            directors = get_unique_sorted_values(df_shorts, 'primaryName')

            if movie_title:
                result = shorts_reco(df, df_shorts, movie_title)
                
                if result is None:
                    st.error(f"Aucun film trouvé correspondant à '{movie_title}'")
                else:
                    closest_title, match_score, recommendations = result  # Déballage du tuple

                    st.info(f"Film trouvé : '{closest_title}' (score de similarité : {match_score:.1f}%)")
                    st.write(f"Voici des courts métrages qui vont bien avec **{closest_title}** :")
                    
                    for _, row in recommendations.iterrows():
                        genres = ', '.join([g.strip() for g in row['genres']])  # Nettoie aussi les espaces
                        directors = ', '.join([d.strip() for d in row['primaryName']])
                        poster_url = f"https://image.tmdb.org/t/p/w200{row['poster_path']}"

                        # Vérifier si poster_path existe
                        if 'poster_path' in row and pd.notna(row['poster_path']):
                            poster_url = f"https://image.tmdb.org/t/p/w200{row['poster_path']}"
                        else:
                            poster_url = None

                        col1, col2 = st.columns([1, 2])
                        with col1:
                            if poster_url:
                                st.image(poster_url, use_container_width=True)
                            else:
                                st.write("Aucune image disponible.")

                        with col2:
                            st.write(f"- **{row['akas_title']}** ({row['startYear']})  \n"
                                    f"  \n**Genres** : {genres}  \n"
                                    f"  \n**Réalisateur(s)** : {directors}  \n"
                                    f"  \n**Durée** : {row['runtimeMinutes']} min  \n"
                                    f"  \n**Résumé** : {row['overview_fr']}")

    # Menu de navigation
    with st.sidebar:
        selection = on_hover_tabs(tabName=['Accueil', 'Recherche de films', 'Recherche de courts métrages', 'La base de données'], 
                         iconName=['home', 'movie', 'theaters', 'analytics'], default_choice=0)

    afficher_contenu(selection)

if __name__ == "__main__":
    main()