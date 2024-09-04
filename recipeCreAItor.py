import streamlit as st
import torch
import numpy as np
import pandas as pd
import json
from openai import OpenAI # type: ignore
from sentence_transformers import util, SentenceTransformer
from time import perf_counter as timer
import tiktoken
import os
import requests
import warnings


language_texts_fr = {
    "language": "Français",
    "title_my_plans": "Mes Plans",  
    "no_plans_saved_yet": "Aucun plan enregistré pour l'instant.",  
    "label_plan": "Plan",  
    "label_total_nutrition": "Informations nutritionnelles totales pour",  
    "button_delete_plan": "Supprimer le plan",  
    "title_main": "Créateur de Recettes",
    "welcome_message": "Bienvenue dans votre générateur de recettes optimisé par l'IA !",
    "joke_caption": "Ne rigole pas, ton nom n'est pas mieux !",
    "main_board_description": "Créez de nouvelles recettes, menus et plans hebdomadaires",
    "main_board_section_new": "Créer Nouveau",
    "button_create_recipe": "Créer une Recette",
    "button_create_menu": "Créer un Menu",
    "button_create_weekly_plan": "Créer un Plan Hebdomadaire",
    "main_board_section_creations": "Mes Créations",
    "button_my_recipes": "Mes Recettes",
    "button_my_menus": "Mes Menus",
    "button_my_weekly_plans": "Mes Plans Hebdomadaires",
    "title_create_recipe": "Créer une Recette",
    "step_select_dish_type": "Étape 1 : Sélectionnez le type de plat",
    "select_dish_type_prompt": "Quel type de plat souhaitez-vous préparer ?",
    "step_number_of_people": "Étape 2 : Nombre de personnes",
    "slider_number_of_people_prompt": "Pour combien de personnes ? (entre 1 et 8)",
    "step_dietary_preferences": "Étape 3 : Préférences alimentaires et restrictions",
    "label_diet": "Régime",
    "label_restrictions": "Restrictions ou aversions (ex : 'pas de fruits rouges, pas de noix...')",
    "step_ingredients": "Étape 4 : Ingrédients",
    "input_ingredients_prompt": "Quels ingrédients souhaitez-vous inclure dans la recette ? (listez jusqu'à 5 ingrédients séparés par des virgules, vous pouvez également dire simplement 'fruits' ou 'légumes')",
    "step_preparation_time": "Étape 5 : Temps de préparation",
    "select_preparation_time_prompt": "Combien de temps souhaitez-vous que la recette prenne à préparer ?",
    "step_cooking_tools": "Étape 6 : Ustensiles de cuisine",
    "multiselect_cooking_tools_prompt": "Quels sont vos ustensiles de cuisine ?",
    "button_generate_recipe": "Générer une Recette",
    "message_generating_recipe": "Génération de la recette...",
    "label_recipe_prompt": "Demande de recette",
    "label_generated_recipe": "Recette Générée",
    "success_recipe_generated": "Recette générée en {elapsed_time:.2f} secondes.",
    "input_token_count_label": "Nombre de jetons d'entrée : {input_token_count}",
    "output_token_count_label": "Nombre de jetons de sortie : {output_token_count}",
    "button_save_recipe": "Enregistrer la Recette",
    "success_recipe_saved": "Recette enregistrée avec succès !",
    "title_create_menu": "Créer un Menu",
    "step_menu_name": "Étape 1 : Nom du Menu",
    "input_menu_name_prompt": "Entrez un nom pour votre menu",
    "step_number_of_recipes": "Étape 2 : Nombre de Recettes",
    "slider_number_of_recipes_prompt": "Combien de recettes souhaitez-vous inclure dans votre menu ? (2 à 6)",
    "step_menu_number_of_people": "Étape 3 : Nombre de Personnes",
    "slider_menu_number_of_people_prompt": "Pour combien de personnes ? (entre 1 et 8)",
    "step_menu_dietary_preferences": "Étape 4 : Préférences alimentaires et restrictions",
    "step_menu_cooking_tools": "Étape 5 : Ustensiles de cuisine",
    "button_generate_menu": "Générer un Menu",
    "message_generating_menu": "Génération du menu...",
    "label_menu_prompt": "Demande de menu",
    "label_generated_menu": "Menu Généré",
    "success_menu_generated": "Menu généré en {elapsed_time:.2f} secondes.",
    "input_menu_token_count_label": "Nombre de jetons d'entrée : {input_token_count}",
    "output_menu_token_count_label": "Nombre de jetons de sortie : {output_token_count}",
    "button_save_menu": "Enregistrer le Menu",
    "success_menu_saved": "Menu enregistré avec succès !",
    "label_person": "Personne",
    "label_recipe": "Recette",
    "label_type": "Type",
    "label_number_of_recipes_generated": "Nombre de recettes générées",
    "button_return_to_main_page": "Retourner à la page principale",
    "label_menu": "Menu",
    "title_create_weekly_plan": "Créer un Plan de Repas Hebdomadaire",
    "select_plan_type_for_day": "Sélectionnez une option pour",
    "option_none": "Aucune",
    "option_add_existing_menu": "Ajouter un Menu Existant",
    "option_add_existing_recipe": "Ajouter une Recette Existante",
    "select_menu_for_day": "Sélectionnez un menu pour",
    "select_recipe_for_day": "Sélectionnez une recette pour",
    "no_existing_menus_available": "Aucun menu existant disponible.",
    "no_existing_recipes_available": "Aucune recette existante disponible.",
    "add_another_plan_for_day": "Ajouter un autre plan pour",
    "input_plan_name_prompt": "Nommez votre plan hebdomadaire",
    "button_save_weekly_plan": "Enregistrer le Plan Hebdomadaire",
    "success_weekly_plan_saved": "Plan hebdomadaire enregistré avec succès !",
    "error_provide_plan_name_before_saving": "Veuillez fournir un nom pour le plan hebdomadaire avant d'enregistrer.",
    "title_my_recipes": "Mes Recettes",
    "search_recipe_by_title": "Rechercher une recette par titre",
    "no_recipes_saved_yet": "Aucune recette enregistrée pour l'instant.",
    "label_ingredients": "Ingrédients",
    "label_directions": "Instructions",
    "label_nutritional_info": "Informations Nutritionnelles",
    "label_calories": "Calories",
    "label_fat": "Graisse",
    "label_carbs": "Glucides",
    "label_protein": "Protéines",
    "label_prep_time": "Temps de Préparation",
    "select_for_grocery_list": "Sélectionner pour la Liste des Courses",
    "button_delete_recipe": "Supprimer la Recette",
    "button_generate_grocery_list": "Générer la Liste des Courses",
    "title_grocery_list": "Liste des Courses",
    "no_grocery_items": "Aucun article de courses trouvé.",
    "title_my_menus": "Mes Menus",  
    "search_menu_by_name": "Rechercher un menu par nom",  
    "no_menus_saved_yet": "Aucun menu enregistré pour l'instant.",  
    "button_generate_grocery_list": "Générer la Liste des Courses",  
    "button_delete_menu": "Supprimer le Menu"  
}



language_texts_en = {
    "language": "English",
    "title_my_plans": "My Plans",  # Added this key
    "no_plans_saved_yet": "No plans saved yet.",  # Added this key
    "label_plan": "Plan",  # Added this key
    "label_total_nutrition": "Total Nutritional Information for",  # Added this key
    "button_delete_plan": "Delete Plan",  # Added this key
    "title_main": "Recipe CreAItor",
    "welcome_message": "Welcome to your AI buffed Recipe generator!",
    "joke_caption": "Don't laugh, your name isn't any better!",
    "main_board_description": "Create New Recipes, Menus, and Weekly Plans",
    "main_board_section_new": "Create New",
    "button_create_recipe": "Create Recipe",
    "button_create_menu": "Create Menu",
    "button_create_weekly_plan": "Create Weekly Plan",
    "main_board_section_creations": "My Creations",
    "button_my_recipes": "My Recipes",
    "button_my_menus": "My Menus",
    "button_my_weekly_plans": "My Weekly Plans",
    "title_create_recipe": "Create Recipe",
    "step_select_dish_type": "Step 1: Select Dish Type",
    "select_dish_type_prompt": "What kind of dish do you want to make?",
    "step_number_of_people": "Step 2: Number of People",
    "slider_number_of_people_prompt": "For how many people? (between 1 and 8)",
    "step_dietary_preferences": "Step 3: Dietary Preferences and Restrictions",
    "label_diet": "Diet",
    "label_restrictions": "Restrictions or dislikes (ex: 'no red fruits, no nuts...')",
    "step_ingredients": "Step 4: Ingredients",
    "input_ingredients_prompt": "What ingredients would you like the recipe to have? (list up to 5 ingredients separated by commas, you can also just say 'fruits' or 'vegetables')",
    "step_preparation_time": "Step 5: Preparation Time",
    "select_preparation_time_prompt": "How much time would you like the recipe to take to prepare?",
    "step_cooking_tools": "Step 6: Cooking Tools",
    "multiselect_cooking_tools_prompt": "What are your cooking tools?",
    "button_generate_recipe": "Generate Recipe",
    "message_generating_recipe": "Generating recipe...",
    "label_recipe_prompt": "Recipe Prompt",
    "label_generated_recipe": "Generated Recipe",
    "success_recipe_generated": "Recipe generated in {elapsed_time:.2f} seconds.",
    "input_token_count_label": "Input token count: {input_token_count}",
    "output_token_count_label": "Output token count: {output_token_count}",
    "button_save_recipe": "Save Recipe",
    "success_recipe_saved": "Recipe saved successfully!",
    "title_create_menu": "Create Menu",
    "step_menu_name": "Step 1: Menu Name",
    "input_menu_name_prompt": "Enter a name for your menu",
    "step_number_of_recipes": "Step 2: Number of Recipes",
    "slider_number_of_recipes_prompt": "How many recipes do you want to have in your menu? (2 to 6)",
    "step_menu_number_of_people": "Step 3: Number of People",
    "slider_menu_number_of_people_prompt": "For how many people? (between 1 and 8)",
    "step_menu_dietary_preferences": "Step 4: Dietary Preferences and Restrictions",
    "step_menu_cooking_tools": "Step 5: Cooking Tools",
    "button_generate_menu": "Generate Menu",
    "message_generating_menu": "Generating menu...",
    "label_menu_prompt": "Menu Prompt",
    "label_generated_menu": "Generated Menu",
    "success_menu_generated": "Menu generated in {elapsed_time:.2f} seconds.",
    "input_menu_token_count_label": "Input token count: {input_token_count}",
    "output_menu_token_count_label": "Output token count: {output_token_count}",
    "button_save_menu": "Save Menu",
    "success_menu_saved": "Menu saved successfully!",
    "label_person": "Person",
    "label_recipe": "Recipe",
    "label_type": "Type",
    "label_number_of_recipes_generated": "Number of recipes generated",
    "button_return_to_main_page": "Return to Main Page",
    "label_menu": "Menu",
    "title_create_weekly_plan": "Create Weekly Meal Plan",
    "select_plan_type_for_day": "Select an option for",
    "option_none": "None",
    "option_add_existing_menu": "Add Existing Menu",
    "option_add_existing_recipe": "Add Existing Recipe",
    "select_menu_for_day": "Select a menu for",
    "select_recipe_for_day": "Select a recipe for",
    "no_existing_menus_available": "No existing menus available.",
    "no_existing_recipes_available": "No existing recipes available.",
    "add_another_plan_for_day": "Add another plan for",
    "input_plan_name_prompt": "Name your weekly plan",
    "button_save_weekly_plan": "Save Weekly Plan",
    "success_weekly_plan_saved": "Weekly plan saved successfully!",
    "error_provide_plan_name_before_saving": "Please provide a name for the weekly plan before saving.",
    "title_my_recipes": "My Recipes",
    "search_recipe_by_title": "Search for a recipe by title",
    "no_recipes_saved_yet": "No recipes saved yet.",
    "label_ingredients": "Ingredients",
    "label_directions": "Directions",
    "label_nutritional_info": "Nutritional Information",
    "label_calories": "Calories",
    "label_fat": "Fat",
    "label_carbs": "Carbs",
    "label_protein": "Protein",
    "label_prep_time": "Prep Time",
    "select_for_grocery_list": "Select for Grocery List",
    "button_delete_recipe": "Delete Recipe",
    "button_generate_grocery_list": "Generate Grocery List",
    "title_grocery_list": "Grocery List",
    "no_grocery_items": "No grocery items found.",
    "title_my_menus": "My Menus",  # Added this key
    "search_menu_by_name": "Search for a menu by name",  # Added this key
    "no_menus_saved_yet": "No menus saved yet.",  # Added this key
    "button_generate_grocery_list": "Generate Grocery List",  # Added this key
    "button_delete_menu": "Delete Menu"  # Added this key
}


language_texts_de = {
    "language": "Deutsch",
    "title_my_plans": "Meine Pläne",  # Added this key
    "no_plans_saved_yet": "Noch keine Pläne gespeichert.",  # Added this key
    "label_plan": "Plan",  # Added this key
    "label_total_nutrition": "Gesamt-Nährwertangaben für",  # Added this key
    "button_delete_plan": "Plan löschen",  # Added this key
    "title_main": "Rezept Ersteller",
    "welcome_message": "Willkommen bei deinem KI-gestärkten Rezeptgenerator!",
    "joke_caption": "Lach nicht, dein Name ist auch nicht besser!",
    "main_board_description": "Erstelle neue Rezepte, Menüs und Wochenpläne",
    "main_board_section_new": "Erstelle Neues",
    "button_create_recipe": "Rezept erstellen",
    "button_create_menu": "Menü erstellen",
    "button_create_weekly_plan": "Wochenplan erstellen",
    "main_board_section_creations": "Meine Kreationen",
    "button_my_recipes": "Meine Rezepte",
    "button_my_menus": "Meine Menüs",
    "button_my_weekly_plans": "Meine Wochenpläne",
    "title_create_recipe": "Rezept erstellen",
    "step_select_dish_type": "Schritt 1: Wählen Sie die Art des Gerichts",
    "select_dish_type_prompt": "Welche Art von Gericht möchten Sie zubereiten?",
    "step_number_of_people": "Schritt 2: Anzahl der Personen",
    "slider_number_of_people_prompt": "Für wie viele Personen? (zwischen 1 und 8)",
    "step_dietary_preferences": "Schritt 3: Ernährungsvorlieben und Einschränkungen",
    "label_diet": "Ernährung",
    "label_restrictions": "Einschränkungen oder Abneigungen (z.B.: 'keine roten Früchte, keine Nüsse...')",
    "step_ingredients": "Schritt 4: Zutaten",
    "input_ingredients_prompt": "Welche Zutaten möchten Sie im Rezept verwenden? (bis zu 5 Zutaten, getrennt durch Kommas, Sie können auch einfach 'Früchte' oder 'Gemüse' sagen)",
    "step_preparation_time": "Schritt 5: Zubereitungszeit",
    "select_preparation_time_prompt": "Wie viel Zeit möchten Sie für die Zubereitung des Rezepts aufwenden?",
    "step_cooking_tools": "Schritt 6: Küchengeräte",
    "multiselect_cooking_tools_prompt": "Welche Küchengeräte haben Sie zur Verfügung?",
    "button_generate_recipe": "Rezept generieren",
    "message_generating_recipe": "Rezept wird generiert...",
    "label_recipe_prompt": "Rezeptaufforderung",
    "label_generated_recipe": "Generiertes Rezept",
    "success_recipe_generated": "Rezept in {elapsed_time:.2f} Sekunden generiert.",
    "input_token_count_label": "Anzahl der Eingabe-Token: {input_token_count}",
    "output_token_count_label": "Anzahl der Ausgabe-Token: {output_token_count}",
    "button_save_recipe": "Rezept speichern",
    "success_recipe_saved": "Rezept erfolgreich gespeichert!",
    "title_create_menu": "Menü erstellen",
    "step_menu_name": "Schritt 1: Name des Menüs",
    "input_menu_name_prompt": "Geben Sie einen Namen für Ihr Menü ein",
    "step_number_of_recipes": "Schritt 2: Anzahl der Rezepte",
    "slider_number_of_recipes_prompt": "Wie viele Rezepte möchten Sie in Ihrem Menü haben? (2 bis 6)",
    "step_menu_number_of_people": "Schritt 3: Anzahl der Personen",
    "slider_menu_number_of_people_prompt": "Für wie viele Personen? (zwischen 1 und 8)",
    "step_menu_dietary_preferences": "Schritt 4: Ernährungsvorlieben und Einschränkungen",
    "step_menu_cooking_tools": "Schritt 5: Küchengeräte",
    "button_generate_menu": "Menü generieren",
    "message_generating_menu": "Menü wird generiert...",
    "label_menu_prompt": "Menüaufforderung",
    "label_generated_menu": "Generiertes Menü",
    "success_menu_generated": "Menü in {elapsed_time:.2f} Sekunden generiert.",
    "input_menu_token_count_label": "Anzahl der Eingabe-Token: {input_token_count}",
    "output_menu_token_count_label": "Anzahl der Ausgabe-Token: {output_token_count}",
    "button_save_menu": "Menü speichern",
    "success_menu_saved": "Menü erfolgreich gespeichert!",
    "label_person": "Person",
    "label_recipe": "Rezept",
    "label_type": "Typ",
    "label_number_of_recipes_generated": "Anzahl der generierten Rezepte",
    "button_return_to_main_page": "Zurück zur Hauptseite",
    "label_menu": "Menü",
    "title_create_weekly_plan": "Wochenplan erstellen",
    "select_plan_type_for_day": "Wählen Sie eine Option für",
    "option_none": "Keine",
    "option_add_existing_menu": "Vorhandenes Menü hinzufügen",
    "option_add_existing_recipe": "Vorhandenes Rezept hinzufügen",
    "select_menu_for_day": "Wählen Sie ein Menü für",
    "select_recipe_for_day": "Wählen Sie ein Rezept für",
    "no_existing_menus_available": "Keine vorhandenen Menüs verfügbar.",
    "no_existing_recipes_available": "Keine vorhandenen Rezepte verfügbar.",
    "add_another_plan_for_day": "Einen weiteren Plan hinzufügen für",
    "input_plan_name_prompt": "Nennen Sie Ihren Wochenplan",
    "button_save_weekly_plan": "Wochenplan speichern",
    "success_weekly_plan_saved": "Wochenplan erfolgreich gespeichert!",
    "error_provide_plan_name_before_saving": "Bitte geben Sie einen Namen für den Wochenplan an, bevor Sie speichern.",
    "title_my_recipes": "Meine Rezepte",
    "search_recipe_by_title": "Suche nach einem Rezept nach Titel",
    "no_recipes_saved_yet": "Noch keine Rezepte gespeichert.",
    "label_ingredients": "Zutaten",
    "label_directions": "Anweisungen",
    "label_nutritional_info": "Nährwertinformationen",
    "label_calories": "Kalorien",
    "label_fat": "Fett",
    "label_carbs": "Kohlenhydrate",
    "label_protein": "Eiweiß",
    "label_prep_time": "Vorbereitungszeit",
    "select_for_grocery_list": "Für Einkaufsliste auswählen",
    "button_delete_recipe": "Rezept löschen",
    "button_generate_grocery_list": "Einkaufsliste erstellen",
    "title_grocery_list": "Einkaufsliste",
    "no_grocery_items": "Keine Einkaufsartikel gefunden.",
    "title_my_menus": "Meine Menüs",  # Added this key
    "search_menu_by_name": "Suche nach einem Menü nach Name",  # Added this key
    "no_menus_saved_yet": "Noch keine Menüs gespeichert.",  # Added this key
    "button_generate_grocery_list": "Einkaufsliste erstellen",  # Added this key
    "button_delete_menu": "Menü löschen"  # Added this key
}




warnings.filterwarnings("ignore")

client = OpenAI(
    api_key=st.secrets['OPENAI_API_KEY'],
    organization=st.secrets['ORGANIZATION_ID']
)

language="English"

data_dir = "data/"

# Load models and data
model="gpt-4o-mini" #replace with a m

# we download the embedding model from the github repository
embedding_model_name = data_dir + "embeddings/embedding_model.pt"

# URL to the file in GitHub Releases
embedding_model_url = "https://github.com/MikeDuran-git/Recipe_CreAItor/releases/download/0.1/embedding_model.pt"

# Check if the model file already exists
if not os.path.exists(embedding_model_name):
    print("Embedding model not found locally. Downloading from GitHub...")
    try:
        # Stream the download with progress bar
        with requests.get(embedding_model_url, stream=True) as r:
            r.raise_for_status()
            total_size_in_bytes = int(r.headers.get('content-length', 0))
            block_size = 8192  # 8 Kibibytes
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(embedding_model_name), exist_ok=True)
            
            with open(embedding_model_name, 'wb') as f:
                for chunk in r.iter_content(chunk_size=block_size):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
        print("Download complete!")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while downloading the file: {e}")
else:
    #print("Embedding model found locally.")
    pass
test_chunks_and_embeddings_df_name=data_dir+"embeddings/text_chunks_and_embeddings_df.csv"

# files that are used in the app
recipe_dataset=data_dir+"recipes/recipe_dataset.csv"
saved_recipe_filename=data_dir+"recipes/saved_recipes.json"
saved_menu_filename=data_dir+"recipes/saved_menus.json"
saved_plan_filename=data_dir+"recipes/saved_plans.json"

recipe_dataset = pd.read_csv(recipe_dataset)
embedding_model = torch.load(embedding_model_name)
text_chunks_and_embedding_df = pd.read_csv(test_chunks_and_embeddings_df_name)
text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(
    lambda x: np.fromstring(x.strip("[]"), sep=" ")
)
pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")
# Load embeddings to the correct device
device = "cuda" if torch.cuda.is_available() else "cpu"
embeddings = torch.tensor(
    np.array(text_chunks_and_embedding_df["embedding"].tolist()),
    dtype=torch.float32
).to(device)

def send_message_to_recipe_model(msg, 
                                 content=(
                    "You are a helpful assistant that generates a new recipe based on a given list of ingredients. "
                    "Follow these guidelines strictly: "
                    "1. Ensure the generated recipes include only the specified elements in the following order and format, all in a single line: "
                    "'<recipe_start> <title_start>title_name<title_end> <ingredient_start>ingredient1<ingredient_next>ingredient2<ingredient_next>...<ingredient_end> "
                    "<directions_start>direction1<directions_next>direction2<directions_next>...<directions_end> <calories_start>calories_value<calories_end> "
                    "<fatcontent_start>fat_content_value<fatcontent_end> <carbohydratecontent_start>carbohydrate_content_value<carbohydratecontent_end> "
                    "<proteincontent_start>protein_content_value<proteincontent_end> <prep_time_min_start>prep_time_value<prep_time_min_end> "
                    "<type_start>type_value<type_end> <diet_start>diet_value<diet_end> <recipe_end>'. 2. The user will only provide a list of ingredients. Based on your training, "
                    "you must define the dosage of each ingredient. 3. Correct diet misclassification issues by ensuring the diet matches one of the following categories: vegan, vegetarian, contains_meat. "
                    "4. Maintain proper formatting in the output, including consistent spacing. 5. Accurately classify ingredients into meal types and diets based on enhanced training datasets. "
                    "6. Verify caloric and macronutrient calculations using the formula: Calories = 9 * fatContent + 4 * carbohydrateContent + 4 * proteinContent. "
                    "7. Ensure that the correct units are used for each ingredient. Solids should be measured in grams (g) and liquids in milliliters (ml). "
                    "8. Strictly enforce the expected output format by using the specified tokens: <title_start><title_end>, <ingredient_start><ingredient_end>, <directions_start><directions_end>, etc. "
                    "9. Ensure that the units are consistent and appropriate for the type of ingredient. 10. Normalize text to ensure consistent capitalization and remove unnecessary punctuation or characters. "
                    "11. Ensure no additional tokens or unexpected characters are included in the output. "
                    "12. Correct type misclassification issues by ensuring the type matches one of the following categories: appetizer, dinner, lunch, breakfast_and_brunch, desert "
                    "13. If a vegan recipe is requested and you see that there is an ingredient that is not vegan, you can specify the vegan version of it. For example, if the ingredient is Honey, you can specify (vegan) Honey, or (vegan) chicken."
                    "14. if you see 14 bananas, then that must mean 1/4 bananas"
                    "Example Input: ingredients: [mayonnaise, package knorr leek mix, sour cream, package spinach, chopped, drained well, loaf] "
                    "Expected Output: <recipe_start> <title_start> Bread Bowl Spinach Dip <title_end> <ingredient_start> 118.29 ml mayonnaise <ingredient_next> 26.57 g package knorr leek mix <ingredient_next> 118.29 ml sour cream "
                    "<ingredient_next> 141.74 g package spinach, chopped, drained well <ingredient_next> round sourdough loaf <ingredient_end> <directions_start> Mix first four ingredients well and refrigerate for 6 hours. "
                    "<directions_next> Create cavity inside French or sourdough bread loaf. <directions_next> Reserve pieces for dip. <directions_next> Fill the cavity with spinach dip. <directions_next> Makes 2 cups. "
                    "<directions_end> <calories_start> 84.9 <calories_end> <fatcontent_start> 7.8 <fatcontent_end> <carbohydratecontent_start> 2.8 <carbohydratecontent_end> <proteincontent_start> 2.1 <proteincontent_end> "
                    "<prep_time_min_start> 365 <prep_time_min_end> <type_start> appetizer <type_end> <diet_start> vegetarian <diet_end> <recipe_end>'"
                ) ,
                model='ft:gpt-3.5-turbo-1106:personal:recipecreatorv7b:9XU88CKN'):
    """
    This method sends a message to the recipe model and returns the response message.
    The form of the message is as follows: 
    ingredients: [ingredient1, ingredient2, ingredient3, ...]
    The model used by default is the RecipeCreatorV7b model, which is the model with the best performance.
    """
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "text"},
        messages=[
            {
                "role": "system",
                "content": content
            },
            {
                "role": "user",
                "content": msg
            },
        ],
    )
    return response.choices[0].message.content


# Load saved recipes
def load_recipes():
    try:
        with open(saved_recipe_filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


# Save recipes
def save_recipes(recipes):
    with open(saved_recipe_filename, "w") as f:
        json.dump(recipes, f)


# Load saved menus
def load_menus():
    try:
        with open(saved_menu_filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


# Save menus
def save_menus(menus):
    with open(saved_menu_filename, "w") as f:
        json.dump(menus, f)


# Load saved plans
def load_plans():
    try:
        with open(saved_plan_filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


# Save plans
def save_plans(plans):
    with open(saved_plan_filename, "w") as f:
        json.dump(plans, f)


# Navigation Function
def navigate():
    page = st.sidebar.selectbox("Select Page", ["Main Board", "Create Recipe", "Create Menu", "My Recipes", "My Menus", "Create Weekly Plan", "My Plans"])
    return page


# Main Board Layout
def main_board(language_texts):
    st.title(language_texts["title_main"])
    st.markdown(f"### {language_texts['welcome_message']}")
    st.markdown(f"_{language_texts['joke_caption']}_")
    st.caption(language_texts["main_board_description"])
    st.markdown("---")

    st.header(language_texts["main_board_section_new"])
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button(language_texts["button_create_recipe"]):
            st.session_state.page = "Create Recipe"
            st.rerun()
    with col2:
        if st.button(language_texts["button_create_menu"]):
            st.session_state.page = "Create Menu"
            st.rerun()
    with col3:
        if st.button(language_texts["button_create_weekly_plan"]):
            st.session_state.page = "Create Weekly Plan"
            st.rerun()
    
    st.markdown("---")
    st.header(language_texts["main_board_section_creations"])
    col4, col5, col6 = st.columns(3)
    with col4:
        if st.button(language_texts["button_my_recipes"]):
            st.session_state.page = "My Recipes"
            st.rerun()
    with col5:
        if st.button(language_texts["button_my_menus"]):
            st.session_state.page = "My Menus"
            st.rerun()
    with col6:
        if st.button(language_texts["button_my_weekly_plans"]):
            st.session_state.page = "My Plans"
            st.rerun()



def calculate_token_count(text, model_name=model):
    encoder = tiktoken.encoding_for_model(model_name)
    tokens = encoder.encode(text)
    return len(tokens)


def suggest_alternative(ingredient, diet):
    meat_alternatives = {
        'chicken': 'vegan chicken',
        'beef': 'vegan beef',
        'pork': 'vegan pork',
        'lamb': 'vegan lamb',
        'turkey': 'vegan turkey',
        'fish': 'vegan fish',
        'shrimp': 'vegan shrimp',
        'sausage': 'vegan sausage',
        'bacon': 'vegan bacon',
        'ham': 'vegan ham'
    }

    alternatives = {
        'vegan': meat_alternatives,
        'vegetarian': {
            'chicken': 'tofu',
            'beef': 'soy protein',
            'fish': 'tempeh',
            'gelatin': 'agar-agar'
        },
        'no gluten': {
            'wheat': 'gluten-free flour',
            'soy sauce': 'tamari',
            'barley': 'quinoa'
        },
        'no lactose': {
            'milk': 'lactose-free milk',
            'cheese': 'lactose-free cheese',
            'cream': 'coconut cream'
        }
    }

    if diet == 'vegan' and ingredient.lower() in meat_alternatives:
        return meat_alternatives[ingredient.lower()]
    return alternatives.get(diet, {}).get(ingredient.lower(), ingredient)


def retrieve_relevant_recipe(query: str,
                             embeddings: torch.tensor,
                             pages_and_chunks: list[dict] = pages_and_chunks,
                             n_resources_to_return: int = 5):
    scores, indices = retrieve_relevant_resources(query=query,
                                                  embeddings=embeddings,
                                                  n_resources_to_return=n_resources_to_return)

    recipes = []
    for score, index in zip(scores, indices):
        recipes.append(recipe_dataset.loc[recipe_dataset['page_number'] == pages_and_chunks[index]['page_number'], 'text'].values[0])

    context = "\n-------------------------------------\n".join(recipes)

    return context


def retrieve_relevant_resources(query: str,
                                embeddings: torch.tensor,
                                model: SentenceTransformer = embedding_model,
                                n_resources_to_return: int = 5,
                                print_time: bool = True):
    device = embeddings.device
    query_embedding = model.encode(query, convert_to_tensor=True).to(device)

    start_time = timer()
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    end_time = timer()
    if print_time:
        print(f"[INFO] Time taken to get scores on {len(embeddings)} embeddings: {end_time - start_time:.5f} seconds.")
    scores, indices = torch.topk(input=dot_scores, k=n_resources_to_return)
    return scores, indices


def prompt_formatter(query: str, context_items: str) -> str:
    context = context_items
    base_prompt = """
    You are an expert culinary chef with more than 10 years of experience. Create a recipe based on the following criteria:
    {context}

    The recipe must strictly adhere to the following guidelines:
    1. Respect all dietary restrictions and dislikes. Do not include any ingredients that are listed as restrictions. 
       If a restriction such as "red fruits" is specified, ensure no red fruits are included.
    2. Use the preferred ingredients where possible, substituting alternatives if they conflict with any dietary restrictions or dislikes.
    3. The preparation time should not exceed the specified maximum time.
    4. The recipe should be suitable for the number of people specified.
    5. The recipe should only require the specified cooking tools.

    Please provide the recipe with the following details:
    - Title
    - Ingredients (including quantities)
    - Directions
    - Nutritional information (calories, fat, carbs, protein)
    - Preparation time
    - Category
    - Diet
    """
    prompt = base_prompt.format(context=context, query=query)
    return prompt


def generate_recipe_based_on_questions_with_RAG(language: str = "English"):
    dish_type = st.session_state.dish_type
    number_of_people = st.session_state.number_of_people
    diets = st.session_state.diets
    restrictions = st.session_state.restrictions
    ingredients = st.session_state.ingredients
    max_time = st.session_state.max_time
    cooking_tools = st.session_state.cooking_tools

    compatible_ingredients = []
    meat_alternatives = {
        'chicken': 'vegan chicken',
        'beef': 'vegan beef',
        'pork': 'vegan pork',
        'lamb': 'vegan lamb',
        'turkey': 'vegan turkey',
        'fish': 'vegan fish',
        'shrimp': 'vegan shrimp',
        'sausage': 'vegan sausage',
        'bacon': 'vegan bacon',
        'ham': 'vegan ham'
    }

    for ingredient in ingredients:
        original_ingredient = ingredient
        for diet in diets:
            alternative = suggest_alternative(ingredient, diet)
            if alternative != ingredient:
                st.write(f"For the {diet} diet, replacing {ingredient} with {alternative}.")
            ingredient = alternative
        if ingredient in meat_alternatives.values():
            compatible_ingredients.append(f"(vegan) {ingredient}")
        else:
            compatible_ingredients.append(ingredient)

    context = f"""
    Dish Type: {dish_type}
    Number of People: {number_of_people}
    Diets: {', '.join(diets)}
    Restrictions:
    """
    for i in range(number_of_people):
        context += f"Person {i + 1}: {', '.join(restrictions[i])}\n"
    context += f"Preferred Ingredients: {', '.join(compatible_ingredients)}\n"
    context += f"Maximum Preparation Time: {max_time}\n"
    context += f"Available Cooking Tools: {', '.join(cooking_tools)}"

    prompt = f"""
    You are an expert culinary chef. Create a recipe based on the following criteria:
    {context}

    The recipe must strictly adhere to the following guidelines:
    1. Respect all dietary restrictions and dislikes. Do not include any ingredients that are listed as restrictions. If a restriction such as "red fruits" is specified, ensure no red fruits are included.
    2. Use the preferred ingredients where possible, substituting alternatives if they conflict with any dietary restrictions or dislikes.
    3. The preparation time should not exceed the specified maximum time.
    4. The recipe should be suitable for the number of people specified.
    5. The recipe should only require the specified cooking tools.

    For the recipe, provide the following details:
    - Title
    - Ingredients (including quantities)
    - Directions
    - Nutritional information for the whole recipe (calories, fat, carbs, protein)
    - Preparation time
    - Category
    - Diet

    return it in {language} but keep the token structure.
    """

    RAG_context = retrieve_relevant_recipe(query=f"{', '.join(compatible_ingredients)}", embeddings=embeddings, pages_and_chunks=pages_and_chunks)
    try:
        recipe_prompt = prompt + "\nInspire yourself from the these recipe to make a high protein and healthy recipe if possible:\n" + RAG_context

        recipe = send_message_to_recipe_model(recipe_prompt, model=model)
    except Exception as e:
        print(f"Error generating recipe: {e}")
        return "An error occurred while generating the recipe."

    return recipe_prompt, recipe


def format_recipe(recipe):
    recipe = recipe.replace("<recipe_start>", "").replace("<title_start>", "TITLE: ").replace("<ingredient_start>", "INGREDIENTS: \n-").replace("<ingredient_next>", "\n-").replace("<directions_start>", "DIRECTIONS: \n-").replace("<directions_next>", "\n-").replace("<calories_start>", "CALORIES: ").replace("<fatcontent_start>", "FAT: ").replace("<carbohydratecontent_start>", "CARBS: ").replace("<proteincontent_start>", "PROTEIN: ").replace("<prep_time_min_start>", "PREP TIME: ").replace("<type_start>", "TYPE: ").replace("<diet_start>", "DIET: ").replace("<title_end>", "\n").replace("<ingredient_end>", "\n").replace("<directions_end>", "\n").replace("<calories_end>", "\n").replace("<fatcontent_end>", "\n").replace("<carbohydratecontent_end>", "\n").replace("<proteincontent_end>", "\n").replace("<prep_time_min_end>", "\n").replace("<type_end>", "\n").replace("<diet_end>", "\n").replace("<recipe_end>", "")
    return recipe


# Recipe Creation Steps
def create_recipe(language_texts):
    if st.button(language_texts["button_return_to_main_page"]):
        st.session_state.page = "Main Board"
        st.rerun()
    
    st.title(language_texts["title_create_recipe"])
    st.write(f"### {language_texts['step_select_dish_type']}")
    dish_type = st.selectbox(language_texts["select_dish_type_prompt"], 
                             ["Breakfast", "Lunch", "Dinner", "Dessert", "Appetizer", "Snacks"])
    st.session_state.dish_type = dish_type

    st.write(f"### {language_texts['step_number_of_people']}")
    number_of_people = st.slider(language_texts["slider_number_of_people_prompt"], 1, 8, 1)
    st.session_state.number_of_people = number_of_people

    st.write(f"### {language_texts['step_dietary_preferences']}")
    diet_options = ["none", "vegan", "vegetarian", "pescetarian", "no gluten", "no lactose", "no pork"]
    diets = []
    restrictions = []
    for i in range(number_of_people):
        st.write(f"#### Person {i + 1}")
        col1, col2 = st.columns(2)
        with col1:
            diet = st.selectbox(language_texts["label_diet"], diet_options, key=f"diet_{i}")
        with col2:
            restriction = st.text_input(language_texts["label_restrictions"], key=f"restriction_{i}")
        diets.append(diet)
        restrictions.append([r.strip().lower() for r in restriction.split(',')])
    st.session_state.diets = diets
    st.session_state.restrictions = restrictions

    st.write(f"### {language_texts['step_ingredients']}")
    ingredients = st.text_input(language_texts["input_ingredients_prompt"]).split(',')
    ingredients = [ing.strip().lower() for ing in ingredients]
    st.session_state.ingredients = ingredients

    st.write(f"### {language_texts['step_preparation_time']}")
    time_options = ["at most 15 min", "between 15-30 min", "30 min or more"]
    max_time = st.selectbox(language_texts["select_preparation_time_prompt"], time_options)
    st.session_state.max_time = max_time

    st.write(f"### {language_texts['step_cooking_tools']}")
    tool_options = ["stovetop", "oven", "blender", "microwave", "automatic cooker", "fryer"]
    cooking_tools = st.multiselect(language_texts["multiselect_cooking_tools_prompt"], tool_options)
    st.session_state.cooking_tools = cooking_tools

    if st.button(language_texts["button_generate_recipe"]):
        with st.spinner(language_texts["message_generating_recipe"]):
            start_time = timer()
            recipe_prompt, generated_recipe = generate_recipe_based_on_questions_with_RAG(language_texts['language'])
            end_time = timer()
            elapsed_time = end_time - start_time
            input_token_count = calculate_token_count(recipe_prompt)
            output_token_count = calculate_token_count(generated_recipe)
            st.write(f"### {language_texts['label_recipe_prompt']}")
            st.text_area("Prompt", recipe_prompt, height=300)
            st.write(f"### {language_texts['label_generated_recipe']}")
            formatted_recipe = format_recipe(generated_recipe)
            st.session_state.formatted_recipe = formatted_recipe
            st.text_area("Recipe", formatted_recipe, height=300)
            st.success(language_texts["success_recipe_generated"].format(elapsed_time=elapsed_time))
            st.write(language_texts["input_token_count_label"].format(input_token_count=input_token_count))
            st.write(language_texts["output_token_count_label"].format(output_token_count=output_token_count))

    if 'formatted_recipe' in st.session_state:
        if st.button(language_texts["button_save_recipe"]):
            saved_recipes = load_recipes()
            saved_recipes.append(st.session_state.formatted_recipe)
            save_recipes(saved_recipes)
            st.success(language_texts["success_recipe_saved"])
            st.session_state.page = "Main Board"
            st.rerun()



def create_menu(language_texts):
    if st.button(language_texts['button_return_to_main_page']):
        st.session_state.page = "Main Board"
        st.rerun()

    st.title(language_texts['title_create_menu'])

    st.write(f"### {language_texts['step_menu_name']}")
    menu_name = st.text_input(language_texts['input_menu_name_prompt'])
    st.session_state.menu_name = menu_name

    st.write(f"### {language_texts['step_number_of_recipes']}")
    num_recipes = st.slider(language_texts['slider_number_of_recipes_prompt'], 2, 6, 2)
    st.session_state.num_recipes = num_recipes

    st.write(f"### {language_texts['step_menu_number_of_people']}")
    number_of_people = st.slider(language_texts['slider_menu_number_of_people_prompt'], 1, 8, 1)
    st.session_state.number_of_people = number_of_people

    st.write(f"### {language_texts['step_menu_dietary_preferences']}")
    diet_options = ["none", "vegan", "vegetarian", "pescetarian", "no gluten", "no lactose", "no pork"]
    diets = []
    restrictions = []
    for i in range(number_of_people):
        st.write(f"#### {language_texts['label_person']} {i + 1}")
        col1, col2 = st.columns(2)
        with col1:
            diet = st.selectbox(language_texts['label_diet'], diet_options, key=f"menu_diet_{i}")
        with col2:
            restriction = st.text_input(language_texts['label_restrictions'], key=f"menu_restriction_{i}")
        diets.append(diet)
        restrictions.append([r.strip().lower() for r in restriction.split(',')])
    st.session_state.diets = diets
    st.session_state.restrictions = restrictions

    st.write(f"### {language_texts['step_menu_cooking_tools']}")
    tool_options = ["stovetop", "oven", "blender", "microwave", "automatic cooker", "fryer"]
    cooking_tools = st.multiselect(language_texts['multiselect_cooking_tools_prompt'], tool_options)
    st.session_state.cooking_tools = cooking_tools

    recipes = []
    for i in range(num_recipes):
        st.write(f"### {language_texts['label_recipe']} {i + 1}")
        recipe_type = st.selectbox(language_texts['label_type'], ["Breakfast", "Lunch", "Dinner", "Dessert", "Appetizer", "Snacks"], key=f"recipe_type_{i}")
        ingredients = st.text_input(language_texts['input_ingredients_prompt'], key=f"recipe_ingredients_{i}").split(',')
        ingredients = [ing.strip().lower() for ing in ingredients]
        time_options = ["at most 15 min", "between 15-30 min", "30 min or more"]
        max_time = st.selectbox(language_texts['select_preparation_time_prompt'], time_options, key=f"recipe_time_{i}")
        recipes.append((recipe_type, ingredients, max_time))
    st.session_state.recipes = recipes

    if st.button(language_texts['button_generate_menu']):
        with st.spinner(language_texts['message_generating_menu']):
            start_time = timer()
            menu_prompt, generated_menu = generate_menu_based_on_questions_with_RAG(language_texts['language'])
            end_time = timer()
            elapsed_time = end_time - start_time
            input_token_count = calculate_token_count(menu_prompt)
            output_token_count = calculate_token_count("".join(generated_menu))
            st.write(f"### {language_texts['label_menu_prompt']}")
            st.text_area("Prompt", menu_prompt, height=300)
            st.write(f"### {language_texts['label_generated_menu']}")
            st.success(language_texts['success_menu_generated'].format(elapsed_time=elapsed_time))
            st.write(language_texts['input_menu_token_count_label'].format(input_token_count=input_token_count))
            st.write(language_texts['output_menu_token_count_label'].format(output_token_count=output_token_count))
            generated_menu = generated_menu.split("<recipe_end>")
            generated_menu = [m.strip() for m in generated_menu if m.strip() != ""]
            st.write(f"{language_texts['label_number_of_recipes_generated']}: {len(generated_menu)}")
            formatted_menu = [format_recipe(recipe) for recipe in generated_menu]
            st.session_state.formatted_menu = formatted_menu
            for i, recipe in enumerate(formatted_menu):
                st.write(f"### {language_texts['label_recipe']} {i + 1}")
                st.text_area(f"Recipe {i + 1}", recipe, height=300)

    if 'formatted_menu' in st.session_state:
        if st.button(language_texts['button_save_menu']):
            saved_menus = load_menus()
            saved_menus.append({"name": st.session_state.menu_name, "recipes": st.session_state.formatted_menu})
            save_menus(saved_menus)
            st.success(language_texts['success_menu_saved'])
            st.session_state.page = "Main Board"
            st.rerun()

def generate_menu_based_on_questions_with_RAG(language: str = "English"):
    num_recipes = st.session_state.num_recipes
    number_of_people = st.session_state.number_of_people
    diets = st.session_state.diets
    restrictions = st.session_state.restrictions
    cooking_tools = st.session_state.cooking_tools
    recipes = st.session_state.recipes

    menu_context = f"""
    Number of Recipes: {num_recipes}
    Number of People: {number_of_people}
    Diets: {', '.join(diets)}
    Restrictions:
    """
    for i in range(number_of_people):
        menu_context += f"Person {i + 1}: {', '.join(restrictions[i])}\n"
    menu_context += f"Available Cooking Tools: {', '.join(cooking_tools)}\n\n"

    menu = []
    for i, (recipe_type, ingredients, max_time) in enumerate(recipes):
        compatible_ingredients = []
        for ingredient in ingredients:
            original_ingredient = ingredient
            for diet in diets:
                alternative = suggest_alternative(ingredient, diet)
                if alternative != ingredient:
                    print(f"For the {diet} diet, replacing {ingredient} with {alternative}.")
                ingredient = alternative
            compatible_ingredients.append(ingredient)

        recipe_context = f"""
        Recipe {i + 1}:
        Type: {recipe_type}
        Preferred Ingredients: {', '.join(compatible_ingredients)}
        Maximum Preparation Time: {max_time}
        """
        menu_context += recipe_context

    prompt = f"""You are an expert culinary chef. Create a cohesive and harmonious menu based on the following criteria:
    {menu_context}

    Instructions:
    1. You must respect the number of recipes specified. For example, if the number of recipes is 2, then only return 2 recipes, not more, not less.
    2. Each recipe must strictly adhere to the following guidelines:
        a. Respect all dietary restrictions and dislikes. Do not include any ingredients that are listed as restrictions. If a restriction such as "red fruits" is specified, ensure no red fruits are included.
        b. Use the preferred ingredients where possible, substituting alternatives if they conflict with any dietary restrictions or dislikes.
        c. The preparation time should not exceed the specified maximum time.
        d. The recipes should be suitable for the number of people specified.
        e. The recipes should only require the specified cooking tools.
        f. Ensure the dishes are logically sequenced and have a harmonious flow from one to the next.
        g. if the user provides you for example with "vegetables" as an ingredient, you can use any type of vegetables that based on your knowledge fits well with the other informations provided and your knowledge (preferably of season).
        h. If the user provides you for example fruits, and it is not clear which fruits, you can use any type of fruits that based on your knowledge fits well with the other informations provided and your knowledge (preferably of season).
        i. Specify the macros per serving per person (calories, fat, carbs, protein).

    For each recipe, provide the following details:
    - Title
    - Ingredients (including quantities)
    - Directions
    - Nutritional information (calories, fat, carbs, protein)
    - Preparation time
    - Category
    - Diet

    return it in {language} but keep the token structure.
    """

    RAG_context = retrieve_relevant_recipe(query=f"{', '.join(compatible_ingredients)}", embeddings=embeddings, pages_and_chunks=pages_and_chunks)
    try:
        menu_prompt = prompt + "\nInspire yourself from the these recipe to make a high protein and healthy recipe if possible:\n" + RAG_context

        menu = send_message_to_recipe_model(menu_prompt, model=model)
        print(f"Generated menu: {menu}")
    except Exception as e:
        print(f"Error generating recipe: {e}")
        return "An error occurred while generating the recipe."

    return menu_prompt, menu


def extract_title(recipe):
    if isinstance(recipe, dict):
        return recipe.get("TITLE", "")
    title_start = recipe.find("TITLE:") + len("TITLE: ")
    title_end = recipe.find("\n", title_start)
    return recipe[title_start:title_end]

def extract_ingredients(recipe):
    if isinstance(recipe, dict):
        return recipe.get("INGREDIENTS", "")
    ingredients_start = recipe.find("INGREDIENTS:") + len("INGREDIENTS: ")
    ingredients_end = recipe.find("DIRECTIONS:")
    return recipe[ingredients_start:ingredients_end].strip()

def extract_directions(recipe):
    if isinstance(recipe, dict):
        return recipe.get("DIRECTIONS", "")
    directions_start = recipe.find("DIRECTIONS:") + len("DIRECTIONS: ")
    directions_end = recipe.find("CALORIES:")
    return recipe[directions_start:directions_end].strip()

def extract_nutritional_info(recipe):
    if isinstance(recipe, dict):
        calories = recipe.get("CALORIES", "")
        fat = recipe.get("FAT", "")
        carbs = recipe.get("CARBS", "")
        protein = recipe.get("PROTEIN", "")
    else:
        calories_start = recipe.find("CALORIES:") + len("CALORIES: ")
        calories_end = recipe.find("FAT:")
        fat_start = recipe.find("FAT:") + len("FAT: ")
        fat_end = recipe.find("CARBS:")
        carbs_start = recipe.find("CARBS:") + len("CARBS: ")
        carbs_end = recipe.find("PROTEIN:")
        protein_start = recipe.find("PROTEIN:") + len("PROTEIN: ")
        protein_end = recipe.find("PREP TIME:")
        calories = recipe[calories_start:calories_end].strip()
        fat = recipe[fat_start:fat_end].strip()
        carbs = recipe[carbs_start:carbs_end].strip()
        protein = recipe[protein_start:protein_end].strip()

    return calories, fat, carbs, protein

def extract_prep_time(recipe):
    if isinstance(recipe, dict):
        return recipe.get("PREP TIME", "")
    prep_time_start = recipe.find("PREP TIME:") + len("PREP TIME: ")
    prep_time_end = recipe.find("TYPE:")
    return recipe[prep_time_start:prep_time_end].strip()

def extract_type(recipe):
    if isinstance(recipe, dict):
        return recipe.get("TYPE", "")
    type_start = recipe.find("TYPE:") + len("TYPE: ")
    type_end = recipe.find("DIET:")
    return recipe[type_start:type_end].strip()

def extract_diet(recipe):
    if isinstance(recipe, dict):
        return recipe.get("DIET", "")
    diet_start = recipe.find("DIET:") + len("DIET: ")
    diet_end = recipe.find("\n", diet_start)
    return recipe[diet_start:diet_end].strip()


# My Recipes Page
# My Recipes Page
def my_creations(language_texts):
    if st.button(language_texts["button_return_to_main_page"]):
        st.session_state.page = "Main Board"
        st.rerun()
    
    st.title(language_texts["title_my_recipes"])

    search_query = st.text_input(language_texts["search_recipe_by_title"])
    saved_recipes = load_recipes()
    selected_recipes = []

    if not saved_recipes:
        st.write(language_texts["no_recipes_saved_yet"])
    else:
        filtered_recipes = [recipe for recipe in saved_recipes if search_query.lower() in extract_title(recipe).lower()] if search_query else saved_recipes
        print(f"Filtered Recipes: {[extract_title(recipe) for recipe in filtered_recipes]}")  # Debug print

        for idx, recipe in enumerate(filtered_recipes):
            title = extract_title(recipe)
            ingredients = extract_ingredients(recipe)
            directions = extract_directions(recipe)
            calories, fat, carbs, protein = extract_nutritional_info(recipe)
            prep_time = extract_prep_time(recipe)
            type_ = extract_type(recipe)
            diet = extract_diet(recipe)

            st.write(f"### {title}")
            st.write(f"**{language_texts['label_ingredients']}:**\n{ingredients}")
            st.write(f"**{language_texts['label_directions']}:**\n{directions}")
            st.write(f"**{language_texts['label_nutritional_info']}:**")
            st.write(f"- {language_texts['label_calories']}: {calories}")
            st.write(f"- {language_texts['label_fat']}: {fat}")
            st.write(f"- {language_texts['label_carbs']}: {carbs}")
            st.write(f"- {language_texts['label_protein']}: {protein}")
            st.write(f"**{language_texts['label_prep_time']}:** {prep_time}")
            st.write(f"**{language_texts['label_type']}:** {type_}")
            st.write(f"**{language_texts['label_diet']}:** {diet}")

            if st.checkbox(f"{language_texts['select_for_grocery_list']} {title}", key=f"select_{idx}"):
                selected_recipes.append(recipe)
                print(f"Selected Recipe: {title}")  # Debug print

            if st.button(f"{language_texts['button_delete_recipe']} {idx + 1}", key=f"delete_{idx}"):
                del saved_recipes[idx]
                save_recipes(saved_recipes)
                st.session_state.page = language_texts["title_my_recipes"]
                st.rerun()

    if selected_recipes:
        print(f"Selected Recipes for Grocery List: {[extract_title(recipe) for recipe in selected_recipes]}")  # Debug print
        if st.button(language_texts["button_generate_grocery_list"]):
            grocery_list = generate_grocery_list(selected_recipes)
            print(f"Generated Grocery List: {grocery_list}")  # Debug print
            st.session_state.grocery_list = grocery_list
            st.session_state.page = "Grocery List"
            st.rerun()



def my_menus(language_texts):
    if st.button(language_texts["button_return_to_main_page"]):
        st.session_state.page = "Main Board"
        st.rerun()
    st.title(language_texts["title_my_menus"])

    search_query = st.text_input(language_texts["search_menu_by_name"])
    saved_menus = load_menus()
    if not saved_menus:
        st.write(language_texts["no_menus_saved_yet"])
    else:
        filtered_menus = [menu for menu in saved_menus if search_query.lower() in menu['name'].lower()] if search_query else saved_menus
        for idx, menu in enumerate(filtered_menus):
            st.write(f"### {language_texts['label_menu']} {idx + 1}: {menu['name']}")
            for j, recipe in enumerate(menu['recipes']):
                title = extract_title(recipe)
                ingredients = extract_ingredients(recipe)
                directions = extract_directions(recipe)
                calories, fat, carbs, protein = extract_nutritional_info(recipe)
                prep_time = extract_prep_time(recipe)
                type_ = extract_type(recipe)
                diet = extract_diet(recipe)

                st.write(f"#### {language_texts['label_recipe']} {j + 1}: {title}")
                st.write(f"**{language_texts['label_ingredients']}:**\n{ingredients}")
                st.write(f"**{language_texts['label_directions']}:**\n{directions}")
                st.write(f"**{language_texts['label_nutritional_info']}:**")
                st.write(f"- {language_texts['label_calories']}: {calories}")
                st.write(f"- {language_texts['label_fat']}: {fat}")
                st.write(f"- {language_texts['label_carbs']}: {carbs}")
                st.write(f"- {language_texts['label_protein']}: {protein}")
                st.write(f"**{language_texts['label_prep_time']}:** {prep_time}")
                st.write(f"**{language_texts['label_type']}:** {type_}")
                st.write(f"**{language_texts['label_diet']}:** {diet}")

            if st.button(f"{language_texts['button_generate_grocery_list']} {idx + 1}", key=f"menu_select_{idx}"):
                grocery_list = generate_grocery_list(menu['recipes'])
                st.session_state.grocery_list = grocery_list
                st.session_state.page = "Grocery List"
                st.rerun()

            if st.button(f"{language_texts['button_delete_menu']} {idx + 1}", key=f"delete_menu_{idx}"):
                del saved_menus[idx]
                save_menus(saved_menus)
                st.session_state.page = language_texts["title_my_menus"]
                st.rerun()



def generate_grocery_list(selected_recipes):
    grocery_list = {}
    for recipe in selected_recipes:
        ingredients = extract_ingredients(recipe).split('\n-')
        for ingredient in ingredients:
            ingredient = ingredient.strip()
            if ingredient:
                if ingredient in grocery_list:
                    grocery_list[ingredient] += 1
                else:
                    grocery_list[ingredient] = 1
    return grocery_list


# Create Weekly Plan Page
def create_weekly_plan(language_texts):
    if st.button(language_texts["button_return_to_main_page"], key="return_main_page"):
        st.session_state.page = "Main Board"
        st.rerun()

    st.title(language_texts["title_create_weekly_plan"])

    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    daily_plans = {}

    for day in days_of_week:
        st.write(f"### {day}")
        daily_plans[day] = daily_plans.get(day, [])

        add_more = True
        plan_index = 0
        while add_more:
            plan_type = st.selectbox(
                f"{language_texts['select_plan_type_for_day']} {day}", 
                [language_texts['option_none'], language_texts['option_add_existing_menu'], language_texts['option_add_existing_recipe']], 
                key=f"{day}_plan_type_{plan_index}"
            )

            if plan_type == language_texts['option_add_existing_menu']:
                existing_menus = load_menus()
                if existing_menus:
                    menu_selection = st.selectbox(
                        f"{language_texts['select_menu_for_day']} {day}", 
                        [f"{language_texts['label_menu']} {i+1}" for i in range(len(existing_menus))], 
                        key=f"{day}_menu_selection_{plan_index}"
                    )
                    selected_menu = existing_menus[int(menu_selection.split()[-1]) - 1]
                    daily_plans[day].append(selected_menu)
                else:
                    st.write(language_texts['no_existing_menus_available'])

            elif plan_type == language_texts['option_add_existing_recipe']:
                existing_recipes = load_recipes()
                if existing_recipes:
                    recipe_titles = [extract_title(recipe) for recipe in existing_recipes]
                    recipe_selection = st.selectbox(
                        f"{language_texts['select_recipe_for_day']} {day}", 
                        recipe_titles, 
                        key=f"{day}_recipe_selection_{plan_index}"
                    )
                    selected_recipe = existing_recipes[recipe_titles.index(recipe_selection)]
                    daily_plans[day].append(selected_recipe)
                else:
                    st.write(language_texts['no_existing_recipes_available'])

            plan_index += 1
            add_more = st.checkbox(f"{language_texts['add_another_plan_for_day']} {day}", key=f"{day}_add_more_{plan_index}")

    plan_name = st.text_input(language_texts["input_plan_name_prompt"], key="plan_name_input")
    save_plan_button = st.button(language_texts["button_save_weekly_plan"], key="save_plan_button")

    if save_plan_button and plan_name:
        weekly_plan = {
            "plan_name": plan_name,
            "days": daily_plans
        }
        saved_plans = load_plans()
        saved_plans.append(weekly_plan)
        save_plans(saved_plans)
        st.success(language_texts["success_weekly_plan_saved"])
        st.session_state.page = "Main Board"
        st.rerun()
    elif save_plan_button:
        st.error(language_texts["error_provide_plan_name_before_saving"])



def display_recipe(recipe):
    title = extract_title(recipe)
    ingredients = extract_ingredients(recipe)
    directions = extract_directions(recipe)
    calories, fat, carbs, protein = extract_nutritional_info(recipe)
    prep_time = extract_prep_time(recipe)
    type_ = extract_type(recipe)
    diet = extract_diet(recipe)

    st.write(f"**Title:** {title}")
    st.write(f"**Ingredients:**\n{ingredients}")
    st.write(f"**Directions:**\n{directions}")
    st.write(f"**Nutritional Information:**")
    st.write(f"- Calories: {calories}")
    st.write(f"- Fat: {fat}")
    st.write(f"- Carbs: {carbs}")
    st.write(f"- Protein: {protein}")
    st.write(f"**Prep Time:** {prep_time}")
    st.write(f"**Type:** {type_}")
    st.write(f"**Diet:** {diet}")


def my_plans(language_texts):
    if st.button(language_texts["button_return_to_main_page"], key="return_main_page"):
        st.session_state.page = "Main Board"
        st.rerun()
    
    st.title(language_texts["title_my_plans"])
    saved_plans = load_plans()
    if not saved_plans:
        st.write(language_texts["no_plans_saved_yet"])
    else:
        for idx, plan in enumerate(saved_plans):
            st.write(f"### {language_texts['label_plan']} {idx + 1}: {plan['plan_name']}")
            for day, items in plan["days"].items():
                st.write(f"#### {day}")

                # Initialize totals for the day's nutritional information
                total_calories = 0
                total_fat = 0
                total_carbs = 0
                total_protein = 0

                for item in items:
                    if isinstance(item, dict) and 'recipes' in item:
                        for recipe in item['recipes']:
                            display_recipe(recipe)
                            # Sum up the nutritional information
                            calories, fat, carbs, protein = extract_nutritional_info(recipe)
                            total_calories += float(calories)
                            total_fat += float(fat)
                            total_carbs += float(carbs)
                            total_protein += float(protein)
                    else:
                        display_recipe(item)
                        # Sum up the nutritional information
                        calories, fat, carbs, protein = extract_nutritional_info(item)
                        total_calories += float(calories)
                        total_fat += float(fat)
                        total_carbs += float(carbs)
                        total_protein += float(protein)

                # Display the total nutritional information for the day
                st.write(f"**{language_texts['label_total_nutrition']} {day}:**")
                st.write(f"- {language_texts['label_calories']}: {total_calories}")
                st.write(f"- {language_texts['label_fat']}: {total_fat}g")
                st.write(f"- {language_texts['label_carbs']}: {total_carbs}g")
                st.write(f"- {language_texts['label_protein']}: {total_protein}g")
                st.write("---")

            if st.button(f"{language_texts['button_generate_grocery_list']} {language_texts['label_plan']} {idx + 1}", key=f"plan_select_{idx}"):
                all_recipes = [recipe for day_items in plan["days"].values() for item in day_items for recipe in (item if isinstance(item, list) else [item])]
                grocery_list = generate_grocery_list(all_recipes)
                st.session_state.grocery_list = grocery_list
                st.session_state.page = "Grocery List"
                st.rerun()

            if st.button(f"{language_texts['button_delete_plan']} {idx + 1}", key=f"delete_plan_{idx}"):
                del saved_plans[idx]
                save_plans(saved_plans)
                st.session_state.page = "My Plans"
                st.rerun()



# Grocery List Page
def grocery_list(language_texts):
    if st.button(language_texts["button_return_to_main_page"]):
        st.session_state.page = "Main Board"
        st.rerun()
    
    st.title(language_texts["title_grocery_list"])
    grocery_list = st.session_state.get("grocery_list", {})
    
    if grocery_list:
        for ingredient, quantity in grocery_list.items():
            st.write(f"{ingredient}: {quantity}")
    else:
        st.write(language_texts.get("no_grocery_items", "No grocery items found."))

def main():
    if 'page' not in st.session_state:
        st.session_state.page = "Main Board"
    
    # Add a dropdown menu to select the language with a unique key
    language = st.sidebar.selectbox("Select Language", ["English", "Deutsch", "Français"], key="language_select_main")

    # Load the appropriate language dictionary based on the selection
    if language == "Deutsch":
        language_texts = language_texts_de
    elif language == "Français":
        language_texts = language_texts_fr
    else:
        language_texts = language_texts_en

    page = st.session_state.page

    # Use the selected language dictionary
    if page == "Main Board":
        main_board(language_texts)
    elif page == "Create Recipe":
        create_recipe(language_texts)
    elif page == "Create Menu":
        create_menu(language_texts)
    elif page == "Create Weekly Plan":
        create_weekly_plan(language_texts)
    elif page == "My Recipes":
        my_creations(language_texts)
    elif page == "My Menus":
        my_menus(language_texts)
    elif page == "My Plans":
        my_plans(language_texts)
    elif page == "Grocery List":
        grocery_list(language_texts)


if __name__ == "__main__":
    main()
