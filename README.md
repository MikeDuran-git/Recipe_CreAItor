# Recipe CreAItor

## Overview

**Recipe CreAItor** is an AI-powered application designed to help users generate new recipes, menus, and weekly meal plans based on their specific preferences, dietary restrictions, and available ingredients. The app leverages OpenAI's GPT models and embedding-based retrieval techniques to provide personalized recipe suggestions. The application is built using Python, Streamlit, and several other dependencies, making it easy to deploy and use on your local machine.

## Features

- **Create Custom Recipes**: Generate new recipes based on your preferred ingredients, dietary restrictions, and cooking tools.
- **Generate Menus**: Create cohesive menus with multiple recipes that fit together in terms of flavor and nutritional content.
- **Weekly Meal Planning**: Plan meals for an entire week, ensuring they meet your dietary needs and culinary preferences.
- **Recipe Storage**: Save your favorite recipes, menus, and plans for future use.
- **Grocery List Generation**: Automatically generate a grocery list based on your selected recipes or meal plans.

## Requirements

The application requires the following Python packages:

```txt
streamlit
torch
openai
numpy==1.26.4
pandas
sentence-transformers==2.5.1
tiktoken
```
Additionally, you'll need Python version 3.11.9 to run this application.

## Installation and Setup
To set up the project on your local machine, follow these steps:

1. Clone the Repository: 
```bash
git clone https://github.com/yourusername/recipe-creator.git
cd recipe-creator
```
2. **Create a Virtual Environment**:
```
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
3. **Install Dependencies**: It's recommended to use a virtual environment to manage dependencies. Install the required packages by running:
```
pip install -r requirements.txt
```

4. **Create the openaiCredentials.json File**: First, ensure that you have a JSON file named openaiCredentials.json in the data/embeddings folder with the following structure:
```
{
    "OPENAI_API_KEY": "your-openai-api-key",
    "ORGANIZATION_ID": "your-organization-id"
}
```
5. **Modify the Python Code:** Update your Python script to load the API credentials from this JSON file. Here’s how you can modify the relevant part of your code:
```
import json
import os

# Load OpenAI credentials from JSON file
credentials_path = os.path.join(data_dir, 'embeddings', 'openaiCredentials.json')

with open(credentials_path, 'r') as f:
    openai_credentials = json.load(f)

api_key = openai_credentials['OPENAI_API_KEY']
organization_id = openai_credentials['ORGANIZATION_ID']

client = OpenAI(
    api_key=api_key,
    organization=organization_id
)
```