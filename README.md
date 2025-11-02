# Cooking-Recipe-Generator
Fine-Tuning GPT-2 for Recipe Generation

This application is live on Streamlit Community Cloud.

[https://cooking-recipie-generator-taqi-tallal.streamlit.app/]()

*(Replace the URL above with your actual Streamlit Cloud link)*

## 1. Project Overview

This project fine-tunes a GPT-2 model to generate creative cooking recipes from a title and a list of ingredients. It is the first of three tasks in a transformer fine-tuning assignment.

- **Model**: GPT-2 (124M parameters)
- **Dataset**: 3.2M+ Cooking Recipe Dataset
- **Fine-Tuned Model**: syedtaqi/recipe-generator

## 2. Objective

The goal was to fine-tune a Decoder-Only (Causal LM) model, GPT-2, to understand a specific new format. As a "text completion" engine, it was taught a pattern to act as a creative assistant that can:

- Generate a full recipe from just a title
- Infer ingredients from a title
- Generate a recipe from a list of ingredients

## 3. Methodology

The core of this task was data formatting. The model was trained on 100,000 recipes from the dataset, where each recipe was formatted into a single, consistent string:

```
TITLE: [Recipe Title]
INGREDIENTS: [Ingredient 1, Ingredient 2, ...]
RECIPE:
[Step 1...]
[Step 2...]
```

By fine-tuning on this pattern, the model learned that after seeing `TITLE:` and `INGREDIENTS:`, the most statistically probable text to generate next is `RECIPE:`, followed by the cooking steps.

## 4. Qualitative Analysis & Key Findings

The model performs well but shows classic, interesting behaviors of a Causal LM.

✅ **High-Quality Generation (Success)**: When given a prompt like `TITLE: Chocolate Chip Cookies` (with ingredients left blank), the model successfully infers all the standard ingredients (flour, sugar, eggs, etc.) and generates a complete, logical recipe. This shows it learned the deep relationship between the fields.

⚠️ **Vague/Lazy Generation (Weakness)**: When given a simple prompt like `TITLE: Guacamole` and all its ingredients, the model can be "lazy" and produce a technically correct but useless recipe: "Mix all ingredients together and serve." It fails to provide the necessary, nuanced steps like "mash the avocados" or "finely chop the onion."

❌ **Hallucination (Failure)**: When given a title-only prompt for a complex dish like `TITLE: Classic Beef Lasagna`, the model sometimes "hallucinates." In one test, it began correctly ("Cook noodles...") but then "lost the plot" and started generating steps for making "beef patties." This demonstrates the model following a statistically probable but contextually incorrect path (beef -> patties).

## 5. How to Run This App Locally

This app loads the model directly from the Hugging Face Hub.

### 1. Clone This Repository

```bash
git clone https://github.com/SyedTaqii/Cooking-Recipie-Generator.git
cd Cooking-Recipie-Generator
```

### 2. Install Dependencies

*(It is highly recommended to use a Python virtual environment)*

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

Your browser will automatically open, and the app will download the fine-tuned model from Hugging Face on the first run.