import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# --- 1. Load Model and Tokenizer (Cached) ---
# This @st.cache_resource decorator is crucial.
# It tells Streamlit to load the model only ONCE,
# not every time the user clicks a button.
@st.cache_resource
def load_model():
    """
    Loads the fine-tuned model and tokenizer from the local directory.
    """
    print("Loading fine-tuned model...")
    # This path assumes 'final_recipe_model' is in the same
    # directory as this 'app.py' file.
    model_path = './final_recipe_model'
    
    # Check if a GPU is available, otherwise use CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load the model and tokenizer
    try:
        model = GPT2LMHeadModel.from_pretrained(model_path)
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"ERROR: Could not load model. Ensure './final_recipe_model' exists.")
        print(e)
        return None, None, None

    # Move model to the correct device
    model.to(device)
    # Set model to evaluation mode (disables dropout, etc.)
    model.eval()

    # CRITICAL: Ensure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        
    print(f"Model loaded successfully on {device}.")
    return model, tokenizer, device

# Load the model once when the app starts
model, tokenizer, device = load_model()

# --- 2. The Inference Function (Corrected) ---
# --- 2. The Inference Function (Corrected) ---
def generate_recipe(title, ingredients):
    """
    Generates a full recipe based on the title and/or ingredients.
    """
    if not model:
        return "Error: Model is not loaded."

    # 1. Format the prompt
    prompt_text = f"TITLE: {title}\nINGREDIENTS: {ingredients}\nRECIPE:\n"
    
    # 2. Tokenize the prompt
    # --- THIS IS THE FIX ---
    # Instead of just .encode(), we call the tokenizer directly
    # to get both 'input_ids' and the 'attention_mask'.
    encoding = tokenizer(
        prompt_text, 
        return_tensors='pt'
    )
    input_ids = encoding.input_ids.to(device)
    attention_mask = encoding.attention_mask.to(device) # <-- We now have the mask

    prompt_length = input_ids.shape[1]
    
    # 3. Generate text
    with torch.no_grad():
        output_sequences = model.generate(
            input_ids=input_ids,
            
            # --- ADD THE MASK HERE ---
            attention_mask=attention_mask, 
            
            max_new_tokens=400, 
            do_sample=True,       
            temperature=0.7,      
            top_k=50,             
            top_p=0.95,           
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 4. Decode
    generated_text = tokenizer.decode(
        output_sequences[0][prompt_length:], 
        skip_special_tokens=True
    )
    
    return generated_text

# --- 3. The Streamlit User Interface ---
st.title("🔥 Recipe Generator")
st.markdown("Enter a recipe title and/or ingredients to generate a new recipe.")

# Input fields
title_input = st.text_input(
    "Recipe Title", 
    placeholder="e.g., Spicy Chicken Pasta"
)
ingredients_input = st.text_input(
    "Ingredients", 
    placeholder="e.g., chicken, pasta, chili flakes, tomato sauce"
)

# Generate button
if st.button("Generate Recipe"):
    # Check if at least one field is filled
    if not title_input and not ingredients_input:
        st.error("Please provide at least a title or some ingredients.")
    else:
        # Show a spinner while the model is working
        with st.spinner("Generating your recipe... This might take a moment."):
            # Run the generation
            recipe_output = generate_recipe(title_input, ingredients_input)
            
            # Display the output
            st.subheader("Generated Recipe")
            st.write(recipe_output)

# st.sidebar.header("About")
# st.sidebar.info(
#     "This app uses a GPT-2 model fine-tuned on the 3A2M+ recipe dataset "
#     "as part of a fine-tuning assignment."
# )