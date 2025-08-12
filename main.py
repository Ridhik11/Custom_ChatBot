import pandas as pd
import openai
import os

openai.api_key = "voc-2064621629126677393550067da90ba9cf680.63522755"
# --- End of API Key section ---

# --- Dataset Selection and Scenario Explanation ---
# We're using the '2023_fashion_trends.csv' for this project.
# This dataset covers fashion trends from 2023. Imagine building a specialized chatbot
# for fashion enthusiasts, stylists, or anyone needing quick info on past trends.
# A chatbot customized with this data would offer detailed, specific answers about
# 2023 fashion that a general AI might not provide, or would give in a less focused way.
# It's all about getting curated information fast!

# --- Data Preparation ---
# Load our fashion trends data into a pandas DataFrame.
try:
    df = pd.read_csv('data/2023_fashion_trends.csv')
except FileNotFoundError:
    print("Oops! '2023_fashion_trends.csv' not found. Make sure it's in a 'data' folder.")
    
    dummy_data = {
        'source_url': ['https://example.com/trend1', 'https://example.com/trend2'],
        'article_title': ['The Rise of Comfort Wear', 'Sustainable Style Dominates'],
        'text_snippet': [
            'Oversized blazers were a big hit in 2023, mixing comfort with a sharp look for any occasion.',
            'Sustainable fashion really took off in 2023, focusing on recycled materials and ethical production.'
        ]
    }
    df = pd.DataFrame(dummy_data)


# Chatbot needs a 'text' column,  let's rename 'text_snippet'.
df = df.rename(columns={'text_snippet': 'text'})


# Quick peek at the data to ensure it's loaded correctly.
print("Here's how our fashion data looks after preparation:")
print(df.head())
print(f"\nTotal trend snippets loaded: {len(df)}.")

# --- Custom Query Process ---

# This function helps us find relevant snippets from our DataFrame for a given query.
# For this project, we're doing a simple keyword search.
# In a more advanced system,we use something like text embeddings for smarter matching.
def get_relevant_text(query, dataframe, top_n=3):
    """
    Finds and returns relevant text snippets from the DataFrame based on keywords in the query.
    It's a basic search for demonstration purposes.
    """
    query_lower = query.lower()
    found_snippets = []
    
    # We're just checking if query keywords are in the text.
    for index, row in dataframe.iterrows():
        if query_lower in row['text'].lower():
            found_snippets.append(row['text'])
        if len(found_snippets) >= top_n:
            break # Let's not overwhelm the model, grab just a few top snippets

    return "\n".join(found_snippets) if found_snippets else ""

# This function sends our query (and any custom context) to the OpenAI model.
# We're using the 'openai' v0 package as specified for this project.
def ask_openai_with_context(prompt, custom_context=""):
    # Setting up the conversation roles for the AI.
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]
    
    # If we have custom data, we'll add it as a system message for the AI to consider.
    if custom_context:
        messages.append({"role": "system", "content": f"Here's some background info on fashion trends:\n{custom_context}"})
    
    # Finally, the user's actual question.
    messages.append({"role": "user", "content": prompt})

    try:
        # Making the call to OpenAI's chat completion endpoint.
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", # A good general-purpose model for our task.
            messages=messages,
            temperature=0.7, # Controls creativity; 0.7 is fairly balanced.
            max_tokens=250 # Limit the response length.
        )
        return response.choices[0].message['content'].strip()
    except openai.error.OpenAIError as e:
        return f"Whoops, an OpenAI API error happened: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

print("\nChatbot functions are ready to go!")

# --- Demonstrate Custom Performance with Questions ---

# Question 1: A general query where our custom data should really shine.
question_1 = "What were the major fashion trends in 2023?"

print(f"\n--- Testing Question 1: '{question_1}' ---")

# First, let's ask without giving the chatbot any special data.
print("\nModel's answer WITHOUT our custom fashion data:")
response_without_custom_data_1 = ask_openai_with_context(question_1)
print(response_without_custom_data_1)

# Now, let's try again, but this time providing the chatbot with relevant info
# from our '2023_fashion_trends.csv' file.
print("\nModel's answer WITH our custom fashion data:")
context_1 = get_relevant_text(question_1, df)
response_with_custom_data_1 = ask_openai_with_context(question_1, custom_context=context_1)
print(response_with_custom_data_1)


# Question 2: A more specific question that our dataset should ideally answer well.
question_2 = "Tell me about sustainable fashion trends in 2023."

print(f"\n--- Testing Question 2: '{question_2}' ---")

# Again, start without our custom data.
print("\nModel's answer WITHOUT our custom fashion data:")
response_without_custom_data_2 = ask_openai_with_context(question_2)
print(response_without_custom_data_2)

# And now, with our custom data about sustainable fashion.
print("\nModel's answer WITH our custom fashion data:")
context_2 = get_relevant_text(question_2, df)
response_with_custom_data_2 = ask_openai_with_context(question_2, custom_context=context_2)
print(response_with_custom_data_2)
