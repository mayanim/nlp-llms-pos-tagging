# --- Imports ---
import os
from google import genai
from pydantic import BaseModel, Field, RootModel
from typing import List, Literal, Optional, Dict

from enum import Enum
from sklearn.metrics import classification_report, accuracy_score
import json
import time

# 
gemini_model = 'gemini-2.0-flash-lite'

# --- Define Pydantic Models for Structured Output ---

# --- Define the Universal Dependencies POS Tagset (17 core tags) as an enum ---
class UDPosTag(str, Enum):
    ADJ = "ADJ"
    ADP = "ADP"
    ADV = "ADV"
    AUX = "AUX"
    CCONJ = "CCONJ"
    DET = "DET"
    INTJ = "INTJ"
    NOUN = "NOUN"
    NUM = "NUM"
    PART = "PART"
    PRON = "PRON"
    PROPN = "PROPN"
    PUNCT = "PUNCT"
    SCONJ = "SCONJ"
    SYM = "SYM"
    VERB = "VERB"
    X = "X"

# Define more Pydantic models for structured output

class TokenPOS(BaseModel):
    """Represents a token with its part-of-speech tag."""
    word: str = Field(description="The token itself.")
    pos_tag: UDPosTag = Field(description="The part-of-speech tag of the token.")

class SentencePOS(BaseModel):
    """Represents a sentence with its tokens and their tags."""
    tokens: List[TokenPOS] = Field(description="A list of tokens with their POS tags.")

class TaggedSentences(BaseModel):
    """Represents a list of sentences with their tagged tokens."""
    sentences: List[SentencePOS] = Field(description="A list of sentences, each containing tagged tokens.")

class ErrorExplanation(BaseModel):
    explanation: str = Field(description="Explanation of the tagging error.")
    category: str = Field(description="Error category: a single label describing the error.")   

class SegmentedSentences(RootModel[List[List[str]]]):
    pass


# --- Configure the Gemini API ---
# Get a key https://aistudio.google.com/plan_information 
# Use os.environ.get for production environments.
# For Colab/AI Studio, you might use userdata.get
# Example:
# from google.colab import userdata
# GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
# genai.configure(api_key=GOOGLE_API_KEY)

# Make sure to replace "YOUR_API_KEY" with your actual key if running locally
# and not using environment variables or userdata.
try:
    # Attempt to get API key from environment variable
    api_key = os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        # Fallback or specific instruction for local setup
        # Replace with your actual key if needed, but environment variables are safer
        api_key = "AIzaSyD-GCxfd6ertCOHDaHhW_Pj4Y8FTZQ54Ac"
        if api_key == "YOUR_API_KEY":
           print("⚠️ Warning: API key not found in environment variables. Using placeholder.")
           print("   Please set the GOOGLE_API_KEY environment variable or replace 'YOUR_API_KEY' in the code.")

    #genai.configure(api_key=api_key) #removed

except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    print("Please ensure you have a valid API key set.")
    # Depending on the environment, you might want to exit here
    # import sys
    # sys.exit(1)


# --- Function to Perform POS Tagging ---

def tag_sentences_ud(text_to_tag: List[str]) -> Optional[TaggedSentences]: # for one sentence use: text_to_tag: str
    """
    Performs POS tagging on the input text using the Gemini API and
    returns the result structured according to the SentencePOS Pydantic model.

    Args:
        text_to_tag: The sentence or text to be tagged.

    Returns:
        A TaggedSentences object containing the tagged tokens, or None if an error occurs.
    """
    #joined = "\n".join(f"{i+1}. {s}" for i, s in enumerate(text_to_tag)) # for one sentence use: f"1. {text_to_tag}"
    #joined = "\n\n".join(f"{i+1}.\n" + "\n".join(s.split()) for i, s in enumerate(text_to_tag))
    joined = "\n\n".join(f"{i+1}. {s}" for i, s in enumerate(text_to_tag))


    # Construct the prompt
    prompt = f"""
                You are a POS tagger.

                Your task is to tag each token in the given sentence(s) using the Universal Dependencies (UD) POS tagset.

                Use **exactly one** of the following 17 tags for each word token:
                ADJ, ADP, ADV, AUX, CCONJ, DET, INTJ, NOUN, NUM, PART, PRON, PROPN, PUNCT, SCONJ, SYM, VERB, X

                Do not change, interpret, combine, or skip any words.
                Do not include any explanation, commentary, or additional text.

                Return your response **exactly** in the following JSON format (with double quotes and field names as shown):

                {{
                "sentences": [
                    {{
                    "tokens": [
                        {{"word": "The", "pos_tag": "DET"}},
                        {{"word": "quick", "pos_tag": "ADJ"}},
                        ...
                    ]
                    }},
                    ...
                ]
                }}

                **The sentences are pre-tokenized**: each word is already split as it should be tagged. Do not change or split any tokens.

                Now tag the following sentence(s):

                {joined}
                """

    client = genai.Client(api_key=api_key) 
  
    try:
        response = client.models.generate_content(
            model=gemini_model,
            contents=prompt,
            config={
                'response_mime_type': 'application/json',
                'response_schema': TaggedSentences,
            },
        )

    except Exception as e:
        print(f"Error during API call: {e}")
        return None
        
    # Use the response as a JSON string.
    #print(response.text)

    # Use instantiated objects.
    res: TaggedSentences = response.parsed
    return res

def explanations_for_errors(errors: List[Dict[str, str]]) -> Optional[List[ErrorExplanation]]:
    """
    Generates explanations and error categories for tagging errors made by both LR and LLM taggers.

    Args:
        errors: A list of dictionaries with keys:
            - "token"
            - "gold_tag"
            - "lr_tag"
            - "gemini_tag"
            - "sentence"

    Returns:
        A list of ErrorExplanation objects with explanation and category, or None if API fails.
    """

    results = []

    for i, e in enumerate(errors):
        prompt = f"""
        You are given a sentence and a token in that sentence that was incorrectly tagged by an LLM-based POS tagger.

        Your task is to analyze the tagging error of the given token, and return:

        1. An explanation of **why the token was misclassified**, based on the context of the sentence and the provided tags.
        2. A **single error category label**, based on the explanation you provided.

        Do not return any text outside the JSON structure.
        Do not explain the other tags — just the token's confusion.

        Return your response **exactly** in the following JSON format (with double quotes and field names as shown):

        {{
        "explanation": "Your explanation here.",
        "category": "Your error category here."
        }}

        Example:

        {{
        "explanation": "'Gravity' is the name of a specific company (Gravity Corp.), making it a proper noun in this context. However, 'gravity' is also a common noun referring to the physical force. Unless the tagger has specific knowledge or strong contextual clues (like surrounding proper nouns or capitalization patterns) to identify it as part of a named entity ('Gravity CEO Kim Jung-Ryool'), it might default to the more frequent common noun tag.",
        "category": "Proper Noun vs Common Noun"
        }}

        Now analyze the following error:

        Sentence: {e['sentence']}
        Token: "{e['token']}"
        Gold POS tag: {e['gold_tag']}
        LLM tag: {e['gemini_tag']}
        """

        client = genai.Client(api_key=api_key)

        try:
            response = client.models.generate_content(
                model=gemini_model,
                contents=prompt,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': ErrorExplanation,
                },
            )
            results.append(response.parsed)

        except Exception as ex:
            print(f"❌ Error on example {i + 1}: {ex}")
            results.append(None)

    return results


# --- section 3 ---

few_shot_examples = [
        {
            "original": "What if Google expanded on its search-engine (and now e-mail) wares into a full-fledged operating system?",
            "tokens": ["What", "if", "Google", "expanded", "on", "its", "search", "-", "engine", "(", "and", "now", "e-mail", ")", "wares", "into", "a", "full", "-", "fledged", "operating", "system", "?"]
        },
        {
            "original": "They can't believe it's already mid-September!",
            "tokens": ["They", "ca", "n't", "believe", "it", "'s", "already", "mid", "-", "September", "!"]
        },
        {
            "original": "I'll re-schedule the follow-up for post-launch.",
            "tokens": ["I", "'ll", "re", "-", "schedule", "the", "follow", "-", "up", "for", "post", "-", "launch", "."]
        }
    ]

def segment_sentences_ud(sentences: List[str]) -> Optional[List[List[str]]]:
    """
    Segments each sentence using the Gemini LLM and returns a list of token lists.
    
    Args:
        sentences: A list of original, unsegmented sentences.

    Returns:
        A list of lists, where each inner list contains the UD-style tokens of a sentence.
    """
    

    example_prompt = "\n".join(
        f"Input: {ex['original']}\nOutput: {json.dumps(ex['tokens'])}"
        for ex in few_shot_examples
    )

    user_sentences = "\n".join(
        f"Sentence {i+1}: {s}" for i, s in enumerate(sentences)
    )

    system_instructions = """
    You are a tokenizer that strictly follows the Universal Dependencies (UD) English word segmentation rules.
    Your task is to segment each sentence into individual tokens.

    Follow these instructions carefully:
    - Use the UD guidelines: https://universaldependencies.org/en/tokenization.html
    - Split punctuation and symbols (e.g., '.', ',', '(', ')')
    - Split hyphenated words (e.g., "full-fledged" → "full", "-", "fledged")
    - Split contractions (e.g., "can't" → "ca", "n't")
    - Keep emoji and special symbols as separate tokens
    - Do not invent or explain anything
    - Only return valid JSON: a **list of lists of strings** — one list per sentence
    - **Do not include any explanation or extra text**
    """

    user_prompt = f"""


    ### Examples:

    {example_prompt}

    ### Now segment the following sentences:

    {user_sentences}
    """

    client = genai.Client(api_key=api_key)

    try:
            response = client.models.generate_content(
                model=gemini_model,
                contents=system_instructions.strip() + "\n\n" + user_prompt.strip(),
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': SegmentedSentences,
                },
            )
            return response.parsed.root

    except Exception as e:
            print(f"❌ Error during segmentation API call: {e}")
            return None   


# --- 3.3 - Improved Model ----
def pipeline_model(sentences: List[str]) -> Optional[List[SentencePOS]]:
    """
    First segments each sentence using the LLM segmenter,
    then tags the segmented sentence using the POS tagger.
    
    Args:
        sentences: List of original (unsegmented) sentences
    
    Returns:
        List of SentencePOS objects (tagged sentences)
    """
    segmented = segment_sentences_ud(sentences)
    if segmented is None:
        return None
    
    rejoined = [" ".join(tokens) for tokens in segmented]

    tagged = tag_sentences_ud(rejoined)
    if tagged is None:
        return None
    
    return tagged.sentences

few_shot_joint_examples = [
        {
            "original": "What if Google expanded on its search-engine (and now e-mail) wares into a full-fledged operating system?",
            "tokens": [
                {"word": "What", "pos_tag": "PRON"},
                {"word": "if", "pos_tag": "SCONJ"},
                {"word": "Google", "pos_tag": "PROPN"},
                {"word": "expanded", "pos_tag": "VERB"},
                {"word": "on", "pos_tag": "ADP"},
                {"word": "its", "pos_tag": "PRON"},
                {"word": "search", "pos_tag": "NOUN"},
                {"word": "-", "pos_tag": "PUNCT"},
                {"word": "engine", "pos_tag": "NOUN"},
                {"word": "(", "pos_tag": "PUNCT"},
                {"word": "and", "pos_tag": "CCONJ"},
                {"word": "now", "pos_tag": "ADV"},
                {"word": "e-mail", "pos_tag": "NOUN"},
                {"word": ")", "pos_tag": "PUNCT"},
                {"word": "wares", "pos_tag": "NOUN"},
                {"word": "into", "pos_tag": "ADP"},
                {"word": "a", "pos_tag": "DET"},
                {"word": "full", "pos_tag": "ADJ"},
                {"word": "-", "pos_tag": "PUNCT"},
                {"word": "fledged", "pos_tag": "ADJ"},
                {"word": "operating", "pos_tag": "ADJ"},
                {"word": "system", "pos_tag": "NOUN"},
                {"word": "?", "pos_tag": "PUNCT"}
            ]
        },
        {
            "original": "They can't believe it's already mid-September!",
            "tokens": [
                {"word": "They", "pos_tag": "PRON"},
                {"word": "ca", "pos_tag": "AUX"},
                {"word": "n't", "pos_tag": "PART"},
                {"word": "believe", "pos_tag": "VERB"},
                {"word": "it", "pos_tag": "PRON"},
                {"word": "'s", "pos_tag": "AUX"},
                {"word": "already", "pos_tag": "ADV"},
                {"word": "mid", "pos_tag": "NOUN"},
                {"word": "-", "pos_tag": "PUNCT"},
                {"word": "September", "pos_tag": "PROPN"},
                {"word": "!", "pos_tag": "PUNCT"}
            ]
        },
        {
            "original": "I'll re-schedule the follow-up for post-launch.",
            "tokens": [
                {"word": "I", "pos_tag": "PRON"},
                {"word": "'ll", "pos_tag": "AUX"},
                {"word": "re", "pos_tag": "ADV"},
                {"word": "-", "pos_tag": "PUNCT"},
                {"word": "schedule", "pos_tag": "VERB"},
                {"word": "the", "pos_tag": "DET"},
                {"word": "follow", "pos_tag": "NOUN"},
                {"word": "-", "pos_tag": "PUNCT"},
                {"word": "up", "pos_tag": "PART"},
                {"word": "for", "pos_tag": "ADP"},
                {"word": "post", "pos_tag": "NOUN"},
                {"word": "-", "pos_tag": "PUNCT"},
                {"word": "launch", "pos_tag": "NOUN"},
                {"word": ".", "pos_tag": "PUNCT"}
            ]
        }
    ]

def joint_model(sentences: List[str]) -> Optional[List[SentencePOS]]:
    """
    Performs joint segmentation and POS tagging using the Gemini API.
    Returns a list of SentencePOS where each token has a word and its UD POS tag.

    Args:
        sentences: A list of original, unsegmented sentences.

    Returns:
        A list of SentencePOS objects or None if API call fails.
    """

    example_prompt = "\n\n".join(
        f"Input: {ex['original']}\nOutput: {json.dumps(ex['tokens'])}"
        for ex in few_shot_joint_examples
    )

    user_sentences = "\n".join(f"Sentence {i+1}: {s}" for i, s in enumerate(sentences))

    system_instruction = """
    You are a tokenizer and POS tagger that strictly follows the Universal Dependencies (UD) English guidelines.
    Given a raw sentence, first segment it into tokens, then assign a POS tag from the UD tagset to each token.

    - Follow the UD segmentation rules: https://universaldependencies.org/en/tokenization.html
    - Use only these POS tags: ADJ, ADP, ADV, AUX, CCONJ, DET, INTJ, NOUN, NUM, PART, PRON, PROPN, PUNCT, SCONJ, SYM, VERB, X
    - Return only valid JSON: a list of lists of token-POS dictionaries, one per sentence
    - Do not return any explanation or extra formatting

    Each sentence must be returned as:
    [
    {"word": "The", "pos_tag": "DET"},
    {"word": "quick", "pos_tag": "ADJ"},
    ...
    ]
    """

    user_prompt = f"""
    ### Examples:
    {example_prompt}

    ### Now tag the following sentences:
    {user_sentences}
    """

    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model=gemini_model,
            contents=system_instruction.strip() + "\n\n" + user_prompt.strip(),
            config={
                'response_mime_type': 'application/json',
                'response_schema': TaggedSentences,
            },
        )
        return response.parsed.sentences

    except Exception as e:
        print(f"❌ Error during joint tagging API call: {e}")
        return None


# --- Example Usage ---
if __name__ == "__main__":
    #example_text = "The quick brown fox jumps over the lazy dog."
    #example_text = "What if Google expanded on its search-engine (and now e-mail) wares into a full-fledged operating system?"
    #example_text = "Google Search is a web search engine developed by Google LLC."
    #example_text = "החתול המהיר קופץ מעל הכלב העצלן." # Example in Hebrew

    #print(f"\nTagging text: \"{example_text}\"")

    example_sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "Google Search is a web search engine developed by Google LLC.",
    "He acted like it never happened.",
    "We order take out from here all the time.",
    "החתול המהיר קופץ מעל הכלב העצלן.",
    "I try to add more examples to the list.",
    "Is this working?"
]   


    tagged_result = tag_sentences_ud(example_sentences) # for one sentence use: tag_sentences_ud(example_text)

    if tagged_result:
        print("\n--- Tagging Results ---")
        for s in tagged_result.sentences:
            #Retrieve tokens and tags from each sentence:
            for token_pos in s.tokens:
                token = token_pos.word
                tag = token_pos.pos_tag
                # Handle potential None for pos_tag if model couldn't assign one
                ctag = tag if tag is not None else "UNKNOWN"
                print(f"Token: {token:<15} {str(ctag)}")
                print("----------------------")
    else:
        print("\nFailed to get POS tagging results.")









"""
# --- 2.2 Error Analysis ---

# load LR data for hard sentences with 1-3 errors
with open("hard_sentences_lr_results.json", "r", encoding="utf-8") as f:
    hard_sentences_data = json.load(f)

# extract the sentences and their corresponding tags
hard_sentences_1_3 = []
correct_tags = []
lr_tags = []

for item in hard_sentences_data:
    hard_sentences_1_3.append(item["sentence"])
    correct_tags.extend(item["gold_tags"])
    lr_tags.extend(item["lr_tags"])

# tag sentences in batches by BaseModel
batch_size = 4
skipped = 0
skipped_indices = set()

tagged_hard = []
for i in range(0, len(hard_sentences_1_3), batch_size):
    batch = hard_sentences_1_3[i:i + batch_size]
    print(f"Tagging batch {i // batch_size + 1} / {len(hard_sentences_1_3) // batch_size + 1}")
    tagged_result = tag_sentences_ud(batch)

    if tagged_result is None:
        print(f"⚠️ Skipping batch {i // batch_size + 1} due to API failure.")
        skipped += 1
        skipped_indices.update(range(i, i + batch_size))
        continue
    
    tagged_hard.extend(tagged_result.sentences)

    time.sleep(3)  # delay to avoid hitting API limits

# extract tags from the tagged sentences
tagged_tags = []
for s in tagged_hard:
    for token_pos in s.tokens:
        token = token_pos.word
        tag = token_pos.pos_tag
        # Handle potential None for pos_tag if model couldn't assign one
        ctag = tag if tag is not None else "UNKNOWN"
        tagged_tags.append(ctag)


def filter_flat_list_by_sentence_indices(flat_list, lengths, skipped_set):
    result = []
    index = 0
    for i, length in enumerate(lengths):
        if i not in skipped_set:
            result.extend(flat_list[index:index + length])
        index += length
    return result


print("check before")
# compare accuracy of the model with the LR model
print("\n--- Error Analysis ---")
print("Gemini Model vs. LR Model")
print("-------------------------------------------------")
print(f"Number of sentences: {len(correct_tags)}")

if (len(correct_tags)== len(tagged_tags)) and (len(lr_tags)== len(tagged_tags)):
    print(f"Accuracy of the Gemini model: {accuracy_score(correct_tags, tagged_tags):.2f}")
    print(f"Accuracy of the LR model: {accuracy_score(correct_tags, lr_tags):.2f}")

else:
    #print(f"⚠️ Warning: Length mismatch between correct_tags and tagged_tags. Skipping accuracy calculation.")
    sentence_lengths = [len(s["gold_tags"]) for s in hard_sentences_data]
    filtered_correct_tags = filter_flat_list_by_sentence_indices(correct_tags, sentence_lengths, skipped_indices)
    filtered_lr_tags = filter_flat_list_by_sentence_indices(lr_tags, sentence_lengths, skipped_indices)
    filtered_tagged_tags = filter_flat_list_by_sentence_indices(tagged_tags, sentence_lengths, skipped_indices)

    print(f"Accuracy of the Gemini model: {accuracy_score(filtered_correct_tags, filtered_tagged_tags):.2f}")
    print(f"Accuracy of the LR model: {accuracy_score(filtered_lr_tags, filtered_tagged_tags):.2f}")


print("finish")

error_fixed_LLM_Tagger = 0
new_errors_LLM_Tagger = 0

# continue later

# --- section 3 ----

few_shot_examples = [
    {
        "sentence": "The quick brown fox jumps over the lazy dog.",
        "tags": ["DET", "ADJ", "ADJ", "NOUN", "VERB", "ADP", "DET", "ADJ", "NOUN"]
    },
    {
        "sentence": "Google Search is a web search engine developed by Google LLC.",
        "tags": ["PROPN", "PROPN", "AUX", "DET", "NOUN", "NOUN", "NOUN", "VERB", "ADP", "PROPN", "PROPN"]
    }
]

"""




    








