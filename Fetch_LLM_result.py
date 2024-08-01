# Fetch_LLM_result.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from huggingface_hub import InferenceClient
import numpy as np
import config
import json
import re
from receipt_OCR_processor import extract_text_from_file
import sys
import warnings
# Suppress FutureWarnings from huggingface_hub
warnings.filterwarnings('ignore', category=FutureWarning, module='huggingface_hub')


# Define the prompt template
prompt_template = """
The information you need to extract includes the following fields: Date, Vendor, Without tax total amount, Tax, Total amount, Country, Province. Sometimes, the country and province information may be missing, and you will need to infer the most likely country and province based on the other information in the text.

Here is an example of the receipt text:
---
Text from Original Image with Orientation Detection:
{receipt_text}
---

Please provide the extracted information in the following JSON format:
{{
    "Date": "2024-08-01",
    "Vendor": "Church's Chicken",
    "Without tax total amount": "27.50",
    "Tax": "1.38",
    "Total amount": "28.88",
    "Country": "Canada",
    "Province": "British Columbia",
    "Comment":"I inferred the country as Canada and the province as British Columbia based on the vendor location "Surrey, Bila" which is a city in the province of British Columbia, Canada."
}}

Remember, if the any information is not explicitly mentioned,
 you should infer it from the other details in the receipt or leave it empty, and explain it in the Comment,
 you should be succint. 
 DO NOT add comment inside the JSON, like this  "Total amount": "25.20", // Inferred as total amount since only one amount is provided".
 If there is only one amount information, you should infer that this is total amount.
 "Without tax total amount","Tax","Total amount" values should be in numberic format.
 If you can't detect , just leave it as "". DO NOT use "Unknown" or "Unavailuable"
"""

# Create a function to generate the prompt
def generate_prompt(receipt_text):
    return prompt_template.format(receipt_text=receipt_text)

# Function to query HuggingFace
def query_huggingface(client, prompt):
    messages = [
        {"role": "system", "content": "You are an intelligent assistant tasked with extracting important information from receipt texts. "},
        {"role": "user", "content": f"{prompt}"}
    ]

    response_text = ""
    for message in client.chat_completion(
        messages=messages,
        max_tokens=500,
        stream=True,
    ):
        if "choices" in message and message["choices"]:
            response_text += message["choices"][0]["delta"].get("content", "")
    # print('messages',messages)
    return response_text

# Use HuggingFace InferenceClient for question answering
client = InferenceClient(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    token=config.HuggingFace_Token_KEY
)

def extract_information_from_receipt(receipt_text):
    prompt = generate_prompt(receipt_text)
    response_text = query_huggingface(client, prompt)
    # print(response_text)
    return response_text

def extract_json_from_result(result_string):
    # Use regular expression to find the JSON part, considering comments
    json_match = re.search(r'\{[\s\S]*?\}', result_string)
    if json_match:
        json_str = json_match.group(0)
        
        # Remove comments and trailing commas (if any)
        json_str = re.sub(r'//.*', '', json_str)  # Remove single line comments
        json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas before closing brace
        json_str = re.sub(r',\s*\]', ']', json_str)  # Remove trailing commas before closing bracket
        
        try:
            # Load the cleaned string as JSON to validate and return it
            json_data = json.loads(json_str)
            return json_data
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON: {e}")
            return None
    else:
        print("No JSON found in the result string")
        return None


def main(file_path):
    # Extract text from the provided file (PDF or image)
    receipt_text = extract_text_from_file(file_path)
    # print(receipt_text)
    # This function will be called with the extracted text from the receipt.
    extracted_information = extract_information_from_receipt(receipt_text)
    extracted_json = extract_json_from_result(extracted_information)
    return extracted_json




# Example usage
if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]  # Get file path from command line argument
        result = main(file_path)
        print(result)
    else:
        print("Please provide a file path as a command line argument.")





