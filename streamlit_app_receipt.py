import streamlit as st
from receipt_OCR_processor import preprocess_image, extract_text_from_file, convert_pdf_to_images
from Fetch_LLM_result import process_receipt_file
import os
import cv2
from PIL import Image
import pandas as pd

# Create directories for uploaded and processed images
uploaded_dir = "USER_uploaded_receipts"
processed_dir = "Processed_USER_receipts"
os.makedirs(uploaded_dir, exist_ok=True)
os.makedirs(processed_dir, exist_ok=True)

# Add "Built with Meta Llama 3" at the top with colorful text
st.markdown(
    "<h1 style='text-align: center; color: #FF6347;'>Built with <span style='color: #4682B4;'>Meta Llama 3</span></h1>", 
    unsafe_allow_html=True
)

# Streamlit app title
st.title("Receipt Information Extraction")

# File uploader
uploaded_files = st.file_uploader("Upload your receipts (JPEG/PNG/PDF)", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)

uploaded_file_list = []
results = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Save uploaded file
        file_path = os.path.join(uploaded_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        uploaded_file_list.append(uploaded_file.name)
        
        # Process the file and extract JSON
        extracted_json,extracted_text = process_receipt_file(file_path)
        if extracted_json:
            extracted_json["file_name"] = uploaded_file.name  # Add file name to JSON result
            results.append((extracted_json, extracted_text))

# Display uploaded files
st.write("## Uploaded Files")
st.write(uploaded_file_list)

# Function to display images
def display_images(file_name):
    file_path = os.path.join(uploaded_dir, file_name)
    if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        st.image(file_path, caption="Original Image", use_column_width=True)
        processed_image_path = os.path.join(processed_dir, f"Processed_{file_name}")
        if not os.path.exists(processed_image_path):
            processed_image = preprocess_image(file_path)
            cv2.imwrite(processed_image_path, processed_image)
        st.image(processed_image_path, caption="Processed Image", use_column_width=True)
    elif file_name.lower().endswith('.pdf'):
        st.write(f"PDF uploaded: {file_name}")
        # Convert the first page of the PDF to an image
        image_paths = convert_pdf_to_images(file_path, processed_dir)
        if image_paths:
            st.image(image_paths[0], caption="First Page of PDF", use_column_width=True)

# Display results table with radio buttons
if results:
    st.write("## Extracted Information")
    df = pd.DataFrame([res[0] for res in results])
    
    # Convert appropriate columns to numeric type
    numeric_columns = ["Without tax total amount", "Tax", "Total amount"]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Display the table with radio buttons
    selected_idx = st.radio("Select a row", range(len(df)), format_func=lambda x: f"{df['file_name'][x]}")
    
    # Display the original and processed images for the selected row
    if st.button("Show Images for Selected Row"):
        selected_file = df.iloc[selected_idx]['file_name']
        st.write(f"Selected File: {selected_file}")
        display_images(selected_file)

    # Display the table
    st.dataframe(df)

    # Button to display extracted text
    if st.button("Show Extracted Text"):
        st.write("## Extracted Text")
        st.text(results[selected_idx][1])  # Display the extracted text corresponding to the selected row