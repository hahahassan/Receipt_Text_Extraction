import os
import cv2
import pandas as pd
import streamlit as st
from PIL import Image
from receipt_OCR_processor import preprocess_image, extract_text_from_file, convert_pdf_to_images
from Fetch_LLM_result import process_receipt_file
from calculate_accuracy import accuracy  # Placeholder import, to be implemented later

# Directory setup for uploaded and processed images
UPLOADED_DIR = "USER_uploaded_receipts"
PROCESSED_DIR = "Processed_USER_receipts"
os.makedirs(UPLOADED_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# App Title and Introduction with Custom Styling
st.markdown(
    """
    <h1 style='text-align: center; color: #FF6347;'>Receipt Insight Hub</h1>
    <p style='text-align: center;'>This application allows you to upload receipt images or PDFs, extracts relevant information, 
    and displays it in an organized table format. Built using advanced OCR and NLP techniques, and enhanced by Meta-Llama 3.</p>
    """, 
    unsafe_allow_html=True
)

# Custom CSS to increase sidebar width and adjust scrollbar for showing more records
st.markdown(
    """
    <style>

    /* Footer styling */
    footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #f8f9fa;
        color: #333;
        padding: 5px 0;
        text-align: center;
        font-size: 15px;
        z-index: 100;
    }

    /* Increase table width */
    .dataframe {
        width: 100% !important;
    }

    </style>
    """, 
    unsafe_allow_html=True
)

# Sidebar for File Upload and Controls
st.sidebar.header("Upload Your Receipts")
uploaded_files = st.sidebar.file_uploader(
    "Upload your receipts (JPEG/PNG/PDF)", 
    type=["jpg", "jpeg", "png", "pdf"], 
    accept_multiple_files=True
)

# Initialize lists to store uploaded file names and results
uploaded_file_list = []
results = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Save the uploaded file
        file_path = os.path.join(UPLOADED_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        uploaded_file_list.append(uploaded_file.name)
        
        # Process the file and extract information
        extracted_json, extracted_text = process_receipt_file(file_path)
        if extracted_json:
            extracted_json["file_name"] = uploaded_file.name  # Include file name in JSON
            results.append((extracted_json, extracted_text))

# Function to display original and processed images side by side
def display_images_side_by_side(file_name):
    file_path = os.path.join(UPLOADED_DIR, file_name)
    col1, col2 = st.columns(2)

    if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        with col1:
            st.markdown("### Original Image")
            st.image(file_path, use_column_width=True)
        
        processed_image_path = os.path.join(PROCESSED_DIR, f"Processed_{file_name}")
        if not os.path.exists(processed_image_path):
            processed_image = preprocess_image(file_path)
            cv2.imwrite(processed_image_path, processed_image)
        
        with col2:
            st.markdown("### Processed Image")
            st.image(processed_image_path, use_column_width=True)
    
    elif file_name.lower().endswith('.pdf'):
        st.write(f"PDF uploaded: {file_name}")
        # Convert the first page of the PDF to an image
        image_paths = convert_pdf_to_images(file_path, PROCESSED_DIR)
        if image_paths:
            with col1:
                st.markdown("### Original Image (First Page)")
                st.image(image_paths[0], use_column_width=True)

# Display the results if any files were uploaded
if results:
    st.write("## Extracted Information")
    
    # Create a DataFrame from the results
    df = pd.DataFrame([res[0] for res in results])
    
    # Reformat the date column to be consistent
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
    
    # Convert relevant columns to numeric types and format them to 2 decimal places
    numeric_columns = ["Without tax total amount", "Tax", "Total amount"]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').map('{:,.2f}'.format)
    
    # Reorder columns to start with 'file_name'
    df = df[['file_name', 'Date', 'Without tax total amount', 'Tax', 'Total amount'] + [col for col in df.columns if col not in ['file_name', 'Date', 'Without tax total amount', 'Tax', 'Total amount']]]
    
    # Display the DataFrame without the index column
    st.dataframe(df.reset_index(drop=True))
    
    # Radio button selection for choosing a specific file to view details
    selected_idx = st.radio(
        "Select a file to view details", 
        range(len(df)), 
        format_func=lambda x: f"{df['file_name'][x]}"
    )
    
    # Tabbed interface for images and accuracy
    tab1, tab2 = st.tabs(["Images", "Accuracy"])
    
    with tab1:
        # Automatically display the original and processed images side by side for the selected file
        selected_file = df.iloc[selected_idx]['file_name']
        display_images_side_by_side(selected_file)
    
    with tab2:
        # Placeholder for accuracy results (to be implemented later)
        st.write("## Accuracy Results")
        accuracy_result = accuracy(df.iloc[selected_idx])  # Placeholder for actual accuracy calculation
        st.write(accuracy_result)

# Footer with additional information and credits
st.markdown(
    """
    <footer>
    Credits: Built with <a href='https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct' target='_blank'>Meta-Llama 3</a>
    </footer>
    """, 
    unsafe_allow_html=True
)