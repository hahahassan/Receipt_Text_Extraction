# receipt_OCR_processor
import cv2
import numpy as np
import fitz  # PyMuPDF
import pytesseract
import os

def convert_pdf_to_images(pdf_path, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    pdf_document = fitz.open(pdf_path)  # Open the PDF document
    image_paths = []
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]  # Get base name of the file without extension
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)  # Load each page
        pix = page.get_pixmap()  # Get the pixmap (image representation)
        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        
        if pix.n > 3:  # Convert RGBA to RGB if needed
            img_data = cv2.cvtColor(img_data, cv2.COLOR_BGRA2BGR)
        
        # Save the image with the processed prefix
        image_path = os.path.join(output_folder, f"Processed_{base_name}_page_{page_num + 1}.png")
        cv2.imwrite(image_path, img_data)
        image_paths.append(image_path)  # Keep track of saved image paths
    return image_paths

def preprocess_image(image_path):
    image = cv2.imread(image_path)  # Read the image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)  # Apply Gaussian blur to reduce noise
    denoised_image = cv2.fastNlMeansDenoising(blurred_image, h=30)  # Denoise the image
    adaptive_thresh_image = cv2.adaptiveThreshold(denoised_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)  # Adaptive thresholding
    kernel = np.ones((3, 3), np.uint8)  # Kernel for morphological operations
    morph_image = cv2.morphologyEx(adaptive_thresh_image, cv2.MORPH_CLOSE, kernel)  # Morphological closing
    contrast_image = cv2.convertScaleAbs(morph_image, alpha=1.5, beta=0)  # Adjust contrast
    return contrast_image

def extract_text_from_images(image_paths):
    texts = []
    for image_path in image_paths:
        img = cv2.imread(image_path)  # Read the image
        text = pytesseract.image_to_string(img, config='--psm 1')  # Extract text using Tesseract OCR
        texts.append(text)
    return texts

def extract_text_from_file(file_path, output_folder="Processed_images_PDFs"):
    if file_path.lower().endswith('.pdf'):
        image_paths = convert_pdf_to_images(file_path, output_folder)  # Convert PDF to images
        texts = extract_text_from_images(image_paths)  # Extract text from images without preprocessing
    else:
        processed_image = preprocess_image(file_path)  # Preprocess the image
        base_name = os.path.splitext(os.path.basename(file_path))[0]  # Get base name of the file without extension
        processed_image_path = os.path.join(output_folder, f"Processed_{base_name}.png")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        cv2.imwrite(processed_image_path, processed_image)  # Save the processed image
        texts = [pytesseract.image_to_string(processed_image, config='--psm 1')]  # Extract text from the processed image
    return "\n".join(texts)  # Join extracted texts from all images


if __name__ == "__main__":
    file_path = r"uploaded-receipts\IMG_1305_9yP7EuA.jpg"  # Replace with your file path (PDF or image)
    extracted_text = extract_text_from_file(file_path)  # Extract text from the file
    print(extracted_text)  # Print the extracted text
    
    




