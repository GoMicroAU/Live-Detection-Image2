import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import cv2
from ultralytics import YOLO
import numpy as np
import PIL.Image
import io
import logging
from PIL import Image



# Set the logging level to ERROR
logging.getLogger('streamlit').setLevel(logging.ERROR)


st.set_page_config(page_title="Web App - Beta", layout="centered")


st.markdown("""
    <style>
   .stCamera {
        width: 300px;  /* Adjust the width as needed */
        height: 300px; /* Height same as width for square shape */
        object-fit: cover; /* Adjust the video to cover the area */
    }


    /* Main styles */
    .stButton>button {
        width: 100%;
        height: 50px; /* Adjust height as necessary */
    }
    
    .st-emotion-cache-1kyxreq {
    display: flex;
    flex-flow: wrap;
    row-gap: 1rem;
    justify-content: center;

}
    
    /* Custom styles for mobile */
    @media (max-width: 768px) {
        /* Centering buttons */
        .stButton {
            display: flex;
            justify-content: center;
            width: 100%;
            margin-bottom: 10px; /* Space between buttons */
        }
        /* Centering images */
        .stImage {
            display: flex;
            justify-content: center;
            width: 100%;
        }
        .stImage img {
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 80%;
        }
        .stImage img {
            max-width: 100px; /* Adjust image width as necessary */
            height: auto; /* Maintain aspect ratio */
            margin-left: auto;
            margin-right: auto;
        }
        
        .stCamera {
        width: 300px;  /* Adjust the width as needed */
        height: 300px; /* Height same as width for square shape */
        object-fit: cover; /* Adjust the video to cover the area */
    }
        
        .stVideo video {
        height: 600px;
}
        
        .st-emotion-cache-1663gsk {
  
    height: 600px;
    
}
            
        
            

            .st-emotion-cache-1erivf3 {
    
    color: rgb(255 255 255 /);
}
            
            .st-emotion-cache-7oyrr6 {
    color: rgb(250 250 250 / 0%);
    
}

.st-emotion-cache-1ec2a3d{
    color: rgb(255 255 255);
    
}
        
        
     
    </style>
""", unsafe_allow_html=True)

import cv2
import numpy as np
from PIL import Image

if 'country' not in st.session_state:
    st.session_state['country'] = None
if 'language' not in st.session_state:
    st.session_state['language'] = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = "country_language_page"


# Page functions
def country_language_page():
    st.title("Country & Language")
    st.header("Country")
    if st.button("Australia", on_click=lambda: set_country_language("Australia", "English")):
        pass
    if st.button("India", on_click=lambda: set_country_language("India", None)):
        pass

def set_country_language(country, language):
    st.session_state['country'] = country
    st.session_state['language'] = language
    if language:
        st.session_state['current_page'] = "selection_page"
    else:
        st.session_state['current_page'] = "language_page"

def language_page():
    st.title("Language")
    st.header("Language")
    if st.button("English", on_click=lambda: set_language("English")):
        pass
    if st.button("Hindi", on_click=lambda: set_language("Hindi")):
        pass

def set_language(language):
    st.session_state['language'] = language
    st.session_state['current_page'] = "selection_page"


def toImgOpenCV(imgPIL):
    """Convert PIL Image to OpenCV format."""
    return np.array(imgPIL)


def crop_image_pil(img, folder_name, string_val):
    """Crop images into a square shape."""
    width, height = img.size
    # Use the minimum of width and height to create a square
    square_side = min(width, height)
    
    # Calculate the top-left corner of the square
    left = (width - square_side) / 2
    top = (height - square_side) / 2
    right = (width + square_side) / 2
    bottom = (height + square_side) / 2
    
    # Crop the image
    cropped_img = img.crop((left, top, right, bottom))
    open_cv_image = toImgOpenCV(cropped_img)
    return open_cv_image
    
    
def handle_close_click():
    st.session_state.current_page = "selection_page"


def crop_image_extra(image_np, folder_name, row_num, col_num):
    """Crop the images further into smaller squares."""
    height, width, _ = image_np.shape
    w, h = width // int(row_num), height // int(col_num)
    cropped_images = []
    for row in range(0, width, int(w)):
        for col in range(0, height, int(h)):
            cropped_img = image_np[col:(col + int(h)), row:(row+int(w))]
            cropped_images.append(cropped_img)
    return cropped_images


def process_uploaded_images(uploaded_files, model, class_names):
    total_class_counts = {class_name: 0 for class_name in class_names.values()}
    total_images = 0

    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            # Convert the file to an image
            image_stream = io.BytesIO(uploaded_file.read())
            image = Image.open(image_stream).convert("RGB")
            image_np = np.array(image)

            # Process and display cropped images
            num_rows, num_cols = 4, 3  # Define rows and columns for cropping
            process_and_display_cropped_images(image_np, model, class_names, num_rows, num_cols)
            st.session_state.image_processed = True  # Set the flag when an image is processed

            total_images += 1
            
# Initialize the flag in session state
if 'image_processed' not in st.session_state:
    st.session_state.image_processed = False


   


def show_header(center=False):
    if center:
        # Creates three columns for centering
        col1, padding,col2,padding, col3 = st.columns([1,1,2,1,1])
        with col2:  # This is the center column
            st.image("unnamed.png", width=150)  # Adjust the width as needed
    else:
        st.image("unnamed.png", width=150)
        
        
        



# Define your pages as functions
def change_page_to_detection(selected_item):
    st.session_state.current_page = "detection_page"
    st.session_state.selected_item = selected_item



        
        
def selection_page():
    show_header(center=False)

    # Display available options
    cols = st.columns([0.5,2,2,2,0.5])
    with cols[1]:
        st.image("lentil_icon.png",width=120 )  # Replace with your lentil image path
        st.button("Lentils", on_click=lambda: change_page_to_detection("Lentils"))

        

    with cols[2]:
        st.image("grain_icon.png",width=120 )  # Replace with your coming soon image path
        st.button("Wheat", on_click=lambda: change_page_to_detection("Wheat"))

    with cols[3]:
        st.image("coffee_icon.png",width=120 )  # Use the same coming soon image path
        st.button("Coffee", on_click=lambda: change_page_to_detection("Coffee"))


    # Repeat for the second row
    cols = st.columns([0.5,2,2,2,0.5])
    with cols[1]:
        st.image("almond_icon.png", width=120)  # Update this path
        st.button("Almonds", on_click=lambda: change_page_to_detection("Almonds"))



    with cols[2]:
        st.image("chickpea_icon.png",width=120 )
        st.button("Coming Soon Chick pea", disabled=True)


    with cols[3]:
        st.image("barley_icon.png",width=120 )
        st.button("Coming Soon Barley", disabled=True)


  # Import the io module


# Load YOLO model  # Update this path
lentil_model = YOLO('lentils-model.pt')

lentil_class_names = {
    0: 'A1', 1: 'DE', 2: 'P', 3: 'SP', 4: 'CANO',
    5: 'FI', 6: 'FM', 7: 'FN', 8: 'OM', 9: 'PC',
    10: 'SD', 11: 'SN', 12: 'UM', 13: 'V',
    14: 'SC', 15: 'PCS', 16: 'FS', 17: 'ID'
}


# Load Wheat Model
  # Update this path
wheat_model = YOLO('wheat-model.pt')
# Lentil Classes

# Wheat Classes
wheat_class_names = {
    0: 'G', 1: 'DST', 2: 'SD', 3: 'SP', 4: 'FS', 5: 'WG'
}

coffee_model = YOLO('coffee-model.pt')

coffee_class_names = {
    0: 'BL', 1: 'PB', 2: 'BB', 3: 'BN', 4: 'HS',
    5: 'BK', 6: 'IM', 7: 'HO', 8: 'GB'
}

almond_model_path = 'yolo-almonds.pt'  # Update this path
almond_model = YOLO(almond_model_path)

almond_class_names = {
    0: 'CS', 1: 'D', 2: 'G', 3: 'M', 4: 'N', 5: 'S'
}




# Full name dictionary
# Full name dictionary with four main categories
full_name_dict = {
    'G':'Good Quality',
    'A1': 'Good Quality',
    'DE': 'Defective Grain',
'CANO': 'Foreign Seed',
'FI': 'Foreign Insects',
'FM': 'Foreign Material',
'FN': 'Foreign Material',
'OM': 'Objectionable Material',
'PC': 'Defective Grain',
'SD': 'Defective Grain',
'SN': 'Snails',
'UM': 'unmillable Material',
'V': 'Foreign Seed',
'FS': 'Foreign Material',
    'SC': 'Good Quality',
    'PCS': 'Defective Grain',
    'SP': 'Defective Grain',
    'CKS': 'Defective Grain',
    'P': 'Defective Grain',
    'FG/B': 'Foreign Seed',
    'FG/W': 'Foreign Seed',
    'CH': 'Defective Grain',
    'WR': 'Defective Grain',
    'S6/MM': 'Others',
    '7B/BS': 'Others',
    'RSNL': 'Others',
    'CSNL': 'Others',
    'LSC': 'Others',
    'FMP': 'Others',
    'ID': 'Others',
    'BL': 'Black Beans',
    'PB': 'Partly Black Beans',
    'BB': 'Broken Black Beans',
    'BN': 'Brown Beans',
    'HS': 'Husk Fragments',
    'BK': 'Broken Beans',
    'IM': 'Immature Beans',
    'HO': 'Beans With Holes',
    'GB': 'Good Beans',
    'CS': 'Chips Scratch',
    'D': 'Double',
    'G': 'Gum',
    'M': 'Mild',
    'N': 'Normal',
    'S': 'Stain'
}





def object_detection(image, model, class_names):
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    results = model(image_bgr)[0]
    class_counts = {class_name: 0 for class_name in class_names.values()}

    # Font settings
    font_scale = 0.7
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (255, 255, 255)  # White color

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > 0.2:
            class_name = class_names.get(int(class_id), "Unknown")
            class_counts[class_name] += 1
            
           # cv2.rectangle(image_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            
            label = f"{class_name.upper()}"

            # Calculate text size
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]

            # Calculate center position for text
            text_x = int((x1 + x2) / 2 - text_size[0] / 2)
            text_y = int((y1 + y2) / 2 + text_size[1] / 2)

            # Draw label in the middle of the box
            cv2.putText(image_bgr, label, (text_x, text_y), font, font_scale, text_color, thickness)

    processed_image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return processed_image_rgb, class_counts
    
    
def crop_smaller_center_square(img):
    """Crop a smaller square from the center of the image."""
    width, height = img.size
    new_side_length = min(width, height) / 2  # Half the size of the smaller dimension

    left = width / 2 - new_side_length / 2
    top = height / 2 - new_side_length / 2
    right = width / 2 + new_side_length / 2
    bottom = height / 2 + new_side_length / 2

    # Crop the center square
    cropped_img = img.crop((left, top, right, bottom))
    return cropped_img




    
def crop_image_to_squares(image, num_rows, num_cols):
    height, width, _ = image.shape
    cropped_images = []
    w, h = width // num_rows, height // num_cols

    for row in range(num_rows):
        for col in range(num_cols):
            start_x, start_y = row * w, col * h
            cropped_img = image[start_y:start_y + h, start_x:start_x + w]
            cropped_images.append(cropped_img)

    return cropped_images

def process_and_display_cropped_images(image, model, class_names, num_rows, num_cols):
    cropped_images = crop_image_to_squares(image, num_rows, num_cols)
    total_class_counts = {}

    for i, cropped_image in enumerate(cropped_images):
        processed_image, class_counts = object_detection(cropped_image, model, class_names)
        
        # Update total_class_counts with class_counts
        for class_name, count in class_counts.items():
            total_class_counts[class_name] = total_class_counts.get(class_name, 0) + count

        st.image(processed_image,)

    # Display summary of detected categories
    display_detected_categories_summary(total_class_counts)

def display_detected_categories_summary(class_counts):
    # Define the order of categories
    ordered_categories = [
    "Good Quality",
    "Defective Grain",
    "Foreign Seed",
    "Others",
    "Foreign Insects",
    "Foreign Material",
    "Objectionable Material",
    "Snails",
    "unmillable Material",
    "Black Beans",
    "Partly Black Beans",
    "Broken Black Beans",
    "Brown Beans",
    "Husk Fragments",
    "Broken Beans",
    "Immature Beans",
    "Beans With Holes",
    "Good Beans",
    "Chips Scratch",
        "Double",
        "Gum",
        "Mild",
        "Normal",
        "Stain"
]
    # Initialize a dictionary to hold counts for each ordered category
    ordered_category_counts = {category: 0 for category in ordered_categories}

    # Aggregate counts into the defined categories
    for class_name, count in class_counts.items():
        category_name = full_name_dict.get(class_name, "Others")
        ordered_category_counts[category_name] += count

    # Display results in the specified order
    for category in ordered_categories:
        if ordered_category_counts[category] > 0:
            st.write(f"{category}: {ordered_category_counts[category]}")

    
import streamlit as st
import streamlit.components.v1 as components



# Initialize session state variables
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = []

# Set default detection mode to "Camera Upload"
if 'detection_mode' not in st.session_state:
    st.session_state.detection_mode = "Camera Upload"   


RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

def detection_page():
    if 'selected_item' in st.session_state:
        if st.session_state['selected_item'] == 'Lentils':
            current_model = lentil_model
            current_class_names = lentil_class_names
        elif st.session_state['selected_item'] == 'Wheat':
            current_model = wheat_model
            current_class_names = wheat_class_names
        elif st.session_state['selected_item'] == 'Coffee':
            current_model = coffee_model
            current_class_names = coffee_class_names
        elif st.session_state['selected_item'] == 'Almonds':  # Add condition for Almonds
            current_model = almond_model
            current_class_names = almond_class_names
        else:
            st.error("Please select an item.")
            return
        
        

    
    

    def camera_upload():
        st.write("Place grains in A4 sheet")

        total_class_counts = {class_name: 0 for class_name in current_class_names.values()}
        total_images = 0

    # Replace video capture with file uploader
        uploaded_files = st.file_uploader("For indoor use", accept_multiple_files=True,)

        if 'selected_item' in st.session_state:
        # [Existing logic to determine the model based on selected item]

            if uploaded_files:
                process_uploaded_images(uploaded_files, current_model, current_class_names)


        for processed_image, _ in st.session_state.processed_images:
            st.image(processed_image, caption=f'Image {len(st.session_state.processed_images)}')
    # Implement the functionality for Camera Upload

    def big_assessor():
        st.write("For use with GoMicro Assessor")
        uploaded_files = st.file_uploader("Place grains in tray", accept_multiple_files=True, type=["jpg", "jpeg"])

        total_class_counts = {class_name: 0 for class_name in current_class_names.values()}  # Initialize total counts

        if uploaded_files:
            for uploaded_file in uploaded_files:
            # Read and process the original image
                img = Image.open(uploaded_file)
                img_np = np.array(img)  # Convert to numpy array for processing
                processed_image, class_counts = object_detection(img_np, current_model, current_class_names)
                st.image(img, caption='Assessor Image', use_column_width=True)

            # Update total class counts with current image counts
                for class_name, count in class_counts.items():
                    total_class_counts[class_name] += count

            # Crop the original image
                cropped_image = crop_image_pil(img, "folder_name", "s")  # Assuming square cropping
                cropped_image_np = np.array(cropped_image)

            # Process further cropped images
                row_num = 3
                col_num = 3
                further_cropped_images = crop_image_extra(cropped_image_np, "folder_name", row_num, col_num)

                for i, img in enumerate(further_cropped_images):
                # Apply detection to each further cropped image
                    processed_sub_image, class_counts = object_detection(img, current_model, current_class_names)
                    st.image(processed_sub_image, use_column_width=True)

                # Update total class counts again
                    for class_name, count in class_counts.items():
                        total_class_counts[class_name] += count

                    st.session_state.image_processed = True    

        # Display summary of detected categories
            display_detected_categories_summary(total_class_counts)

        
    # Implement the functionality for Big Assessor

    def small_assessor():
        st.write("Upload images using assessor")
        uploaded_files = st.file_uploader("", accept_multiple_files=True, type=["jpg", "jpeg"], label_visibility="hidden")

        total_class_counts = {class_name: 0 for class_name in current_class_names.values()}  # Initialize total counts

        if uploaded_files:
            for uploaded_file in uploaded_files:
                img = Image.open(uploaded_file).convert("RGB")

            # Crop a smaller center square of the original image
                cropped_image = crop_smaller_center_square(img)
                cropped_image_np = np.array(cropped_image)

            # Process the cropped image
                processed_image, class_counts = object_detection(cropped_image_np, current_model, current_class_names)
                st.image(processed_image, use_column_width=True)

            # Update total class counts with current image counts
                for class_name, count in class_counts.items():
                    total_class_counts[class_name] += count

                st.session_state.image_processed = True    

    # Display summary of detected categories
        display_detected_categories_summary(total_class_counts)

        





 
        
    with st.sidebar:
        st.title("Settings")
        st.session_state.detection_mode = st.selectbox("Choose Mode", ["Phone", "Assessor", "Spot check"])

    # Functionality based on chosen mode
    if st.session_state.detection_mode == "Phone":
        camera_upload()
    elif st.session_state.detection_mode == "Assessor":
        big_assessor()
    elif st.session_state.detection_mode == "Spot check":
        small_assessor()
        
        
    if st.session_state.image_processed:
        st.write("### You can perform the QC process again with a new image if needed.")
        
        
    # Create a row with two columns
    col1, col2 = st.columns(2)

    # Add the Close button in the first column
    with col1:
        if st.button("Close", on_click=handle_close_click):
            # This button will now call handle_close_click when clicked
            pass

    # Add the Register button in the second column
    with col2:
        if st.session_state.image_processed:
            st.markdown("<a href='https://www.gomicro.co/register/' target='_blank'><button style='width: 100%; color: white; background-color: rgb(33, 75, 65); padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer;'>Register</button></a>", unsafe_allow_html=True)




# Page navigation and session state code...

if st.session_state.current_page == "country_language_page":
    country_language_page()
elif st.session_state.current_page == "language_page":
    language_page()
elif st.session_state.current_page == "selection_page":
    selection_page()
elif st.session_state.current_page == "detection_page":
    detection_page()
