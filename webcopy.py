import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import io
import logging
from PIL import Image
import pandas as pd




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

if 'current_page' not in st.session_state:
    st.session_state.current_page = "selection_page"

if 'total_class_counts' not in st.session_state:
    st.session_state.total_class_counts = {}


def detect_circle_and_crop(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    rows = gray.shape[0]

    # Detect circles
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,
                               param1=100, param2=30,
                               minRadius=0, maxRadius=0)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])  # center of the circle
            radius = i[2]  # radius of the circle

            # Assuming the square to be half the diameter of the circle
            half_side = radius // 2
            x1, y1 = center[0] - half_side, center[1] - half_side
            x2, y2 = center[0] + half_side, center[1] + half_side

            # Crop to square
            cropped_img = img[y1:y2, x1:x2]
            return cropped_img

    return None  # Return None if no circle is detected


def remove_anomalous_detections(detections, anomalous_threshold=0.1):
    """
    Remove detections that are too large or too small compared to the average size.
    :param detections: List of detections.
    :param anomalous_threshold: Threshold to determine anomalous detections.
    :return: Filtered list of detections.
    """
    areas = [((det['box'][2] - det['box'][0]) * (det['box'][3] - det['box'][1])) for det in detections]
    average_area = np.mean(areas)
    filtered_detections = [det for det in detections if average_area * anomalous_threshold < areas[i] < average_area / anomalous_threshold for i, det in enumerate(detections)]
    return filtered_detections

def combine_detections(detections, combine_threshold=0.5):
    """
    Combine detections that are close to each other or overlapping.
    :param detections: List of detections.
    :param combine_threshold: Threshold to determine when to combine detections.
    :return: List of combined detections.
    """
    combined_detections = []
    used = set()
    for i, det1 in enumerate(detections):
        if i in used:
            continue
        for j, det2 in enumerate(detections):
            if j in used or i == j:
                continue
            if is_overlap(det1['box'], det2['box']) > combine_threshold:
                used.add(j)
                # Combine det1 and det2. For simplicity, we take the average.
                combined_box = [(det1['box'][k] + det2['box'][k]) / 2 for k in range(4)]
                det1['box'] = combined_box
        combined_detections.append(det1)
    return combined_detections


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

#def display_counts_and_percentages(class_counts):
#    total_count = sum(class_counts.values())
#    if total_count > 0:
#        st.write("### Class Counts and Percentages")
#        for class_name, count in class_counts.items():
#            percentage = (count / total_count) * 100
#            st.write(f"{class_name}: {count} ({percentage:.2f}%)")
#    else:
#        st.write("No items detected.")   




def display_class_details(detection_counts, selected_item='Lentils'):
    if sum(detection_counts.values()) > 0:
        total_count = sum(detection_counts.values())
        data = []  # List to store row data

        # Collect data
        for class_id, count in detection_counts.items():
            class_description = class_descriptions[selected_item].get(class_id, "Description not available")
            percentage = (count / total_count) * 100
            data.append({"Class ID": class_id, "Description": class_description, "Count": count, "Percentage": f"{percentage:.1f}%"})

        # Create DataFrame
        df = pd.DataFrame(data)

        # Display as a table
        st.table(df)
    else:
        st.write("No detections found.")



     


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


def crop_image_to_square(image_np, row_index, col_index, row_num, col_num):
    """
    Crop the image into a square based on row and column index.
    :param image_np: Numpy array of the image.
    :param row_index: Row index of the square.
    :param col_index: Column index of the square.
    :param row_num: Total number of rows in the grid.
    :param col_num: Total number of columns in the grid.
    :return: Cropped image as a numpy array.
    """
    height, width, _ = image_np.shape
    square_height = height // row_num
    square_width = width // col_num
    y1 = square_height * row_index
    y2 = y1 + square_height
    x1 = square_width * col_index
    x2 = x1 + square_width
    return image_np[y1:y2, x1:x2]

def crop_further(image_np, row_index, col_index, row_num, col_num):
    """
    Further crop the image into smaller squares.
    :param image_np: Numpy array of the already cropped image.
    :param row_index: Row index of the square to crop.
    :param col_index: Column index of the square to crop.
    :param row_num: Total number of rows in the grid for further cropping.
    :param col_num: Total number of columns in the grid for further cropping.
    :return: Further cropped image as a numpy array.
    """
    height, width, _ = image_np.shape
    square_height = height // row_num
    square_width = width // col_num
    y1 = square_height * row_index
    y2 = y1 + square_height
    x1 = square_width * col_index
    x2 = x1 + square_width
    return image_np[y1:y2, x1:x2]



def adjust_detections(detections, row_index, col_index, row_num, col_num, image_height, image_width):
    square_height = image_height // row_num
    square_width = image_width // col_num
    offset_y = square_height * row_index
    offset_x = square_width * col_index

    adjusted_detections = []
    for det in detections:
        # Check if the detection format is correct
        if 'box' in det and isinstance(det['box'], (list, tuple)) and len(det['box']) == 4:
            x1, y1, x2, y2 = det['box']
            adjusted_box = [x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y]
            adjusted_detections.append({
                'box': adjusted_box, 
                'score': det['score'], 
                'class_id': det['class_id']
            })
        else:
            print("Invalid detection format or missing 'box' key:", det)

    return adjusted_detections

def remove_overlap_detections(detections, overlap_threshold):
    """
    Remove overlapping detections based on the specified threshold.
    :param detections: List of detections.
    :param overlap_threshold: Threshold for overlap.
    :return: List of detections after removing overlaps.
    """
    if not detections:
        return []

    # Sort detections based on confidence score
    detections.sort(key=lambda x: x['score'], reverse=True)

    final_detections = []
    while detections:
        # Take the detection with the highest score
        current = detections.pop(0)
        final_detections.append(current)

        # Compute overlap with the remaining detections and remove if necessary
        detections = [det for det in detections if not is_overlap(current['box'], det['box'], overlap_threshold)]

    return final_detections

def is_overlap(boxA, boxB, threshold):
    """
    Check if two boxes overlap more than the given threshold.
    :param boxA: First bounding box.
    :param boxB: Second bounding box.
    :param threshold: Overlap threshold.
    :return: Boolean indicating if boxes overlap more than the threshold.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou > threshold




def process_uploaded_images(uploaded_files, model, class_names):
    st.session_state.total_class_counts = {class_name: 0 for class_name in class_names.values()}
    
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            # Existing code to process the image...

            all_detections = []
            for i in range(num_rows):
                for j in range(num_cols):
                    # Existing code for cropping and detecting...

                    # Adjust detections to global coordinates
                    detections = adjust_detections(detections, i, j, num_rows, num_cols)
                    all_detections.extend(detections)

            # Remove anomalous detections
            all_detections = remove_anomalous_detections(all_detections, 0.2)

            # Combine close detections
            all_detections = combine_detections(all_detections, 0.8)

            # Remove overlapping detections
            final_detections = remove_overlap_detections(all_detections, 0.8)


            # Count the final detections
            for detection in final_detections:
                class_name = detection['class_name']
                st.session_state.total_class_counts[class_name] += 1

            st.session_state.image_processed = True
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

    # Update the class names for the selected item
    if selected_item == 'Lentils':
        st.session_state.selected_item_class_names = lentil_class_names
    elif selected_item == 'Wheat':
        st.session_state.selected_item_class_names = wheat_class_names
    elif selected_item == 'Coffee':
        st.session_state.selected_item_class_names = coffee_class_names
    elif selected_item == 'Almonds':
        st.session_state.selected_item_class_names = almond_class_names
    elif selected_item == 'Soy':
        st.session_state.selected_item_class_names = soy_class_names
    elif selected_item == 'Corn':
        st.session_state.selected_item_class_names = corn_class_names
    elif selected_item == 'Rice':
        st.session_state.selected_item_class_names = rice_class_names
    elif selected_item == 'Chickpeas':
        st.session_state.selected_item_class_names = chickpea_class_names        
    # Add similar lines for any other items you have




        
        
def selection_page():

    
    show_header(center=False)

    st.markdown("<h2 style='text-align: center; color: grey; font-weight: bold; margin-bottom: 20px'>Demo Defect Count</h2>", unsafe_allow_html=True)
    

    # Display available options
    cols = st.columns([0.5,2,2,2,2,0.5])
    with cols[1]:
        st.image("lentil_icon.png",width=120 )  # Replace with your lentil image path
        st.button("Lentils: Australia/India", on_click=lambda: change_page_to_detection("Lentils"))

        

    with cols[2]:
        st.image("grain_icon.png",width=120 )  
        st.button("Wheat: Australia", on_click=lambda: change_page_to_detection("Wheat"))

    with cols[3]:
        st.image("coffee_icon.png",width=120 )  
        st.button("Coffee: Indonesia", on_click=lambda: change_page_to_detection("Coffee"))

    with cols[4]:
        st.image("rice_icon.png",width=120 )  
        st.button("Rice: India", on_click=lambda: change_page_to_detection("Rice"))    


    # Repeat for the second row
    cols = st.columns([0.5,2,2,2,2,0.5])
    with cols[1]:
        st.image("almond_icon.png", width=120)  
        st.button("Almonds: Australia/US", on_click=lambda: change_page_to_detection("Almonds"))



    with cols[2]:
        st.image("chickpea_icon.png",width=120 )
        st.button("Soy: Australia", on_click=lambda: change_page_to_detection("Soy"))


    with cols[3]:
        st.image("corn_icon.png",width=120 )
        st.button("Corn: Indonesia", on_click=lambda: change_page_to_detection("Corn"))

    with cols[4]:
        st.image("chickpea_icon.png",width=120 )  
        st.button("Chickpeas: Australia/India", on_click=lambda: change_page_to_detection("Chickpeas"))    


  # Import the io module


# Load YOLO model
model = YOLO('yolov8n.pt') 
lentil_model = YOLO('lentils-model.pt')

lentil_class_names = {
    0: 'A1',
    1: 'DE',
    2: 'P',
    3: 'CANO',
    4: 'FI',
    5: 'FM',
    6: 'FN',
    7: 'OB',
    8: 'PC',
    9: 'SD',
    10: 'SN',
    11: 'UM',
    12: 'V',
    13: 'SC',
    14: 'PCS',
    15: 'FS',
    16: 'ID'
}



# Load Wheat Model
wheat_model = YOLO('wheat-model.pt')
# Lentil Classes

# Wheat Classes
wheat_class_names = {
    0: 'G', 1: 'DST', 2: 'SD', 3: 'SP', 4: 'FS', 5: 'WG'
}

coffee_model = YOLO('coffee-model.pt')

coffee_class_names = {
    0: 'BL',
    1: 'PB',
    2: 'BB',
    3: 'BN',
    4: 'HS',
    5: 'BK',
    6: 'IM',
    7: 'HO',
    8: 'GB'
}


almond_model = YOLO('yolo-almonds.pt')

almond_class_names = {
    0: 'CS', 1: 'D', 2: 'G', 3: 'M', 4: 'N', 5: 'S'
}

soy_model = YOLO('soy-model.pt')
corn_model = YOLO('model-corn.pt')

# Soy class names
soy_class_names = {
    0: 'N',  # Normal
    1: 'P',  # Purple
    2: 'G',  # Green
    3: 'D',  # Damaged
    4: 'R'   # Dirty
}

# Corn class names
corn_class_names = {
    0: 'G',   # Good
    1: 'B',   # Broken
    2: 'D',   # Damaged
    3: 'V',   # Weeviled
    4: 'F',   # Fungus
    5: 'FM1'  # Foreign Matter
}

# Load new models
rice_model = YOLO('rice-model.pt')
chickpea_model = YOLO('chickpea-model.pt')

# Soy class names
rice_class_names = {
    0: 'G',  # Normal
    1: 'B',  # Purple
    2: 'D'  # Green
}

# Corn class names
chickpea_class_names = {
    0: 'DG_BR',   # Good
    1: 'DG_CH',   # Broken
    2: 'DG_ID',   # Damaged
    3: 'DG_MS',   # Weeviled
    4: 'DG_S',   # Fungus
    5: 'DG_SD',
    6: 'DG_SP',   # Good
    7: 'DG_WR',   # Broken
    8: 'G',   # Damaged
    9: 'PC_G',   # Weeviled
    10: 'PC_PC'
}




category_dicts = {
    'Lentils': {
        'A1': 'Good Quality',  # Add category for 'A1'
        'DE': 'Defective Grain',  # Add category for 'DE'
        'P': 'Defective Grain',   # Add category for 'P'
        'CANO': 'Foreign Seed', # Add category for 'CANO'
        'FI': 'Foreign Insects',   # Add category for 'FI'
        'FM': 'Foreign Material',   # Add category for 'FM'
        'FN': 'Foreign Material',   # Add category for 'FN'
        'OB': 'Objectionable Material',   # Add category for 'OB'
        'PC': 'Defective Grain',   # Add category for 'PC'
        'SD': 'Defective Grain',   # Add category for 'SD'
        'SN': 'Snails',   # Add category for 'SN'
        'UM': 'Unmillable Material',   # Add category for 'UM'
        'V': 'Foreign Seed',    # Add category for 'V'
        'SC': 'Good Quality',   # Add category for 'SC'
        'PCS': 'Defective Grain',  # Add category for 'PCS'
        'FS': 'Foreign Material',   # Add category for 'FS'
        'ID': 'Others'    # Add category for 'ID'
    },
    'Coffee': {
        'BL': 'Defective',
        'PB': 'Defective',
        'BB': 'Defective',
        'BN': 'Defective',
        'HS': 'Defective',
        'BK': 'Defective',
        'IM': 'Defective',
        'HO': 'Defective',
        'GB': 'Good Beans'
    },
    'Soy': {
        'N': 'Normal',
        'P': 'Defective',
        'G': 'Defective',
        'D': 'Defective',
        'R': 'Defective'
    },
    'Corn': {
        'G': 'Good',
        'B': 'Defective',
        'D': 'Defective',
        'V': 'Defective',
        'F': 'Defective',
        'FM1': 'Defective'
    },
    'Almonds': {
        'CS': 'Defective',
        'D': 'Defective',
        'G': 'Defective',
        'M': 'Defective',
        'N': 'Normal',
        'S': 'Defective'
    },
    'Wheat': {
        'G': 'Good Quality',
        'DST': 'Defective Grain',
        'SD': 'Defective Grain',
        'SP': 'Defective Grain',
        'FS': 'Foreign Material',
        'WG': 'Good Quality'
    },
    'Rice': {
        'G': 'Good Quality',
        'B': 'Defective Grain',
        'D': 'Defective Grain'
    },
    'Chickpeas': {
        'DG_BR': 'Defective Grain',
        'DG_CH': 'Defective Grain',
        'DG_ID': 'Defective Grain',
        'DG_MS': 'Defective Grain',
        'DG_S': 'Defective Grain',
        'DG_SD': 'Defective Grain',
        'DG_SP': 'Defective Grain',
        'G': 'Good Quality',
        'PC_G': 'Defective Grain',
        'PC_PC': 'Defective Grain',


    },
}




# Full name dictionary
# Full name dictionary with four main categories
class_descriptions = {
    'Lentils': {
        'A1': 'Good',
        'DE': 'Defective',
        'P': 'Foreign Material',
        'CANO': 'Foreign Seeds',
        'FI': 'Field Insects',
        'FM': 'Foreign Material',
        'FN': 'Foreign Material',
        'OB': 'Objectionable Material',
        'PC': 'Poor Colour Seed',
        'SD': 'Severely Damaged',
        'SN': 'Snails',
        'UM': 'Unmillable Material',
        'V': 'Foreign Seeds',
        'SC': 'Defective',
        'PCS': 'Poor Colour Seed',
        'FS': 'Foreign Material',
        'ID': 'Defective'
    },
    'Coffee': {
        'BL': 'Black Beans',
        'PB': 'Partly Black Beans',
        'BB': 'Broken Black Beans',
        'BN': 'Brown Beans',
        'HS': 'Husk Fragments',
        'BK': 'Broken Beans',
        'IM': 'Immature Beans',
        'HO': 'Beans With Holes',
        'GB': 'Good Beans'
    },
    'Soy': {
        'N': 'Normal',
        'P': 'Purple',
        'G': 'Green',
        'D': 'Damaged',
        'R': 'Dirty'
    },
    'Corn': {
        'G': 'Good',
        'B': 'Broken',
        'D': 'Damaged',
        'V': 'Weeviled',
        'F': 'Fungus',
        'FM1': 'Foreign Matter'
    },
    'Almonds': {
        'CS': 'Chips Scratch',
        'D': 'Double',
        'G': 'Hum',
        'M': 'Mild',
        'N': 'Normal',
        'S': 'Stain'
    },
    'Wheat': {
        'G': 'Good Quality',
        'DST': 'Defective Grain',
        'SD': 'Defective Grain',
        'SP': 'Defective Grain',
        'FS': 'Foreign Material',
        'WG': 'Good Quality'
    },
    'Rice': {
        'G': 'Good',
        'B': 'Broken',
        'D': 'Discoloured'
    },
    'Chickpeas': {
        'DG_BR': 'Broken',
        'DG_CH': 'Chipped',
        'DG_ID': 'Insect Damage',
        'DG_MS': 'Missing Seed Coat',
        'DG_S': 'Split',
        'DG_SD': 'Skin Damage',
        'DG_SP': 'Sprouted',
        'G': 'Good',
        'PC_G': 'Green',
        'PC_PC': 'Poor Colour',


    },
}


def reset_class_counts(class_names):
    st.session_state.total_class_counts = {class_name: 0 for class_name in class_names.values()}


def object_detection(image, model, class_names):
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    results = model(image_bgr)[0]
    class_counts = {class_name: 0 for class_name in class_names.values()}

    font_scale = 0.7
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (255, 255, 255)

    detections = []
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > 0.2:
            # Ensure class_id is an integer
            class_id_int = int(class_id)

            # Get the class name using class_id
            class_name = class_names.get(class_id_int, "Unknown")
            class_counts[class_name] += 1

            detections.append({
                'box': [int(x1), int(y1), int(x2), int(y2)],
                'score': score,
                'class_id': class_id_int  # Store as an integer
            })
            
            label = f"{class_name.upper()}" 

            # Calculate text size
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]

            # Calculate center position for text
            text_x = int((x1 + x2) / 2 - text_size[0] / 2)
            text_y = int((y1 + y2) / 2 + text_size[1] / 2)

            # Draw label in the middle of the box
            cv2.putText(image_bgr, label, (text_x, text_y), font, font_scale, text_color, thickness)

    processed_image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return processed_image_rgb, detections
    
    
def crop_center_square(img, size=1280):
    """Crop a centered square from the image with a specific size."""
    width, height = img.size

    # Calculate the top-left corner of the square
    left = (width - size) / 2
    top = (height - size) / 2
    right = (width + size) / 2
    bottom = (height + size) / 2

    # Ensuring the crop area is within the image
    left, top, right, bottom = map(lambda x: max(0, x), [left, top, right, bottom])

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

        st.session_state.total_class_counts = total_class_counts    

    # Display summary of detected categories
    display_detected_categories_summary(total_class_counts)

    # Return the total_class_counts
    return total_class_counts


def show_class_details():
    if sum(st.session_state.total_class_counts.values()) > 0:
        st.write("### Class Details: Label, Count, and Percentage")
        total_count = sum(st.session_state.total_class_counts.values())

        for class_id, count in st.session_state.total_class_counts.items():
            # Get the class label from selected_item_class_names
            class_label = st.session_state.selected_item_class_names.get(class_id, "Unknown")

            # Calculate percentage
            percentage = (count / total_count) * 100

            # Display label, count, and percentage
            st.write(f"Label: {class_label}, Count: {count}, Percentage: {percentage:.2f}%")
    else:
        st.write("No items detected.")






def display_detected_categories_summary(class_counts, selected_item):
    if selected_item not in category_dicts:
        st.error("Selected item category dictionary not found.")
        return

    category_dict = category_dicts[selected_item]

    # Initialize a dictionary to hold counts for each category
    category_counts = {category: 0 for category in category_dict.values()}
    category_counts['Others'] = 0  # Add this line to initialize 'Others'

    # Aggregate counts into the defined categories
    for class_id, count in class_counts.items():
        category_name = category_dict.get(class_id, "Others")
        category_counts[category_name] += count

    # Display results
    for category, count in category_counts.items():
        if count > 0:
            st.write(f"{category}: {count}")


    
import streamlit.components.v1 as components



# Initialize session state variables
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = []

# Set default detection mode to "Camera Upload"
if 'detection_mode' not in st.session_state:
    st.session_state.detection_mode = "Camera Upload"   


if 'selected_item_class_names' not in st.session_state:
    st.session_state.selected_item_class_names = {}




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
        elif st.session_state['selected_item'] == 'Soy':
            current_model = soy_model
            current_class_names = soy_class_names
        elif st.session_state['selected_item'] == 'Corn':
            current_model = corn_model
            current_class_names = corn_class_names
        elif st.session_state['selected_item'] == 'Rice':
            current_model = rice_model
            current_class_names = rice_class_names
        elif st.session_state['selected_item'] == 'Chickpeas':
            current_model = chickpea_model
            current_class_names = chickpea_class_names            
        else:
            st.error("Please select an item.")
            return
        
        

    

    def camera_upload():
        st.write("Place grains in A4 sheet")

        reset_class_counts(current_class_names)

        total_class_counts = {class_name: 0 for class_name in current_class_names.values()}

        uploaded_files = st.file_uploader("For indoor use", accept_multiple_files=True)

        if uploaded_files:
            for uploaded_file in uploaded_files:
                img = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
                img_np = np.array(img)

            # Assuming you want to crop the original image into 4x3 grid
                row_num, col_num = 4, 3
                all_detections = []

                for i in range(row_num):
                    for j in range(col_num):
                    # Crop the image
                        cropped_image_np = crop_image_to_square(img_np, i, j, row_num, col_num)

                    # Perform detection on the cropped image
                        processed_image, detections = object_detection(cropped_image_np, current_model, current_class_names)

                    # Display each cropped image with detections
                        st.image(processed_image, caption=f"Cropped Image {i},{j}")

                    # Aggregate detections
                        for detection in detections:
                            class_name = current_class_names.get(detection['class_id'], "Unknown")
                            total_class_counts[class_name] += 1

            # Update session state after processing each uploaded file
                st.session_state.total_class_counts = total_class_counts
                st.session_state.image_processed = True

    # Display summary of detected categories if any image has been processed
        if st.session_state.image_processed:
            display_detected_categories_summary(st.session_state.total_class_counts, st.session_state.selected_item)

    # Implement the functionality for Camera Upload

    def big_assessor():
        st.write("For use with GoMicro Assessor")
        reset_class_counts(current_class_names)
        uploaded_files = st.file_uploader("Place grains in tray", accept_multiple_files=True, type=["jpg", "jpeg"])

        local_total_class_counts = {class_name: 0 for class_name in current_class_names.values()}

        if uploaded_files:
            for uploaded_file in uploaded_files:
                img = Image.open(uploaded_file)
                img_np = np.array(img)

            # Assuming you want to crop the original image into 3x3 grid
                row_num, col_num = 3, 3
                all_detections = []

                for i in range(row_num):
                    for j in range(col_num):
                    # Crop the image
                        cropped_image_np = crop_image_to_square(img_np, i, j, row_num, col_num)

                    # Perform detection on the cropped image
                        processed_image, detections = object_detection(cropped_image_np, current_model, current_class_names)

                    # Display each cropped image with detections
                        st.image(processed_image, caption=f"Cropped Image {i},{j}")

                    # Aggregate detections
                        for detection in detections:
                            class_name = current_class_names.get(detection['class_id'], "Unknown")
                            local_total_class_counts[class_name] += 1

            # Update session state after processing each uploaded file
                st.session_state.total_class_counts = local_total_class_counts
                st.session_state.image_processed = True

    # Display summary of detected categories if any image has been processed
        if st.session_state.image_processed:
            display_detected_categories_summary(st.session_state.total_class_counts, st.session_state.selected_item)

        # Display summary of detected categories

        
    # Implement the functionality for Big Assessor

    def small_assessor():
        st.write("Turn on the flashlight while clicking picture for Spot Check")
        reset_class_counts(current_class_names)
        uploaded_files = st.file_uploader("", accept_multiple_files=True, type=["jpg", "jpeg"], label_visibility="hidden")

        total_class_counts = {class_name: 0 for class_name in current_class_names.values()}

        if uploaded_files:
            for uploaded_file in uploaded_files:
                img = Image.open(uploaded_file).convert("RGB")

            # Crop center square
                cropped_img = crop_center_square(img)
                img_np = np.array(cropped_img)

            # Process the cropped image
                processed_image, detections = object_detection(img_np, current_model, current_class_names)
                st.image(processed_image, use_column_width=True)

            # Update total class counts with current image counts
                for detection in detections:
                    class_name = current_class_names.get(detection['class_id'], "Unknown")
                    total_class_counts[class_name] += 1

                st.session_state.total_class_counts = total_class_counts
                st.session_state.image_processed = True

    # Display summary of detected categories if any image has been processed
        if st.session_state.image_processed:
            display_detected_categories_summary(st.session_state.total_class_counts, st.session_state.selected_item)


        


   


 
        
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


    if st.session_state.image_processed:
        if st.button("Show Class Details"):
            display_class_details(st.session_state.total_class_counts, st.session_state.selected_item)

    
          




# Page navigation and session state code...

# Page navigation and session state code...
if st.session_state.current_page == "selection_page":
    selection_page()
elif st.session_state.current_page == "detection_page":
    detection_page()

