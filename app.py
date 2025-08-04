import cvzone
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
from PIL import Image
import streamlit as st
import pyttsx3
import pytesseract
from io import BytesIO

# Set the Tesseract path explicitly
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Streamlit setup
st.set_page_config(layout="wide")
st.image('C:/Users/pc/Downloads/MathGestures.png')

# Layout for Streamlit app
col1, col2 = st.columns([3, 2])
with col1:
run = st.checkbox('Run', value=True)
FRAME_WINDOW = st.image([])

with col2:
st.title("Math Problem Solver")
math_input = st.text_input("Enter a math problem:")
output_text_area = st.subheader("")

# File uploader for image input
uploaded_image = st.file_uploader("Upload an image with a math problem", type=["jpg", "jpeg", "png"])

# Google AI configuration
genai.configure(api_key="API-KEY")
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize the webcam to capture video
cap = cv2.VideoCapture(0) # Change this to '1' or '2' if you're using a different camera
cap.set(3, 1280)
cap.set(4, 720)

# Initialize the HandDetector class for hand tracking
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

# Function to get hand information
def getHandInfo(img):
if img is None or img.size == 0:
print("Error: Empty image passed to hand detection.")
return None

hands, img = detector.findHands(img, draw=False, flipType=True)

if hands:
hand = hands[0]
lmList = hand["lmList"]
fingers = detector.fingersUp(hand)
return fingers, lmList
else:
return None

# Function to draw on the canvas
def draw(info, prev_pos, canvas):
fingers, lmList = info
current_pos = None
if fingers == [0, 1, 0, 0, 0]: # If index finger is up
current_pos = lmList[8][0:2] # Get position of the tip of the index finger
if prev_pos is None:
prev_pos = current_pos
cv2.line(canvas, current_pos, prev_pos, (255, 0, 255), 10)
elif fingers == [1, 0, 0, 0, 0]: # If no finger is up (reset)
canvas = np.zeros_like(img)

return current_pos, canvas

# Function to send canvas to AI for solving math problems
def sendToAI(model, math_input=None, canvas=None, fingers=None):
if math_input: # If a math problem is entered in the text box
response = model.generate_content([f"Solve this math problem: {math_input}"])
return response.text
if fingers == [1, 1, 1, 1, 0]: # If all fingers are up except the pinky
pil_image = Image.fromarray(canvas)
response = model.generate_content(["Solve this math problem", pil_image])
return response.text

# Function to speak the answer using text-to-speech
def speak_answer(answer):
engine = pyttsx3.init()
engine.say(answer)
engine.runAndWait()

# Function to extract text from an uploaded image (OCR)
def extract_text_from_image(image):
img = Image.open(image)
text = pytesseract.image_to_string(img)
return text

# Main loop to process webcam frames and gestures
prev_pos = None
canvas = None
image_combined = None
output_text = ""

# Continuously get frames from the webcam
while True:
# Capture each frame from the webcam
success, img = cap.read()
if not success:
print("Error: Failed to capture image from webcam.")
break

img = cv2.flip(img, 1)

# Initialize canvas if not already initialized
if canvas is None:
canvas = np.zeros_like(img)

# Get hand info
info = getHandInfo(img)
if info:
fingers, lmList = info
prev_pos, canvas = draw(info, prev_pos, canvas)
output_text = sendToAI(model, canvas=canvas, fingers=fingers)

# Combine the webcam frame with the canvas for drawing
image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
FRAME_WINDOW.image(image_combined, channels="BGR")

# Display AI response in Streamlit
if output_text:
output_text_area.text(output_text)
speak_answer(output_text) # Speak the output

# Check if user has entered a math problem in the text box
if math_input:
output_text = sendToAI(model, math_input=math_input) # Send the input math problem
output_text_area.text(output_text)
speak_answer(output_text) # Speak the output

# Check if user has uploaded an image with a math problem
if uploaded_image:
extracted_text = extract_text_from_image(uploaded_image)
output_text = sendToAI(model, math_input=extracted_text)
output_text_area.text(output_text)
speak_answer(output_text) # Speak the output

# Wait for the next frame
cv2.waitKey(1)

# Release the webcam capture object when done
cap.release()
