from dotenv import load_dotenv 
import os
from groq import Groq
import cv2
from faster_whisper import WhisperModel
from PIL import ImageGrab
import pyperclip
import speech_recognition as sr
import base64
from openai import OpenAI
import pyaudio
import os
import time
import re
import ollama
from gtts import gTTS
from pygame import mixer
from io import BytesIO

#------------------------------------------------------------------------------------------------

#API keys are stored in a .env file in the same directory as the script
load_dotenv()
groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
openai_client = None

#If OpenAI API key is set, use OpenAI API
if os.getenv('OPENAI_API_KEY'):
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#------------------------------------------------------------------------------------------------

#System prompt and wake word
wake_word = 'jarvis'
sys_msg = (
    'You are a multi-model AI voice assistant. Your user may or may not have attached a photo for context '
    '(either a screenshot or a webcam capture). Any photo has already been processes into a highly detailed '
    'text prompt that will be attached to their transcribe voice prompt. Generate the most useful and '
    'factual response possible, carefully considering all previous generated text in your response before '
    'adding new tokens to the response. Do not expect or request images, just use the context if added. '
    'Use all of the context of this conversation so your response is relevant to the conversation. Make '
    'your responses clear and concise, avoiding any verbosity.'
)
convo = [{"role": "system", "content": sys_msg}]

#------------------------------------------------------------------------------------------------
#Funcion Definitions

#Whisper model for transcribing audio
whisper_size = 'base'
whisper_model = WhisperModel(
    model_size_or_path=whisper_size,
    device='cpu',
    compute_type='int8'
)

#Main AI assistant function
def groq_prompt(prompt, img_context):
    if img_context:
        prompt = f'USER PROMPT: {prompt}\n\n  IMAGE CONTEXT: {img_context}'
    convo.append({"role": "user", "content": prompt})
    chat_completion = groq_client.chat.completions.create(messages=convo, model="llama3-70b-8192")
    response = chat_completion.choices[0].message
    convo.append(response)

    return response.content
    
#Function to determine which function to call based on user prompt
def function_call(prompt):
    sys_msg = (
        'You are an AI function calling model. You will determine whether extracting the users clipboard content,'
        'taking a screenshot, capturing the webcam or calling no functions is best for the voice assistant to respond'
        'to the user prompt. The webcam can be assumed to be a normal laptop webcam facing the user. Use caution '
        'when calling functions. If you are unsure it is best to output None, then provide the wrong function. YOu should '
        'respond with only one selection from this list: ["extract_clipboard", "take_screenshot", "capture_webcam", "None"] \n'
        'Do not respond with anything but the most logical selection from that list with no explanations. Fromat the '
        'function call name exactly as I listed.'
    )

    function_convo = [{"role": "system", "content": sys_msg},{"role": "user", "content": prompt}]
    chat_completion = groq_client.chat.completions.create(messages=function_convo, model="llama3-70b-8192")
    response = chat_completion.choices[0].message
    return response.content

#Function to take a screenshot
def take_screenshot():
    path = 'screenshot.png'
    screenshot = ImageGrab.grab()
    rgb_screenshot = screenshot.convert('RGB')
    rgb_screenshot.save(path, quality=15)

#Function to capture webcam image
def capture_webcam():
    web_cam = cv2.VideoCapture(0)
    if not web_cam.isOpened():
        print("Error: Could not open webcam.")
        return

    ret, frame = web_cam.read()
    if not ret:
        print("Error: Could not capture frame.")
        web_cam.release()
        return

    path = 'webcam_capture.png'
    if cv2.imwrite(path, frame):
        print(f"Image saved to  {path}")
    else:
        print("Error: Could not write image.")

    web_cam.release()

#Function to extract clipboard content
def extract_clipboard():
    clipboard_content = pyperclip.paste()
    if isinstance(clipboard_content, str):
        return clipboard_content
    else:
        print('Could not extract clipboard content.')
        return None
    
#Function to encode image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

#OpenAI Vision model for image analysis
def openai_vision_prompt(prompt, photo_path):
    base64_image = encode_image(photo_path)
    prompt=(
            'You are the vision analysis AI that provides semantic meaning from images to provide context '
            'to send to another AI that will create a response to the user. Do not respond as the AI assistant to the user. '
            'Instead take the user prompt input and try to extract all meaning from the photo '
            'relevant to the user prompt. Then generate as much objective data about the image for the AI '
            f'assistant who will respond to the user. \n USER PROMPT: {prompt}'
        )
    

    response = openai_client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": "high"
            },
            },
        ],
        }
    ],
    max_tokens=300,
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content

#Ollama Vision model for image analysis
def ollama_vision_prompt(prompt, photo_path):
    prompt=(
            'You are the vision analysis AI that provides semantic meaning from images to provide context '
            'to send to another AI that will create a response to the user. Do not respond as the AI assistant to the user. '
            'Instead take the user prompt input and try to extract all meaning from the photo '
            'relevant to the user prompt. Then generate as much objective data about the image for the AI '
            f'assistant who will respond to the user. \n USER PROMPT: {prompt}'
        )
    response = ollama.generate(
        model='llava:13b',
        prompt=prompt,
        images=[photo_path],
    )
    return response['response']

#Function to speak text using OpenAI API
def openai_speak(text):
    player_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
    stream_start = False

    with openai_client.audio.speech.with_streaming_response.create(
        model='tts-1',
        voice='nova',
        response_format='pcm',
        input=text,
    ) as response:
            silence_threshold = 0.01
            for chunk in response.iter_bytes(chunk_size=1024):
                if stream_start:
                    player_stream.write(chunk)
                else:
                    if max(chunk) > silence_threshold:
                        player_stream.write(chunk)
                        stream_start = True

#Function to speak text using gTTS
def gtts_speak(text):
    mixer.init()
    mp3_fp = BytesIO()
    tts = gTTS(text, lang='en')
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    mixer.music.load(mp3_fp, "mp3")
    mixer.music.play()

    while mixer.music.get_busy():
        time.sleep(0.1)
    
    mixer.music.stop()
    mixer.quit()

#Function to transcribe audio to text
def wav_to_text(audio_path):
    segments, _ = whisper_model.transcribe(audio_path)
    text = ' '.join([seg.text for seg in segments])
    return text

#------------------------------------------------------------------------------------------------
#Main Loop Functions

#Callback function for listening to audio and responding to user prompts
def callback(recognizer, audio):
    prompt_audio_path = 'prompt.wav'
    print('Listening...')
    with open(prompt_audio_path, 'wb') as f:
        f.write(audio.get_wav_data())  

    prompt_text = wav_to_text(prompt_audio_path)
    print(f'\nTRANSCRIBED: {prompt_text}')
    clean_prompt = extract_prompt(prompt_text, wake_word)

    if clean_prompt:
        print(f'USER: {clean_prompt}')
        call = function_call(clean_prompt)

        if 'take_screenshot' in call:
            print('Taking screenshot...')
            take_screenshot()
            if openai_client:
                visual_context = openai_vision_prompt(clean_prompt, 'screenshot.png')
            else:
                visual_context = ollama_vision_prompt(clean_prompt, 'screenshot.png')
        elif 'capture_webcam' in call:
            print('Capturing webcam...')
            capture_webcam()
            if openai_client:
                visual_context = openai_vision_prompt(clean_prompt, 'webcam_capture.png')
            else:
                visual_context = ollama_vision_prompt(clean_prompt, 'webcam_capture.png')
        elif 'extract_clipboard' in call:
            print('Extracting clipboard content...')
            paste = extract_clipboard()
            clean_prompt = f'{clean_prompt}\n\n CLIPBOARD CONTENT: {paste}'
            visual_context = None
        else:
            visual_context = None

        response = groq_prompt(clean_prompt, visual_context)
        print("\nASSISTANT: ", response)
        if openai_client:
            openai_speak(response)
        else:
            gtts_speak(response)


#Function to start listening to audio
def start_listening(source, r):
    with source as source:
        r.adjust_for_ambient_noise(source, duration=2)
    print('\nSay', wake_word, 'followed with your prompt. \n')
    r.listen_in_background(source, callback)

    while True:
        time.sleep(.3)

#Function to extract prompt from transcribed text
def extract_prompt(trascribed_text, wake_word):
    pattern = rf'\b{re.escape(wake_word)}[\s,.?!]*([A-Za-z0-9].*)'
    match = re.search(pattern, trascribed_text, re.IGNORECASE)

    if match:
        prompt = match.group(1).strip()
        return prompt
    else:
        return None


#Main function
if __name__ == '__main__':
    r = sr.Recognizer()
    source = sr.Microphone()
    start_listening(source, r)

