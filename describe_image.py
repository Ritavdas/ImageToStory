import os

import requests
import streamlit as st
from dotenv import find_dotenv, load_dotenv
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from transformers import pipeline

load_dotenv(find_dotenv())
HUGGINGFACE_API_KEY = os.getenv("HF_API_KEY")


# image to text
def describe_image(image_url):
    """
    Describes an image using a pre-trained image captioning model.

    Args:
        image_url (str): The URL of the image to be described.

    Returns:
        str: The description of the image.
    """
    captioner = pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base",
    )
    return captioner(image_url)


# llm
def generate_story(scenario):
    """
    Generate a short story based on a simple narrative.

    Args:
        scenario (str): The narrative scenario.

    Returns:
        str: The generated story.

    """
    template = """
        You are a story teller. You can generate a short story based on a simple narrative, The story should not be more than 50 words.
        CONTEXT: {scenario}
        STORY:
    """
    prompt = PromptTemplate(template=template, input_variables=["scenario"])
    story_llm = LLMChain(
        llm=OpenAI(temperature=1),
        prompt=prompt,
        verbose=True,
    )
    story = story_llm.predict(scenario=scenario)
    print(story)
    return str(story)


# text to speech
def text_to_speech(text):
    """
    Converts the given text to speech using the Hugging Face API.

    Args:
        text (str): The text to be converted to speech.

    Returns:
        None
    """

    API_URL = "https://api-inference.huggingface.co/models/facebook/mms-tts-eng"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.content

    audio_bytes = query({"inputs": text})

    # Store the audio in a file
    with open("audio.flac", "wb") as file:
        file.write(audio_bytes)


def main():
    """
    Main function to turn an image into an audio story.

    This function allows the user to choose an image file, convert it into an audio story,
    and display the scenario and story generated from the image. The audio story is also played.

    Args:
        None

    Returns:
        None
    """
    st.set_page_config(
        page_title="Image to Audio Story",
        page_icon="📸",
        layout="centered",
        initial_sidebar_state="auto",
    )
    st.header("Turn an image into an audio story! 📸🔊")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    # If a file has been uploaded
    if uploaded_file is not None:
        # Read the file content into bytes
        bytes_data = uploaded_file.getvalue()
        # Open a file with the same name as the uploaded file in write-binary mode
        with open(uploaded_file.name, "wb") as file:
            # Write the bytes data into the file
            file.write(bytes_data)
        # Display the uploaded image in the Streamlit app with a caption
        st.image(bytes_data, caption="Uploaded Image", use_column_width=True)
        # Use the pre-trained image captioning model to describe the image
        scenario = describe_image(uploaded_file.name)[0]["generated_text"]
        # Generate a short story based on the image description
        story = generate_story(scenario)
        # Convert the story text to speech
        text_to_speech(story)

        # Create an expander in the Streamlit app for the image description
        with st.expander("scenario"):
            # Write the image description into the expander
            st.write(scenario)
        # Create another expander for the story
        with st.expander("story"):
            # Write the story into the expander
            st.write(story)

        # Play the audio file in the Streamlit app
        st.audio("audio.flac")


if __name__ == "__main__":
    main()
