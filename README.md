# Image to Audio Story Converter

## Overview

This project is an innovative application that transforms images into audio stories. Utilizing cutting-edge AI models, it captions images, crafts stories based on those captions, and converts the text to speech, offering a unique auditory experience from visual inputs.

## Features

-  **Image Captioning:** Leverages a pre-trained model to describe images.
-  **Story Generation:** Creates short, engaging narratives based on image descriptions.
-  **Text-to-Speech Conversion:** Transforms generated stories into audio format using Hugging Face's API.
-  **Streamlit Web App:** Provides an interactive interface for users to upload images and receive audio stories.

## Technologies

-  Python
-  Streamlit
-  Hugging Face Transformers
-  Langchain
-  dotenv for environment management

## Setup and Usage

1. Install dependencies: `pip install -r requirements.txt`
2. Rename `.env.example` to `.env` and update it with your API Keys.
3. Run the Streamlit app: `streamlit run describe_image.py`

## Contribution

Contributions are welcome! Feel free to fork the repo and submit pull requests.

## License

This project is open-sourced under the MIT license.
