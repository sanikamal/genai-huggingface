#  Practical AI Solutions and Generative AI Applications with Hugging Face Models and Gradio 🤖🚀

## Overview
This repository contains Practical AI solutions and generative AI applications using open-source models available on the Hugging Face Hub. Through a series of mini projects and notebooks to showcase the practical applications of Hugging Face's transformers library for various tasks including Natural Language Processing (NLP), audio processing, image analysis, and multimodal tasks. The applications are built with Gradio to provide user-friendly interfaces.

## Notebooks

### Chatbot 🤖
- **Description**: Build a chatbot capable of multi-turn conversations. Use a small language model from Hugging Face Hub.
- **Notebook**: [Chatbot Notebook](notebooks/NLP_chatbot_pipeline.ipynb)

### Translation and Summarization 🌍
- **Description**: Translate between languages and summarize. Use Hugging Face models from Meta for translation and summarization tasks.
- **Notebook**: [Translation and Summarization Notebook](notebooks/translation_and_summarization.ipynb)

  ### Sentence Embeddings 🌍
- **Description**: Build the sentence embedding pipeline using 🤗 Transformers Library
- **Notebook**: [Sentence Embeddings Notebook](notebooks/sentence_embeddings.ipynb)

### Zero-shot Audio Classification 🎧
- **Description**: Perform audio classification without fine-tuning the model. Use Hugging Face audio classification pipeline from 🤗 Transformers Library
- **Notebook**: [Zero-shot Audio Classification Notebook](notebooks/zero-shot_audio_classification.ipynb)

### Automatic Speech Recognition 🎧
- **Description**: Convert audio to text with Automatic Speech Recognition (ASR). Use Hugging Face automatic speech recognition pipeline from 🤗 Transformers Library and whisper model.
- **Notebook**: [Automatic Speech Recognition Notebook](notebooks/automatic_speech_recognition.ipynb)

### Text to Speech 🎧
- **Description**: Generate audio from text using text-to-speech (TTS). Use Hugging Face to build the text-to-speech pipeline using 🤗 Transformers Library
- **Notebook**: [Text to Speech Notebook](notebooks/text_to_speech.ipynb)

### Object Detection Audio Descriptions for Images 🔊
- **Description**: Generate audio narrations describing images using object detection and TTS. Use Hugging Face models for object detection and TTS.
- **Notebook**: [Audio Descriptions for Images Notebook](notebooks/object_detection.ipynb)
- 
### Image Segmentation 🖼️
- **Description**: Identify objects or regions in an image using zero-shot image segmentation. Use the Hugging Face SAM model from Meta for image segmentation and Depth Estimation with DPT.
- **Notebook**: [Image Segmentation Notebook](notebooks/segmentation.ipynb)

### Image Retrieval 🖼️
- **Description**: Image Retrieval using the Salesforce blip model from Hugging Face.
- **Notebook**: [Image Retrieval Notebook](notebooks/image_retrieval.ipynb)
  
### Image Captioning 📷
- **Description**: Upload an image and generate a caption for it using an image captioning model. Use the Salesforce blip model from Hugging Face.
- **Notebook**: [Image Captioning Notebook](notebooks/image_captioning.ipynb)

### Multimodel Visual Question & Answering 🖼️
- **Description**: Multimodel Visual Question & Answering using the Salesforce blip model from Hugging Face.
- **Notebook**: [Multimodel Visual QA Notebook](notebooks/visual_q_and_a.ipynb)

### Zero-Shot Image Classification 🖼️
- **Description**: Perform image classification without fine-tuning the model using the OpenAI clip model from Hugging Face.
- **Notebook**: [Multimodel Visual QA Notebook](notebooks/zero_shot_image_classification.ipynb)

## Deployment with Gradio and Hugging Face Spaces ☁️
- **Description**: Share AI applications using Gradio and Hugging Face Spaces for user-friendly cloud deployment. Create a simple interface with Gradio and deploy it using Hugging Face Spaces.
- **Notebook**: [Deployment with Gradio Notebook](notebooks/hf_deployment.ipynb)

## Mini Projects

### Text-to-Image Generation 🎨
- **Description**: Generate an image from text using a diffusion model.
- **Implementation**: Use Hugging Face models for text-to-image generation.
- **Notebook**: [Text-to-Image Generation Notebook](notebooks/text_to_image.ipynb)

### Chat with LLM using Falcon 🦅
- **Description**: Create an interface to chat with an open-source Large Language Model (LLM) using Falcon.
- **Implementation**: Use Falcon to interact with the LLM.
- **Notebook**: [Chat with LLM Notebook](notebooks/chat_with_llm.ipynb)






3. 🔊 **Audio Conversion**
   - **Description**: Convert audio to text with ASR and generate audio from text using TTS.
   - **Implementation**: Use Hugging Face models for ASR and TTS.
   - [Link to Notebook](audio_conversion/audio_conversion.ipynb)

4. 🎧 **Zero-shot Audio Classification**
   - **Description**: Perform audio classification without fine-tuning the model.
   - **Implementation**: Use Hugging Face models for zero-shot learning.
   - [Link to Notebook](audio_classification/audio_classification.ipynb)

5. 📷 **Image Segmentation**
   - **Description**: Identify objects or regions in an image using zero-shot image segmentation.
   - **Implementation**: Use Hugging Face models for image segmentation.
   - [Link to Notebook](image_segmentation/image_segmentation.ipynb)

6. 🌐 **Chat with LLM using Falcon**
   - **Description**: Create an interface to chat with an open-source LLM using Falcon.
   - **Implementation**: Use Falcon to create a chat interface with a large language model.
   - [Link to Notebook](chat_with_llm/chat_with_llm.ipynb)

7. 🌅 **Image Generation and Captioning**
   - **Description**: Generate images from text descriptions and caption images.
   - **Implementation**: Use Hugging Face models for image generation and captioning.
   - [Link to Notebook](image_generation_captioning/image_generation_captioning.ipynb)

8. 🔄 **Image Captioning and Generation**
   - **Description**: Upload an image, caption the image, and use the caption to generate a new image.
   - **Implementation**: Combine image captioning and generation using Hugging Face models.
   - [Link to Notebook](image_captioning_generation/image_captioning_generation.ipynb)


1. **Text Summarization App** 📄
   - Create a user-friendly app (usable for non-coders) to take input text, summarize it with an open-source large language model, and display the summary.

2. **Image Captioning App** 🖼️
   - Allow users to upload an image, which uses an image-to-text (image captioning) model to describe the uploaded image, and display both the image and the caption in the app.

3. **Text-to-Image Generation App** 📝➡️🖼️
   - Generate an image from text input using a diffusion model and display the generated image within the app.

4. **Combined Image and Caption Generation App** 🖼️➡️📝➡️🖼️
   - Upload an image, caption the image, and use the caption to generate a new image, all within the same app.

5. **Chat Interface with Open Source LLM** 💬
   - Create an interface to chat with an open-source Large Language Model using Falcon.

1. **Text Summarization App** 📄
   - Create a user-friendly app (usable for non-coders) to take input text, summarize it with an open-source large language model, and display the summary.

2. **Image Captioning App** 🖼️
   - Allow users to upload an image, which uses an image-to-text (image captioning) model to describe the uploaded image, and display both the image and the caption in the app.

3. **Text-to-Image Generation App** 📝➡️🖼️
   - Generate an image from text input using a diffusion model and display the generated image within the app.

4. **Combined Image and Caption Generation App** 🖼️➡️📝➡️🖼️
   - Upload an image, caption the image, and use the caption to generate a new image, all within the same app.

5. **Chat Interface with Open Source LLM** 💬
   - Create an interface to chat with an open-source Large Language Model using Falcon.

### How to Use
1. Clone this repository.
2. Install the necessary dependencies.
3. Run the notebooks or deploy the mini projects using Gradio and Hugging Face Spaces.

## Resources

- [Hugging Face Hub](https://huggingface.co/models)
- [Transformers Library Documentation](https://huggingface.co/transformers/)
- [Gradio Documentation](https://gradio.app/docs)
- [Hugging Face Spaces Documentation](https://huggingface.co/spaces/)

### Contribution
You can feel free to contribute by adding mini-projects, improving existing implementations, or providing feedback.

### Acknowledgements
This project is based on the tutorials and resources provided by Hugging Face, Gradio and Deeplearning.ai short course.

### License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

🚀 Happy coding and exploring the world of AI with Hugging Face and Gradio! 🤖
