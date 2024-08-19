import os
import requests
import vertexai.preview.generative_models as generative_models
from vertexai.generative_models import GenerativeModel, Part

from google.cloud import aiplatform, storage
from google.oauth2 import service_account

# Initialize VertexAI API
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp-service-account.json"

credentials = service_account.Credentials.from_service_account_file("gcp-service-account.json")

aiplatform.init(project = 'gcp-vpcx-acl',
                credentials = credentials)

class GeminiAgent():
    # user(str): Username for creating the Cloud Storage folder
    # doc_paths(list): The list of document paths to be uploaded

    def __init__(self, user, doc_paths):
        # Upload docs to CS and get them ready for the LLM
        self.bucket = "gcp-cloud-storage-bucket-url"
        self.user_path = "/".join(["cs-bucket-folder", user])
        # cs_paths = []
        
        # for path in doc_paths:
        #     with open(path, 'rb') as file:
        #         file_content = file.read()

        #     cs_paths.append(upload_file_to_storage(self, path, file_content))

        # Create the docs for the LLM
        self.docs = create_docs(self, doc_paths)

        # Initialize the LLM for the chatbot
        self.chat = initialize_gemini().start_chat(response_validation = False)
        print("Ready!")

    def invoke(self, prompt, history):
        # Runs an iteration of the conversation with the LLM
        # Inputs:
        #     - prompt: The user instruction and the documents to be used

        chat_history = "Chat History:\n" + str(history)
        responses = self.chat.send_message(self.docs + [chat_history, prompt],
                                        generation_config = {"temperature": 0,
                                                                "max_output_tokens": 8192,
                                                                "top_p": 0.95},
                                        safety_settings = {generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
                                                            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
                                                            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
                                                            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE})
        return responses.text

def create_docs(self, paths):
    # Formats the documents from the user's Cloud Storage folder so the LLM can read them
    # Inputs:
    #     - paths: The list of documents in cloud storage uris

    docs = []

    for path in paths:
        # Read from the file
        filename = path.rsplit("/", 1)[1]
        storage_client = storage.Client()
        bkt = storage_client.bucket(self.bucket)
        blob = bkt.blob("/".join([self.user_path, filename]))
        
        with blob.open("rb") as f:
            data = f.read()
            
        # Create the VertexAI document
        docs.append(Part.from_data(mime_type = "application/pdf",
                                        data = data))

    return docs

def initialize_gemini():
    # Initialize the LLM to get resopnses from it.

    safety_settings = {generative_models.HarmCategory.HARM_CATEGORY_UNSPECIFIED: generative_models.HarmBlockThreshold.BLOCK_NONE,
                       generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
                       generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
                       generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
                       generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE}
    
    agent_prompt = """You're a service agent in charge of evaluating different documents to answer the user's questions.
    Talk in a formal and professional manner.
    If you need to evaluate multiple documents, evaluate the user request on each document individually, and join all the answers in the end.
    Use ALL the documents when necessary. Do NOT ignore any of them.
    Use ONLY the information on the given documents to generate your answer.
    If you receive multiple questions, answer each one individually."""
    #Use the following answer when you can't respond to the user's questions: "I'm sorry, but the documents you provided do not have enough information to solve your request"."""
    
    model = GenerativeModel("gemini-1.5-pro-001",
                            generation_config = {"temperature": 0.2,
                                                    "max_output_tokens": 8192,
                                                    "top_p": 0.95},
                            system_instruction = agent_prompt,
                            safety_settings = safety_settings)
    return model