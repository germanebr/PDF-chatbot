import pickle
import requests
from google.cloud import storage
from gpt_chatbot import GPT
from gemini_chatbot import GeminiAgent

class Chatbot:

    def __init__(self, model : str, user : str, paths : list, content=None):
        '''Initialize the Chatbot LLM.
        @param: model: String containing model name (Acceptable values: Gemini, GPT)
        @param: paths: paths to all pdfs (uploaded to Cloud Storage) to be given to model, must be given in the form of a list, even if it's only one path.
        '''
        if model == 'GPT':
            print("Creating GPT model")
            self.LLM = GPT(paths=paths, user=user, content=content)
        else:
            print("Creating Gemini model")
            self.LLM = GeminiAgent(user=user, doc_paths=paths) #Will be Gemini

def upload_file_to_storage(user, file_bytes, filename="agent"):
    # Upload to GCP Cloud Storage
    # Inputs:
    #     - file_path: The path of the file you want to store
    #     - file_bytes: Bytes of the document to upload

    # This url comes from an internal microservice that uploads into a bucket into GCP Cloud Storage.
    # You can replace this code for the one shown on GCP's documentation
    url = "pdf-upload-to-gcp-cloud-storage-api"
    
    file_name = f"{filename}.pkl"
    tgt_path = "/".join(["gcp-cs-folder", user, file_name])
    response = requests.post(url,
                            files={"file": (file_name, file_bytes, "multipart/form-data")},
                            data={"filepath": tgt_path, "bucketname": "gcp-cs-bucket-name"},
                            verify=False)
    if response.status_code == 200:
        return "/".join(["gs:/", "gcp-cs-bucket-name", tgt_path])
    else:
        print("Error in File Upload to GCS")
        print(f"Status Code: {response.status_code}")
        print(f"Status Text: {response.text}")
        raise Exception("File Upload Error")


def getBlob(user, filename="agent"):
    storage_client = storage.Client()
    bkt = storage_client.bucket("gcp-cs-bucket-name")
    blob = bkt.blob(f"gcp-cs-folder/{user}/{filename}.pkl")
    return blob


def cloud_start(input_json):
    '''
    Entry point for cloud function.
    @params: input_json: JSON containing the Step (Upload or Chat):
        if Upload: Model (Gemini or GPT) and paths (paths to files) --> Returns Status and LLM Object
        if Chat: LLM (LLM Object received from Upload call) and query (User Query) --> Returns Status, LLM Object, and LLM Answer
    '''
    try:
        ip_request_json = input_json.get_json(silent=True)
        step = ip_request_json["step"]
        step = step.lower()

        if step == "upload":
            model = ip_request_json["model"]
            paths = ip_request_json["paths"]
            user = ip_request_json["user"]
            obj = Chatbot(model=model, user=user, paths=paths)
            if model.lower() == "gemini":
                pkl = pickle.dumps(obj.LLM)
            else:
                pkl = pickle.dumps(obj.LLM.compress)
            upload_file_to_storage(user, pkl)

            chat_history_dict = {"System" : "Waiting for user query."}
            pklHistory = pickle.dumps(chat_history_dict)
            upload_file_to_storage(user, pklHistory, filename="history")

            op_response = {"status": True, "ans": "Agent stored in user folder."}

        elif step == "chat":
            user = ip_request_json["user"]
            query = ip_request_json["query"]
            model = ip_request_json["model"]

            blob = getBlob(user=user)
            
            if model.lower() == "gemini":
                LLM = pickle.loads(blob.download_as_string())
            else:
                content = pickle.loads(blob.download_as_string())
                obj = Chatbot(model="GPT", user=user, paths=[], content=content)
                LLM = obj.LLM
            print(type(LLM))

            historyBlob = getBlob(user=user, filename="history")
            historyDict = pickle.loads(historyBlob.download_as_string())
            
            LLM_response = LLM.invoke(query, historyDict)

            historyDict[f"Message_{len(historyDict)}"] = (f"User: {query}", f"System (LLM): {LLM_response}")
            # print(f"Updated History Dict: {historyDict}")
            pklHistory = pickle.dumps(historyDict)
            upload_file_to_storage(user, pklHistory, filename="history")

            op_response = {"status": True, "ans": LLM_response}

        else:
            raise ValueError("Invalid step value, acceptable step values are: 'Upload', 'Chat'.")
    except Exception as ex:
        op_response = {"status": False, "error": ex.__str__()}
    return op_response