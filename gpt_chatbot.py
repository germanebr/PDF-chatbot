#Generic
import os
# import time
#Cosine Similarity
import numpy as np
from numpy.linalg import norm
#LLM
from langchain_openai import AzureChatOpenAI
#PDF Reader
from pypdf import PdfReader
#Embeddings
from langchain_openai import AzureOpenAIEmbeddings
#Memory
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
#Req/Resp
import requests
#Google Cloud
from google.cloud import storage
from google.oauth2 import service_account
#Pickle
import pickle


class GPT:
    '''GPT Model to be used as a PDF chatbot'''

    def __init__(self, paths: list, user, content = None):
        '''
        Initialize the GPT 4o model to be used as a chatbot.
        @param paths: List containing one or more paths to PDF's to be read.
        @returns: LLM Model Object.
        '''
        self.__embeddingsList = {} #NOT IN ORIGINAL
        #Step 1, get LLM:
        self.llm = self.__gptInit(0)
        print("LLM initialized")
        self.compress = []
        if not content:
            self.pdfs = self.__prepDocs(paths=paths, user=user)
            self.compress = [self.pdfs, self.__embeddingsList]
        else:
            self.pdfs = content[0]
            self.__embeddingsList = content[1]
        '''
        To access self.pdfs, use the syntax: var = self.pdfs[pdfName]
        '''

        self.conversation_with_summary = ConversationChain(
            llm=self.llm,
            memory=ConversationSummaryBufferMemory(llm=self.llm)
        )
        print('Ready!')

    
    ''' ----------------- PUBLIC METHODS ----------------- '''

    def invoke(self, query, history):
        '''
        Invokes the LLM based on the users query.
        @param query: Question from the user for the LLM.
        '''
        prompt = self.__promptInit(query=query, history=history)

        res = self.conversation_with_summary.predict(input=prompt)

        return res

    ''' ----------------- INITIALIZATION AND DOCUMENT PREPROCESSING ----------------- '''

    def __gptInit(self, temp):
        '''
        Initialize GPT model and return model object.
        @param temp: Temperature to initialize model with.
        '''
        # Initialize Azure OpenAI
        path = "API_key.txt"
        with open(path) as f:
            API_key = f.readlines()[0]
            
        os.environ['OPENAI_API_VERSION'] = '2023-05-15'
        os.environ['AZURE_OPENAI_API_KEY'] = API_key
        os.environ['AZURE_OPENAI_ENDPOINT'] = "azure-openai-endpoint-url"

        # Specify the model
        return AzureChatOpenAI(model_name = 'gpt-4o',
                            openai_api_key=API_key,
                            temperature = temp)
    

    def __prepDocs(self, paths: list, user):
        '''
        Prepares the docs for use with the chatbot.
        @param paths: paths to pdfs.
        '''
        pdfContentDict = {}
        for path in paths:
            pdfName = path.split('/')[-1]
            print(pdfName)
            # self.__call_upload(path=path, user=user)
            pdfContentDict[pdfName] = self.__getDocs(path=path, pdfName=pdfName, user=user)
        return pdfContentDict


    def __call_upload(self, path, user):
        print(f"Calling Upload: {path}")
        user_path = f"pdf-chatbot/{user}"
        bucket = "gcp-cloud-storage-bucket-url"
        cs_paths = []
        
        with open(path ,'rb') as file:
            file_content = file.read()
        cs_paths.append(self.__upload_file_to_storage(path, file_content, user_path, bucket))


    #Upload zip to some cloud storage path 
    def __upload_file_to_storage(self, file_path, file_bytes, tgt_path, tgt_bkt):
        """Upload to GCP Cloud Storage
        Inputs:
            - file_path: The path of the file you want to store
            - file_bytes: Bytes of the document to upload
            - tgt_path: Child path (folder) in which the file will be stored
            - tgt_kbt: Name of the main bucket where the data will be stored"""
        
        # This url comes from an internal microservice that uploads into a bucket into GCP Cloud Storage.
        # You can replace this code for the one shown on GCP's documentation
        url = "pdf-upload-to-gcp-cloud-storage-api"
        print("Uploading file to Cloud Bucket...")
        file_name = file_path.split("/")[-1]
        tgt_path = tgt_path + '/' + file_name
        response = requests.post(url,
                                files={"file": (file_name, file_bytes, "multipart/form-data")},
                                data={"filepath": tgt_path, "bucketname": tgt_bkt},
                                verify=False)
        if response.status_code == 200:
            return "/".join(["gs/", tgt_bkt, tgt_path])
        else:
            print("Error in File Upload to GCS")
            print(f"Status Code: {response.status_code}")
            print(f"Status Text: {response.text}")
            raise Exception("File Upload Error")
    

    def __getDocs(self, path, pdfName, user="user00", bucket="gcp-cs-bucket-name"):
        '''
        Gets docs from cloud bucket (TODO)
        @param path: path to file
        '''
        #reader = PdfReader(path) 
        #pages = reader.pages
        print("Retrieving Document from Cloud...")
        credentials = service_account.Credentials.from_service_account_file('gcp-service-account.json')

        user_path = f"gcp-cs-bucket-user-folder/{user}"
        print("User path = ", user_path)
        storage_client = storage.Client(credentials=credentials)
        bkt = storage_client.bucket(bucket)
        blob = bkt.blob("/".join([user_path, pdfName]))
        with blob.open("rb") as f:
            reader = PdfReader(f)
            pages = reader.pages

            return self.__getChunks(pages, pdfName=pdfName)
    

    def __getChunks(self, pages, pdfName):
        '''
        Get chunks for PDF.
        @param pages: pages of pdf.
        '''
        chunksEmbedded = {}
        self.__embeddingsList[pdfName] = []

        chunks = [pages[i].extract_text() + "/n/n" + pages[i+1].extract_text() for i in range(len(pages)-1)]
        
        print('Embedding document...')
        #for chunk in chunks: ORIGINAL
            #chunksEmbedded[chunk] = self.__getEmbeddings(chunk) ORIGINAL

        for i in range(len(chunks)): #NOT IN ORIGINAL
            chunksEmbedded[f"chunk_{i}"] = chunks[i] #NOT IN ORIGINAL
            self.__embeddingsList[pdfName].append(self.__getEmbeddings(chunks[i])) #NOT IN ORIGINAL

        print("Embeddings List size after embedding:", len(self.__embeddingsList))
        print("Embeddings List size after embedding:", len(self.__embeddingsList[pdfName]))
        
        return chunksEmbedded
    
    
    def __getEmbeddings(self, texts, qty = 'SINGLE'):
        embeddings = AzureOpenAIEmbeddings(azure_deployment = "text-embedding-ada-002",
                                           openai_api_version = "2023-05-15")
    
        if qty == 'MULTIPLE':
            return embeddings.embed_documents(texts)
        else:
            return embeddings.embed_query(texts)
        

    ''' ----------------- MODEL ANSWERING ----------------- '''


    def __promptInit(self, query, history):
        relevantChunks = []
        prompt = f"""You are an LLM who's primary goal is to provide information about documents based on context from them.
        Answer the question based on the context below, which is a section from each relevant document.
        If you need to evaluate multiple documents, evaluate the user request on each document individually, and join all the answers in the end.
        If the question seems unrelated to the document, try to answer off of previous chat history.
        If the question still cannot be answered using the information provided, answer with "I don't know".
        If you receive multiple questions, answer each one individually.
 
        Previous Chat History: {history}
        
        Context: """
        for doc in self.pdfs.keys():
            relevantChunks.append(self.__getRelevantChunks(query=query, pdfName=doc))

        for i in range(len(relevantChunks)):
            prompt += f"Document {i+1}: {relevantChunks[i]}\n\n"
        prompt += f"""\n\nQuestion: {query}
                    
        Answer:
                    """

        return prompt
    

    def __getRelevantChunks(self, query, pdfName):
        chunkSimilarities = self.__cosineSimilarity(query=query, pdfName=pdfName)

        for chunks in chunkSimilarities.keys():
            mostRelevantChunk = chunks
            break

        print(mostRelevantChunk)
        return self.pdfs[pdfName][mostRelevantChunk]


    def __cosineSimilarity(self, query, pdfName):
        cosineSimilarities = {}

        embeddedQuery = self.__getEmbeddings(query)
        # for key in self.pdfs.keys(): ORIGINAL
        #     for key2 in self.pdfs[key].keys(): ORIGINAL
        #         embeddedChunk = self.pdfs[key][key2] ORIGINAL
        #         cosine = np.dot(embeddedQuery,embeddedChunk)/(norm(embeddedQuery)*norm(embeddedChunk)) ORIGINAL

        for i in range(len(self.__embeddingsList[pdfName])):
            embeddedChunk = self.__embeddingsList[pdfName][i]
            cosine = np.dot(embeddedQuery,embeddedChunk)/(norm(embeddedQuery)*norm(embeddedChunk))
            cosineSimilarities[f"chunk_{i}"] = cosine

        return dict(sorted(cosineSimilarities.items(), key=lambda item: item[1], reverse=True))