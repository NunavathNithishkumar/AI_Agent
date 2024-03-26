import PyPDF2
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove punctuation and convert to lowercase
    tokens = [token.lower() for token in tokens if token.isalnum()]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back into text
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

def answer_question(context, question):
    # Load DistilBERT tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased')

    # Tokenize input text and question
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt", max_length=512, truncation=True)

    # Get answer start and end positions
    with torch.no_grad():
        outputs = model(**inputs)
    
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # Find the tokens with the highest start and end scores
    start_index = torch.argmax(start_scores[0])
    end_index = torch.argmax(end_scores[0])

    # Get the answer tokens and decode them
    answer_tokens = inputs['input_ids'][0][start_index:end_index+1]
    answer = tokenizer.decode(answer_tokens)

    return answer

# Example usage:
pdf_path = "D:\App\PathOr AI Intern task 2.pdf"
pdf_text = extract_text_from_pdf(pdf_path)
processed_text = preprocess_text(pdf_text)

question = input("ASK")
answer = answer_question(processed_text, question)
print("Answer:", answer)




# import webbrowser
# import PyPDF2
# import re
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from transformers import AutoModelForCausalLM, AutoTokenizer, DistilBertTokenizer, DistilBertForQuestionAnswering
# import torch

# # Download NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# def extract_text_from_pdf(pdf_path):
#     text = ""
#     with open(pdf_path, 'rb') as file:
#         reader = PyPDF2.PdfReader(file)
#         for page_num in range(len(reader.pages)):
#             page = reader.pages[page_num]
#             text += page.extract_text()
#     return text

# def preprocess_text(text):
#     # Tokenize the text
#     tokens = word_tokenize(text)
    
#     # Remove punctuation and convert to lowercase
#     tokens = [token.lower() for token in tokens if token.isalnum()]
    
#     # Remove stopwords
#     stop_words = set(stopwords.words('english'))
#     tokens = [token for token in tokens if token not in stop_words]
    
#     # Lemmatize tokens
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
#     # Join tokens back into text
#     preprocessed_text = ' '.join(tokens)
    
#     return preprocessed_text

# def answer_question(context, question):
#     # Load DistilBERT tokenizer and model
#     tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
#     model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased')

#     # Tokenize input text and question
#     inputs = tokenizer.encode_plus(question, context, return_tensors="pt", max_length=512, truncation=True)

#     # Get answer start and end positions
#     with torch.no_grad():
#         outputs = model(**inputs)
    
#     start_scores = outputs.start_logits
#     end_scores = outputs.end_logits

#     # Find the tokens with the highest start and end scores
#     start_index = torch.argmax(start_scores[0])
#     end_index = torch.argmax(end_scores[0])

#     # Get the answer tokens and decode them
#     answer_tokens = inputs['input_ids'][0][start_index:end_index+1]
#     answer = tokenizer.decode(answer_tokens)

#     return answer
# # For any other input, assume it's context for question answering
# pdf_path = "C:\Desktop\App\PathOr AI Intern task 2.pdf"
# pdf_text = extract_text_from_pdf(pdf_path)
# processed_text = preprocess_text(pdf_text)
# user_input=input("Ask the question")
# answer = answer_question(processed_text, user_input)
# print("Answer:", answer)



# # Load the DialoGPT model and tokenizer
# model_name = "microsoft/DialoGPT-medium"
# dialogpt_tokenizer = AutoTokenizer.from_pretrained(model_name)
# dialogpt_model = AutoModelForCausalLM.from_pretrained(model_name)



# print("Hello! How can I assist you today?")

# while True:
#     # Get user input
#     user_input = input("You: ")

#     # Exit condition
#     if user_input.lower() == 'exit':
#         print("Agent: Goodbye!")
#         break

#     # Check if the user wants to open a website
#     if user_input.lower().startswith('open'):
#         website = user_input.split(' ', 1)[1]
#         print("Agent: Opening", website)
#         webbrowser.open_new_tab(website)
#         continue

#     # Check if the user wants to close a website
#     if user_input.lower().startswith('close'):
#         print("Agent: Closing the current tab.")
#         # Note: webbrowser.close() doesn't work reliably across all platforms.
#         # You might need to find an alternative way to handle closing tabs.
#         continue

#     # Check if the user is asking a question
#     if '?' in user_input:
#         # Tokenize the input text
#         input_ids = dialogpt_tokenizer.encode(user_input + dialogpt_tokenizer.eos_token, return_tensors="pt")

#         # Generate response
#         response_ids = dialogpt_model.generate(input_ids, max_length=1000, pad_token_id=dialogpt_tokenizer.eos_token_id)

#         # Decode the response and print it
#         response = dialogpt_tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
#         print("Agent:", response)
#         continue

    