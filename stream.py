import streamlit as st
import webbrowser
from transformers import AutoModelForCausalLM, AutoTokenizer
import PyPDF2
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch


model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

st.title("Conversational Agent")

def agent_response(user_input):
    # Tokenize the input text
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")

    # Generate response
    response_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode the response and return it
    response = tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

user_input = st.text_input("You:")
if st.button("Ask"):
    if user_input.lower() == 'exit':
        st.write("Agent: Goodbye!")
    elif user_input.lower().startswith('open'):
        website = user_input.split(' ', 1)[1]
        st.write(f"Agent: Opening {website}")
        webbrowser.open_new_tab(website)
    elif user_input.lower().startswith('close'):
        st.write("Agent: Closing the current tab.")
        webbrowser.close()
    else:
        agent_response_text = agent_response(user_input)
        st.write("Agent:", agent_response_text)




# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    reader = PyPDF2.PdfReader(uploaded_file)
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text()
    return text

# Function to preprocess text
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

# Function to answer question using DistilBERT
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

# Main function to run Streamlit app
def main():
    st.title("PDF Question Answering System")

    # Upload PDF file
    uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])

    if uploaded_file is not None:
        # Extract text from PDF
        pdf_text = extract_text_from_pdf(uploaded_file)

        # Preprocess text
        processed_text = preprocess_text(pdf_text)

        # Ask question
        question = st.text_input("Enter your question:")
        if st.button("Get Answer"):
            if question:
                # Get answer
                answer = answer_question(processed_text, question)
                st.write("Answer:", answer)
            else:
                st.warning("Please enter a question.")
    else:
        st.warning("Please upload a PDF file.")

if __name__ == "__main__":
    main()
