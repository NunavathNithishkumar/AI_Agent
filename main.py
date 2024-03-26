import webbrowser
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print("Hello! How can I assist you today?")

while True:
    # Get user input
    user_input = input("You: ")

    # Exit condition
    if user_input.lower() == 'exit':
        print("Agent: Goodbye!")
        break

    # Check if the user wants to open a website
    if user_input.lower().startswith('open'):
        website = user_input.split(' ', 1)[1]
        print("Agent: Opening", website)
        webbrowser.open_new_tab(website)
        continue

    # Check if the user wants to close a website
    if user_input.lower().startswith('close'):
        print("Agent: Closing the current tab.")
        webbrowser.close()
        continue

    # Tokenize the input text
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")

    # Generate response
    response_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode the response and print it
    response = tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    print("Agent:", response)
