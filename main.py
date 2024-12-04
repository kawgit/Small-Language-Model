import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import threading

torch.random.manual_seed(0)
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-128k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

# Initialize messages outside the loop
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
    {
        "role": "assistant",
        "content": "Sure! Here are some ways to eat bananas and dragonfruits together:\n\n1. **Banana and dragonfruit smoothie**: Blend bananas and dragonfruits together with some milk and honey.\n\n2. **Banana and dragonfruit salad**: Mix sliced bananas and dragonfruits together with some lemon juice and honey.",
    },
]

def messages_to_prompt(messages):
    prompt = ''
    for message in messages:
        role = message['role']
        content = message['content']
        prompt += f"<|{role}|>\n{content}<|end|>\n"
    prompt += "<|assistant|>"
    return prompt

while True:
    user_input = input("=========== User ===========\n")
    messages.append({"role": "user", "content": user_input})

    prompt = messages_to_prompt(messages)
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)

    # Initialize the streamer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_args = {
        "input_ids": input_ids,
        "max_new_tokens": 500,
        "temperature": 0.7,  # Set temperature for variability
        "do_sample": True,   # Enable sampling for more natural responses
        "streamer": streamer,
    }

    # Run generation in a separate thread
    generation_thread = threading.Thread(target=model.generate, kwargs=generation_args)
    generation_thread.start()

    print("=========== AI ===========")

    assistant_reply = ''
    for new_text in streamer:
        print(new_text, end='', flush=True)
        assistant_reply += new_text

    generation_thread.join()
    print()

    messages.append({"role": "assistant", "content": assistant_reply})
