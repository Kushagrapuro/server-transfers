from transformers import AutoTokenizer, AutoModelForCausalLM


tokenizer = AutoTokenizer.from_pretrained("abhilash2599/llama-2-7b-medlm-2k")
model = AutoModelForCausalLM.from_pretrained("abhilash2599/llama-2-7b-medlm-2k")


input_text = "What are the symptoms of diabetes?"


inputs = tokenizer(input_text, return_tensors="pt")


outputs = model.generate(**inputs, max_length=100, num_return_sequences=1)


response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Model Response:", response)