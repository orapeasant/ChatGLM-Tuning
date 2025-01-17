from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('chatglm2',trust_remote_code = True)

model= AutoModel.from_pretrained('chatglm2', trust_remote_code = True).half().cuda()

model=model.eval()

sentences = ['Who is Donald Trump']

for sentence in sentences:
	response, history = model.chat(tokenizer, sentence, history = [])
	print(sentence, response)
	response, history = model.chat(tokenizer, "what is I cannot fall asleep",history =history)
	print(response)
while True:
  pass
