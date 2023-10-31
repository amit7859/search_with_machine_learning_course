import fasttext

# Config 

threshold = 0.75
top_words_file = '/workspace/datasets/fasttext/top_words.txt'
model_file = '/workspace/datasets/fasttext/title_model_100_25_epoch.bin'
synonyms_file = '/workspace/datasets/fasttext/synonyms.csv'

model = fasttext.load_model(model_file)

with open(top_words_file, 'r') as file:
    words = file.readlines()

lines = []

for word in words:
    word = word.strip()
    model_output = model.get_nearest_neighbors(word)
    synonyms = [candidate for score, candidate in model_output if score >= threshold]
    if (len(synonyms) > 0):
        lines.append(word + ',' + ','.join(synonyms) + "\n")

with open(synonyms_file, "w") as output_file:
    for line in lines:
        output_file.write(line)
