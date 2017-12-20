from keras.preprocessing.text import text_to_word_sequence, one_hot, hashing_trick, Tokenizer
# define the document
text = 'The quick brown fox jumped over the lazy dog.'
print(text)

# tokenize the document
'''
result = text_to_word_sequence(text)
print(result)

words = set(text_to_word_sequence(text))
print(words)
vocab_size = len(words)
print(vocab_size)

result = one_hot(text, round(vocab_size*1.3))
print(result)

result = hashing_trick(text, round(vocab_size*1.3), hash_function='md5')
print(result)
'''
# define documents
docs = []
with open('all_dga.txt', 'r') as f:
	for line in f.readlines():
		dga_domain, _ = line.split(' ')
		docs.append(dga_domain)

# create the tokenizer
t = Tokenizer()
# fit the tokenizer on the documents
t.fit_on_texts(docs)
'''
print("")
print(t.word_counts)
print("")
print(t.word_docs)
print("")
print(t.word_index)
print("")
print(t.document_count)
print("")
'''
# integer encode documents
encoded_docs = t.texts_to_matrix(docs, mode='count')
print(encoded_docs)