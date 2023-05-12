!pip install tensorflow==1.1.0
import tensorflow as tf
print(tf.__version__)

# In[1]:

import pandas as pd
import numpy as np
import tensorflow as tf
import re
from nltk.corpus import stopwords
import time
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
print('TensorFlow Version: {}'.format(tf.__version__))


# ## Inspecting data

# In[2]:
!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

link = 'https://drive.google.com/open?id=1OidyyQrBQvViZkOKtBY7TJm9GWx4Chg0'

fluff, id = link.split('=')
print (id)

downloaded = drive.CreateFile({'id':id})
downloaded.GetContentFile('Reviews.csv')
reviews = pd.read_csv('Reviews.csv')


# In[3]:

reviews.shape


# In[4]:

reviews.head()


# In[5]:

# Check for any nulls values
reviews.isnull().sum()


# In[6]:

# Remove null values and irrelevant features
reviews = reviews.dropna()
reviews = reviews.drop(['Id','ProductId','UserId','ProfileName','HelpfulnessNumerator','HelpfulnessDenominator',
                        'Score','Time'], 1)
reviews = reviews.reset_index(drop=True)


# In[7]:

reviews.head()


# In[8]:

# Inspecting some of the reviews
for i in range(5):
    print("Review #",i+1)
    print(reviews.Summary[i])
    print(reviews.Text[i])
    print()


# ## Preparing data

# In[9]:


contractions = {
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'll": "i will",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"oughtn't": "ought not",
"shan't": "shall not",
"sha'n't": "shall not",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that would",
"that's": "that is",
"there'd": "there had",
"there's": "there is",
"they'd": "they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"where'd": "where did",
"where's": "where is",
"who'll": "who will",
"who's": "who is",
"won't": "will not",
"wouldn't": "would not",
"you'd": "you would",
"you'll": "you will",
"you're": "you are"
}


# In[203]:

def clean_text(text, remove_stopwords = True):
    '''Remove unwanted characters, stopwords, and format the text to create fewer nulls word embeddings'''

    # Convert words to lower case
    text = text.lower()

    # Replace contractions with their longer forms
    if True:
        text = text.split()
        new_text = []
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        text = " ".join(new_text)

    # Format words and remove unwanted characters
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text)
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)

    # Optionally, remove stop words
    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)

    return text


# We will remove the stopwords from the texts because they do not provide much use for training our model. However, we will keep them for our summaries so that they sound more like natural phrases.

# In[204]:

# Clean the summaries and texts
clean_summaries = []
for summary in reviews.Summary:
    clean_summaries.append(clean_text(summary, remove_stopwords=False))
print("Summaries are complete.")

clean_texts = []
for text in reviews.Text:
    clean_texts.append(clean_text(text))
print("Texts are complete.")


# In[206]:

# Inspect the cleaned summaries and texts to ensure they have been cleaned well
for i in range(5):
    print("Clean Review #",i+1)
    print(clean_summaries[i])
    print(clean_texts[i])
    print()


# In[207]:

def count_words(count_dict, text):
    '''Count the number of occurrences of each word in a set of text'''
    for sentence in text:
        for word in sentence.split():
            if word not in count_dict:
                count_dict[word] = 1
            else:
                count_dict[word] += 1


# In[208]:

# Find the number of times each word was used and the size of the vocabulary
word_counts = {}

count_words(word_counts, clean_summaries)
count_words(word_counts, clean_texts)

print("Size of Vocabulary:", len(word_counts))


# In[209]:

from google.colab import drive
drive.mount('/content/drive')


embeddings_index = {}
with open('/content/drive/My Drive/numberbatch-en-17.02.txt', 'r') as f:
    for line in f:
        values = line.split(' ')
        word = values[0]
        embedding = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = embedding

print('Word embeddings:', len(embeddings_index))


# In[210]:

# Find the number of words that are missing from CN, and are used more than our threshold.
missing_words = 0
threshold = 20

for word, count in word_counts.items():
    if count > threshold:
        if word not in embeddings_index:
            missing_words += 1

missing_ratio = round(missing_words/len(word_counts),4)*100

print("Number of words missing from CN:", missing_words)
print("Percent of words that are missing from vocabulary: {}%".format(missing_ratio))


# I use a threshold of 20, so that words not in CN can be added to our word_embedding_matrix, but they need to be common enough in the reviews so that the model can understand their meaning.

# In[211]:


#dictionary to convert words to integers
vocab_to_int = {}

value = 0
for word, count in word_counts.items():
    if count >= threshold or word in embeddings_index:
        vocab_to_int[word] = value
        value += 1
codes = ["<UNK>","<PAD>","<EOS>","<GO>"]

for code in codes:
    vocab_to_int[code] = len(vocab_to_int)

# Dictionary to convert integers to words
int_to_vocab = {}
for word, value in vocab_to_int.items():
    int_to_vocab[value] = word

usage_ratio = round(len(vocab_to_int) / len(word_counts),4)*100

print("Total number of unique words:", len(word_counts))
print("Number of words we will use:", len(vocab_to_int))
print("Percent of words we will use: {}%".format(usage_ratio))


# In[212]:

embedding_dim = 300
nb_words = len(vocab_to_int)

# Create matrix with default values of zero
word_embedding_matrix = np.zeros((nb_words, embedding_dim), dtype=np.float32)
for word, i in vocab_to_int.items():
    if word in embeddings_index:
        word_embedding_matrix[i] = embeddings_index[word]
    else:
        new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
        embeddings_index[word] = new_embedding
        word_embedding_matrix[i] = new_embedding

# Check if value matches len(vocab_to_int)
print(len(word_embedding_matrix))


# In[213]:

def convert_to_ints(text, word_count, unk_count, eos=False):

    ints = []
    for sentence in text:
        sentence_ints = []
        for word in sentence.split():
            word_count += 1
            if word in vocab_to_int:
                sentence_ints.append(vocab_to_int[word])
            else:
                sentence_ints.append(vocab_to_int["<UNK>"])
                unk_count += 1
        if eos:
            sentence_ints.append(vocab_to_int["<EOS>"])
        ints.append(sentence_ints)
    return ints, word_count, unk_count


# In[214]:

word_count = 0
unk_count = 0

int_summaries, word_count, unk_count = convert_to_ints(clean_summaries, word_count, unk_count)
int_texts, word_count, unk_count = convert_to_ints(clean_texts, word_count, unk_count, eos=True)

unk_percent = round(unk_count/word_count,4)*100

print("Total number of words in headlines:", word_count)
print("Total number of UNKs in headlines:", unk_count)
print("Percent of words that are UNK: {}%".format(unk_percent))


# In[215]:

def create_lengths(text):

    lengths = []
    for sentence in text:
        lengths.append(len(sentence))
    return pd.DataFrame(lengths, columns=['counts'])


# In[216]:

lengths_summaries = create_lengths(int_summaries)
lengths_texts = create_lengths(int_texts)

print("Summaries:")
print(lengths_summaries.describe())
print()
print("Texts:")
print(lengths_texts.describe())


# In[217]:

# Inspect the length of texts
print(np.percentile(lengths_texts.counts, 90))
print(np.percentile(lengths_texts.counts, 95))
print(np.percentile(lengths_texts.counts, 99))


# In[218]:

# Inspect the length of summaries
print(np.percentile(lengths_summaries.counts, 90))
print(np.percentile(lengths_summaries.counts, 95))
print(np.percentile(lengths_summaries.counts, 99))


# In[219]:

def unk_counter(sentence):
    unk_count = 0
    for word in sentence:
        if word == vocab_to_int["<UNK>"]:
            unk_count += 1
    return unk_count


# In[220]:

# Sort the summaries and texts by the length of the texts, shortest to longest
# Limit the length of summaries and texts based on the min and max ranges.
# Remove reviews that include too many UNKs

sorted_summaries = []
sorted_texts = []
max_text_length = 84
max_summary_length = 13
min_length = 2
unk_text_limit = 1
unk_summary_limit = 0

for length in range(min(lengths_texts.counts), max_text_length):
    for count, words in enumerate(int_summaries):
        if (len(int_summaries[count]) >= min_length and
            len(int_summaries[count]) <= max_summary_length and
            len(int_texts[count]) >= min_length and
            unk_counter(int_summaries[count]) <= unk_summary_limit and
            unk_counter(int_texts[count]) <= unk_text_limit and
            length == len(int_texts[count])
           ):
            sorted_summaries.append(int_summaries[count])
            sorted_texts.append(int_texts[count])

# Compare lengths to ensure they match
print(len(sorted_summaries))
print(len(sorted_texts))


# ## Building the Model

# In[221]:

def model_inputs():


    input_data = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    summary_length = tf.placeholder(tf.int32, (None,), name='summary_length')
    max_summary_length = tf.reduce_max(summary_length, name='max_dec_len')
    text_length = tf.placeholder(tf.int32, (None,), name='text_length')

    return input_data, targets, lr, keep_prob, summary_length, max_summary_length, text_length


# In[222]:

def process_encoding_input(target_data, vocab_to_int, batch_size):


    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

    return dec_input


# In[223]:

def encoding_layer(rnn_size, sequence_length, num_layers, rnn_inputs, keep_prob):


    for layer in range(num_layers):
        with tf.variable_scope('encoder_{}'.format(layer)):
            cell_fw = tf.contrib.rnn.LSTMCell(rnn_size,
                                              initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw,
                                                    input_keep_prob = keep_prob)

            cell_bw = tf.contrib.rnn.LSTMCell(rnn_size,
                                              initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw,
                                                    input_keep_prob = keep_prob)

            enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                    cell_bw,
                                                                    rnn_inputs,
                                                                    sequence_length,
                                                                    dtype=tf.float32)

    enc_output = tf.concat(enc_output,2)

    return enc_output, enc_state


# In[224]:

def training_decoding_layer(dec_embed_input, summary_length, dec_cell, initial_state, output_layer,
                            vocab_size, max_summary_length):


    training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                        sequence_length=summary_length,
                                                        time_major=False)

    training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                       training_helper,
                                                       initial_state,
                                                       output_layer)

    training_logits, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                           output_time_major=False,
                                                           impute_finished=True,
                                                           maximum_iterations=max_summary_length)
    return training_logits


# In[225]:

def inference_decoding_layer(embeddings, start_token, end_token, dec_cell, initial_state, output_layer,
                             max_summary_length, batch_size):


    start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [batch_size], name='start_tokens')

    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,
                                                                start_tokens,
                                                                end_token)

    inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                        inference_helper,
                                                        initial_state,
                                                        output_layer)

    inference_logits, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                            output_time_major=False,
                                                            impute_finished=True,
                                                            maximum_iterations=max_summary_length)

    return inference_logits


# In[226]:

def decoding_layer(dec_embed_input, embeddings, enc_output, enc_state, vocab_size, text_length, summary_length,
                   max_summary_length, rnn_size, vocab_to_int, keep_prob, batch_size, num_layers):


    for layer in range(num_layers):
        with tf.variable_scope('decoder_{}'.format(layer)):
            lstm = tf.contrib.rnn.LSTMCell(rnn_size,
                                           initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            dec_cell = tf.contrib.rnn.DropoutWrapper(lstm,
                                                     input_keep_prob = keep_prob)

    output_layer = Dense(vocab_size,
                         kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))

    attn_mech = tf.contrib.seq2seq.BahdanauAttention(rnn_size,
                                                  enc_output,
                                                  text_length,
                                                  normalize=False,
                                                  name='BahdanauAttention')

    dec_cell = tf.contrib.seq2seq.DynamicAttentionWrapper(dec_cell,
                                                          attn_mech,
                                                          rnn_size)

    initial_state = tf.contrib.seq2seq.DynamicAttentionWrapperState(enc_state[0],
                                                                    _zero_state_tensors(rnn_size,
                                                                                        batch_size,
                                                                                        tf.float32))
    with tf.variable_scope("decode"):
        training_logits = training_decoding_layer(dec_embed_input,
                                                  summary_length,
                                                  dec_cell,
                                                  initial_state,
                                                  output_layer,
                                                  vocab_size,
                                                  max_summary_length)
    with tf.variable_scope("decode", reuse=True):
        inference_logits = inference_decoding_layer(embeddings,
                                                    vocab_to_int['<GO>'],
                                                    vocab_to_int['<EOS>'],
                                                    dec_cell,
                                                    initial_state,
                                                    output_layer,
                                                    max_summary_length,
                                                    batch_size)

    return training_logits, inference_logits


# In[227]:

def seq2seq_model(input_data, target_data, keep_prob, text_length, summary_length, max_summary_length,
                  vocab_size, rnn_size, num_layers, vocab_to_int, batch_size):

    embeddings = word_embedding_matrix

    enc_embed_input = tf.nn.embedding_lookup(embeddings, input_data)
    enc_output, enc_state = encoding_layer(rnn_size, text_length, num_layers, enc_embed_input, keep_prob)

    dec_input = process_encoding_input(target_data, vocab_to_int, batch_size)
    dec_embed_input = tf.nn.embedding_lookup(embeddings, dec_input)

    training_logits, inference_logits  = decoding_layer(dec_embed_input,
                                                        embeddings,
                                                        enc_output,
                                                        enc_state,
                                                        vocab_size,
                                                        text_length,
                                                        summary_length,
                                                        max_summary_length,
                                                        rnn_size,
                                                        vocab_to_int,
                                                        keep_prob,
                                                        batch_size,
                                                        num_layers)

    return training_logits, inference_logits


# In[228]:

def pad_sentence_batch(sentence_batch):

    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]


# In[229]:

def get_batches(summaries, texts, batch_size):

    for batch_i in range(0, len(texts)//batch_size):
        start_i = batch_i * batch_size
        summaries_batch = summaries[start_i:start_i + batch_size]
        texts_batch = texts[start_i:start_i + batch_size]
        pad_summaries_batch = np.array(pad_sentence_batch(summaries_batch))
        pad_texts_batch = np.array(pad_sentence_batch(texts_batch))

        pad_summaries_lengths = []
        for summary in pad_summaries_batch:
            pad_summaries_lengths.append(len(summary))

        pad_texts_lengths = []
        for text in pad_texts_batch:
            pad_texts_lengths.append(len(text))

        yield pad_summaries_batch, pad_texts_batch, pad_summaries_lengths, pad_texts_lengths


# In[230]:

epochs = 100
batch_size = 64
rnn_size = 256
num_layers = 2
learning_rate = 0.005
keep_probability = 0.75


# In[231]:

# Build the graph
train_graph = tf.Graph()

with train_graph.as_default():

    input_data, targets, lr, keep_prob, summary_length, max_summary_length, text_length = model_inputs()

    training_logits, inference_logits = seq2seq_model(tf.reverse(input_data, [-1]),
                                                      targets,
                                                      keep_prob,
                                                      text_length,
                                                      summary_length,
                                                      max_summary_length,
                                                      len(vocab_to_int)+1,
                                                      rnn_size,
                                                      num_layers,
                                                      vocab_to_int,
                                                      batch_size)

    training_logits = tf.identity(training_logits.rnn_output, 'logits')
    inference_logits = tf.identity(inference_logits.sample_id, name='predictions')

    masks = tf.sequence_mask(summary_length, max_summary_length, dtype=tf.float32, name='masks')

    with tf.name_scope("optimization"):

        cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            targets,
            masks)

        optimizer = tf.train.AdamOptimizer(learning_rate)

        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)
print("Graph is built.")


# ## Training the Model


# In[234]:

# Subset the data for training
start = 200000
end = start + 50000
sorted_summaries_short = sorted_summaries[start:end]
sorted_texts_short = sorted_texts[start:end]
print("The shortest text length:", len(sorted_texts_short[0]))
print("The longest text length:",len(sorted_texts_short[-1]))


# In[258]:

# Train the Model
learning_rate_decay = 0.95
min_learning_rate = 0.0005
display_step = 20
stop_early = 0
stop = 3
per_epoch = 3
update_check = (len(sorted_texts_short)//batch_size//per_epoch)-1

update_loss = 0
batch_loss = 0
summary_update_loss = []

checkpoint = "best_model.ckpt"
with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())



    for epoch_i in range(1, epochs+1):
        update_loss = 0
        batch_loss = 0
        for batch_i, (summaries_batch, texts_batch, summaries_lengths, texts_lengths) in enumerate(
                get_batches(sorted_summaries_short, sorted_texts_short, batch_size)):
            start_time = time.time()
            _, loss = sess.run(
                [train_op, cost],
                {input_data: texts_batch,
                 targets: summaries_batch,
                 lr: learning_rate,
                 summary_length: summaries_lengths,
                 text_length: texts_lengths,
                 keep_prob: keep_probability})

            batch_loss += loss
            update_loss += loss
            end_time = time.time()
            batch_time = end_time - start_time

            if batch_i % display_step == 0 and batch_i > 0:
                print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                      .format(epoch_i,
                              epochs,
                              batch_i,
                              len(sorted_texts_short) // batch_size,
                              batch_loss / display_step,
                              batch_time*display_step))
                batch_loss = 0

            if batch_i % update_check == 0 and batch_i > 0:
                print("Average loss for this update:", round(update_loss/update_check,3))
                summary_update_loss.append(update_loss)

                if update_loss <= min(summary_update_loss):
                    print('New Record!')
                    stop_early = 0
                    saver = tf.train.Saver()
                    saver.save(sess, checkpoint)

                else:
                    print("No Improvement.")
                    stop_early += 1
                    if stop_early == stop:
                        break
                update_loss = 0

        learning_rate *= learning_rate_decay
        if learning_rate < min_learning_rate:
            learning_rate = min_learning_rate

        if stop_early == stop:
            print("Stopping Training.")
            break




# In[264]:

def text_to_seq(text):

    text = clean_text(text)
    return [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in text.split()]


# In[267]:

input_sentence = "i love this food" #@param {type:"string"}


print('\nOriginal Text:', input_sentence)
text = text_to_seq(input_sentence)
checkpoint = "./best_model.ckpt"

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(checkpoint + '.meta')
    loader.restore(sess, checkpoint)

    input_data = loaded_graph.get_tensor_by_name('input:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    text_length = loaded_graph.get_tensor_by_name('text_length:0')
    summary_length = loaded_graph.get_tensor_by_name('summary_length:0')
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
    answer_logits = sess.run(logits, {input_data: [text]*batch_size,
                                      summary_length: [np.random.randint(5,8)],
                                      text_length: [len(text)]*batch_size,
                                      keep_prob: 1.0})[0]

pad = vocab_to_int["<PAD>"]

print('Original Text:', input_sentence)

print('\nText')
print('  Word Ids:    {}'.format([i for i in text]))
print('  Input Words: {}'.format(" ".join([int_to_vocab[i] for i in text])))

print('\nSummary')
print('  Word Ids:       {}'.format([i for i in answer_logits if i != pad]))
print('  Response Words: {}'.format(" ".join([int_to_vocab[i] for i in answer_logits if i != pad])))

#html file for front view of tool
%mkdir templates -p
%%writefile templates/portfolio.html
<!doctype html>
<html class="no-js" lang="zxx">
<head>
    <meta charset="utf-8">
    <meta http-equiv="x-ua-compatible" content="ie=edge">
    <title>Text-Summarization</title>
    <meta name="description" content="">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
    body {
padding:0 0.5em;
margin:0;
font-family:Verdana, Helvetica, Arial, sans-serif;
background-color:#eee;
}
#header, #main, #footer {
position:relative;
padding:0.5em;
margin:0.5em auto;
-moz-border-radius:10px;
-webkit-border-radius:10px;
border-radius:10px;
background-color:#a6dba0;
box-shadow:0 0 3px #000;
width:800px;
}

#tc-logo {
margin: 0 auto;
text-align: center;
}
#logo-text {
-webkit-transition: all 1s ease 0s;
letter-spacing: 4px;
display: inline-block;
font-weight: bold;
font-size: 2em;
margin:0;
padding:0 4px 0 8px;
}
#tc-logo:before, #tc-logo:after {
display: inline-block;
content: "";
background: #b5bdc8;
background: -moz-linear-gradient(top,  #b5bdc8 0%, #828c95 36%, #28343b 100%);
background: -webkit-gradient(linear, left top, left bottom, color-stop(0%,#b5bdc8), color-stop(36%,#828c95), color-stop(100%,#28343b));
background: -webkit-linear-gradient(top,  #b5bdc8 0%,#828c95 36%,#28343b 100%);
background: -o-linear-gradient(top,  #b5bdc8 0%,#828c95 36%,#28343b 100%);
background: -ms-linear-gradient(top,  #b5bdc8 0%,#828c95 36%,#28343b 100%);
background: linear-gradient(top,  #b5bdc8 0%,#828c95 36%,#28343b 100%);
filter: progid:DXImageTransform.Microsoft.gradient( startColorstr='#b5bdc8', endColorstr='#28343b',GradientType=0 );
-moz-background-clip: padding;
-webkit-background-clip: padding;
background-clip: padding-box;
width: 40px;
height: 25px;
vertical-align: -25%;
border-top: 10px solid transparent;
border-bottom: 10px solid transparent;
}
#tc-logo:before {
border-right: 10px solid #555;
}
#tc-logo:after {
border-left: 10px solid #555;
}
#tc-logo:hover #logo-text {
letter-spacing: -3px;
padding: 0;
}
#header h2 {
margin:0;
padding:0;
text-align:center;
font-size:1.2em;
font-weight:normal;
}
#nav {
position:absolute;
top:0.5em;
right:0.5em;
}
#nav ul {
padding:0;
margin:0;
list-style:none;
text-align:right;
}
#nav li {
padding:0.2em;
margin-bottom:0.8em;
}
#nav li a{
text-decoration:none;
color:#000;
padding:0.4em;
margin-bottom:0.4em;
background-color:#DDD;
border:1px solid #999;
font-weight:bold;
}
#nav li a:hover {
background-color:#FFF;
}
#nav .current {
background-color:#FFF;
}
#description {
padding:0.5em;
}
#description p {
padding:0;
margin:0;
font-size:1.1em;
}
.step {
padding:0.5em;
margin:0.5em 0;
-moz-border-radius:10px;
-webkit-border-radius:10px;
-khtml-border-radius:10px;
border-radius:10px;
background-color:#d9f0d3;
border:2px solid #999;
}
.step h3 {
margin:0;
padding:0;
}
.step label {
display:block;
margin-bottom:0.2em;
}
.step p {
margin-top:0;
margin-bottom:0.6em;
}
.step textarea {
border:1px solid #000;
background-color: #FFF;
box-sizing: border-box;
width: 100%;
}
#range-cont {
position:relative;
height:35px;
}
#output {
border:1px solid #000;
background-color: #FFF;
padding:0.5em;
}
#output p {
margin-top:0.5em;
}
#output p:first-child {
margin-top:0;
}
#globalerror {
padding:0 0.5em;
margin:0.5em;
}
#globalerror p {
padding:0;
margin:0;
font-size:1.1em;
}
#js-globalerror {
display:none;
position:fixed;
top:0;
width:100%;
}
#js-globalerror p {
margin-left:auto;
margin-right:auto;
margin-top:0;
padding:1em;
text-align:center;
background-color:#FFF;
border:1px solid #AAA;
width:20em;
}
.error {
color:#FF0000;
font-weight:bold;
}
#footer p {
margin:0;
padding:0;
text-align:center;
}
/* slider root element */
.slider {
height:9px;
position:relative;
cursor:pointer;
border:1px solid #333;
width:675px;
float:left;
clear:right;
margin-top:10px;
-moz-border-radius:5px;
-webkit-border-radius:5px;
border-radius:5px;
}
/* progress bar (enabled with progress: true) */
.progress {
height:9px;
background-color:#555;
-moz-border-radius:5px;
-webkit-border-radius:5px;
border-radius:5px;
}
#percent-label {
float:left;
padding:4px;
font-size:20px;
}
/* drag handle */
.handle {
background:#fff;
height:28px;
width:28px;
top:-12px;
position:absolute;
display:block;
margin-top:1px;
border:1px solid #000;
cursor:move;
-moz-box-shadow:0 0 6px #000;
-webkit-box-shadow:0 0 6px #000;
box-shadow:0 0 6px #000;
-moz-border-radius:14px;
-webkit-border-radius:14px;
border-radius:14px;
}
/* the input field */
.range {
border:1px inset #ddd;
float:left;
font-size:20px;
margin:0 0 0 15px;
padding:3px 0;
text-align:center;
width:50px;
-moz-border-radius:5px;
-webkit-border-radius:5px;
border-radius:5px;
}
.button {
background: #90EE90;
padding: 15px 32px;
color: #fff;
border: 1px solid #eee;
border-radius: 20px;
box-shadow: 5px 5px 5px #eee;
text-shadow:none;
}
.button:hover {
background: #4CAF50;
padding: 15px 32px;
color: #fff;
border: 1px solid #eee;
border-radius: 20px;
box-shadow: 5px 5px 5px #eee;
text-shadow:none;
}
.btn{
    text-align:center
}
</style>
    <!-- <link rel="stylesheet" href="css/responsive.css"> -->
</head>
<body>
<div class="bg-img">
<div id="header">
    <header class="mdl-layout__header mdl-color-text--white mdl-color--light-blue-700">
      <div class="mdl-cell mdl-cell--12-col mdl-cell--12-col-tablet mdl-grid">
        <div class="mdl-layout__header-row mdl-cell mdl-cell--12-col mdl-cell--12-col-tablet mdl-cell--12-col-desktop">
    <div id="tc-logo"><h1 id="logo-text">Text Compactor</h1></div>
    <h2> Automatic Text Summarization Tool</h2>
    </div>
    </div>
    </header>
</div>
<div id="main">
<form method="post" action="{{ url_for('result') }}">
    <div class="step">

        <label for="text_in"><strong>Type or paste your text into the box.</strong></label>
        <textarea cols="60" id="text_in" name="text_in" rows="15" placeholder="enter text here">{{ text_in }}</textarea>
    </div>

    <div class="btn">
                    <button class="button" type="submit" id="button">
                        Submit
                    </button>
                </div>
     <h2>Summary</h2>
    <p>{{ text_out }}</p>
</form>
<br>
<br>
{{ prediction_text }}
</div>
</div>
</body>
</html>


#Flask Part
%%writefile requirements.txt
Flask==0.12.2
flask-socketio
eventlet==0.17.4
gunicorn==18.0.0
!pip install flask-ngrok
from flask_ngrok import run_with_ngrok
from flask import Flask, jsonify, request,render_template
import numpy as np
import tensorflow as tf
import os
import logging
import random
import time
app = Flask(__name__)
run_with_ngrok(app)
app.config.from_object(__name__)


checkpoint = "/content/drive/My Drive/Text_Summarization/best_model.ckpt"

loaded_graph = tf.Graph()
tf.train.write_graph(loaded_graph, "colab ", "tf_model.pb", as_text=False)

@app.route('/',methods=['GET', 'POST'])
def predict():
    return render_template('portfolio.html',text_in='', text_out='')

@app.route('/result/', methods=['GET', 'POST'])
def result():
    input_sentence = request.form['text_in']
    text = text_to_seq(input_sentence)
    with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
      loader = tf.train.import_meta_graph("/content/drive/My Drive/Text_Summarization/best_model.ckpt.meta")
      #loader = tf.train.import_meta_graph("/content/drive/My Drive/Text_Summarization/Trained Model/best_model.ckpt.meta")
      loader.restore(sess, checkpoint)

      input_data = loaded_graph.get_tensor_by_name('input:0')
      logits = loaded_graph.get_tensor_by_name('predictions:0')
      text_length = loaded_graph.get_tensor_by_name('text_length:0')
      summary_length = loaded_graph.get_tensor_by_name('summary_length:0')
      keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

    #Multiply by batch_size to match the model's input parameters
      answer_logits = sess.run(logits, {input_data: [text]*batch_size,
                                      summary_length: [np.random.randint(5,8)],
                                      text_length: [len(text)]*batch_size,
                                      keep_prob: 1.0})[0]

      pad = vocab_to_int["<PAD>"]
    return render_template('portfolio.html', text_in=input_sentence, text_out='{}'.format(" ".join([int_to_vocab[i] for i in answer_logits if i != pad])))
app.run()
