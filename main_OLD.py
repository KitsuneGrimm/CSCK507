import pandas as pd
import numpy as np
import string
import nltk
import tensorflow as tf
import os
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, AdditiveAttention, Concatenate, Dropout, Bidirectional
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer

# Ensure you have the necessary NLTK data files
nltk.download('stopwords')
nltk.download('wordnet')

# Enable memory growth for GPUs
def enable_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

# Preprocess text
def preprocess_text(text):
    stemmer = PorterStemmer()

    # Lowercase the text
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize
    tokens = text.split()

    # Stem tokens
    tokens = [stemmer.stem(word) for word in tokens]

    return ' '.join(tokens)

# Load and preprocess dataset
def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath, sep='\t')
    data['Original Question'] = data['Question']
    data['Original Sentence'] = data['Sentence']
    data['Question'] = data['Question'].apply(preprocess_text)
    data['Sentence'] = data['Sentence'].apply(preprocess_text)
    data['Label'] = data['Label'].apply(lambda x: 'startseq ' + str(x) + ' endseq')

    le_doc_title = LabelEncoder()
    data['DocumentTitle'] = le_doc_title.fit_transform(data['DocumentTitle'])

    return data, le_doc_title

# Tokenize and pad sequences
def tokenize_and_pad(data, tokenizer, max_seq_length):
    # Combine question and answer tokenization and padding
    questions_seq = tokenizer.texts_to_sequences(data['Question'].tolist())
    answers_seq = tokenizer.texts_to_sequences(data['Sentence'].tolist())

    questions_padded = pad_sequences(questions_seq, maxlen=max_seq_length, padding='post')
    answers_padded = pad_sequences(answers_seq, maxlen=max_seq_length, padding='post')

    return questions_padded, answers_padded

# Load pre-trained embedding matrix
def load_embedding_matrix(word_index, embedding_dim, embedding_file_path):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))

    with open(embedding_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = coefs

    return embedding_matrix

# Build Seq2Seq model
def build_seq2seq_model(vocab_size, embedding_matrix, lstm_units, attention=False):
    clear_session()

    # Encoder
    encoder_inputs = Input(shape=(None,), name='encoder_inputs')
    encoder_embedding = Embedding(vocab_size, embedding_matrix.shape[1], weights=[embedding_matrix], trainable=False)(encoder_inputs)
    encoder_lstm = Bidirectional(LSTM(lstm_units, return_sequences=attention, return_state=True, dropout=0.3, recurrent_dropout=0.3, kernel_regularizer=l2(1e-6)), name='bidirectional_lstm')
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm(encoder_embedding)
    state_h = Concatenate(name='state_h')([forward_h, backward_h])
    state_c = Concatenate(name='state_c')([forward_c, backward_c])

    # Decoder
    decoder_inputs = Input(shape=(None,), name='decoder_inputs')
    decoder_embedding = Embedding(vocab_size, embedding_matrix.shape[1], weights=[embedding_matrix], trainable=False, name='decoder_embedding')(decoder_inputs)
    decoder_lstm = LSTM(lstm_units * 2, return_sequences=True, return_state=True, dropout=0.3, recurrent_dropout=0.3, kernel_regularizer=l2(1e-6), name='decoder_lstm')

    if attention:
        attention_layer = AdditiveAttention(name='additive_attention')
        decoder_lstm_output, _, _ = decoder_lstm(decoder_embedding, initial_state=[state_h, state_c])
        attention_output = attention_layer([decoder_lstm_output, encoder_outputs])
        decoder_concat_input = Concatenate(axis=-1, name='decoder_concat')([decoder_lstm_output, attention_output])
        decoder_dense = Dense(vocab_size, activation='softmax', kernel_regularizer=l2(1e-6), name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_concat_input)
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        return model, encoder_inputs, encoder_outputs, state_h, state_c, decoder_inputs, decoder_lstm, decoder_dense, attention_layer
    else:
        decoder_lstm_output, _, _ = decoder_lstm(decoder_embedding, initial_state=[state_h, state_c])
        decoder_dense = Dense(vocab_size, activation='softmax', kernel_regularizer=l2(1e-6), name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_lstm_output)
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        return model, encoder_inputs, state_h, state_c, decoder_inputs, decoder_lstm, decoder_dense


def evaluate_metrics(true_answers, pred_answers):
    smoothing_function = SmoothingFunction().method1
    rouge = Rouge()

    bleu_scores = []
    rouge_scores = []
    meteor_scores = []

    for true_answer, pred_answer in zip(true_answers, pred_answers):
        true_answer_tokens = true_answer.split()
        pred_answer_tokens = pred_answer.split()
        bleu_score_value = sentence_bleu([true_answer_tokens], pred_answer_tokens, smoothing_function=smoothing_function)
        rouge_score = rouge.get_scores(pred_answer, true_answer, avg=True)['rouge-l']['f']
        meteor = meteor_score([true_answer], pred_answer)

        bleu_scores.append(bleu_score_value)
        rouge_scores.append(rouge_score)
        meteor_scores.append(meteor)

    avg_bleu = np.mean(bleu_scores)
    avg_rouge = np.mean(rouge_scores)
    avg_meteor = np.mean(meteor_scores)

    return avg_bleu, avg_rouge, avg_meteor

def train_model_with_callbacks(model, train_data, val_data, tokenizer, max_seq_length, batch_size, epochs, callbacks, encoder_model, decoder_model, attention):
    train_questions_padded, train_answers_padded = tokenize_and_pad(train_data, tokenizer, max_seq_length)
    val_questions_padded, val_answers_padded = tokenize_and_pad(val_data, tokenizer, max_seq_length)

    decoder_input_data = np.zeros_like(train_answers_padded)
    decoder_input_data[:, 1:] = train_answers_padded[:, :-1]
    decoder_input_data[:, 0] = tokenizer.word_index['startseq']

    val_decoder_input_data = np.zeros_like(val_answers_padded)
    val_decoder_input_data[:, 1:] = val_answers_padded[:, :-1]
    val_decoder_input_data[:, 0] = tokenizer.word_index['startseq']

    train_decoder_target_data = np.expand_dims(train_answers_padded, -1)
    val_decoder_target_data = np.expand_dims(val_answers_padded, -1)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

    model.fit(
        [train_questions_padded, decoder_input_data],
        train_decoder_target_data,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=([val_questions_padded, val_decoder_input_data], val_decoder_target_data),
        callbacks=callbacks
    )

    val_pred_answers = [decode_sequence(np.expand_dims(question_padded, axis=0), encoder_model, decoder_model, tokenizer, max_seq_length, attention=attention) for question_padded in val_questions_padded]
    true_answers = val_data['Sentence'].tolist()
    avg_bleu, avg_rouge, avg_meteor = evaluate_metrics(true_answers, val_pred_answers)

    print(f"Validation BLEU Score: {avg_bleu}")
    print(f"Validation ROUGE Score: {avg_rouge}")
    print(f"Validation METEOR Score: {avg_meteor}")

def load_or_train_model(model_path, vocab_size, embedding_matrix, lstm_units, max_seq_length, train_data, val_data, tokenizer, attention=False):
    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}")
        model = tf.keras.models.load_model(model_path, custom_objects={'AdditiveAttention': AdditiveAttention})
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    else:
        if attention:
            print(f"Training new model with attention and saving to {model_path}")
            model, encoder_inputs, encoder_outputs, state_h, state_c, decoder_inputs, decoder_lstm, decoder_dense, attention_layer = build_seq2seq_model(vocab_size, embedding_matrix, lstm_units, attention=True)
        else:
            print(f"Training new model without attention and saving to {model_path}")
            model, encoder_inputs, state_h, state_c, decoder_inputs, decoder_lstm, decoder_dense = build_seq2seq_model(vocab_size, embedding_matrix, lstm_units, attention=False)

        encoder_model, decoder_model = prepare_inference_models(model, lstm_units, attention=attention)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-5)
        ]

        train_model_with_callbacks(model, train_data, val_data, tokenizer, max_seq_length, batch_size=32, epochs=50, callbacks=callbacks, encoder_model=encoder_model, decoder_model=decoder_model, attention=attention)
        model.save(model_path)

    return model

# Prepare inference models
def prepare_inference_models(model, lstm_units, attention=False):
    # Encoder model
    encoder_inputs = model.input[0]
    encoder_lstm_layer = model.get_layer('bidirectional_lstm')
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm_layer.output
    state_h_enc = Concatenate(name='state_h_enc')([forward_h, backward_h])
    state_c_enc = Concatenate(name='state_c_enc')([forward_c, backward_c])

    if attention:
        encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs, state_h_enc, state_c_enc])
    else:
        encoder_model = Model(inputs=encoder_inputs, outputs=[state_h_enc, state_c_enc])

    # Decoder model
    decoder_inputs = model.input[1]
    decoder_state_input_h = Input(shape=(lstm_units * 2,), name='decoder_state_input_h')
    decoder_state_input_c = Input(shape=(lstm_units * 2,), name='decoder_state_input_c')
    decoder_embedding_layer = model.get_layer('decoder_embedding')
    decoder_embedding_output = decoder_embedding_layer(decoder_inputs)
    decoder_lstm_layer = model.get_layer('decoder_lstm')
    decoder_lstm_output, state_h_dec, state_c_dec = decoder_lstm_layer(decoder_embedding_output, initial_state=[decoder_state_input_h, decoder_state_input_c])

    if attention:
        attention_layer = model.get_layer('additive_attention')
        attention_output = attention_layer([decoder_lstm_output, encoder_outputs])
        decoder_concat_input = Concatenate(axis=-1, name='decoder_concat')([decoder_lstm_output, attention_output])
        decoder_dense_layer = model.get_layer('decoder_dense')
        decoder_outputs = decoder_dense_layer(decoder_concat_input)
        decoder_model = Model([decoder_inputs, encoder_outputs, decoder_state_input_h, decoder_state_input_c], [decoder_outputs, state_h_dec, state_c_dec])
    else:
        decoder_dense_layer = model.get_layer('decoder_dense')
        decoder_outputs = decoder_dense_layer(decoder_lstm_output)
        decoder_model = Model([decoder_inputs, decoder_state_input_h, decoder_state_input_c], [decoder_outputs, state_h_dec, state_c_dec])

    return encoder_model, decoder_model

# Decode sequence
def decode_sequence(input_seq, encoder_model, decoder_model, tokenizer, max_seq_length, attention=False):
    if attention:
        encoder_outputs, state_h, state_c = encoder_model.predict(input_seq)
    else:
        state_h, state_c = encoder_model.predict(input_seq)

    states_value = [state_h, state_c]
    target_seq = np.array([[tokenizer.word_index['startseq']]])
    decoded_sentence = []

    for _ in range(max_seq_length):
        if attention:
            output_tokens, h, c = decoder_model.predict([target_seq, encoder_outputs] + states_value, verbose=0)
        else:
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tokenizer.index_word.get(sampled_token_index, '')

        if sampled_word == 'endseq':
            break

        decoded_sentence.append(sampled_word)
        target_seq = np.array([[sampled_token_index]])
        states_value = [h, c]

    return ' '.join(decoded_sentence)

# Evaluate model
def evaluate_model(test_data, encoder_model, decoder_model, tokenizer, max_seq_length, attention=False, batch_size=32):
    results = []
    smoothing_function = SmoothingFunction().method1
    rouge = Rouge()

    test_questions_seq = tokenizer.texts_to_sequences(test_data['Question'])
    test_questions_padded = pad_sequences(test_questions_seq, maxlen=max_seq_length, padding='post')

    num_batches = int(np.ceil(len(test_data) / batch_size))

    print(f"Evaluating the model on {len(test_data)} questions in {num_batches} batches.")

    for batch_idx in tqdm(range(num_batches), desc="Evaluating model", unit="batch"):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, len(test_data))

        batch_questions_padded = test_questions_padded[batch_start:batch_end]
        batch_true_answers = test_data['Sentence'].iloc[batch_start:batch_end].tolist()
        batch_original_questions = test_data['Original Question'].iloc[batch_start:batch_end].tolist()
        batch_original_sentences = test_data['Original Sentence'].iloc[batch_start:batch_end].tolist()
        batch_question_ids = test_data['QuestionID'].iloc[batch_start:batch_end].tolist()

        batch_pred_answers = [decode_sequence(np.expand_dims(question_padded, axis=0), encoder_model, decoder_model, tokenizer, max_seq_length, attention=attention) for question_padded in batch_questions_padded]

        for i in range(len(batch_pred_answers)):
            true_answer = batch_true_answers[i]
            pred_answer = batch_pred_answers[i]
            bleu_score = sentence_bleu([true_answer.split()], pred_answer.split(), smoothing_function=smoothing_function)
            rouge_score = rouge.get_scores(pred_answer, true_answer)[0]['rouge-l']['f']
            meteor_score_value = meteor_score([true_answer], pred_answer)
            is_correct = int(true_answer.strip() == pred_answer.strip())
            question_len = len(batch_original_questions[i].split())

            results.append({
                'QuestionID': batch_question_ids[i],
                'Original Question': batch_original_questions[i],
                'Original Sentence': batch_original_sentences[i],
                'Predicted Sentence': pred_answer,
                'BLEU Score': bleu_score,
                'ROUGE Score': rouge_score,
                'METEOR Score': meteor_score_value,
                'Correct': is_correct,
                'Question Length': question_len
            })

    print("Evaluation completed.")
    return pd.DataFrame(results)

# Method to plot comparison graphs
def plot_comparison_graphs(combined_results):
    plt.figure(figsize=(14, 7))
    plt.title("BLEU Scores Comparison")
    plt.xlabel("Question Length")
    plt.ylabel("BLEU Score")
    plt.scatter(combined_results[combined_results['Model'] == 'No Attention']['Question Length'], combined_results[combined_results['Model'] == 'No Attention']['BLEU Score'], label='No Attention', alpha=0.6)
    plt.scatter(combined_results[combined_results['Model'] == 'Attention']['Question Length'], combined_results[combined_results['Model'] == 'Attention']['BLEU Score'], label='Attention', alpha=0.6)
    plt.legend()
    plt.show()

    plt.figure(figsize=(14, 7))
    plt.title("Accuracy Comparison")
    plt.xlabel("Question Length")
    plt.ylabel("Accuracy")
    no_attention_accuracy = combined_results[combined_results['Model'] == 'No Attention']['Correct'].mean()
    attention_accuracy = combined_results[combined_results['Model'] == 'Attention']['Correct'].mean()
    plt.bar(['No Attention', 'Attention'], [no_attention_accuracy, attention_accuracy])
    plt.show()

# Main execution
def main():
    enable_memory_growth()

    # File paths and parameters
    data_path = 'Data/WikiQA-train.tsv'
    test_data_path = 'Data/WikiQA-test.tsv'
    save_dir = 'Model/'
    predict_dir = 'Predictions/'
    model_path_no_attention = os.path.join(save_dir, 'seq2seq_no_attention_model.h5')
    model_path_attention = os.path.join(save_dir, 'seq2seq_attention_model.h5')
    predictions_path_no_attention = os.path.join(predict_dir, 'WikiQA-Predictions-Seq2Seq-no_attention.csv')
    predictions_path_attention = os.path.join(predict_dir, 'WikiQA-Predictions-Seq2Seq-attention.csv')
    embedding_file_path = 'Embedding/glove.6B.300d.txt'
    embedding_dim = 300
    lstm_units = 1024

    # Load and preprocess data
    data, _ = load_and_preprocess_data(data_path)

    # Add 'startseq' and 'endseq' tokens to the questions and sentences
    data['Question'] = data['Question'].apply(lambda x: 'startseq ' + x + ' endseq')
    data['Sentence'] = data['Sentence'].apply(lambda x: 'startseq ' + x + ' endseq')

    # Split data into training and validation sets
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    # Tokenize and pad sequences
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_data['Question'].tolist() + train_data['Sentence'].tolist())

    vocab_size = len(tokenizer.word_index) + 1
    max_seq_length = max(
        max(len(seq.split()) for seq in train_data['Question']),
        max(len(seq.split()) for seq in train_data['Sentence'])
    )

    # Assume embedding_matrix is loaded from pre-trained embeddings
    embedding_matrix = load_embedding_matrix(tokenizer.word_index, embedding_dim, embedding_file_path)

    # Load or train model without attention
    model_no_attention = load_or_train_model(model_path_no_attention, vocab_size, embedding_matrix, lstm_units, max_seq_length, train_data, val_data, tokenizer, attention=False)
    encoder_model_no_attention, decoder_model_no_attention = prepare_inference_models(model_no_attention, lstm_units, attention=False)

    # Load or train model with attention
    model_attention = load_or_train_model(model_path_attention, vocab_size, embedding_matrix, lstm_units, max_seq_length, train_data, val_data, tokenizer, attention=True)
    encoder_model_attention, decoder_model_attention = prepare_inference_models(model_attention, lstm_units, attention=True)

    # Preprocess and prepare test data for evaluation
    test_data, _ = load_and_preprocess_data(test_data_path)
    test_data['Question'] = test_data['Question'].apply(lambda x: 'startseq ' + x + ' endseq')
    test_data['Sentence'] = test_data['Sentence'].apply(lambda x: 'startseq ' + x + ' endseq')
    test_questions_list = test_data['Question'].tolist()
    test_answers_list = test_data['Sentence'].tolist()

    # Evaluate and save/load predictions without attention
    if os.path.exists(predictions_path_no_attention):
        results_df_no_attention = pd.read_csv(predictions_path_no_attention)
        print("Loaded predictions without attention from file.")
    else:
        results_df_no_attention = evaluate_model(test_data, encoder_model_no_attention, decoder_model_no_attention, tokenizer, max_seq_length, attention=False)
        results_df_no_attention.to_csv(predictions_path_no_attention, index=False)
        print("Computed and saved predictions without attention.")

    # Evaluate and save/load predictions with attention
    if os.path.exists(predictions_path_attention):
        results_df_attention = pd.read_csv(predictions_path_attention)
        print("Loaded predictions with attention from file.")
    else:
        results_df_attention = evaluate_model(test_data, encoder_model_attention, decoder_model_attention, tokenizer, max_seq_length, attention=True)
        results_df_attention.to_csv(predictions_path_attention, index=False)
        print("Computed and saved predictions with attention.")

    # Compare and visualize results
    results_df_no_attention['Model'] = 'No Attention'
    results_df_attention['Model'] = 'Attention'

    combined_results = pd.concat([results_df_no_attention, results_df_attention], ignore_index=True)

    print("Combined Results:")
    print(combined_results.head())

    # Plot the comparison graphs
    plot_comparison_graphs(combined_results)

if __name__ == '__main__':
    main()
