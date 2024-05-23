"""
/* THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE
WRITTEN BY OTHER STUDENTS OR LARGE LANGUAGE MODELS SUCH AS CHATGPT.
Tommy Skodje */

I collaborated with the following classmates for this homework:
None
"""

import argparse
import numpy as np
import pandas as pd
from sklearn import preprocessing


def model_assessment(filename):
    """
    Given the entire data, split it into training and test set 
    so you can assess your different models 
    to compare perceptron, logistic regression,
    and naive bayes. 
    """

    # Read the file at filename
    with open(filename, "r") as input_file:
        read_input_file = input_file.read()
        input_file_lines = read_input_file.split("\n")

    input_file.close()

    # There is a blank line at the end of the file. Remove it from the blank value from the array.
    input_file_lines = input_file_lines[:-1]

    # Store the '0' or '1' at the start of each email as y, and store the rest of the email as text.
    y_values = []
    text_values = []
    for line in input_file_lines:
        words = line.split(" ")

        y_value = words[0]
        y_values.append(y_value)
        text_values.append(words[1:])

    # Create a pandas dataframe from y_values and text_values
    height = len(y_values)
    width = 2
    total_data = pd.DataFrame()
    total_data["y"] = y_values
    total_data["text"] = text_values

    # Randomly shuffle the dataset
    shuffled = total_data.sample(frac=1)

    # Split the data into train and test splits.
    # I chose to use 70-30 train-test split because we learned 70-30 and 80-20 as heuristics
    # for train-test splits in class, and I felt that having more test data was beneficial to
    # determine the effectiveness of the model.
    num_rows = shuffled.shape[0]
    seventy_percent = int(num_rows * 0.70)
    training = shuffled.iloc[:seventy_percent]
    test = shuffled.iloc[seventy_percent:]

    return training, test


def build_vocab_map(traindf):
    """
    Construct the vocabulary map such that it returns
    (1) the vocabulary dictionary contains words as keys and
    the number of emails the word appears in as values, and
    (2) a list of words that appear in at least 30 emails.

    ---input:
    dataset: pandas dataframe containing the 'text' column
             and 'y' label column

    ---output:
    dict: key-value is word-count pair
    list: list of words that appear in at least 30 emails
    """

    # The dictionary and list that is to be returned
    vocab_dict = {}
    common_words = []

    words = traindf["text"]
    words_numpy = words.to_numpy()

    for row in words_numpy:
        seen = False
        for word in row:
            # If the word has already been seen before in another email, increment its value by 1
            # and mark it as seen in the current email.
            if seen is False and word in vocab_dict:
                vocab_dict[word] += 1
                seen = True

            # If the word has not been seen before in any email, add it to the dictionary and mark
            # it as seen in the current email.
            elif seen is False and word not in vocab_dict:
                vocab_dict[word] = 1
                seen = True
            # Don't do anything if the word has been seen before in this email.

    # Fill in the common_words list
    for word, count in vocab_dict.items():
        if count >= 30:
            common_words.append(word)

    return vocab_dict, common_words


def construct_binary(dataset, freq_words):
    """
    Construct email datasets based on
    the binary representation of the email.
    For each e-mail, transform it into a
    feature vector where the ith entry,
    $x_i$, is 1 if the ith word in the 
    vocabulary occurs in the email,
    or 0 otherwise

    ---input:
    dataset: pandas dataframe containing the 'text' column

    freq_word: the vocabulary map built in build_vocab_map()

    ---output:
    numpy array
    """

    text = dataset["text"]

    text_numpy = text.to_numpy()

    # Create a 2d numpy array to store the feature vectors
    columns = len(freq_words)
    rows = np.shape(text_numpy)[0]
    output_array = np.zeros((rows, columns), dtype=int)

    # Loop through the emails, checking if each word in freq_words is contained within it.
    for email_index, email in enumerate(text_numpy):
        for word_index, word in enumerate(freq_words):
            if word in email:
                # If the word is in the email, set the appropriate index to 1.
                output_array[email_index][word_index] = 1

    return output_array


def construct_count(dataset, freq_words):
    """
    Construct email datasets based on
    the count representation of the email.
    For each e-mail, transform it into a
    feature vector where the ith entry,
    $x_i$, is the number of times the ith word in the 
    vocabulary occurs in the email,
    or 0 otherwise

    ---input:
    dataset: pandas dataframe containing the 'text' column

    freq_word: the vocabulary map built in build_vocab_map()

    ---output:
    numpy array
    """

    text = dataset["text"]

    text_numpy = text.to_numpy()

    # Create a 2d numpy array to store the feature vectors
    columns = len(freq_words)
    rows = np.shape(text_numpy)[0]
    output_array = np.zeros((rows, columns), dtype=int)

    # Loop through the emails, checking if each word in freq_words is contained within it.
    for email_index, email in enumerate(text_numpy):
        for word_index, word in enumerate(freq_words):
            if word in email:
                # If the word is in the email, increment the appropriate index.
                output_array[email_index][word_index] = email.count(word)

    return output_array


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",
                        default="spamAssassin.data",
                        help="filename of the input data")
    args = parser.parse_args()

    # 1a. Return the train-test split. This is our model assessment strategy.
    train_test_split = model_assessment(args.data)
    train = train_test_split[0]
    test = train_test_split[1]

    # 1b. Build the vocabulary map.
    frequently_occurring_words = build_vocab_map(train)

    # 1b, 1c. Build a vocabulary map of if the ith word appears in the dataset for
    # both train and test, and create binary datasets for them.
    freq_words_keys = frequently_occurring_words[1]
    print("words that appear in more than 30 emails: ", freq_words_keys)
    train_binary = construct_binary(train, freq_words_keys)
    test_binary = construct_binary(test, freq_words_keys)

    # 1d. Create a count dataset of how many times the ith word appears in the dataset for
    # both train and test.
    train_count = construct_count(train, freq_words_keys)
    test_count = construct_count(test, freq_words_keys)

    # Store the extracted datasets for use in questions 2 and 3.
    pandas_binary_xTrain = pd.DataFrame(train_binary)
    pandas_binary_xTest = pd.DataFrame(test_binary)
    pandas_count_xTrain = pd.DataFrame(train_count)
    pandas_count_xTest = pd.DataFrame(test_count)
    pandas_yTrain = train["y"]
    pandas_yTest = test["y"]

    pandas_binary_xTrain.to_csv("binary_xTrain.csv", index=False)
    pandas_binary_xTest.to_csv("binary_xTest.csv", index=False)
    pandas_count_xTrain.to_csv("count_xTrain.csv", index=False)
    pandas_count_xTest.to_csv("count_xTest.csv", index=False)
    pandas_yTrain.to_csv("yTrain.csv", index=False)
    pandas_yTest.to_csv("yTest.csv", index=False)


if __name__ == "__main__":
    main()
