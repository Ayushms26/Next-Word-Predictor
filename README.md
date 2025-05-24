## Next Word Predictor

A deep learning-based Next Word Predictor that uses an LSTM neural network to predict the next word in a given sentence fragment. This project demonstrates the end-to-end process of building a text prediction model using Keras and TensorFlow, from data preparation to model training and inference.

**Key Features**
- Predicts the next word in a sentence based on previous context
- Built using Python, TensorFlow, and Keras
- Utilizes an Embedding layer and LSTM for sequence modeling
- Trained on a conversational English dataset
- Demonstrates text preprocessing, tokenization, padding, and one-hot encoding

## How It Works

- **Dataset Preparation:**  
  The model is trained on a dataset of conversational English sentences. Each sentence is tokenized, and input sequences are generated such that for a sequence of words, the model learns to predict the next word in the sequence.

- **Text Processing:**  
  - Tokenization using Keras' `Tokenizer`
  - Creation of input-output pairs for supervised learning
  - Padding of sequences to ensure uniform input length
  - Output labels are one-hot encoded for categorical prediction

- **Model Architecture:**  
  - **Embedding Layer:** Converts word indices into dense vectors of fixed size
  - **LSTM Layer:** Captures sequential dependencies and context
  - **Dense Output Layer:** Predicts the next word from the vocabulary using softmax activation

- **Training:**  
  The model is trained using categorical cross-entropy loss and the Adam optimizer for 100 epochs, achieving high accuracy on the training data.

## Example Workflow

1. **Input:**  
   User provides a fragment of a sentence (e.g., "I am going to").
2. **Prediction:**  
   The model outputs the most probable next word (e.g., "study").

## Usage

- Clone the repository and install the required dependencies (`tensorflow`, `keras`, `numpy`, `pandas`).
- Prepare your own dataset or use the provided sample.
- Run the notebook or script to train the model.
- Use the prediction function to generate next-word suggestions for any input phrase.

## Model Details

| Layer        | Output Shape      | Parameters |
|--------------|------------------|------------|
| Embedding    | (None, 10, 100)  | 58,100     |
| LSTM         | (None, 200)      | 240,800    |
| Dense        | (None, 581)      | 116,781    |
| **Total**    |                  | 415,681    |

- **Vocabulary Size:** 581 words
- **Sequence Length:** 10 tokens (padded)
- **Embedding Dimension:** 100

## Applications

- Smart keyboard suggestions
- Chatbots and conversational agents
- Text autocompletion tools
- Language modeling research

## Limitations

- Trained on a small conversational dataset; accuracy may vary on broader or more formal corpora.
- Model performance depends on the quality and diversity of the training data.

## Getting Started

1. Install dependencies:
   ```bash
   pip install tensorflow keras numpy pandas
   ```
2. Train the model using the provided notebook or script.
3. Use the trained model to predict the next word for any input sequence.

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements, bug fixes, or new features.

---
