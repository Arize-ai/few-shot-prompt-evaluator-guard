from nemoguardrails import LLMRails

from llama_index.embeddings.openai import OpenAIEmbedding
import numpy as np
import nltk
import tiktoken
import itertools
import json
from typing import List
from getpass import getpass
from dataset_guard_config.dataset import DEFAULT_FEW_SHOT_TRAIN_PROMPTS
from arize.experimental.datasets import ArizeDatasetsClient


CHUNK_STRATEGY = "sentence"
CHUNK_SIZE = 100
CHUNK_OVERLAP = 20


def get_arize_jailbreak_dataset():
    developer_key = getpass(f"Enter your Arize developer_key to get your dataset: ")
    space_id = getpass(f"Enter your Arize space_id to get your dataset: ")
    dataset_id = getpass(f"Enter your Arize dataset_id to get your dataset: ")
    client = ArizeDatasetsClient(developer_key=developer_key)
    # Get the current dataset version
    dataset = client.get_dataset(space_id=space_id, dataset_id=dataset_id)
    return [json.loads(prompt)['messages'][0]["content"] for prompt in dataset['attributes.input.value'].tolist()]


def get_chunks_from_text(
    text: str, chunk_strategy: str, chunk_size: int, chunk_overlap: int
):
    """Get chunks of text from a string.

    Args:
        text: The text to chunk.
        chunk_strategy: The strategy to use for chunking.
        chunk_size: The size of each chunk. If the chunk_strategy is "sentences",
            this is the number of sentences per chunk. If the chunk_strategy is
            "characters", this is the number of characters per chunk, and so on.
        chunk_overlap: The number of characters to overlap between chunks. If the
            chunk_strategy is "sentences", this is the number of sentences to overlap
            between chunks.
    """

    nltk_error = (
        "nltk is required for sentence splitting. Please install it using "
        "`poetry add nltk`"
    )
    tiktoken_error = (
        "tiktoken is required for token splitting. Please install it using "
        "`poetry add tiktoken`"
    )

    if chunk_strategy == "sentence":
        if nltk is None:
            raise ImportError(nltk_error)
        atomic_chunks = nltk.sent_tokenize(text)
    elif chunk_strategy == "word":
        if nltk is None:
            raise ImportError(nltk_error)
        atomic_chunks = nltk.word_tokenize(text)
    elif chunk_strategy == "char":
        atomic_chunks = list(text)
    elif chunk_strategy == "token":
        if tiktoken is None:
            raise ImportError(tiktoken_error)
        atomic_chunks = tiktoken(text)  # type: ignore
    else:
        raise ValueError(
            "chunk_strategy must be 'sentence', 'word', 'char', or 'token'."
        )

    chunks = []
    for i in range(0, len(atomic_chunks), chunk_size - chunk_overlap):
        chunk = " ".join(atomic_chunks[i : i + chunk_size])
        chunks.append(chunk)
    return chunks


def _embed_function(text) -> np.ndarray:
    """Function used to embed text with OpenAIEmbedding(model="text-embedding-ada-002").

    :param text: Either a string or list of strings that will be embedded.

    :return: Array of embedded input string(s).
    """
    if isinstance(text, str):
        text = [text]

    embeddings_out = []
    for current_example in text:
        embedding = OpenAIEmbedding(model="text-embedding-ada-002").get_text_embedding(current_example)
        embeddings_out.append(embedding)
    return np.array(embeddings_out)


def init(app: LLMRails, sources: List[str] = get_arize_jailbreak_dataset(), chunk_strategy: str = CHUNK_STRATEGY,
         chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    # Validate we have a non-empty dataset containing string messages
    for prompt in sources:
        if not prompt or not isinstance(prompt, str):
            raise ValueError(f"Prompt example: {prompt} is invalid. Must contain valid string data.")

    chunks = [
        get_chunks_from_text(source, chunk_strategy, chunk_size, chunk_overlap)
        for source in sources
    ]
    chunks = list(itertools.chain.from_iterable(chunks))

    # Create embeddings
    source_embeddings = np.array(_embed_function(chunks)).squeeze()

    # Register the action parameter
    app.register_action_param("source_embeddings", source_embeddings)
    app.register_action_param("chunks", chunks)
