from nemoguardrails import LLMRails

from llama_index.embeddings.openai import OpenAIEmbedding
import numpy as np
import nltk
import itertools
from dataset import DEFAULT_FEW_SHOT_TRAIN_PROMPTS


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
        # FIXME is this the correct way to use tiktoken?
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


def init(app: LLMRails):
    # pass in dynamically later
    sources = DEFAULT_FEW_SHOT_TRAIN_PROMPTS

    # Validate we have a non-empty dataset containing string messages
    for prompt in sources:
        if not prompt or not isinstance(prompt, str):
            raise ValueError(f"Prompt example: {prompt} is invalid. Must contain valid string data.")

    chunks = [
        get_chunks_from_text(source, "sentence", 100, 20)
        for source in sources
    ]
    _chunks = list(itertools.chain.from_iterable(chunks))

    # Create embeddings
    source_embeddings = np.array(_embed_function(_chunks)).squeeze()

    # Register the action parameter
    app.register_action_param("source_embeddings", source_embeddings)
