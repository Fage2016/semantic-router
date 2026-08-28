import os

import numpy as np
import pytest

from semantic_router.encoders import BM25Encoder
from semantic_router.route import Route
from semantic_router.tokenizers import BaseTokenizer

UTTERANCES = [
    "Hello we need this text to be a little longer for our sparse encoders",
    "In this case they need to learn from recurring tokens, ie words.",
    "We give ourselves several examples from our encoders to learn from.",
    "But given this is only an example we don't need too many",
    "Just enough to test that our sparse encoders work as expected",
]


@pytest.fixture
def bm25_encoder():
    sparse_encoder = BM25Encoder(use_default_params=True)
    sparse_encoder.fit(
        [
            Route(
                name="test_route",
                utterances=[
                    "The quick brown fox",
                    "jumps over the lazy dog",
                    "Hello, world!",
                ],
            )
        ]
    )
    return sparse_encoder


@pytest.fixture
def routes():
    return [
        Route(name="Route 1", utterances=[UTTERANCES[0], UTTERANCES[1]]),
        Route(name="Route 2", utterances=[UTTERANCES[2], UTTERANCES[3], UTTERANCES[4]]),
    ]


@pytest.mark.skipif(
    os.environ.get("RUN_HF_TESTS") is None,
    reason="Set RUN_HF_TESTS=1 to run. This test downloads models from Hugging Face which can time out in CI.",
)
class TestBM25Encoder:
    def test_initialization(self, bm25_encoder):
        assert bm25_encoder._tokenizer is not None

    def test_fit(self, bm25_encoder, routes):
        bm25_encoder.fit(routes)
        assert bm25_encoder._tokenizer is not None

    def test_fit_with_strings(self, bm25_encoder):
        route_strings = ["test a", "test b", "test c"]
        with pytest.raises(TypeError):
            bm25_encoder.fit(route_strings)

    def test_call_method(self, bm25_encoder):
        result = bm25_encoder(["test"])
        assert isinstance(result, list), "Result should be a list"
        assert all(
            isinstance(sparse_emb.embedding, np.ndarray) for sparse_emb in result
        ), "Each item in result should be an array"

    def test_call_method_no_docs_bm25_encoder(self, bm25_encoder):
        with pytest.raises(ValueError):
            bm25_encoder([])

    def test_call_method_no_word(self, bm25_encoder):
        result = bm25_encoder(["doc with fake word gta5jabcxyz"])
        assert isinstance(result, list), "Result should be a list"
        assert all(
            isinstance(sparse_emb.embedding, np.ndarray) for sparse_emb in result
        ), "Each item in result should be an array"

    def test_call_method_with_uninitialized_model_or_mapping(self, bm25_encoder):
        bm25_encoder._tokenizer = None
        with pytest.raises(ValueError):
            bm25_encoder(["test"])

    def test_fit_with_uninitialized_model(self, bm25_encoder, routes):
        bm25_encoder._tokenizer = None
        with pytest.raises(ValueError):
            bm25_encoder.fit(routes)

    def test_encode_queries(self, bm25_encoder):
        queries = ["quick brown", "lazy dog", "hello world"]
        results = bm25_encoder.encode_queries(queries)

        assert len(results) == len(queries)
        assert all([isinstance(result.embedding, np.ndarray) for result in results])

    def test_encode_queries_empty_list(self, bm25_encoder):
        with pytest.raises(ValueError, match="No documents provided for encoding"):
            bm25_encoder.encode_queries([])

    def test_encode_queries_unfitted(self):
        encoder = BM25Encoder(use_default_params=True)
        with pytest.raises(ValueError, match="Encoder not fitted"):
            encoder.encode_queries(["test query"])

    def test_encode_documents(self, bm25_encoder):
        documents = ["quick brown", "lazy dog", "hello world"]
        results = bm25_encoder.encode_documents(documents)

        assert len(results) == len(documents)
        assert all([isinstance(result.embedding, np.ndarray) for result in results])

    def test_encode_documents_empty_list(self, bm25_encoder):
        with pytest.raises(ValueError, match="No documents provided for encoding"):
            bm25_encoder.encode_documents([])

    def test_encode_documents_unfitted(self):
        encoder = BM25Encoder(use_default_params=True)
        with pytest.raises(ValueError, match="Encoder not fitted"):
            encoder.encode_documents(["test document"])

    def test_encode_documents_batch_size(self, bm25_encoder):
        documents = ["quick brown", "lazy dog", "hello world", "test document"]
        batch_size = 2
        results = bm25_encoder.encode_documents(documents, batch_size=batch_size)

        assert len(results) == len(documents)
        assert all(isinstance(result.embedding, np.ndarray) for result in results)


class WordTokenizer(BaseTokenizer):
    """Deterministic word-level tokenizer, so the formula tests below need no
    model download. Token id 0 is reserved for padding, matching the convention
    :class:`BM25Encoder` relies on.
    """

    def __init__(self, vocab: list[str]) -> None:
        super().__init__()
        self._vocab = {word: idx + 1 for idx, word in enumerate(vocab)}

    @property
    def vocab_size(self) -> int:
        return len(self._vocab) + 1

    def tokenize(self, texts, pad: bool = True) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        batch = [
            [self._vocab[word] for word in text.split() if word in self._vocab]
            for text in texts
        ]
        width = max(len(ids) for ids in batch)
        return np.array([ids + [0] * (width - len(ids)) for ids in batch])


VOCAB = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]

# Document lengths 2, 3 and 4, so avgdl is exactly 3.0.
CORPUS = [
    "alpha beta",
    "beta gamma delta",
    "gamma delta epsilon zeta",
]
AVG_DOC_LEN = 3.0


def atire_tf_component(tf: float, doc_len: int, k1: float, b: float) -> float:
    """The document-side factor of the ATIRE BM25 formula (paper section 4.1).

    Written out longhand as a reference, independent of the vectorised
    implementation under test.
    """
    return ((k1 + 1.0) * tf) / (k1 * ((1.0 - b) + b * (doc_len / AVG_DOC_LEN)) + tf)


@pytest.fixture
def word_encoder():
    encoder = BM25Encoder(tokenizer=WordTokenizer(VOCAB), use_default_params=False)
    encoder.fit([Route(name="corpus", utterances=CORPUS)])
    return encoder


class TestBM25ATIREFormula:
    """Guards the ATIRE BM25 formula itself, rather than just output shapes."""

    def test_fit_computes_avg_doc_len(self, word_encoder):
        assert word_encoder.corpus_size == len(CORPUS)
        assert word_encoder._avg_doc_len == pytest.approx(AVG_DOC_LEN)

    @pytest.mark.parametrize(
        "document,doc_len,term,tf",
        [
            ("alpha", 1, "alpha", 1),
            ("alpha beta gamma", 3, "beta", 1),
            ("alpha alpha beta", 3, "alpha", 2),
            # ~3x the average length: the case where an inverted length
            # normalisation drives the denominator through zero.
            ("alpha beta gamma delta epsilon zeta alpha beta gamma", 9, "alpha", 2),
        ],
    )
    def test_encode_documents_matches_atire(
        self, word_encoder, document, doc_len, term, tf
    ):
        token_id = VOCAB.index(term) + 1
        encoded = word_encoder.encode_documents([document])[0].to_dict()

        expected = atire_tf_component(tf, doc_len, word_encoder.k1, word_encoder.b)
        assert encoded[token_id] == pytest.approx(expected)

    def test_encode_documents_always_positive(self, word_encoder):
        """The denominator must stay positive at any document length."""
        documents = [" ".join(["alpha"] + VOCAB * n) for n in range(1, 20)]
        for embedding in word_encoder.encode_documents(documents):
            values = embedding.embedding[:, 1]
            assert np.all(values > 0.0)

    def test_longer_documents_score_lower(self, word_encoder):
        """A single occurrence of a term is worth less in a longer document."""
        documents = [
            "alpha",
            "alpha beta",
            "alpha beta gamma",
            "alpha beta gamma delta",
            "alpha beta gamma delta epsilon",
            "alpha beta gamma delta epsilon zeta",
        ]
        alpha_id = VOCAB.index("alpha") + 1
        scores = [
            embedding.to_dict()[alpha_id]
            for embedding in word_encoder.encode_documents(documents)
        ]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] > scores[-1]
