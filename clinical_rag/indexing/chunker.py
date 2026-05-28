"""
Section-aware parent-child chunking strategy.
"""
import re
from typing import List, Dict, Any, Optional, Tuple
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import spacy
from ..config import settings


def download_nltk_data():
    """Download required NLTK data."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')


def load_spacy_model():
    """Load spaCy model for sentence segmentation."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        print("spaCy model en_core_web_sm not found. Please install it:")
        print("python -m spacy download en_core_web_sm")
        return None


class SectionAwareChunker:
    def __init__(self,
                 chunk_size: int = None,
                 chunk_overlap: int = None,
                 parent_chunk_size: int = None):
        """
        Initialize section-aware chunker.

        Args:
            chunk_size: Size of child chunks in tokens
            chunk_overlap: Overlap between child chunks in tokens
            parent_chunk_size: Size of parent chunks in tokens
        """
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.parent_chunk_size = parent_chunk_size or settings.PARENT_CHUNK_SIZE

        # Initialize NLP tools
        download_nltk_data()
        self.nlp = load_spacy_model()

        # Common section headers in biomedical papers
        self.section_patterns = [
            r'^(abstract|background|introduction|methods?|methodology|experimental|results?|discussion|conclusion|references?|acknowledgments?)[\s:-]*$',
            r'^\d+\.\s*(abstract|background|introduction|methods?|methodology|experimental|results?|discussion|conclusion|references?|acknowledgments?)[\s:-]*$',
            r'^[A-Z][A-Z\s\-]+[A-Z]$',  # ALL CAPS sections
        ]

    def detect_sections(self, text: str) -> List[Tuple[str, str, int, int]]:
        """
        Detect sections in text and return section boundaries.

        Args:
            text: Input text to analyze

        Returns:
            List of tuples (section_title, section_content, start_char, end_char)
        """
        lines = text.split('\n')
        sections = []
        current_section = "unknown"
        current_content = []
        start_char = 0

        for i, line in enumerate(lines):
            line_stripped = line.strip()

            # Check if line is a section header
            is_header = False
            for pattern in self.section_patterns:
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    is_header = True
                    break

            # Also check for common section patterns
            if not is_header and line_stripped.isupper() and len(line_stripped) > 3:
                # Likely a section header if it's short and all caps
                if len(line_stripped.split()) <= 5:  # Reasonable section name length
                    is_header = True

            if is_header and current_content:
                # Save previous section
                section_text = '\n'.join(current_content)
                sections.append((current_section, section_text, start_char, start_char + len(section_text)))
                start_char += len(section_text) + 1  # +1 for newline
                current_content = []
                current_section = line_stripped
            elif is_header and not current_content:
                # First section header
                current_section = line_stripped
                start_char = sum(len(l) + 1 for l in lines[:i])  # Calculate start position
            else:
                current_content.append(line)

        # Add final section
        if current_content:
            section_text = '\n'.join(current_content)
            sections.append((current_section, section_text, start_char, start_char + len(section_text)))

        # If no sections detected, treat whole text as one section
        if not sections:
            sections = [("full_text", text, 0, len(text))]

        return sections

    def chunk_text_by_tokens(self, text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
        """
        Chunk text into pieces based on approximate token count.

        Args:
            text: Text to chunk
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Overlap between chunks in tokens

        Returns:
            List of text chunks
        """
        # Simple approximation: 1 token ≈ 0.75 words for English
        words_per_token = 0.75
        max_words = int(max_tokens * words_per_token)
        overlap_words = int(overlap_tokens * words_per_token)

        # Tokenize into sentences
        sentences = sent_tokenize(text)

        chunks = []
        current_chunk = []
        current_word_count = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())

            # If adding this sentence would exceed max_words, save current chunk
            if current_word_count + sentence_words > max_words and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)

                # Start new chunk with overlap
                overlap_sentences = []
                overlap_word_count = 0
                # Go backwards through current chunk to get overlap
                for sent in reversed(current_chunk):
                    sent_words = len(sent.split())
                    if overlap_word_count + sent_words <= overlap_words:
                        overlap_sentences.insert(0, sent)
                        overlap_word_count += sent_words
                    else:
                        break

                current_chunk = overlap_sentences + [sentence]
                current_word_count = overlap_word_count + sentence_words
            else:
                current_chunk.append(sentence)
                current_word_count += sentence_words

        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)

        return chunks

    def create_parent_child_chunks(self, text: str, metadata: Dict[str, Any] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Create parent-child chunks from text.

        Args:
            text: Input text to chunk
            metadata: Base metadata to include with each chunk

        Returns:
            Tuple of (parent_chunks, child_chunks) where each is a list of chunk dictionaries
        """
        if metadata is None:
            metadata = {}

        # Detect sections
        sections = self.detect_sections(text)

        parent_chunks = []
        child_chunks = []

        for section_title, section_text, start_char, end_char in sections:
            # Create parent chunk for this section
            parent_metadata = metadata.copy()
            parent_metadata.update({
                "section": section_title,
                "section_start_char": start_char,
                "section_end_char": end_char,
                "chunk_type": "parent",
                "word_count": len(section_text.split()),
                "char_count": len(section_text)
            })

            parent_chunk = {
                "text": section_text,
                "metadata": parent_metadata
            }
            parent_chunks.append(parent_chunk)

            # Create child chunks from this section
            child_texts = self.chunk_text_by_tokens(
                section_text,
                self.chunk_size,
                self.chunk_overlap
            )

            for i, child_text in enumerate(child_texts):
                child_metadata = metadata.copy()
                child_metadata.update({
                    "section": section_title,
                    "parent_section_start_char": start_char,
                    "parent_section_end_char": end_char,
                    "chunk_index": i,
                    "chunk_type": "child",
                    "word_count": len(child_text.split()),
                    "char_count": len(child_text)
                })

                child_chunk = {
                    "text": child_text,
                    "metadata": child_metadata
                }
                child_chunks.append(child_chunk)

        return parent_chunks, child_chunks

    def process_document(self, document: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Process a document and create parent-child chunks.

        Args:
            document: Dictionary containing document text and metadata
                   Expected keys: 'text', plus any metadata fields

        Returns:
            Tuple of (parent_chunks, child_chunks)
        """
        text = document.get("text", "")
        # Extract metadata (everything except 'text')
        metadata = {k: v for k, v in document.items() if k != "text"}

        return self.create_parent_child_chunks(text, metadata)


# Convenience functions
def chunk_document(document: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Chunk a document using section-aware parent-child strategy.

    Args:
        document: Dictionary containing document text and metadata

    Returns:
        Tuple of (parent_chunks, child_chunks)
    """
    chunker = SectionAwareChunker()
    return chunker.process_document(document)


def chunk_text(text: str, metadata: Dict[str, Any] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Chunk raw text using section-aware parent-child strategy.

    Args:
        text: Raw text to chunk
        metadata: Metadata to attach to chunks

    Returns:
        Tuple of (parent_chunks, child_chunks)
    """
    if metadata is None:
        metadata = {}
    document = {"text": text, **metadata}
    chunker = SectionAwareChunker()
    return chunker.process_document(document)


if __name__ == "__main__":
    # Test the chunker
    sample_text = """
    ABSTRACT
    This study investigates the effects of metformin on liver function in elderly patients with type 2 diabetes.
    We conducted a randomized controlled trial involving 200 participants aged 65-80 years.

    METHODS
    Participants were randomly assigned to receive either metformin 500mg twice daily or placebo for 24 weeks.
    Liver function tests including ALT, AST, and bilirubin were measured at baseline, 12 weeks, and 24 weeks.

    RESULTS
    Metformin treatment was associated with a significant reduction in ALT levels compared to placebo (p<0.05).
    No serious adverse events related to liver toxicity were observed.

    CONCLUSION
    Metformin appears to be safe for liver function in elderly patients with type 2 diabetes.
    """

    parent_chunks, child_chunks = chunk_text(sample_text, {"source": "test", "year": 2023})

    print(f"Created {len(parent_chunks)} parent chunks and {len(child_chunks)} child chunks")
    print("\nParent chunks:")
    for i, chunk in enumerate(parent_chunks):
        print(f"  Parent {i}: {chunk['metadata']['section']} ({chunk['metadata']['word_count']} words)")

    print("\nChild chunks:")
    for i, chunk in enumerate(child_chunks[:3]):  # Show first 3
        print(f"  Child {i}: Section '{chunk['metadata']['section']}' ({chunk['metadata']['word_count']} words)")
        print(f"    Text preview: {chunk['text'][:100]}...")