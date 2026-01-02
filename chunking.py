# chunking.py
import re
import hashlib
from typing import List, Dict, Any, Tuple
import json
from dataclasses import dataclass, asdict

@dataclass
class TextChunk:
    """Represents a chunk of text with metadata"""
    text: str
    chunk_id: str
    chunk_number: int
    total_chunks: int
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    def __repr__(self) -> str:
        return f"Chunk {self.chunk_number}/{self.total_chunks} ({len(self.text)} chars)"

class ArticleChunker:
    """Handle text chunking with semantic boundaries"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize chunker
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def _generate_chunk_id(self, text: str, chunk_num: int) -> str:
        """Generate unique ID for chunk"""
        content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        return f"chunk_{chunk_num}_{content_hash}"
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """Split text by paragraphs (preserves semantic boundaries)"""
        # Split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # If no paragraphs found, split by sentences
        if not paragraphs or len(paragraphs) == 1:
            # Simple sentence splitting (can be improved with NLTK)
            sentences = re.split(r'(?<=[.!?])\s+', text)
            paragraphs = [s.strip() for s in sentences if s.strip()]
        
        return paragraphs
    
    def _merge_paragraphs_to_chunks(self, paragraphs: List[str]) -> List[str]:
        """Merge paragraphs into chunks of appropriate size"""
        chunks = []
        current_chunk = []
        current_size = 0
        
        for paragraph in paragraphs:
            para_size = len(paragraph)
            
            # If adding this paragraph would exceed chunk size, finalize current chunk
            if current_size + para_size > self.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0 and current_chunk:
                    # Keep last few paragraphs for overlap
                    overlap_text = ' '.join(current_chunk[-2:]) if len(current_chunk) >= 2 else current_chunk[-1]
                    if len(overlap_text) <= self.chunk_overlap:
                        current_chunk = [overlap_text]
                        current_size = len(overlap_text)
                    else:
                        current_chunk = []
                        current_size = 0
            
            # Add paragraph to current chunk
            current_chunk.append(paragraph)
            current_size += para_size
            
            # If current chunk is large enough, finalize it
            if current_size >= self.chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
        
        # Add remaining paragraphs as final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def chunk_article(self, article_text: str, metadata: Dict[str, Any] = None) -> List[TextChunk]:
        """
        Chunk article text into manageable pieces
        
        Args:
            article_text: Full article text
            metadata: Article metadata (title, url, date, etc.)
            
        Returns:
            List of TextChunk objects
        """
        if metadata is None:
            metadata = {}
        
        # Clean text
        text = re.sub(r'\s+', ' ', article_text.strip())
        
        # Split by paragraphs
        paragraphs = self._split_by_paragraphs(text)
        
        # Merge into chunks
        chunks = self._merge_paragraphs_to_chunks(paragraphs)
        
        # Create TextChunk objects
        text_chunks = []
        total_chunks = len(chunks)
        
        for i, chunk_text in enumerate(chunks, 1):
            chunk_id = self._generate_chunk_id(chunk_text, i)
            
            text_chunk = TextChunk(
                text=chunk_text,
                chunk_id=chunk_id,
                chunk_number=i,
                total_chunks=total_chunks,
                metadata={
                    **metadata,
                    'chunk_size_chars': len(chunk_text),
                    'chunk_size_tokens': len(chunk_text) // 4  # Rough estimate
                }
            )
            
            text_chunks.append(text_chunk)
        
        return text_chunks
    
    def save_chunks(self, chunks: List[TextChunk], filename: str):
        """Save chunks to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json_data = [chunk.to_dict() for chunk in chunks]
            json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    def load_chunks(self, filename: str) -> List[TextChunk]:
        """Load chunks from JSON file"""
        with open(filename, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        return [TextChunk(**data) for data in json_data]

def test_chunking():
    """Test the chunking functionality in terminal"""
    print("=" * 60)
    print("TESTING CHUNKING FUNCTIONALITY")
    print("=" * 60)
    
    # Sample article text (from Wikipedia scraping)
    sample_text = """Web scraping, web harvesting, or web data extraction is data scraping used for extracting data from websites. Web scraping software may access the World Wide Web directly using the Hypertext Transfer Protocol, or through a web browser. While web scraping can be done manually by a software user, the term typically refers to automated processes implemented using a bot or web crawler. It is a form of copying, in which specific data is gathered and copied from the web, typically into a central local database or spreadsheet, for later retrieval or analysis.

    Web scraping is a technique to automatically access and extract large amounts of information from a website, which can save a huge amount of time and effort. The term "scraping" refers to obtaining the information from a source that is not intended for sharing. Newer forms of web scraping involve listening to data feeds from web servers. For example, JSON is commonly used as a transport mechanism between the client and the web server.

    There are methods that some websites use to prevent web scraping, such as detecting and disallowing bots from crawling (viewing) their pages. In response, there are web scraping systems that rely on using techniques in DOM parsing, computer vision and natural language processing to simulate human browsing to enable gathering web page content for offline parsing.

    Web scraping is used for a wide range of applications including price monitoring, news aggregation, market research, and data journalism. However, web scraping can also be controversial due to its potential to violate terms of service, copyright laws, and privacy concerns."""
    
    print(f"\nOriginal text length: {len(sample_text)} characters")
    
    # Create chunker instance
    chunker = ArticleChunker(chunk_size=300, chunk_overlap=50)
    
    # Create metadata
    metadata = {
        'title': 'Web Scraping Article',
        'url': 'https://en.wikipedia.org/wiki/Web_scraping',
        'source': 'Wikipedia',
        'date': '2024-01-15'
    }
    
    # Chunk the article
    chunks = chunker.chunk_article(sample_text, metadata)
    
    print(f"\nCreated {len(chunks)} chunks:")
    print("-" * 40)
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\n{i}. {chunk}")
        print(f"   Text preview: {chunk.text[:100]}...")
        print(f"   Metadata: {chunk.metadata.get('chunk_size_chars')} chars, ~{chunk.metadata.get('chunk_size_tokens')} tokens")
    
    # Test saving and loading
    print("\n" + "=" * 60)
    print("TESTING SAVE/LOAD FUNCTIONALITY")
    print("=" * 60)
    
    # Save chunks
    save_filename = "test_chunks.json"
    chunker.save_chunks(chunks, save_filename)
    print(f"\n Saved {len(chunks)} chunks to '{save_filename}'")
    
    # Load chunks back
    loaded_chunks = chunker.load_chunks(save_filename)
    print(f" Loaded {len(loaded_chunks)} chunks from '{save_filename}'")
    
    # Verify
    if len(chunks) == len(loaded_chunks):
        print(" Chunk count matches!")
    else:
        print(" Chunk count mismatch!")
    
    print("\n" + "=" * 60)
    print("CHUNKING TEST COMPLETE")
    print("=" * 60)
    
    return chunks

if __name__ == "__main__":
    test_chunking()