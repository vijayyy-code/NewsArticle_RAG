# embedding_store.py
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import numpy as np
import json
import os
from datetime import datetime
from chunking import TextChunk  # Import from our previous module

class VectorStore:
    """Handle embedding generation and vector storage"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", persist_dir: str = "vector_db"):
        """
        Initialize vector store
        
        Args:
            model_name: Sentence transformer model name
            persist_dir: Directory to store ChromaDB data
        """
        self.model_name = model_name
        self.persist_dir = persist_dir
        
        # Initialize embedding model
        print(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        print(f" Model loaded. Vector dimension: {self.embedding_model.get_sentence_embedding_dimension()}")
        
        # Initialize ChromaDB client
        self._init_chroma_client()
        
    def _init_chroma_client(self):
        """Initialize or connect to ChromaDB"""
        os.makedirs(self.persist_dir, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="news_articles",
            metadata={"description": "News article chunks for RAG system"}
        )
        
        print(f" Connected to ChromaDB collection: {self.collection.name}")
        print(f"  Collection count: {self.collection.count()} documents")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        print(f"Generating embeddings for {len(texts)} texts...")
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Convert to list of lists for ChromaDB
        embeddings_list = embeddings.tolist()
        print(f" Generated {len(embeddings_list)} embeddings")
        
        return embeddings_list
    
    def add_chunks_to_store(self, chunks: List[TextChunk], article_id: str = None):
        """
        Add text chunks to vector store
        
        Args:
            chunks: List of TextChunk objects
            article_id: Optional article identifier
        """
        if not chunks:
            print("No chunks to add")
            return
        
        # Prepare data
        texts = [chunk.text for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]
        
        # Prepare metadata for each chunk
        metadatas = []
        for chunk in chunks:
            metadata = {
                **chunk.metadata,
                'chunk_number': chunk.chunk_number,
                'total_chunks': chunk.total_chunks,
                'article_id': article_id or chunk.metadata.get('url', 'unknown'),
                'added_date': datetime.now().isoformat()
            }
            metadatas.append(metadata)
        
        print(f"Adding {len(chunks)} chunks to vector store...")
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f" Successfully added {len(chunks)} chunks to vector store")
        print(f"  Total documents in collection: {self.collection.count()}")
    
    def search_similar(self, query: str, n_results: int = 3, filter_metadata: Dict = None) -> List[Dict]:
        """
        Search for similar chunks
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of search results with metadata
        """
        print(f"Searching for: '{query}'")
        
        # Generate embedding for query
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_metadata,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        search_results = []
        if results['documents']:
            for i in range(len(results['documents'][0])):
                result = {
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'similarity_score': 1 - results['distances'][0][i]  # Convert distance to similarity
                }
                search_results.append(result)
        
        return search_results
    
    def get_collection_info(self) -> Dict:
        """Get information about the collection"""
        return {
            'name': self.collection.name,
            'count': self.collection.count(),
            'metadata': self.collection.metadata
        }
    
    def reset_collection(self):
        """Reset the collection (delete all documents)"""
        print("Resetting collection...")
        self.client.delete_collection("news_articles")
        self._init_chroma_client()
        print(" Collection reset")

def test_embedding_store():
    """Test the embedding and vector store functionality"""
    print("=" * 60)
    print("TESTING EMBEDDING & VECTOR STORE")
    print("=" * 60)
    
    # Initialize vector store
    print("\n1. Initializing Vector Store...")
    vector_store = VectorStore()
    
    # Test with sample chunks from our chunking test
    print("\n2. Creating test chunks...")
    from chunking import ArticleChunker
    
    # Sample text
    sample_text = """Artificial intelligence is transforming many industries. 
    Machine learning algorithms can analyze large datasets. 
    Natural language processing enables computers to understand human language.
    Deep learning uses neural networks with multiple layers.
    AI ethics is an important consideration for responsible development."""
    
    chunker = ArticleChunker(chunk_size=200, chunk_overlap=50)
    metadata = {'title': 'AI Overview', 'url': 'https://example.com/ai', 'source': 'test'}
    chunks = chunker.chunk_article(sample_text, metadata)
    
    print(f"Created {len(chunks)} test chunks")
    
    # Add chunks to vector store
    print("\n3. Adding chunks to vector store...")
    vector_store.add_chunks_to_store(chunks, article_id="test_ai_article")
    
    # Test search functionality
    print("\n4. Testing search functionality...")
    
    test_queries = [
        "machine learning algorithms",
        "natural language processing",
        "AI ethics considerations"
    ]
    
    for query in test_queries:
        print(f"\nSearch query: '{query}'")
        results = vector_store.search_similar(query, n_results=2)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"  Result {i}:")
                print(f"    Similarity: {result['similarity_score']:.3f}")
                print(f"    Text: {result['document'][:80]}...")
                print(f"    Source: {result['metadata'].get('source', 'unknown')}")
        else:
            print("  No results found")
    
    # Get collection info
    print("\n5. Collection information:")
    info = vector_store.get_collection_info()
    print(f"   Collection name: {info['name']}")
    print(f"   Document count: {info['count']}")
    
    print("\n" + "=" * 60)
    print("EMBEDDING STORE TEST COMPLETE")
    print("=" * 60)
    
    return vector_store

def test_with_actual_article():
    """Test with actual article from our previous fetch"""
    print("\n" + "=" * 60)
    print("TESTING WITH ACTUAL ARTICLE")
    print("=" * 60)
    
    # Try to load actual article from cache
    import glob
    cache_files = glob.glob("cache/articles/*.json")
    
    if not cache_files:
        print("No cached articles found. Run article fetcher first.")
        return
    
    # Load first cached article
    cache_file = cache_files[0]
    print(f"Loading article from: {cache_file}")
    
    with open(cache_file, 'r', encoding='utf-8') as f:
        article_data = json.load(f)
    
    print(f"Article: {article_data.get('title', 'No title')}")
    print(f"Text length: {len(article_data.get('text', ''))} chars")
    
    # Chunk the article
    from chunking import ArticleChunker
    chunker = ArticleChunker(chunk_size=500, chunk_overlap=100)
    
    chunks = chunker.chunk_article(
        article_data['text'],
        {
            'title': article_data.get('title', ''),
            'url': article_data.get('url', ''),
            'source': article_data.get('source', ''),
            'date': article_data.get('date', ''),
            'fetch_method': article_data.get('fetch_method', '')
        }
    )
    
    print(f"Created {len(chunks)} chunks")
    
    # Initialize vector store
    vector_store = VectorStore()
    
    # Add to vector store
    article_id = article_data.get('url', 'unknown')
    vector_store.add_chunks_to_store(chunks, article_id=article_id)
    
    # Test search
    print("\nTesting search with actual article content...")
    
    # Try to extract some keywords from the article for testing
    if "web scraping" in article_data.get('text', '').lower():
        test_query = "web scraping techniques"
    elif "artificial intelligence" in article_data.get('text', '').lower():
        test_query = "artificial intelligence applications"
    else:
        test_query = "technology development"
    
    results = vector_store.search_similar(test_query, n_results=3)
    
    print(f"\nSearch results for '{test_query}':")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Similarity: {result['similarity_score']:.3f}")
        print(f"   Chunk: {result['document'][:100]}...")
        print(f"   From: {result['metadata'].get('title', 'Unknown')}")
    
    return vector_store, chunks

if __name__ == "__main__":
    # Run basic test
    store = test_embedding_store()
    
    # Ask if user wants to test with actual article
    print("\n" + "=" * 60)
    choice = input("Test with actual article from cache? (y/n): ")
    
    if choice.lower() == 'y':
        test_with_actual_article()
    
    print("\n Embedding store ready for RAG pipeline!")