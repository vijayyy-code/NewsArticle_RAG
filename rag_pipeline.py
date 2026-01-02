# rag_pipeline.py
"""
Complete RAG Pipeline with Groq LLM Integration
"""
import os
import sys
import json
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from embedding_store import VectorStore
from chunking import ArticleChunker
from llm_integration import GroqLLM

class NewsRAGPipeline:
    """Complete RAG pipeline for news articles"""
    
    def __init__(self):
        """Initialize all components"""
        print(" Initializing News RAG Pipeline...")
        
        # Initialize components
        self.vector_store = VectorStore()
        self.chunker = ArticleChunker(chunk_size=500, chunk_overlap=100)
        self.llm = GroqLLM()
        
        print(" All components initialized")
    
    def process_news_url(self, url: str, use_playwright: bool = False) -> Dict:
        """
        Process a news URL through the entire pipeline
        
        Args:
            url: News article URL
            use_playwright: Whether to use headless browser for JS-heavy sites
            
        Returns:
            Dictionary with processing results
        """
        print(f"\n Processing: {url}")
        
        # Step 1: Fetch article
        article_data = self._fetch_article(url, use_playwright)
        if not article_data:
            return {'error': 'Failed to fetch article'}
        
        # Step 2: Chunk article
        chunks = self.chunker.chunk_article(
            article_data['text'],
            {
                'title': article_data['title'],
                'url': article_data['url'],
                'source': article_data['source'],
                'date': article_data.get('date', ''),
                'fetch_method': article_data.get('fetch_method', '')
            }
        )
        
        print(f" Created {len(chunks)} chunks")
        
        # Step 3: Add to vector store
        self.vector_store.add_chunks_to_store(chunks, article_id=article_data['url'])
        
        return {
            'article': article_data,
            'chunks_count': len(chunks),
            'total_documents': self.vector_store.collection.count()
        }
    
    def ask_question(self, question: str, n_context_chunks: int = 3) -> Dict:
        """
        Ask a question about processed articles
        
        Args:
            question: User question
            n_context_chunks: Number of chunks to retrieve
            
        Returns:
            Dictionary with answer and context
        """
        print(f"\n Question: {question}")
        
        # Step 1: Retrieve relevant chunks
        print(f" Retrieving {n_context_chunks} most relevant chunks...")
        context_chunks = self.vector_store.search_similar(question, n_results=n_context_chunks)
        
        if not context_chunks:
            return {
                'answer': "I couldn't find any relevant information in the processed articles.",
                'context_chunks': [],
                'sources': []
            }
        
        # Display retrieved context
        print(f" Found {len(context_chunks)} relevant chunks:")
        for i, chunk in enumerate(context_chunks, 1):
            print(f"  {i}. Similarity: {chunk['similarity_score']:.3f}")
            print(f"     {chunk['document'][:100]}...")
        
        # Step 2: Generate answer with LLM
        llm_result = self.llm.generate_answer(question, context_chunks)
        
        # Extract sources from context
        sources = []
        for chunk in context_chunks:
            metadata = chunk.get('metadata', {})
            source_info = {
                'title': metadata.get('title', 'Unknown'),
                'url': metadata.get('url', ''),
                'chunk': metadata.get('chunk_number', 0),
                'similarity': chunk.get('similarity_score', 0)
            }
            sources.append(source_info)
        
        return {
            'answer': llm_result['answer'],
            'context_chunks': context_chunks,
            'sources': sources,
            'llm_usage': llm_result.get('usage'),
            'model': llm_result.get('model')
        }
    
    def _fetch_article(self, url: str, use_playwright: bool = False) -> Optional[Dict]:
        """Fetch article from URL"""
        # Try multiple methods in order
        try:
            print(f" Fetching article...")
            
            # Method 1: Try newspaper3k
            try:
                print("  Trying newspaper3k...")
                from newspaper import Article
                
                article = Article(url)
                article.download()
                article.parse()
                
                return {
                    'url': url,
                    'title': article.title,
                    'text': article.text,
                    'source': url.split('//')[1].split('/')[0],
                    'date': str(article.publish_date) if article.publish_date else '',
                    'fetch_method': 'newspaper3k'
                }
            except Exception as e:
                print(f"  newspaper3k failed: {e}")
            
            # Method 2: Try trafilatura
            try:
                print("  Trying trafilatura...")
                import trafilatura
                
                downloaded = trafilatura.fetch_url(url)
                text = trafilatura.extract(downloaded)
                
                if text and len(text) > 100:
                    return {
                        'url': url,
                        'title': 'Extracted Article',  # Trafilatura might not get title
                        'text': text,
                        'source': url.split('//')[1].split('/')[0],
                        'date': '',
                        'fetch_method': 'trafilatura'
                    }
            except Exception as e:
                print(f"  trafilatura failed: {e}")
            
            # Method 3: Try requests + BeautifulSoup (fallback)
            try:
                print("  Trying requests + BeautifulSoup...")
                import requests
                from bs4 import BeautifulSoup
                
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(url, headers=headers, timeout=10)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Get title
                title = soup.find('title')
                title_text = title.text if title else 'News Article'
                
                # Get main content
                body = soup.find('body')
                text = body.get_text(separator='\n', strip=True) if body else ''
                
                return {
                    'url': url,
                    'title': title_text,
                    'text': text[:10000],  # Limit size
                    'source': url.split('//')[1].split('/')[0],
                    'date': '',
                    'fetch_method': 'requests'
                }
            except Exception as e:
                print(f"  requests failed: {e}")
            
            print(" All fetch methods failed")
            return None
            
        except Exception as e:
            print(f" Error fetching article: {e}")
            return None
    
    def list_processed_articles(self) -> List[Dict]:
        """List all articles in the vector store"""
        # Get unique articles from collection metadata
        articles = set()
        
        # Note: This is simplified - in production, you'd query metadata properly
        collection_info = self.vector_store.get_collection_info()
        
        return [{
            'total_documents': collection_info['count'],
            'collection': collection_info['name']
        }]

def main():
    """Main function to test the complete pipeline"""
    print("=" * 70)
    print("NEWS RAG PIPELINE WITH GROQ LLM INTEGRATION")
    print("=" * 70)
    
    # Initialize pipeline
    pipeline = NewsRAGPipeline()
    
    # Test with a news URL
    test_urls = [
        "https://www.bbc.com/news/technology",  # BBC Technology page
        # "https://www.reuters.com/technology/",  # Reuters Technology
        # "https://www.theverge.com/tech"  # The Verge
    ]
    
    # Process each URL
    for url in test_urls:
        print(f"\n{'='*70}")
        print(f"PROCESSING URL: {url}")
        print('='*70)
        
        try:
            # Process the URL
            result = pipeline.process_news_url(url)
            
            if 'error' in result:
                print(f" Error: {result['error']}")
                continue
            
            print(f" Processed: {result['article']['title']}")
            print(f"  Chunks: {result['chunks_count']}")
            print(f"  Total in DB: {result['total_documents']} documents")
            
            # Ask a question about the article
            if "technology" in url.lower():
                question = "What are the main technology topics discussed?"
            else:
                question = "What is this article about?"
            
            print(f"\n Asking: {question}")
            answer_result = pipeline.ask_question(question)
            
            print(f"\n ANSWER:")
            print("-" * 50)
            print(answer_result['answer'])
            print("-" * 50)
            
            if answer_result.get('llm_usage'):
                usage = answer_result['llm_usage']
                print(f"\n Token Usage:")
                print(f"  Prompt: {usage['prompt_tokens']}")
                print(f"  Completion: {usage['completion_tokens']}")
                print(f"  Total: {usage['total_tokens']}")
            
            print(f"\n Sources:")
            for source in answer_result.get('sources', [])[:3]:
                print(f"  • {source['title']} (similarity: {source['similarity']:.3f})")
            
        except Exception as e:
            print(f" Error processing {url}: {e}")
    
    # Interactive mode
    print(f"\n{'='*70}")
    print("INTERACTIVE MODE")
    print("=" * 70)
    print("You can now ask questions about the processed articles.")
    print("Type 'exit' to quit, 'new url' to process another article.")
    print("-" * 70)
    
    while True:
        user_input = input("\n Your question (or 'new url'): ").strip()
        
        if user_input.lower() == 'exit':
            break
        
        if user_input.lower() == 'new url':
            new_url = input("Enter news URL: ").strip()
            if new_url:
                pipeline.process_news_url(new_url)
            continue
        
        if user_input:
            result = pipeline.ask_question(user_input)
            
            print(f"\n ANSWER:")
            print("-" * 50)
            print(result['answer'])
            print("-" * 50)
            
            if result.get('sources'):
                print(f"\n Based on {len(result['sources'])} sources:")
                for source in result['sources'][:3]:
                    print(f"  • {source['title']}")

if __name__ == "__main__":
    main()