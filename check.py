# news_qa_test.py
"""
Simple News Q&A Test
Provide a URL, then ask questions about it
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_pipeline import NewsRAGPipeline
import time

def news_qa_test():
    """Test Q&A for a specific news article"""
    print("=" * 70)
    print(" NEWS ARTICLE Q&A TEST")
    print("=" * 70)
    
    # Step 1: Get URL from user
    print("\nEnter a news article URL:")
    print("Example: https://www.bbc.com/news/technology-68979323")
    print("Or: https://indianexpress.com/article/india/nadda-big-claim-congress-insiders-2013-naxal-attack-killed-leaders-10433435/")
    print("-" * 50)
    
    news_url = input("News URL: ").strip()
    
    if not news_url:
        print("No URL provided. Using default.")
        news_url = "https://www.bbc.com/news/technology-68979323"
    
    # Step 2: Initialize pipeline
    print("\n Initializing RAG Pipeline...")
    pipeline = NewsRAGPipeline()
    
    # Step 3: Process the URL
    print(f"\n Processing: {news_url}")
    start_time = time.time()
    
    try:
        # Try with newspaper3k first
        result = pipeline.process_news_url(news_url, use_playwright=False)
        
        if 'error' in result:
            print(f"⚠️ Standard fetch failed: {result['error']}")
            print("Trying with headless browser...")
            result = pipeline.process_news_url(news_url, use_playwright=True)
        
        if 'error' in result:
            print(f" Failed to process URL: {result['error']}")
            return
        
        process_time = time.time() - start_time
        
        print(f" Successfully processed in {process_time:.1f} seconds")
        print(f" Title: {result['article']['title']}")
        print(f" Method: {result['article'].get('fetch_method', 'unknown')}")
        print(f" Text length: {len(result['article']['text'])} characters")
        print(f" Chunks created: {result['chunks_count']}")
        
    except Exception as e:
        print(f" Error: {e}")
        return
    
    # Step 4: Q&A Loop
    print("\n" + "=" * 70)
    print(" QUESTION & ANSWER MODE")
    print("=" * 70)
    print("Ask questions about the article. Type 'exit' to quit.")
    print("Type 'new' to process a different URL.")
    print("-" * 50)
    
    question_count = 0
    
    while True:
        question = input(f"\nQ{question_count + 1}: ").strip()
        
        if question.lower() == 'exit':
            break
        
        if question.lower() == 'new':
            print("\n Enter new URL:")
            new_url = input("URL: ").strip()
            if new_url:
                news_url = new_url
                print(f"Processing: {news_url}")
                result = pipeline.process_news_url(news_url)
                if 'error' not in result:
                    print(f" Processed: {result['article']['title']}")
                question_count = 0
            continue
        
        if not question:
            continue
        
        # Get answer
        print(f"\n Searching for answer...")
        start_q_time = time.time()
        
        answer_result = pipeline.ask_question(question, n_context_chunks=3)
        
        q_time = time.time() - start_q_time
        
        print(f"\n ANSWER ({q_time:.1f}s):")
        print("-" * 60)
        print(answer_result['answer'])
        print("-" * 60)
        
        # Show sources
        if answer_result.get('sources'):
            print(f"\n Based on {len(answer_result['sources'])} parts of the article:")
            for source in answer_result['sources'][:2]:  # Show top 2 sources
                title = source.get('title', 'Article')
                similarity = source.get('similarity', 0)
                print(f"  • {title} (relevance: {similarity:.3f})")
        
        # Show token usage
        if answer_result.get('llm_usage'):
            usage = answer_result['llm_usage']
            print(f"\n Tokens: {usage['total_tokens']} total")
        
        question_count += 1
    
    print("\n" + "=" * 70)
    print(" Q&A Session Complete")
    print(f" Summary: {question_count} questions answered")
    print(f" Total documents in knowledge base: {pipeline.vector_store.collection.count()}")
    print("=" * 70)

if __name__ == "__main__":
    news_qa_test()