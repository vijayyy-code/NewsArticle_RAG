# main_improved.py
"""
NEWS RAG SYSTEM - IMPROVED VERSION
Better URL validation and error handling
"""
import os
import sys
import time
import re
from datetime import datetime
from typing import Optional, Dict
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from rag_pipeline import NewsRAGPipeline
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

class ImprovedRAGSystem:
    """Improved system with better URL handling"""
    
    def __init__(self):
        self.pipeline = None
        self.current_article = None
        
    def initialize(self):
        """Initialize system"""
        print("Initializing...")
        try:
            self.pipeline = NewsRAGPipeline()
            print("System ready!")
            return True
        except Exception as e:
            print(f"Failed: {e}")
            return False
    
    def is_valid_article_url(self, url: str) -> bool:
        """Check if URL looks like a specific article"""
        # Patterns that indicate section/list pages (not articles)
        section_patterns = [
            r'/incoming/$',           # Just /incoming/
            r'/section/',            # Section pages
            r'/category/',           # Category pages
            r'/tag/',               # Tag pages
            r'/search/',            # Search results
            r'\?.*page=',           # Pagination
            r'/$'                   # Directory listing
        ]
        
        # Patterns that indicate actual articles
        article_patterns = [
            r'/article\d+\.ece$',    # The Hindu articles
            r'/story/',              # Story pages
            r'-\d+$',                # Ending with numbers (common for news)
            r'\.html$',              # HTML pages
            r'\.php\?id=',          # PHP articles
        ]
        
        url_lower = url.lower()
        
        # Check if it's a section page
        for pattern in section_patterns:
            if re.search(pattern, url_lower):
                return False
        
        # Check if it looks like an article
        for pattern in article_patterns:
            if re.search(pattern, url_lower):
                return True
        
        # Default: assume it's valid if it has decent length
        return len(url) > 40  # Article URLs are usually longer
    
    def get_valid_article_url(self) -> Optional[str]:
        """Get a valid article URL from user"""
        print("\n" + "="*70)
        print("ENTER A SPECIFIC NEWS ARTICLE URL")
        print("="*70)
        print("\nDon't paste section/list pages like: https://www.thehindu.com/incoming/")
        print("Paste a FULL ARTICLE URL like:")
        print("\nEXAMPLES:")
        print("1. The Hindu (Chess):")
        print("   https://www.thehindu.com/incoming/gcl-caruana-ends-firouzjas-winning-run-to-help-pipers-pip-continental-kings/article70417370.ece")
        print("\n2. The Hindu (Garbage):")
        print("   https://www.thehindu.com/incoming/a-tour-ofmini-garbagedumps-inmedavakkam/article70420330.ece")
        print("\n3. BBC Technology:")
        print("   https://www.bbc.com/news/technology-68979323")
        print("\n4. Indian Express:")
        print("   https://indianexpress.com/article/india/nadda-big-claim-congress-insiders-2013-naxal-attack-killed-leaders-10433435/")
        print("\n" + "-"*70)
        
        while True:
            url = input("\nPaste FULL article URL: ").strip()
            
            if not url:
                print("Please enter a URL")
                continue
            
            # Add https:// if missing
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            # Validate it's an article, not a section page
            if not self.is_valid_article_url(url):
                print("\nWARNING: This looks like a SECTION PAGE, not a specific article.")
                print(f"   You pasted: {url[:60]}...")
                print("   Article URLs are usually longer and contain '/article' or story IDs.")
                
                confirm = input("   Try anyway? (y/n): ").lower()
                if confirm != 'y':
                    print("   Please paste a complete article URL.")
                    continue
            
            return url
    
    def process_article(self, url: str) -> Optional[Dict]:
        """Process an article with validation"""
        print(f"\nProcessing article...")
        print(f"   URL: {url[:80]}...")
        
        start_time = time.time()
        
        try:
            result = self.pipeline.process_news_url(url, use_playwright=False)
            
            if 'error' in result:
                print(f"   Standard fetch failed: {result['error'][:100]}...")
                print("   Trying headless browser...")
                result = self.pipeline.process_news_url(url, use_playwright=True)
            
            if 'error' in result:
                print(f"   Failed: {result['error'][:100]}...")
                return None
            
            process_time = time.time() - start_time
            
            # Check if we got meaningful content
            article = result['article']
            text_length = len(article['text'])
            
            print(f"\nProcessed in {process_time:.1f}s")
            print(f"Title: {article['title']}")
            print(f"Text length: {text_length} characters")
            print(f"Method: {article.get('fetch_method', 'unknown')}")
            print(f"Chunks: {result['chunks_count']}")
            
            # Warn if content seems too short
            if text_length < 500:
                print(f"Warning: Article seems short ({text_length} chars). Might be limited content.")
            
            return result
            
        except Exception as e:
            print(f"Error: {str(e)[:100]}...")
            return None
    
    def qna_loop(self):
        """Question and Answer loop"""
        if not self.current_article:
            print("No article to ask about!")
            return
        
        article = self.current_article['article']
        
        print("\n" + "="*70)
        print("ASK QUESTIONS ABOUT THE ARTICLE")
        print("="*70)
        print(f"{article['title']}")
        print(f"{article.get('source', 'Unknown source')}")
        print(f"{len(article['text'])} characters | {self.current_article['chunks_count']} sections")
        print("\nCommands:")
        print("  'exit' or 'quit' - End session")
        print("  'new' - Process different article")
        print("  'summary' - Get article summary")
        print("-"*70)
        
        question_num = 1
        
        while True:
            try:
                # Get question
                question = input(f"\nQ{question_num}: ").strip()
                
                if not question:
                    continue
                
                # Check commands
                if question.lower() in ['exit', 'quit']:
                    print("\nSession ended.")
                    break
                
                if question.lower() == 'new':
                    print("\nStarting new article...")
                    return 'new'
                
                if question.lower() == 'summary':
                    question = "Provide a concise 3-point summary of this article."
                
                # Get answer
                print("Searching...")
                start = time.time()
                
                answer = self.pipeline.ask_question(question)
                response_time = time.time() - start
                
                # Display answer
                print(f"\n{'='*60}")
                print("ANSWER:")
                print('='*60)
                print(answer['answer'])
                print('='*60)
                
                # Show stats
                print(f"\nResponse: {response_time:.2f}s")
                if answer.get('llm_usage'):
                    tokens = answer['llm_usage']['total_tokens']
                    print(f"Tokens: {tokens}")
                
                question_num += 1
                
            except KeyboardInterrupt:
                print("\nInterrupted. Type 'exit' to quit.")
            except Exception as e:
                print(f"Error: {e}")
    
    def run(self):
        """Main run loop"""
        print("\n" + "="*70)
        print("NEWS RAG SYSTEM - IMPROVED")
        print("="*70)
        
        if not self.initialize():
            return
        
        while True:
            # Get URL
            url = self.get_valid_article_url()
            if not url:
                break
            
            # Process article
            result = self.process_article(url)
            if not result:
                retry = input("\nTry different URL? (y/n): ").lower()
                if retry != 'y':
                    break
                continue
            
            self.current_article = result
            
            # Q&A loop
            result = self.qna_loop()
            if result != 'new':
                break
        
        print("\nSession completed!")

def main():
    """Main entry point"""
    try:
        system = ImprovedRAGSystem()
        system.run()
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"\nFatal error: {e}")

if __name__ == "__main__":
    main()