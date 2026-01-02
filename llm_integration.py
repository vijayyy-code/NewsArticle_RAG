# llm_integration.py
import os
from groq import Groq
from typing import List, Dict, Any
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GroqLLM:
    """Handle LLM interactions with Groq API"""
    
    def __init__(self):
        """Initialize Groq client with API key"""
        api_key = os.getenv("GROQ_API_KEY")
        model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.client = Groq(api_key=api_key)
        self.model = model_name
        
        print(f" Groq LLM initialized")
        print(f"  Model: {self.model}")
    
    def generate_answer(self, query: str, context_chunks: List[Dict], max_tokens: int = 500) -> Dict:
        """
        Generate answer using retrieved context
        
        Args:
            query: User question
            context_chunks: List of retrieved chunks with metadata
            max_tokens: Maximum tokens in response
            
        Returns:
            Dictionary with answer and metadata
        """
        # Prepare context from retrieved chunks
        context_text = self._prepare_context(context_chunks)
        
        # Create system prompt
        system_prompt = """You are a helpful news assistant. Use the provided news article context to answer questions accurately.
        If the context doesn't contain relevant information, say so.
        Always cite sources by mentioning which part of the article you're referencing.
        Keep answers concise and factual."""
        
        # Create user prompt with context
        user_prompt = f"""CONTEXT FROM NEWS ARTICLE:
{context_text}

USER QUESTION: {query}

Based ONLY on the context provided above, answer the question. If you cannot answer from the context, say "I cannot find specific information about this in the provided article."
"""
        
        try:
            print(f"\n Generating answer with {self.model}...")
            
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Lower temperature for factual responses
                max_tokens=max_tokens,
                top_p=0.9
            )
            
            answer = response.choices[0].message.content
            
            # Extract usage info
            usage = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
            
            return {
                'answer': answer,
                'usage': usage,
                'model': self.model,
                'context_chunks_used': len(context_chunks)
            }
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            return {
                'answer': f"Error: {str(e)}",
                'usage': None,
                'model': self.model,
                'context_chunks_used': len(context_chunks)
            }
    
    def _prepare_context(self, context_chunks: List[Dict]) -> str:
        """Format context chunks for LLM prompt"""
        context_parts = []
        
        for i, chunk in enumerate(context_chunks, 1):
            chunk_text = chunk.get('document', '')
            metadata = chunk.get('metadata', {})
            similarity = chunk.get('similarity_score', 0)
            
            context_part = f"[CHUNK {i} - Relevance: {similarity:.3f}]"
            if metadata.get('title'):
                context_part += f" From: {metadata['title']}"
            if metadata.get('chunk_number'):
                context_part += f" (Part {metadata['chunk_number']}/{metadata.get('total_chunks', '?')})"
            
            context_part += f"\n{chunk_text}\n"
            context_parts.append(context_part)
        
        return "\n---\n".join(context_parts)

def test_groq_connection():
    """Test Groq API connection"""
    print("=" * 60)
    print("TESTING GROQ LLM CONNECTION")
    print("=" * 60)
    
    try:
        llm = GroqLLM()
        
        # Simple test query
        test_query = "What is artificial intelligence in one sentence?"
        
        # Use minimal context for test
        test_context = [{
            'document': "Artificial intelligence is the simulation of human intelligence processes by machines, especially computer systems.",
            'metadata': {'title': 'AI Definition'},
            'similarity_score': 0.9
        }]
        
        print(f"\nTest query: {test_query}")
        result = llm.generate_answer(test_query, test_context, max_tokens=100)
        
        print(f"\n API Connection Successful!")
        print(f"Model: {result['model']}")
        print(f"Answer: {result['answer']}")
        
        if result['usage']:
            print(f"Token usage: {result['usage']['total_tokens']} tokens")
        
        return llm
        
    except Exception as e:
        print(f" Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check GROQ_API_KEY in .env file")
        print("2. Ensure you have API credits at https://console.groq.com")
        print("3. Check internet connection")
        return None

if __name__ == "__main__":
    test_groq_connection()