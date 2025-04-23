import httpx
from sentence_transformers import SentenceTransformer
import logging
import nltk
from nltk.tokenize import sent_tokenize
from collections import Counter
from transformers import pipeline
import torch
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download necessary NLTK data (run once)
try:
    # Try to download punkt
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    nltk.download('punkt', quiet=True)
    logger.info("Successfully downloaded NLTK punkt data")
except Exception as e:
    logger.warning(f"Failed to download NLTK punkt data: {str(e)}")
    logger.warning("Some tokenization features may not work correctly.")
    # Fallback - define a simple sentence tokenizer function
    def simple_sent_tokenize(text):
        # Simple fallback tokenizer that splits on ., ! and ?
        if not hasattr(nltk, 'sent_tokenize'):
            return re.split(r'(?<=[.!?])\s+', text)
        return nltk.sent_tokenize(text)
    # Replace nltk.sent_tokenize with our fallback if needed
    if not hasattr(nltk, 'sent_tokenize'):
        nltk.sent_tokenize = simple_sent_tokenize

class NLPProcessor:
    def __init__(self, document_service_url, model_name="all-MiniLM-L6-v2"):
        """
        Initialize the NLP processor with the document service URL and the sentence transformer model.
        
        Args:
            document_service_url (str): URL of the Document Service
            model_name (str): Name of the sentence-transformers model to use
        """
        self.document_service_url = document_service_url
        logger.info(f"Loading sentence-transformers model: {model_name}")
        
        # Load the sentence transformer model - this will download the model on first run
        # all-MiniLM-L6-v2 is a good balance of speed and accuracy for beginners
        self.model = SentenceTransformer(model_name)
        logger.info("Model loaded successfully")
        
        # Initialize HTTP client for async requests
        self.http_client = httpx.AsyncClient(timeout=30.0)  # 30 second timeout
    
    async def generate_embedding(self, text):
        """
        Generate an embedding vector for the input text.
        
        Args:
            text (str): The input text
            
        Returns:
            list: The embedding vector as a list of floats
        """
        try:
            # Generate the embedding - this returns a numpy array
            embedding = self.model.encode(text)
            
            # Convert to list for JSON serialization
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    async def search_documents(self, query_text):
        """
        Send the query to the Document Service to search for relevant chunks.
        
        Args:
            query_text (str): The user's original query text
            
        Returns:
            list: Relevant text chunks from the Document Service
        """
        try:
            # Endpoint at the Document Service
            search_url = f"{self.document_service_url}/search"
            
            # Prepare the request data according to SearchRequest schema from Document Service
            data = {
                "query": query_text,  # Use the original query text
                "n_results": 5  # Number of results to return
            }
            
            # Send POST request to the Document Service
            logger.info(f"Sending search request to Document Service at {search_url}")
            response = await self.http_client.post(search_url, json=data)
            
            # Check if the request was successful
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Document Service returned error: {response.status_code}, {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error communicating with Document Service: {str(e)}")
            return []
    
    async def synthesize_answer(self, query, chunks):
        """
        Simple function to synthesize an answer from the retrieved chunks.
        In a real implementation, this could use a more sophisticated approach.
        
        Args:
            query (str): The original query
            chunks (list): The retrieved text chunks
            
        Returns:
            str: A synthesized answer
        """
        if not chunks:
            return "I couldn't find relevant information for your query."
        
        try:
            # Extract query keywords
            query_words = set(w.lower() for w in query.split() if len(w) > 3)
            
            # Tokenize chunks into sentences
            all_sentences = []
            for chunk in chunks:
                # Get the text from the chunk - handle different possible formats
                if isinstance(chunk, dict) and 'text' in chunk:
                    text = chunk['text']
                elif isinstance(chunk, str):
                    text = chunk
                else:
                    # Try to convert to string if it's some other type
                    text = str(chunk)
                
                # Try to get metadata
                metadata = {}
                if isinstance(chunk, dict):
                    metadata = chunk.get('metadata', {})
                
                try:
                    # Try to split into sentences
                    if hasattr(nltk, 'sent_tokenize'):
                        sentences = nltk.sent_tokenize(text)
                    else:
                        # Fallback - simple split on periods, exclamation marks, and question marks
                        sentences = re.split(r'(?<=[.!?])\s+', text)
                except Exception:
                    # If tokenization fails, treat the whole chunk as one sentence
                    sentences = [text]
                
                for sentence in sentences:
                    # Score each sentence by counting query keywords
                    score = sum(1 for word in query_words if word in sentence.lower())
                    all_sentences.append((sentence, score, metadata))
            
            # Sort sentences by relevance score
            all_sentences.sort(key=lambda x: x[1], reverse=True)
            
            # Take top sentences (avoid very low scores)
            top_sentences = [s for s in all_sentences if s[1] > 0][:5]
            
            # If we don't have enough scored sentences, add some high-quality ones
            if len(top_sentences) < 3:
                # Get sentences with 15+ words as they're likely more informative
                quality_sentences = [(s[0], s[1], s[2]) for s in all_sentences 
                                    if len(s[0].split()) > 15][:3]
                for sentence in quality_sentences:
                    if sentence not in top_sentences:
                        top_sentences.append(sentence)
            
            # Format the answer
            answer = f"Based on our ESG policies regarding '{query}':\n\n"
            
            for sentence, score, metadata in top_sentences:
                source = ""
                if metadata:
                    if 'title' in metadata:
                        source = f" (Source: {metadata['title']})"
                    elif 'filename' in metadata:
                        source = f" (Source: {metadata['filename']})"
                answer += f"• {sentence}{source}\n\n"
            
            return answer
            
        except Exception as e:
            logger.error(f"Error in synthesize_answer: {str(e)}")
            # Fallback to simpler method if the sophisticated one fails
            answer = f"Here are some relevant policies regarding '{query}':\n\n"
            for i, chunk in enumerate(chunks[:5], 1):
                if isinstance(chunk, dict) and 'text' in chunk:
                    text = chunk['text']
                else:
                    text = str(chunk)
                answer += f"• {text}\n\n"
            return answer
    
    async def process_query(self, query):
        """
        Process a user query:
        1. Generate embedding for the query
        2. Send the embedding to the Document Service
        3. Optionally synthesize an answer
        
        Args:
            query (str): The user's query text
            
        Returns:
            tuple: (list of relevant chunks, synthesized answer string)
        """
        try:
            logger.info(f"Processing query: {query}")
            
            # Modified to send the original query text directly to the Document Service
            # instead of using embeddings
            chunks = await self.search_documents(query)
            
            # Synthesize an answer (optional step)
            synthesized_answer = await self.synthesize_answer(query, chunks)
            
            return chunks, synthesized_answer
        except Exception as e:
            logger.error(f"Error in query processing pipeline: {str(e)}")
            raise

class AdvancedNLPProcessor(NLPProcessor):
    def __init__(self, document_service_url, model_name="all-MiniLM-L6-v2"):
        super().__init__(document_service_url, model_name)
        # Load a small summarization model - will work on M1 Mac with 16GB RAM
        self.summarizer = pipeline(
            "summarization", 
            model="facebook/bart-large-cnn", 
            device=-1  # CPU, use 0 for GPU if available
        )
        
    async def synthesize_answer(self, query, chunks):
        if not chunks:
            return "I couldn't find relevant information for your query."
        
        # Combine relevant chunks
        combined_text = " ".join([chunk['text'] for chunk in chunks])
        
        # For very long texts, split and summarize in parts
        max_length = 1024  # BART limitation
        if len(combined_text) > max_length:
            parts = [combined_text[i:i+max_length] for i in range(0, len(combined_text), max_length)]
            summaries = []
            
            for part in parts:
                if len(part.split()) > 50:  # Only summarize substantial parts
                    summary = self.summarizer(part, max_length=100, min_length=30)[0]['summary_text']
                    summaries.append(summary)
                else:
                    summaries.append(part)
            
            final_text = " ".join(summaries)
        else:
            # If text is short enough, summarize directly
            if len(combined_text.split()) > 50:
                final_text = self.summarizer(combined_text, max_length=150, min_length=50)[0]['summary_text']
            else:
                final_text = combined_text
        
        # Format with original query for context
        answer = f"In response to your question about {query}:\n\n{final_text}\n\n"
        
        # Add sources
        answer += "Sources:\n"
        seen_sources = set()
        for chunk in chunks:
            if 'metadata' in chunk and chunk['metadata']:
                source = chunk['metadata'].get('title', 'ESG Policy')
                if source not in seen_sources:
                    seen_sources.add(source)
                    answer += f"- {source}\n"
        
        return answer 

    async def enhanced_synthesize_answer(self, query, chunks):
        if not chunks:
            return "I couldn't find relevant information for your query."
        
        # 1. Extract query terms for highlighting
        query_terms = [term.lower() for term in re.findall(r'\b\w{4,}\b', query.lower())]
        
        # 2. Score chunks by relevance and de-duplicate
        unique_chunks = {}
        for chunk in chunks:
            text = chunk['text']
            # Simple scoring: count query terms in the chunk
            score = 0
            for term in query_terms:
                score += text.lower().count(term)
            
            # Store in dict with text as key to de-duplicate
            simplified = ' '.join(text.lower().split())[:100]  # First 100 chars for comparison
            if simplified not in unique_chunks or score > unique_chunks[simplified]['score']:
                unique_chunks[simplified] = {
                    'text': text,
                    'score': score,
                    'metadata': chunk.get('metadata', {})
                }
        
        # 3. Sort chunks by score
        sorted_chunks = sorted(unique_chunks.values(), key=lambda x: x['score'], reverse=True)
        
        # 4. Generate a simple introduction based on the query
        topic_words = Counter(query_terms).most_common(2)
        if topic_words:
            main_topic = topic_words[0][0].capitalize()
            answer = f"Regarding our ESG policies on {main_topic}:\n\n"
        else:
            answer = "Based on our ESG policies, here's what I found:\n\n"
        
        # 5. Add the top chunks with their source information
        for i, chunk in enumerate(sorted_chunks[:5]):  # Limit to top 5
            # Try to get source information
            source_info = ""
            if chunk.get('metadata'):
                meta = chunk['metadata']
                if 'title' in meta:
                    source_info = f" (Source: {meta['title']})"
                elif 'filename' in meta:
                    source_info = f" (Source: {meta['filename']})"
            
            # Highlight query terms in the text (simple approach)
            highlighted_text = chunk['text']
            for term in query_terms:
                if len(term) >= 4:  # Only highlight meaningful terms
                    pattern = re.compile(f'\\b{re.escape(term)}\\w*\\b', re.IGNORECASE)
                    highlighted_text = pattern.sub(lambda m: m.group(0), highlighted_text)
            
            answer += f"• {highlighted_text}{source_info}\n\n"
        
        # 6. Add a conclusion with next steps
        answer += "If you need more specific information or have follow-up questions, please let me know."
        
        return answer 