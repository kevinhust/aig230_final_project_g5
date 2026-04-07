"""
RAG Pipeline for Multilingual E-Commerce Customer Support

This module implements the core RAG engine with:
- Multilingual support (EN, ZH, FR, ES)
- Code-switching detection
- Sentiment analysis
- Source citations

Supports:
- ZhipuAI (GLM) API
- HuggingFace API
- Ollama local
- Mock mode for testing
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables FIRST
# Find .env file - check multiple locations
env_locations = [
    Path.cwd() / '.env',  # Current working directory
    Path(__file__).parent.parent / '.env',  # Parent of src directory
    Path(__file__).parent / '.env',  # Same directory as this file
]

env_path = None
for loc in env_locations:
    if loc.exists():
        env_path = loc
        break

load_dotenv(dotenv_path=env_path)

# Debug: Uncomment to check if key is loaded
# print(f"DEBUG: ZAI_API_KEY loaded: {bool(os.environ.get('ZAI_API_KEY'))}")
# print(f"DEBUG: env_path: {env_path}")

from langdetect import detect
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Import dedicated sentiment module
try:
    from sentiment import analyze_sentiment as _analyze_sentiment, get_escalation_response as _get_escalation_response
except ImportError:
    from src.sentiment import analyze_sentiment as _analyze_sentiment, get_escalation_response as _get_escalation_response

# Configuration
CHROMA_PATH = "data/chroma_db"
EMBEDDING_MODEL = "BAAI/bge-m3"
LLM_REPO_ID = "Qwen/Qwen2.5-7B-Instruct"

# API Keys - read AFTER load_dotenv()
ZAI_API_KEY = os.environ.get("ZAI_API_KEY", "")
HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN", "")

# Sentiment thresholds
SENTIMENT_ANGRY_THRESHOLD = -0.3


def detect_primary_lang(query):
    """Detect the primary language of the query."""
    if not query or len(query.strip()) < 3:
        return 'English'
    try:
        lang = detect(query)
        mapping = {
            'zh': '中文', 'ko': 'Korean',
            'ja': 'Japanese', 'en': 'English', 'fr': 'Français',
            'es': 'Español', 'de': 'Deutsch', 'it': 'Italiano',
            'pt': 'Português', 'ru': 'Russian', 'ar': 'Arabic'
        }
        return mapping.get(lang.split('-')[0], 'English')
    except:
        return 'English'


def detect_code_switching(query):
    """Check if query contains mixed languages."""
    try:
        has_chinese = any('\u4e00' <= c <= '\u9fff' for c in query)
        has_latin = any(c.isalpha() and ord(c) < 128 for c in query)
        return has_chinese and has_latin
    except:
        return False


def simple_sentiment_analysis(query):
    """Rule-based sentiment analysis using the dedicated sentiment module."""
    result = _analyze_sentiment(query)
    return result['score'], result['is_angry'], result['escalation_needed']


def get_escalation_response(lang):
    """Get appropriate escalation response based on language."""
    return _get_escalation_response(lang)


def format_docs(docs):
    """Format documents into a single string."""
    return "\n\n".join(f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}" for doc in docs)


def generate_mock_response(query, docs, primary_lang):
    """Generate a mock response for testing without LLM."""
    # Extract key information from retrieved docs
    sources = [doc.metadata.get('source', 'Unknown') for doc in docs]

    # Simple template-based response
    responses = {
        'English': f"Based on our knowledge base, I found information related to your query. Please refer to our policies for details. [Source: {', '.join(set(sources))}]",
        '中文': f"根据我们的知识库，我找到了与您查询相关的信息。请参阅我们的政策了解详情。[来源: {', '.join(set(sources))}]",
        'Français': f"Selon notre base de connaissances, j'ai trouvé des informations liées à votre demande. Veuillez vous référer à nos politiques pour plus de détails. [Source: {', '.join(set(sources))}]",
        'Español': f"Según nuestra base de conocimientos, encontré información relacionada con su consulta. Consulte nuestras políticas para obtener más detalles. [Fuente: {', '.join(set(sources))}]"
    }
    return responses.get(primary_lang, responses['English'])


class RAGEngine:
    def __init__(self, use_ollama=False, use_mock=False, similarity_threshold=0.3):
        print(f"Initializing RAG Engine (embeddings: {EMBEDDING_MODEL})...")
        self.similarity_threshold = similarity_threshold
        self.chat_history = {}  # session_id -> list of messages

        # Determine mode: prefer ZhipuAI > HuggingFace > Ollama > Mock
        self.use_mock = use_mock
        self.llm_type = None

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )

        # Load vector store
        self.vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=self.embeddings
        )

        # Initialize LLM
        self.llm = None
        if not self.use_mock:
            # Try ZhipuAI first (GLM-4.7)
            if ZAI_API_KEY:
                try:
                    from langchain_openai import ChatOpenAI
                    self.llm = ChatOpenAI(
                        model="glm-4.7",
                        openai_api_key=ZAI_API_KEY,
                        openai_api_base="https://open.bigmodel.cn/api/paas/v4",
                        temperature=0.3,
                        max_tokens=512,
                    )
                    self.llm_type = "zhipuai"
                    print("LLM initialized: ZhipuAI GLM-4.7")
                except Exception as e:
                    print(f"Warning: Could not initialize ZhipuAI ({e})")

            # Fallback to HuggingFace
            if self.llm is None and HUGGINGFACEHUB_API_TOKEN:
                try:
                    from langchain_huggingface import HuggingFaceEndpoint
                    self.llm = HuggingFaceEndpoint(
                        repo_id=LLM_REPO_ID,
                        task="text-generation",
                        max_new_tokens=512,
                        do_sample=False,
                        repetition_penalty=1.03,
                        temperature=0.3,
                    )
                    self.llm_type = "huggingface"
                    print("LLM initialized: HuggingFace")
                except Exception as e:
                    print(f"Warning: Could not initialize HuggingFace ({e})")

            # Fallback to Ollama
            if self.llm is None and use_ollama:
                try:
                    from langchain_ollama import ChatOllama
                    self.llm = ChatOllama(model="qwen2.5:7b", temperature=0.3)
                    self.llm_type = "ollama"
                    print("LLM initialized: Ollama")
                except Exception as e:
                    print(f"Warning: Could not initialize Ollama ({e})")

            # If all fail, use mock
            if self.llm is None:
                print("No LLM available. Using mock mode.")
                self.use_mock = True

        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        mode = "MOCK" if self.use_mock else self.llm_type.upper()
        print(f"RAG Engine initialized successfully! (Mode: {mode})")

    def retrieve_with_scores(self, query, k=5):
        """Retrieve documents with similarity scores."""
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
        return [(doc, score) for doc, score in docs_with_scores]

    def ask(self, query, include_scores=False, session_id="default"):
        """
        Process a query and return response with metadata.
        """
        # Detect language and sentiment
        primary_lang = detect_primary_lang(query)
        is_code_switching = detect_code_switching(query)
        sentiment_score, is_angry, escalation_needed = simple_sentiment_analysis(query)

        # Check for escalation - return early with escalation message
        if escalation_needed:
            return {
                "answer": get_escalation_response(primary_lang),
                "sources": [],
                "detected_lang": primary_lang,
                "is_code_switching": is_code_switching,
                "sentiment": {
                    "score": sentiment_score,
                    "is_angry": is_angry,
                    "escalation_needed": escalation_needed
                },
                "escalated": True
            }

        # Get retrieval scores if requested
        retrieval_results = None
        if include_scores:
            retrieval_results = self.retrieve_with_scores(query)

        # Retrieve documents
        docs = self.retriever.invoke(query)
        sources = list(set([doc.metadata.get('source', 'Unknown') for doc in docs]))

        # Generate response
        if self.use_mock:
            answer = generate_mock_response(query, docs, primary_lang)
        else:
            try:
                from langchain_core.prompts import ChatPromptTemplate

                prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are a multilingual e-commerce customer service bot.

Instructions:
1. Respond primarily in the user's detected language: {primary_lang}
2. If the user uses code-switching (mixed languages), naturally mix languages
3. ALWAYS cite sources using [Source: filename] format
4. Base your answer ONLY on the provided context
5. If you don't know the answer, say so politely in {primary_lang}
6. Keep responses concise (2-4 sentences)

Context:
{context}"""),
                    ("human", "{question}")
                ])

                context_str = format_docs(docs)
                prompt_value = prompt.invoke({
                    "context": context_str,
                    "question": query,
                    "primary_lang": primary_lang
                })
                result = self.llm.invoke(prompt_value)
                answer = result.content if hasattr(result, 'content') else str(result)

            except Exception as e:
                print(f"Error in LLM call: {e}")
                answer = generate_mock_response(query, docs, primary_lang)

        # Add apologetic prefix for angry customers
        if is_angry and not escalation_needed:
            apology_prefix = {
                'English': "I sincerely apologize for this inconvenience. ",
                '中文': "非常抱歉给您带来不便。 ",
                'Français': "Je vous présente mes sincères excuses pour ce désagrément. ",
                'Español': "Le pido sinceras disculpas por este inconveniente. "
            }
            answer = apology_prefix.get(primary_lang, apology_prefix['English']) + answer

        # Store in session history
        if session_id not in self.chat_history:
            self.chat_history[session_id] = []
        self.chat_history[session_id].append({"role": "user", "content": query})
        self.chat_history[session_id].append({"role": "assistant", "content": answer})

        response = {
            "answer": answer,
            "sources": sources,
            "detected_lang": primary_lang,
            "is_code_switching": is_code_switching,
            "sentiment": {
                "score": sentiment_score,
                "is_angry": is_angry,
                "escalation_needed": escalation_needed
            },
            "escalated": False
        }

        if include_scores:
            response["retrieval_scores"] = retrieval_results

        return response

    def clear_memory(self, session_id="default"):
        """Clear conversation memory for a session."""
        if session_id in self.chat_history:
            del self.chat_history[session_id]
            print(f"Conversation memory cleared for session: {session_id}")
        else:
            print("No session memory to clear.")

    def get_memory_history(self, session_id="default"):
        """Get the current conversation history for a session."""
        return self.chat_history.get(session_id, [])


if __name__ == "__main__":
    # Quick Test
    print("=" * 50)
    print("RAG Engine Test")
    print("=" * 50)

    # Auto-detect mode based on available API keys
    engine = RAGEngine()

    # Test queries
    test_queries = [
        "我的 fridge 噪音大，保修怎么走？",
        "How do I return a broken washer?",
        "This is UNACCEPTABLE! I want to speak to a manager!",
        "Mon téléphone ne s'allume pas, que faire ?",
        "¿Cómo solicito una devolución?",
        "我的laptop screen flickering，warranty能cover吗？"
    ]

    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print("-" * 50)
        response = engine.ask(query, include_scores=True)
        print(f"Language: {response['detected_lang']}")
        print(f"Code-switching: {response['is_code_switching']}")
        print(f"Sentiment: score={response['sentiment']['score']:.2f}, angry={response['sentiment']['is_angry']}")
        print(f"Escalated: {response['escalated']}")
        print(f"Sources: {response['sources']}")
        print(f"\nAnswer: {response['answer']}")
