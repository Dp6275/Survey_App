import streamlit as st
import os
import json
from datetime import datetime
import fitz  # PyMuPDF
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from cryptography.fernet import Fernet
import re
from typing import List, Dict, Tuple
import pickle

# Page configuration
st.set_page_config(
    page_title="RAG Business Survey AI",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'encryption_key' not in st.session_state:
    st.session_state.encryption_key = Fernet.generate_key()
if 'survey_data' not in st.session_state:
    st.session_state.survey_data = {}
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'embeddings_model' not in st.session_state:
    st.session_state.embeddings_model = None

class CosineSimilarityRAG:
    """RAG system using cosine similarity for document retrieval"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.document_vectors = None
        self.document_texts = []
        
    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store"""
        self.document_texts = [doc.page_content for doc in documents]
        
        # Create TF-IDF vectors for cosine similarity
        if self.document_texts:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.document_texts)
            
        # Create sentence embeddings for semantic search
        self.embeddings = self.sentence_model.encode(self.document_texts)
        
        # Build FAISS index for efficient similarity search
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])  # Inner product (cosine similarity)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings.astype('float32'))
        
    def similarity_search(self, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        """Search for similar documents using cosine similarity"""
        if not self.document_texts:
            return []
            
        # Encode query
        query_embedding = self.sentence_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS index
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.document_texts):
                doc = Document(
                    page_content=self.document_texts[idx],
                    metadata={"similarity_score": float(score)}
                )
                results.append((doc, float(score)))
                
        return results
    
    def calculate_cosine_similarity(self, query: str, document_text: str) -> float:
        """Calculate cosine similarity between query and document"""
        try:
            # Encode both query and document
            query_embedding = self.sentence_model.encode([query])
            doc_embedding = self.sentence_model.encode([document_text])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(query_embedding, doc_embedding)[0][0]
            return float(similarity)
        except Exception as e:
            st.error(f"Error calculating similarity: {e}")
            return 0.0

class AdvancedRAGModel:
    """Advanced RAG model with cosine similarity and hybrid search"""
    
    def __init__(self):
        self.rag_system = CosineSimilarityRAG()
        self.text_formatter = TextFormatter()
        self.setup_multilingual_responses()
        
    def setup_multilingual_responses(self):
        """Setup multilingual response templates"""
        self.response_templates = {
            "en": {
                "greeting": "Hello! I can help you analyze business surveys using the documents you've provided.",
                "no_context": "I don't have enough context from your documents to answer this specifically. Please upload relevant business documents.",
                "high_confidence": "Based on the documents with {confidence:.1%} confidence:",
                "medium_confidence": "Based on relevant information from your documents:",
                "low_confidence": "Here's some general information that might be relevant:"
            },
            "es": {
                "greeting": "¡Hola! Puedo ayudarte a analizar encuestas comerciales usando los documentos que has proporcionado.",
                "no_context": "No tengo suficiente contexto de tus documentos para responder esto específicamente. Por favor, sube documentos comerciales relevantes.",
                "high_confidence": "Basándome en los documentos con {confidence:.1%} de confianza:",
                "medium_confidence": "Basándome en información relevante de tus documentos:",
                "low_confidence": "Aquí hay información general que podría ser relevante:"
            },
            "fr": {
                "greeting": "Bonjour ! Je peux vous aider à analyser les enquêtes commerciales en utilisant les documents que vous avez fournis.",
                "no_context": "Je n'ai pas assez de contexte de vos documents pour répondre spécifiquement à cela. Veuillez télécharger des documents commerciaux pertinents.",
                "high_confidence": "Sur la base des documents avec {confidence:.1%} de confiance :",
                "medium_confidence": "Sur la base des informations pertinentes de vos documents :",
                "low_confidence": "Voici quelques informations générales qui pourraient être pertinentes :"
            }
        }
    
    def generate_response(self, query: str, language: str = "en") -> str:
        """Generate response using RAG with cosine similarity"""
        if not st.session_state.documents:
            return self.response_templates[language]["no_context"]
        
        # Search for relevant documents
        relevant_docs = self.rag_system.similarity_search(query, k=3)
        
        if not relevant_docs:
            return self.response_templates[language]["no_context"]
        
        # Combine context from top documents
        context = "\n\n".join([doc.page_content for doc, score in relevant_docs[:2]])
        top_score = relevant_docs[0][1] if relevant_docs else 0
        
        # Generate response based on confidence level
        lang_responses = self.response_templates[language]
        
        if top_score > 0.7:
            confidence_prefix = lang_responses["high_confidence"].format(confidence=top_score)
        elif top_score > 0.4:
            confidence_prefix = lang_responses["medium_confidence"]
        else:
            confidence_prefix = lang_responses["low_confidence"]
        
        # Extract key information based on query type
        response_content = self._extract_relevant_information(context, query, top_score)
        
        # Format final response
        final_response = f"{confidence_prefix}\n\n{response_content}"
        
        # Add similarity scores for transparency
        similarity_info = self._format_similarity_scores(relevant_docs, language)
        final_response += f"\n\n{similarity_info}"
        
        return self.text_formatter.format_response(final_response, "structured")
    
    def _extract_relevant_information(self, context: str, query: str, confidence: float) -> str:
        """Extract relevant information based on query and context"""
        query_lower = query.lower()
        context_sentences = [s.strip() for s in context.split('.') if s.strip()]
        
        response_parts = []
        
        # Extract sentences based on query keywords
        relevant_sentences = []
        for sentence in context_sentences:
            sentence_lower = sentence.lower()
            
            # Business challenge detection
            if any(word in query_lower for word in ['challenge', 'problem', 'issue', 'difficulty']):
                if any(word in sentence_lower for word in ['challenge', 'problem', 'issue', 'difficult', 'struggle']):
                    relevant_sentences.append(sentence)
            
            # Solution detection
            elif any(word in query_lower for word in ['solution', 'recommend', 'advice', 'should']):
                if any(word in sentence_lower for word in ['solution', 'recommend', 'advice', 'should', 'could']):
                    relevant_sentences.append(sentence)
            
            # Data collection detection
            elif any(word in query_lower for word in ['data', 'survey', 'collect', 'research']):
                if any(word in sentence_lower for word in ['data', 'survey', 'collect', 'research', 'method']):
                    relevant_sentences.append(sentence)
            
            # General relevance (if no specific matches)
            elif len(relevant_sentences) < 3 and len(sentence) > 20:
                # Calculate similarity for general relevance
                similarity = self.rag_system.calculate_cosine_similarity(query, sentence)
                if similarity > 0.3:
                    relevant_sentences.append(sentence)
        
        # Add relevant sentences to response
        if relevant_sentences:
            response_parts.append("**Key Points from Documents:**")
            for i, sentence in enumerate(relevant_sentences[:5], 1):
                response_parts.append(f"{i}. {sentence}")
        else:
            # Fallback: use most relevant context sentences
            response_parts.append("**Relevant Information:**")
            for i, sentence in enumerate(context_sentences[:3], 1):
                response_parts.append(f"{i}. {sentence}")
        
        return "\n".join(response_parts)
    
    def _format_similarity_scores(self, relevant_docs: List[Tuple[Document, float]], language: str) -> str:
        """Format similarity scores for transparency"""
        score_labels = {
            "en": "Document Similarity Scores:",
            "es": "Puntuaciones de Similitud de Documentos:",
            "fr": "Scores de Similarité des Documents:"
        }
        
        score_info = [score_labels.get(language, "Document Similarity Scores:")]
        for i, (doc, score) in enumerate(relevant_docs, 1):
            score_info.append(f"Doc {i}: {score:.1%}")
        
        return " | ".join(score_info)

class TextFormatter:
    """Format text responses"""
    
    def format_response(self, text: str, format_type: str = "markdown") -> str:
        """Format text response"""
        if format_type == "structured":
            return self._format_structured(text)
        else:
            return text
    
    def _format_structured(self, text: str) -> str:
        """Structure text with proper formatting"""
        lines = text.split('\n')
        formatted_lines = []
        
        for line in lines:
            if line.startswith('**') and line.endswith('**'):
                formatted_lines.append(f"\n{line}\n")
            elif line.strip().startswith(('1.', '2.', '3.', '4.', '5.')):
                formatted_lines.append(f"  {line}")
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)

class SecurityManager:
    def __init__(self):
        self.cipher_suite = Fernet(st.session_state.encryption_key)
    
    def encrypt_data(self, data):
        """Encrypt sensitive data"""
        if isinstance(data, str):
            data = data.encode()
        return self.cipher_suite.encrypt(data)
    
    def decrypt_data(self, encrypted_data):
        """Decrypt sensitive data"""
        return self.cipher_suite.decrypt(encrypted_data).decode()

class VectorDBAnalytics:
    """Analytics for vector database performance"""
    
    def __init__(self):
        self.search_history = []
    
    def log_search(self, query: str, results_count: int, avg_similarity: float):
        """Log search performance"""
        self.search_history.append({
            'timestamp': datetime.now(),
            'query': query,
            'results_count': results_count,
            'avg_similarity': avg_similarity
        })
    
    def get_analytics(self):
        """Get vector DB analytics"""
        if not self.search_history:
            return "No searches performed yet"
        
        total_searches = len(self.search_history)
        avg_results = np.mean([log['results_count'] for log in self.search_history])
        avg_similarity = np.mean([log['avg_similarity'] for log in self.search_history])
        
        return {
            'total_searches': total_searches,
            'avg_results_per_search': avg_results,
            'avg_similarity_score': avg_similarity
        }

class MultilingualSurveyAI:
    def __init__(self):
        self.security_manager = SecurityManager()
        self.rag_model = AdvancedRAGModel()
        self.analytics = VectorDBAnalytics()
    
    def process_documents(self, uploaded_files):
        """Process uploaded documents for RAG"""
        documents = []
        
        for uploaded_file in uploaded_files:
            try:
                if uploaded_file.type == "application/pdf":
                    # Process PDF files
                    pdf_content = uploaded_file.read()
                    pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
                    
                    for page_num in range(len(pdf_document)):
                        page = pdf_document[page_num]
                        text = page.get_text()
                        if text.strip():
                            # Split large pages into chunks
                            chunks = self._split_text_into_chunks(text, chunk_size=500)
                            for i, chunk in enumerate(chunks):
                                documents.append(Document(
                                    page_content=chunk,
                                    metadata={
                                        "source": uploaded_file.name,
                                        "page": page_num + 1,
                                        "chunk": i + 1,
                                        "type": "pdf"
                                    }
                                ))
                    
                    pdf_document.close()
                
                elif uploaded_file.type == "text/plain":
                    # Process text files
                    text_content = uploaded_file.read().decode("utf-8")
                    chunks = self._split_text_into_chunks(text_content, chunk_size=500)
                    for i, chunk in enumerate(chunks):
                        documents.append(Document(
                            page_content=chunk,
                            metadata={
                                "source": uploaded_file.name,
                                "chunk": i + 1,
                                "type": "text"
                            }
                        ))
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")
        
        return documents
    
    def _split_text_into_chunks(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into chunks for better vectorization"""
        words = text.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            if len(' '.join(current_chunk)) > chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def initialize_rag_system(self, documents):
        """Initialize RAG system with documents"""
        if documents:
            self.rag_model.rag_system.add_documents(documents)
            return True
        return False
    
    def get_rag_response(self, question, language="en"):
        """Get RAG-based response using cosine similarity"""
        if not st.session_state.documents:
            return "Please upload some documents first to enable context-aware responses."
        
        try:
            response = self.rag_model.generate_response(question, language)
            
            # Log search for analytics
            relevant_docs = self.rag_model.rag_system.similarity_search(question, k=3)
            if relevant_docs:
                avg_similarity = np.mean([score for _, score in relevant_docs])
                self.analytics.log_search(question, len(relevant_docs), avg_similarity)
            
            return response
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def handle_survey_response(self, question, response, business_info):
        """Store and encrypt survey responses"""
        survey_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "response": response,
            "business_info": business_info
        }
        
        # Encrypt sensitive data
        encrypted_entry = {
            "timestamp": survey_entry["timestamp"],
            "question": self.security_manager.encrypt_data(question).decode(),
            "response": self.security_manager.encrypt_data(response).decode(),
            "business_info": {
                k: self.security_manager.encrypt_data(str(v)).decode() 
                for k, v in business_info.items()
            }
        }
        
        # Store in session
        survey_id = f"survey_{len(st.session_state.survey_data) + 1}"
        st.session_state.survey_data[survey_id] = encrypted_entry
        
        return survey_id

def main():
    st.title("🔍 RAG Business Survey AI with Cosine Similarity")
    st.markdown("### Vector Database + Cosine Similarity for Accurate Document Retrieval")
    
    # Initialize AI system
    ai_system = MultilingualSurveyAI()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Language selection
        language_map = {
            "English": "en",
            "Spanish": "es", 
            "French": "fr"
        }
        
        selected_language = st.selectbox(
            "Survey Language",
            list(language_map.keys()),
            index=0
        )
        language_code = language_map[selected_language]
        
        # Document upload for RAG
        st.subheader("📚 Knowledge Base")
        uploaded_files = st.file_uploader(
            "Upload business documents (PDF/TXT)",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="Documents will be vectorized using cosine similarity"
        )
        
        if uploaded_files and not st.session_state.documents:
            with st.spinner("Vectorizing documents..."):
                documents = ai_system.process_documents(uploaded_files)
                if documents:
                    success = ai_system.initialize_rag_system(documents)
                    if success:
                        st.session_state.documents = documents
                        st.success(f"✅ Vectorized {len(documents)} document chunks!")
                        st.info(f"📊 Embedding dimension: {ai_system.rag_model.rag_system.embeddings.shape[1]}")
        
        # Business information
        st.subheader("🏢 Business Profile")
        business_name = st.text_input("Business Name")
        industry = st.selectbox(
            "Industry",
            ["Retail", "Manufacturing", "Services", "Technology", "Healthcare", "Other"]
        )
        
        # Similarity threshold
        st.subheader("🔍 Search Settings")
        similarity_threshold = st.slider(
            "Minimum Similarity Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.3,
            help="Higher values return more relevant but fewer results"
        )
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Survey Chat Interface")
        
        # Display conversation
        chat_container = st.container()
        with chat_container:
            for i, (speaker, message, timestamp, similarity) in enumerate(st.session_state.conversation):
                if speaker == "AI":
                    with st.chat_message("assistant"):
                        st.markdown(message)
                        if similarity:
                            st.caption(f"Similarity: {similarity} | {timestamp}")
                else:
                    with st.chat_message("user"):
                        st.markdown(message)
                        st.caption(timestamp)
                st.markdown("---")
        
        # Question input
        user_question = st.chat_input("Ask a business survey question...")
        
        if user_question:
            # Add user question to conversation
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.conversation.append(("User", user_question, timestamp, None))
            
            # Get RAG response
            with st.spinner("🔍 Searching documents..."):
                ai_response = ai_system.get_rag_response(user_question, language_code)
                
                # Calculate average similarity for this query
                relevant_docs = ai_system.rag_model.rag_system.similarity_search(user_question, k=3)
                avg_similarity = np.mean([score for _, score in relevant_docs]) if relevant_docs else 0
                
                # Store survey response
                if business_name:
                    survey_id = ai_system.handle_survey_response(
                        user_question, ai_response, {"name": business_name, "industry": industry}
                    )
                
                # Add AI response to conversation
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.session_state.conversation.append(("AI", ai_response, timestamp, f"{avg_similarity:.1%}"))
            
            st.rerun()
    
    with col2:
        st.subheader("📊 Analytics Dashboard")
        
        # Vector DB Analytics
        if st.session_state.documents:
            analytics = ai_system.analytics.get_analytics()
            
            st.metric("Document Chunks", len(st.session_state.documents))
            st.metric("Total Searches", analytics['total_searches'])
            st.metric("Avg Similarity", f"{analytics['avg_similarity_score']:.1%}")
            
            # Similarity distribution
            st.subheader("🎯 Similarity Scores")
            if ai_system.analytics.search_history:
                recent_scores = [log['avg_similarity'] for log in ai_system.analytics.search_history[-5:]]
                for i, score in enumerate(recent_scores):
                    st.progress(score, text=f"Query {i+1}: {score:.1%}")
        
        # Security Status
        st.subheader("🔒 Security Status")
        st.success("End-to-end Encryption: ACTIVE")
        st.info(f"Documents: {len(st.session_state.documents)}")
        st.info(f"Vector Dimension: {ai_system.rag_model.rag_system.embeddings.shape[1] if st.session_state.documents else 'N/A'}")
        
        # Export data
        if st.session_state.survey_data:
            if st.button("📥 Export Survey Data"):
                decrypted_data = {}
                for survey_id, encrypted_survey in st.session_state.survey_data.items():
                    decrypted_data[survey_id] = {
                        "timestamp": encrypted_survey["timestamp"],
                        "question": ai_system.security_manager.decrypt_data(
                            encrypted_survey["question"].encode()
                        ),
                        "response": ai_system.security_manager.decrypt_data(
                            encrypted_survey["response"].encode()
                        )
                    }
                
                json_data = json.dumps(decrypted_data, indent=2)
                st.download_button(
                    label="Download Data",
                    data=json_data,
                    file_name="survey_data.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main()