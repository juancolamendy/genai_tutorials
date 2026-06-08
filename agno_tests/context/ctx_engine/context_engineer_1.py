import os
from typing import List, Dict, Any

import lancedb

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.anthropic import Claude
from openai import OpenAI as OpenAIClient

from rich.pretty import pprint

# Initialize OpenAI client for embeddings
openai_client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))

def get_embeddings_batch(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """Generate embeddings for a batch of texts"""
    response = openai_client.embeddings.create(
        input=texts,
        model=model
    )
    return [item.embedding for item in response.data]

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# ==========================================
# 1. DEFINE SEMANTIC BLUEPRINTS
# ==========================================

context_blueprints = [
    {
        "id": "suspenseful_narrative",
        "description": "Creative writing with suspenseful narrative style, dramatic tension, and vivid imagery",
        "blueprint": {
            "tone": "mysterious and suspenseful",
            "style": "creative narrative",
            "techniques": [
                "Use cliffhangers and dramatic pauses",
                "Build tension through pacing",
                "Include vivid sensory details",
                "Create atmospheric descriptions"
            ],
            "structure": "Start with a hook, build tension gradually, end with revelation or twist",
            "vocabulary": "Evocative, descriptive, dramatic language"
        }
    },
    {
        "id": "technical_explanation",
        "description": "Technical and factual explanations with precise terminology and logical structure",
        "blueprint": {
            "tone": "professional and authoritative",
            "style": "technical documentation",
            "techniques": [
                "Define terms clearly before use",
                "Use numbered steps for processes",
                "Include relevant technical specifications",
                "Reference standards and best practices"
            ],
            "structure": "Overview → detailed explanation → summary of key points",
            "vocabulary": "Domain-specific terminology, precise and unambiguous"
        }
    },
    {
        "id": "casual_summary",
        "description": "Casual and friendly summaries for general audiences with conversational tone",
        "blueprint": {
            "tone": "friendly and approachable",
            "style": "conversational summary",
            "techniques": [
                "Use simple language and analogies",
                "Break down complex concepts",
                "Include relatable examples",
                "Keep paragraphs short"
            ],
            "structure": "Main point → supporting details → practical takeaway",
            "vocabulary": "Everyday language, minimal jargon"
        }
    }
]

# ==========================================
# 2. SETUP LANCEDB DATABASES
# ==========================================

# Connect to LanceDB (creates local directory if doesn't exist)
db = lancedb.connect("./context_enginer_1.db")

# ==========================================
# 3. POPULATE CONTEXT LIBRARY
# ==========================================

print("Building Context Library...")

# Prepare context library data
context_data = []
context_descriptions = [bp["description"] for bp in context_blueprints]
context_embeddings = get_embeddings_batch(context_descriptions)

for i, blueprint in enumerate(context_blueprints):
    context_data.append({
        "id": blueprint["id"],
        "description": blueprint["description"],
        "blueprint": str(blueprint["blueprint"]),  # Convert dict to string for storage
        "vector": context_embeddings[i]
    })

# Create context library table
try:
    db.drop_table("context_library")
except:
    pass

context_table = db.create_table("context_library", data=context_data)
print(f"Context Library created with {len(context_data)} blueprints")

# ==========================================
# 4. POPULATE KNOWLEDGE BASE
# ==========================================

print("\nBuilding Knowledge Base...")

# Sample knowledge content
raw_knowledge = """
Artificial Intelligence (AI) refers to computer systems capable of performing tasks that typically require human intelligence. 
Machine learning is a subset of AI that enables systems to learn from data without explicit programming. 
Deep learning uses neural networks with multiple layers to process complex patterns in large datasets.
Natural Language Processing (NLP) allows computers to understand, interpret, and generate human language.
Computer vision enables machines to interpret and understand visual information from the world.
Reinforcement learning trains agents to make decisions by rewarding desired behaviors.
Transfer learning allows models trained on one task to be adapted for related tasks.
The transformer architecture revolutionized NLP by introducing attention mechanisms for processing sequences.
Large Language Models (LLMs) like GPT use billions of parameters to generate human-like text.
Vector databases store embeddings for efficient similarity search in AI applications.
"""

# Chunk the knowledge
knowledge_chunks = chunk_text(raw_knowledge, chunk_size=200, overlap=30)

# Process in batches
batch_size = 100
knowledge_data = []

for i in range(0, len(knowledge_chunks), batch_size):
    batch = knowledge_chunks[i:i + batch_size]
    embeddings = get_embeddings_batch(batch)
    
    for j, chunk in enumerate(batch):
        knowledge_data.append({
            "id": f"chunk_{i+j}",
            "text": chunk,
            "vector": embeddings[j]
        })

# Create knowledge base table
try:
    db.drop_table("knowledge_base")
except:
    pass

knowledge_table = db.create_table("knowledge_base", data=knowledge_data)
print(f"Knowledge Base created with {len(knowledge_data)} chunks")

# ==========================================
# 5. DEFINE LIBRARIAN AGENT
# ==========================================

def search_context_library(style_request: str) -> Dict[str, Any]:
    """Search for the best matching semantic blueprint"""
    query_embedding = get_embeddings_batch([style_request])[0]
    
    results = context_table.search(query_embedding).limit(1).to_pandas()
    
    if len(results) > 0:
        return {
            "id": results.iloc[0]["id"],
            "description": results.iloc[0]["description"],
            "blueprint": results.iloc[0]["blueprint"],
            "distance": results.iloc[0]["_distance"]
        }
    return None

def search_knowledge_base(query: str, limit: int = 3) -> List[Dict[str, Any]]:
    """Search knowledge base for relevant information"""
    query_embedding = get_embeddings_batch([query])[0]
    
    results = knowledge_table.search(query_embedding).limit(limit).to_pandas()
    
    return [
        {
            "text": row["text"],
            "distance": row["_distance"]
        }
        for _, row in results.iterrows()
    ]

# ==========================================
# 6. CREATE ORCHESTRATOR AGENT
# ==========================================

class ContextAwareOrchestrator:
    """Orchestrator that uses context engineering for multi-agent systems"""
    
    def __init__(self, model_provider="openai"):
        if model_provider == "openai":
            self.model = OpenAIChat(id="gpt-4o")
        else:
            self.model = Claude(id="claude-sonnet-4-20250514")
        
    def _write_response(self, context_blueprint, retrieved_knowledge, user_query):
        system_prompt = f"""You are an AI assistant using a semantic blueprint to guide your response style.

STYLE BLUEPRINT:
{context_blueprint['blueprint']}

KNOWLEDGE CONTEXT:
{retrieved_knowledge}

Generate a response following the style blueprint exactly, using only information from the knowledge context provided."""

        # Step 4: Create agent with contextual instructions
        agent = Agent(
            model=self.model,
            description=system_prompt,
            markdown=True
        )
        
        # Step 5: Generate response
        print(f"\n💬 Generating response...\n")
        response = agent.run(user_query)
        return response

    def generate_response(self, user_query: str, style_preference: str):
        """Generate a response using context library and knowledge base"""
        
        # Step 1: Librarian searches for appropriate style
        print(f"\n🔍 Searching context library for: '{style_preference}'")
        context_blueprint = search_context_library(style_preference)
        
        if context_blueprint:
            print(f"✅ Found blueprint: {context_blueprint['id']}")
            print(f"   Description: {context_blueprint['description']}")
        else:
            print("❌ No matching blueprint found")
            return
        
        # Step 2: Search knowledge base for relevant information
        print(f"\n🔍 Searching knowledge base for: '{user_query}'")
        knowledge_results = search_knowledge_base(user_query)
        
        print(f"✅ Found {len(knowledge_results)} relevant chunks")
        
        # Step 3: Construct context-aware prompt
        retrieved_knowledge = "\n\n".join([
            f"[Source {i+1}]: {result['text']}" 
            for i, result in enumerate(knowledge_results)
        ])

        # Step4: Write the response
        response = self._write_response(context_blueprint, retrieved_knowledge, user_query)
        pprint(response.content)
        
# ==========================================
# 7. EXAMPLE USAGE
# ==========================================

orchestrator = ContextAwareOrchestrator(model_provider="anthropic")

# Test different style requests
print("\n" + "="*60)
print("EXAMPLE 1: Technical Style")
print("="*60)
orchestrator.generate_response(
    user_query="Explain what machine learning is",
    style_preference="technical explanation with precise terminology"
)

print("\n" + "="*60)
print("EXAMPLE 2: Casual Style")
print("="*60)
orchestrator.generate_response(
    user_query="What is deep learning?",
    style_preference="friendly casual summary for beginners"
)

print("\n" + "="*60)
print("EXAMPLE 3: Suspenseful Style")
print("="*60)
orchestrator.generate_response(
    user_query="Tell me about artificial intelligence",
    style_preference="creative suspenseful narrative"
)
