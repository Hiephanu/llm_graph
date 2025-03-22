import os
import google.generativeai as genai
from neo4j import GraphDatabase
from dotenv import load_dotenv
import json

# Load API key từ file .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Cấu hình Gemini
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash")

# Kết nối Neo4j
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "doanvanhiep"
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

domains = [
    "Science", "Healthcare", "Social Sciences", "E-commerce", "Information Retrieval",
    "Finance", "Education", "Entertainment", "Government", "Agriculture", "Manufacturing"
]

# Hàm truy vấn Neo4j
def query_neo4j(tx, entity_name):
    query = """
    MATCH (e:Entity {name: $entity_name})
    OPTIONAL MATCH (e)-[r:RELATES_TO]->(related:Entity)
    RETURN e.name AS name, e.label AS label, r.type AS relation, 
           related.name AS related_name, related.label AS related_label
    """
    result = tx.run(query, entity_name=entity_name)
    return [dict(record) for record in result]

# Bước 1: Query Processing and Retrieval
def process_query(query_text):
    print(f"\n🔹 **Processing Query:** {query_text}\n")
    
    # Phân tích truy vấn bằng Gemini
    prompt = f"""
    Analyze the following query:
    "{query_text}"
    
    1. Identify key entities (e.g., nouns or subjects).
    2. Identify relationships (e.g., verbs or actions).
    3. Suggest a domain from the list: {domains}
    
    Format the output as JSON:
    {{
      "entities": ["entity1", "entity2", ...],
      "relationships": ["relation1", "relation2", ...],
      "domain": "Domain"
    }}
    """
    response = model.generate_content(prompt)
    result = response.text.strip()
    
    # Làm sạch JSON
    if result.startswith("```json"):
        result = result.replace("```json", "", 1).strip()
    if result.endswith("```"):
        result = result.rsplit("```", 1)[0].strip()
    
    try:
        data = json.loads(result)
        entities = data.get("entities", [])
        relationships = data.get("relationships", [])
        domain = data.get("domain", "Unknown")
        
        # Truy vấn Neo4j cho từng entity
        kg_results = []
        with driver.session() as session:
            for entity in entities:
                result = session.execute_read(query_neo4j, entity)
                if result:
                    kg_results.extend(result)
        
        return {
            "entities": entities,
            "relationships": relationships,
            "domain": domain,
            "kg_results": kg_results
        }
    except json.JSONDecodeError as e:
        print(f"⚠️ Error parsing JSON: {e}")
        print(f"Raw output: {repr(result)}")
        return None

# Bước 2: Query-Focused Summarization
def summarize_results(query_text, analysis_result):
    if not analysis_result:
        return None
    
    kg_results = analysis_result["kg_results"]
    if not kg_results:
        return "No relevant data found in the knowledge graph."
    
    # Tạo prompt để tinh chỉnh thông tin
    prompt = f"""
    Given the query: "{query_text}"
    And the following data from a knowledge graph:
    {json.dumps(kg_results, indent=2)}
    
    Summarize the information to directly answer the query, removing irrelevant details.
    Return a concise summary as a string.
    """
    response = model.generate_content(prompt)
    summary = response.text.strip()
    
    return summary

# Bước 3: Generation of Global Answer with LLMs
def generate_global_answer(query_text, summary):
    if not summary:
        return "Sorry, I couldn't find enough information to answer your query."
    
    # Tạo prompt để sinh câu trả lời tự nhiên
    prompt = f"""
    Based on the query: "{query_text}"
    And the summarized information: "{summary}"
    
    Generate a natural, coherent, and contextually appropriate response.
    """
    response = model.generate_content(prompt)
    global_answer = response.text.strip()
    
    print("\n📝 Global Answer:")
    print(global_answer)
    return global_answer

# Hàm chính xử lý truy vấn
def handle_query(query_text):
    # Bước 1: Phân tích và truy xuất
    analysis_result = process_query(query_text)
    if not analysis_result:
        return "Error processing query."
    
    # Bước 2: Tinh chỉnh kết quả
    summary = summarize_results(query_text, analysis_result)
    
    # Bước 3: Tạo câu trả lời cuối cùng
    global_answer = generate_global_answer(query_text, summary)
    return global_answer

# Ví dụ sử dụng
query = "Talk about something about Hiep? Hiep can code in languages specific?"
answer = handle_query(query)

# Đóng kết nối Neo4j
driver.close()