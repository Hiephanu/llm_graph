import os
import google.generativeai as genai
import spacy
from neo4j import GraphDatabase
from dotenv import load_dotenv
import json

# Load API key từ file .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Cấu hình Gemini
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash")

# Load mô hình NLP
nlp = spacy.load("en_core_web_lg")

# Kết nối Neo4j
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "doanvanhiep"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Danh sách domain
domains = [
    "Science",
    "Healthcare",
    "Social Sciences",
    "E-commerce",
    "Information Retrieval",
    "Finance",
    "Education",
    "Entertainment",
    "Government",
    "Agriculture",
    "Manufacturing"
]

# Hàm lưu thực thể và quan hệ vào Neo4j với label
def save_to_neo4j(tx, entity1, label1, relation, entity2, label2):
    query = """
    MERGE (e1:Entity {name: $entity1})
    SET e1.label = $label1
    MERGE (e2:Entity {name: $entity2})
    SET e2.label = $label2
    MERGE (e1)-[r:RELATES_TO {type: $relation}]->(e2)
    RETURN e1, r, e2
    """
    tx.run(query, entity1=entity1, label1=label1, relation=relation, entity2=entity2, label2=label2)

# Văn bản đầu vào
with open("dataset/datasource.txt", "r", encoding="utf-8") as file:
    text = file.read().strip()

# Tách văn bản thành từng câu (chunks)
doc = nlp(text)
chunks = [sent.text.strip() for sent in doc.sents]

# Duyệt từng chunk để trích xuất thực thể, quan hệ và gắn nhãn
for i, chunk in enumerate(chunks):
    print(f"\n🔹 **Processing Chunk {i+1}:** {chunk}\n")

    prompt = f"""
    Extract entities and their relationships from the following text:
    "{chunk}"
    
    Additionally, assign a label to each entity from the following list of domains:
    {domains}
    
    Format the output as JSON:
    {{
      "entities": [
        {{"name": "Entity1", "label": "Domain"}},
        {{"name": "Entity2", "label": "Domain"}}
      ],
      "relationships": [
        {{"source": "Entity1", "relation": "RELATION_TYPE", "target": "Entity2"}}
      ]
    }}
    """

    response = model.generate_content(prompt)
    result = response.text
    result = response.text.strip()

    if result.startswith("```json"):
        result = result.replace("```json", "", 1).strip()
    if result.endswith("```"):
        result = result.rsplit("```", 1)[0].strip()

    try:
        data = json.loads(result)
        entities = {e["name"]: e["label"] for e in data.get("entities", [])}
        relationships = data.get("relationships", [])

        with driver.session() as session:
            for rel in relationships:
                entity1 = rel["source"]
                entity2 = rel["target"]
                label1 = entities.get(entity1, "Unknown")  # Default label nếu không có
                label2 = entities.get(entity2, "Unknown")
                session.execute_write(
                    save_to_neo4j, entity1, label1, rel["relation"], entity2, label2
                )

    except json.JSONDecodeError as e:
        print(f"⚠️ Lỗi khi parse JSON từ Gemini: {e}")

# Đóng kết nối Neo4j
driver.close()

print("✅ Dữ liệu đã được lưu vào Neo4j thành công!")