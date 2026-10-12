from sentence_transformers import SentenceTransformer, util
import chromadb

# Takes a user question and a list of retrieved document chunks (simulated)

client = chromadb.Client()
collection = client.get_or_create_collection("example_docs")

documents = [
    "A relational database stores information in tables made up of rows and columns. Each table represents an entity, such as customers or orders, and each row represents one record.",
    "A primary key uniquely identifies each row in a database table. Primary keys help prevent duplicate records and let other tables refer to a specific record reliably.",
    "A foreign key links a record in one table to a record in another table. This relationship helps maintain referential integrity, such as ensuring every order belongs to an existing customer.",
    "SQL is the standard language used to define, query, and modify relational data. Commands such as SELECT, INSERT, UPDATE, and DELETE cover common data operations.",
    "The SELECT statement retrieves data from one or more tables. WHERE filters the rows returned, while ORDER BY sorts the results by one or more columns.",
    "SQL joins combine related rows from multiple tables using matching columns. An INNER JOIN returns matching rows, while a LEFT JOIN also keeps every row from the left table.",
    "Database normalization organizes data to reduce unnecessary duplication and update errors. Splitting related facts into linked tables can make data more consistent and easier to maintain.",
    "An index is a database structure that can speed up searches and joins on selected columns. Indexes use extra storage and can make writes slower, so they should be chosen for common query patterns.",
    "A REST API exposes resources through predictable URLs and standard HTTP methods. Clients use methods such as GET, POST, PUT, PATCH, and DELETE to read or change resource state.",
    "HTTP status codes communicate the outcome of an API request. For example, 200 indicates success, 201 indicates a resource was created, and 404 means the requested resource was not found.",
    "JSON is a common format for exchanging structured data between REST API clients and servers. Objects and arrays allow responses to represent records and collections in a readable way.",
    "HTTP headers carry information about a request or response, such as the content type or authorization credentials. The Content-Type header tells the server how to interpret a request body.",
    "API endpoints should validate incoming parameters and request bodies before processing them. Clear validation errors help clients correct malformed or incomplete requests.",
    "Pagination divides a large API result set into smaller portions. Page numbers, offsets, or cursors let clients retrieve results without transferring an entire collection at once.",
    "FastAPI uses Python type hints to define request data and validate it automatically. Typed models also help generate clear API documentation.",
    "Pydantic models describe the structure and constraints of data accepted or returned by a FastAPI application. FastAPI can use these models to parse request bodies and serialize responses.",
    "Path parameters identify a resource within an endpoint URL, while query parameters often control filtering or pagination. FastAPI lets developers declare both directly in route function signatures.",
    "FastAPI can generate interactive API documentation from route definitions and data models. The OpenAPI schema supports tools such as Swagger UI for trying endpoints in a browser.",
    "Streamlit turns Python scripts into interactive web applications with widgets such as buttons, sliders, and text inputs. A widget interaction reruns the script so the displayed output reflects the latest input.",
    "Streamlit's session state preserves values across script reruns for a user's session. It is useful for retaining selections, counters, or intermediate results as users interact with an app.",
]

sources = [
    "relational_db.md",
    "primary_keys.md",
    "foreign_keys.md",
    "sql_basics.md",
    "select_queries.md",
    "sql_joins.md",
    "database_normalization.md",
    "database_indexes.md",
    "rest_apis.md",
    "http_status_codes.md",
    "json_data_format.md",
    "http_headers.md",
    "api_validation.md",
    "api_pagination.md",
    "fastapi_basics.md",
    "pydantic_models.md",
    "fastapi_path_and_query_parameters.md",
    "fastapi_openapi_docs.md",
    "streamlit_basics.md",
    "streamlit_session_state.md",
]

collection.add(
    documents=documents,
    metadatas=[{"source": s} for s in sources],
    ids=[f"doc_{i}" for i in range(len(documents))],
)
print(f"Knowledge base loaded: {collection.count()} document chunks\n")

system_prompt = (
    "You are a helpful AI assistant for students learning AI engineering. "
    "Answer the user's question based ONLY on the context provided below. "
    "If the context doesn't contain enough information to answer, say so. "
    "Always cite which source document your answer comes from. "
    "Keep your response under 150 words."
)


def build_prompt(user_question):
    print("=" * 60)
    print("Step 1: User Question")
    print("=" * 60)
    print(f"{user_question}\n")

    results = collection.query(query_texts=[user_question], n_results=5)
    print(f"Found {len(results['documents'][0])} relevant chunks:\n")
    for i in range(len(results["documents"][0])):
        doc = results["documents"][0][i]
        source = results["metadatas"][0][i]["source"]
        distance = results["distances"][0][i]
        print(f"  [{i+1}] (dist: {distance:.4f}) [{source}]")
        print(f"      {doc[:80]}...\n")

    print("=" * 60)
    print("STEP 2: Build the Prompt")
    print("=" * 60)

    # Retrieved chunks formatted with source labels
    context = "\n\n".join(
        [
            f"[Source: {results['metadatas'][0][i]['source']}]\n{results['documents'][0][i]}"
            for i in range(len(results["documents"][0]))
        ]
    )

    full_prompt = f"""SYSTEM: {system_prompt}

CONTEXT: {context}

USER QUESTION: {user_question}

ANSWER:
"""

    # Token estimate: characters / 4 as a rough approximation
    print(
        f"Prompt length: {len(full_prompt)} characters (Roughly {len(full_prompt) / 4:.0f} tokens)"
    )
    print("\n--- Full Prompt (what gets sent to LLM) ---")
    print(full_prompt[:500] + "\n....\n")
    return full_prompt


# Tests
test_queries = [
    "What are the primary functions of SQL in coding?",
    "What are the benefits of using Streamlit for coding AI powered applications?",
]

for query in test_queries:
    build_prompt(query)
