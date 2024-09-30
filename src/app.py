from flask import Flask, request, jsonify
import logging
import jsonschema
from flask_cors import CORS

app = Flask(__name__)

# if os.environ.get('FLASK_ENV') == 'development':
cors = CORS(app, resources={r"/ask": {"origins": "*"}})



# Define the JSON schema for the request payload
schema = {
    "type": "object",
    "properties": {
        "question": {"type": "string", "minLength": 1}
    },
    "required": ["question"]
}

# Create embeddings
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Create vector store
from langchain_community.vectorstores import FAISS
vector_store = FAISS.load_local("../notebooks/faiss_sberquad_index", embeddings, allow_dangerous_deserialization=True)

# Create retriever
retriever = vector_store.as_retriever()

# Create LLM
from langchain_ollama import OllamaLLM
llm = OllamaLLM(model="llama3.1:8b")

# Create prompt
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Вы являетесь ассистентом для задач вопрос-ответ. Используйте следующие извлеченные данные в качестве контекста для ответа на вопрос. Если вы не знаете ответа, скажите, что не знаете. Используйте не более трех предложений и старайтесь отвечать кратко.\n\n{context}"),
        ("human", "{input}"),
    ]
)

# Create chains
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        # Validate the request payload against the schema
        data = request.json
        jsonschema.validate(instance=data, schema=schema)

        # Process the question
        question = data['question']
        response = rag_chain.invoke({"input": question})
        return jsonify({"answer":response["answer"]})
    except jsonschema.ValidationError as e:
        return jsonify({"error": "Invalid request payload: " + e.message}), 400
    except Exception as e:
        logging.error(f"Error: {e}")
        return jsonify({"error": "Failed to process question"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)