
import os
from huggingface_hub import login
from datasets import load_dataset
import gradio as gr
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import chromadb
from sentence_transformers import SentenceTransformer


# Charger le token depuis les secrets
hf_token = os.getenv("HF_TOKEN")  # Assurez-vous que 'HF_TOKEN' est bien le nom du secret Hugging Face

# Connecting à Hugging Face
login(hf_token)
# Charger le dataset
dataset = load_dataset("Maryem2025/final_dataset")  # Changez le nom si nécessaire

# Initialisation du modèle Llama
llm = Llama(
    model_path=hf_hub_download(
        repo_id="TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF",
        filename="capybarahermes-2.5-mistral-7b.Q2_K.gguf",
    ),
    n_ctx=2048,
    #n_gpu_layers=50,  # Ajustez selon votre VRAM
)

# Initialisation de ChromaDB Vector Store
class VectorStore:
    def __init__(self, collection_name):
        self.embedding_model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
        self.chroma_client = chromadb.Client()

        # Supprimer la collection existante si elle existe
        if collection_name in self.chroma_client.list_collections():
            self.chroma_client.delete_collection(collection_name)
        
        # Créer une nouvelle collection
        self.collection = self.chroma_client.create_collection(name=collection_name)

    def populate_vectors(self, dataset):
        # Sélectionner les colonnes pertinentes à concaténer
        names = dataset['train']['name'][:200]
        ingredients = dataset['train']['ingredients'][:200]
        instructions = dataset['train']['instructions'][:200]
        
        cuisine = dataset['train']['cuisine'][:200]
        total_time = dataset['train']['total_time'][:200]

        # Concaténer les textes à partir des colonnes sélectionnées
        texts = [
            f"Name: {name}. Ingredients: {ingr}. Instructions: {instr}. Cuisine: {cui}. Total time: {total} minutes."
            for name, ingr, instr, cui, total in zip(names, ingredients, instructions, cuisine, total_time)
                   
        ]

        # Ajouter les embeddings au store de vecteurs
        for i, item in enumerate(texts):
            embeddings = self.embedding_model.encode(item).tolist()
            self.collection.add(embeddings=[embeddings], documents=[item], ids=[str(i)])

    def search_context(self, query, n_results=1):
        query_embedding = self.embedding_model.encode([query]).tolist()
        results = self.collection.query(query_embeddings=query_embedding, n_results=n_results)
        return results['documents']

# Initialisation du store de vecteurs et peuplement
dataset = load_dataset('Maryem2025/dataset-test')
vector_store = VectorStore("embedding_vector")
vector_store.populate_vectors(dataset)

# Fonction pour générer du texte
def generate_text(message, max_tokens, temperature, top_p):
    # Récupérer le contexte depuis le store de vecteurs
    context_results = vector_store.search_context(message, n_results=1)
    context = context_results[0] if context_results else ""

    # Créer le modèle de prompt
    prompt_template = (
        f"SYSTEM: You are a recipe generating bot.\n"
        f"SYSTEM: {context}\n"
        f"USER: {message}\n"
        f"ASSISTANT:\n"
    )

    # Générer le texte avec le modèle de langue
    output = llm(
        prompt_template,
        temperature=0.3,
        top_p=0.95,
        top_k=40,
        repeat_penalty=1.1,
        max_tokens=600,
    )

    # Traiter la sortie
    input_string = output['choices'][0]['text'].strip()
    cleaned_text = input_string.strip("[]'").replace('\\n', '\n')
    continuous_text = '\n'.join(cleaned_text.split('\n'))
    return continuous_text

# Définir l'interface Gradio
demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter your message here...", label="Message"),
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="FALFOUL's Kitchen",
    description="Running LLM with context retrieval from ChromaDB",
    examples=[
        ["I have  rice, what can I make out of it?"],
        ["I just have some milk and chocolate, what dessert can I make?"],
       
        ["Can you suggest a vegan breakfast recipe?"],
        ["How do I make a perfect scrambled egg?"],
        ["Can you guide me through making a tajine?"],
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch()
