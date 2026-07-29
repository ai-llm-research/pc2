from wiki_search import process_single_country_or_content
from languages import language_dict
from langchain_openai import OpenAIEmbeddings
import os

os.makedirs("data", exist_ok=True)
# Set your key in the environment before running:  export OPENAI_API_KEY=sk-...
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("Error: OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY=<your key>")
embedding = OpenAIEmbeddings(openai_api_key=api_key, model="text-embedding-3-large")

# Pre-compute country common knowledge embeddings
country_common_knowledge_embedding_list = {}
countries = language_dict.keys()
for country in countries:
    print(country)
    country_common_knowledge = process_single_country_or_content(country)["paragraphs"]
    country_common_knowledge_embedding_list[country] = embedding.embed_documents(country_common_knowledge)

with open("data/country_common_knowledge_embedding_list.pkl", "wb") as f:
    import pickle
    pickle.dump(country_common_knowledge_embedding_list, f)

# Pre-compute country conflict embeddings
country_conflict_embedding_list = {}
countries = language_dict.keys()
for country in countries:
    print(country)
    country_conflict_embedding_list[country] = embedding.embed_query("conflict with " + country)
with open("data/country_conflict_embedding_list.pkl", "wb") as f:
    import pickle
    pickle.dump(country_conflict_embedding_list, f)