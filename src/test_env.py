from langdetect import detect
from sentence_transformers import SentenceTransformer

def test_langdetect():
    queries = [
        "我的 fridge 噪音大，保修怎么走？",
        "How do I request a return for my broken washer?",
        "Mon frigo fait du bruit, comment marche la garantie ?",
        "¿Cómo solicito una devolución?"
    ]
    print("--- Language Detection Test ---")
    for q in queries:
        try:
            lang = detect(q)
            mapping = {'zh-cn': '中文', 'zh-tw': '中文', 'ko': 'Korean', 'ja': 'Japanese', 'en': 'English', 'fr': 'Français', 'es': 'Español'}
            print(f"Query: {q} \nDetected: {lang} ({mapping.get(lang, lang)})")
        except Exception as e:
            print(f"Error detecting {q}: {e}")

def test_embeddings():
    print("\n--- Embedding Model Test (bge-m3) ---")
    try:
        model = SentenceTransformer('BAAI/bge-m3')
        sentences = ["This is a test sentence.", "这是一个测试句子。"]
        embeddings = model.encode(sentences)
        print(f"Successfully loaded model and generated embeddings of shape: {embeddings.shape}")
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    test_langdetect()
    # Commenting out embeddings test for now to avoid huge download in background if unsupervised
    # But I should probably test it. I'll uncomment and run.
    test_embeddings()
