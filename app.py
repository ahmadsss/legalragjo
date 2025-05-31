# app.py
import streamlit as st
import openai
import weaviate
from weaviate.classes.init import Auth
from dotenv import load_dotenv
import os

# Load credentials
load_dotenv()
openai.api_key = "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"
WEAVIATE_API_KEY = "MUV1leWNYVHCSrkNGNbG43eflSiDzXBpl4Sf"
WEAVIATE_URL = "kvpz5nvzqewg1rtv2qwjsg.c0.europe-west3.gcp.weaviate.cloud"

# Connect to Weaviate Cloud
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,
    auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
)

# Embed query
def embed_query(text):
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-3-large"
    )
    return response.data[0].embedding

# Run semantic search
def retrieve_articles(query, limit=5):
    vector = embed_query(query)
    results = client.collections.get("legalragjo").query.near_vector(
        near_vector=vector,
        limit=limit
    )
    return results.objects

# Generate answer
def generate_answer(question, context):
    context_text = "\n\n".join(
        f"المادة {o.properties.get('article_number', '')}: {o.properties.get('article_title', '')}\n{o.properties.get('text', '')}"
        for o in context
    )
    prompt = f"""أنت مساعد قانوني ذكي. استنادًا إلى المواد التالية من القانون، أجب على السؤال التالي باللغة العربية الفصحى:

{context_text}

السؤال: {question}

الإجابة:"""
    completion = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "أجب فقط بناءً على النصوص القانونية المتاحة."},
                  {"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content.strip()

# Streamlit UI
st.set_page_config(layout="centered", page_title="مساعد قانوني ذكي")
st.markdown("<h1 style='text-align: right; direction: rtl;'>💼 مساعد القانون الأردني</h1>", unsafe_allow_html=True)

question = st.text_input("✍️ اكتب سؤالك القانوني هنا:", key="query", placeholder="ما التعديل الذي جرى على المادة 8؟")

if question:
    with st.spinner("🔍 يتم البحث في النصوص القانونية..."):
        articles = retrieve_articles(question)
    if not articles:
        st.error("لم يتم العثور على مواد قانونية مناسبة لهذا السؤال.")
    else:
        with st.spinner("🤖 يتم توليد الإجابة..."):
            answer = generate_answer(question, articles)
        st.markdown("### 🧠 الإجابة", unsafe_allow_html=True)
        st.success(answer)

        with st.expander("📜 عرض المواد القانونية المسترجعة"):
            for obj in articles:
                st.markdown(f"**المادة {obj.properties.get('article_number')}** - {obj.properties.get('article_title')}", unsafe_allow_html=True)
                st.text(obj.properties.get("text"))
