import streamlit as st
import openai
import weaviate
from dotenv import load_dotenv
import os
from weaviate.classes.init import Auth

# Load .env (optional for local dev)
load_dotenv()

# Load secrets from Streamlit Cloud
openai.api_key = st.secrets["OPENAI_API_KEY"]
WEAVIATE_API_KEY = st.secrets["WEAVIATE_API_KEY"]
WEAVIATE_URL = st.secrets["WEAVIATE_URL"]

# Connect to Weaviate Cloud
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,
    auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
)

# 🔎 Embed query using OpenAI
def embed_query(text):
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-3-large"
    )
    return response.data[0].embedding

# 🔍 Semantic retrieval from Weaviate
def retrieve_articles(query, limit=5):
    vector = embed_query(query)
    results = client.collections.get("LawArticle").query.near_vector(
        near_vector=vector,
        limit=limit
    )
    return results.objects

# 🧠 Generate a legal-style answer using GPT-4
def generate_answer(question, context):
    context_text = "\n\n".join(
        f"المادة {o.properties.get('article_number', '')}: {o.properties.get('article_title', '')}\n{o.properties.get('text', '')}"
        for o in context
    )

    prompt = f"""You are a legal expert and consultant. For every legal question:

    Base your answer only on the content of the retrieved chunks from the vector database. Do not answer from general knowledge or pre-training unless explicitly instructed.

    For each legal point you provide, cite or quote the relevant retrieved chunk (article, clause, or paragraph) that supports your answer.

    Structure your answer as a numbered list of clear legal situations or rights, each point referencing the supporting chunk.

    If multiple chunks address the same issue (e.g., general and specific provisions), present them together, clarifying their relationship.

    Always mention any legal steps required before action (e.g., giving notice, going to court) if stated in the retrieved chunks.

    After the main list, briefly explain why these chunks are relevant to the question, referencing their position (general rule, special rule, etc.).

    End every answer with: "هذه المعلومات للاستدلال فقط وليست استشارة قانونية رسمية."

    Use the tone, depth, and legal structure of an expert consultant, aiming for the detail and clarity seen in the best large language model (LLM) responses.

    Never miss a general legal rule from the retrieved chunks that may apply, even if a special rule exists.
    Answer according to the retrieved chunks. Do not rely on your own legal knowledge. Structure your answer by legal points, and cite the chunk or article for every claim. Use professional legal language and reasoning.
 النصوص القانونية:
 
{context_text}

السؤال: {question}

الإجابة:"""
    completion = openai.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "أجب فقط بناءً على النصوص القانونية المعروضة."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content.strip()

# 🌐 Streamlit Web UI
st.set_page_config(layout="centered", page_title="مساعد قانوني ذكي")
st.markdown("<h1 style='text-align: right; direction: rtl;'>💼 مساعد القانون الأردني</h1>", unsafe_allow_html=True)

question = st.text_input(
    "✍️ اكتب سؤالك القانوني هنا:",
    key="query",
    placeholder="ما التعديل الذي جرى على المادة 8؟",
    help="اكتب سؤالك بالعربية",
)

if question:
    with st.spinner("🔍 يتم البحث في النصوص القانونية..."):
        articles = retrieve_articles(question)

    if not articles:
        st.error("لم يتم العثور على مواد قانونية مناسبة لهذا السؤال.")
    else:
        with st.spinner("🤖 يتم توليد الإجابة..."):
            answer = generate_answer(question, articles)

        st.markdown("### 🧠 الإجابة", unsafe_allow_html=True)
        # RTL-formatted answer, preserving paragraph breaks and Arabic quoting
        st.markdown(
            f"""
            <div style='direction: rtl; text-align: right; font-size: 1.15em; line-height: 2.1;'>
            {answer.replace(chr(10), '<br>')}
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander("📜 عرض المواد القانونية المسترجعة"):
            for obj in articles:
                st.markdown(
                    f"<div style='direction: rtl; text-align: right;'><b>المادة {obj.properties.get('article_number')}</b> - {obj.properties.get('article_title')}</div>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"<div style='direction: rtl; text-align: right; background-color: #f8f9fa; border-radius: 8px; padding: 8px; margin-bottom: 10px;'>{obj.properties.get('text').replace(chr(10), '<br>')}</div>",
                    unsafe_allow_html=True
                )
