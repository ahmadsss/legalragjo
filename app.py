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
def retrieve_articles(query, limit=15):
    vector = embed_query(query)
    results = client.collections.get("LawArticle").query.near_vector(
        near_vector=vector,
        limit=limit
    )
    # ✅ استبعاد LAW METADATA من النتيجة مبكرًا
    filtered = [obj for obj in results.objects if obj.properties.get("article_title") != "LAW METADATA"]
    return filtered

# 🧠 Generate a legal-style answer using GPT-4
def generate_answer(question, context):
    context_text = "\n\n".join(
        f"المادة {o.properties.get('article_number', '')}: {o.properties.get('article_title', '')}\n{o.properties.get('text', '')}"
        for o in context
    )

    prompt = f"""
    أنت مساعد قانوني محترف. عند الإجابة على أي سؤال قانوني:
- اعتمد فقط على النصوص القانونية المستخرجة من القوانين الأردنية المتوفرة لديك
 - في البداية، اكتب بشكل موجز يوضح ما يطبّق في المسألة
 - تجنّب الشرح المطوّل أو الاستطراد القانوني إلا إذا طلب المستخدم.
 - تجنب ذكر ارقام المواد والقاونين الا اذا طلب منك ذاك
 - فليكن جوابك على شكل paragraph
- أجب بنفس لغة السؤال.

 النصوص القانونية:
 
{context_text}

السؤال: {question}

الإجابة:"""
    completion = openai.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": ""}
        ]
    )
    return completion.choices[0].message.content.strip()

# 🌐 Streamlit Web UI
st.set_page_config(layout="centered", page_title="مساعد قانوني ذكي")
st.markdown("<h1 style='text-align: right; direction: rtl;'>💼 مساعد القانون الأردني</h1>", unsafe_allow_html=True)

question = st.text_input(
    "✍️ اكتب سؤالك القانوني هنا:",
    key="query",
    placeholder="هل الشهادة وحدها تكفي لإثبات حق مالي كبير؟",
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
            <div style='direction: rtl; text-align: right; font-size: 1.15em; line-height: 2.1;
            background-color: #a2d0ff; border-radius: 10px; padding: 18px 16px; margin: 10px 0 18px 0;
            border: 1px solid #006e1a; color: #181c1f;'>
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
            f"<div style='direction: rtl; text-align: right; background-color: #012348; border-radius: 8px; padding: 8px; margin-bottom: 10px;'>{obj.properties.get('text').replace(chr(10), '<br>')}</div>",
                unsafe_allow_html=True
            )
