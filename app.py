import streamlit as st
import openai
import weaviate
from dotenv import load_dotenv
import os
import re
from weaviate.classes.init import Auth

# ✅ Load .env (for local development)
load_dotenv()

# ✅ Load secrets from Streamlit Cloud or .env
openai.api_key = st.secrets["OPENAI_API_KEY"]
WEAVIATE_API_KEY = st.secrets["WEAVIATE_API_KEY"]
WEAVIATE_URL = st.secrets["WEAVIATE_URL"]

# ✅ Connect to Weaviate Cloud
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
    # ✅ Remove metadata chunk
    return [obj for obj in results.objects if obj.properties.get("article_title") != "LAW METADATA"]

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
- تجنّب الشرح المطوّل أو الاستطراد القانوني إلا إذا طلب المستخدم
- تجنب ذكر أرقام المواد والقوانين إلا إذا طلب منك ذلك
- فليكن جوابك على شكل فقرة واحدة واضحة
- أجب بنفس لغة السؤال
- تجاهل كل ماسبق اذا طلب منك التفصيل وليكن جوابك استشارة قانونية
النصوص القانونية:

{context_text}

السؤال: {question}

الإجابة:
"""
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
    placeholder="ما هي مدة التقادم في الدعاوى المدنية، ومتى يبدأ سريانها؟",
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
                law_title = obj.properties.get("law_title", "قانون غير معروف")
                article_number = obj.properties.get("article_number", "")
                article_title = obj.properties.get("article_title", "")
                article_body = obj.properties.get("text", "")

                # 🧼 Remove "المادة N" header from body if present
                cleaned_body = re.sub(rf"^المادة\s+{article_number}\s*","", article_body).strip()

                # 🧷 Combined heading inside blue box
                full_heading = f"{law_title} - المادة {article_number}: {article_title}"

                st.markdown(
                    f"""
                    <div style='direction: rtl; text-align: right; background-color: #012348; color: white;
                    border-radius: 8px; padding: 10px; margin-bottom: 12px; font-size: 1.05em; line-height: 1.9;'>
                    <strong>{full_heading}</strong><br><br>
                    {cleaned_body.replace(chr(10), '<br>')}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
