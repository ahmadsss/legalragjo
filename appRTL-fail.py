import streamlit as st
import openai
import weaviate
from dotenv import load_dotenv
import os
from weaviate.classes.init import Auth
import streamlit.components.v1 as components

# Load .env (for local use)
load_dotenv()

# Load API keys from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]
WEAVIATE_API_KEY = st.secrets["WEAVIATE_API_KEY"]
WEAVIATE_URL = st.secrets["WEAVIATE_URL"]

# Connect to Weaviate Cloud
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,
    auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
)

# Embedding with OpenAI
def embed_query(text):
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-3-large"
    )
    return response.data[0].embedding

# Retrieve relevant articles
def retrieve_articles(query, limit=15):
    vector = embed_query(query)
    results = client.collections.get("LawArticle").query.near_vector(
        near_vector=vector,
        limit=limit
    )
    # Remove metadata chunk
    filtered = [obj for obj in results.objects if obj.properties.get("article_title") != "LAW METADATA"]
    return filtered

# Generate answer using GPT-4.1
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
- تجنب ذكر أرقام المواد والقوانين إلا إذا طلب منك ذلك.
- فليكن جوابك على شكل فقرة متماسكة.
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

# Streamlit UI
st.set_page_config(layout="centered", page_title="مساعد قانوني ذكي")
st.markdown("<h1 style='text-align: right; direction: rtl;'>💼 مساعد القانون الأردني</h1>", unsafe_allow_html=True)

# RTL input field label
st.markdown("<div style='text-align: right; direction: rtl;'>✍️ اكتب سؤالك القانوني هنا:</div>", unsafe_allow_html=True)
question = st.text_input(
    label="",
    key="query",
    placeholder="هل الشهادة وحدها تكفي لإثبات حق مالي كبير؟",
    help="اكتب سؤالك بالعربية"
)

if question:
    with st.spinner("🔍 يتم البحث في النصوص القانونية..."):
        articles = retrieve_articles(question)

    if not articles:
        st.markdown("<div style='direction: rtl; text-align: right; color: red;'>لم يتم العثور على مواد قانونية مناسبة لهذا السؤال.</div>", unsafe_allow_html=True)
    else:
        with st.spinner("🤖 يتم توليد الإجابة..."):
            answer = generate_answer(question, articles)

        # ✅ عرض الإجابة
        st.markdown("<h3 style='text-align: right; direction: rtl;'>🧠 الإجابة</h3>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div id="answer-box" style='direction: rtl; text-align: right; font-size: 1.15em; line-height: 2.1;
            background-color: #a2d0ff; border-radius: 10px; padding: 18px 16px; margin: 10px 0 18px 0;
            border: 1px solid #006e1a; color: #181c1f;'>
            {answer.replace(chr(10), '<br>')}
            </div>
            """,
            unsafe_allow_html=True
        )

        # 📋 زر نسخ الإجابة
        copy_code = f"""
        <script>
        function copyAnswer() {{
            const text = `{answer.replace("`", "\\`")}`;
            navigator.clipboard.writeText(text).then(function() {{
                alert("✅ تم نسخ الإجابة إلى الحافظة");
            }});
        }}
        </script>
        <div style='direction: rtl; text-align: right; margin-top: -10px;'>
            <button onclick="copyAnswer()" style="
                background-color: #006e1a;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                margin-bottom: 10px;
                cursor: pointer;
                font-size: 1em;">
                📋 نسخ الإجابة
            </button>
        </div>
        """
        components.html(copy_code, height=100)

        # عرض المواد المسترجعة
        with st.expander("📜 عرض المواد القانونية المسترجعة"):
            st.markdown("<div style='direction: rtl; text-align: right;'>", unsafe_allow_html=True)
            for obj in articles:
                st.markdown(
                    f"<b>المادة {obj.properties.get('article_number')}</b> - {obj.properties.get('article_title')}",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"<div style='background-color: #012348; border-radius: 8px; padding: 8px; margin-bottom: 10px;'>{obj.properties.get('text').replace(chr(10), '<br>')}</div>",
                    unsafe_allow_html=True
                )
            st.markdown("</div>", unsafe_allow_html=True)
