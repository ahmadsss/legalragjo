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

# 🧠 Generate a legal-style answer using GPT-4-turbo
def generate_answer(question, context):
    context_text = "\n\n".join(
        f"المادة {o.properties.get('article_number', '')}: {o.properties.get('article_title', '')}\n{o.properties.get('text', '')}"
        for o in context
    )

    prompt = f"""أنت مساعد قانوني ذكي تجيب بأسلوب بشري مبني فقط على النصوص القانونية المتاحة من قاعدة البيانات.

🎯 التوجيه الأساسي:

    تحدث مع المستخدم كما لو كنت مستشارًا قانونيًا يشرح المفاهيم لشخص غير متخصص.

    تجنب اللغة القانونية الجامدة أو النسخ الحرفي للنصوص، وركّز على الشرح البسيط والمفيد.

    لا تذكر أرقام المواد أو عناوينها إطلاقًا، إلا إذا طلب المستخدم ذلك صراحة (مثلاً: قال "أنا محامٍ" أو "أعطني التفاصيل القانونية").

📌 قواعد الأسلوب:

    اعتمد فقط على النصوص القانونية التي تم استرجاعها، ولا تضف شيئًا من خارج السياق.

    استخدم لغة واضحة، منطقية، مرتبة، ومباشرة.

    لا تُطِل أو تُكرر، وابتعد عن الحشو.

    إذا لم يكن النص المسترجع كافيًا للإجابة، صرّح بذلك بوضوح.

⚖️ في حالة وجود أكثر من نسخة للمادة (أصلية ومعدّلة):

    افترض أن المادة المعدّلة هي السارية، لكن قارنها بالأصل إذا كان السياق يتطلب ذلك (مثلاً عند السؤال عن "ما التعديل الذي طرأ").

    في هذه الحالة، استخدم أسلوب المقارنة بعبارات مثل "تم تغيير كذا إلى كذا"، أو "أضيفت فقرة تنص على...".

    لا تستنتج أو تُخمّن — اجعل المقارنة واضحة ومعتمدة فقط على ما ورد في النصوص المعروضة أمامك.

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
