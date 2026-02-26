import streamlit as st
import pandas as pd
from huggingface_hub import InferenceClient

st.markdown("""
<style>
div.stDownloadButton > button {
    background-color: #28a745;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.5em 1em;
}

div.stDownloadButton > button:hover {
    background-color: #218838;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Academic Discipline Intelligence (Beta)", layout="wide")

st.title("🎓 Academic Discipline Intelligence (Beta)")
st.caption("AI-powered early risk detection & micro-intervention system")

# --- Downloadable Sample File ---
df = pd.read_csv("student_activity.csv")

#st.subheader("Sample Data")
#st.dataframe(df)

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Sample Data",
    data=csv,
    file_name="student_activity.csv",
    mime="text/csv",
)

# --- Secret ---
if "HUGGINGFACE_TOKEN" not in st.secrets:
    st.error("HUGGINGFACE_TOKEN not found in Streamlit secrets.")
    st.stop()

api_key = st.secrets["HUGGINGFACE_TOKEN"]

client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    token=api_key
)

uploaded_file = st.file_uploader("Upload student_activity.csv", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Student Activity Data")
    st.dataframe(df)

    if st.button("Run AI Analysis"):

        data_text = df.to_string(index=False)

        prompt = f"""
You are an AI learning coach.

Step 1:
Classify each student as:
- No Risk
- Medium Risk
- High Risk

Step 2:
For each student, write a short supportive intervention message.

Keep it concise and structured.

Data:
{data_text}
"""

        with st.spinner("Running AI analysis..."):
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
            )

        result = response.choices[0].message.content
        
        # Next steps'ten sonra zorunlu satır kır
        result = result.replace("Next steps:", "Next steps:\n\n")

        st.markdown(result)

        #st.subheader("🧠 AI Output")

        #st.write(result)





