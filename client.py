import streamlit as st
import requests
import json

API_URL = "http://localhost:8000/recommend"

st.set_page_config(page_title="SHL Assessment Recommender", layout="centered")

st.title("🔍 SHL Assessment Recommendation Engine")

st.write(
    "Paste a **job description**, a **role name**, or a **query** and get the top 5 relevant SHL assessments."
)

# Input box
user_query = st.text_area("Enter your query:", height=150)

if st.button("Get Recommendations"):
    if not user_query.strip():
        st.error("Please enter a query.")
    else:
        # Make request to FastAPI
        payload = {"query": user_query}
        try:
            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                result = response.json()

                recs = result.get("recommendations", [])

                # Display results cleanly
                st.subheader("✅ Top Recommended Assessments")
                
                if isinstance(recs, str):
                    # Sometimes output is a string; try to parse it
                    try:
                        recs = json.loads(recs)
                    except:
                        st.write("Raw Output:")
                        st.code(recs)
                        st.stop()

                for i, item in enumerate(recs, start=1):
                    st.markdown(f"### {i}. {item.get('Assessment Name', '')}")
                    st.write(f"**Job Levels:** {item.get('Job Levels', '')}")
                    st.write(f"**Description:** {item.get('Description', '')}")
                    st.write(f"**Language:** {item.get('Language', '')}")
                    st.write(f"**Assessment Length:** {item.get('Assessment Length', '')}")
                    st.write(f"**Test Type:** {item.get('Test Type', '')}")
                    st.write(f"**URL:** {item.get('URL', '')}")
                    st.markdown("---")
            else:
                st.error(f"API Error: {response.text}")

        except Exception as e:
            st.error(f"Could not connect to the API: {e}")
