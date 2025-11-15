import streamlit as st
import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("AI Customer Query Classifier")
st.write("Enter any customer message and the AI will classify it.")

query = st.text_input("Enter your query:")

if st.button("Classify"):
    if query.strip() == "":
        st.warning("Please enter a message.")
    else:
        vector = vectorizer.transform([query])
        prediction = model.predict(vector)[0]
        st.success(f"Predicted Category: {prediction}") 
