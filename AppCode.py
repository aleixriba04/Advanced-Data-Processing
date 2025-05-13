#command to run app on terminal: python -m streamlit run FinalProjectV3.py

import streamlit as st
import requests
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Real-Time Market Intelligence Dashboard", layout="wide")

# Load OpenAI key
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Load products
@st.cache_data
def load_products():
    res = requests.get("https://dummyjson.com/products?limit=100")
    return res.json()["products"]

products = load_products()
categories = sorted(set(p["category"] for p in products if "category" in p))

# Sidebar filters
st.sidebar.header("🔍 Filters")
category = st.sidebar.selectbox("Choose a product category", ["All"] + categories)
max_price = st.sidebar.slider("Maximum price", 10, 1500, 1500)

# Filter products
filtered = [
    p for p in products
    if p["price"] <= max_price and (category == "All" or p["category"] == category)
]

# App header
st.markdown("### 📦 Real-Time Market Intelligence Dashboard for Ecommerce")
st.markdown("**This tool helps small business owners discover profitable, high-rated ecommerce products in seconds.**")

with st.expander("🔹 How to Use This App", expanded=False):
    st.markdown(
        """
        - Use the filters on the left to narrow down product results.  
        - Browse the filtered products and examine their **price**, **brand**, and **user rating**.  
        - Scroll down and click the button to generate AI-powered recommendations and graphs.
        """
    )

st.success(f"Found {len(filtered)} products.")

# Display products
cols = st.columns(2)
for idx, p in enumerate(filtered):
    with cols[idx % 2]:
        st.markdown(f"### {p['title']}")
        st.image(p["thumbnail"], width=200)
        st.markdown(f"💰 **Price:** ${p['price']}")
        st.markdown(f"⭐ **Rating:** {p['rating']}")
        if "brand" in p:
            st.markdown(f"📦 **Brand:** {p['brand']}")
        st.markdown(f"🗒️ {p['description'][:200]}...")

# === Analysis Section ===
st.markdown("---")
st.markdown("## 📊 Product Analysis & Market Insights")

if st.button("Generate Insights and Visuals"):
    # GPT Insight Section
    def get_gpt_insights(products):
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }

        prompt = f"""
You are an expert market analyst. A small business owner wants to identify the **top 3 products** to sell in an online store based on the following data.

Each product includes:
- title
- price
- rating
- brand
- stock
- discount percentage
- minimum order quantity (MOQ)

Analyze this list of products and give a **clear recommendation** with:
1. A short **ranked list** of the top 3 products to sell online.
2. **Business reasoning** (Why are they good? E.g., high margin, demand, low MOQ, high rating...).
3. A **1-sentence summary** with your final advice.

Here is the product data (JSON):
{json.dumps(products[:10], indent=2)}
"""


        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}]
        }

        res = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)

        if res.status_code == 200:
            return res.json()["choices"][0]["message"]["content"]
        else:
            return f"❌ GPT Error: {res.status_code} – {res.text}"

    # Create DataFrame for visualizations
    df = pd.DataFrame(filtered)
    #df = df[["title", "price", "rating", "brand"]]
    expected_cols = ["title", "price", "rating", "brand"]
    available_cols = [col for col in expected_cols if col in df.columns]
    df = df[available_cols]

    df_sorted = df.sort_values(by="rating", ascending=False)

    # Extract values from filtered products
    titles = [p["title"] for p in filtered]
    prices = [p["price"] for p in filtered]
    ratings = [p["rating"] for p in filtered]

    # Show scatter plot
    plt.figure(figsize=(8, 4))  # Smaller size
    plt.scatter(prices, ratings, s=200, color="orange")

    for i, title in enumerate(titles):
        plt.annotate(
            title,
            (prices[i], ratings[i]),
            textcoords="offset points",
            xytext=(5, 5),
            ha='left',
            fontsize=8,
            color="black",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8)
        )
    # Fit a linear regression line (trend line)
    z = np.polyfit(prices, ratings, 1)
    p = np.poly1d(z)
    plt.plot(prices, p(prices), linestyle="--", color="blue", label="Trend Line")

    # Shade a "Best Value Zone"
    plt.axhspan(4.0, 5.0, xmin=0, xmax=0.3, facecolor='green', alpha=0.1, label="Best Value Zone")

    plt.title("Best Value Products: Comparing Price & User Ratings")
    plt.xlabel("Price ($)")
    plt.ylabel("Rating (0-5)")
    plt.legend(loc="lower right")
    st.pyplot(plt.gcf())


    # Show product table
    st.markdown("### 📋 Best Rated Products")
    st.markdown("Sorted by **rating** to help you quickly spot top-rated products.")
    st.dataframe(df_sorted.reset_index(drop=True), use_container_width=True)

    # Show GPT Recommendation
    st.markdown("### 🤖 GPT-Powered Market Recommendation")
    with st.spinner("Analyzing data with GPT..."):
        insights = get_gpt_insights(filtered)
        st.markdown(insights)
else:
    st.info("⬆️ Click 'Generate Insights and Visuals' to run analysis and see GPT suggestions.")
