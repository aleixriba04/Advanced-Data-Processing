# 📦 Real-Time Market Intelligence Dashboard

## Description

The **Real-Time Market Intelligence Dashboard** is an intuitive, AI-powered web application built with **Streamlit**, designed to help **small business owners, entrepreneurs, and e-commerce sellers** quickly identify profitable and highly-rated products for online stores.

By combining **real-time product data** and **GPT-powered insights**, the app allows users to filter, visualize, and analyze products to support **data-driven business decisions**.  
This dashboard removes guesswork from product research and empowers users with actionable recommendations based on live market data.

---

## Table of Contents

- [Description](#description)
- [Installation Instructions](#installation-instructions)
- [Usage](#usage)
---

## Installation Instructions

### Prerequisites

Ensure you have the following installed:

- Python 3.8 or higher
- Git (optional, for cloning the repository)
- OpenAI API key (used for GPT-powered recommendations)

### Setup Steps

#### 1. Install Required Libraries
```python
import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import re
import time
import random
from bs4 import BeautifulSoup
import concurrent.futures
from fake_useragent import UserAgent
import plotly.express as px
```

#### 2. Set Up your API Keys
- Create a .streamlit/secrets.toml file in the project directory.
- Add your API keys:
```python
[general]
OPENAI_API_KEY = "openai_api_key_here"
SCRAPER_API_KEY = "scraper_api_key_here"
```

## Usage

#### 1. Launch inside an integrated terminal using:
```
python -m streamlit run AppCodeFinal.py
```


#### 2. Use the sidebar filters to:

- Select the product category.

- Adjust the maximum price.


#### 3. Browse the filtered product listings with detailed insights like:
- Price

- Brand

- Rating

- Description


#### 4. Click the "Find Cheaper Alternatives" button to:

- View scatter plots of Price vs Rating.

- Identify Best Value Zone for smart product picks.

- Get GPT-powered recommendations with reasoning and summary.










