import os
import re
import pickle
import urllib.request
import nltk
from flask import Flask, render_template, request
from ddgs import DDGS
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from difflib import SequenceMatcher
from newspaper import Article, Config
from transformers import pipeline

app = Flask(__name__, template_folder='./templates', static_folder='./static')

# --- MODEL LOADING ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Traditional ML (Emergency Fallback Only)
try:
    model_path = os.path.join(BASE_DIR, 'model.pkl')
    vector_path = os.path.join(BASE_DIR, 'vector.pkl')
    loaded_model = pickle.load(open(model_path, 'rb'))
    vector = pickle.load(open(vector_path, 'rb'))
except Exception as e:
    print(f"Legacy model load error: {e}")

# 2. Deep Learning Transformer (Primary Authority)
try:
    # Initializing RoBERTa as the global authority
    dl_classifier = pipeline("text-classification", model="hamzab/roberta-fake-news-detector")
except Exception as e:
    print(f"Deep Learning model load error: {e}")
    dl_classifier = None

lemmatizer = WordNetLemmatizer()
stpwrds = set(stopwords.words('english'))

# --- GLOBAL SOURCES ---
TRUSTED_SOURCES = ['reuters.com', 'apnews.com', 'bbc.com', 'nytimes.com', 'theguardian.com', 'indiatimes.com', 'ndtv.com', 'aljazeera.com']
FACT_CHECKERS = ['snopes.com', 'politifact.com', 'factcheck.org', 'altnews.in']

def get_similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def scrape_url(url):
    try:
        # Using a custom User-Agent to prevent news sites from blocking repeat requests
        config = Config()
        config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        config.request_timeout = 10
        
        article = Article(url, config=config)
        article.download()
        article.parse()
        return f"{article.title}. {article.text}"
    except Exception as e:
        print(f"Scraping Error: {e}")
        return None

def live_internet_verification(news_query):
    try:
        # Targeted search to avoid rate limits
        search_query = news_query[:150]
        results = DDGS().text(search_query, max_results=8)
        
        if not results: return None, None, None, None
        
        real_score, fake_score = 0, 0
        found_sources = []
        
        for res in results:
            url, title = res.get('href', '').lower(), res.get('title', '').lower()
            snippet = res.get('body', '').lower()
            
            # Ensure the search results actually match the input news
            if get_similarity(search_query, title) < 0.25: continue 

            if any(domain in url for domain in FACT_CHECKERS):
                if any(word in title or word in snippet for word in ['false', 'fake', 'hoax', 'debunked', 'misleading']):
                    fake_score += 4
                    found_sources.append("Fact-Checker")
            elif any(domain in url for domain in TRUSTED_SOURCES):
                real_score += 2
                found_sources.append(url.split('/')[2].replace('www.', ''))
        
        if fake_score > real_score and fake_score >= 4:
            return "Fake", 98, "Verified fact-checking organizations have flagged this story.", "Live Web Verification"
        elif real_score > fake_score and real_score >= 2:
            return "Real", min(99, 82 + (real_score * 3)), f"This is actively reported by trusted outlets: {', '.join(set(found_sources[:2]))}.", "Live Web Verification"
            
        return None, None, None, None
    except Exception as e:
        print(f"Internet Verification error: {e}")
        return None, None, None, None

def fake_news_det(news_content):
    # --- PHASE 1: INTERNET VERIFICATION ---
    res, conf, reason, method = live_internet_verification(news_content)
    if res is not None:
        print(f"Decision: Phase 1 (Internet) - Result: {res}")
        return res, conf, reason, method

    # --- PHASE 2: DEEP LEARNING ANALYSIS (RoBERTa) ---
    if dl_classifier:
        try:
            # Cleaning and truncating to avoid RoBERTa token limits
            clean_text = news_content.strip()[:1500] 
            dl_result = dl_classifier(clean_text)[0]
            
            label = dl_result['label']
            confidence = round(dl_result['score'] * 100, 1)
            
            # Map model labels to UI labels
            is_fake = label.upper() in ['FAKE', 'LABEL_1', 'DISPUTED']
            verdict = "Fake" if is_fake else "Real"
            
            reason = "The Deep Neural Engine detected linguistic patterns and semantic structures typical of misinformation." if is_fake else "The Deep Neural Engine identified high linguistic coherence typical of factual journalism."
            
            print(f"Decision: Phase 2 (RoBERTa) - Result: {verdict}")
            return verdict, confidence, reason, "RoBERTa Deep Learning"
        except Exception as e:
            print(f"Phase 2 inference error: {e}")

    # --- PHASE 3: LEGACY ML ANALYSIS (Emergency Fallback) ---
    # This only triggers if Phase 1 returns no results AND Phase 2 crashes.
    try:
        review = re.sub(r'[^a-zA-Z\s]', '', news_content).lower()
        tokens = nltk.word_tokenize(review)
        corpus = [' '.join([lemmatizer.lemmatize(y) for y in tokens if y not in stpwrds])]
        vec_input = vector.transform(corpus)
        prediction = loaded_model.predict(vec_input)
        
        score = abs(float(loaded_model.decision_function(vec_input)[0]))
        confidence = min(92, 65 + (score * 7))
        
        verdict = "Fake" if prediction[0] == 1 else "Real"
        reason = "Safety Fallback: Basic stylistic markers match non-credible source patterns." if verdict == "Fake" else "Safety Fallback: Basic linguistic traits align with reporting standards."
        
        print(f"Decision: Phase 3 (Legacy) - Result: {verdict}")
        return verdict, round(confidence, 1), reason, "Legacy ML Analysis"
    except Exception as e:
        return "Unknown", 0, f"System Error: {str(e)}", "Error Handler"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    url_input = request.form.get('news_url', '').strip()
    text_input = request.form.get('news_text', '').strip()
    
    content = ""
    if url_input:
        content = scrape_url(url_input)
        if not content:
            return render_template("prediction.html", result="Error", confidence=0, reason="Scraper could not access the URL content.", method="System Error")
    else:
        content = text_input

    if not content or len(content) < 10:
        return render_template("prediction.html", result="Error", confidence=0, reason="Insufficient content provided for analysis.", method="Input Error")

    res, conf, reason, method = fake_news_det(content)
    return render_template("prediction.html", result=res, confidence=conf, reason=reason, method=method)

if __name__ == '__main__':
    app.run(debug=True)