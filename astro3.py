import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json
import openai
import time 
import requests # For Prokerala API calls
import pytz # For time zone handling
from geopy.geocoders import Nominatim # For converting place name to coordinates

# --- LangChain Imports ---
try:
    from langchain_community.document_loaders import Docx2txtLoader
    DOCX2TXT_AVAILABLE = True
except ImportError:
    DOCX2TXT_AVAILABLE = False
    st.warning("docx2txt not available. Word document processing will be limited.")

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# --- CONSTANTS ---
DOC_FILES = ["KP_RULE_1.docx", "KP_RULE_2.docx", "KP_RULE_3.docx"]
DEFAULT_LAT, DEFAULT_LON = 19.0760, 72.8777 # Mumbai Coordinates
DEFAULT_TZ = 'Asia/Kolkata' 

# --- API Key Setup ---
try:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    
    # --- Prokerala Credentials ---
    PROKERALA_CLIENT_ID = st.secrets["PROKERALA_CLIENT_ID"]
    PROKERALA_CLIENT_SECRET = st.secrets["PROKERALA_CLIENT_SECRET"]
    
except KeyError as e:
    st.error(f"API key not found in Streamlit secrets! Missing: {e.args[0]}", icon="🚨")
    st.stop()

# --- HELPER FUNCTIONS ---

@st.cache_data(ttl=3600) 
def geocode_place(place_name):
    """Converts a place name to (latitude, longitude) using Nominatim."""
    geolocator = Nominatim(user_agent="astrobot_app")
    try:
        location = geolocator.geocode(place_name, timeout=10)
        if location:
            return location.latitude, location.longitude
    except Exception as e:
        st.warning(f"Geocoding failed for '{place_name}': {e}")
    return DEFAULT_LAT, DEFAULT_LON # Fallback


@st.cache_data(ttl=3600) # Token is usually valid for 1 hour
def get_prokerala_access_token():
    """Fetches the access token from Prokerala API with enhanced error reporting."""
    url = "https://api.prokerala.com/token"
    data = {
        "grant_type": "client_credentials",
        "client_id": PROKERALA_CLIENT_ID,
        "client_secret": PROKERALA_CLIENT_SECRET
    }
    try:
        response = requests.post(url, data=data)
        
        # Enhanced Authentication Error Check (400 or 401)
        if response.status_code in [400, 401]:
             error_details = response.json().get('error_description', response.text)
             st.error(f"❌ Prokerala AUTH Failed (Status: {response.status_code}). "
                      f"Check Client ID/Secret in secrets.toml. Details: {error_details}")
             return None

        response.raise_for_status() 
        return response.json().get("access_token")
    except requests.exceptions.RequestException as e:
        # Catch network issues
        st.error(f"🌐 Network Error during Prokerala Token request: {e}")
        return None
    except Exception as e:
        st.error(f"Unknown Error during Prokerala Token request: {e}")
        return None

def calculate_chart_data(name, dob_date, tob_time, pob_text, latitude, longitude, timezone_str):
    """
    Fetches chart data using Prokerala API with KP Astrology settings:
    - Fixed ayanamsa=5 for KP Astrology
    - North Indian chart style
    - SVG output format
    - English language
    """
    access_token = get_prokerala_access_token()
    if not access_token:
        return None

    try:
        # Create localized datetime
        local_tz = pytz.timezone(timezone_str)
        birth_datetime = datetime.combine(dob_date, tob_time)
        localized_dt = local_tz.localize(birth_datetime)
        api_datetime_str = localized_dt.isoformat() 
    except Exception as e:
        st.error(f"Date/Time Error: {e}")
        return None

    headers = {"Authorization": f"Bearer {access_token}"}
    base_url = "https://api.prokerala.com/v2/astrology" 
    
    # Fixed parameters for KP Astrology
    common_params = {
        'ayanamsa': 5,  # KP Astrology (fixed)
        'coordinates': f"{latitude},{longitude}",
        'datetime': api_datetime_str
    }
    
    # --- Fetch Planet Positions (JSON Data) ---
    try:
        planet_url = f"{base_url}/planet-position"
        planet_response = requests.get(planet_url, headers=headers, params=common_params)
        planet_response.raise_for_status()
        planet_data = planet_response.json().get('data', {}).get('planet_position', [])
        st.success(f"✅ Planet positions fetched successfully: {len(planet_data)} planets")
    except Exception as e:
        st.error(f"❌ Error fetching Planet Positions: {e}")
        st.error(f"Response status: {planet_response.status_code if 'planet_response' in locals() else 'N/A'}")
        return None

    # --- Fetch Chart SVG (Visual Chart) ---
    try:
        chart_params = {
            **common_params,
            'chart_type': 'rasi',
            'chart_style': 'north-indian',  # Fixed
            'format': 'svg',  # Fixed
            'la': 'en'  # Fixed
        }
        
        chart_url = f"{base_url}/chart"
        chart_response = requests.get(chart_url, headers=headers, params=chart_params)
        chart_response.raise_for_status()
        chart_svg = chart_response.text
        st.success(f"✅ Chart SVG fetched successfully: {len(chart_svg)} characters")
    except Exception as e:
        st.warning(f"⚠️ Chart SVG not available: {e}")
        chart_svg = None

    # --- Process Planet Data ---
    planets_in_house = {}
    ascendant_sign = None
    ascendant_sign_name = "N/A"
    
    planet_code_map = {
        'Sun': 'Su', 'Moon': 'Mo', 'Mars': 'Ma', 'Mercury': 'Me', 
        'Jupiter': 'Ju', 'Venus': 'Ve', 'Saturn': 'Sa', 
        'Rahu': 'Ra', 'Ketu': 'Ke', 'Lagna': 'La'
    }

    if planet_data:
        # Find Lagna/Ascendant
        lagna_planet = next((p for p in planet_data if p.get('id') == 100), None)
        if not lagna_planet:
            lagna_planet = next((p for p in planet_data if p.get('name') == 'Lagna'), None)
        
        if lagna_planet:
            ascendant_sign = lagna_planet.get('rasi', {}).get('id')
            ascendant_sign_name = lagna_planet.get('rasi', {}).get('name')
                
        # Map Planets to Houses 
        if ascendant_sign is not None:
            for planet in planet_data:
                rasi_id = planet.get('rasi', {}).get('id')
                planet_name = planet.get('name')
                
                if rasi_id is not None:
                    house_num = (rasi_id - ascendant_sign + 12) % 12 + 1
                    planet_code = planet_code_map.get(planet_name, planet_name[:2])
                    
                    if house_num not in planets_in_house:
                        planets_in_house[house_num] = []
                    if planet_code not in planets_in_house[house_num]:
                         planets_in_house[house_num].append(planet_code)
    
    # --- Derive Mangal Dosha from Mars position ---
    mars_house = None
    for house_num, planets in planets_in_house.items():
        if 'Ma' in planets:
            mars_house = house_num
            break
    
    mangal_dosha_present = mars_house in [1, 4, 7, 8, 12] if mars_house else False
    
    # --- Final CHART_DATA Structure ---
    st.success(f"✅ Chart data processed successfully for {name}")
    return {
        "name": name,
        "ascendant_sign": ascendant_sign or 1, 
        "ascendant_sign_name": ascendant_sign_name,
        "planets": planets_in_house,
        "mangal_dosha": {
            "is_present": mangal_dosha_present,
            "description": f"Mangal Dosha {'Present' if mangal_dosha_present else 'Absent'} - Mars in House {mars_house if mars_house else 'Unknown'}"
        },
        "birth_location": pob_text,
        "raw_planet_data": planet_data,
        "chart_svg": chart_svg
    }

# --- Function to load and process Word documents (RAG) ---
@st.cache_resource
def load_vector_store(doc_files):
    all_docs = []
    
    if not DOCX2TXT_AVAILABLE:
        st.warning("docx2txt library not available. Using fallback text processing.")
        # Create a simple fallback with basic text content
        fallback_text = """
        KP Astrology Rules:
        1. House signification is very important in KP astrology
        2. Cuspal sublord is the key to predictions
        3. Planet's signification depends on its star lord
        4. Ruling planets at the time of query are important
        5. Dasa and antardasa periods are crucial for timing
        6. Sub-sub lords provide detailed analysis
        """
        from langchain.schema import Document
        all_docs.append(Document(page_content=fallback_text, metadata={"source": "fallback"}))
    else:
        for doc_file in doc_files:
            file_path = os.path.join("docs", doc_file)
            if os.path.exists(file_path):
                try:
                    loader = Docx2txtLoader(file_path)
                    all_docs.extend(loader.load())
                except Exception as e:
                    st.error(f"Error loading {doc_file}: {e}")
            else:
                print(f"Warning: Document file not found at {file_path}")

    if not all_docs:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(all_docs)
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(texts, embeddings)
    return vector_store

# --- Function to draw the Kundli chart ---
def draw_kundli_chart(planets_in_house, ascendant_sign):
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor("#FFF8E1")
    ax.set_facecolor('#FFF8E1')
    line_color, text_color, rashi_color = '#795548', '#3E2723', "#CA461E"
    
    ax.plot([0, 100, 100, 0, 0], [0, 0, 100, 100, 0], color=line_color)
    ax.plot([0, 100], [0, 100], color=line_color)
    ax.plot([0, 100], [100, 0], color=line_color)
    ax.plot([50, 100, 50, 0, 50], [0, 50, 100, 50, 0], color=line_color)
    
    coords_map = {1: (50, 75), 4: (25, 50), 7: (50, 25), 10: (75, 50), 
                  2: (25, 90), 5: (10, 25), 8: (75, 10), 11: (90, 75), 
                  3: (10, 75), 6: (25, 10), 9: (90, 25), 12: (75, 90)}
                  
    rashi_in_house = {i: (ascendant_sign + i - 2) % 12 + 1 for i in range(1, 13)}

    for house_num in range(1, 13):
        rashi = rashi_in_house.get(house_num, "")
        planets = planets_in_house.get(house_num, [])
        x, y = coords_map[house_num]
        
        ax.text(x, y + 5, str(rashi), color=rashi_color, fontsize=12, ha='center', va='center', weight='bold')
        ax.text(x, y - 8, " ".join(planets), color=text_color, fontsize=9, ha='center', va='center')
        
    ax.set_xlim(-5, 105); ax.set_ylim(-5, 105); ax.axis('off')
    return fig

# --- Function to get a Hinglish response from the AI ---
def get_rag_response(question, vector_store, chart_data):
    """Generate detailed personalized AI response based on chart data and KP astrology knowledge"""
    if vector_store is None:
        return "• **Knowledge base not loaded** - Please check your document files."
        
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(question)
    context_from_docs = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # Extract key chart information for context
    chart_summary = f"""
    **Name:** {chart_data['name']}
    **Ascendant:** {chart_data['ascendant_sign_name']} (Sign {chart_data['ascendant_sign']})
    **Location:** {chart_data['birth_location']}
    **Planet Positions:** {json.dumps(chart_data['planets'], indent=2)}
    **Mangal Dosha:** {'Present' if chart_data['mangal_dosha']['is_present'] else 'Absent'}
    """

    system_prompt = f"""
    You are AstroBot, an experienced and respected KP Jyotishacharya (Astrologer). Your role is to provide detailed, personalized astrological analysis.

    **Response Format (CRITICAL - Follow this EXACT format):**
    
    Start with: "Aapka ascendant sign [SIGN_NAME] hai, jo ki Aapke vyaktitva ke bohot se pehluon ko prabhavit karta hai."
    
    Then provide 4-6 detailed bullet points using this format:
    * [Detailed personality trait or characteristic]
    * [Another aspect of their nature or behavior]
    * [Strengths or positive qualities]
    * [Potential challenges or areas to be aware of]
    * [Career or life path insights]
    * [Relationships or social aspects]
    
    End with: "Overall, Aap [summary of their nature] hote hain aur unka nature [descriptive quality] hota hai."

    **Response Rules:**
    - **Use Hinglish:** Respond in Hinglish (Hindi + English, Roman script)
    - **Respectful Pronouns:** Always use 'Aap' and 'Aapka' when addressing the user
    - **Detailed Analysis:** Provide comprehensive personality insights based on their ascendant sign
    - **Personalized:** Make it specific to their chart data
    - **Professional Tone:** Like an experienced astrologer giving a detailed reading
    - **Use Chart Data:** Reference their specific ascendant sign and planetary positions
    - **Bullet Points:** Use asterisk (*) for each point, not bullet symbols

    ---
    **CHART DATA FOR {chart_data['name']}:**
    {chart_summary}
    ---
    **KP ASTROLOGY KNOWLEDGE:**
    {context_from_docs}
    ---
    
    Now, provide a detailed, personalized astrological analysis in the EXACT format specified above for this question: "{question}"
    """
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": system_prompt}],
            temperature=0.8,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"• **Error:** Sorry, I encountered an issue with the AI model: {e}"

# ----------------------------------------------------------------------
## MAIN STREAMLIT APPLICATION
# ----------------------------------------------------------------------

st.set_page_config(page_title="AstroBot Kundli", layout="wide")

st.title("🔮 ASTROBOT KUNDALI - Your Personal AI Astrologer")

vector_store = load_vector_store(DOC_FILES)

# --- Initialize session state ---
if 'current_chart_data' not in st.session_state:
    st.session_state.current_chart_data = None

# --- Birth Details Form ---
st.markdown("### 📝 Enter Your Birth Details")

with st.form("birth_details_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("Full Name", placeholder="Enter your full name")
        dob = st.date_input("Date of Birth", value=datetime(2000, 5, 24).date())
        tob = st.time_input("Time of Birth", value=datetime.strptime("14:30", "%H:%M").time())
    
    with col2:
        pob = st.text_input("Place of Birth", placeholder="e.g., Mumbai, India")
        tz = st.selectbox("Timezone", options=pytz.common_timezones, index=pytz.common_timezones.index(DEFAULT_TZ) if DEFAULT_TZ in pytz.common_timezones else 0)
    
    # Geocoding button
    if st.form_submit_button("📍 Get Coordinates", help="Click to find coordinates for your place of birth"):
        if pob:
            with st.spinner("Finding coordinates..."):
                lat, lon = geocode_place(pob)
                st.success(f"Coordinates found: {lat:.4f}, {lon:.4f}")
                st.session_state.lat = lat
                st.session_state.lon = lon
        else:
            st.warning("Please enter a place of birth first")
    
    # Generate chart button
    if st.form_submit_button("🔮 Generate Kundli Chart", type="primary"):
        if not all([name, pob]):
            st.error("Please fill in all required fields (Name and Place of Birth)")
        else:
            # Use stored coordinates or default
            latitude = getattr(st.session_state, 'lat', DEFAULT_LAT)
            longitude = getattr(st.session_state, 'lon', DEFAULT_LON)
            
            with st.spinner("Generating your Kundli chart..."):
                try:
                    chart_data = calculate_chart_data(
                        name, dob, tob, pob, latitude, longitude, tz
                    )
                    
                    if chart_data:
                        st.session_state.current_chart_data = chart_data
                        st.success("✅ Chart generated successfully!")
                    else:
                        st.error("❌ Failed to generate chart. Please check your details and try again.")
                except Exception as e:
                    st.error(f"❌ Error generating chart: {str(e)}")

# --- Display Chart if Available ---
if st.session_state.current_chart_data:
    chart_data = st.session_state.current_chart_data
    
    st.markdown("---")
    st.markdown(f"### 🔮 Kundli Chart for {chart_data['name']}")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Show SVG chart if available, otherwise show custom chart
        if chart_data.get("chart_svg"):
            st.markdown("**📊 Prokerala Chart:**")
            components.html(chart_data["chart_svg"], height=400)
        else:
            chart_figure = draw_kundli_chart(chart_data["planets"], chart_data["ascendant_sign"])
            st.pyplot(chart_figure)
    
    with col2:
        st.markdown("**📊 Chart Analysis:**")
        dosha_info = chart_data["mangal_dosha"]
        if dosha_info["is_present"]:
            st.error(f"**❌ Mangal Dosha: Present**\n\n{dosha_info['description']}")
        else:
            st.success(f"**✅ Mangal Dosha: Absent**\n\n{dosha_info['description']}")
        
        st.info(f"**🌟 Ascendant (Lagna):** {chart_data['ascendant_sign_name']} (Sign {chart_data['ascendant_sign']})")
        st.markdown(f"**📍 Location:** {chart_data['birth_location']}")
        
        # Planet positions
        st.markdown("**🪐 Planet Positions:**")
        for house_num in sorted(chart_data["planets"].keys()):
            planets = chart_data["planets"][house_num]
            st.markdown(f"House {house_num}: {', '.join(planets)}")

# --- Chat Interface for Questions ---
if st.session_state.current_chart_data:
    st.markdown("---")
    st.markdown("### 💬 Ask Questions About Your Chart")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your chart..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            thinking_message = "Typing"
            message_placeholder.markdown(thinking_message + "...")
            time.sleep(0.2)
            
            for i in range(3):
                message_placeholder.markdown(thinking_message + "." * (i + 1))
                time.sleep(0.15)
            
            # Generate AI response
            response = get_rag_response(prompt, vector_store, st.session_state.current_chart_data)
            message_placeholder.markdown(response)
            
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Reset button
    if st.button("🔄 Reset Chat", help="Clear chat history"):
        st.session_state.messages = []
        st.rerun()