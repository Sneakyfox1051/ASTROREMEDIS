# 🔮 ASTROBOT KUNDALI - Dynamic Chart Generator

A powerful Streamlit-based astrological application that generates personalized Kundli charts and provides AI-powered astrological insights using the Prokerala API and OpenAI GPT-4.

## ✨ Features

- **Dynamic Kundli Chart Generation**: Generate accurate astrological charts based on birth details
- **AI-Powered Analysis**: Get personalized astrological insights in Hinglish using OpenAI GPT-4
- **Interactive Form Interface**: Easy-to-use form for collecting birth information
- **Real-time Geocoding**: Automatic location lookup and coordinate conversion
- **Multiple Chart Types**: Support for various astrological chart styles
- **KP Astrology Support**: Uses KP (Krishnamurti Paddhati) astrology system

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Streamlit
- Required Python packages (see requirements.txt)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd astro-main
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API keys**
   Create a `.streamlit/secrets.toml` file in your project directory:
   ```toml
   # .streamlit/secrets.toml
   OPENAI_API_KEY="your_openai_api_key_here"
   PROKERALA_CLIENT_ID="your_prokerala_client_id_here"
   PROKERALA_CLIENT_SECRET="your_prokerala_client_secret_here"
   ```

4. **Run the application**
   ```bash
   streamlit run astro3.py
   ```

5. **Access the app**
   Open your browser and go to `http://localhost:8501`

## 📋 Usage

1. **Enter Birth Details**: Fill in the form with:
   - Name
   - Birth Date
   - Birth Time
   - Birth Place

2. **Get Coordinates**: Click "Get Coordinates" to automatically fetch latitude and longitude

3. **Generate Chart**: Click "Generate Kundli Chart" to create your astrological chart

4. **Ask Questions**: Use the chat interface to ask questions about your chart

## 🔧 Configuration

### API Keys Setup

#### OpenAI API Key
1. Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Add it to your `secrets.toml` file

#### Prokerala API Credentials
1. Sign up at [Prokerala API](https://www.prokerala.com/api/)
2. Get your Client ID and Client Secret
3. Add them to your `secrets.toml` file

### Environment Variables (Alternative)
You can also set environment variables:
```bash
export OPENAI_API_KEY="your_key_here"
export PROKERALA_CLIENT_ID="your_client_id_here"
export PROKERALA_CLIENT_SECRET="your_client_secret_here"
```

## 📁 Project Structure

```
astro-main/
├── astro3.py                          # Main Streamlit application
├── requirements.txt                   # Python dependencies
├── .streamlit/
│   └── secrets.toml                   # API keys (not in version control)
├── docs/                              # Documentation files
│   ├── KP_RULE_1.docx
│   ├── KP_RULE_2.docx
│   └── KP_RULE_3.docx
├── astro remedis ai astrologer model.docx
├── BNn concept for AI bot training.docx
├── kp 12 houses and significator for chat bot.docx
└── LICENSE
```

## 🌟 Key Features Explained

### Dynamic Chart Generation
- Uses Prokerala API for accurate astrological calculations
- Supports KP Astrology (Ayanamsa: 5)
- Generates SVG charts with North Indian style
- Real-time planet position calculations

### AI-Powered Analysis
- Leverages OpenAI GPT-4 for intelligent responses
- Provides insights in Hinglish for better user experience
- Uses RAG (Retrieval Augmented Generation) for context-aware responses
- Structured output format with bullet points

### Interactive Interface
- Clean, modern Streamlit UI
- Form-based data collection
- Real-time geocoding integration
- Responsive design

## 🔒 Security

- API keys are stored in `secrets.toml` (excluded from version control)
- No sensitive data is committed to the repository
- Secure API key management for production deployment

## 🚀 Deployment

### Streamlit Cloud
1. Push your code to a GitHub repository
2. Connect your GitHub account to Streamlit Cloud
3. Deploy using the repository URL
4. Add your API keys in the Streamlit Cloud secrets section

### Local Development
```bash
streamlit run astro3.py
```

## 📊 Dependencies

- `streamlit` - Web application framework
- `requests` - HTTP library for API calls
- `geopy` - Geocoding library
- `pytz` - Timezone handling
- `matplotlib` - Chart visualization
- `langchain` - AI framework
- `openai` - OpenAI API client
- `faiss-cpu` - Vector similarity search

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

If you encounter any issues:
1. Check that all API keys are correctly set
2. Verify your internet connection
3. Ensure all dependencies are installed
4. Check the console for error messages

## 🔮 About Astrology

This application uses KP (Krishnamurti Paddhati) astrology, which is a modern system of Vedic astrology. It provides accurate predictions and insights based on precise astronomical calculations.

---

**Note**: This application is for educational and entertainment purposes. Always consult with professional astrologers for important life decisions.
