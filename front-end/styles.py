import streamlit as st

def load_enhanced_neon_theme():
    st.markdown(
        """
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
        
        /* Global Styles */
        * {
            font-family: 'Poppins', sans-serif;
        }
        
        /* Main background with gradient */
        .stApp {
            background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0a0e27 100%);
            background-attachment: fixed;
        }
        
        /* Remove default Streamlit padding */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 900px;
        }
        
        /* Animated Header */
        .header-container {
            text-align: center;
            padding: 2rem 0;
            animation: fadeInDown 1s ease-out;
        }
        
        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .neon-title {
            font-size: 3rem;
            font-weight: 700;
            background: linear-gradient(90deg, #00d4ff, #00ffff, #00d4ff);
            background-size: 200% auto;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: neonGlow 3s ease-in-out infinite;
            margin-bottom: 0.5rem;
            text-shadow: 0 0 20px rgba(0, 229, 255, 0.5);
        }
        
        @keyframes neonGlow {
            0%, 100% {
                filter: drop-shadow(0 0 10px rgba(0, 229, 255, 0.7));
            }
            50% {
                filter: drop-shadow(0 0 20px rgba(0, 229, 255, 1));
            }
        }
        
        .subtitle {
            color: #a0aec0;
            font-size: 1.1rem;
            font-weight: 300;
            margin-top: 0;
        }
        
        /* Info Cards */
        .info-card {
            background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(0, 255, 255, 0.05));
            border: 1px solid rgba(0, 229, 255, 0.3);
            border-radius: 15px;
            padding: 1.5rem;
            text-align: center;
            transition: all 0.3s ease;
            animation: fadeInUp 0.8s ease-out;
            backdrop-filter: blur(10px);
        }
        
        .info-card:hover {
            transform: translateY(-5px);
            border-color: rgba(0, 229, 255, 0.6);
            box-shadow: 0 10px 30px rgba(0, 229, 255, 0.3);
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .info-icon {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }
        
        .info-text {
            color: #00e5ff;
            font-weight: 600;
            font-size: 1rem;
        }
        
        /* Form Container */
        .form-container {
            background: rgba(26, 31, 58, 0.6);
            border: 1px solid rgba(0, 229, 255, 0.2);
            border-radius: 20px;
            padding: 2rem;
            margin-top: 2rem;
            animation: fadeIn 1s ease-out;
            backdrop-filter: blur(10px);
        }
        
        @keyframes fadeIn {
            from {
                opacity: 0;
            }
            to {
                opacity: 1;
            }
        }
        
        .section-header {
            color: #00e5ff;
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid rgba(0, 229, 255, 0.3);
        }
        
        /* Input Fields */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input {
            background-color: rgba(10, 14, 39, 0.8) !important;
            border: 1px solid rgba(0, 229, 255, 0.3) !important;
            border-radius: 10px !important;
            color: #e6e6e6 !important;
            padding: 0.75rem !important;
            transition: all 0.3s ease !important;
        }
        
        .stTextInput > div > div > input:focus,
        .stNumberInput > div > div > input:focus {
            border-color: #00e5ff !important;
            box-shadow: 0 0 15px rgba(0, 229, 255, 0.3) !important;
            transform: scale(1.02);
        }
        
        /* Select Boxes */
        .stSelectbox > div > div {
            background-color: rgba(10, 14, 39, 0.8) !important;
            border: 1px solid rgba(0, 229, 255, 0.3) !important;
            border-radius: 10px !important;
            transition: all 0.3s ease !important;
        }
        
        .stSelectbox > div > div:hover {
            border-color: #00e5ff !important;
            box-shadow: 0 0 15px rgba(0, 229, 255, 0.2) !important;
        }
        
        /* Labels */
        .stTextInput > label,
        .stNumberInput > label,
        .stSelectbox > label {
            color: #a0aec0 !important;
            font-weight: 500 !important;
            font-size: 0.95rem !important;
        }
        
        /* Submit Button */
        .stButton > button {
            background: linear-gradient(90deg, #00d4ff, #00ffff) !important;
            color: #0a0e27 !important;
            border: none !important;
            border-radius: 12px !important;
            font-weight: 700 !important;
            font-size: 1.1rem !important;
            padding: 0.75rem 2rem !important;
            transition: all 0.3s ease !important;
            text-transform: uppercase;
            letter-spacing: 1px;
            box-shadow: 0 5px 20px rgba(0, 229, 255, 0.4) !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(0, 229, 255, 0.6) !important;
            background: linear-gradient(90deg, #00ffff, #00d4ff) !important;
        }
        
        .stButton > button:active {
            transform: translateY(0);
        }
        
        /* Result Cards */
        .result-card {
            border-radius: 20px;
            padding: 2.5rem;
            text-align: center;
            margin: 2rem 0;
            animation: scaleIn 0.5s ease-out;
            backdrop-filter: blur(10px);
        }
        
        @keyframes scaleIn {
            from {
                opacity: 0;
                transform: scale(0.9);
            }
            to {
                opacity: 1;
                transform: scale(1);
            }
        }
        
        .success-card {
            background: linear-gradient(135deg, rgba(0, 212, 255, 0.15), rgba(0, 255, 255, 0.1));
            border: 2px solid rgba(0, 229, 255, 0.5);
            box-shadow: 0 10px 40px rgba(0, 229, 255, 0.3);
        }
        
        .error-card {
            background: linear-gradient(135deg, rgba(255, 82, 82, 0.15), rgba(255, 107, 107, 0.1));
            border: 2px solid rgba(255, 82, 82, 0.5);
            box-shadow: 0 10px 40px rgba(255, 82, 82, 0.3);
        }
        
        .result-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
            animation: bounceIn 0.8s ease-out;
        }
        
        @keyframes bounceIn {
            0% {
                opacity: 0;
                transform: scale(0.3);
            }
            50% {
                opacity: 1;
                transform: scale(1.1);
            }
            100% {
                transform: scale(1);
            }
        }
        
        .result-title {
            color: #a0aec0;
            font-size: 1.2rem;
            font-weight: 500;
            margin-bottom: 0.5rem;
        }
        
        .result-price {
            color: #00e5ff;
            font-size: 3rem;
            font-weight: 700;
            margin: 1rem 0;
            text-shadow: 0 0 20px rgba(0, 229, 255, 0.5);
            animation: pulse 2s ease-in-out infinite;
        }
        
        @keyframes pulse {
            0%, 100% {
                transform: scale(1);
            }
            50% {
                transform: scale(1.05);
            }
        }
        
        .result-subtitle {
            color: #718096;
            font-size: 1rem;
            font-weight: 300;
        }
        
        /* Info Box */
        .info-box {
            background: rgba(0, 212, 255, 0.1);
            border-left: 4px solid #00e5ff;
            border-radius: 8px;
            padding: 1rem;
            margin: 1.5rem 0;
            color: #cbd5e0;
            animation: slideInLeft 0.6s ease-out;
        }
        
        @keyframes slideInLeft {
            from {
                opacity: 0;
                transform: translateX(-30px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        /* Spinner */
        .stSpinner > div {
            border-top-color: #00e5ff !important;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            color: #718096;
            font-size: 0.9rem;
            padding: 2rem 0;
            border-top: 1px solid rgba(0, 229, 255, 0.2);
            margin-top: 3rem;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(10, 14, 39, 0.5);
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, #00d4ff, #00ffff);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(180deg, #00ffff, #00d4ff);
        }
        </style>
        """,
        unsafe_allow_html=True
    )