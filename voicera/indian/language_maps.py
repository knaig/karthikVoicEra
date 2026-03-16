"""Language code mappings for STT and TTS providers."""

STT_LANGUAGE_MAP = {
    "Deepgram": {
        "Hindi": "hi", "English": "en-US", "English (India)": "en-IN",
        "English (United States)": "en-US", "Bengali": "bn", "Tamil": "ta",
        "Telugu": "te", "Gujarati": "gu", "Kannada": "kn", "Malayalam": "ml",
        "Marathi": "mr", "Punjabi": "pa",
    },
    "Google": {
        "Hindi": "hi-IN", "English": "en-US", "English (India)": "en-IN",
        "English (United States)": "en-US", "Bengali": "bn-IN", "Tamil": "ta-IN",
        "Telugu": "te-IN", "Gujarati": "gu-IN", "Kannada": "kn-IN",
        "Malayalam": "ml-IN", "Marathi": "mr-IN", "Punjabi": "pa-IN",
    },
    "OpenAI": {
        "Hindi": "hi", "English": "en", "English (India)": "en",
        "English (United States)": "en", "Bengali": "bn", "Tamil": "ta",
        "Telugu": "te", "Gujarati": "gu", "Kannada": "kn", "Malayalam": "ml",
        "Marathi": "mr", "Punjabi": "pa",
    },
    "Sarvam": {
        "Hindi": "hi-IN", "English": "en-IN", "English (India)": "en-IN",
        "English (United States)": "en-US", "Bengali": "bn-IN", "Tamil": "ta-IN",
        "Telugu": "te-IN", "Gujarati": "gu-IN", "Kannada": "kn-IN",
        "Malayalam": "ml-IN", "Marathi": "mr-IN", "Punjabi": "pa-IN", "Odia": "od-IN",
    },
    "AI4Bharat": {
        "Hindi": "hi", "Bengali": "bn", "Tamil": "ta", "Telugu": "te",
        "Gujarati": "gu", "Kannada": "kn", "Malayalam": "ml", "Marathi": "mr",
        "Punjabi": "pa", "Assamese": "as", "Odia": "or", "Nepali": "ne",
    },
    "Bhashini": {
        "Hindi": "hi", "Bengali": "bn", "Tamil": "ta", "Telugu": "te",
        "Gujarati": "gu", "Kannada": "kn", "Malayalam": "ml", "Marathi": "mr",
        "Punjabi": "pa", "Assamese": "as", "Odia": "or", "Nepali": "ne",
        "Bodo": "brx", "Dogri": "doi", "Kashmiri": "ks", "Konkani": "kok",
        "Maithili": "mai", "Manipuri": "mni", "Sanskrit": "sa", "Santali": "sat",
        "Sindhi": "sd", "Urdu": "ur",
    },
}

TTS_LANGUAGE_MAP = {
    "Cartesia": {
        "Hindi": "hi", "English": "en", "English (India)": "en",
        "English (United States)": "en", "Bengali": "bn", "Tamil": "ta",
        "Telugu": "te", "Gujarati": "gu", "Kannada": "kn", "Malayalam": "ml",
        "Marathi": "mr", "Punjabi": "pa",
    },
    "Google": {
        "Hindi": "hi-IN", "English": "en-US", "English (India)": "en-IN",
        "English (United States)": "en-US", "Bengali": "bn-IN", "Tamil": "ta-IN",
        "Telugu": "te-IN", "Gujarati": "gu-IN", "Kannada": "kn-IN",
        "Malayalam": "ml-IN", "Marathi": "mr-IN", "Punjabi": "pa-IN",
    },
    "OpenAI": {
        "Hindi": "en", "English": "en", "English (India)": "en",
        "English (United States)": "en", "Bengali": "en", "Tamil": "en",
        "Telugu": "en", "Gujarati": "en", "Kannada": "en", "Malayalam": "en",
        "Marathi": "en", "Punjabi": "en",
    },
    "Sarvam": {
        "Hindi": "hi-IN", "English (India)": "en-IN", "Bengali": "bn-IN",
        "Tamil": "ta-IN", "Telugu": "te-IN", "Gujarati": "gu-IN",
        "Kannada": "kn-IN", "Malayalam": "ml-IN", "Marathi": "mr-IN",
        "Punjabi": "pa-IN", "Odia": "od-IN",
    },
    "AI4Bharat": {
        "Hindi": "hi", "Bengali": "bn", "Tamil": "ta", "Telugu": "te",
        "Gujarati": "gu", "Kannada": "kn", "Malayalam": "ml", "Marathi": "mr",
        "Punjabi": "pa", "Assamese": "as", "Odia": "or",
    },
}
