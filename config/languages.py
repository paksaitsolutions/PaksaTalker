"""
Language and Voice Configuration for PaksaTalker
Comprehensive mapping of all supported languages and voices
"""

# Comprehensive language and voice mapping
SUPPORTED_LANGUAGES = {
    # European Languages
    "en-US": {
        "name": "English (US)",
        "flag": "🇺🇸",
        "voices": {
            "en-US-JennyNeural": {"name": "Jenny", "gender": "Female"},
            "en-US-ChristopherNeural": {"name": "Christopher", "gender": "Male"},
            "en-US-AriaNeural": {"name": "Aria", "gender": "Female"},
            "en-US-GuyNeural": {"name": "Guy", "gender": "Male"},
            "en-US-SaraNeural": {"name": "Sara", "gender": "Female"},
            "en-US-TonyNeural": {"name": "Tony", "gender": "Male"},
        }
    },
    "en-GB": {
        "name": "English (UK)",
        "flag": "🇬🇧",
        "voices": {
            "en-GB-SoniaNeural": {"name": "Sonia", "gender": "Female"},
            "en-GB-RyanNeural": {"name": "Ryan", "gender": "Male"},
            "en-GB-LibbyNeural": {"name": "Libby", "gender": "Female"},
        }
    },
    "es": {
        "name": "Spanish",
        "flag": "🇪🇸",
        "voices": {
            "es-ES-ElviraNeural": {"name": "Elvira", "gender": "Female", "region": "Spain"},
            "es-ES-AlvaroNeural": {"name": "Alvaro", "gender": "Male", "region": "Spain"},
            "es-MX-DaliaNeural": {"name": "Dalia", "gender": "Female", "region": "Mexico"},
            "es-MX-JorgeNeural": {"name": "Jorge", "gender": "Male", "region": "Mexico"},
        }
    },
    "fr": {
        "name": "French",
        "flag": "🇫🇷",
        "voices": {
            "fr-FR-DeniseNeural": {"name": "Denise", "gender": "Female", "region": "France"},
            "fr-FR-HenriNeural": {"name": "Henri", "gender": "Male", "region": "France"},
            "fr-CA-SylvieNeural": {"name": "Sylvie", "gender": "Female", "region": "Canada"},
        }
    },
    "de": {
        "name": "German",
        "flag": "🇩🇪",
        "voices": {
            "de-DE-KatjaNeural": {"name": "Katja", "gender": "Female", "region": "Germany"},
            "de-DE-ConradNeural": {"name": "Conrad", "gender": "Male", "region": "Germany"},
            "de-AT-IngridNeural": {"name": "Ingrid", "gender": "Female", "region": "Austria"},
        }
    },
    "it": {
        "name": "Italian",
        "flag": "🇮🇹",
        "voices": {
            "it-IT-ElsaNeural": {"name": "Elsa", "gender": "Female"},
            "it-IT-DiegoNeural": {"name": "Diego", "gender": "Male"},
            "it-IT-IsabellaNeural": {"name": "Isabella", "gender": "Female"},
        }
    },
    "pt": {
        "name": "Portuguese",
        "flag": "🇵🇹",
        "voices": {
            "pt-BR-FranciscaNeural": {"name": "Francisca", "gender": "Female", "region": "Brazil"},
            "pt-BR-AntonioNeural": {"name": "Antonio", "gender": "Male", "region": "Brazil"},
            "pt-PT-RaquelNeural": {"name": "Raquel", "gender": "Female", "region": "Portugal"},
        }
    },
    "ru": {
        "name": "Russian",
        "flag": "🇷🇺",
        "voices": {
            "ru-RU-SvetlanaNeural": {"name": "Svetlana", "gender": "Female"},
            "ru-RU-DmitryNeural": {"name": "Dmitry", "gender": "Male"},
        }
    },
    "nl": {
        "name": "Dutch",
        "flag": "🇳🇱",
        "voices": {
            "nl-NL-ColetteNeural": {"name": "Colette", "gender": "Female"},
            "nl-NL-MaartenNeural": {"name": "Maarten", "gender": "Male"},
        }
    },
    "sv": {
        "name": "Swedish",
        "flag": "🇸🇪",
        "voices": {
            "sv-SE-SofieNeural": {"name": "Sofie", "gender": "Female"},
            "sv-SE-MattiasNeural": {"name": "Mattias", "gender": "Male"},
        }
    },
    "no": {
        "name": "Norwegian",
        "flag": "🇳🇴",
        "voices": {
            "nb-NO-PernilleNeural": {"name": "Pernille", "gender": "Female"},
            "nb-NO-FinnNeural": {"name": "Finn", "gender": "Male"},
        }
    },
    "da": {
        "name": "Danish",
        "flag": "🇩🇰",
        "voices": {
            "da-DK-ChristelNeural": {"name": "Christel", "gender": "Female"},
            "da-DK-JeppeNeural": {"name": "Jeppe", "gender": "Male"},
        }
    },
    "fi": {
        "name": "Finnish",
        "flag": "🇫🇮",
        "voices": {
            "fi-FI-NooraNeural": {"name": "Noora", "gender": "Female"},
            "fi-FI-HarriNeural": {"name": "Harri", "gender": "Male"},
        }
    },
    "pl": {
        "name": "Polish",
        "flag": "🇵🇱",
        "voices": {
            "pl-PL-ZofiaNeural": {"name": "Zofia", "gender": "Female"},
            "pl-PL-MarekNeural": {"name": "Marek", "gender": "Male"},
        }
    },
    "cs": {
        "name": "Czech",
        "flag": "🇨🇿",
        "voices": {
            "cs-CZ-VlastaNeural": {"name": "Vlasta", "gender": "Female"},
            "cs-CZ-AntoninNeural": {"name": "Antonin", "gender": "Male"},
        }
    },
    "hu": {
        "name": "Hungarian",
        "flag": "🇭🇺",
        "voices": {
            "hu-HU-NoemiNeural": {"name": "Noemi", "gender": "Female"},
            "hu-HU-TamasNeural": {"name": "Tamas", "gender": "Male"},
        }
    },
    "el": {
        "name": "Greek",
        "flag": "🇬🇷",
        "voices": {
            "el-GR-AthinaNeural": {"name": "Athina", "gender": "Female"},
            "el-GR-NestorNeural": {"name": "Nestor", "gender": "Male"},
        }
    },
    "tr": {
        "name": "Turkish",
        "flag": "🇹🇷",
        "voices": {
            "tr-TR-EmelNeural": {"name": "Emel", "gender": "Female"},
            "tr-TR-AhmetNeural": {"name": "Ahmet", "gender": "Male"},
        }
    },

    # Asian Languages
    "zh": {
        "name": "Chinese",
        "flag": "🇨🇳",
        "voices": {
            "zh-CN-XiaoxiaoNeural": {"name": "Xiaoxiao", "gender": "Female", "region": "Mandarin"},
            "zh-CN-YunxiNeural": {"name": "Yunxi", "gender": "Male", "region": "Mandarin"},
            "zh-CN-XiaohanNeural": {"name": "Xiaohan", "gender": "Female", "region": "Mandarin"},
            "zh-HK-HiuMaanNeural": {"name": "HiuMaan", "gender": "Female", "region": "Cantonese"},
            "zh-HK-WanLungNeural": {"name": "WanLung", "gender": "Male", "region": "Cantonese"},
            "zh-TW-HsiaoChenNeural": {"name": "HsiaoChen", "gender": "Female", "region": "Taiwanese"},
            "zh-TW-YunJheNeural": {"name": "YunJhe", "gender": "Male", "region": "Taiwanese"},
        }
    },
    "ja": {
        "name": "Japanese",
        "flag": "🇯🇵",
        "voices": {
            "ja-JP-NanamiNeural": {"name": "Nanami", "gender": "Female"},
            "ja-JP-KeitaNeural": {"name": "Keita", "gender": "Male"},
            "ja-JP-AoiNeural": {"name": "Aoi", "gender": "Female"},
        }
    },
    "ko": {
        "name": "Korean",
        "flag": "🇰🇷",
        "voices": {
            "ko-KR-SunHiNeural": {"name": "SunHi", "gender": "Female"},
            "ko-KR-InJoonNeural": {"name": "InJoon", "gender": "Male"},
        }
    },
    "th": {
        "name": "Thai",
        "flag": "🇹🇭",
        "voices": {
            "th-TH-AcharaNeural": {"name": "Achara", "gender": "Female"},
            "th-TH-NiwatNeural": {"name": "Niwat", "gender": "Male"},
        }
    },
    "vi": {
        "name": "Vietnamese",
        "flag": "🇻🇳",
        "voices": {
            "vi-VN-HoaiMyNeural": {"name": "HoaiMy", "gender": "Female"},
            "vi-VN-NamMinhNeural": {"name": "NamMinh", "gender": "Male"},
        }
    },
    "id": {
        "name": "Indonesian",
        "flag": "🇮🇩",
        "voices": {
            "id-ID-GadisNeural": {"name": "Gadis", "gender": "Female"},
            "id-ID-ArdiNeural": {"name": "Ardi", "gender": "Male"},
        }
    },
    "ms": {
        "name": "Malay",
        "flag": "🇲🇾",
        "voices": {
            "ms-MY-YasminNeural": {"name": "Yasmin", "gender": "Female"},
            "ms-MY-OsmanNeural": {"name": "Osman", "gender": "Male"},
        }
    },
    "my": {
        "name": "Myanmar",
        "flag": "🇲🇲",
        "voices": {
            "my-MM-NilarNeural": {"name": "Nilar", "gender": "Female"},
            "my-MM-ThihaNeural": {"name": "Thiha", "gender": "Male"},
        }
    },
    "km": {
        "name": "Khmer",
        "flag": "🇰🇭",
        "voices": {
            "km-KH-SreymomNeural": {"name": "Sreymom", "gender": "Female"},
            "km-KH-PisachNeural": {"name": "Pisach", "gender": "Male"},
        }
    },
    "lo": {
        "name": "Lao",
        "flag": "🇱🇦",
        "voices": {
            "lo-LA-KeomanyNeural": {"name": "Keomany", "gender": "Female"},
            "lo-LA-ChanthavongNeural": {"name": "Chanthavong", "gender": "Male"},
        }
    },
    "mn": {
        "name": "Mongolian",
        "flag": "🇲🇳",
        "voices": {
            "mn-MN-YesuiNeural": {"name": "Yesui", "gender": "Female"},
            "mn-MN-BatbayarNeural": {"name": "Batbayar", "gender": "Male"},
        }
    },

    # South Asian Languages
    "hi": {
        "name": "Hindi",
        "flag": "🇮🇳",
        "voices": {
            "hi-IN-SwaraNeural": {"name": "Swara", "gender": "Female"},
            "hi-IN-MadhurNeural": {"name": "Madhur", "gender": "Male"},
        }
    },
    "ur": {
        "name": "Urdu",
        "flag": "🇵🇰",
        "voices": {
            "ur-PK-UzmaNeural": {"name": "Uzma", "gender": "Female"},
            "ur-PK-AsadNeural": {"name": "Asad", "gender": "Male"},
        }
    },
    "bn": {
        "name": "Bengali",
        "flag": "🇧🇩",
        "voices": {
            "bn-BD-NabanitaNeural": {"name": "Nabanita", "gender": "Female"},
            "bn-BD-PradeepNeural": {"name": "Pradeep", "gender": "Male"},
        }
    },
    "ta": {
        "name": "Tamil",
        "flag": "🇱🇰",
        "voices": {
            "ta-IN-PallaviNeural": {"name": "Pallavi", "gender": "Female"},
            "ta-IN-ValluvarNeural": {"name": "Valluvar", "gender": "Male"},
        }
    },
    "si": {
        "name": "Sinhala",
        "flag": "🇱🇰",
        "voices": {
            "si-LK-ThiliniNeural": {"name": "Thilini", "gender": "Female"},
            "si-LK-SameeraNeural": {"name": "Sameera", "gender": "Male"},
        }
    },
    "ne": {
        "name": "Nepali",
        "flag": "🇳🇵",
        "voices": {
            "ne-NP-HemkalaNeural": {"name": "Hemkala", "gender": "Female"},
            "ne-NP-SagarNeural": {"name": "Sagar", "gender": "Male"},
        }
    },

    # Pakistani Regional Languages
    "ps": {
        "name": "Pashto",
        "flag": "🇦🇫",
        "voices": {
            "ps-AF-LatifaNeural": {"name": "Latifa", "gender": "Female"},
            "ps-AF-GulNawazNeural": {"name": "Gul Nawaz", "gender": "Male"},
        }
    },
    "fa": {
        "name": "Persian",
        "flag": "🇮🇷",
        "voices": {
            "fa-IR-DilaraNeural": {"name": "Dilara", "gender": "Female"},
            "fa-IR-FaridNeural": {"name": "Farid", "gender": "Male"},
        }
    },
    "pa": {
        "name": "Punjabi",
        "flag": "🇮🇳",
        "voices": {
            "pa-IN-HarpreetNeural": {"name": "Harpreet", "gender": "Female"},
            "pa-IN-GaganNeural": {"name": "Gagan", "gender": "Male"},
        }
    },
    "sd": {
        "name": "Sindhi",
        "flag": "🇵🇰",
        "voices": {
            "sd-PK-AminaNeural": {"name": "Amina", "gender": "Female"},
            "sd-PK-AsharNeural": {"name": "Ashar", "gender": "Male"},
        }
    },
    "bal": {
        "name": "Balochi",
        "flag": "🇵🇰",
        "voices": {
            "bal-PK-BibiNeural": {"name": "Bibi", "gender": "Female"},
            "bal-PK-JamNeural": {"name": "Jam", "gender": "Male"},
        }
    },
    "gjk": {
        "name": "Gojri",
        "flag": "🇵🇰",
        "voices": {
            "gjk-PK-RubinaNeural": {"name": "Rubina", "gender": "Female"},
            "gjk-PK-RashidNeural": {"name": "Rashid", "gender": "Male"},
        }
    },

    # Central Asian Languages
    "uz": {
        "name": "Uzbek",
        "flag": "🇺🇿",
        "voices": {
            "uz-UZ-MadinaNeural": {"name": "Madina", "gender": "Female"},
            "uz-UZ-SardorNeural": {"name": "Sardor", "gender": "Male"},
        }
    },
    "kk": {
        "name": "Kazakh",
        "flag": "🇰🇿",
        "voices": {
            "kk-KZ-AigulNeural": {"name": "Aigul", "gender": "Female"},
            "kk-KZ-DauletNeural": {"name": "Daulet", "gender": "Male"},
        }
    },
    "ky": {
        "name": "Kyrgyz",
        "flag": "🇰🇬",
        "voices": {
            "ky-KG-AidaNeural": {"name": "Aida", "gender": "Female"},
            "ky-KG-TentekNeural": {"name": "Tentek", "gender": "Male"},
        }
    },
    "tg": {
        "name": "Tajik",
        "flag": "🇹🇯",
        "voices": {
            "tg-TJ-HulkarNeural": {"name": "Hulkar", "gender": "Female"},
            "tg-TJ-AbdullohNeural": {"name": "Abdulloh", "gender": "Male"},
        }
    },

    # Arabic Variants
    "ar": {
        "name": "Arabic",
        "flag": "🇦🇪",
        "voices": {
            "ar-SA-ZariyahNeural": {"name": "Zariyah", "gender": "Female", "region": "Saudi"},
            "ar-SA-HamedNeural": {"name": "Hamed", "gender": "Male", "region": "Saudi"},
            "ar-EG-SalmaNeural": {"name": "Salma", "gender": "Female", "region": "Egypt"},
            "ar-EG-ShakirNeural": {"name": "Shakir", "gender": "Male", "region": "Egypt"},
            "ar-AE-FatimaNeural": {"name": "Fatima", "gender": "Female", "region": "UAE"},
            "ar-AE-HamadNeural": {"name": "Hamad", "gender": "Male", "region": "UAE"},
            "ar-JO-SanaNeural": {"name": "Sana", "gender": "Female", "region": "Jordan"},
            "ar-LB-LaylaNeural": {"name": "Layla", "gender": "Female", "region": "Lebanon"},
        }
    },
}

# Voice ID to language mapping for quick lookup
VOICE_TO_LANGUAGE = {}
for lang_code, lang_data in SUPPORTED_LANGUAGES.items():
    for voice_id in lang_data["voices"].keys():
        VOICE_TO_LANGUAGE[voice_id] = lang_code

# Language code normalization
LANGUAGE_ALIASES = {
    "en": "en-US",
    "english": "en-US",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "portuguese": "pt",
    "russian": "ru",
    "chinese": "zh",
    "japanese": "ja",
    "korean": "ko",
    "hindi": "hi",
    "urdu": "ur",
    "bengali": "bn",
    "tamil": "ta",
    "arabic": "ar",
    "turkish": "tr",
    "dutch": "nl",
    "swedish": "sv",
    "norwegian": "no",
    "danish": "da",
    "finnish": "fi",
    "polish": "pl",
    "czech": "cs",
    "hungarian": "hu",
    "greek": "el",
    "thai": "th",
    "vietnamese": "vi",
    "indonesian": "id",
    "malay": "ms",
    "myanmar": "my",
    "khmer": "km",
    "lao": "lo",
    "mongolian": "mn",
    "sinhala": "si",
    "nepali": "ne",
    "pashto": "ps",
    "persian": "fa",
    "punjabi": "pa",
    "sindhi": "sd",
    "balochi": "bal",
    "gojri": "gjk",
    "uzbek": "uz",
    "kazakh": "kk",
    "kyrgyz": "ky",
    "tajik": "tg",
}

def get_language_info(lang_code: str) -> dict:
    """Get language information by code."""
    normalized_code = LANGUAGE_ALIASES.get(lang_code.lower(), lang_code)
    return SUPPORTED_LANGUAGES.get(normalized_code, {})

def get_voice_info(voice_id: str) -> dict:
    """Get voice information by voice ID."""
    lang_code = VOICE_TO_LANGUAGE.get(voice_id)
    if lang_code:
        lang_info = SUPPORTED_LANGUAGES[lang_code]
        voice_info = lang_info["voices"].get(voice_id, {})
        return {
            "language": lang_info["name"],
            "language_code": lang_code,
            "flag": lang_info["flag"],
            **voice_info
        }
    return {}

def is_voice_supported(voice_id: str) -> bool:
    """Check if a voice ID is supported."""
    return voice_id in VOICE_TO_LANGUAGE

def get_default_voice(lang_code: str) -> str:
    """Get the default voice for a language."""
    lang_info = get_language_info(lang_code)
    if lang_info and lang_info.get("voices"):
        return list(lang_info["voices"].keys())[0]
    return "en-US-JennyNeural"  # Fallback to English

def get_all_languages() -> list:
    """Get all supported languages."""
    return [
        {
            "code": code,
            "name": info["name"],
            "flag": info["flag"],
            "voice_count": len(info["voices"])
        }
        for code, info in SUPPORTED_LANGUAGES.items()
    ]

def get_all_voices() -> list:
    """Get all supported voices."""
    voices = []
    for lang_code, lang_info in SUPPORTED_LANGUAGES.items():
        for voice_id, voice_info in lang_info["voices"].items():
            voices.append({
                "voice_id": voice_id,
                "language": lang_info["name"],
                "language_code": lang_code,
                "flag": lang_info["flag"],
                **voice_info
            })
    return voices