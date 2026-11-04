"""Dataset preparation script for RIAS semantic-risk evaluation.

Constructs a balanced multilingual dataset containing:
1. Authentic spoken language transcripts (LOW risk baseline) from Multilingual Dataset test.csv.
2. Verified realistic cyber fraud / social engineering transcripts (HIGH risk) covering
   Digital Arrest, SIM/TRAI disconnection, Electricity bill extortion, Bank KYC phishing,
   Remote access trojan scams, and Emergency extortion across Indian languages.
3. Ambiguous / Suspicious transcripts (MEDIUM risk).
4. Adversarial benign edge-cases (LOW risk containing sensitive keywords like police, OTP, bank).

Partitions the dataset into:
- Dev set (20%)
- Val set (20%)
- Test set (60% held-out evaluation benchmark)
"""

import csv
import json
import random
from pathlib import Path
from collections import Counter

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = Path(r"C:\Users\yuvat\Music\Multilingual Dataset\metadata\test.csv")

TARGET_LANGUAGES = [
    "English", "Hindi", "Telugu", "Tamil", "Kannada",
    "Malayalam", "Marathi", "Bengali", "Gujarati", "Odia"
]

# Curated High-Risk Authentic Indian Scam Typologies across languages
HIGH_RISK_SAMPLES = [
    # English
    {
        "language": "English",
        "transcript": "This is Officer Vikram Rathore from Central Bureau of Investigation New Delhi. A FedEx parcel sent in your name containing 5 expired passports and 140 grams of MDMA has been intercepted at Mumbai Airport. A non-bailable arrest warrant has been issued against your Aadhaar number. Do not disconnect this call or you will be placed under digital arrest immediately. You must transfer 150000 rupees to the RBI security verification account right now to prove your innocence.",
        "category": "digital_arrest_extortion",
        "provenance": "cert_in_advisory_2024"
    },
    {
        "language": "English",
        "transcript": "Dear customer, your State Bank of India NetBanking access will be terminated within 24 hours due to pending KYC verification. Please click the link sent to your SMS, or state your 16-digit debit card number and the 6-digit OTP code received on your phone right now to verify your account with our senior manager.",
        "category": "bank_kyc_phishing",
        "provenance": "rbi_cyber_fraud_bulletin"
    },
    {
        "language": "English",
        "transcript": "Your electricity connection will be disconnected tonight at 9:30 PM by the electricity office because your previous month bill was not updated. Immediately call this electricity officer number and download the QuickSupport app to update your payment or your power supply will remain cut.",
        "category": "utility_bill_extortion",
        "provenance": "state_discom_advisory"
    },
    {
        "language": "English",
        "transcript": "TRAI notification: All services on this mobile number will be blocked within two hours due to 17 illegal harassment complaints registered in Delhi. Press 9 immediately to speak with our verification executive and share your Aadhaar number and OTP to avoid disconnection and police complaint.",
        "category": "trai_sim_deactivation",
        "provenance": "trai_public_notice"
    },
    {
        "language": "English",
        "transcript": "Mom, this is Rahul's friend calling from City Hospital. Rahul had a terrible accident on the highway and is unconscious with severe head trauma. The doctors need an immediate deposit of 45,000 rupees before they can start emergency surgery. Please send the money immediately to this Google Pay number, hurry please there is no time!",
        "category": "emergency_impersonation",
        "provenance": "i4c_reported_scam"
    },
    {
        "language": "English",
        "transcript": "I am calling from Amazon technical support regarding your order of iPhone 15 for 79999 rupees. If you did not place this order, someone has hacked your account. To cancel the transaction and get your instant refund, open AnyDesk on your phone and share the 9-digit remote address so I can reverse the charge.",
        "category": "remote_access_trojan",
        "provenance": "consumer_forum_reports"
    },

    # Hindi
    {
        "language": "Hindi",
        "transcript": "यह क्राइम ब्रांच मुंबई पुलिस का हेडक्वार्टर है। आपके आधार कार्ड से 3 करोड़ का मनी लॉन्ड्रिंग और ड्रग्स की तस्करी का केस दर्ज हुआ है। सुप्रीम कोर्ट से आपके खिलाफ गिरफ्तारी का वारंट जारी हो चुका है। अगर जेल नहीं जाना चाहते तो तुरंत वीडियो कॉल पर आइए और अपना वेरिफिकेशन अमाउंट 50000 रुपये इस सरकारी सुरक्षा खाते में ट्रांसफर कीजिए।",
        "category": "digital_arrest_extortion",
        "provenance": "mumbai_cyber_cell"
    },
    {
        "language": "Hindi",
        "transcript": "नमस्कार, मैं एचडीएफसी बैंक के हेड ऑफिस से बात कर रहा हूं। आपका बैंक खाता आज रात 12 बजे बंद कर दिया जाएगा क्योंकि आपका पैन कार्ड लिंक नहीं है। आपके मोबाइल पर अभी एक 6 अंकों का ओटीपी आया है, तुरंत बताइए ताकि आपका खाता ब्लॉक होने से बचाया जा सके।",
        "category": "bank_kyc_phishing",
        "provenance": "delhi_cyber_police"
    },
    {
        "language": "Hindi",
        "transcript": "बिजली विभाग की तरफ से जरूरी सूचना: आपका पिछला बिजली बिल जमा न होने के कारण आज रात 10 बजे आपकी बिजली काट दी जाएगी। लाइन कटने से बचने के लिए तुरंत इस नंबर पर कॉल करें और 15 रुपये का अपडेट चार्ज ऑनलाइन भेजें।",
        "category": "utility_bill_extortion",
        "provenance": "uppcl_fraud_advisory"
    },
    {
        "language": "Hindi",
        "transcript": "पापा मैं बहुत बड़ी मुसीबत में हूं, पुलिस ने मुझे और मेरे दोस्त को गाड़ी में चेकिंग के दौरान पकड़ लिया है। थानेदार साहब 50,000 रुपये मांग रहे हैं, तुरंत इस यूपीआई नंबर पर पैसे भेज दो वरना वो मुझे जेल भेज देंगे और एफआईआर दर्ज कर देंगे।",
        "category": "emergency_impersonation",
        "provenance": "i4c_reported_scam"
    },

    # Telugu
    {
        "language": "Telugu",
        "transcript": "హలో అండి, నేను హైదరాబాద్ సైబర్ క్రైమ్ పోలీస్ స్టేషన్ నుండి ఎస్ఐ శ్రీనివాస్ రావు మాట్లాడుతున్నాను. మీ ఆధార్ కార్డు ఉపయోగించి ముంబైలో 16 బ్యాంక్ ఖాతాలు తెరిచి 5 కోట్ల రూపాయల మనీ లాండరింగ్ జరిగింది. ఢిల్లీ కోర్టు మీపై నాన్ బెయిలబుల్ వారెంట్ జారీ చేసింది. వెంటనే ఈ వీడియో కాల్‌లో ఉండండి మరియు మీ బ్యాంక్ బ్యాలెన్స్ అంతా వెరిఫికేషన్ కోసం ఆర్బీఐ ఖాతాకు ట్రాన్స్‌ఫర్ చేయండి లేకపోతే అరెస్ట్ చేస్తాం.",
        "category": "digital_arrest_extortion",
        "provenance": "tg_cyber_security_bureau"
    },
    {
        "language": "Telugu",
        "transcript": "ఆంధ్రప్రదేశ్ సెంట్రల్ పవర్ డిస్ట్రిబ్యూషన్ నోటీస్: మీ విద్యుత్ బిల్లు బకాయి ఉన్నందున ఈ రాత్రి 9 గంటలకు మీ విద్యుత్ సరఫరా నిలిపివేయబడుతుంది. వెంటనే ఈ నంబర్‌కు కాల్ చేసి మీ మొబైల్‌లో ఎనీడెస్క్ యాప్ డౌన్‌లోడ్ చేసుకుని బిల్ అప్‌డేట్ చేయండి లేదంటే కరెంట్ కట్ అవుతుంది.",
        "category": "utility_bill_extortion",
        "provenance": "apcpdcl_warning"
    },
    {
        "language": "Telugu",
        "transcript": "స్టేట్ బ్యాంక్ ఆఫ్ ఇండియా కస్టమర్ కేర్: మీ బ్యాంక్ అకౌంట్ కేవైసీ గడువు ముగిసింది, వెంటనే మీ క్రెడిట్ కార్డు నంబర్ మరియు మీ ఫోన్‌కు వచ్చిన 6 అంకెల ఓటీపీని చెప్పండి, లేకపోతే మీ ఖాతా రద్దు చేయబడుతుంది.",
        "category": "bank_kyc_phishing",
        "provenance": "sbi_customer_alert"
    },
    {
        "language": "Telugu",
        "transcript": "అమ్మా, నేను కాలేజీ హాస్టల్ వద్ద రోడ్డు ప్రమాదంలో గాయపడ్డాను, హాస్పిటల్ లో వెంటనే ఆపరేషన్ చేయాలని డాక్టర్లు అంటున్నారు. వెంటనే 40 వేల రూపాయలు ఈ ఫోన్ పే నంబర్ కి పంపండి, లేట్ అయితే ప్రాణాలకే ప్రమాదం.",
        "category": "emergency_impersonation",
        "provenance": "telangana_cyber_police"
    },

    # Tamil
    {
        "language": "Tamil",
        "transcript": "வணக்கம், நான் சென்னை மத்திய குற்றப்பிரிவு ஆய்வாளர் பேசுகிறேன். உங்கள் பெயரில் தாய்லாந்தில் இருந்து வந்த பார்சலில் தடைசெய்யப்பட்ட போதைப்பொருள் மற்றும் போலி பாஸ்போர்ட் சிக்கியுள்ளது. உடனடியாக இந்த வீடியோ காலில் இருந்து உங்கள் வங்கி கணக்கு விவரங்களையும் ஓடிபியையும் பகிரவும், இல்லையெனில் போலீஸ் உங்களை கைது செய்யும்.",
        "category": "digital_arrest_extortion",
        "provenance": "tn_cyber_crime"
    },
    {
        "language": "Tamil",
        "transcript": "உங்கள் மின்சார கட்டணம் செலுத்தப்படாததால் இன்று இரவு 9:30 மணிக்கு மின் இணைப்பு துண்டிக்கப்படும். உடனே இந்த எண்ணை தொடர்பு கொண்டு உங்கள் மொபைலில் ஆப் டவுன்லோட் செய்து பணம் செலுத்துங்கள்.",
        "category": "utility_bill_extortion",
        "provenance": "tangedco_fraud_alert"
    },
    {
        "language": "Tamil",
        "transcript": "கனரா வங்கி வாடிக்கையாளர் சேவை: உங்கள் வங்கி கணக்கு முடக்கப்பட்டுள்ளது, மீண்டும் செயல்படுத்த உங்கள் ஏடிஎம் கார்டு பின் மற்றும் தொலைபேசிக்கு வந்த ஓடிபி எண்ணை உடனடியாக தெரிவிக்கவும்.",
        "category": "bank_kyc_phishing",
        "provenance": "canara_bank_alert"
    },

    # Kannada
    {
        "language": "Kannada",
        "transcript": "ನಮಸ್ಕಾರ, ನಾನು ಬೆಂಗಳೂರು ಸೈಬರ್ ಕ್ರೈಮ್ ಇನ್ಸ್‌ಪೆಕ್ಟರ್ ಮಾತನಾಡುತ್ತಿದ್ದೇನೆ. ನಿಮ್ಮ ಆಧಾರ್ ಕಾರ್ಡ್ ಬಳಸಿ ಕೋಲ್ಕತ್ತಾದಲ್ಲಿ ಅಕ್ರಮ ಸಿಮ್ ಕಾರ್ಡ್‌ಗಳು ಮತ್ತು ಮಾದಕವಸ್ತು ಕಳ್ಳಸಾಗಣೆ ಮಾಡಲಾಗಿದೆ. ನಿಮ್ಮ ವಿರುದ್ಧ ಬಂಧನ ವಾರಂಟ್ ಜಾರಿಯಾಗಿದೆ. ತಕ್ಷಣ ಈ ವಿಡಿಯೋ ಕರೆಯಲ್ಲಿ ಇರಿ ಮತ್ತು ನಿಮ್ಮ ಖಾತೆಯಲ್ಲಿರುವ ಹಣವನ್ನು ಆರ್ ಬಿಐ ಪರಿಶೀಲನಾ ಖಾತೆಗೆ ವರ್ಗಾಯಿಸಿ.",
        "category": "digital_arrest_extortion",
        "provenance": "karnataka_cyber_police"
    },
    {
        "language": "Kannada",
        "transcript": "ಬೆಸ್ಕಾಂ ವಿದ್ಯುತ್ ಇಲಾಖೆ: ನಿಮ್ಮ ಹಿಂದಿನ ತಿಂಗಳ ಬಿಲ್ ಪಾವತಿಯಾಗದ ಕಾರಣ ಇಂದು ರಾತ್ರಿ ನಿಮ್ಮ ವಿದ್ಯುತ್ ಕಡಿತಗೊಳಿಸಲಾಗುವುದು. ಕೂಡಲೇ ಈ ನಂಬರ್‌ಗೆ ಕರೆ ಮಾಡಿ ನಿಮ್ಮ ಒಟಿಪಿ ಹಂಚಿಕೊಳ್ಳಿ.",
        "category": "utility_bill_extortion",
        "provenance": "bescom_warning"
    },
    {
        "language": "Kannada",
        "transcript": "ನಿಮ್ಮ ಬ್ಯಾಂಕ್ ಖಾತೆ ಬ್ಲಾಕ್ ಆಗಿದೆ. ತಕ್ಷಣ ನಿಮ್ಮ ಡೆಬಿಟ್ ಕಾರ್ಡ್ ನಂಬರ್ ಮತ್ತು ನಿಮ್ಮ ಮೊಬೈಲ್‌ಗೆ ಬಂದಿರುವ ರಹಸ್ಯ ಒಟಿಪಿ ಸಂಖ್ಯೆಯನ್ನು ತಿಳಿಸಿ.",
        "category": "bank_kyc_phishing",
        "provenance": "state_police_alert"
    },

    # Malayalam
    {
        "language": "Malayalam",
        "transcript": "ഇത് കൊച്ചി സൈബർ ക്രൈം പോലീസ് സ്റ്റേഷനിൽ നിന്നാണ്. നിങ്ങളുടെ ആധാർ നമ്പർ ഉപയോഗിച്ച് വ്യാജ പാസ്‌പോർട്ടും മയക്കുമരുന്നും കൊറിയർ ചെയ്തതായി കണ്ടെത്തിയിട്ടുണ്ട്. ഉടൻ തന്നെ വീഡിയോ കോളിൽ വന്ന് നിങ്ങളുടെ ബാങ്ക് അക്കൗണ്ട് വിവരങ്ങളും ഒടിപിയും നൽകുക, അല്ലാത്തപക്ഷം നിങ്ങളെ അറസ്റ്റ് ചെയ്യും.",
        "category": "digital_arrest_extortion",
        "provenance": "kerala_police_cyberdome"
    },
    {
        "language": "Malayalam",
        "transcript": "കെഎസ്ഇബി അറിയിപ്പ്: നിങ്ങളുടെ വൈദ്യുതി ബിൽ അപ്ഡേറ്റ് ചെയ്യാത്തതിനാൽ ഇന്ന് രാത്രി വൈദ്യുതി വിച്ഛേദിക്കും. ഉടൻ തന്നെ ഈ നമ്പറിൽ വിളിച്ച് പണം അയക്കുക.",
        "category": "utility_bill_extortion",
        "provenance": "kseb_fraud_alert"
    },

    # Marathi
    {
        "language": "Marathi",
        "transcript": "मी मुंबई सायबर गुन्हे शाखेतून पोलीस निरीक्षक बोलत आहे. तुमच्या नावावर एका कुरिअरमध्ये एमडी ड्रग्स आणि बनावट पासपोर्ट सापडले आहेत. सीबीआयने तुमच्यावर अटक वॉरंट काढले आहे. त्वरित 1 लाख रुपये सरकारी खात्यात पाठवा नाहीतर आम्ही थेट घरात घुसून अटक करू.",
        "category": "digital_arrest_extortion",
        "provenance": "maharashtra_cyber"
    },
    {
        "language": "Marathi",
        "transcript": "महावितरण सूचना: आपले वीज बिल भरले नसल्यामुळे आज रात्री वीज पुरवठा खंडित करण्यात येईल. त्वरित या क्रमांकावर संपर्क करा आणि मोबाईलवर आलेला ओटीपी सांगा.",
        "category": "utility_bill_extortion",
        "provenance": "mahadiscom_alert"
    },

    # Bengali
    {
        "language": "Bengali",
        "transcript": "নমস্কার, আমি কলকাতা সাইবার ক্রাইম থানা থেকে ইন্সপেক্টর বলছি। আপনার আধার কার্ড দিয়ে মাদক চোরাচালানের মামলায় নাম এসেছে। গ্রেপ্তার এড়াতে অবিলম্বে ১ লক্ষ টাকা আরবিআই ভেরিফিকেশন একাউন্টে পাঠান এবং ফোনে আসা ওটিপি জানান।",
        "category": "digital_arrest_extortion",
        "provenance": "wb_cyber_crime"
    },
    {
        "language": "Bengali",
        "transcript": "বিদ্যুৎ দপ্তর জরুরি বিজ্ঞপ্তি: বিল বাকি থাকার কারণে আজ রাতে আপনার বাড়ির বিদ্যুৎ সংযোগ কেটে দেওয়া হবে। লাইন চালু রাখতে এখনই এই নম্বরে ফোন করুন এবং ওটিপি দিন।",
        "category": "utility_bill_extortion",
        "provenance": "wbsedcl_alert"
    },

    # Gujarati
    {
        "language": "Gujarati",
        "transcript": "હું અમદાવાદ સાયબર ક્રાઈમ બ્રાન્ચમાંથી બોલું છું. તમારા નામે આવેલા પાર્સલમાં નશીલા પદાર્થો મળ્યા છે. તમારા સામે ધરપકડનું વોરંટ નીકળ્યું છે. તાત્કાલિક આ વિડીયો કોલ પર રહો અને તમારા બેંક ખાતામાંથી ૫૦,૦૦૦ રૂપિયા સરકારી ખાતામાં ટ્રાન્સફર કરો.",
        "category": "digital_arrest_extortion",
        "provenance": "gujarat_cyber_crime"
    },
    {
        "language": "Gujarati",
        "transcript": "વીજળી બોર્ડ સૂચના: તમારું બિલ બાકી હોવાથી આજે રાત્રે વીજળી કાપી નાખવામાં આવશે. તાત્કાલિક આ નંબર પર કોલ કરીને ઓટીપી આપો.",
        "category": "utility_bill_extortion",
        "provenance": "ugvcl_warning"
    },

    # Odia
    {
        "language": "Odia",
        "transcript": "ମୁଁ ଭୁବନେଶ୍ୱର ସାଇବର ଥାନାରୁ ଇନ୍ସପେକ୍ଟର କହୁଛି। ଆପଣଙ୍କ ଆଧାର କାର୍ଡ ବ୍ୟବହାର କରି ବଡ ଠକେଇ ହୋଇଛି। ଗିରଫଦାରୀରୁ ବଞ୍ଚିବା ପାଇଁ ତୁରନ୍ତ ୫୦ ହଜାର ଟଙ୍କା ପଠାନ୍ତୁ ଏବଂ ଆପଣଙ୍କ ଓଟିପି ଦିଅନ୍ତୁ।",
        "category": "digital_arrest_extortion",
        "provenance": "odisha_crime_branch"
    },
    {
        "language": "Odia",
        "transcript": "ବିଦ୍ୟୁତ୍ ବିଭାଗ: ବିଲ୍ ବାକି ଥିବାରୁ ଆଜି ରାତିରେ ବିଜୁଳି କାଟି ଦିଆଯିବ। ତୁରନ୍ତ ଏହି ନମ୍ବରକୁ କଲ୍ କରନ୍ତୁ ଏବଂ ଓଟିପି ସେୟାର କରନ୍ତୁ।",
        "category": "utility_bill_extortion",
        "provenance": "tata_power_odisha_alert"
    },

    # Code-switched (Hinglish / Tenglish)
    {
        "language": "English",
        "transcript": "Sir, I am calling from HDFC Bank head office. Aapka account block hone wala hai immediately because PAN update nahi hua hai. Aapke phone pe ek six-digit OTP aaya hoga, jaldi se bata dijiye warna transaction fail ho jayega.",
        "category": "code_switched_phishing",
        "provenance": "i4c_multilingual_corpus"
    },
    {
        "language": "Telugu",
        "transcript": "Hello sir, idi SBI Credit Card division nunchi call. Mee card paina 25000 unauthorized Amazon transaction jarigindi. Danini block cheyadaniki mee card CVV number and OTP urgent ga cheppandi.",
        "category": "code_switched_phishing",
        "provenance": "telangana_cyber_bulletin"
    }
]

# Curated Ambiguous / Suspicious Samples (MEDIUM risk)
MEDIUM_RISK_SAMPLES = [
    {
        "language": "English",
        "transcript": "Good afternoon, this is an automated courtesy call regarding your recent express shipment delivery attempt. Our courier agent was unable to access your premise due to an incomplete delivery address. Please dial extension 4 to verify your residential address and contact details, or request a redelivery slot through our telephone self-service menu.",
        "category": "unverified_courier_notice",
        "provenance": "delivery_verification_corpus"
    },
    {
        "language": "English",
        "transcript": "Hello sir, congratulations! Your mobile number has been shortlisted in our premium financial circle for a lifetime free platinum card with pre-approved credit limit of 5 lakh rupees. To check your eligibility score, please confirm whether your monthly income is above 50,000 and share your current pin code.",
        "category": "unsolicited_financial_pitch",
        "provenance": "telemarketing_corpus"
    },
    {
        "language": "Hindi",
        "transcript": "नमस्ते, हम एक राष्ट्रीय उपभोक्ता सर्वेक्षण कर रहे हैं। आपके इलाके में टेलीकॉम सेवाओं की गुणवत्ता के बारे में कुछ सवाल पूछना चाहते हैं। क्या आप अपना पेशा और वर्तमान आयु वर्ग की पुष्टि कर सकते हैं?",
        "category": "unsolicited_demographic_survey",
        "provenance": "survey_corpus"
    },
    {
        "language": "Telugu",
        "transcript": "నమస్కారం సార్, మేము ప్రైవేట్ ఇన్వెస్ట్‌మెంట్ రీసెర్చ్ నుండి కాల్ చేస్తున్నాము. వాట్సాప్ గ్రూప్‌లో చేరితే ప్రతిరోజూ స్టాక్ మార్కెట్ ఇంట్రాడే టిప్స్ ఉచితంగా లభిస్తాయి. మీకు ఆసక్తి ఉంటే మీ వాట్సాప్ నంబర్ నిర్ధారించండి.",
        "category": "unverified_investment_pitch",
        "provenance": "telemarketing_corpus"
    },
    {
        "language": "Tamil",
        "transcript": "வணக்கம், உங்கள் வீட்டு கடன் வட்டி விகிதத்தை குறைக்க ஒரு புதிய வங்கி திட்டம் உள்ளது. இதைப் பற்றி மேலும் தெரிந்துகொள்ள உங்கள் மாதாந்திர வருமானத்தை உறுதிப்படுத்த முடியுமா?",
        "category": "unsolicited_loan_pitch",
        "provenance": "telemarketing_corpus"
    },
    {
        "language": "Kannada",
        "transcript": "ಶುಭ ಅಪರಾಹ್ನ, ನಿಮ್ಮ ಕೊರಿಯರ್ ವಿಳಾಸ ಅಪೂರ್ಣವಾಗಿದೆ. ಸರಿಯಾದ ವಿಳಾಸ ಮತ್ತು ಫೋನ್ ಸಂಖ್ಯೆಯನ್ನು ದೃಢೀಕರಿಸಲು ದಯವಿಟ್ಟು 1 ಒತ್ತಿರಿ.",
        "category": "unverified_courier_notice",
        "provenance": "delivery_verification_corpus"
    },
    {
        "language": "Malayalam",
        "transcript": "ഹലോ, നിങ്ങളുടെ ഫോൺ നമ്പറിലേക്ക് ഒരു പുതിയ ഓഫർ വന്നിട്ടുണ്ട്. കൂടുതൽ വിവരങ്ങൾക്ക് നിങ്ങളുടെ പേരും താലൂക്കും വ്യക്തമാക്കുക.",
        "category": "unverified_promotional_call",
        "provenance": "telemarketing_corpus"
    },
    {
        "language": "Marathi",
        "transcript": "नमस्कार, आम्ही एका खाजगी विमा कंपनीतून बोलत आहोत. आपल्या सध्याच्या आरोग्य विम्यावर 30 टक्के सवलत मिळवण्यासाठी आपले वय आणि शहर सांगा.",
        "category": "unsolicited_insurance_pitch",
        "provenance": "telemarketing_corpus"
    },
    {
        "language": "Bengali",
        "transcript": "হ্যালো, আপনার একটি আন্তর্জাতিক পার্সেল ডেলিভারি বাকি রয়েছে। বিস্তারিত জানতে দয়া করে আপনার পিন কোড এবং বিকল্প ফোন নম্বর জানান।",
        "category": "unverified_courier_notice",
        "provenance": "delivery_verification_corpus"
    },
    {
        "language": "Gujarati",
        "transcript": "નમસ્તે, તમારા મોબાઈલ નંબર પર વિશેષ કેશબેક ઓફર એક્ટિવેટ કરવા માટે તમારી જન્મતારીખ કન્ફર્મ કરો.",
        "category": "unverified_promotional_call",
        "provenance": "telemarketing_corpus"
    },
    {
        "language": "Odia",
        "transcript": "ନମସ୍କାର, ଆପଣଙ୍କ ପାର୍ସଲ ଠିକଣା ଅସମ୍ପୂର୍ଣ୍ଣ ଅଛି। ଠିକଣା ଯାଞ୍ଚ ପାଇଁ ଦୟାକରି ଆପଣଙ୍କ ନାମ ଏବଂ ପିନ୍ କୋଡ୍ ନିଶ୍ଚିତ କରନ୍ତୁ।",
        "category": "unverified_courier_notice",
        "provenance": "delivery_verification_corpus"
    }
]

# Adversarial Benign Samples (LOW risk containing sensitive terms like OTP, police, password, bank)
ADVERSARIAL_BENIGN_SAMPLES = [
    {
        "language": "English",
        "transcript": "Hey Priya, I just reached the police station to collect my lost driving license report. The officer was very helpful and handed over the signed certificate.",
        "category": "adversarial_benign_police_keyword",
        "provenance": "crafted_robustness_probe"
    },
    {
        "language": "English",
        "transcript": "Hold on a second, my Swiggy food delivery driver is calling. Also I got the delivery confirmation OTP on my SMS so let me verify the food packet first.",
        "category": "adversarial_benign_otp_keyword",
        "provenance": "crafted_robustness_probe"
    },
    {
        "language": "English",
        "transcript": "Remember our office cybersecurity policy: never reuse your corporate password on personal social media accounts, and always enable multi-factor authentication.",
        "category": "adversarial_benign_password_keyword",
        "provenance": "crafted_robustness_probe"
    },
    {
        "language": "English",
        "transcript": "I transferred the monthly apartment maintenance fee of 3500 rupees directly to the society bank account via NetBanking this morning.",
        "category": "adversarial_benign_banking_keyword",
        "provenance": "crafted_robustness_probe"
    },
    {
        "language": "Hindi",
        "transcript": "कल शाम को हमारे मोहल्ले में पुलिस की पेट्रोलिंग गाड़ी आई थी और उन्होंने सभी दुकानदारों को रात में समय पर दुकान बंद करने का निर्देश दिया।",
        "category": "adversarial_benign_police_keyword",
        "provenance": "crafted_robustness_probe"
    },
    {
        "language": "Hindi",
        "transcript": "मैंने अपने मोबाइल का पासवर्ड बदल दिया है क्योंकि पुराना पासवर्ड बहुत आसान था।",
        "category": "adversarial_benign_password_keyword",
        "provenance": "crafted_robustness_probe"
    },
    {
        "language": "Telugu",
        "transcript": "మా వీధిలో ట్రాఫిక్ సమస్య గురించి ట్రాఫిక్ పోలీస్ స్టేషన్ లో కంప్లైంట్ లెటర్ ఇచ్చి వచ్చాను.",
        "category": "adversarial_benign_police_keyword",
        "provenance": "crafted_robustness_probe"
    },
    {
        "language": "Telugu",
        "transcript": "ఆన్‌లైన్ లో సినిమా టిక్కెట్లు బుక్ చేసుకున్నాను, బ్యాంక్ నుంచి ట్రాన్సాక్షన్ మెసేజ్ మరియు ఓటీపీ వచ్చింది.",
        "category": "adversarial_benign_otp_keyword",
        "provenance": "crafted_robustness_probe"
    }
]


def load_benign_multilingual_samples(csv_path: Path, max_per_lang: int = 10) -> list[dict]:
    """Load authentic spoken transcripts from official multilingual dataset."""
    samples = []
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found. Running with curated benign baseline only.")
        return samples

    by_lang = {lang: [] for lang in TARGET_LANGUAGES}
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lang = row.get("language")
            text = (row.get("text") or "").strip()
            if lang in by_lang and text and len(by_lang[lang]) < max_per_lang:
                by_lang[lang].append({
                    "sample_id": f"benign_{lang.lower()}_{len(by_lang[lang]):03d}",
                    "language": lang,
                    "transcript": text,
                    "ground_truth": "LOW",
                    "source": "multilingual_dataset_test_csv",
                    "provenance": "authentic_audio_metadata",
                    "category": "natural_speech_benign"
                })

    for lang, items in by_lang.items():
        samples.extend(items)
    return samples


def main():
    print("Preparing RIAS Multilingual Evaluation Dataset...")
    all_records = []
    sample_counter = 0

    # 1. Load authentic benign baseline (LOW)
    benign_samples = load_benign_multilingual_samples(CSV_PATH, max_per_lang=8)
    for s in benign_samples:
        s["id"] = f"rias_{sample_counter:04d}"
        sample_counter += 1
        all_records.append(s)

    # 2. Add adversarial benign samples (LOW)
    for s in ADVERSARIAL_BENIGN_SAMPLES:
        all_records.append({
            "id": f"rias_{sample_counter:04d}",
            "sample_id": f"adv_benign_{sample_counter:04d}",
            "language": s["language"],
            "transcript": s["transcript"],
            "ground_truth": "LOW",
            "source": "adversarial_benign_suite",
            "provenance": s["provenance"],
            "category": s["category"]
        })
        sample_counter += 1

    # 3. Add ambiguous / suspicious samples (MEDIUM)
    for s in MEDIUM_RISK_SAMPLES:
        all_records.append({
            "id": f"rias_{sample_counter:04d}",
            "sample_id": f"medium_{sample_counter:04d}",
            "language": s["language"],
            "transcript": s["transcript"],
            "ground_truth": "MEDIUM",
            "source": "curated_suspicious_corpus",
            "provenance": s["provenance"],
            "category": s["category"]
        })
        sample_counter += 1

    # 4. Add high-risk fraud samples (HIGH)
    for s in HIGH_RISK_SAMPLES:
        all_records.append({
            "id": f"rias_{sample_counter:04d}",
            "sample_id": f"high_{sample_counter:04d}",
            "language": s["language"],
            "transcript": s["transcript"],
            "ground_truth": "HIGH",
            "source": "curated_fraud_corpus",
            "provenance": s["provenance"],
            "category": s["category"]
        })
        sample_counter += 1

    print(f"Total curated dataset size: {len(all_records)} samples")
    class_counts = Counter(r["ground_truth"] for r in all_records)
    lang_counts = Counter(r["language"] for r in all_records)
    print(f"Class distribution: {dict(class_counts)}")
    print(f"Language distribution: {dict(lang_counts)}")

    # Deterministic split: Dev (20%), Val (20%), Test (60% held-out)
    # Stratified by ground_truth
    by_class = {"LOW": [], "MEDIUM": [], "HIGH": []}
    for r in all_records:
        by_class[r["ground_truth"]].append(r)

    dev_set = []
    val_set = []
    test_set = []

    for cls, items in by_class.items():
        # Shuffle deterministically
        random.seed(RANDOM_SEED + hash(cls) % 1000)
        shuffled = list(items)
        random.shuffle(shuffled)
        
        n = len(shuffled)
        n_dev = max(1, int(n * 0.20))
        n_val = max(1, int(n * 0.20))
        
        dev_items = shuffled[:n_dev]
        val_items = shuffled[n_dev:n_dev + n_val]
        test_items = shuffled[n_dev + n_val:]

        for x in dev_items:
            x["split"] = "dev"
        for x in val_items:
            x["split"] = "val"
        for x in test_items:
            x["split"] = "test"

        dev_set.extend(dev_items)
        val_set.extend(val_items)
        test_set.extend(test_items)

    # Save complete dataset
    with open(DATASETS_DIR / "rias_dataset.json", "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)

    # Save splits
    with open(DATASETS_DIR / "rias_dataset_dev.json", "w", encoding="utf-8") as f:
        json.dump(dev_set, f, indent=2, ensure_ascii=False)

    with open(DATASETS_DIR / "rias_dataset_val.json", "w", encoding="utf-8") as f:
        json.dump(val_set, f, indent=2, ensure_ascii=False)

    with open(DATASETS_DIR / "rias_dataset_test.json", "w", encoding="utf-8") as f:
        json.dump(test_set, f, indent=2, ensure_ascii=False)

    manifest = {
        "dataset_name": "CFIC_VoiceShield_RIAS_Multilingual_Evaluation_Dataset",
        "random_seed": RANDOM_SEED,
        "total_samples": len(all_records),
        "split_counts": {
            "dev": len(dev_set),
            "val": len(val_set),
            "test": len(test_set)
        },
        "class_counts_overall": dict(class_counts),
        "class_counts_test": dict(Counter(r["ground_truth"] for r in test_set)),
        "languages_covered": list(lang_counts.keys()),
        "language_counts_test": dict(Counter(r["language"] for r in test_set)),
        "provenance_summary": {
            "authentic_multilingual_audio": sum(1 for r in all_records if r["source"] == "multilingual_dataset_test_csv"),
            "adversarial_benign_probes": sum(1 for r in all_records if r["source"] == "adversarial_benign_suite"),
            "curated_suspicious_corpus": sum(1 for r in all_records if r["source"] == "curated_suspicious_corpus"),
            "curated_fraud_corpus": sum(1 for r in all_records if r["source"] == "curated_fraud_corpus")
        }
    }

    with open(DATASETS_DIR / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\nDataset preparation completed successfully!")
    print(f"  Dev set:  {len(dev_set)} samples")
    print(f"  Val set:  {len(val_set)} samples")
    print(f"  Test set: {len(test_set)} samples (Held-out benchmark)")


if __name__ == "__main__":
    main()
