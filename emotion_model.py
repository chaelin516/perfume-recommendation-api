
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import random

# ✅ 감정 레이블
emotion_labels = ["기쁨", "불안", "당황", "분노", "상처", "슬픔", "우울", "흥분"]

# ✅ 감정별 태그 사전
emotion_tag_map = {
    "기쁨": {
        "emotion": ["#joyful", "#bright", "#happy", "#sunny", "#breezy", "#zesty", "#delight", "#uplifted"],
        "scent": ["#citrus", "#lemon", "#grapefruit", "#orange", "#freshfruit"]
    },
    "불안": {
        "emotion": ["#nervous", "#sharp", "#spicy", "#calming", "#restless", "#uneasy"],
        "scent": ["#herbal", "#cool", "#minty", "#softwood", "#lavender", "#softgreen", "#eucalyptus", "#soothingmint", "#camomile"]
    },
    "당황": {
        "emotion": ["#confused", "#unsteady", "#dazed", "#awkward"],
        "scent": ["#soapy", "#neutral", "#coolfloral", "#freshcotton", "#powderclean", "#airy", "#sparkling"]
    },
    "분노": {
        "emotion": ["#angry", "#hot", "#furious", "#intense", "#bold"],
        "scent": ["#burntwood", "#pepper", "#smoky", "#darkleather", "#chili", "#blackpepper"]
    },
    "상처": {
        "emotion": ["#hurt", "#broken", "#fragile", "#dusty"],
        "scent": ["#dryrose", "#woody", "#iris", "#violet", "#quietfloral", "#tea"]
    },
    "슬픔": {
        "emotion": ["#sad", "#deep", "#blue", "#sorrow", "#moody", "#heartache", "#tearful"],
        "scent": ["#deepfloral", "#musk", "#sandalwood", "#darkrose", "#wetwood", "#quietforest", "#oldpaper", "#coldrose"]
    },
    "우울": {
        "emotion": ["#depressed", "#dark", "#gloomy", "#dustyroom", "#softleather", "#cozyblanket", "#foggy", "#lost", "#graymood"],
        "scent": ["#darkmusk", "#vanilla", "#coldfloral", "#smokedwood", "#heavy", "#cloudysmoke", "#cocoa"]
    },
    "흥분": {
        "emotion": ["#excited", "#thrilled", "#buzzing", "#energyup"],
        "scent": ["#sweet", "#fruity", "#peach", "#candy", "#berry", "#tropical", "#juicy", "#candyfloral", "#sweetberry", "#coconut", "#bubblygum", "#juicyapple", "#sugarvanilla", "#pineapple"]
    }
}

# ✅ 모델 로드
emotion_model_name = "nlp04/korean-sentiment-analysis"
emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name)

# ✅ 감정 예측 함수
def predict_emotion_from_diary(diary_text):
    inputs = emotion_tokenizer(diary_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = emotion_model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs).item()
    return emotion_labels[pred]

# ✅ 태그 추천 함수
def recommend_tags_from_emotion(predicted_emotion, diary_text=None):
    base_tags = random.sample(emotion_tag_map[predicted_emotion]["emotion"], 2)
    scent_tag = random.choice(emotion_tag_map[predicted_emotion]["scent"])
    return base_tags + [scent_tag]

# ✅ 예시 실행
if __name__ == "__main__":
    diary_text = "처음엔 좀 낯설었지만 시간이 지나면서 따뜻하고 부드러워졌다. 기본에 충실한 느낌이었고 안정감을 주는 향이었다."
    predicted_emotion = predict_emotion_from_diary(diary_text)
    recommended_tags = recommend_tags_from_emotion(predicted_emotion, diary_text)

    print("🧠 예측 감정:", predicted_emotion)
    print("🏷️ 추천 태그:", recommended_tags)
