import ollama
from pydantic import BaseModel

class Emotion(BaseModel):
    name: str
    score: float

class EmotionResponse(BaseModel):
    emotions: list[Emotion]

res = ollama.chat(
    model="llama3.2-vision",
    messages=[
        {
             'role': 'user',
            'content': """**Analyze facial expressions with high precision** 
            
            For each face in this image, carefully:
            1. Examine key facial features: eyes, eyebrows, mouth, and facial muscle tension
            2. Cross-reference with known emotional patterns from psychological studies
            3. Rate these emotions on a 0-1 scale: happiness, sadness, anger, fear, surprise, disgust, neutral
            4. Calculate confidence scores for each emotion assessment
            5. Identify the dominant emotion for each face
            
            **Critical Requirements:**
            - Consider subtle micro-expressions
            - Account for cultural expression differences
            - Handle occlusions gracefully
            - Provide detailed confidence metrics
            - Include a summary of emotional context
            
            Example Output Structure:
            {
                "faces": [{
                    "bounding_box": [x,y,w,h],
                    "dominant_emotion": "happy",
                    "emotions": [
                        {"name": "happiness", "score": 0.92, "confidence": 0.89},
                        ...
                    ]
                }],
                "analysis_summary": "Primary emotions detected with high confidence..."
            }
            """,
            'images': ['images/picture.jpg']
        }
    ],
    format=EmotionResponse.model_json_schema(),
    options={'temperature': 0}
)

print(res['message']['content'])
