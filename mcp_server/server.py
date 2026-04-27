import os
import sys
import math
import json
import google.generativeai as genai
from dotenv import load_dotenv
from mcp.server import FastMCP
from typing import List, Optional
from pydantic import BaseModel

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.ophthalmology_tools import (
    Question, UserAnswer, DoctorIdentification, QdrantResult, DoctorSummary,
    InputValidationResult, LLMGeneratedQuestion,
    generate_followup_question, identify_doctor, query_qdrant, generate_doctor_summary
)
from app.utils.api_key_manager import api_key_manager

load_dotenv()

# Configure Gemini using API key manager
reasoning_model = os.getenv("GEMINI_REASONING_MODEL", "gemini-2.5-flash")
print(f"[MCP Server] API Key Manager initialized with {len(api_key_manager.api_keys)} keys")

mcp = FastMCP("OphthalmologyAssistant")


@mcp.tool()
def validate_input(user_input: str) -> dict:
    """
    Validate user input using LLM. Checks if input is a valid eye/vision symptom description,
    detects language (english, urdu, roman_urdu), and returns error message if invalid.
    """
    try:
        prompt = f'''You are a smart and friendly input validator for a medical eye/vision symptom assessment system.

User Input: "{user_input}"

TASK: Analyze if this is a VALID medical symptom description about eyes/vision, and detect the language.

INVALID INPUTS (is_valid = false):
- Greetings: "hello", "hi", "kaise ho", "salam", "how are you", "assalam o alaikum", etc.
- Offensive/inappropriate content: slang, insults, vulgar words, offensive phrases
- Random gibberish: "asdfgh", "xyz123", "hi skdfhdskfj 234", meaningless text, random characters
- Non-medical questions: "what time is it", "who are you", general chat, jokes
- Empty or very short non-descriptive text (less than 3 meaningful words)
- General statements without eye/vision symptoms

VALID INPUTS (is_valid = true):
- Eye symptoms: pain, redness, itching, burning, watering, dryness, swelling
- Vision problems: blurry vision, double vision, floaters, flashes, difficulty seeing
- Roman Urdu symptoms: "aankh mein dard", "nazar kamzor", "aankh laal", "kharish", "dhundla dikhai dena"
- Urdu symptoms: آنکھ میں درد، نظر کمزور، آنکھ لال، خارش، دھندلا دکھائی دینا

LANGUAGE DETECTION (detect from the user's input):
- "english" = English words only
- "roman_urdu" = Urdu/Hindi written in English letters (aankh, mujhe, meri, hai, hain, nazar, etc.)
- "urdu" = Urdu script (آنکھ, میں, ہے, نظر, etc.)

ERROR MESSAGES - Generate a helpful, friendly error message in the SAME LANGUAGE as the user's input:
- For greetings: Acknowledge politely, then ask them to describe their eye symptoms
- For gibberish: Kindly ask them to describe their eye/vision problem clearly
- For non-medical: Explain this is for eye health assessment and ask for symptoms
- Always be polite and provide examples of what they can say

Example error messages (generate similar ones dynamically):
- English greeting: "Hello! I'm here to help with your eye health. Please describe any eye or vision symptoms you're experiencing. For example: 'I have blurry vision' or 'My eyes are red and itchy'."
- Roman Urdu greeting: "Assalam o alaikum! Main aap ki aankh ke maslay mein madad kar sakta hoon. Barah karam apni aankh ya nazar ki takleef bataein. Maslan: 'Meri aankh mein dard hai' ya 'Nazar dhundli hai'."
- Urdu greeting: "السلام علیکم! میں آپ کی آنکھوں کے مسائل میں مدد کر سکتا ہوں۔ براہ کرم اپنی آنکھ یا نظر کی تکلیف بتائیں۔ مثال: 'میری آنکھ میں درد ہے' یا 'نظر دھندلی ہے'۔"

Respond with ONLY valid JSON (no markdown, no explanation):
{{"is_valid": true/false, "detected_language": "english/urdu/roman_urdu", "error_message": "helpful message if invalid, empty string if valid"}}'''

        model = genai.GenerativeModel(reasoning_model)
        
        # Use API key manager for retry with rotation
        def call_gemini():
            return model.generate_content(prompt)
        
        response = api_key_manager.call_sync_with_retry(call_gemini)
        
        response_text = response.text.strip()
        
        # Clean up response to extract JSON
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].strip()
        
        if not response_text.startswith("{"):
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start != -1 and end > start:
                response_text = response_text[start:end]
        
        result = json.loads(response_text)
        return InputValidationResult(**result).dict()
        
    except Exception as e:
        print(f"Error in LLM validation: {e}")
        return InputValidationResult(
            is_valid=True,
            detected_language="english",
            error_message=""
        ).dict()


@mcp.tool()
def generate_llm_question(symptoms: str, previous_answers: Optional[List[dict]] = None, detected_language: str = "english", rag_context: str = "") -> dict:
    """
    Generate a question using LLM in the detected language.
    Questions and options are dynamically generated based on symptoms and context.
    """
    try:
        # Build conversation context
        conversation = ""
        if previous_answers:
            for ans in previous_answers:
                conversation += f"Q: {ans.get('question', '')} A: {ans.get('answer', '')}. "
        
        # Language-specific instructions
        language_instructions = {
            "english": "Generate the question and all options in ENGLISH.",
            "urdu": "Generate the question and all options in URDU (using Urdu script). Example: 'آپ کی آنکھ میں درد کب سے ہے؟' with options like 'کل سے', 'ایک ہفتے سے', etc.",
            "roman_urdu": "Generate the question and all options in ROMAN URDU (Urdu written in English letters). Example: 'Aap ki aankh mein dard kab se hai?' with options like 'Kal se', 'Ek haftay se', etc."
        }
        
        lang_instruction = language_instructions.get(detected_language, language_instructions["english"])
        
        # Determine "Other" option text based on language
        other_text = {"english": "Other", "urdu": "دیگر", "roman_urdu": "Dusra/Other"}.get(detected_language, "Other")
        
        prompt = f'''You are a medical assistant helping to assess eye conditions. Generate ONE focused question to better understand the patient's condition.

IMPORTANT LANGUAGE INSTRUCTION: {lang_instruction}
The patient's input language is: {detected_language}

Patient's symptoms: {symptoms}

{f"Conversation so far: {conversation}" if conversation else ""}

{f"Medical context: {rag_context}" if rag_context else ""}

Generate a single, clear question with 4-5 specific answer options. The question should:
1. Be easy to understand for a layperson
2. Help narrow down the diagnosis based on the symptoms
3. Be clinically relevant to the medical context
4. ALWAYS include "{other_text}" as the last option
5. GENERATE CONTEXTUALLY RELEVANT OPTIONS based on the specific symptoms - DO NOT use generic/hardcoded options

CRITICAL: Each option must be unique and specifically relevant to the symptoms described. Options should be dynamic and change based on the patient's specific condition.

Format your response as JSON:
{{
    "question_text": "Your question here in {detected_language}",
    "options": ["Option 1 in {detected_language}", "Option 2 in {detected_language}", "Option 3 in {detected_language}", "Option 4 in {detected_language}", "{other_text}"]
}}'''

        model = genai.GenerativeModel(reasoning_model)
        
        # Use API key manager for retry with rotation
        def call_gemini():
            return model.generate_content(prompt)
        
        response = api_key_manager.call_sync_with_retry(call_gemini)
        
        response_text = response.text.strip()
        
        # Clean up response to extract JSON
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].strip()
        
        if not response_text.startswith("{"):
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start != -1 and end > start:
                response_text = response_text[start:end]
        
        result = json.loads(response_text)
        result["detected_language"] = detected_language
        return LLMGeneratedQuestion(**result).dict()
        
    except Exception as e:
        print(f"Error generating LLM question: {e}")
        # Fallback question
        return LLMGeneratedQuestion(
            question_text="How long have you been experiencing these symptoms?",
            options=["Less than 24 hours", "1-7 days", "1-4 weeks", "More than a month", "Other"],
            detected_language="english"
        ).dict()


# Define MCP tools
@mcp.tool()
def generate_question(condition: str, previous_answers: Optional[List[dict]] = None) -> dict:
    user_answers = None
    if previous_answers:
        user_answers = [UserAnswer(**answer) for answer in previous_answers]
    
    question = generate_followup_question(condition, user_answers)
    return question.dict()

@mcp.tool()
def identify_eye_doctor(condition: str, answers: List[dict]) -> dict:
    user_answers = [UserAnswer(**answer) for answer in answers]
    
    doctor_id = identify_doctor(condition, user_answers)
    
    return doctor_id.dict()

@mcp.tool()
def search_medical_knowledge(condition: str, answers: List[dict]) -> List[dict]:
    
    user_answers = [UserAnswer(**answer) for answer in answers]
    
    results = query_qdrant(condition, user_answers)
    
    return [result.dict() for result in results]

@mcp.tool()
def create_doctor_summary(condition: str, answers: List[dict], doctor_type: str, qdrant_results: List[dict]) -> dict:
    
    user_answers = [UserAnswer(**answer) for answer in answers]
    
    qdrant_docs = [QdrantResult(**result) for result in qdrant_results]
    
    summary = generate_doctor_summary(condition, user_answers, doctor_type, qdrant_docs)
    
    return summary.dict()


@mcp.tool()
def generate_english_clinical_summary(
    initial_symptoms: str, 
    questions_and_answers: List[dict], 
    detected_language: str = "english",
    medical_history: Optional[str] = None
) -> dict:
    """
    Generate a comprehensive clinical summary in ENGLISH for medical documentation.
    This translates all patient information to English regardless of the assessment language.
    
    Args:
        initial_symptoms: Patient's initial symptoms (may be in any language)
        questions_and_answers: List of {"question": str, "answer": str} dicts
        detected_language: The language used during assessment
        medical_history: Patient's medical history if available
    
    Returns:
        Complete clinical summary with doctor recommendation in English
    """
    try:
        # Build conversation text
        conversation_parts = []
        for i, qa in enumerate(questions_and_answers):
            conversation_parts.append(f"Q{i+1}: {qa.get('question', '')}")
            conversation_parts.append(f"A{i+1}: {qa.get('answer', '')}")
        
        conversation_text = "\n".join(conversation_parts)
        
        prompt = f'''You are a medical documentation assistant. Generate a comprehensive CLINICAL SUMMARY in ENGLISH for medical records.

PATIENT ASSESSMENT DATA:
- Initial Symptoms (may be in any language): {initial_symptoms}
- Patient Medical History: {medical_history or "Not provided"}
- Assessment Language Used: {detected_language}

QUESTIONS AND ANSWERS FROM ASSESSMENT:
{conversation_text}

TASK: Create a professional medical summary in ENGLISH that includes:

1. **Chief Complaint**: Translate and summarize the patient's main symptoms in clear medical English
2. **History of Present Illness**: Duration, onset, progression, severity based on answers
3. **Associated Symptoms**: Any related symptoms mentioned
4. **Relevant Medical History**: If provided
5. **Clinical Assessment**: Brief analysis of the condition
6. **Recommended Specialist**: Choose the most appropriate from:
   - Ophthalmologist (eye diseases, serious conditions, glaucoma, retinal issues)
   - Optometrist (vision problems, refractive errors, routine exams)
   - Ocular Surgeon (surgical conditions, cataracts, trauma)
   - Optician (glasses fitting, lens dispensing)
7. **Urgency Level**: Low / Medium / High / Emergency
8. **Recommended Next Steps**: What the patient should do

Format your response as JSON:
{{
    "chief_complaint": "Patient's main complaint in English",
    "history_of_present_illness": "Detailed history based on Q&A",
    "associated_symptoms": ["symptom1", "symptom2"],
    "medical_history": "Relevant history or 'None provided'",
    "clinical_assessment": "Brief clinical impression",
    "recommended_specialist": "Ophthalmologist/Optometrist/Ocular Surgeon/Optician",
    "confidence_score": 0.85,
    "urgency_level": "Low/Medium/High/Emergency",
    "reasoning": "Why this specialist is recommended",
    "recommended_actions": ["action1", "action2"],
    "full_summary": "A complete narrative summary paragraph in professional medical English documenting everything"
}}'''

        model = genai.GenerativeModel(reasoning_model)
        response = model.generate_content(prompt)
        
        response_text = response.text.strip()
        
        # Clean up response to extract JSON
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].strip()
        
        if not response_text.startswith("{"):
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start != -1 and end > start:
                response_text = response_text[start:end]
        
        result = json.loads(response_text)
        return {"success": True, "summary": result}
        
    except Exception as e:
        print(f"Error generating English clinical summary: {e}")
        return {
            "success": False,
            "summary": {
                "chief_complaint": initial_symptoms,
                "full_summary": f"Patient presented with: {initial_symptoms}. Professional eye examination recommended.",
                "recommended_specialist": "Ophthalmologist",
                "urgency_level": "Medium"
            },
            "error": str(e)
        }

if __name__ == "__main__":
    mcp.run()