import asyncio
import re
from typing import List, Dict, Any, Optional, Tuple
import google.generativeai as genai
import os
from dotenv import load_dotenv
from rag.qdrant.client import QdrantManager
from app.core.session_manager import Session, Answer
from app.utils.api_key_manager import api_key_manager

# Load environment variables
load_dotenv()

class QuestionGenerationService:
    
    def __init__(self):
        self.qdrant_manager = QdrantManager()
        self.reasoning_model = os.getenv("GEMINI_REASONING_MODEL", "gemini-2.5-flash")
        
        print(f"[QuestionService] Initializing with model: {self.reasoning_model}")
        print(f"[QuestionService] API Key Manager active with {len(api_key_manager.api_keys)} keys")

    async def _validate_and_analyze_input(self, text: str) -> Dict[str, Any]:
        """
        Use LLM to validate input and detect language.
        Returns: {"is_valid": bool, "error_message": str, "detected_language": str, "error_type": str}
        """
        try:
            # Only check for completely empty input
            if not text or not text.strip():
                return {
                    "is_valid": False,
                    "detected_language": "english",
                    "error_message": "Please describe your eye symptoms.",
                    "error_type": "invalid_symptoms"
                }
            
            prompt = f'''You are a smart and friendly input validator for a medical eye/vision symptom assessment system.

User Input: "{text}"

TASK: Analyze if this is a VALID medical symptom description about eyes/vision, and detect the language.

INVALID INPUTS (is_valid = false, error_type = "invalid_symptoms"):
- Greetings: "hello", "hi", "kaise ho", "salam", "how are you", "assalam o alaikum", etc.
- Offensive/inappropriate content: slang, insults, vulgar words, offensive phrases
- Random gibberish: "asdfgh", "xyz123", "hi skdfhdskfj 234", "xyz", meaningless text, random characters
- Non-medical questions: "what time is it", "who are you", general chat, jokes
- Empty or very short non-descriptive text (less than 3 meaningful words)
- General statements without eye/vision symptoms
- NON-EYE-RELATED SYMPTOMS: headache only (without eye symptoms), stomach pain, fever, cough, cold, body pain, leg pain, back pain, chest pain, or any symptoms NOT related to eyes/vision

VALID INPUTS (is_valid = true):
- Eye symptoms: pain in eyes, redness, itching, burning, watering, dryness, swelling around eyes
- Vision problems: blurry vision, double vision, floaters, flashes, difficulty seeing, night blindness
- Eye conditions: cataract, glaucoma symptoms, eye infection, pink eye, stye
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
- For non-eye-related symptoms: Politely explain that this system is ONLY for eye/vision problems, and ask them to enter eye-related symptoms
- Always be polite and provide examples of what they can say

Example error messages (generate similar ones dynamically):
- English gibberish/invalid: "Please enter valid eye-related symptoms. For example: 'I have blurry vision', 'My eyes are red and itchy', or 'I have pain in my eyes'. This system is specifically for eye and vision problems."
- English non-eye symptoms: "This system is designed for eye and vision problems only. Please describe your eye-related symptoms. For example: 'I have blurry vision' or 'My eyes are red and itchy'."
- Roman Urdu invalid: "Barah karam aankh se mutaliq alamat bataein. Maslan: 'Meri aankh mein dard hai' ya 'Nazar dhundli hai'. Yeh system sirf aankhon ke maslay ke liye hai."
- Urdu invalid: "براہ کرم صرف آنکھوں سے متعلق علامات بتائیں۔ مثال: 'میری آنکھ میں درد ہے' یا 'نظر دھندلی ہے'۔ یہ سسٹم صرف آنکھوں کے مسائل کے لیے ہے۔"

Respond with ONLY valid JSON (no markdown, no explanation):
{{"is_valid": true/false, "detected_language": "english/urdu/roman_urdu", "error_message": "helpful message if invalid, empty string if valid"}}'''
            
            model = genai.GenerativeModel(self.reasoning_model)
            
            # Use sync version wrapped in asyncio with API key rotation
            def call_gemini():
                return model.generate_content(prompt)
            
            # Use API key manager for retry with rotation
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: api_key_manager.call_sync_with_retry(call_gemini)
            )
            
            import json
            response_text = response.text.strip()
            print(f"LLM Validation Response: {response_text}")
            
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
            # Add error_type for LLM validation results
            if not result.get("is_valid", True):
                result["error_type"] = "invalid_symptoms"
            else:
                result["error_type"] = None
            print(f"LLM Validation Result: {result}")
            return result
            
        except Exception as e:
            import traceback
            print(f"Error in LLM validation: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            print(f"Active API Keys: {len(api_key_manager.api_keys)}")
            print(f"Model: {self.reasoning_model}")
            # If LLM fails, allow the input through (assume valid) - don't block users due to API issues
            print("[QuestionService] LLM validation failed, allowing input through")
            return {
                "is_valid": True,
                "detected_language": "english",
                "error_message": "",
                "error_type": None
            }
    
    async def generate_first_question(self, symptoms: str, user_history: Optional[str] = None) -> Dict[str, Any]:
        try:
            # Use LLM to validate input and detect language
            validation_result = await self._validate_and_analyze_input(symptoms)
            
            if not validation_result.get("is_valid", True):
                return {
                    "error": True,
                    "error_type": validation_result.get("error_type", "invalid_symptoms"),
                    "error_message": validation_result.get("error_message", "Please describe your eye symptoms."),
                    "detected_language": validation_result.get("detected_language", "english"),
                    "question": None,
                    "doctor_guess": None,
                    "rag_context": []
                }
            
            detected_language = validation_result.get("detected_language", "english")
            
            rag_results = await self.qdrant_manager.search(symptoms)
            rag_context = self._format_rag_context(rag_results)
            
            doctor_guess = await self._generate_doctor_guess(symptoms, rag_context, user_history)
            
            prompt = self._create_first_question_prompt(symptoms, rag_context, user_history, detected_language)
            
            question_data = await self._call_gemini_for_question(prompt)
            
            return {
                "question": question_data,
                "doctor_guess": doctor_guess,
                "rag_context": rag_context[:2],
                "detected_language": detected_language
            }
            
        except Exception as e:
            import traceback
            print(f"Error generating first question: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            
            # Check if it's a quota/rate limit error
            error_str = str(e).lower()
            if "429" in error_str or "quota" in error_str or "rate" in error_str or "exhausted" in error_str:
                return {
                    "error": True,
                    "error_type": "quota_exceeded",
                    "error_message": "Service is temporarily busy. Please wait a moment and try again.",
                    "detected_language": "english",
                    "question": None,
                    "doctor_guess": None,
                    "rag_context": []
                }
            
            return {
                "error": True,
                "error_type": "service_error",
                "error_message": "Unable to process your request. Please try again.",
                "detected_language": "english",
                "question": None,
                "doctor_guess": None,
                "rag_context": []
            }
    
    async def generate_followup_question(self, session: Session) -> Dict[str, Any]:

        try:
            conversation_context = self._build_conversation_context(session)
            
            # Use stored language from session, or detect from conversation as fallback
            detected_language = session.detected_language
            if not detected_language:
                validation_result = await self._validate_and_analyze_input(session.initial_symptoms + " " + conversation_context)
                detected_language = validation_result.get("detected_language", "english")
            
            search_query = f"{session.initial_symptoms} {conversation_context}"
            rag_results = await self.qdrant_manager.search(search_query)
            rag_context = self._format_rag_context(rag_results)
            
            prompt = self._create_followup_question_prompt(session, rag_context, detected_language)
            
            question_data = await self._call_gemini_for_question(prompt)
            
            return {
                "question": question_data,
                "rag_context": rag_context[:2],
                "detected_language": detected_language
            }
            
        except Exception as e:
            import traceback
            print(f"Error generating follow-up question: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            
            # Check if it's a quota/rate limit error
            error_str = str(e).lower()
            if "429" in error_str or "quota" in error_str or "rate" in error_str or "exhausted" in error_str:
                return {
                    "error": True,
                    "error_type": "quota_exceeded",
                    "error_message": "Service is temporarily busy. Please wait a moment and try again.",
                    "question": None,
                    "rag_context": []
                }
            
            return {
                "error": True,
                "error_type": "service_error",
                "error_message": "Unable to process your response. Please try again.",
                "question": None,
                "rag_context": []
            }
    
    def _format_rag_context(self, rag_results: List[Dict[str, Any]]) -> str:
        if not rag_results:
            return "No specific medical context found."
        
        context = "Relevant medical information:\n"
        for i, result in enumerate(rag_results[:3], 1): 
            context += f"{i}. {result.get('content', '')} (Relevance: {result.get('relevance_score', 0):.2f})\n"
        
        return context
    
    def _build_conversation_context(self, session: Session) -> str:
        context = ""
        for question, answer in zip(session.questions, session.answers):
            context += f"Q: {question.text} A: {answer.option_text}"
            if answer.custom_text:
                context += f" ({answer.custom_text})"
            context += ". "
        return context
    
    async def _generate_doctor_guess(self, symptoms: str, rag_context: str, user_history: Optional[str] = None) -> str:
        try:
            history_context = f"\n\nPatient's medical history: {user_history}" if user_history else ""
            
            prompt = f"""
            Based on these symptoms: {symptoms}
            
            And this medical context:
            {rag_context}
            {history_context}
            
            What type of doctor should the patient see? Choose from:
            - Ophthalmologist (eye diseases, serious conditions, retinal issues, glaucoma)
            - Optometrist (vision problems, glasses, contacts, routine eye exams)
            - Ocular Surgeon (surgical conditions, cataracts, corneal issues)
            - Optician (glasses fitting, basic vision aids)
            
            Consider:
            - Severity and urgency of symptoms
            - Patient's medical history (if provided)
            - Risk factors from the medical context
            - Previous eye conditions or treatments
            
            Respond with just the doctor type, no explanation.
            """
            
            model = genai.GenerativeModel(self.reasoning_model)
            
            # Use sync version wrapped in asyncio with API key rotation
            def call_gemini():
                return model.generate_content(prompt)
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: api_key_manager.call_sync_with_retry(call_gemini)
            )
            
            doctor_types = ["Ophthalmologist", "Optometrist", "Ocular Surgeon", "Optician"]
            guess = response.text.strip()
            
            for doc_type in doctor_types:
                if doc_type.lower() in guess.lower():
                    return doc_type
            
            return "Ophthalmologist"  
            
        except Exception as e:
            print(f"Error generating doctor guess: {e}")
            return "Ophthalmologist"
    
    def _create_first_question_prompt(self, symptoms: str, rag_context: str, user_history: Optional[str], detected_language: str = "english") -> str:
        history_context = ""
        if user_history:
            history_context = f"\n\nPatient's medical history: {user_history}"
        
        # Language-specific instructions
        language_instructions = {
            "english": "Generate the question and all options in ENGLISH.",
            "urdu": "Generate the question and all options in URDU (using Urdu script). Example: 'آپ کی آنکھ میں درد کب سے ہے؟' with options like 'کل سے', 'ایک ہفتے سے', etc.",
            "roman_urdu": "Generate the question and all options in ROMAN URDU (Urdu written in English letters). Example: 'Aap ki aankh mein dard kab se hai?' with options like 'Kal se', 'Ek haftay se', etc."
        }
        
        lang_instruction = language_instructions.get(detected_language, language_instructions["english"])
        
        return f"""
        You are a medical assistant helping to assess eye conditions. Generate ONE focused question to better understand the patient's condition.
        
        IMPORTANT LANGUAGE INSTRUCTION: {lang_instruction}
        The patient's input language is: {detected_language}
        
        Patient's initial symptoms: {symptoms}
        
        Medical context from database:
        {rag_context}
        {history_context}
        
        Generate a single, clear question with 4-5 specific answer options. The question should:
        1. Be easy to understand for a layperson
        2. Help narrow down the diagnosis based on the symptoms and any medical history provided
        3. Be clinically relevant to the medical context and patient background
        4. Consider the patient's medical history when formulating questions (if provided)
        5. ALWAYS include "Other" as the last option (in the same language: "Other" for English, "دیگر" for Urdu, "Dusra/Other" for Roman Urdu)
        6. GENERATE CONTEXTUALLY RELEVANT OPTIONS based on the specific symptoms - DO NOT use generic/hardcoded options
        
        CRITICAL: Each option must be unique and specifically relevant to the symptoms described. Options should be dynamic and change based on the patient's specific condition.
        
        When medical history is provided, consider:
        - How current symptoms might relate to past conditions
        - Risk factors that could influence the assessment
        - Previous treatments or surgeries that might be relevant
        - Medications that could cause side effects
        - Family history or genetic predisposition
        
        Format your response as JSON:
        {{
            "question_text": "Your question here in {detected_language}",
            "options": ["Option 1 in {detected_language}", "Option 2 in {detected_language}", "Option 3 in {detected_language}", "Option 4 in {detected_language}", "Other/دیگر/Dusra"]
        }}
        
        Make the question specific and tailored to the patient's complete profile.
        """
    
    def _create_followup_question_prompt(self, session: Session, rag_context: str, detected_language: str = "english") -> str:
        conversation = self._build_conversation_context(session)
        
        history_context = ""
        if session.user_history:
            history_context = f"\n\nPatient's medical history: {session.user_history}"
        
        # Language-specific instructions
        language_instructions = {
            "english": "Generate the question and all options in ENGLISH.",
            "urdu": "Generate the question and all options in URDU (using Urdu script). Example: 'کیا آپ کو روشنی سے تکلیف ہوتی ہے؟' with options like 'ہاں، بہت زیادہ', 'ہاں، تھوڑی', etc.",
            "roman_urdu": "Generate the question and all options in ROMAN URDU (Urdu written in English letters). Example: 'Kya aap ko roshni se takleef hoti hai?' with options like 'Haan, bohat zyada', 'Haan, thodi', etc."
        }
        
        lang_instruction = language_instructions.get(detected_language, language_instructions["english"])
        
        return f"""
        You are a medical assistant continuing an eye condition assessment. Generate the NEXT question based on the conversation so far.
        
        IMPORTANT LANGUAGE INSTRUCTION: {lang_instruction}
        The patient's input language is: {detected_language}
        
        Initial symptoms: {session.initial_symptoms}
        
        Conversation so far:
        {conversation}
        
        Medical context from database:
        {rag_context}
        {history_context}
        
        Generate ONE focused follow-up question that:
        1. Builds logically on previous answers
        2. Is easy to understand for a layperson  
        3. Helps further narrow the diagnosis
        4. Uses the medical context and patient background to be specific
        5. Considers the patient's complete medical profile (if provided)
        6. ALWAYS includes "Other" as the last option (in the same language: "Other" for English, "دیگر" for Urdu, "Dusra/Other" for Roman Urdu)
        7. GENERATE CONTEXTUALLY RELEVANT OPTIONS based on the specific symptoms and previous answers - DO NOT use generic/hardcoded options
        
        CRITICAL: Each option must be unique and specifically relevant to the conversation context. Options should be dynamic and logically follow from previous questions and answers.
        
        When medical history is available, consider:
        - How current symptoms might relate to past conditions
        - Progression patterns for chronic conditions
        - Medication-related side effects that might be relevant
        - Complications related to existing health conditions
        - Family history or genetic factors
        
        Format your response as JSON:
        {{
            "question_text": "Your follow-up question here in {detected_language}",
            "options": ["Option 1 in {detected_language}", "Option 2 in {detected_language}", "Option 3 in {detected_language}", "Option 4 in {detected_language}", "Other/دیگر/Dusra"]
        }}
        
        Make sure the question is different from previous questions and progresses the assessment meaningfully.
        """
    
    async def _call_gemini_for_question(self, prompt: str) -> Dict[str, Any]:
        try:
            model = genai.GenerativeModel(self.reasoning_model)
            
            # Use sync version wrapped in asyncio with API key rotation
            def call_gemini():
                return model.generate_content(prompt)
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: api_key_manager.call_sync_with_retry(call_gemini)
            )
            
            import json
            response_text = response.text.strip()
            
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].strip()
            
            return json.loads(response_text)
            
        except Exception as e:
            print(f"Error calling Gemini for question: {e}")
            raise

    async def generate_english_clinical_summary(self, session: Session) -> Dict[str, Any]:
        """
        Generate a comprehensive clinical summary in ENGLISH regardless of the language
        used during the assessment. This is for medical documentation purposes.
        """
        try:
            # Build the conversation context
            conversation_parts = []
            for i, (question, answer) in enumerate(zip(session.questions, session.answers)):
                conversation_parts.append(f"Q{i+1}: {question.text}")
                answer_text = answer.option_text
                if answer.custom_text:
                    answer_text += f" - {answer.custom_text}"
                conversation_parts.append(f"A{i+1}: {answer_text}")
            
            conversation_text = "\n".join(conversation_parts)
            
            prompt = f'''You are a medical documentation assistant. Generate a comprehensive CLINICAL SUMMARY in ENGLISH for medical records.

PATIENT ASSESSMENT DATA:
- Initial Symptoms (may be in any language): {session.initial_symptoms}
- Patient Medical History: {session.user_history or "Not provided"}
- Assessment Language Used: {session.detected_language or "english"}

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

            model = genai.GenerativeModel(self.reasoning_model)
            
            # Use sync version wrapped in asyncio with API key rotation
            def call_gemini():
                return model.generate_content(prompt)
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: api_key_manager.call_sync_with_retry(call_gemini)
            )
            
            import json
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
            return {
                "success": True,
                "summary": result
            }
            
        except Exception as e:
            print(f"Error generating English clinical summary: {e}")
            # Fallback to basic summary
            return {
                "success": False,
                "summary": {
                    "chief_complaint": session.initial_symptoms,
                    "history_of_present_illness": "Assessment completed",
                    "associated_symptoms": [],
                    "medical_history": session.user_history or "None provided",
                    "clinical_assessment": "Requires professional evaluation",
                    "recommended_specialist": session.initial_guess or "Ophthalmologist",
                    "confidence_score": 0.5,
                    "urgency_level": "Medium",
                    "reasoning": "Based on reported symptoms",
                    "recommended_actions": ["Schedule appointment with eye specialist"],
                    "full_summary": f"Patient presented with: {session.initial_symptoms}. Professional eye examination recommended."
                },
                "error": str(e)
            }

question_service = QuestionGenerationService()
