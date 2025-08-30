"""
Advanced Prompt Engineering API Endpoints
Real implementation with system prompts and safety filters
"""

from fastapi import APIRouter, Form, HTTPException
from typing import Optional, Dict, Any, List
from models.prompt_engine import (
    AdvancedPromptEngine, 
    PersonaType, 
    SafetyLevel,
    prompt_engine
)
from models.qwen_omni import get_qwen_model

router = APIRouter(prefix="/prompt", tags=["prompt-engineering"])

@router.post("/generate")
async def generate_enhanced_prompt(
    topic: str = Form(...),
    persona: str = Form("professional"),
    duration: int = Form(60),
    emotion: str = Form("neutral"),
    context: str = Form(""),
    safety_level: str = Form("moderate"),
    include_examples: bool = Form(True)
):
    """Generate enhanced prompt with advanced engineering"""
    
    try:
        # Convert string enums
        persona_enum = PersonaType(persona)
        safety_enum = SafetyLevel(safety_level)
        
        # Generate dynamic prompt
        enhanced_prompt = prompt_engine.construct_dynamic_prompt(
            user_input=f"Create a {duration}-second presentation about {topic}",
            persona=persona_enum,
            context=context,
            emotion=emotion,
            safety_level=safety_enum,
            include_examples=include_examples,
            max_length=int((duration / 60) * 150)  # ~150 words per minute
        )
        
        # Get Qwen model for generation
        qwen_model = get_qwen_model()
        
        # Generate content using enhanced prompt
        generated_content = qwen_model.generate_text(
            prompt=enhanced_prompt,
            max_length=int((duration / 60) * 150),
            temperature=0.7,
            top_p=0.9
        )
        
        # Validate and enhance output
        validation_result = prompt_engine.validate_and_enhance_output(
            generated_text=generated_content,
            persona=persona_enum,
            safety_level=safety_enum
        )
        
        return {
            "success": True,
            "enhanced_prompt": enhanced_prompt,
            "generated_content": validation_result["moderated_text"],
            "quality_metrics": {
                "quality_score": validation_result["quality_score"],
                "word_count": validation_result["word_count"],
                "estimated_duration": validation_result["estimated_duration"],
                "safety_passed": validation_result["safety_passed"]
            },
            "enhancements": validation_result["enhancements"],
            "persona_applied": persona,
            "safety_level": safety_level
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid parameter: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@router.get("/personas")
async def get_available_personas():
    """Get all available persona types with descriptions"""
    
    personas = {}
    for persona_type in PersonaType:
        system_prompt = prompt_engine.system_prompts[persona_type]
        personas[persona_type.value] = {
            "name": persona_type.value.replace("_", " ").title(),
            "description": system_prompt.role_description,
            "guidelines": system_prompt.behavioral_guidelines,
            "response_format": system_prompt.response_format,
            "constraints": system_prompt.constraints
        }
    
    return {
        "success": True,
        "personas": personas,
        "total_count": len(personas)
    }

@router.get("/examples/{category}")
async def get_few_shot_examples(category: str):
    """Get few-shot learning examples for a category"""
    
    if category not in prompt_engine.few_shot_templates:
        raise HTTPException(status_code=404, detail=f"Category '{category}' not found")
    
    examples = prompt_engine.few_shot_templates[category]
    
    return {
        "success": True,
        "category": category,
        "examples": [
            {
                "input": example.input_text,
                "output": example.expected_output,
                "context": example.context,
                "emotion": example.emotion,
                "style": example.style
            }
            for example in examples
        ],
        "count": len(examples)
    }

@router.get("/examples")
async def list_example_categories():
    """List all available few-shot example categories"""
    
    categories = {}
    for category, examples in prompt_engine.few_shot_templates.items():
        categories[category] = {
            "name": category.replace("_", " ").title(),
            "count": len(examples),
            "description": f"Examples for {category.replace('_', ' ')} scenarios"
        }
    
    return {
        "success": True,
        "categories": categories
    }

@router.post("/validate")
async def validate_content(
    content: str = Form(...),
    safety_level: str = Form("moderate"),
    persona: str = Form("professional")
):
    """Validate content against safety filters and persona guidelines"""
    
    try:
        safety_enum = SafetyLevel(safety_level)
        persona_enum = PersonaType(persona)
        
        # Check safety filters
        passes_safety = prompt_engine._passes_safety_filter(content, safety_enum)
        
        # Apply moderation
        moderated_content = prompt_engine.apply_safety_moderation(content, safety_enum)
        
        # Calculate quality score
        quality_score = prompt_engine._calculate_quality_score(content, persona_enum)
        
        # Get enhancement suggestions
        enhancements = prompt_engine._suggest_enhancements(content, persona_enum)
        
        return {
            "success": True,
            "validation_results": {
                "passes_safety": passes_safety,
                "original_content": content,
                "moderated_content": moderated_content,
                "quality_score": quality_score,
                "word_count": len(content.split()),
                "estimated_duration": len(content.split()) / 2.5,
                "enhancements": enhancements
            },
            "safety_level": safety_level,
            "persona": persona
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid parameter: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@router.post("/enhance")
async def enhance_existing_content(
    content: str = Form(...),
    target_persona: str = Form("professional"),
    target_emotion: str = Form("neutral"),
    target_duration: int = Form(60),
    safety_level: str = Form("moderate")
):
    """Enhance existing content with advanced prompt engineering"""
    
    try:
        persona_enum = PersonaType(target_persona)
        safety_enum = SafetyLevel(safety_level)
        
        # Create enhancement prompt
        enhancement_prompt = f"""
TASK: Enhance the following content to match the specified persona and requirements.

TARGET PERSONA: {prompt_engine.system_prompts[persona_enum].role_description}
BEHAVIORAL GUIDELINES: {'; '.join(prompt_engine.system_prompts[persona_enum].behavioral_guidelines)}
TARGET EMOTION: {target_emotion}
TARGET DURATION: {target_duration} seconds
TARGET WORDS: {int((target_duration / 60) * 150)} words

ORIGINAL CONTENT: {content}

ENHANCED VERSION:
"""
        
        # Get Qwen model for enhancement
        qwen_model = get_qwen_model()
        
        # Generate enhanced content
        enhanced_content = qwen_model.generate_text(
            prompt=enhancement_prompt,
            max_length=int((target_duration / 60) * 150),
            temperature=0.6,
            top_p=0.8
        )
        
        # Validate enhanced content
        validation_result = prompt_engine.validate_and_enhance_output(
            generated_text=enhanced_content,
            persona=persona_enum,
            safety_level=safety_enum
        )
        
        return {
            "success": True,
            "original_content": content,
            "enhanced_content": validation_result["moderated_text"],
            "enhancement_prompt": enhancement_prompt,
            "quality_improvement": {
                "original_quality": prompt_engine._calculate_quality_score(content, persona_enum),
                "enhanced_quality": validation_result["quality_score"],
                "improvement": validation_result["quality_score"] - prompt_engine._calculate_quality_score(content, persona_enum)
            },
            "metrics": {
                "word_count": validation_result["word_count"],
                "estimated_duration": validation_result["estimated_duration"],
                "safety_passed": validation_result["safety_passed"]
            },
            "suggestions": validation_result["enhancements"]
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid parameter: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {str(e)}")

@router.get("/safety-filters")
async def get_safety_filter_info():
    """Get information about available safety filters"""
    
    safety_info = {}
    for level in SafetyLevel:
        config = prompt_engine.safety_filters[level]
        safety_info[level.value] = {
            "name": level.value.title(),
            "blocked_topics": config["blocked_topics"],
            "content_guidelines": config["content_guidelines"],
            "response_modifications": config["response_modifications"]
        }
    
    return {
        "success": True,
        "safety_levels": safety_info,
        "moderation_categories": list(prompt_engine.moderation_patterns.keys())
    }

@router.post("/batch-generate")
async def batch_generate_content(
    topics: List[str] = Form(...),
    persona: str = Form("professional"),
    duration: int = Form(60),
    safety_level: str = Form("moderate")
):
    """Generate content for multiple topics in batch"""
    
    try:
        persona_enum = PersonaType(persona)
        safety_enum = SafetyLevel(safety_level)
        
        results = []
        
        for topic in topics:
            try:
                # Generate prompt for this topic
                enhanced_prompt = prompt_engine.generate_persona_prompt(
                    topic=topic,
                    persona=persona_enum,
                    duration_seconds=duration,
                    emotion="neutral",
                    context=""
                )
                
                # Generate content
                qwen_model = get_qwen_model()
                generated_content = qwen_model.generate_text(
                    prompt=enhanced_prompt,
                    max_length=int((duration / 60) * 150),
                    temperature=0.7
                )
                
                # Validate
                validation_result = prompt_engine.validate_and_enhance_output(
                    generated_text=generated_content,
                    persona=persona_enum,
                    safety_level=safety_enum
                )
                
                results.append({
                    "topic": topic,
                    "success": True,
                    "content": validation_result["moderated_text"],
                    "quality_score": validation_result["quality_score"],
                    "word_count": validation_result["word_count"],
                    "estimated_duration": validation_result["estimated_duration"]
                })
                
            except Exception as e:
                results.append({
                    "topic": topic,
                    "success": False,
                    "error": str(e)
                })
        
        successful_generations = sum(1 for r in results if r["success"])
        
        return {
            "success": True,
            "results": results,
            "summary": {
                "total_topics": len(topics),
                "successful_generations": successful_generations,
                "failed_generations": len(topics) - successful_generations,
                "success_rate": (successful_generations / len(topics)) * 100
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch generation failed: {str(e)}")

@router.post("/custom-persona")
async def create_custom_persona(
    name: str = Form(...),
    role_description: str = Form(...),
    behavioral_guidelines: List[str] = Form(...),
    response_format: str = Form(...),
    constraints: List[str] = Form(...)
):
    """Create a custom persona for prompt engineering"""
    
    try:
        # Create custom persona (in real implementation, this would be saved to database)
        custom_persona = {
            "name": name,
            "role_description": role_description,
            "behavioral_guidelines": behavioral_guidelines,
            "response_format": response_format,
            "constraints": constraints,
            "created_at": "2024-01-01T00:00:00Z"  # Would use actual timestamp
        }
        
        # Generate sample prompt with custom persona
        sample_prompt = f"""
SYSTEM: {role_description}
BEHAVIORAL GUIDELINES: {'; '.join(behavioral_guidelines)}
RESPONSE FORMAT: {response_format}
CONSTRAINTS: {'; '.join(constraints)}

USER INPUT: Create a sample presentation about artificial intelligence
RESPONSE:
"""
        
        return {
            "success": True,
            "custom_persona": custom_persona,
            "sample_prompt": sample_prompt,
            "message": f"Custom persona '{name}' created successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create custom persona: {str(e)}")