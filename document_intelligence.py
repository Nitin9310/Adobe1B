import json
import re
import logging
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import spacy

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

@dataclass
class DocumentSection:
    content: str
    section_type: str
    page_number: int
    confidence_score: float
    keywords: List[str]
    section_id: str

@dataclass
class Persona:
    role: str
    job_to_be_done: str
    priority_keywords: List[str]
    context_requirements: List[str]

class DocumentIntelligenceSystem:
    def __init__(self):
        self.setup_logging()
        self.load_models()
        self.section_patterns = self.define_section_patterns()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def load_models(self):
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.nlp = spacy.load('en_core_web_sm')
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            self.sentence_model = None
            self.nlp = None
            self.stop_words = set()

    def define_section_patterns(self) -> Dict[str, List[str]]:
        return {
            'executive_summary': [r'executive\s+summary', r'summary', r'overview', r'key\s+points', r'highlights'],
            'methodology': [r'methodology', r'method', r'approach', r'process', r'framework', r'procedure'],
            'results': [r'results', r'findings', r'outcomes', r'conclusion', r'analysis', r'data'],
            'recommendations': [r'recommendations?', r'suggestions?', r'next\s+steps', r'action\s+items?', r'proposals?'],
            'technical_details': [r'technical', r'implementation', r'specifications?', r'requirements?', r'details'],
            'financial': [r'financial', r'budget', r'cost', r'revenue', r'profit', r'investment', r'roi'],
            'legal': [r'legal', r'compliance', r'regulatory', r'terms', r'conditions', r'agreement']
        }

    def extract_text_from_pdf(self, pdf_path: str) -> Dict[int, str]:
        try:
            text_by_page = {}
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    text_by_page[page_num + 1] = text
            return text_by_page
        except Exception as e:
            self.logger.error(f"Error extracting PDF text: {e}")
            return {}

    def preprocess_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\!\?\;\:]', ' ', text)
        return text.strip()

    def identify_section_type(self, text: str) -> Tuple[str, float]:
        text_lower = text.lower()
        best_match = 'general'
        best_score = 0.0
        for section_type, patterns in self.section_patterns.items():
            score = sum(len(re.findall(pattern, text_lower)) for pattern in patterns)
            normalized_score = score / max(len(text.split()), 1)
            if normalized_score > best_score:
                best_score = normalized_score
                best_match = section_type
        return best_match, min(best_score, 1.0)

    def extract_keywords(self, text: str) -> List[str]:
        if self.nlp:
            doc = self.nlp(text)
            keywords = [token.lemma_.lower() for token in doc if token.pos_ in ['NOUN', 'ADJ', 'VERB']
                        and not token.is_stop and len(token.text) > 2 and token.text.lower() not in self.stop_words]
            return list(set(keywords))[:10]
        else:
            words = word_tokenize(text.lower())
            keywords = [word for word in words if word.isalpha() and len(word) > 3 and word not in self.stop_words]
            return list(set(keywords))[:10]

    def segment_document(self, text_by_page: Dict[int, str]) -> List[DocumentSection]:
        sections = []
        section_id = 1
        for page_num, page_text in text_by_page.items():
            if not page_text.strip():
                continue
            paragraphs = [p.strip() for p in page_text.split('\n\n') if p.strip()]
            for paragraph in paragraphs:
                if len(paragraph.split()) < 10:
                    continue
                processed_text = self.preprocess_text(paragraph)
                section_type, confidence = self.identify_section_type(processed_text)
                keywords = self.extract_keywords(processed_text)
                section = DocumentSection(
                    content=processed_text,
                    section_type=section_type,
                    page_number=page_num,
                    confidence_score=confidence,
                    keywords=keywords,
                    section_id=f"section_{section_id}"
                )
                sections.append(section)
                section_id += 1
        return sections

    def calculate_persona_relevance(self, section: DocumentSection, persona: Persona) -> float:
        relevance_score = 0.0
        section_text = section.content.lower()
        job_keywords = persona.job_to_be_done.lower().split()
        for keyword in persona.priority_keywords:
            if keyword.lower() in section_text:
                relevance_score += 0.3
        for keyword in job_keywords:
            if keyword in section_text:
                relevance_score += 0.2
        for req in persona.context_requirements:
            if req.lower() in section_text:
                relevance_score += 0.25
        relevance_score += self.get_section_type_boost(section.section_type, persona.role)
        if self.sentence_model:
            try:
                section_embedding = self.sentence_model.encode([section.content])
                persona_embedding = self.sentence_model.encode([f"{persona.role} {persona.job_to_be_done}"])
                similarity = cosine_similarity(section_embedding, persona_embedding)[0][0]
                relevance_score += similarity * 0.4
            except:
                pass
        return min(relevance_score, 1.0)

    def get_section_type_boost(self, section_type: str, role: str) -> float:
        role_section_mapping = {
            'executive': {'executive_summary': 0.4, 'recommendations': 0.3, 'financial': 0.3, 'results': 0.2},
            'manager': {'recommendations': 0.4, 'results': 0.3, 'methodology': 0.2, 'executive_summary': 0.2},
            'analyst': {'results': 0.4, 'methodology': 0.4, 'technical_details': 0.3, 'executive_summary': 0.1},
            'developer': {'technical_details': 0.5, 'methodology': 0.3, 'recommendations': 0.2},
            'financial_analyst': {'financial': 0.5, 'results': 0.3, 'recommendations': 0.2}
        }
        role_key = role.lower().replace(' ', '_')
        return role_section_mapping.get(role_key, {}).get(section_type, 0.0)

    def rank_sections(self, sections: List[DocumentSection], persona: Persona) -> List[Tuple[DocumentSection, float]]:
        scored_sections = [(section, self.calculate_persona_relevance(section, persona)) for section in sections]
        return sorted(scored_sections, key=lambda x: x[1], reverse=True)

    def generate_output(self, ranked_sections: List[Tuple[DocumentSection, float]], persona: Persona, max_sections: int = 5) -> Dict[str, Any]:
        top_sections = ranked_sections[:max_sections]
        output = {
            "persona": {"role": persona.role, "job_to_be_done": persona.job_to_be_done},
            "analysis_summary": {
                "total_sections_analyzed": len(ranked_sections),
                "sections_returned": len(top_sections),
                "confidence_threshold": 0.3
            },
            "prioritized_sections": []
        }
        for i, (section, score) in enumerate(top_sections):
            output["prioritized_sections"].append({
                "rank": i + 1,
                "section_id": section.section_id,
                "relevance_score": round(score, 3),
                "section_type": section.section_type,
                "page_number": section.page_number,
                "content": section.content[:500] + "..." if len(section.content) > 500 else section.content,
                "key_keywords": section.keywords[:5],
                "confidence": round(section.confidence_score, 3)
            })
        return output

    def process_document(self, pdf_path: str, persona: Persona) -> Dict[str, Any]:
        self.logger.info(f"Processing document: {pdf_path}")
        self.logger.info(f"Persona: {persona.role} - {persona.job_to_be_done}")
        text_by_page = self.extract_text_from_pdf(pdf_path)
        if not text_by_page:
            return {"error": "Failed to extract text from PDF"}
        sections = self.segment_document(text_by_page)
        self.logger.info(f"Extracted {len(sections)} sections")
        ranked_sections = self.rank_sections(sections, persona)
        return self.generate_output(ranked_sections, persona)

# -------------------------
# 🔧 Helper Functions
# -------------------------

def create_sample_personas() -> List[Persona]:
    return [
        Persona(
            role="Executive",
            job_to_be_done="Need high-level insights and strategic recommendations for quarterly board meeting",
            priority_keywords=["strategy", "revenue", "growth", "market", "competitive", "roi"],
            context_requirements=["executive summary", "key metrics", "strategic insights"]
        ),
        Persona(
            role="Financial Analyst",
            job_to_be_done="Analyze budget allocation and cost optimization opportunities",
            priority_keywords=["budget", "cost", "financial", "savings", "investment", "profit"],
            context_requirements=["financial data", "cost analysis", "budget breakdown"]
        ),
        Persona(
            role="Product Manager",
            job_to_be_done="Understand user feedback and feature prioritization for next release",
            priority_keywords=["user", "feature", "feedback", "requirements", "priority", "roadmap"],
            context_requirements=["user insights", "feature analysis", "product roadmap"]
        ),
        Persona(
            role="Technical Lead",
            job_to_be_done="Review technical implementation details and architecture decisions",
            priority_keywords=["technical", "architecture", "implementation", "system", "design", "scalability"],
            context_requirements=["technical specifications", "system design", "implementation details"]
        )
    ]

def main():
    doc_system = DocumentIntelligenceSystem()
    personas = create_sample_personas()
    pdf_path = "sample_document.pdf"  # Replace with actual PDF path
    print("🚀 Adobe Hackathon Round 1B - Document Intelligence System")
    print("=" * 60)
    for persona in personas:
        print(f"\n📋 Processing for {persona.role}")
        print(f"Job to be done: {persona.job_to_be_done}")
        print("-" * 40)
        try:
            result = doc_system.process_document(pdf_path, persona)
            print(json.dumps(result, indent=2))
            output_file = f"output_{persona.role.lower().replace(' ', '_')}.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"✅ Results saved to {output_file}")
        except Exception as e:
            print(f"❌ Error processing document for {persona.role}: {e}")
        print("-" * 40)

if __name__ == "__main__":
    main()
