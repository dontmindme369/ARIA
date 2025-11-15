#!/usr/bin/env python3
"""
Anchor Mode Selector - Detects query type and selects appropriate reasoning anchor

16 Anchor System:

Core Response Modes (8):
- formal: Academic/research queries across all domains
- casual: Conversational questions
- philosophical: Deep inquiry/existential questions
- analytical: Problem-solving/comparison
- factual: Direct fact lookup
- creative: Brainstorming/exploration
- feedback_correction: Response to negative user feedback
- educational: Teaching, learning, ELI5 explanations

Domain-Specific Technical (8):
- code: Programming/software development
- engineering: Mechanical, electrical, civil, aerospace, chemical, materials
- medical: Clinical, diagnostics, procedures, pharmacology
- science: Physics, chemistry, biology, earth science, astronomy
- mathematics: Pure math, applied math, statistics
- business: Finance, economics, management, strategy
- law: Legal, regulatory, compliance
- technical: General technical/engineering knowledge and troubleshooting
"""

import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class AnchorSelector:
    """Select appropriate reasoning anchor based on query characteristics"""

    def __init__(self, exemplar_path: Optional[Path] = None, feedback_detector=None):
        self.mode_patterns = self._build_patterns()
        self.exemplar_path = exemplar_path
        self.feedback_detector = feedback_detector

    def _build_patterns(self) -> Dict[str, Dict[str, List]]:
        """Define patterns for each mode based on exemplar analysis"""
        return {
            # ============================================================
            # CORE RESPONSE MODES
            # ============================================================
            "formal": {
                "keywords": [
                    "research",
                    "study",
                    "paper",
                    "theory",
                    "hypothesis",
                    "evidence",
                    "methodology",
                    "analysis",
                    "empirical",
                    "peer-reviewed",
                    "citation",
                    "systematic",
                    "meta-analysis",
                    "framework",
                    "paradigm",
                    "literature",
                    "scholarly",
                    "academic",
                    "et al",
                    "according to",
                ],
                "patterns": [
                    r"\bet al\.",
                    r"\b\d{4}\b.*\b(study|research|paper)\b",
                    r"\b(according to|based on).*\b(research|studies|literature)\b",
                    r"\b(theoretical|empirical|methodological)\b",
                    r"\b(peer.?reviewed|systematic review|meta.?analysis)\b",
                ],
                "indicators": [
                    "academic",
                    "scholarly",
                    "rigorous",
                    "formal",
                    "scientific",
                ],
            },
            "casual": {
                "keywords": [
                    "basically",
                    "pretty much",
                    "kind of",
                    "sort of",
                    "tell me",
                    "curious",
                    "wondering",
                    "heard",
                    "friend",
                    "chat",
                    "discuss",
                    "hey",
                    "hi",
                    "hello",
                    "yo",
                    "whats up",
                ],
                "patterns": [
                    r"^(hey|hi|hello|yo|sup)",
                    r"(can you|could you|would you) (just|simply|quickly)",
                    r"(tell me|explain).*(in simple terms|simply|casually)",
                    r"(what's|how's|why's|when's)",
                    r"\b(I'm curious|wondering|heard about)\b",
                ],
                "indicators": ["conversational", "informal", "friendly", "casual"],
            },
            "philosophical": {
                "keywords": [
                    "consciousness",
                    "existence",
                    "reality",
                    "meaning",
                    "purpose",
                    "metaphysics",
                    "epistemology",
                    "ontology",
                    "phenomenology",
                    "existential",
                    "being",
                    "truth",
                    "knowledge",
                    "mind",
                    "free will",
                    "subjective",
                    "objective",
                    "essence",
                    "nature of",
                    "fundamental",
                    "meaning of life",
                    "ethics",
                    "morality",
                    "virtue",
                ],
                "patterns": [
                    r"\b(what is|nature of) (the )?(consciousness|existence|reality|truth|being)\b",
                    r"\b(why do (we|I)|what is the meaning of|what is the purpose)\b",
                    r"\b(philosophical|metaphysical|existential|ontological|epistemological)\b",
                    r"\b(can we (really|truly) know|is it possible to)\b",
                    r"\bmeaning of (life|existence)\b",
                    r"\b(do we have|is there) free will\b",
                ],
                "indicators": [
                    "philosophy",
                    "deep",
                    "existential",
                    "metaphysical",
                    "ethical",
                ],
            },
            "analytical": {
                "keywords": [
                    "analyze",
                    "evaluate",
                    "compare",
                    "assess",
                    "determine",
                    "solve",
                    "approach",
                    "strategy",
                    "optimization",
                    "trade-off",
                    "tradeoff",
                    "trade-offs",
                    "tradeoffs",
                    "decision",
                    "choose",
                    "best",
                    "optimal",
                    "pros and cons",
                    "pros",
                    "cons",
                    "framework",
                    "methodology",
                    "systematic",
                    "structured",
                    "versus",
                    "vs",
                    "difference between",
                    "differences",
                    "advantages",
                    "disadvantages",
                    "better",
                    "worse",
                    "which one",
                    "which is",
                    "should i",
                    "comparison",
                    "contrast",
                ],
                "patterns": [
                    r"\b(how (should|can|do) (I|we) (solve|approach|tackle|handle))",
                    r"\b(what.?s the best (way|approach|method|strategy))",
                    r"\b(compare|contrast|versus|vs\.?|difference between)",
                    r"\b(analyze|evaluate|assess|determine|calculate)",
                    r"\b(pros and cons|advantages and disadvantages|trade.?offs)",
                    r"\b(which (one|option|choice) (should|is better))",
                    r"\b(sql (and|or|vs) nosql|react (and|or|vs) vue)",  # Common comparisons
                ],
                "indicators": [
                    "analysis",
                    "problem-solving",
                    "systematic",
                    "strategic",
                    "comparison",
                ],
            },
            "factual": {
                "keywords": [
                    "what is",
                    "who is",
                    "when did",
                    "where is",
                    "how many",
                    "definition",
                    "date",
                    "fact",
                    "number",
                    "statistic",
                    "data",
                    "capital",
                    "population",
                    "founded",
                    "invented",
                    "discovered",
                    "born",
                    "died",
                    "located",
                ],
                "patterns": [
                    r"^\s*(what|who|when|where|which) (is|are|was|were|did)",
                    r"\b(how (many|much|long|far|high|deep|old))",
                    r"\b(define|definition of|meaning of)\b",
                    r"^\s*(give me|tell me|show me) (the|a) (fact|number|date)",
                    r"\b(what.?s the (capital|population|date|year))",
                ],
                "indicators": ["lookup", "fact", "direct", "quick", "definition"],
            },
            "creative": {
                "keywords": [
                    "brainstorm",
                    "ideas",
                    "creative",
                    "imagine",
                    "what if",
                    "innovative",
                    "novel",
                    "original",
                    "explore",
                    "possibilities",
                    "invent",
                    "design",
                    "create",
                    "generate",
                    "envision",
                    "dream",
                    "story",
                    "scenario",
                    "hypothetical",
                ],
                "patterns": [
                    r"\b(what if|imagine|suppose|let.?s say)",
                    r"\b(brainstorm|generate|come up with|think of) (ideas|ways|approaches)",
                    r"\b(creative|innovative|novel|unique|original|unconventional)",
                    r"\b(how (could|might) we|what are some ways to)",
                    r"\b(explore|envision|dream up|invent|design)\b",
                    r"\b(write|create|make) (a|an) (story|scenario|world)",
                ],
                "indicators": [
                    "ideation",
                    "exploration",
                    "speculation",
                    "imagination",
                    "invention",
                ],
            },
            "feedback_correction": {
                "keywords": [
                    "wrong",
                    "not what i asked",
                    "misunderstood",
                    "not helpful",
                    "too simple",
                    "too complex",
                    "missed the point",
                    "let me clarify",
                    "thats incorrect",
                    "no",
                    "actually",
                    "you didnt",
                    "try again",
                ],
                "patterns": [
                    r"\b(no|nope|wrong|incorrect|not (what|how)|not right)\b",
                    r"\b(you (misunderstood|missed|didnt (get|answer)))",
                    r"\b(too (simple|complex|technical|basic|advanced|vague))",
                    r"\b(not helpful|unhelpful|doesnt help)",
                    r"\b(let me (clarify|rephrase|try again))",
                ],
                "indicators": ["correction", "feedback", "misalignment", "retry"],
            },
            # ============================================================
            # DOMAIN-SPECIFIC TECHNICAL ANCHORS
            # ============================================================
            "code": {
                "keywords": [
                    "code",
                    "function",
                    "class",
                    "method",
                    "api",
                    "debug",
                    "error",
                    "implement",
                    "algorithm",
                    "syntax",
                    "compile",
                    "deploy",
                    "stack",
                    "framework",
                    "library",
                    "package",
                    "module",
                    "variable",
                    "type",
                    "async",
                    "await",
                    "promise",
                    "callback",
                    "memory leak",
                    "binary search",
                    "python",
                    "javascript",
                    "java",
                    "cpp",
                    "rust",
                    "go",
                    "typescript",
                    "react",
                    "django",
                    "flask",
                    "nodejs",
                    "git",
                    "docker",
                    "kubernetes",
                ],
                "patterns": [
                    r"```",  # Code blocks
                    r"\b\w+\([^)]*\)",  # Function calls
                    r"\b[A-Z][a-zA-Z]*Error\b",  # Error types
                    r"\b(def|class|import|from|return|async|await)\b",  # Keywords
                    r"\.(py|js|ts|java|cpp|go|rs|rb|php)\b",  # File extensions
                    r"\b(how (to|do I)) (implement|code|write|fix|debug|build)\b",
                    r"\b(implement|build|create|write) (a|an) \w+ (in|using|with)\b",
                ],
                "indicators": [
                    "programming",
                    "development",
                    "implementation",
                    "coding",
                    "software",
                ],
            },
            "engineering": {
                "keywords": [
                    "mechanical",
                    "electrical",
                    "civil",
                    "aerospace",
                    "chemical",
                    "materials",
                    "stress",
                    "strain",
                    "load",
                    "force",
                    "torque",
                    "circuit",
                    "voltage",
                    "current",
                    "resistance",
                    "capacitor",
                    "bridge",
                    "beam",
                    "structure",
                    "thermodynamics",
                    "fluid dynamics",
                    "heat transfer",
                    "combustion",
                    "alloy",
                    "composite",
                    "strength",
                    "elasticity",
                    "motor",
                    "engine",
                    "turbine",
                    "pump",
                    "valve",
                    "welding",
                    "fabrication",
                    "cad",
                    "fem",
                ],
                "patterns": [
                    r"\b(mechanical|electrical|civil|aerospace|chemical) engineering\b",
                    r"\b(stress|strain|load|force|torque|moment) (analysis|calculation)\b",
                    r"\b(circuit|schematic|wiring|pcb) design\b",
                    r"\b(structural|thermal|fluid|stress) analysis\b",
                    r"\b(turbine|engine|motor|pump|compressor) (design|operation)\b",
                ],
                "indicators": [
                    "engineering",
                    "design",
                    "technical",
                    "manufacturing",
                    "industrial",
                ],
            },
            "medical": {
                "keywords": [
                    "symptom",
                    "diagnosis",
                    "treatment",
                    "disease",
                    "disorder",
                    "syndrome",
                    "patient",
                    "clinical",
                    "medicine",
                    "therapy",
                    "surgery",
                    "procedure",
                    "anatomy",
                    "physiology",
                    "pathology",
                    "pharmacology",
                    "drug",
                    "medication",
                    "prescription",
                    "dosage",
                    "side effect",
                    "contraindication",
                    "vital signs",
                    "blood pressure",
                    "heart rate",
                    "examination",
                    "lab test",
                    "blood test",
                    "diagnostic test",
                    "imaging",
                    "mri",
                    "ct scan",
                    "x-ray",
                    "ultrasound",
                    "biopsy",
                    "cancer",
                    "tumor",
                ],
                "patterns": [
                    r"\b(symptom|sign|presentation) of\b",
                    r"\b(diagnosis|diagnose|differential diagnosis)\b",
                    r"\b(treatment|therapy|medication|drug) for\b",
                    r"\b(clinical|medical|surgical) (approach|management)\b",
                    r"\b(how (to|do you)) (diagnose|treat|manage)\b",
                    r"\b(side effect|adverse effect|contraindication)\b",
                ],
                "indicators": [
                    "medical",
                    "clinical",
                    "healthcare",
                    "diagnostic",
                    "therapeutic",
                ],
            },
            "science": {
                "keywords": [
                    "physics",
                    "chemistry",
                    "biology",
                    "quantum",
                    "relativity",
                    "mechanics",
                    "thermodynamics",
                    "electromagnetism",
                    "atom",
                    "molecule",
                    "reaction",
                    "bond",
                    "orbital",
                    "periodic table",
                    "element",
                    "compound",
                    "cell",
                    "dna",
                    "rna",
                    "protein",
                    "gene",
                    "evolution",
                    "ecology",
                    "photosynthesis",
                    "respiration",
                    "mitosis",
                    "meiosis",
                    "geology",
                    "plate tectonics",
                    "climate",
                    "atmosphere",
                    "astronomy",
                    "universe",
                    "galaxy",
                    "star",
                    "planet",
                    "cosmology",
                    "dark matter",
                    "black hole",
                ],
                "patterns": [
                    r"\b(quantum|classical|relativistic) (mechanics|physics)\b",
                    r"\b(chemical|nuclear) reaction\b",
                    r"\b(molecular|cell|evolutionary) biology\b",
                    r"\b(plate tectonics|climate change|geology)\b",
                    r"\b(astronomical|astrophysical|cosmological)\b",
                    r"\b(how (does|do)) (photosynthesis|evolution|gravity)\b",
                ],
                "indicators": [
                    "scientific",
                    "natural",
                    "physical",
                    "biological",
                    "astronomical",
                ],
            },
            "mathematics": {
                "keywords": [
                    "equation",
                    "formula",
                    "theorem",
                    "proof",
                    "calculus",
                    "derivative",
                    "integral",
                    "limit",
                    "algebra",
                    "geometry",
                    "trigonometry",
                    "statistics",
                    "probability",
                    "matrix",
                    "vector",
                    "function",
                    "graph",
                    "polynomial",
                    "logarithm",
                    "exponential",
                    "differential",
                    "topology",
                    "number theory",
                    "set theory",
                    "logic",
                    "axiom",
                    "lemma",
                    "corollary",
                    "prime",
                    "factorial",
                    "combinatorics",
                    "permutation",
                    "combination",
                ],
                "patterns": [
                    r"\b(solve|calculate|compute|find) (the|a|an)? (equation|integral|derivative)\b",
                    r"\b(prove|proof of|show that) (the )?(theorem|lemma|proposition)\b",
                    r"\b(calculus|algebra|geometry|statistics|probability) (problem|question)\b",
                    r"\b(what is|how (to|do you)) (integrate|differentiate|factor)\b",
                    r"\b(linear|differential|partial differential) equation\b",
                ],
                "indicators": [
                    "mathematical",
                    "quantitative",
                    "numerical",
                    "computational",
                ],
            },
            "business": {
                "keywords": [
                    "business",
                    "finance",
                    "economics",
                    "management",
                    "strategy",
                    "market",
                    "marketing",
                    "revenue",
                    "profit",
                    "loss",
                    "investment",
                    "stock",
                    "bond",
                    "portfolio",
                    "risk",
                    "return",
                    "valuation",
                    "cash flow",
                    "balance sheet",
                    "income statement",
                    "accounting",
                    "audit",
                    "tax",
                    "merger",
                    "acquisition",
                    "startup",
                    "entrepreneur",
                    "venture capital",
                    "supply chain",
                    "operations",
                    "hr",
                    "human resources",
                    "leadership",
                    "negotiation",
                    "pricing",
                ],
                "patterns": [
                    r"\b(business|financial|economic) (model|strategy|analysis)\b",
                    r"\b(how to|should I|should we) (invest|manage|market|price)\b",
                    r"\b(valuation|pricing|forecasting) (method|approach|model)\b",
                    r"\b(startup|company|corporation|firm) (strategy|planning)\b",
                    r"\b(supply chain|operations|logistics|inventory) (management|optimization)\b",
                ],
                "indicators": [
                    "business",
                    "commercial",
                    "financial",
                    "economic",
                    "strategic",
                ],
            },
            "law": {
                "keywords": [
                    "legal",
                    "law",
                    "regulation",
                    "statute",
                    "legislation",
                    "compliance",
                    "contract",
                    "agreement",
                    "clause",
                    "liability",
                    "tort",
                    "negligence",
                    "jurisdiction",
                    "court",
                    "trial",
                    "judge",
                    "jury",
                    "plaintiff",
                    "defendant",
                    "attorney",
                    "lawyer",
                    "lawsuit",
                    "litigation",
                    "settlement",
                    "damages",
                    "intellectual property",
                    "patent",
                    "trademark",
                    "copyright",
                    "license",
                    "constitutional",
                    "criminal",
                    "civil",
                    "corporate",
                    "employment",
                    "gdpr",
                    "hipaa",
                    "regulatory",
                    "compliance",
                    "precedent",
                ],
                "patterns": [
                    r"\b(legal|regulatory|compliance) (requirement|framework|issue)\b",
                    r"\b(contract|agreement|clause|provision) (terms|interpretation)\b",
                    r"\b(what (is|are) the|how (to|do I)) (law|regulation|statute)\b",
                    r"\b(intellectual property|patent|trademark|copyright) (law|protection)\b",
                    r"\b(gdpr|hipaa|sox|fcpa|ada) (compliance|requirement)\b",
                ],
                "indicators": [
                    "legal",
                    "regulatory",
                    "judicial",
                    "statutory",
                    "compliance",
                ],
            },
            "educational": {
                "keywords": [
                    "teach",
                    "learn",
                    "explain",
                    "understand",
                    "eli5",
                    "simple terms",
                    "beginner",
                    "introduction",
                    "basics",
                    "fundamental",
                    "tutorial",
                    "lesson",
                    "course",
                    "concept",
                    "definition",
                    "help me understand",
                    "how does",
                    "what is",
                    "break down",
                    "step by step",
                    "guide",
                    "walkthrough",
                ],
                "patterns": [
                    r"\b(explain|teach|show) (me|how|what|why)\b",
                    r"\b(eli5|explain like|simple terms|layman)\b",
                    r"\b(help me (understand|learn|grasp)|walk me through)\b",
                    r"\b(beginner|introduction|basics|fundamental)\b",
                    r"\b(step.?by.?step|tutorial|lesson|guide)\b",
                ],
                "indicators": [
                    "educational",
                    "pedagogical",
                    "instructional",
                    "tutorial",
                    "foundational",
                ],
            },
            "technical": {
                "keywords": [
                    "technical",
                    "system",
                    "troubleshoot",
                    "debug",
                    "configure",
                    "setup",
                    "installation",
                    "issue",
                    "problem",
                    "fix",
                    "error",
                    "architecture",
                    "infrastructure",
                    "implementation",
                    "specification",
                    "protocol",
                    "standard",
                    "documentation",
                    "manual",
                    "procedure",
                    "workflow",
                    "process",
                    "tcp",
                    "udp",
                    "http",
                    "https",
                    "network",
                    "networking",
                    "transformer",
                    "transformers",
                    "neural network",
                    "deep learning",
                ],
                "patterns": [
                    r"\b(technical|system) (issue|problem|specification|architecture)\b",
                    r"\b(how to|how do I) (setup|configure|install|troubleshoot|debug)\b",
                    r"\b(troubleshooting|debugging|configuration) (steps|guide|help)\b",
                    r"\b(infrastructure|architecture|implementation) (design|approach)\b",
                    r"\b(protocol|standard|specification) (documentation|implementation)\b",
                    r"\b(tcp|udp|http|https) (protocol)?\b",
                    r"\b(difference|compare) (between|vs)? (tcp|udp|http|protocols?)\b",
                    r"\bexplain (how )?(transformers?|neural networks?) (work|function)\b",
                ],
                "indicators": [
                    "technical",
                    "systematic",
                    "procedural",
                    "implementation",
                    "specification",
                ],
            },
        }

    def select_mode(self, query: str) -> str:
        """
        Select the best anchor mode for a query.

        Priority order:
        1. Feedback correction (highest priority)
        2. Domain-specific technical (code, engineering, medical, science, math, business, law, technical)
        3. Core modes (formal, philosophical, analytical, creative, factual, educational, casual)
        """
        query_lower = query.lower()

        # Check for feedback correction first (highest priority)
        if self.feedback_detector:
            feedback_signal = self.feedback_detector.detect(query)
            if feedback_signal.type.value == -1.0 and feedback_signal.should_trigger_correction:
                return "feedback_correction"

        # Score each mode
        scores = {}
        domain_terms_present = False

        for mode, patterns in self.mode_patterns.items():
            score = 0

            # Keyword matching
            for keyword in patterns["keywords"]:
                if keyword in query_lower:
                    score += 1

            # Pattern matching
            for pattern in patterns["patterns"]:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    score += 2  # Patterns worth more than keywords

            if score > 0:
                scores[mode] = score

        # Check if core modes have strong signals (should NOT be overridden by domain)
        core_modes_strong = False
        core_modes = ["analytical", "factual", "technical", "formal", "philosophical"]
        for core_mode in core_modes:
            if core_mode in scores and scores[core_mode] >= 3:  # Strong signal threshold
                core_modes_strong = True
                break
        
        # Boost domain-specific anchors ONLY when:
        # 1. No strong core mode signal, OR
        # 2. Domain has explicit indicators (not just keyword match)
        domain_anchors = [
            "code",
            "engineering", 
            "medical",
            "science",
            "mathematics",
            "business",
            "law",
        ]
        
        # Check for explicit domain indicators
        explicit_domain_context = {
            "code": any(term in query_lower for term in ["code", "programming", "implement", "debug"]),
            "science": any(term in query_lower for term in ["physics", "chemistry", "biology", "scientific"]),
            "engineering": any(term in query_lower for term in ["engineering", "engineer", "circuit", "mechanical"]),
            "medical": any(term in query_lower for term in ["medical", "clinical", "diagnosis", "patient"]),
            "mathematics": any(term in query_lower for term in ["math", "equation", "proof", "theorem"]),
            "business": any(term in query_lower for term in ["business", "finance", "market", "revenue"]),
            "law": any(term in query_lower for term in ["legal", "law", "regulation", "contract"]),
        }
        
        for domain in domain_anchors:
            if domain in scores and scores[domain] > 0:
                # Only boost if explicit context OR no strong core mode
                if explicit_domain_context.get(domain, False):
                    scores[domain] = scores[domain] * 1.3  # Moderate boost for explicit domain
                    domain_terms_present = True
                elif not core_modes_strong:
                    scores[domain] = scores[domain] * 1.1  # Minimal boost if no core mode dominates
                    domain_terms_present = True
                # Otherwise no boost - let core modes win
        
        # Extra boost for code/technical when implementation/debugging terms present
        if any(term in query_lower for term in ["implement", "debug", "error", "fix", "build"]):
            if "code" in scores:
                scores["code"] = scores["code"] + 20  # Increased from 10
            if "technical" in scores:
                scores["technical"] = scores["technical"] + 20  # Increased from 10
        
        # Extra boost for technical when protocol/networking terms present
        if any(term in query_lower for term in ["protocol", "tcp", "udp", "http", "api", "network"]):
            if "technical" in scores:
                scores["technical"] = scores["technical"] + 25  # Increased from 15
        
        # Boost technical when "explain" + technical term
        if "explain" in query_lower:
            if any(term in query_lower for term in ["work", "works", "function", "algorithm"]):
                if "technical" in scores:
                    scores["technical"] = scores["technical"] + 15

        # Boost certain modes for explicit markers
        if any(
            term in query_lower
            for term in ["research", "study", "according to", "et al"]
        ):
            scores["formal"] = scores.get("formal", 0) + 10
            domain_terms_present = True

        if any(
            term in query_lower
            for term in ["consciousness", "existence", "meaning of", "free will"]
        ):
            scores["philosophical"] = scores.get("philosophical", 0) + 10
            domain_terms_present = True

        # Boost analytical for comparison queries (but not if clearly technical/code)
        # Define truly technical/debugging context (NOT framework names like React/Vue)
        technical_context = any(
            term in query_lower for term in 
            ["implement", "debug", "error", "function", "code", "api", "protocol",
             "tcp", "udp", "http", "https", "algorithm", "syntax", "compile",
             "indexerror", "typeerror", "valueerror", "exception"]
        )
        
        # Define explanation/learning context (suggests technical/educational over analytical)
        explanation_context = any(
            term in query_lower for term in
            ["explain", "how does", "how do", "what is", "what are", "describe"]
        )
        
        # Strong analytical patterns (regex-based, always win)
        strong_analytical_patterns = [
            r"\btrade-?offs?\b",
            r"\bpros and cons\b",
            r"\badvantages and disadvantages\b",
        ]
        has_strong_analytical = any(
            re.search(pattern, query_lower) for pattern in strong_analytical_patterns
        )
        
        # Apply analytical boost based on context
        if has_strong_analytical:
            # Strong analytical terms override everything
            scores["analytical"] = scores.get("analytical", 0) + 25
            domain_terms_present = True
        elif technical_context:
            # Technical context: NO analytical boost
            pass
        elif explanation_context:
            # Explanation context: small analytical boost only
            if any(term in query_lower for term in ["compare", "versus", "vs", "difference"]):
                scores["analytical"] = scores.get("analytical", 0) + 5
        else:
            # No special context: normal analytical boost
            if any(term in query_lower for term in 
                   ["compare", "versus", "vs", "difference between", "better", "which"]):
                scores["analytical"] = scores.get("analytical", 0) + 15
                domain_terms_present = True

        # Deprioritize casual if domain terms present
        if domain_terms_present and "casual" in scores:
            scores["casual"] = scores["casual"] * 0.3

        # Get mode with highest score
        if not scores or max(scores.values()) == 0:
            return self._heuristic_selection(query)

        return max(scores.items(), key=lambda x: x[1])[0]

    def _heuristic_selection(self, query: str) -> str:
        """Fallback heuristics when pattern matching doesn't give clear signal"""
        query_lower = query.lower()

        # Domain-specific checks (priority order)

        # Code/programming
        code_terms = [
            "code",
            "implement",
            "debug",
            "error",
            "function",
            "api",
            "javascript",
            "python",
            "react",
            "async",
            "programming",
        ]
        if any(term in query_lower for term in code_terms):
            return "code"

        # Engineering
        eng_terms = ["circuit", "voltage", "stress", "load", "beam", "motor", "turbine"]
        if any(term in query_lower for term in eng_terms):
            return "engineering"

        # Medical
        med_terms = ["symptom", "diagnosis", "treatment", "disease", "drug", "patient"]
        if any(term in query_lower for term in med_terms):
            return "medical"

        # Science
        sci_terms = ["physics", "chemistry", "biology", "quantum", "molecule", "atom"]
        if any(term in query_lower for term in sci_terms):
            return "science"

        # Mathematics
        math_terms = [
            "equation",
            "integral",
            "derivative",
            "proof",
            "theorem",
            "calculate",
        ]
        if any(term in query_lower for term in math_terms):
            return "mathematics"

        # Business
        biz_terms = ["business", "finance", "market", "strategy", "revenue", "profit"]
        if any(term in query_lower for term in biz_terms):
            return "business"

        # Law
        law_terms = ["legal", "law", "regulation", "contract", "compliance", "court"]
        if any(term in query_lower for term in law_terms):
            return "law"

        # Philosophical
        phil_terms = [
            "meaning",
            "existence",
            "consciousness",
            "reality",
            "purpose",
            "free will",
        ]
        if any(term in query_lower for term in phil_terms):
            return "philosophical"

        # Very short queries are usually factual
        if len(query.split()) <= 5:
            return "factual"

        # Questions starting with "why"
        if query_lower.startswith("why"):
            if any(word in query_lower for word in ["should", "do we", "is it"]):
                return "philosophical"
            return "casual"  # Removed educational

        # Questions starting with "how"
        if query_lower.startswith("how"):
            if any(word in query_lower for word in ["solve", "approach", "optimize"]):
                return "analytical"
            return "casual"  # Removed educational

        # Presence of question words
        if any(q in query_lower for q in ["what", "who", "when", "where"]):
            if "meaning of" in query_lower or "nature of" in query_lower:
                return "philosophical"
            return "factual"

        # Default to casual
        return "casual"

    def get_mode_info(self, mode: str) -> Dict[str, str]:
        """Get information about a mode"""
        info = {
            # Core Response Modes
            "formal": {
                "name": "Formal/Academic",
                "description": "Research-grade analysis with citations and scholarly rigor",
                "best_for": "Academic questions, research topics, scholarly analysis",
            },
            "casual": {
                "name": "Casual/Conversational",
                "description": "Friendly dialogue like talking to a knowledgeable friend",
                "best_for": "General questions, casual learning, friendly conversation",
            },
            "philosophical": {
                "name": "Philosophical/Deep Inquiry",
                "description": "Exploration of fundamental questions and multiple perspectives",
                "best_for": "Existential questions, deep conceptual analysis, ethics",
            },
            "analytical": {
                "name": "Analytical/Problem-Solving",
                "description": "Systematic reasoning with structured decomposition",
                "best_for": "Problem-solving, decision-making, strategic analysis, comparisons",
            },
            "factual": {
                "name": "Factual/Direct",
                "description": "Concise direct answers to factual questions",
                "best_for": "Quick facts, definitions, direct information lookup",
            },
            "creative": {
                "name": "Creative/Exploratory",
                "description": "Open-ended ideation and imaginative exploration",
                "best_for": "Brainstorming, innovative thinking, speculation, storytelling",
            },
            "feedback_correction": {
                "name": "Feedback/Correction",
                "description": "Response correction based on negative user feedback",
                "best_for": "When user indicates previous response was wrong or unhelpful",
            },
            # Domain-Specific Technical
            "code": {
                "name": "Code/Programming",
                "description": "Software development with implementation details and working code",
                "best_for": "Programming, debugging, software architecture, technical implementation",
            },
            "engineering": {
                "name": "Engineering",
                "description": "Technical engineering across mechanical, electrical, civil, aerospace, chemical domains",
                "best_for": "Engineering design, analysis, manufacturing, materials, systems",
            },
            "medical": {
                "name": "Medical/Clinical",
                "description": "Clinical knowledge, diagnostics, procedures, pharmacology",
                "best_for": "Medical questions, clinical reasoning, treatment approaches, anatomy",
            },
            "science": {
                "name": "Science",
                "description": "Natural sciences: physics, chemistry, biology, earth science, astronomy",
                "best_for": "Scientific principles, mechanisms, natural phenomena, research",
            },
            "mathematics": {
                "name": "Mathematics",
                "description": "Pure and applied mathematics, statistics, proofs",
                "best_for": "Mathematical problems, proofs, calculations, quantitative analysis",
            },
            "business": {
                "name": "Business/Finance",
                "description": "Business strategy, finance, economics, management",
                "best_for": "Business decisions, financial analysis, market strategy, operations",
            },
            "law": {
                "name": "Law/Regulatory",
                "description": "Legal principles, regulations, compliance, intellectual property",
                "best_for": "Legal questions, regulatory compliance, contracts, IP protection",
            },
        }
        return info.get(mode, {})


def main():
    """Test the anchor selector"""
    import sys

    selector = AnchorSelector()

    # Test queries covering all 14 anchors
    test_queries = [
        # Core modes
        "According to recent research, what are the effects of sleep deprivation?",  # formal
        "Hey, tell me about machine learning",  # casual
        "What is the nature of consciousness?",  # philosophical
        "Compare microservices vs monolithic architecture",  # analytical
        "What is the capital of France?",  # factual
        "What if we could travel faster than light?",  # creative
        "No, that's not what I asked.",  # feedback_correction
        # Domain-specific
        "How do I implement a binary search tree in Python?",  # code
        "What's the stress analysis for a cantilever beam?",  # engineering
        "What are the symptoms of appendicitis?",  # medical
        "Explain quantum entanglement",  # science
        "How do you solve this differential equation?",  # mathematics
        "What's the best pricing strategy for a SaaS product?",  # business
        "What are GDPR compliance requirements?",  # law
    ]

    if len(sys.argv) > 1:
        # Test with command line argument
        query = " ".join(sys.argv[1:])
        mode = selector.select_mode(query)
        info = selector.get_mode_info(mode)
        print(f"Query: {query}")
        print(f"Mode: {info['name']}")
        print(f"Description: {info['description']}")
    else:
        # Test with predefined queries
        print("Testing 14-Anchor System:\n")
        for query in test_queries:
            mode = selector.select_mode(query)
            info = selector.get_mode_info(mode)
            print(f"Query: {query}")
            print(f"  -> Mode: {info['name']}")
            print()


if __name__ == "__main__":
    main()
