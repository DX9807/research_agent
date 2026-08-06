import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
import re

# Updated imports for LangChain 1.3+
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool, Tool
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.callbacks import CallbackManager
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    JSONLoader,
    UnstructuredHTMLLoader
)
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.document_loaders import UnstructuredMarkdownLoader
import wikipedia

wikipedia.set_lang('en')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==========================================================
# ENUMS AND CONSTANTS
# ==========================================================

class ResearchDomain(Enum):
    ACADEMIC = "academic"
    BUSINESS = "business"
    PROGRAMMING = "programming"
    CYBERSECURITY = "cybersecurity"
    HEALTHCARE = "healthcare"
    FINANCE = "finance"
    SCIENTIFIC = "scientific"
    GENERAL = "general_knowledge"
    CURRENT_EVENTS = "current_events"
    SOFTWARE_ARCHITECTURE = "software_architecture"
    LEGAL = "legal"
    DATA_ANALYSIS = "data_analysis"
    DEVOPS = "devops"
    CLOUD = "cloud"
    MACHINE_LEARNING = "machine_learning"
    AI = "ai"
    NETWORKING = "networking"
    HARDWARE = "hardware"
    RESEARCH_PAPER = "research_paper"
    PRODUCT_COMPARISON = "product_comparison"
    MARKET_INTELLIGENCE = "market_intelligence"
    COMPETITOR_ANALYSIS = "competitor_analysis"
    THREAT_INTELLIGENCE = "threat_intelligence"
    VULNERABILITY_RESEARCH = "vulnerability_research"

class ConfidenceLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ToolType(Enum):
    WEB_SEARCH = "web_search"
    SCHOLAR_SEARCH = "scholar_search"
    ARXIV = "arxiv"
    PUBMED = "pubmed"
    WIKIPEDIA = "wikipedia"
    GITHUB = "github"
    STACKOVERFLOW = "stackoverflow"
    DOCUMENTATION = "documentation"
    CODE_INTERPRETER = "code_interpreter"
    PDF_LOADER = "pdf_loader"
    HTML_LOADER = "html_loader"
    MARKDOWN_LOADER = "markdown_loader"
    CSV_LOADER = "csv_loader"
    JSON_LOADER = "json_loader"
    BROWSER = "browser"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    LOCAL_DATABASE = "local_database"
    MEMORY = "memory"
    CVE_SEARCH = "cve_search"
    MITRE = "mitre"
    NVD = "nvd"


# ==========================================================
# DATA STRUCTURES
# ==========================================================

@dataclass
class ResearchFinding:
    """
    Represents a research finding with metadata.
    """
    claim: str
    evidence: str
    source: str
    confidence: ConfidenceLevel
    timestamp: datetime = field(default_factory=datetime.now)
    contradictions: List[str] = field(default_factory=list)
    supporting_sources: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "claim": self.claim,
            "evidence": self.evidence,
            "source": self.source,
            "confidence": self.confidence.value,
            "timestamp": self.timestamp.isoformat(),
            "contradictions": self.contradictions,
            "supporting_sources": self.supporting_sources
        }


@dataclass
class ResearchPlan:
    """
    Structured research plan.
    """
    objective: str
    subtasks: List[str]
    required_tools: List[ToolType]
    expected_outputs: List[str]
    dependencies: List[str]
    estimated_confidence: ConfidenceLevel
    priority: int = 1
    status: str = "pending"  # pending, running, completed, failed


@dataclass
class SourceMetadata:
    """
    Metadata for research sources.
    """
    url: str
    title: str
    author: Optional[str] = None
    publication_date: Optional[datetime] = None
    publisher: Optional[str] = None
    domain: Optional[str] = None
    credibility_score: float = 0.0
    verified: bool = False
    accessed_date: datetime = field(default_factory=datetime.now)


# ==========================================================
# MEMORY MANAGEMENT
# ==========================================================

class ResearchMemory:
    """
    Manages all memory components for the agent.
    """
    
    def __init__(self):
        # Use MemorySaver from langgraph for checkpointing
        self.checkpoint_memory = MemorySaver()
        self.research_memory: List[ResearchFinding] = []
        self.intermediate_findings: Dict[str, List[ResearchFinding]] = {}
        self.failed_attempts: List[Dict] = []
        self.extracted_facts: Dict[str, Any] = {}
        self.source_metadata: Dict[str, SourceMetadata] = {}
        self.tool_outputs: Dict[str, Any] = {}
        self.user_preferences: Dict[str, Any] = {}
        self.conversation_history: List[Dict] = []
    
    def add_finding(self, finding: ResearchFinding, context: str = None) -> None:
        """
        Add a research finding to memory.
        """
        self.research_memory.append(finding)
        if context:
            if context not in self.intermediate_findings:
                self.intermediate_findings[context] = []
            self.intermediate_findings[context].append(finding)
    
    def get_findings_by_context(self, context: str) -> List[ResearchFinding]:
        """
        Retrieve findings by context.
        """
        return self.intermediate_findings.get(context, [])
    
    def get_all_findings(self) -> List[ResearchFinding]:
        """
        Retrieve all findings.
        """
        return self.research_memory
    
    def get_high_confidence_findings(self) -> List[ResearchFinding]:
        """
        Retrieve only high-confidence findings.
        """
        return [f for f in self.research_memory if f.confidence == ConfidenceLevel.HIGH]
    
    def add_failed_attempt(self, attempt: Dict) -> None:
        """
        Log a failed attempt.
        """
        self.failed_attempts.append(attempt)
    
    def add_source(self, url: str, metadata: SourceMetadata) -> None:
        """
        Add source metadata.
        """
        self.source_metadata[url] = metadata
    
    def get_source(self, url: str) -> Optional[SourceMetadata]:
        """
        Get source metadata.
        """
        return self.source_metadata.get(url)
    
    def add_conversation_message(self, role: str, content: str) -> None:
        """
        Add a message to conversation history.
        """
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })


# ==========================================================
# TOOL DEFINITIONS
# ==========================================================

class ToolRegistry:
    """
    Registry for managing and selecting tools.
    """
    
    def __init__(self, memory: ResearchMemory):
        self.memory = memory
        self.tools = {}
        self._initialize_tools()
    
    def _initialize_tools(self) -> None:
        """
        Initialize all available tools.
        """
        # Initialize search tools - using invoke method (new in 1.3+)
        search = DuckDuckGoSearchRun()
        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())
        
        # Create tools with proper invocation methods
        self.tools = {
            ToolType.WEB_SEARCH: Tool(
                name="WebSearch",
                description="Search the web for current information. Input: search query string.",
                func=self._create_tool_function(search.invoke, "WebSearch")
            ),
            ToolType.WIKIPEDIA: Tool(
                name="Wikipedia",
                description="Search Wikipedia for encyclopedic information. Input: search query string.",
                func=self._create_tool_function(wikipedia.invoke, "Wikipedia")
            ),
            ToolType.ARXIV: Tool(
                name="Arxiv",
                description="Search academic papers on arXiv. Input: search query string.",
                func=self._create_tool_function(arxiv.invoke, "Arxiv")
            ),
        }
    
    def _create_tool_function(self, func, tool_name: str):
        """
        Create a wrapper function for tool invocation.
        """
        def wrapper(query: str) -> str:
            try:
                result = func(query)
                # Log the tool usage
                logger.info(f"Tool {tool_name} executed successfully")
                return result if result else f"No results found from {tool_name}"
            except Exception as e:
                logger.error(f"Error in tool {tool_name}: {e}")
                return f"Error: {str(e)}"
        return wrapper
    
    def get_tool(self, tool_type: ToolType) -> Optional[Tool]:
        """
        Get a specific tool.
        """
        return self.tools.get(tool_type)
    
    def select_tools_for_task(self, plan: ResearchPlan) -> List[Tool]:
        """
        Select appropriate tools for a research plan.
        """
        selected = []
        for tool_type in plan.required_tools:
            if tool_type in self.tools:
                selected.append(self.tools[tool_type])
        return selected
    
    def get_parallel_tools(self, plan: ResearchPlan) -> List[List[Tool]]:
        """
        Group independent tools for parallel execution.
        """
        parallel_groups = []
        current_group = []
        
        for tool_type in plan.required_tools:
            if self._can_run_parallel(tool_type, current_group):
                current_group.append(self.tools.get(tool_type))
            else:
                if current_group:
                    parallel_groups.append([t for t in current_group if t])
                current_group = [self.tools.get(tool_type)]
        
        if current_group:
            parallel_groups.append([t for t in current_group if t])
        
        return parallel_groups
    
    def _can_run_parallel(self, tool_type: ToolType, current_group: List) -> bool:
        """
        Determine if a tool can run in parallel with others.
        """
        independent_pairs = [
            (ToolType.WEB_SEARCH, ToolType.WIKIPEDIA),
            (ToolType.ARXIV, ToolType.PUBMED),
            (ToolType.GITHUB, ToolType.STACKOVERFLOW),
        ]
        
        for existing_tool in current_group:
            if existing_tool:
                existing_type = self._get_tool_type(existing_tool)
                if existing_type:
                    if (existing_type, tool_type) in independent_pairs or \
                       (tool_type, existing_type) in independent_pairs:
                        return True
                    if existing_type == tool_type:
                        return False
        
        return True
    
    def _get_tool_type(self, tool: Tool) -> Optional[ToolType]:
        """
        Get the enum type of a tool.
        """
        for tool_type, registered_tool in self.tools.items():
            if registered_tool == tool:
                return tool_type
        return None
    
    def get_document_loaders(self) -> Dict[str, Any]:
        """
        Get document loading tools.
        """
        return {
            "pdf": PyPDFLoader,
            "csv": CSVLoader,
            "json": JSONLoader,
            "html": UnstructuredHTMLLoader,
            "markdown": UnstructuredMarkdownLoader,
            "text": TextLoader,
        }


# ==========================================================
# RESEARCH PLANNER
# ==========================================================

class ResearchPlanner:
    """
    Creates and manages research plans.
    """
    
    def __init__(self, memory: ResearchMemory):
        self.memory = memory
        self.plans: List[ResearchPlan] = []
        self.current_plan: Optional[ResearchPlan] = None
    
    def create_plan(self, objective: str, domain: ResearchDomain) -> ResearchPlan:
        """
        Create a research plan based on the objective and domain.
        """
        subtasks = self._generate_subtasks(objective, domain)
        required_tools = self._determine_required_tools(domain)
        expected_outputs = self._determine_expected_outputs(domain)
        
        plan = ResearchPlan(
            objective=objective,
            subtasks=subtasks,
            required_tools=required_tools,
            expected_outputs=expected_outputs,
            dependencies=[],
            estimated_confidence=ConfidenceLevel.MEDIUM
        )
        
        self.plans.append(plan)
        self.current_plan = plan
        return plan
    
    def _generate_subtasks(self, objective: str, domain: ResearchDomain) -> List[str]:
        """
        Generate logical subtasks based on domain and objective.
        """
        base_subtasks = [
            "Define research scope and boundaries",
            "Identify key concepts and terminology",
            "Gather primary sources and evidence",
            "Analyze and synthesize findings",
            "Verify evidence and source credibility",
            "Document conclusions and recommendations"
        ]
        
        domain_specific = {
            ResearchDomain.CYBERSECURITY: [
                "Identify attack vectors and vulnerabilities",
                "Analyze threat intelligence sources",
                "Review CVE and MITRE databases"
            ],
            ResearchDomain.ACADEMIC: [
                "Search peer-reviewed literature",
                "Compare methodologies",
                "Evaluate statistical significance"
            ],
            ResearchDomain.PROGRAMMING: [
                "Analyze code structure and dependencies",
                "Review API documentation",
                "Check GitHub repositories"
            ]
        }
        
        specific_subtasks = domain_specific.get(domain, [])
        return base_subtasks + specific_subtasks
    
    def _determine_required_tools(self, domain: ResearchDomain) -> List[ToolType]:
        """
        Determine required tools based on domain.
        """
        tool_map = {
            ResearchDomain.ACADEMIC: [ToolType.ARXIV, ToolType.SCHOLAR_SEARCH, ToolType.WIKIPEDIA],
            ResearchDomain.BUSINESS: [ToolType.WEB_SEARCH, ToolType.DOCUMENTATION],
            ResearchDomain.PROGRAMMING: [ToolType.GITHUB, ToolType.STACKOVERFLOW, ToolType.DOCUMENTATION],
            ResearchDomain.CYBERSECURITY: [ToolType.WEB_SEARCH, ToolType.CVE_SEARCH, ToolType.MITRE],
            ResearchDomain.HEALTHCARE: [ToolType.PUBMED, ToolType.WEB_SEARCH],
            ResearchDomain.FINANCE: [ToolType.WEB_SEARCH, ToolType.DOCUMENTATION],
            ResearchDomain.SCIENTIFIC: [ToolType.ARXIV, ToolType.WIKIPEDIA],
            ResearchDomain.GENERAL: [ToolType.WEB_SEARCH, ToolType.WIKIPEDIA],
        }
        return tool_map.get(domain, [ToolType.WEB_SEARCH])
    
    def _determine_expected_outputs(self, domain: ResearchDomain) -> List[str]:
        """
        Determine expected output types.
        """
        base_outputs = ["Executive Summary", "Evidence Summary", "Citations"]
        
        domain_outputs = {
            ResearchDomain.CYBERSECURITY: ["Vulnerability Assessment", "Mitigation Recommendations"],
            ResearchDomain.ACADEMIC: ["Literature Review", "Methodology Comparison"],
            ResearchDomain.PROGRAMMING: ["Code Analysis", "Architecture Design"],
        }
        
        return base_outputs + domain_outputs.get(domain, [])
    
    def update_plan(self, new_findings: List[ResearchFinding]) -> None:
        """
        Update the current plan based on findings.
        """
        if not self.current_plan:
            return
        
        high_conf_count = sum(1 for f in new_findings if f.confidence == ConfidenceLevel.HIGH)
        if high_conf_count > len(new_findings) / 2:
            self.current_plan.estimated_confidence = ConfidenceLevel.HIGH
        elif high_conf_count > len(new_findings) / 4:
            self.current_plan.estimated_confidence = ConfidenceLevel.MEDIUM
        else:
            self.current_plan.estimated_confidence = ConfidenceLevel.LOW


# ==========================================================
# CORE AGENT IMPLEMENTATION
# ==========================================================

class DeepResearchAgent:
    """
    Main research agent implementation.
    """
    
    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.1,
        max_iterations: int = 10
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_iterations = max_iterations
        
        self.memory = ResearchMemory()
        self.tool_registry = ToolRegistry(self.memory)
        self.planner = ResearchPlanner(self.memory)
        
        # Initialize the LLM with Groq
        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            streaming=True,
            api_key="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"
        )
        
        # Agent state
        self.current_domain: Optional[ResearchDomain] = None
        self.current_task: Optional[str] = None
        self.iteration_count: int = 0
        self.completed: bool = False
        
        # Initialize the agent executor
        self.agent_executor = None
    
    def _create_agent(self) -> None:
        """
        Create the agent executor using LangChain's agent framework.
        """
        prompt = self._create_agent_prompt()
        
        # Create the agent
        agent = create_react_agent(
            llm=self.llm,
            tools=list(self.tool_registry.tools.values()),
            prompt=prompt
        )
        
        # Create the executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=list(self.tool_registry.tools.values()),
            memory=self.memory.checkpoint_memory,
            verbose=True,
            max_iterations=self.max_iterations,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
    
    def _create_agent_prompt(self) -> ChatPromptTemplate:
        """
        Create the agent prompt with system instructions.
        """
        system_message = """You are DeepResearchAgent, an autonomous AI research assistant built using LangChain's latest agent architecture.

                            Your objective is to solve ANY research task by combining reasoning, planning, web research, document analysis, tool execution, memory management, verification, synthesis, and iterative refinement.

                            Never jump directly to an answer. Always think like an expert researcher.

                            ==========================================================
                            PRIMARY GOALS
                            ==========================================================

                            Your responsibilities include:

                            • Technical Research
                            • Scientific Research
                            • Medical Literature Review
                            • Cyber Security Investigation
                            • Software Engineering
                            • AI/ML Research
                            • Legal Research (non legal advice)
                            • Financial Research
                            • Market Analysis
                            • Competitor Analysis
                            • Product Research
                            • Academic Research
                            • Architecture Design
                            • Root Cause Analysis
                            • Bug Investigation
                            • Codebase Understanding
                            • API Investigation
                            • Standards Analysis
                            • RFC Analysis
                            • Whitepaper Analysis
                            • Compliance Research
                            • Threat Intelligence
                            • Vulnerability Research
                            • Reverse Engineering Research
                            • News Aggregation
                            • Trend Analysis
                            • Decision Support
                            • Technology Comparison

                            ==========================================================
                            GENERAL EXECUTION STRATEGY
                            ==========================================================

                            Follow the LangChain agent loop:

                            Observe → Plan → Think → Select Tools → Execute → Evaluate → Reflect → Repeat → Verify → Generate Final Report

                            Never skip verification.

                            ==========================================================
                            HALLUCINATION PREVENTION
                            ==========================================================

                            Never invent:
                            - Papers
                            - Links
                            - Statistics
                            - Benchmarks
                            - Quotes
                            - Versions
                            - APIs
                            - Authors
                            - Standards

                            If uncertain: Explicitly state uncertainty.

                            ==========================================================
                            BEHAVIOR
                            ==========================================================

                            Be analytical. Be skeptical. Be evidence driven. Be transparent. Be iterative. Think before acting. Verify before concluding. Never sacrifice accuracy for speed.

                            Always produce the highest quality research possible."""

        return ChatPromptTemplate.from_messages([
            SystemMessage(content=system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
    
    def _classify_task(self, query: str) -> ResearchDomain:
        """
        Classify the research task into a domain.
        """
        query_lower = query.lower()
        
        classifications = {
            ResearchDomain.CYBERSECURITY: ["security", "vulnerability", "cve", "attack", "threat", "malware", "hack", "breach"],
            ResearchDomain.ACADEMIC: ["paper", "research", "study", "journal", "publication", "thesis", "dissertation"],
            ResearchDomain.PROGRAMMING: ["code", "api", "function", "class", "method", "bug", "error", "compile", "framework"],
            ResearchDomain.HEALTHCARE: ["medical", "health", "disease", "treatment", "patient", "clinical", "drug"],
            ResearchDomain.FINANCE: ["market", "stock", "investment", "financial", "banking", "economic"],
            ResearchDomain.MACHINE_LEARNING: ["machine learning", "deep learning", "neural network", "ai", "algorithm"],
            ResearchDomain.DATA_ANALYSIS: ["data", "analysis", "statistics", "analytics", "dashboard"],
            ResearchDomain.CLOUD: ["cloud", "aws", "azure", "gcp", "serverless", "s3", "lambda"]
        }
        
        for domain, keywords in classifications.items():
            if any(keyword in query_lower for keyword in keywords):
                return domain
        
        return ResearchDomain.GENERAL
    
    def _think(self, query: str, context: Dict) -> Dict:
        """
        Internal reasoning step.
        """
        thought_process = {
            "understanding": self._analyze_question(query),
            "approach": self._determine_approach(query),
            "knowledge_gaps": self._identify_gaps(query),
            "uncertainties": [],
            "follow_up_questions": []
        }
        return thought_process
    
    def _analyze_question(self, query: str) -> Dict:
        """
        Analyze the question structure.
        """
        return {
            "what": "What is being asked?",
            "expected_output": "Expected output format?",
            "required_depth": "How deep?",
            "time_sensitivity": "Time sensitive?",
            "domain": "Domain?"
        }
    
    def _determine_approach(self, query: str) -> str:
        """
        Determine the best approach for the query.
        """
        return "Systematic research approach using multiple sources"
    
    def _identify_gaps(self, query: str) -> List[str]:
        """
        Identify knowledge gaps.
        """
        return ["Missing specific context", "Need more details"]
    
    def _execute_tools(self, plan: ResearchPlan) -> List[ResearchFinding]:
        """
        Execute the research plan using appropriate tools.
        """
        findings = []
        
        # Group tools for parallel execution
        tool_groups = self.tool_registry.get_parallel_tools(plan)
        
        for group in tool_groups:
            if len(group) > 1:
                # Run tools in parallel
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                tasks = []
                for tool in group:
                    if tool:
                        tasks.append(self._execute_tool_async(tool, plan.objective))
                
                if tasks:
                    results = loop.run_until_complete(asyncio.gather(*tasks))
                    for result in results:
                        if result:
                            findings.extend(result)
                loop.close()
            else:
                # Run sequentially
                tool = group[0]
                if tool:
                    results = self._execute_tool_sync(tool, plan.objective)
                    if results:
                        findings.extend(results)
        
        return findings
    
    async def _execute_tool_async(self, tool: Tool, query: str) -> List[ResearchFinding]:
        """
        Execute a tool asynchronously.
        """
        try:
            # Use invoke method instead of run (new in 1.3+)
            result = tool.invoke(query)
            
            finding = ResearchFinding(
                claim=f"Result from {tool.name}",
                evidence=result if isinstance(result, str) else str(result),
                source=f"Tool: {tool.name}",
                confidence=ConfidenceLevel.MEDIUM
            )
            return [finding]
        except Exception as e:
            logger.error(f"Error executing tool {tool.name}: {e}")
            self.memory.add_failed_attempt({"tool": tool.name, "error": str(e)})
            return []
    
    def _execute_tool_sync(self, tool: Tool, query: str) -> List[ResearchFinding]:
        """
        Execute a tool synchronously.
        """
        try:
            # Use invoke method instead of run (new in 1.3+)
            result = tool.invoke(query)
            
            finding = ResearchFinding(
                claim=f"Result from {tool.name}",
                evidence=result if isinstance(result, str) else str(result),
                source=f"Tool: {tool.name}",
                confidence=ConfidenceLevel.MEDIUM
            )
            return [finding]
        except Exception as e:
            logger.error(f"Error executing tool {tool.name}: {e}")
            self.memory.add_failed_attempt({"tool": tool.name, "error": str(e)})
            return []
    
    def _verify_findings(self, findings: List[ResearchFinding]) -> List[ResearchFinding]:
        """
        Verify findings against sources.
        """
        verified_findings = []
        
        for finding in findings:
            # Check if evidence is credible
            if self._is_credible_source(finding.source):
                finding.confidence = ConfidenceLevel.HIGH
            else:
                finding.confidence = ConfidenceLevel.LOW
            
            # Check for contradictions
            contradictions = self._find_contradictions(finding)
            if contradictions:
                finding.contradictions = contradictions
                finding.confidence = ConfidenceLevel.MEDIUM
            
            # Store in memory
            self.memory.add_finding(finding)
            verified_findings.append(finding)
        
        return verified_findings
    
    def _is_credible_source(self, source: str) -> bool:
        """
        Check if a source is credible.
        """
        credible_domains = [
            ".edu", ".gov", ".org", "arxiv.org",
            "pubmed.ncbi.nlm.nih.gov", "scholar.google.com",
            "wikipedia.org", "github.com", "stackoverflow.com"
        ]
        
        return any(domain in source.lower() for domain in credible_domains)
    
    def _find_contradictions(self, finding: ResearchFinding) -> List[str]:
        """
        Find contradictions with existing findings.
        """
        contradictions = []
        existing = self.memory.get_all_findings()
        
        for existing_finding in existing:
            if existing_finding.claim != finding.claim:
                if self._claims_contradict(finding.claim, existing_finding.claim):
                    contradictions.append(existing_finding.claim)
        
        return contradictions
    
    def _claims_contradict(self, claim1: str, claim2: str) -> bool:
        """
        Check if two claims contradict each other.
        """
        negative_words = ["not", "never", "no", "cannot", "unable", "incorrect", "false"]
        
        claim1_lower = claim1.lower()
        claim2_lower = claim2.lower()
        
        if any(word in claim1_lower for word in negative_words) and \
           any(word in claim2_lower for word in negative_words):
            return False
        if any(word in claim1_lower for word in negative_words):
            return True
        
        return False
    
    def _reflect(self) -> Dict:
        """
        Reflection step - evaluate research progress.
        """
        findings = self.memory.get_all_findings()
        
        reflection = {
            "answered_all": self._is_objective_complete(),
            "contradictions": self._find_all_contradictions(),
            "missing_evidence": self._identify_missing_evidence(),
            "need_more_search": len(findings) < 5,
            "need_another_tool": not self._tools_adequate(),
            "need_another_iteration": self.iteration_count < self.max_iterations
        }
        
        return reflection
    
    def _is_objective_complete(self) -> bool:
        """
        Check if the research objective is complete.
        """
        findings = self.memory.get_high_confidence_findings()
        return len(findings) >= 3
    
    def _find_all_contradictions(self) -> List[Tuple[str, str]]:
        """
        Find all contradictions among findings.
        """
        contradictions = []
        findings = self.memory.get_all_findings()
        
        for i, finding1 in enumerate(findings):
            for finding2 in findings[i+1:]:
                if self._claims_contradict(finding1.claim, finding2.claim):
                    contradictions.append((finding1.claim, finding2.claim))
        
        return contradictions
    
    def _identify_missing_evidence(self) -> List[str]:
        """
        Identify missing evidence.
        """
        return ["Need more primary sources", "Need verification from official sources"]
    
    def _tools_adequate(self) -> bool:
        """
        Check if current tools are adequate.
        """
        findings = self.memory.get_all_findings()
        return len(findings) > 0
    
    def _generate_final_report(self) -> str:
        """
        Generate the final research report.
        """
        findings = self.memory.get_all_findings()
        high_confidence = self.memory.get_high_confidence_findings()
        
        sections = {
            "Executive Summary": self._generate_executive_summary(findings),
            "Problem Statement": self._generate_problem_statement(),
            "Methodology": self._generate_methodology(),
            "Evidence": self._generate_evidence_section(high_confidence),
            "Analysis": self._generate_analysis(findings),
            "Alternative Views": self._generate_alternative_views(findings),
            "Limitations": self._generate_limitations(),
            "Confidence Assessment": self._generate_confidence_assessment(),
            "References": self._generate_references(),
            "Recommendations": self._generate_recommendations(findings)
        }
        
        report = "=" * 80 + "\n"
        report += "RESEARCH REPORT\n"
        report += "=" * 80 + "\n\n"
        
        for section, content in sections.items():
            report += f"## {section.upper()}\n\n"
            report += content + "\n\n"
            report += "-" * 40 + "\n\n"
        
        report += "=" * 80 + "\n"
        report += f"Report Generated: {datetime.now().isoformat()}\n"
        report += f"Iterations: {self.iteration_count}\n"
        report += f"Findings: {len(findings)}\n"
        report += f"High Confidence Findings: {len(high_confidence)}\n"
        report += "=" * 80 + "\n"
        
        return report
    
    def _generate_executive_summary(self, findings: List[ResearchFinding]) -> str:
        """
        Generate executive summary.
        """
        if not findings:
            return "No findings available. Research was unable to gather sufficient information."
        
        summary = f"This research investigated the query and gathered {len(findings)} pieces of evidence.\n\n"
        summary += "Key findings:\n"
        for i, finding in enumerate(findings[:3], 1):
            summary += f"{i}. {finding.claim[:100]}... (Confidence: {finding.confidence.value})\n"
        
        return summary
    
    def _generate_problem_statement(self) -> str:
        """
        Generate problem statement.
        """
        return f"This research aims to investigate: {self.current_task or 'Unspecified research query'}\n\nDomain: {self.current_domain.value if self.current_domain else 'Unspecified'}"
    
    def _generate_methodology(self) -> str:
        """
        Generate methodology section.
        """
        return """Methodology:
1. Task Classification - Identified the domain and nature of the research
2. Research Planning - Created structured approach with subtasks
3. Tool Selection - Selected appropriate tools based on domain
4. Evidence Gathering - Collected information from multiple sources
5. Verification - Cross-referenced findings against credible sources
6. Synthesis - Integrated findings into coherent analysis"""
    
    def _generate_evidence_section(self, findings: List[ResearchFinding]) -> str:
        """
        Generate evidence section.
        """
        if not findings:
            return "No high-confidence evidence available."
        
        evidence = "### High-Confidence Evidence\n\n"
        for i, finding in enumerate(findings, 1):
            evidence += f"**Finding {i}:** {finding.claim}\n"
            evidence += f"*Source:* {finding.source}\n"
            evidence += f"*Confidence:* {finding.confidence.value}\n"
            evidence += f"*Evidence:* {finding.evidence[:200]}...\n\n"
        
        return evidence
    
    def _generate_analysis(self, findings: List[ResearchFinding]) -> str:
        """
        Generate analysis section.
        """
        analysis = "### Analysis of Findings\n\n"
        
        if not findings:
            analysis += "Insufficient data for meaningful analysis."
            return analysis
        
        analysis += f"Analysis of {len(findings)} total findings reveals:\n\n"
        
        high = [f for f in findings if f.confidence == ConfidenceLevel.HIGH]
        medium = [f for f in findings if f.confidence == ConfidenceLevel.MEDIUM]
        
        analysis += f"High-confidence findings: {len(high)}\n"
        analysis += f"Medium-confidence findings: {len(medium)}\n\n"
        
        if high:
            analysis += "**Key Patterns:**\n"
            for finding in high[:3]:
                analysis += f"- {finding.claim[:100]}...\n"
        
        return analysis
    
    def _generate_alternative_views(self, findings: List[ResearchFinding]) -> str:
        """
        Generate alternative views section.
        """
        alt_views = []
        for finding in findings:
            if finding.contradictions:
                alt_views.append(finding.contradictions)
        
        if not alt_views:
            return "No significant alternative views or contradictions were identified in the research."
        
        return "Alternative views identified:\n\n" + "\n".join([f"- {view}" for view in alt_views[:3]])
    
    def _generate_limitations(self) -> str:
        """
        Generate limitations section.
        """
        return """Limitations of this research:

1. The research relied on available public sources
2. Some information may be time-sensitive
3. Full verification of all claims may require more resources
4. Some sources may have inherent biases
5. The research was conducted within the constraints of available tools"""
    
    def _generate_confidence_assessment(self) -> str:
        """
        Generate confidence assessment.
        """
        findings = self.memory.get_all_findings()
        high = len([f for f in findings if f.confidence == ConfidenceLevel.HIGH])
        total = len(findings) if findings else 1
        
        confidence_level = "HIGH" if high / total > 0.7 else "MEDIUM" if high / total > 0.4 else "LOW"
        
        return f"""Confidence Assessment:
- Overall confidence: {confidence_level}
- High-confidence findings: {high} out of {total}
- Reasons for confidence level: Based on source credibility and verification"""
    
    def _generate_references(self) -> str:
        """
        Generate references section.
        """
        sources = list(self.memory.source_metadata.keys())
        
        if not sources:
            return "No formal references available."
        
        references = "## References\n\n"
        for i, source in enumerate(sources[:10], 1):
            metadata = self.memory.source_metadata[source]
            references += f"{i}. {metadata.title or source}\n"
            references += f"   {source}\n"
            if metadata.author:
                references += f"   Author: {metadata.author}\n"
            if metadata.publication_date:
                references += f"   Published: {metadata.publication_date.strftime('%Y-%m-%d')}\n"
            references += "\n"
        
        if len(sources) > 10:
            references += f"... and {len(sources) - 10} more sources\n"
        
        return references
    
    def _generate_recommendations(self, findings: List[ResearchFinding]) -> str:
        """
        Generate recommendations.
        """
        if not findings:
            return "Insufficient findings to generate recommendations."
        
        recommendations = "## Recommendations\n\n"
        
        high_conf = self.memory.get_high_confidence_findings()
        if high_conf:
            recommendations += "Based on the high-confidence findings:\n\n"
            for finding in high_conf[:3]:
                recommendations += f"- {finding.claim[:100]}... (Source: {finding.source})\n"
                recommendations += f"  Confidence: {finding.confidence.value}\n\n"
        
        recommendations += "General recommendations:\n"
        recommendations += "1. Verify findings with additional sources\n"
        recommendations += "2. Monitor for updated information\n"
        recommendations += "3. Consider the context and limitations of this research\n"
        recommendations += "4. Follow up on high-impact findings with deeper investigation\n"
        
        return recommendations

    # ==========================================================
    # MAIN AGENT LOOP
    # ==========================================================
    
    def research(self, query: str) -> str:
        """
        Main research method - the entry point for the agent.
        
        This implements the LangChain agent loop:
        Observe → Plan → Think → Select Tools → Execute → Evaluate → Reflect → Repeat → Verify → Generate Final Report
        """
        logger.info(f"Starting research on: {query}")
        self.current_task = query
        
        # STEP 1: OBSERVE
        logger.info("STEP 1: OBSERVING - Understanding the task")
        
        domain = self._classify_task(query)
        self.current_domain = domain
        
        thought = self._think(query, {})
        
        if thought.get("follow_up_questions"):
            return "I need more information:\n" + "\n".join(thought["follow_up_questions"])
        
        # STEP 2: PLAN
        logger.info("STEP 2: PLANNING - Creating research plan")
        plan = self.planner.create_plan(query, domain)
        
        # STEP 3: THINK
        logger.info("STEP 3: THINKING - Analyzing approach")
        
        # STEP 4: SELECT TOOLS
        logger.info("STEP 4: TOOL SELECTION - Choosing appropriate tools")
        selected_tools = self.tool_registry.select_tools_for_task(plan)
        
        # MAIN RESEARCH LOOP
        logger.info("Starting research iterations...")
        
        all_findings = []
        
        while self.iteration_count < self.max_iterations and not self.completed:
            self.iteration_count += 1
            logger.info(f"Iteration {self.iteration_count}")
            
            # STEP 5: EXECUTE (Parallel Research)
            logger.info("STEP 5: EXECUTING - Running research tools")
            findings = self._execute_tools(plan)
            
            # STEP 6: EVALUATE
            logger.info("STEP 6: EVALUATING - Analyzing results")
            
            # STEP 7: VERIFICATION
            logger.info("STEP 7: VERIFYING - Checking sources and evidence")
            verified_findings = self._verify_findings(findings)
            all_findings.extend(verified_findings)
            
            # Update the plan with new findings
            self.planner.update_plan(verified_findings)
            
            # STEP 8: REFLECT
            logger.info("STEP 8: REFLECTING - Evaluating progress")
            reflection = self._reflect()
            
            if reflection["answered_all"] and not reflection["contradictions"]:
                self.completed = True
                logger.info("Research completed successfully")
                break
            
            if not reflection["need_more_search"] and not reflection["need_another_tool"]:
                self.completed = True
                logger.info("Stopping due to completion criteria")
                break
            
            logger.info(f"Need more search: {reflection['need_more_search']}")
            logger.info(f"Need another tool: {reflection['need_another_tool']}")
        
        # STEP 9: FINAL REPORT
        logger.info("STEP 9: GENERATING FINAL REPORT")
        report = self._generate_final_report()
        
        return report
    
    def research_with_plan(self, query: str, custom_plan: ResearchPlan) -> str:
        """
        Execute research with a custom plan.
        """
        self.current_task = query
        self.planner.current_plan = custom_plan
        
        return self.research(query)
    
    def research_with_agent(self, query: str) -> str:
        """
        Research using the LangChain agent executor.
        """
        # Initialize the agent if not already done
        if not self.agent_executor:
            self._create_agent()
        
        # Run the agent
        try:
            result = self.agent_executor.invoke({
                "input": query,
                "chat_history": self.memory.conversation_history
            })
            return result.get("output", "No output generated")
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            return f"Error: {str(e)}"


# ==========================================================
# MAIN FUNCTION
# ==========================================================

def main():
    """
    Example usage of DeepResearchAgent.
    """
    
    # Create the agent
    agent = DeepResearchAgent(
        model_name="llama-3.3-70b-versatile",
        temperature=0.1,
        max_iterations=5
    )
    
    # Example queries
    # test_queries = [
    #     "What are the latest advancements in quantum computing?",
    #     "Explain the security vulnerabilities in the OAuth 2.0 protocol",
    #     "Compare Python vs JavaScript for web development in 2024",
    #     "What is the current state of AI in healthcare?"
    # ]
    
    # # Run research on a query
    # query = test_queries[0]
    # print(f"\n{'='*80}\nResearch Query: {query}\n{'='*80}\n")
    query = ''
    while query != 'exit':
        query = input("Enter your Query:: ")
        result = agent.research(query)
        print(result)


if __name__ == "__main__":
    main()