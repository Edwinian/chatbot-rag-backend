import json
import os
import requests
import re

from typing import Any, Dict, List, Optional
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from chroma_service import ChromaService
from db_service import DBService
from pydantic_models import ModelName, StructuredChunk, StructuredChunkType
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()  # Loads the .env file


class LangChainService:
    MODEL_ANSWER = {"heading_prefix": "<b>", "bullet_prefix": "-"}
    VALID_LINK_PREFIXES = [
        "http://",
        "https://",
        "www.",
        "ftp://",
        "file://",
    ]
    DEFAULT_CONTEXT_LENGTH = 30000
    MAX_CONTEXT_LENGTH_MAP = {
        ModelName.DeepSeek_R1_Distill_Qwen_32B.value: DEFAULT_CONTEXT_LENGTH,
        ModelName.All_mini_l6_v2.value: 512,
    }

    def __init__(
        self,
        collection_name: Optional[str] = None,
        model_name: Optional[str] = ModelName.DeepSeek_R1_Distill_Qwen_32B.value,
        max_output_length: int = 512,
    ):
        """
        Initialize LangChainService with a ChromaService instance and LLM configuration.

        Args:
            chroma_service: ChromaService instance for retrieval
            model_name: Hugging Face model name (e.g., mistralai/Mistral-7B-Instruct-v0.3)
            max_output_length: Maximum length for generated text
        """
        self.chroma_service = ChromaService(collection_name=collection_name)
        self.db_service = DBService()
        self.output_parser = StrOutputParser()
        self.model_name = model_name
        self.max_output_length = max_output_length
        self.max_context_length = self.MAX_CONTEXT_LENGTH_MAP.get(
            model_name, self.DEFAULT_CONTEXT_LENGTH
        )
        self.max_input_tokens = self.max_context_length - self.max_output_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=os.getenv("HUGGINGFACE_TOKEN")
        )

        # Initialize prompt templates
        self.contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        self.contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        self.qa_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant. Use the following context to answer the user's question.\n\nContext: {context}",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )
        self.chat_llm = self._initialize_chat_llm()

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tokenizer or fallback to character-based estimation."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text, add_special_tokens=True))
        return len(text) // 4 + 1

    def _truncate_to_fit_context(
        self,
        query: str,
        chat_history: List[Dict],
        context: str,
    ) -> tuple[str, List[Dict], str]:
        """Truncate query, chat history, and context to fit within max_input_tokens."""
        # Estimate token counts
        query_tokens = self._count_tokens(query)
        context_tokens = self._count_tokens(context)
        history_tokens = sum(self._count_tokens(str(msg)) for msg in chat_history)

        # Calculate total tokens
        total_tokens = query_tokens + context_tokens + history_tokens

        if total_tokens <= self.max_input_tokens:
            return query, chat_history, context

        # Truncate context first (least critical)
        if context_tokens > self.max_input_tokens // 3:
            target_chars = (
                self.max_input_tokens // 3
            ) * 4  # Convert tokens to chars for fallback
            context = self.summarize_content(context, max_length=target_chars)
            context_tokens = self._count_tokens(context)

        # Recalculate total tokens
        total_tokens = query_tokens + context_tokens + history_tokens

        # Truncate chat history if still over limit
        if total_tokens > self.max_input_tokens:
            remaining_tokens = self.max_input_tokens - query_tokens - context_tokens
            if remaining_tokens < 0:
                # Truncate query as last resort
                query_chars = (self.max_input_tokens - context_tokens) * 4
                query = self.summarize_content(query, max_length=query_chars)
            else:
                # Keep recent messages in history
                kept_history = []
                current_tokens = 0
                for msg in reversed(chat_history):
                    msg_tokens = self._count_tokens(str(msg))
                    if current_tokens + msg_tokens <= remaining_tokens:
                        kept_history.append(msg)
                        current_tokens += msg_tokens
                    else:
                        break
                chat_history = list(reversed(kept_history))

        return query, chat_history, context

    def _initialize_chat_llm(self):
        """Initialize and return the ChatHuggingFace LLM."""
        try:
            endpoint = HuggingFaceEndpoint(
                repo_id=self.model_name,
                huggingfacehub_api_token=os.getenv("HUGGINGFACE_TOKEN"),
                max_new_tokens=self.max_output_length,
                temperature=0.6,  # Controls response randomness
                top_p=0.95,  # Controls response diversity
                return_full_text=False,
            )
            # Wrap with ChatHuggingFace since HuggingFaceEndpoint always defaults to text-generation task, which is not supported by all models
            return ChatHuggingFace(llm=endpoint)
        except Exception as e:
            raise Exception(f"Failed to initialize LLM {self.model_name}: {str(e)}")

    def get_rag_chain(self):
        try:
            retriever = self.chroma_service.get_retriever()
            history_aware_retriever = create_history_aware_retriever(
                self.chat_llm, retriever, self.contextualize_q_prompt
            )
            question_answer_chain = create_stuff_documents_chain(
                self.chat_llm, self.qa_prompt
            )
            rag_chain = create_retrieval_chain(
                history_aware_retriever, question_answer_chain
            )
            return rag_chain
        except Exception as e:
            print(f"Error creating RAG chain: {str(e)}")
            raise

    def get_model_answer(
        self,
        query: str,
        session_id: str = None,
        skip_hybrid: bool = False,
    ):
        chat_history = (
            self.db_service.get_chat_history(session_id) if session_id else []
        )
        rag_chain = self.get_rag_chain()

        # Get context from ChromaService retriever
        retriever = self.chroma_service.get_retriever()
        context_docs = retriever.invoke(query)
        context = "\n".join(doc.page_content for doc in context_docs)

        # Truncate inputs to fit context length
        query, chat_history, context = self._truncate_to_fit_context(
            query, chat_history, context
        )

        # Update context in rag_results
        rag_results = rag_chain.invoke(
            {"input": query, "chat_history": chat_history, "context": context}
        )
        print(
            "Retrieved documents:",
            [doc.metadata["source"] for doc in rag_results["context"]],
        )

        answer = rag_results["answer"]

        if not skip_hybrid:
            need_hybrid = self.need_hybrid_answer(query=query, answer=answer)
            print("need_hybrid", need_hybrid)

            if need_hybrid:
                answer = self.get_hybrid_answer(
                    query=query, chat_history=chat_history, rag_results=rag_results
                )
                print("need_hybrid answer", answer)

        return answer

    def get_urls_from_prompt(self, query: str) -> List[str]:
        urls = []

        if not query:
            return urls

        url_pattern = re.compile(
            r"(?:(?:https?|ftp|file)://)"  # Protocol (http, https, ftp, file)
            r"(?:[\w-]+\.)*[\w-]+"  # Domain (subdomains and main domain)
            r"(?:\.[a-zA-Z]{2,63})"  # TLD
            r"(?:/[\w\-./?%&=]*)?",  # Optional path/query
            re.IGNORECASE,
        )

        potential_urls = re.findall(url_pattern, query)
        urls = [
            url
            for url in potential_urls
            if any(
                url.lower().startswith(prefix) for prefix in self.VALID_LINK_PREFIXES
            )
        ]

        print("Extracted URLs:", urls)
        return urls

    def summarize_content(
        self, text: Optional[str] = None, max_length: int = 1000
    ) -> str:
        """Summarize content to fit within context limits"""
        if not text or len(text) <= max_length:
            return text

        # Use LLM to summarize (or simpler method for demo)
        summary_prompt = f"""
        Summarize the following content in {max_length} characters or less,
        preserving key information:
        
        {text}
        """

        try:
            summary = self.chat_llm.invoke(summary_prompt)
            return summary[:max_length]
        except:
            # Fallback to simple truncation
            return text[:max_length] + "... [truncated]"

    def get_html_content(self, query: str) -> Optional[str]:
        MAX_HTML_LENGTH = 5000  # Limit to prevent excessive content
        urls = self.get_urls_from_prompt(query)

        if not urls:
            return None

        all_content = ""

        for url in urls:
            try:
                headers = {"User-Agent": "Mozilla/5.0..."}
                response = requests.get(url, headers=headers, timeout=10)

                if "text/html" not in response.headers.get("content-type", ""):
                    continue

                soup = BeautifulSoup(response.text, "html.parser")

                # Remove unwanted elements
                for element in soup(["script", "style", "nav", "footer", "iframe"]):
                    element.decompose()

                # Get text from main content areas
                main_content = ""

                for tag in ["article", "main", "div.content", "section"]:
                    elements = soup.find_all(tag)
                    for element in elements:
                        main_content += (
                            element.get_text(separator="\n", strip=True) + "\n\n"
                        )

                # Fallback to body if no specific content found
                if not main_content:
                    main_content = (
                        soup.body.get_text(separator="\n", strip=True)
                        if soup.body
                        else ""
                    )

                if len(main_content) > MAX_HTML_LENGTH:
                    main_content = (
                        main_content[:MAX_HTML_LENGTH] + "... [content truncated]"
                    )

                all_content += f"\n=== Content from {url} ===\n{main_content}\n"

            except Exception as e:
                print(f"Error processing {url}: {str(e)}")

        return self.summarize_content(all_content)

    def get_search_content(self, query: str) -> Optional[str]:
        try:
            api_url = "https://serpapi.com/search"
            headers = {"Content-Type": "application/json"}
            params = {
                "q": query,
                "api_key": os.getenv("SERPAPI_KEY"),
                "engine": "google",
                "num": 5,  # Reduced to focus on top results
            }
            response = requests.get(api_url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            # Score and rank results
            results = []

            # 1. First check answer_box (highest priority)
            if "answer_box" in data:
                answer_box = data["answer_box"]
                content = json.dumps(answer_box)
                source_link = answer_box.get("link", "Direct Answer")
                results.append(
                    {
                        "content": content,
                        "source": source_link,
                        "score": 1.0,  # Highest confidence for direct answers
                    }
                )
            else:
                # 2. Check knowledge_graph
                if "knowledge_graph" in data:
                    kg = data["knowledge_graph"]
                    source_link = kg.get("source", {}).get("link", "Knowledge Graph")

                    if "description" in kg:
                        results.append(
                            {
                                "content": kg["description"],
                                "source": source_link,
                                "score": 0.7,
                            }
                        )

                    # For time-related info in knowledge_graph
                    if kg.get("title", "").lower() in [
                        "time in hong kong",
                        "hong kong time",
                    ]:
                        if "description" in kg:
                            results.append(
                                {
                                    "content": kg["description"],
                                    "source": source_link,
                                    "score": 0.8,
                                }
                            )

                # 3. Check related_questions (often contains featured snippets)
                if "related_questions" in data:
                    for q in data["related_questions"]:
                        source_link = q.get("link", "Related Question")
                        if "snippet" in q:
                            results.append(
                                {
                                    "content": q["snippet"],
                                    "source": source_link,
                                    "score": 0.6,
                                }
                            )
                        elif "table" in q and isinstance(q["table"], list):
                            # Flatten table data for time-related queries
                            table_content = []
                            for row in q["table"]:
                                if isinstance(row, list):
                                    table_content.append(", ".join(str(x) for x in row))
                                elif isinstance(row, dict):
                                    table_content.append(str(row))
                            if table_content:
                                results.append(
                                    {
                                        "content": "; ".join(table_content),
                                        "source": source_link,
                                        "score": 0.65,
                                    }
                                )

                # 4. Process organic results
                if "organic_results" in data:
                    for idx, result in enumerate(
                        data["organic_results"][:5]
                    ):  # Top 5 only
                        source_link = result.get("link", "Organic Result")
                        source_domain = ""

                        # Extract domain for display
                        if "link" in result:
                            try:
                                source_domain = (
                                    result["link"].split("//")[-1].split("/")[0]
                                )
                            except:
                                source_domain = source_link

                        # Calculate base score based on position
                        position_score = 0.6 - (
                            idx * 0.1
                        )  # Higher score for top results
                        content_score = 0.0

                        # Boost score for authoritative domains
                        if any(
                            d in source_domain
                            for d in ["wikipedia.org", "gov.hk", "timeanddate.com"]
                        ):
                            content_score += 0.2

                        # Boost score for time/date mentions
                        snippet = result.get("snippet", "")
                        if "time" in snippet.lower() or "date" in snippet.lower():
                            content_score += 0.1

                        total_score = position_score + content_score

                        if "snippet" in result:
                            results.append(
                                {
                                    "content": result["snippet"],
                                    "source": (
                                        source_domain if source_domain else source_link
                                    ),
                                    "link": source_link,
                                    "score": total_score,
                                }
                            )

            # Sort all results by score (highest first)
            results.sort(key=lambda x: x["score"], reverse=True)

            # Format the final output with sources
            if results:
                formatted_results = []
                seen_content = set()  # To avoid duplicates

                for res in results[:3]:  # Take top 3 results
                    # Skip duplicates
                    content_hash = hash(res["content"])
                    if content_hash in seen_content:
                        continue
                    seen_content.add(content_hash)

                    # Format source display
                    source_display = res["source"]
                    if len(source_display) > 30:
                        source_display = source_display[:27] + "..."

                    # Format the result with source and confidence
                    formatted_results.append(
                        f"[Source: {source_display} | Confidence: {res['score']:.1f}]\n"
                        f"{res['content']}\n"
                    )

                return "\n".join(formatted_results)

        except requests.exceptions.RequestException as e:
            print(f"Failed to retrieve content from web: {str(e)}")
            return ""

    def get_hybrid_answer(self, query: str, chat_history: List[Dict], rag_results: Any):
        html_content = self.get_html_content(query)
        search_content = self.get_search_content(query)

        if all([not search_content, not html_content]):
            return rag_results["answer"]

        # Combine context
        combined_context = f"""
        HTML Content:
        {html_content or ''}

        Web Search Content (most authoritative source if Confidence is 1.0):
        {search_content or ''}
        
        Internal Knowledge (for reference only):
        {rag_results['context']}
        
        Instructions:
        - Always provide the most specific answer available
        - Never say you can't provide real-time information when it's available
        - Remind user to fact check if web search content is used
        - Do not include confidence score or (Confidence: ) in the response
        - HTML content is used as context only and should not be used directly in the answer
        - Add {self.MODEL_ANSWER['bullet_prefix']} in front of bullet points or lists
        - Add {self.MODEL_ANSWER['heading_prefix']} in front of headings
        - Wrap keywords with <b> and </b> to make them bold
        - Paraphrase user's prompt in the beginning to ensure your understanding
        - Give conclusion om the end to conclude your response
        """

        # Truncate inputs
        query, chat_history, combined_context = self._truncate_to_fit_context(
            query, chat_history, combined_context
        )

        hybrid_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant that provides accurate, up-to-date information. "
                    "You have access to both internal knowledge and real-time web search results.\n\n"
                    "Context:\n{context}\n\n"
                    "Response Guidelines:\n"
                    "2. Present the information clearly and directly\n"
                    "3. Never claim you can't provide real-time info when it's available\n"
                    "4. If web results provide a direct answer with confidence 1.0, use it verbatim\n"
                    "5. Only reference internal knowledge if it supplements the web results\n",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )

        hybrid_chain = hybrid_prompt | self.chat_llm | self.output_parser
        answer = hybrid_chain.invoke(
            {"input": query, "chat_history": chat_history, "context": combined_context}
        )

        return answer

    def need_hybrid_answer(self, query: str, answer: str) -> bool:
        """
        Determine if a hybrid answer is needed based on the query.

        """

        def _is_low_confident(answer: str) -> bool:
            """Heuristic to detect when external search might be needed"""
            low_confidence_phrases = [
                "I don't know",
                "I'm not sure",
                "not in my knowledge",
                "no information",
                "can't find",
                "unable to",
                "you may want to check",
                "verify this information",
                "as of my last update",
                "latest",
                "news",
                "update",
                "current",
                "sorry",
                "I cannot provide real-time information",
            ]
            return any(phrase in answer.lower() for phrase in low_confidence_phrases)

        def _llm_decides(query: str, answer: str) -> bool:
            """Use the LLM to evaluate if external search is needed"""
            decision_prompt = f"""
            Based on the following interaction, should we search external sources for more information?
            
            Query: {query}
            Current Answer: {answer}
            
            Respond ONLY with 'YES' or 'NO'. Do not provide any explanation.
            """

            response = self.get_model_answer(
                query=decision_prompt,
                skip_hybrid=True,
            )

            return response.upper() == "YES"

        return any(
            [
                _is_low_confident(answer),
                # _llm_decides(query, answer),
                len(self.get_urls_from_prompt(query)) > 0,
            ]
        )

    def format_llm_response(self, response: str) -> list[StructuredChunk]:
        """
        Parse plain LLM response into structured chunks with formatting metadata.
        Example: Convert the provided LLM response into sections with headings and bullet points.
        """
        chunks = []

        # Example parsing logic (this can be customized based on LLM output patterns)
        lines = response.split("\n")
        for i, line in enumerate(lines):
            line = line.strip()

            if not line:
                continue

            if i == len(lines) - 1:
                line += "."  # Ensure last line ends with a period

            if line.startswith(self.MODEL_ANSWER["heading_prefix"]):
                line = line.replace(self.MODEL_ANSWER["heading_prefix"], "").strip()
                chunks.append(
                    StructuredChunk(
                        type=StructuredChunkType.HEADING, content=line.strip()
                    )
                )
            elif line.startswith(self.MODEL_ANSWER["bullet_prefix"]):
                chunks.append(
                    StructuredChunk(
                        type=StructuredChunkType.BULLET, content=line[1:].strip()
                    )
                )
            else:
                chunks.append(
                    StructuredChunk(type=StructuredChunkType.PARAGRAPH, content=line)
                )

        return chunks
