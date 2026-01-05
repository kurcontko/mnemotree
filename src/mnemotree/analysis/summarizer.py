from __future__ import annotations

from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable

from .models import SummaryResult


class Summarizer:
    """Generates summaries of memory content."""
    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm
        self.text_chain = self._create_text_summary_chain()
        self.structured_chain = self._create_structured_summary_chain()
    
    def _create_text_summary_chain(self) -> Runnable:
        """Create plain-text summarization chain."""
        parser = StrOutputParser()
        prompt = PromptTemplate(
            template=self._summarize_template(),
            input_variables=["content", "context"],
        )
        return prompt | self.llm | parser

    def _create_structured_summary_chain(self) -> Runnable:
        """Create JSON summarization chain."""
        parser = JsonOutputParser(pydantic_object=SummaryResult)
        prompt = PromptTemplate(
            template=self._summarize_structured_template(),
            input_variables=["content", "context"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        return prompt | self.llm | parser
    
    def _summarize_template(self) -> str:
        """Template for summarization."""
        return """
        Please generate a concise summary of the following chat interaction. 
        The summary should capture the key topics discussed, main decisions made, action items, and the entities involved. 
        Present the summary in a clear and structured format suitable for embedding and storage in a vector graph database. 
        Ensure the information is comprehensive enough to support semantic search queries based on embeddings.
        Format the summary as a string.
        
        ---
        
        Content: {content}
        Context: {context}
        
        "Answer only with the summary and nothing else."       
        """.strip()

    def _summarize_structured_template(self) -> str:
        """Template for structured summarization."""
        return """
        Please generate a concise summary of the following chat interaction. 
        The summary should capture the key topics discussed, main decisions made, action items, and the entities involved.
        Provide structured JSON output.

        Content: {content}
        Context: {context}

        {format_instructions}
        """.strip()

    async def summarize(
        self,
        content: str,
        context: str = "",
        format: str = "text"
    ) -> str | dict:
        """Generate a summary of the chat interaction."""
        try:
            if format == "structured":
                result = await self.structured_chain.ainvoke({"content": content, "context": context})
                if isinstance(result, SummaryResult):
                    return result.model_dump()
                return result
            return await self.text_chain.ainvoke({"content": content, "context": context})
        except Exception as e:
            raise e
