from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable


class Summarizer:
    """Generates summaries of memory content."""
    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm
        self.chain = self._create_summary_chain()
    
    def _create_summary_chain(self) -> Runnable:
        """Create summarization chain."""
        parser = StrOutputParser()
        prompt = PromptTemplate(
            template=self._summarize_template(),
            input_variables=["content", "context"]
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
    
    async def summarize(self, content: str, context: str = "") -> str:
        """Generate a concise summary of the chat interaction."""
        try:
            summary = await self.chain.ainvoke({"content": content, "context": context})
            return summary
        except Exception as e:
            raise e