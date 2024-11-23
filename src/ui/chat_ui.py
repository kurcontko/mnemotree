import streamlit as st
import time
import asyncio
from typing import Tuple, Optional, List, Any
from datetime import datetime
from .types import MessageRole
from ..inference.langchain_inference import LangChainInference
from ..memory.processor import MemoryProcessor


class ChatUI:
    """Enhanced Chat UI with memory processing capabilities, progress indicators, and core chat functionality."""

    def __init__(self):
        """Initialize the Chat UI with memory processor and custom styling."""
        self.inference_engine = self.get_inference_engine()
        self.memory_processor = MemoryProcessor()
        self.init_session_state()
        self.apply_custom_css()

    @staticmethod
    def apply_custom_css():
        """Apply minimal CSS styling that only hides Streamlit elements."""
        st.markdown("""
            <style>
            /* Hide Streamlit elements */
            div[data-testid="stToolbar"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
            }
            div[data-testid="stDecoration"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
            }
            div[data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
            }
            #MainMenu {
                visibility: hidden;
                height: 0%;
            }
            header {
                visibility: hidden;
                height: 0%;
            }
            footer {
                visibility: hidden;
                height: 0%;
            }

            /* Minimal styling for sidebar and custom elements */
            .status-indicator {
                display: inline-block;
                width: 10px;
                height: 10px;
                border-radius: 50%;
                margin-right: 5px;
            }
            
            .status-online {
                background-color: #4CAF50;
            }
            
            .status-offline {
                background-color: #F44336;
            }

            /* Message timestamp */
            .message-timestamp {
                font-size: 0.8em;
                color: #666;
                margin-top: 0.2rem;
            }

            /* System message */
            .system-message {
                padding: 0.5rem 1rem;
                border-radius: 5px;
                margin: 0.5rem 0;
                font-size: 0.9em;
                color: #E65100;
            }

            /* Memory styles */
            .memory-container {
                margin-top: 1rem;
                padding: 0.5rem;
                background-color: #f0f7ff;
                border-left: 3px solid #0066cc;
                border-radius: 4px;
            }
            </style>
        """, unsafe_allow_html=True)

    @staticmethod
    @st.cache_resource
    def get_inference_engine() -> LangChainInference:
        """Cache the inference engine instance."""
        return LangChainInference()

    def init_session_state(self):
        """Initialize all session state variables."""
        # Chat-related state
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "auto_clear" not in st.session_state:
            st.session_state.auto_clear = False
        if "last_response_time" not in st.session_state:
            st.session_state.last_response_time = None
        if "conversation_started" not in st.session_state:
            st.session_state.conversation_started = datetime.now()
        
        # Memory-related state
        if "memories" not in st.session_state:
            st.session_state.memories = []
        if "memory_enabled" not in st.session_state:
            st.session_state.memory_enabled = True
        if "memory_search" not in st.session_state:
            st.session_state.memory_search = ""

    async def process_memory(self, prompt: str, response: str, progress_bar) -> Optional[Any]:
        """Process and store memory from the conversation with progress updates."""
        try:
            progress_bar.progress(0, "Starting memory creation...")
            await asyncio.sleep(0.2)
            
            progress_bar.progress(30, "Analyzing conversation...")
            memory = await self.memory_processor.create_memory_from_messages(prompt, response)
            
            progress_bar.progress(60, "Processing memory...")
            await asyncio.sleep(0.1)
            
            progress_bar.progress(90, "Storing memory...")
            st.session_state.memories.append(memory)
            
            progress_bar.progress(100, "Memory created successfully!")
            await asyncio.sleep(0.2)
            
            return memory
            
        except Exception as e:
            progress_bar.error(f"Failed to process memory: {str(e)}")
            return None

    def format_memory_display(self, memory) -> str:
        """Format memory for inline display in the UI."""
        if not memory:
            return ""
        return f"""
<div class="memory-container">
    <div style="font-size: 0.9em; color: #666;">🧠 Memory Created</div>
    <div style="margin-top: 0.5rem;">{memory.to_str()}</div>
</div>
"""

    def filter_memories(self, memories: List, search_term: str) -> List:
        """Filter memories based on search term."""
        if not search_term:
            return memories
        search_term = search_term.lower()
        return [memory for memory in memories if search_term in memory.to_str().lower()]

    def show_memory_sidebar(self):
        """Display memory controls and searchable memory list in sidebar."""
        with st.sidebar:
            st.subheader("Memory Settings")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.session_state.memory_enabled = st.toggle(
                    "Enable Memory Processing",
                    value=st.session_state.memory_enabled,
                    help="Process and store memories from conversations"
                )
            with col2:
                if st.button("Clear All"):
                    st.session_state.memories = []
                    st.rerun()

            if st.session_state.memories:
                st.subheader("Memory Archive")
                st.session_state.memory_search = st.text_input(
                    "Search memories",
                    placeholder="Type to search...",
                    value=st.session_state.memory_search
                )
                
                filtered_memories = self.filter_memories(
                    st.session_state.memories,
                    st.session_state.memory_search
                )
                
                st.caption(f"Showing {len(filtered_memories)} of {len(st.session_state.memories)} memories")
                
                for idx, memory in enumerate(filtered_memories):
                    with st.expander(f"Memory {len(st.session_state.memories) - idx}", expanded=False):
                        st.markdown(f"""
                        <div style="font-size: 0.9em;">
                            <div style="color: #666; margin-bottom: 0.5rem;">
                                {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                            </div>
                            {memory.to_str()}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if st.button("Copy", key=f"copy_{idx}"):
                            st.write("Memory copied to clipboard!")
                            st.experimental_set_query_params(clipboard=memory.to_str())

    def show_sidebar(self):
        """Display complete sidebar with all settings and features."""
        with st.sidebar:
            st.title("Settings")
            
            # Chat Settings
            st.subheader("Chat Settings")
            
            st.session_state.temperature = st.slider(
                "Temperature",
                min_value=0.1,
                max_value=1.0,
                value=0.0,
                step=0.1,
                help="Higher values make the output more random"
            )
            
            st.session_state.max_tokens = st.slider(
                "Max Response Length",
                min_value=100,
                max_value=8192,
                value=2048,
                step=100,
                help="Maximum number of tokens in the response"
            )
            
            # Actions
            st.subheader("Actions")
            if st.button("Clear History", type="primary"):
                st.session_state.messages = []
                st.experimental_rerun()
            
            # Show memory section
            self.show_memory_sidebar()
            
            # Export functionality
            if st.session_state.messages:
                self.export_chat_history()

    def export_chat_history(self):
        """Export chat history as JSON."""
        if not st.session_state.messages:
            st.sidebar.warning("No messages to export!")
            return
            
        chat_export = {
            "timestamp": datetime.now().isoformat(),
            "messages": st.session_state.messages,
            "memories": [m.model_dump() for m in st.session_state.memories] if st.session_state.memories else [],
            "stats": {
                "total_messages": len(st.session_state.messages),
                "duration_minutes": (datetime.now() - st.session_state.conversation_started).total_seconds() / 60
            }
        }
        
        st.sidebar.download_button(
            label="Download JSON",
            data=str(chat_export),
            file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    def validate_input(self, prompt: str) -> Tuple[bool, Optional[str]]:
        """Validate user input."""
        if not prompt.strip():
            return False, "Please enter a non-empty message"
        return True, None

    def display_chat_messages(self):
        """Display chat messages with timestamps."""
        if not st.session_state.messages:
            return

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(self.format_message_for_display(
                    message["role"],
                    message["content"],
                    message.get("timestamp")
                ))

    def format_message_for_display(self, role: str, content: str, timestamp: Optional[float] = None) -> str:
        """Format message with optional timestamp and styling."""
        formatted_message = content
        
        if hasattr(st.session_state, 'show_timestamps') and st.session_state.show_timestamps and timestamp:
            time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
            formatted_message += f"\n<div class='message-timestamp'>{time_str}</div>"
        
        return formatted_message

    def process_user_input(self):
        """Process user input with response time calculation and memory processing."""
        if prompt := st.chat_input("What's on your mind?"):
            is_valid, error_message = self.validate_input(prompt)
            if not is_valid:
                st.warning(error_message)
                return

            st.session_state.messages.append({
                "role": MessageRole.USER,
                "content": prompt,
                "timestamp": time.time()
            })

            with st.chat_message(MessageRole.USER):
                st.markdown(prompt)

            with st.chat_message(MessageRole.ASSISTANT):
                response_container = st.container()
                response_placeholder = response_container.empty()
                progress_container = response_container.empty()
                memory_placeholder = response_container.empty()
                full_response = ""

                try:
                    response_start_time = time.time()
                    st.session_state.last_response_time = response_start_time

                    with st.spinner("Thinking..."):
                        message_list = [
                            {"role": m["role"], "content": m["content"]} 
                            for m in st.session_state.messages
                        ]

                        for chunk in self.inference_engine.chat_completion(
                            message_list,
                            max_tokens=st.session_state.get("max_tokens", 2048),
                            temperature=st.session_state.get("temperature", 0.0)
                        ):
                            full_response += chunk
                            response_placeholder.markdown(full_response + "▌")

                    response_placeholder.markdown(full_response)
                    
                    response_time = time.time() - response_start_time
                    response_container.caption(f"Response time: {response_time:.2f}s")

                    st.session_state.messages.append({
                        "role": MessageRole.ASSISTANT,
                        "content": full_response,
                        "timestamp": time.time()
                    })

                    if st.session_state.memory_enabled:
                        progress_bar = progress_container.progress(0)
                        memory = asyncio.run(self.process_memory(
                            prompt, 
                            full_response,
                            progress_bar
                        ))
                        progress_container.empty()
                        
                        if memory:
                            memory_placeholder.markdown(
                                self.format_memory_display(memory),
                                unsafe_allow_html=True
                            )

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.session_state.messages.pop()


def main():
    """Main function to run the chat interface."""
    chat_ui = ChatUI()
    chat_ui.show_sidebar()
    chat_ui.display_chat_messages()
    chat_ui.process_user_input()


if __name__ == "__main__":
    main()