import streamlit as st
from src.ui.chat_ui import ChatUI


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="mlx chat",
        page_icon="💬",
        layout="wide"
    )

    chat_ui = ChatUI()
    chat_ui.show_sidebar()
    chat_ui.display_chat_messages()
    chat_ui.process_user_input()

if __name__ == "__main__":
    main()