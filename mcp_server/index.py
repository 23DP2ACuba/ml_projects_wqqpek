import asyncio
import random
import nest_asyncio
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import streamlit as st
from utils.models import create_llm
from utils.agent import create_history, ask
from utils.tools import load_tools
from utils.client import connect_to_server

nest_asyncio.apply()

LOADING_MESSAGES = [
    "🤔 Thinking deeply about your request...",
    "✨ Crafting a perfect response...",
    "💭 Processing your productivity goals...",
    "🔄 Organizing your task list...",
    "🎯 Planning the perfect task management strategy...",
]

async def get_response_async(user_query: str, history: list, llm):

    try:
        async with connect_to_server() as session:
            tools = await load_tools(session)
            response_content = await ask(user_query, history.copy(), llm, tools)
            print(response_content)
            return response_content
    except Exception as e:
        print(f"Error in get_response_async: {e}")
        return f"Sorry, I encountered an error: {str(e)}"

def main():
    st.set_page_config(
        page_title="Productivity Assistant",
        page_icon="🎯",
        layout="wide"
    )
    
    if "messages" not in st.session_state:
        st.session_state.messages = create_history()
    
    llm = create_llm()
    
    st.title("🎯 Productivity Assistant")
    st.markdown("Your AI-powered task management companion")
    
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
    
    if prompt := st.chat_input("What would you like to do with your tasks?"):
        st.session_state.messages.append(HumanMessage(content=prompt))
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.status(random.choice(LOADING_MESSAGES), state="running")
            
            response = asyncio.run(
                get_response_async(prompt, st.session_state.messages, llm)
            )
            
            message_placeholder.markdown(response)
            message_placeholder.status("", state="complete")
        
        st.session_state.messages.append(AIMessage(content=response))

if __name__ == "__main__":
    main()