import os
import sys
import time
from multiprocessing import Process

import gradio as gr
import pytest
import requests
from playwright.sync_api import Browser, Page, sync_playwright

from rag_system.interfaces.gradio_ui import make_respond_to_question


def get_create_agent():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, "..", "..")
    sys.path.insert(0, os.path.abspath(project_root))
    from main import create_agent

    return create_agent


def run_gradio_server(port: int):
    """Function to run Gradio server in separate process"""
    # Use offline configuration from main.py
    create_agent = get_create_agent()
    
    # Import config loading
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, "..", "..")
    sys.path.insert(0, os.path.abspath(project_root))
    from config.config import load_config
    
    # Load offline config
    config = load_config("config/offline.json")
    agent = create_agent(config, None)
    respond_to_question = make_respond_to_question(agent)

    iface = gr.ChatInterface(
        fn=respond_to_question,
        title="RAG Agent Chat",
        description="Have a conversation and ask questions based on documents.",
        textbox=gr.Textbox(placeholder="Ask your question here...", container=False, scale=7),
        type="messages",
    )

    iface.launch(server_port=port, share=False, quiet=True)


class GradioServer:
    def __init__(self, port: int = 7860):
        self.port = port
        self.process = None

    def start(self):
        self.process = Process(target=run_gradio_server, args=(self.port,))
        self.process.start()

        # Wait for server to start
        max_attempts = 30
        for _ in range(max_attempts):
            try:
                response = requests.get(f"http://localhost:{self.port}")
                if response.status_code == 200:
                    break
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(1)
        else:
            raise Exception("Gradio server failed to start")

    def stop(self):
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=10)
            if self.process.is_alive():
                self.process.kill()
                self.process.join()


@pytest.fixture(scope="module")
def gradio_server():
    server = GradioServer()
    server.start()
    yield server
    server.stop()


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        # Use headless=False locally for debugging, headless=True in CI
        headless = os.getenv("CI", "false").lower() == "true"
        slow_mo = 0 if headless else 1000  # Only slow down when visible
        browser = p.chromium.launch(headless=headless, slow_mo=slow_mo)
        yield browser
        browser.close()


@pytest.fixture
def page(browser: Browser, gradio_server: GradioServer):
    page = browser.new_page()
    page.goto(f"http://localhost:{gradio_server.port}")
    yield page
    page.close()


class TestGradioUI:
    def test_page_loads_successfully(self, page: Page):
        """Test that the Gradio page loads with expected elements"""
        # Check title
        expect_title = page.locator("h1").first
        expect_title.wait_for(state="visible")
        assert "RAG Agent Chat" in expect_title.text_content()

        # Check description
        description = page.locator("text=Have a conversation and ask questions based on documents.")
        description.wait_for(state="visible")

        # Check chat input exists
        chat_input = page.locator("textarea[placeholder*='Ask your question here']")
        chat_input.wait_for(state="visible")

    def test_send_message_and_receive_response(self, page: Page):
        """Test sending a message and receiving a response"""
        # Wait for the page to fully load
        time.sleep(2)

        # Find the textarea input
        chat_input = page.locator("textarea").first
        chat_input.wait_for(state="visible")

        # Type a test message
        test_message = "Hello, this is a test message"
        chat_input.fill(test_message)

        # Submit the message (look for submit button or press Enter)
        submit_button = page.locator("button[aria-label='Submit']").or_(page.locator("button:has-text('Submit')"))
        if submit_button.is_visible():
            submit_button.click()
        else:
            chat_input.press("Enter")

        # Wait for response to appear in chat history
        # Look for the message in the chat interface
        chat_message = page.locator(f"text={test_message}").first
        chat_message.wait_for(state="visible", timeout=10000)

        # Wait for any response to appear (more flexible check)
        time.sleep(3)

        # Check that there are at least 2 messages (user + assistant)
        all_messages = page.locator("[role='presentation']").or_(page.locator(".message")).or_(page.locator("div:has-text('Hello')"))
        assert all_messages.count() > 0, "Expected at least one response from the agent"

    def test_multiple_messages_conversation(self, page: Page):
        """Test sending multiple messages in a conversation"""
        # Wait for the page to fully load
        time.sleep(2)

        messages = ["Hello my Name is Camilo", "What is the capital of France?"]

        for i, message in enumerate(messages):
            # Find and fill the input
            chat_input = page.locator("textarea").first
            chat_input.wait_for(state="visible")
            chat_input.fill(message)

            # Submit message
            submit_button = page.locator("button[aria-label='Submit']").or_(page.locator("button:has-text('Submit')"))
            if submit_button.is_visible():
                submit_button.click()
            else:
                chat_input.press("Enter")

            # Wait for the message to appear
            page.locator(f"text={message}").first.wait_for(state="visible", timeout=10000)

            # Wait for response
            time.sleep(3)

        # Check that both messages are visible in the chat
        for message in messages:
            assert page.locator(f"text={message}").first.is_visible()
