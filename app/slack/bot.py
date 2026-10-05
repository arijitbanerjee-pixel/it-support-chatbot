import os
import threading

import uvicorn
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from app.rag.chatbot import ITSupportChatbot


load_dotenv()

# Slack application
app = App(
    token=os.environ["SLACK_BOT_TOKEN"]
)

# Existing IT Support RAG chatbot
chatbot = ITSupportChatbot()


@app.event("app_mention")
def handle_app_mention(event, say, logger):

    try:
        # Remove the bot mention from the Slack message
        question = event.get("text", "")

        question = question.split(">", 1)[-1].strip()

        if not question:
            say(
                "Please provide an IT support question after mentioning me."
            )
            return

        # Ask the existing RAG chatbot
        result = chatbot.ask(question)

        answer = result["answer"]

        sources = result.get("sources", [])

        if sources:
            answer += "\n\nSources: " + ", ".join(sources)

        say(answer)

    except Exception:
        logger.exception("Error processing Slack question")

        say(
            "Sorry, I couldn't process your request. Please try again."
        )


def run_fastapi():
    """Run the HTTP server required by Cloud Run."""
    port = int(os.environ.get("PORT", "8080"))

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
    )


if __name__ == "__main__":

    # Start FastAPI in the background.
    threading.Thread(
        target=run_fastapi,
        daemon=True,
    ).start()

    print("⚡️ FastAPI server started!")
    print("⚡️ Bolt app is running!")

    # Keep Slack Socket Mode running.
    handler = SocketModeHandler(
        app,
        os.environ["SLACK_APP_TOKEN"],
    )

    handler.start()