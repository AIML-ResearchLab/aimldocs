![langgraph](image/langgraph.png)

# LangGraph
LangGraph is a low-level orchestration framework for building controllable agents.
While langchain provides integrations and composable components to streamline LLM application development, the LangGraph library enables agent orchestration — offering customizable architectures, long-term memory, and human-in-the-loop to reliably handle complex tasks.

# Install the Library
```
pip install -U langgraph
pip install -U langchain-anthropic
```

# Simple example below of how to create a ReAct agent.

```
# This code depends on pip install langchain[anthropic]
from langgraph.prebuilt import create_react_agent
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Retrieve API tokens from .env
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

def search(query: str):
    """Call to surf the web."""
    if "sf" in query.lower() or "san francisco" in query.lower():
        return "It's 60 degrees and foggy."
    return "It's 90 degrees and sunny."

agent = create_react_agent("anthropic:claude-3-7-sonnet-latest", tools=[search])
agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
```

# Why use LangGraph?
LangGraph is useful for building **robust, modular, and scalable AI agents**.It extends LangChain with graph-based execution, making it ideal for **multi-agent workflows, streaming, and fine-grained control**.

Developers choose LangGraph for:

1. **Reliability and controllability:** 
    ✅ Ensures **structured execution** of agent tasks.
    ✅ Supports **moderation checks**, human approvals, and context persistence for long-running workflows.

2. **Low-level and extensible:** 
    ✅ Provides **full control** over agent behavior using **custom nodes and state management**.
    ✅ Ideal for **multi-agent collaboration**, where each agent has a defined role.

3. **First-Class Streaming Support:**
    ✅ Supports **token-by-token streaming**, making it great for **real-time insights** into agent decisions.
    ✅ Allows **intermediate step streaming**, improving **observability and debugging**.

# Where is LangGraph Useful?

✅ **Multi-Agent Systems:** When you need **multiple specialized agents** working together.
✅ **Long-Running Workflows:** If your agents need **context persistence** over time.
✅ **Interactive Applications:** When **streaming responses** improve user experience.

LangGraph can be useful for designing complex AI pipelines where different agents handle different tasks while **maintaining control and visibility**. 

LangGraph is already **trusted in production** by major companies for AI-powered automation, making it a solid choice for building **scalable, reliable, and controllable AI agents**.

# Real-World Use Cases of LangGraph

✅ **Klarna → Customer Support Bot**
    - Handles **85 million** active users.
    - Manages customer inquiries with **multi-step workflows and automation**.
✅ **Elastic → Security AI Assistant**
    - Helps with **threat detection and security analysis**.
    - Uses **multi-agent collaboration** for investigating security alerts.
✅ **Uber → Automated Unit Test Generation**
    - Generates and refines **unit tests for developers**.
    - Uses LangGraph for **agent-based coding assistants**.
✅ **Replit → AI-Powered Code Generation**
    - Assists developers in **writing, debugging, and optimizing code**.
    - Uses LangGraph’s **streaming and multi-agent capabilities**.

# LangGraph’s Ecosystem & Integrations
LangGraph works **standalone** but integrates seamlessly with **LangChain tools**, making it easier to build, evaluate, and deploy AI agents.

# Key Integrations for Better LLM Application Development

✅ **LangSmith** (Agent Evaluation & Debugging)
    - Debugs **poor-performing LLM runs** and optimizes workflows.
    - Evaluates **agent trajectories** to improve decision-making.
    - Provides **observability** in production.
✅ LangGraph Platform (Scaling & Deployment)
    - Deploys **long-running, stateful AI agents** at scale.
    - Allows **agent discovery, reuse, and configuration** across teams.
    - Features **LangGraph Studio** for **visual prototyping** and fast iteration.

# To integrate LangGraph + LangSmith into your AI projects

1️⃣ **Set Up LangGraph with LangSmith for Debugging & Observability**
    ✅ **Install Dependencies**
        ✅ First, install LangGraph, LangSmith, and LangChain:
        ```
        pip install langgraph langsmith langchain
        ```
    
    ✅ **Set Up LangSmith API Key**
        Sign up for LangSmith at smith.langchain.com and get your API key.
        ```
        https://smith.langchain.com/
        LANGCHAIN_API_KEY="your_actual_api_key"
        ```
        Then, set it in your environment:
    
    ✅ **Enable Debugging for Agents**
      ```
        # This code depends on pip install langchain[anthropic]
        from langgraph.prebuilt import create_react_agent
        import os
        from dotenv import load_dotenv

        from langchain_openai import ChatOpenAI
        from langsmith import traceable


        # Load environment variables
        load_dotenv()

        LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "default"

        # Define the function
        @traceable
        def search(query: str):
            """Call to surf the web."""
            if "sf" in query.lower() or "san francisco" in query.lower():
                return "It's 60 degrees and foggy."
            return "It's 90 degrees and sunny."

        # Use OpenAI's GPT model
        llm = ChatOpenAI(model="gpt-4-turbo")

        # Create the agent
        agent = create_react_agent(llm, tools=[search])

        # Invoke the agent
        response = agent.invoke({"messages": [{"role": "user", "content": "What is the weather in SF?"}]})

        print(response)
       ```

       🔹 Now, all agent runs will be logged in LangSmith for debugging.

2️⃣ **Visualizing & Debugging Agent Trajectories in LangSmith**
Once the agent is running, go to LangSmith UI and check:
✅ Logs of each agent action (inputs, outputs, reasoning).
✅ Failure points in decision-making.
✅ Performance metrics to optimize.

![langsmith](image/langsmith.png)

![langsmith](image/langsmith-1.png)


3️⃣ **Scaling with LangGraph Platform (Long-Running Agents & Deployment)**
    ✅ To make your AI **stateful and scalable**, use **LangGraph Platform:**

    ```
    pip install langgraph[platform]
    ```

✅ Deploy **long-running agents** with **stateful memory**.
✅ Use **LangGraph Studio** for **drag-and-drop workflow design**.
✅ Share & configure agents **across teams**.




