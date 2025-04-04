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
    - Ensures **structured execution** of agent tasks.
    - Supports **moderation checks**, human approvals, and context persistence for long-running workflows.

2. **Low-level and extensible:** 
    - Provides **full control** over agent behavior using **custom nodes and state management**.
    - Ideal for **multi-agent collaboration**, where each agent has a defined role.

3. **First-Class Streaming Support:**
    - Supports **token-by-token streaming**, making it great for **real-time insights** into agent decisions.
    - Allows **intermediate step streaming**, improving **observability and debugging**.

# Where is LangGraph Useful?

- **Multi-Agent Systems:** When you need **multiple specialized agents** working together.
- **Long-Running Workflows:** If your agents need **context persistence** over time.
- **Interactive Applications:** When **streaming responses** improve user experience.

LangGraph can be useful for designing complex AI pipelines where different agents handle different tasks while **maintaining control and visibility**. 

LangGraph is already **trusted in production** by major companies for AI-powered automation, making it a solid choice for building **scalable, reliable, and controllable AI agents**.

# Real-World Use Cases of LangGraph

- **Klarna → Customer Support Bot**
    - Handles **85 million** active users.
    - Manages customer inquiries with **multi-step workflows and automation**.
- **Elastic → Security AI Assistant**
    - Helps with **threat detection and security analysis**.
    - Uses **multi-agent collaboration** for investigating security alerts.
- **Uber → Automated Unit Test Generation**
    - Generates and refines **unit tests for developers**.
    - Uses LangGraph for **agent-based coding assistants**.
- **Replit → AI-Powered Code Generation**
    - Assists developers in **writing, debugging, and optimizing code**.
    - Uses LangGraph’s **streaming and multi-agent capabilities**.

# LangGraph’s Ecosystem & Integrations
LangGraph works **standalone** but integrates seamlessly with **LangChain tools**, making it easier to build, evaluate, and deploy AI agents.

# Key Integrations for Better LLM Application Development

- **LangSmith (Agent Evaluation & Debugging)**
    - Debugs **poor-performing LLM runs** and optimizes workflows.
    - Evaluates **agent trajectories** to improve decision-making.
    - Provides **observability** in production.

- **LangGraph Platform (Scaling & Deployment)**
    - Deploys **long-running, stateful AI agents** at scale.
    - Allows **agent discovery, reuse, and configuration** across teams.
    - Features **LangGraph Studio** for **visual prototyping** and fast iteration.

# To integrate LangGraph + LangSmith into your AI projects

**Set Up LangGraph with LangSmith for Debugging & Observability**
- **Install Dependencies**
    - First, install LangGraph, LangSmith, and LangChain:

```
        pip install langgraph langsmith langchain
```
    
- **Set Up LangSmith API Key**
    Sign up for LangSmith at smith.langchain.com and get your API key.
```
        https://smith.langchain.com/
        LANGCHAIN_API_KEY="your_actual_api_key"
```
Then, set it in your environment:
    
- **Enable Debugging for Agents**

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

- Now, all agent runs will be logged in LangSmith for debugging.

**Visualizing & Debugging Agent Trajectories in LangSmith**

Once the agent is running, go to LangSmith UI and check:

- Logs of each agent action (inputs, outputs, reasoning).
- Failure points in decision-making.
- Performance metrics to optimize.

![langsmith](image/langsmith.png)

![langsmith](image/langsmith-1.png)


**Scaling with LangGraph Platform (Long-Running Agents & Deployment)**

- To make your AI **stateful and scalable**, use **LangGraph Platform:**

```
pip install langgraph[platform]
```

- Deploy **long-running agents** with **stateful memory**.
- Use **LangGraph Studio** for **drag-and-drop workflow design**.
- Share & configure agents **across teams**.


# Graph API Basics
## How to update graph state from nodes

**Define state**
State in **LangGraph** can be a **TypedDict**, **Pydantic** model, or dataclass. 

**State:**
The first thing you do when you define a graph is define the State of the graph. The State consists of the schema of the graph as well as reducer functions which specify how to apply updates to the state.
The schema of the State will be the input schema to all Nodes and Edges in the graph, and can be either a TypedDict or a Pydantic model. All Nodes will emit updates to the State which are then applied using the specified reducer function.

**Schema:**
The main documented way to specify the schema of a graph is by using TypedDict. However, we also support using a Pydantic BaseModel as your graph state to add default values and additional data validation.

## How to use Pydantic model as graph state

**First we need to install the packages required**

```
%%capture --no-stderr
%pip install --quiet -U langgraph
```

```
import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("OPENAI_API_KEY")
```

## Input Validation

```
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from pydantic import BaseModel


# The overall state of the graph (this is the public state shared across nodes)
class OverallState(BaseModel):
    a: str


def node(state: OverallState):
    return {"a": "goodbye"}


# Build the state graph
builder = StateGraph(OverallState)
builder.add_node(node)  # node_1 is the first node
builder.add_edge(START, "node")  # Start the graph with node_1
builder.add_edge("node", END)  # End the graph after node_1
graph = builder.compile()

# Test the graph with a valid input
graph.invoke({"a": "hello"})
```

**Output:**

```
{'a': 'goodbye'}
```

**Invoke the graph with an invalid input**

```
try:
    graph.invoke({"a": 123})  # Should be a string
except Exception as e:
    print("An exception was raised because `a` is an integer rather than a string.")
    print(e)
```

```
An exception was raised because `a` is an integer rather than a string.
1 validation error for OverallState
a
  Input should be a valid string [type=string_type, input_value=123, input_type=int]
    For further information visit https://errors.pydantic.dev/2.9/v/string_type
```

## Multiple Nodes

Run-time validation will also work in a multi-node graph. In the example below bad_node updates a to an integer.


Because run-time validation occurs on inputs, the validation error will occur when ok_node is called (not when bad_node returns an update to the state which is inconsistent with the schema).

```
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from pydantic import BaseModel


# The overall state of the graph (this is the public state shared across nodes)
class OverallState(BaseModel):
    a: str


def bad_node(state: OverallState):
    return {
        "a": 123  # Invalid
    }


def ok_node(state: OverallState):
    return {"a": "goodbye"}


# Build the state graph
builder = StateGraph(OverallState)
builder.add_node(bad_node)
builder.add_node(ok_node)
builder.add_edge(START, "bad_node")
builder.add_edge("bad_node", "ok_node")
builder.add_edge("ok_node", END)
graph = builder.compile()

# Test the graph with a valid input
try:
    graph.invoke({"a": "hello"})
except Exception as e:
    print("An exception was raised because bad_node sets `a` to an integer.")
    print(e)
```

**Output:**

```
An exception was raised because bad_node sets `a` to an integer.
1 validation error for OverallState
a
  Input should be a valid string [type=string_type, input_value=123, input_type=int]
    For further information visit https://errors.pydantic.dev/2.9/v/string_type
```

## Advanced Pydantic Model Usage
This section covers more advanced topics when using Pydantic models with LangGraph.

**Serialization Behavior**

When using Pydantic models as state schemas, it's important to understand how serialization works, especially when: - Passing Pydantic objects as inputs - Receiving outputs from the graph - Working with nested Pydantic models


```
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel


class NestedModel(BaseModel):
    value: str


class ComplexState(BaseModel):
    text: str
    count: int
    nested: NestedModel


def process_node(state: ComplexState):
    # Node receives a validated Pydantic object
    print(f"Input state type: {type(state)}")
    print(f"Nested type: {type(state.nested)}")

    # Return a dictionary update
    return {"text": state.text + " processed", "count": state.count + 1}


# Build the graph
builder = StateGraph(ComplexState)
builder.add_node("process", process_node)
builder.add_edge(START, "process")
builder.add_edge("process", END)
graph = builder.compile()

# Create a Pydantic instance for input
input_state = ComplexState(text="hello", count=0, nested=NestedModel(value="test"))
print(f"Input object type: {type(input_state)}")

# Invoke graph with a Pydantic instance
result = graph.invoke(input_state)
print(f"Output type: {type(result)}")
print(f"Output content: {result}")

# Convert back to Pydantic model if needed
output_model = ComplexState(**result)
print(f"Converted back to Pydantic: {type(output_model)}")
```

**Runtime Type Coercion**
Pydantic performs runtime type coercion for certain data types. This can be helpful but also lead to unexpected behavior if you're not aware of it.

```
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel


class CoercionExample(BaseModel):
    # Pydantic will coerce string numbers to integers
    number: int
    # Pydantic will parse string booleans to bool
    flag: bool


def inspect_node(state: CoercionExample):
    print(f"number: {state.number} (type: {type(state.number)})")
    print(f"flag: {state.flag} (type: {type(state.flag)})")
    return {}


builder = StateGraph(CoercionExample)
builder.add_node("inspect", inspect_node)
builder.add_edge(START, "inspect")
builder.add_edge("inspect", END)
graph = builder.compile()

# Demonstrate coercion with string inputs that will be converted
result = graph.invoke({"number": "42", "flag": "true"})

# This would fail with a validation error
try:
    graph.invoke({"number": "not-a-number", "flag": "true"})
except Exception as e:
    print(f"\nExpected validation error: {e}")
```

**Working with Message Models**
When working with LangChain message types in your state schema, there are important considerations for serialization. You should use AnyMessage (rather than BaseMessage) for proper serialization/deserialization when using message objects over the wire:

```
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
from typing import List


class ChatState(BaseModel):
    messages: List[AnyMessage]
    context: str


def add_message(state: ChatState):
    return {"messages": state.messages + [AIMessage(content="Hello there!")]}


builder = StateGraph(ChatState)
builder.add_node("add_message", add_message)
builder.add_edge(START, "add_message")
builder.add_edge("add_message", END)
graph = builder.compile()

# Create input with a message
initial_state = ChatState(
    messages=[HumanMessage(content="Hi")], context="Customer support chat"
)

result = graph.invoke(initial_state)
print(f"Output: {result}")

# Convert back to Pydantic model to see message types
output_model = ChatState(**result)
for i, msg in enumerate(output_model.messages):
    print(f"Message {i}: {type(msg).__name__} - {msg.content}")
```

## Graphs
At its core, LangGraph models agent workflows as graphs. You define the behavior of your agents using three key components:

1. **State:** A shared data structure that represents the current snapshot of your application. It can be any Python type, but is typically a TypedDict or Pydantic BaseModel.

2. **Nodes:** Python functions that encode the logic of your agents. They receive the current State as input, perform some computation or side-effect, and return an updated State.

3. **Edges:** Python functions that determine which Node to execute next based on the current State. They can be conditional branches or fixed transitions.

By composing Nodes and Edges, you can create complex, looping workflows that evolve the State over time. The real power, though, comes from how LangGraph manages that State.

To emphasize: Nodes and Edges are nothing more than Python functions - they can contain an LLM or just good ol' Python code.

**In short:** **nodes** do the work. **edges** tell what to do next.

LangGraph's underlying graph algorithm uses message passing to define a general program. When a Node completes its operation, it sends messages along one or more edges to other node(s).

These recipient nodes then execute their functions, pass the resulting messages to the next set of nodes, and the process continues.

A super-step can be considered a single iteration over the graph nodes. Nodes that run in parallel are part of the same super-step, while nodes that run sequentially belong to separate super-steps. At the start of graph execution, all nodes begin in an inactive state. A node becomes active when it receives a new message (state) on any of its incoming edges (or "channels"). The active node then runs its function and responds with updates. At the end of each super-step, nodes with no incoming messages vote to halt by marking themselves as inactive. The graph execution terminates when all nodes are inactive and no messages are in transit.

**StateGraph:**
The StateGraph class is the main graph class to use. This is parameterized by a user defined State object.

**Compiling your graph:**
To build your graph, you first define the **state**, you then add **nodes** and **edges**, and then you compile it. What exactly is compiling your graph and why is it needed?

Compiling is a pretty simple step. It provides a few basic checks on the structure of your graph (no orphaned nodes, etc). It is also where you can specify runtime args like **checkpointers** and **breakpoints**. You compile your graph by just calling the **.compile** method:

```
graph = graph_builder.compile(...)
```

**Note:** You MUST compile your graph before you can use it.

**Multiple schemas**
Typically, all graph nodes communicate with a single schema. This means that they will read and write to the same state channels. But, there are cases where we want more control over this:

- Internal nodes can pass information that is not required in the graph's input / output.
- We may also want to use different input / output schemas for the graph. The output might, for example, only contain a single relevant output key.

It is possible to have nodes write to private state channels inside the graph for internal node communication. We can simply define a private schema, **PrivateState**.

**Let's look at an example:**

```
class InputState(TypedDict):
    user_input: str

class OutputState(TypedDict):
    graph_output: str

class OverallState(TypedDict):
    foo: str
    user_input: str
    graph_output: str

class PrivateState(TypedDict):
    bar: str

def node_1(state: InputState) -> OverallState:
    # Write to OverallState
    return {"foo": state["user_input"] + " name"}

def node_2(state: OverallState) -> PrivateState:
    # Read from OverallState, write to PrivateState
    return {"bar": state["foo"] + " is"}

def node_3(state: PrivateState) -> OutputState:
    # Read from PrivateState, write to OutputState
    return {"graph_output": state["bar"] + " Lance"}

builder = StateGraph(OverallState,input=InputState,output=OutputState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", "node_3")
builder.add_edge("node_3", END)

graph = builder.compile()
graph.invoke({"user_input":"My"})
{'graph_output': 'My name is Lance'}
```

There are two subtle and important points to note here:

1. We pass state: InputState as the input schema to node_1. But, we write out to foo, a channel in OverallState. How can we write out to a state channel that is not included in the input schema? This is because a node can write to any state channel in the graph state. The graph state is the union of of the state channels defined at initialization, which includes OverallState and the filters InputState and OutputState.

2. We initialize the graph with StateGraph(OverallState,input=InputState,output=OutputState). So, how can we write to PrivateState in node_2? How does the graph gain access to this schema if it was not passed in the StateGraph initialization? We can do this because nodes can also declare additional state channels as long as the state schema definition exists. In this case, the PrivateState schema is defined, so we can add bar as a new state channel in the graph and write to it.

## Reducers
Reducers are key to understanding how updates from nodes are applied to the State. Each key in the State has its own independent reducer function. If no reducer function is explicitly specified then it is assumed that all updates to that key should override it. There are a few different types of reducers, starting with the default type of reducer:

**Default Reducer**

These two examples show how to use the default reducer:

**Example A:**

```
from typing_extensions import TypedDict

class State(TypedDict):
    foo: int
    bar: list[str]
```

In this example, no reducer functions are specified for any key. Let's assume the input to the graph is {"foo": 1, "bar": ["hi"]}. Let's then assume the first Node returns {"foo": 2}. This is treated as an update to the state. Notice that the Node does not need to return the whole State schema - just an update. After applying this update, the State would then be {"foo": 2, "bar": ["hi"]}. If the second node returns {"bar": ["bye"]} then the State would then be {"foo": 2, "bar": ["bye"]}


**Example B:**

```
from typing import Annotated
from typing_extensions import TypedDict
from operator import add

class State(TypedDict):
    foo: int
    bar: Annotated[list[str], add]
```

In this example, we've used the Annotated type to specify a reducer function (operator.add) for the second key (bar). Note that the first key remains unchanged. Let's assume the input to the graph is {"foo": 1, "bar": ["hi"]}. Let's then assume the first Node returns {"foo": 2}. This is treated as an update to the state. Notice that the Node does not need to return the whole State schema - just an update. After applying this update, the State would then be {"foo": 2, "bar": ["hi"]}. If the second node returns {"bar": ["bye"]} then the State would then be {"foo": 2, "bar": ["hi", "bye"]}. Notice here that the bar key is updated by adding the two lists together.


## Working with Messages in Graph State

**Why use messages?**

Most modern LLM providers have a chat model interface that accepts a list of messages as input. LangChain's ChatModel in particular accepts a list of Message objects as inputs. These messages come in a variety of forms such as HumanMessage (user input) or AIMessage (LLM response). To read more about what message objects are, please refer to this conceptual guide.

**Using Messages in your Graph**
In many cases, it is helpful to store prior conversation history as a list of messages in your graph state. To do so, we can add a key (channel) to the graph state that stores a list of Message objects and annotate it with a reducer function (see messages key in the example below). The reducer function is vital to telling the graph how to update the list of Message objects in the state with each state update (for example, when a node sends an update). If you don't specify a reducer, every state update will overwrite the list of messages with the most recently provided value. If you wanted to simply append messages to the existing list, you could use operator.add as a reducer.

However, you might also want to manually update messages in your graph state (e.g. human-in-the-loop). If you were to use operator.add, the manual state updates you send to the graph would be appended to the existing list of messages, instead of updating existing messages. To avoid that, you need a reducer that can keep track of message IDs and overwrite existing messages, if updated. To achieve this, you can use the prebuilt add_messages function. For brand new messages, it will simply append to existing list, but it will also handle the updates for existing messages correctly.

**Serialization**
In addition to keeping track of message IDs, the add_messages function will also try to deserialize messages into LangChain Message objects whenever a state update is received on the messages channel. See more information on LangChain serialization/deserialization here. This allows sending graph inputs / state updates in the following format:

```
# this is supported
{"messages": [HumanMessage(content="message")]}

# and this is also supported
{"messages": [{"type": "human", "content": "message"}]}
```

Since the state updates are always deserialized into LangChain Messages when using add_messages, you should use dot notation to access message attributes, like state["messages"][-1].content. Below is an example of a graph that uses add_messages as it's reducer function.

```
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict

class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

**MessagesState**

Since having a list of messages in your state is so common, there exists a prebuilt state called MessagesState which makes it easy to use messages. MessagesState is defined with a single messages key which is a list of AnyMessage objects and uses the add_messages reducer. Typically, there is more state to track than just messages, so we see people subclass this state and add more fields, like:

```
from langgraph.graph import MessagesState

class State(MessagesState):
    documents: list[str]
```

## Nodes
In LangGraph, nodes are typically python functions (sync or async) where the first positional argument is the state, and (optionally), the second positional argument is a "config", containing optional configurable parameters (such as a thread_id).

Similar to NetworkX, you add these nodes to a graph using the add_node method:

```
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

builder = StateGraph(dict)


def my_node(state: dict, config: RunnableConfig):
    print("In node: ", config["configurable"]["user_id"])
    return {"results": f"Hello, {state['input']}!"}


# The second argument is optional
def my_other_node(state: dict):
    return state


builder.add_node("my_node", my_node)
builder.add_node("other_node", my_other_node)
...
```

Behind the scenes, functions are converted to RunnableLambdas, which add batch and async support to your function, along with native tracing and debugging.

If you add a node to a graph without specifying a name, it will be given a default name equivalent to the function name.

```
builder.add_node(my_node)
# You can then create edges to/from this node by referencing it as `"my_node"`
```

## START Node
The START Node is a special node that represents the node that sends user input to the graph. The main purpose for referencing this node is to determine which nodes should be called first.

```
from langgraph.graph import START

graph.add_edge(START, "node_a")
```

## END Node
The END Node is a special node that represents a terminal node. This node is referenced when you want to denote which edges have no actions after they are done.

```
from langgraph.graph import END

graph.add_edge("node_a", END)
```

## Edges
Edges define how the logic is routed and how the graph decides to stop. This is a big part of how your agents work and how different nodes communicate with each other. There are a few key types of edges:

- **Normal Edges:** Go directly from one node to the next.
- **Conditional Edges:** Call a function to determine which node(s) to go to next.
- **Entry Point:** Which node to call first when user input arrives.
- **Conditional Entry Point:** Call a function to determine which node(s) to call first when user input arrives.

A node can have MULTIPLE outgoing edges. If a node has multiple out-going edges, all of those destination nodes will be executed in parallel as a part of the next superstep.

**Normal Edges:**
If you always want to go from node A to node B, you can use the add_edge method directly.

```
graph.add_edge("node_a", "node_b")
```

**Conditional Edges:**
If you want to optionally route to 1 or more edges (or optionally terminate), you can use the add_conditional_edges method. This method accepts the name of a node and a "routing function" to call after that node is executed:

```
graph.add_conditional_edges("node_a", routing_function)
```

Similar to nodes, the routing_function accepts the current state of the graph and returns a value.

By default, the return value routing_function is used as the name of the node (or list of nodes) to send the state to next. All those nodes will be run in parallel as a part of the next superstep.

You can optionally provide a dictionary that maps the routing_function's output to the name of the next node.

```
graph.add_conditional_edges("node_a", routing_function, {True: "node_b", False: "node_c"})
```

**Note:** Use Command instead of conditional edges if you want to combine state updates and routing in a single function.

**Command**
It can be useful to combine control flow (edges) and state updates (nodes). For example, you might want to BOTH perform state updates AND decide which node to go to next in the SAME node. LangGraph provides a way to do so by returning a Command object from node functions:

```
def my_node(state: State) -> Command[Literal["my_other_node"]]:
    return Command(
        # state update
        update={"foo": "bar"},
        # control flow
        goto="my_other_node"
    )
```

With Command you can also achieve dynamic control flow behavior (identical to conditional edges):

```
def my_node(state: State) -> Command[Literal["my_other_node"]]:
    if state["foo"] == "bar":
        return Command(update={"foo": "baz"}, goto="my_other_node")
```


**Important**
When returning Command in your node functions, you must add return type annotations with the list of node names the node is routing to, e.g. Command[Literal["my_other_node"]]. This is necessary for the graph rendering and tells LangGraph that my_node can navigate to my_other_node.


## When should I use Command instead of conditional edges?
Use Command when you need to both update the graph state and route to a different node. For example, when implementing multi-agent handoffs where it's important to route to a different agent and pass some information to that agent.

Use conditional edges to route between nodes conditionally without updating the state.

## Navigating to a node in a parent graph
If you are using subgraphs, you might want to navigate from a node within a subgraph to a different subgraph (i.e. a different node in the parent graph). To do so, you can specify graph=Command.PARENT in Command:

```
def my_node(state: State) -> Command[Literal["other_subgraph"]]:
    return Command(
        update={"foo": "bar"},
        goto="other_subgraph",  # where `other_subgraph` is a node in the parent graph
        graph=Command.PARENT
    )
```

**Note:** Setting graph to Command.PARENT will navigate to the closest parent graph.

**State updates with Command.PARENT**
When you send updates from a subgraph node to a parent graph node for a key that's shared by both parent and subgraph state schemas, you must define a reducer for the key you're updating in the parent graph state. See this example.

## Using inside tools
A common use case is updating graph state from inside a tool. For example, in a customer support application you might want to look up customer information based on their account number or ID in the beginning of the conversation. To update the graph state from the tool, you can return Command(update={"my_custom_key": "foo", "messages": [...]}) from the tool:

```
@tool
def lookup_user_info(tool_call_id: Annotated[str, InjectedToolCallId], config: RunnableConfig):
    """Use this to look up user information to better assist them with their questions."""
    user_info = get_user_info(config.get("configurable", {}).get("user_id"))
    return Command(
        update={
            # update the state keys
            "user_info": user_info,
            # update the message history
            "messages": [ToolMessage("Successfully looked up user information", tool_call_id=tool_call_id)]
        }
    )
```


## Human-in-the-loop
Command is an important part of human-in-the-loop workflows: when using interrupt() to collect user input, Command is then used to supply the input and resume execution via Command(resume="User input"). Check out this conceptual guide for more information.

## Persistence
LangGraph provides built-in persistence for your agent's state using checkpointers. Checkpointers save snapshots of the graph state at every superstep, allowing resumption at any time. This enables features like human-in-the-loop interactions, memory management, and fault-tolerance. You can even directly manipulate a graph's state after its execution using the appropriate get and update methods. For more details, see the persistence conceptual guide.

## Threads
Threads in LangGraph represent individual sessions or conversations between your graph and a user. When using checkpointing, turns in a single conversation (and even steps within a single graph execution) are organized by a unique thread ID.

## Storage
LangGraph provides built-in document storage through the BaseStore interface. Unlike checkpointers, which save state by thread ID, stores use custom namespaces for organizing data. This enables cross-thread persistence, allowing agents to maintain long-term memories, learn from past interactions, and accumulate knowledge over time. Common use cases include storing user profiles, building knowledge bases, and managing global preferences across all threads.


## Graph Migrations
LangGraph can easily handle migrations of graph definitions (nodes, edges, and state) even when using a checkpointer to track state.

- For threads at the end of the graph (i.e. not interrupted) you can change the entire topology of the graph (i.e. all nodes and edges, remove, add, rename, etc)
- For threads currently interrupted, we support all topology changes other than renaming / removing nodes (as that thread could now be about to enter a node that no longer exists) -- if this is a blocker please reach out and we can prioritize a solution.
- For modifying state, we have full backwards and forwards compatibility for adding and removing keys
- State keys that are renamed lose their saved state in existing threads
- State keys whose types change in incompatible ways could currently cause issues in threads with state from before the change -- if this is a blocker please reach out and we can prioritize a solution.

## Configuration
When creating a graph, you can also mark that certain parts of the graph are configurable. This is commonly done to enable easily switching between models or system prompts. This allows you to create a single "cognitive architecture" (the graph) but have multiple different instance of it.

You can optionally specify a config_schema when creating a graph.

```
class ConfigSchema(TypedDict):
    llm: str

graph = StateGraph(State, config_schema=ConfigSchema)
```

You can then pass this configuration into the graph using the configurable config field.

```
config = {"configurable": {"llm": "anthropic"}}

graph.invoke(inputs, config=config)
```

You can then access and use this configuration inside a node:

```
def node_a(state, config):
    llm_type = config.get("configurable", {}).get("llm", "openai")
    llm = get_llm(llm_type)
    ...
```

## Recursion Limit
The recursion limit sets the maximum number of super-steps the graph can execute during a single execution. Once the limit is reached, LangGraph will raise GraphRecursionError. By default this value is set to 25 steps. The recursion limit can be set on any graph at runtime, and is passed to .invoke/.stream via the config dictionary. Importantly, recursion_limit is a standalone config key and should not be passed inside the configurable key as all other user-defined configuration. See the example below:

```
graph.invoke(inputs, config={"recursion_limit": 5, "configurable":{"llm": "anthropic"}})
```

## interrupt
Use the interrupt function to pause the graph at specific points to collect user input. The interrupt function surfaces interrupt information to the client, allowing the developer to collect user input, validate the graph state, or make decisions before resuming execution.

```
from langgraph.types import interrupt

def human_approval_node(state: State):
    ...
    answer = interrupt(
        # This value will be sent to the client.
        # It can be any JSON serializable value.
        {"question": "is it ok to continue?"},
    )
    ...
```

Resuming the graph is done by passing a Command object to the graph with the resume key set to the value returned by the interrupt function.


## Breakpoints
Breakpoints pause graph execution at specific points and enable stepping through execution step by step. Breakpoints are powered by LangGraph's persistence layer, which saves the state after each graph step. Breakpoints can also be used to enable human-in-the-loop workflows, though we recommend using the interrupt function for this purpose.

Read more about breakpoints in the Breakpoints conceptual guide.

## Subgraphs
A subgraph is a graph that is used as a node in another graph. This is nothing more than the age-old concept of encapsulation, applied to LangGraph. Some reasons for using subgraphs are:

- building multi-agent systems
- when you want to reuse a set of nodes in multiple graphs, which maybe share some state, you can define them once in a subgraph and then use them in multiple parent graphs
- when you want different teams to work on different parts of the graph independently, you can define each part as a subgraph, and as long as the subgraph interface (the input and output schemas) is respected, the parent graph can be built without knowing any details of the subgraph

There are two ways to add subgraphs to a parent graph:

- add a node with the compiled subgraph: this is useful when the parent graph and the subgraph share state keys and you don't need to transform state on the way in or out

```
builder.add_node("subgraph", subgraph_builder.compile())
```

- add a node with a function that invokes the subgraph: this is useful when the parent graph and the subgraph have different state schemas and you need to transform state before or after calling the subgraph

```
subgraph = subgraph_builder.compile()

def call_subgraph(state: State):
    return subgraph.invoke({"subgraph_key": state["parent_key"]})

builder.add_node("subgraph", call_subgraph)
```

## As a compiled graph
The simplest way to create subgraph nodes is by using a compiled subgraph directly. When doing so, it is important that the parent graph and the subgraph state schemas share at least one key which they can use to communicate. If your graph and subgraph do not share any keys, you should write a function invoking the subgraph instead.


**Note:** If you pass extra keys to the subgraph node (i.e., in addition to the shared keys), they will be ignored by the subgraph node. Similarly, if you return extra keys from the subgraph, they will be ignored by the parent graph.

```
from langgraph.graph import StateGraph
from typing import TypedDict

class State(TypedDict):
    foo: str

class SubgraphState(TypedDict):
    foo: str  # note that this key is shared with the parent graph state
    bar: str

# Define subgraph
def subgraph_node(state: SubgraphState):
    # note that this subgraph node can communicate with the parent graph via the shared "foo" key
    return {"foo": state["foo"] + "bar"}

subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node(subgraph_node)
...
subgraph = subgraph_builder.compile()

# Define parent graph
builder = StateGraph(State)
builder.add_node("subgraph", subgraph)
...
graph = builder.compile()
```

## As a function
You might want to define a subgraph with a completely different schema. In this case, you can create a node function that invokes the subgraph. This function will need to transform the input (parent) state to the subgraph state before invoking the subgraph, and transform the results back to the parent state before returning the state update from the node.



```
class State(TypedDict):
    foo: str

class SubgraphState(TypedDict):
    # note that none of these keys are shared with the parent graph state
    bar: str
    baz: str

# Define subgraph
def subgraph_node(state: SubgraphState):
    return {"bar": state["bar"] + "baz"}

subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node(subgraph_node)
...
subgraph = subgraph_builder.compile()

# Define parent graph
def node(state: State):
    # transform the state to the subgraph state
    response = subgraph.invoke({"bar": state["foo"]})
    # transform response back to the parent state
    return {"foo": response["bar"]}

builder = StateGraph(State)
# note that we are using `node` function instead of a compiled subgraph
builder.add_node(node)
...
graph = builder.compile()
```

## Visualization
It's often nice to be able to visualize graphs, especially as they get more complex. LangGraph comes with several built-in ways to visualize graphs.
[visualize](https://langchain-ai.github.io/langgraph/how-tos/visualization/)

## Streaming
LangGraph is built with first class support for streaming, including streaming updates from graph nodes during the execution, streaming tokens from LLM calls and more. See this conceptual guide for more information.
[streaming](https://langchain-ai.github.io/langgraph/concepts/streaming/)


## How to create branches for parallel node execution¶
Parallel execution of nodes is essential to speed up overall graph operation. LangGraph offers native support for parallel execution of nodes, which can significantly enhance the performance of graph-based workflows. This parallelization is achieved through fan-out and fan-in mechanisms, utilizing both standard edges and conditional_edges. Below are some examples showing how to add create branching dataflows that work for you.

## How to run graph nodes in parallel
In this example, we fan out from Node A to B and C and then fan in to D. With our state, we specify the reducer add operation. This will combine or accumulate values for the specific key in the State, rather than simply overwriting the existing value. For lists, this means concatenating the new list with the existing list. See this guide for more detail on updating state with reducers.

```
import operator
from typing import Annotated, Any

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    # The operator.add reducer fn makes this append-only
    aggregate: Annotated[list, operator.add]


def a(state: State):
    print(f'Adding "A" to {state["aggregate"]}')
    return {"aggregate": ["A"]}


def b(state: State):
    print(f'Adding "B" to {state["aggregate"]}')
    return {"aggregate": ["B"]}


def c(state: State):
    print(f'Adding "C" to {state["aggregate"]}')
    return {"aggregate": ["C"]}


def d(state: State):
    print(f'Adding "D" to {state["aggregate"]}')
    return {"aggregate": ["D"]}


builder = StateGraph(State)
builder.add_node(a)
builder.add_node(b)
builder.add_node(c)
builder.add_node(d)
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("a", "c")
builder.add_edge("b", "d")
builder.add_edge("c", "d")
builder.add_edge("d", END)
graph = builder.compile()
```

```
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))
```

With the reducer, you can see that the values added in each node are accumulated.

```
graph.invoke({"aggregate": []}, {"configurable": {"thread_id": "foo"}})
```

```
Adding "A" to []
Adding "B" to ['A']
Adding "C" to ['A']
Adding "D" to ['A', 'B', 'C']
```

**Note:** In the above example, nodes "b" and "c" are executed concurrently in the same superstep. Because they are in the same step, node "d" executes after both "b" and "c" are finished.

Importantly, updates from a parallel superstep may not be ordered consistently. If you need a consistent, predetermined ordering of updates from a parallel superstep, you should write the outputs to a separate field in the state together with a value with which to order them.

## Parallel node fan-out and fan-in with extra steps
The above example showed how to fan-out and fan-in when each path was only one step. But what if one path had more than one step? Let's add a node b_2 in the "b" branch:

```
def b_2(state: State):
    print(f'Adding "B_2" to {state["aggregate"]}')
    return {"aggregate": ["B_2"]}


builder = StateGraph(State)
builder.add_node(a)
builder.add_node(b)
builder.add_node(b_2)
builder.add_node(c)
builder.add_node(d)
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("a", "c")
builder.add_edge("b", "b_2")
builder.add_edge(["b_2", "c"], "d")
builder.add_edge("d", END)
graph = builder.compile()
```

```
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))
```

```
graph.invoke({"aggregate": []})
```

```
Adding "A" to []
Adding "B" to ['A']
Adding "C" to ['A']
Adding "B_2" to ['A', 'B', 'C']
Adding "D" to ['A', 'B', 'C', 'B_2']
```

```
{'aggregate': ['A', 'B', 'C', 'B_2', 'D']}
```

**Note:** In the above example, nodes "b" and "c" are executed concurrently in the same superstep. What happens in the next step?

We use add_edge(["b_2", "c"], "d") here to force node "d" to only run when both nodes "b_2" and "c" have finished execution. If we added two separate edges, node "d" would run twice: after node b2 finishes and once again after node c (in whichever order those nodes finish).

## Conditional Branching
If your fan-out is not deterministic, you can use add_conditional_edges directly.

```
import operator
from typing import Annotated, Sequence

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    aggregate: Annotated[list, operator.add]
    # Add a key to the state. We will set this key to determine
    # how we branch.
    which: str


def a(state: State):
    print(f'Adding "A" to {state["aggregate"]}')
    return {"aggregate": ["A"]}


def b(state: State):
    print(f'Adding "B" to {state["aggregate"]}')
    return {"aggregate": ["B"]}


def c(state: State):
    print(f'Adding "C" to {state["aggregate"]}')
    return {"aggregate": ["C"]}


def d(state: State):
    print(f'Adding "D" to {state["aggregate"]}')
    return {"aggregate": ["D"]}


def e(state: State):
    print(f'Adding "E" to {state["aggregate"]}')
    return {"aggregate": ["E"]}


builder = StateGraph(State)
builder.add_node(a)
builder.add_node(b)
builder.add_node(c)
builder.add_node(d)
builder.add_node(e)
builder.add_edge(START, "a")


def route_bc_or_cd(state: State) -> Sequence[str]:
    if state["which"] == "cd":
        return ["c", "d"]
    return ["b", "c"]


intermediates = ["b", "c", "d"]
builder.add_conditional_edges(
    "a",
    route_bc_or_cd,
    intermediates,
)
for node in intermediates:
    builder.add_edge(node, "e")

builder.add_edge("e", END)
graph = builder.compile()
```

```
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))
```

```
graph.invoke({"aggregate": [], "which": "bc"})
```

```
graph.invoke({"aggregate": [], "which": "cd"})
```

```
Adding "A" to []
Adding "C" to ['A']
Adding "D" to ['A']
Adding "E" to ['A', 'C', 'D']
```

```
{'aggregate': ['A', 'C', 'D', 'E'], 'which': 'cd'}
```

## How to create map-reduce branches for parallel execution
Map-reduce operations are essential for efficient task decomposition and parallel processing. This approach involves breaking a task into smaller sub-tasks, processing each sub-task in parallel, and aggregating the results across all of the completed sub-tasks.

Consider this example: given a general topic from the user, generate a list of related subjects, generate a joke for each subject, and select the best joke from the resulting list. In this design pattern, a first node may generate a list of objects (e.g., related subjects) and we want to apply some other node (e.g., generate a joke) to all those objects (e.g., subjects). However, two main challenges arise.

(1) the number of objects (e.g., subjects) may be unknown ahead of time (meaning the number of edges may not be known) when we lay out the graph and (2) the input State to the downstream Node should be different (one for each generated object).

LangGraph addresses these challenges through its Send API. By utilizing conditional edges, Send can distribute different states (e.g., subjects) to multiple instances of a node (e.g., joke generation). Importantly, the sent state can differ from the core graph's state, allowing for flexible and dynamic workflow management.

![map-reduce](image/map-reduce.png)


## Setup
First, let's install the required packages and set our API keys

```
%%capture --no-stderr
%pip install -U langchain-anthropic langgraph
```

```
import os
import getpass


def _set_env(name: str):
    if not os.getenv(name):
        os.environ[name] = getpass.getpass(f"{name}: ")


_set_env("ANTHROPIC_API_KEY")
```

## Define the graph

```
import operator
from typing import Annotated
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic

from langgraph.types import Send
from langgraph.graph import END, StateGraph, START

from pydantic import BaseModel, Field

# Model and prompts
# Define model and prompts we will use
subjects_prompt = """Generate a comma separated list of between 2 and 5 examples related to: {topic}."""
joke_prompt = """Generate a joke about {subject}"""
best_joke_prompt = """Below are a bunch of jokes about {topic}. Select the best one! Return the ID of the best one.

{jokes}"""


class Subjects(BaseModel):
    subjects: list[str]


class Joke(BaseModel):
    joke: str


class BestJoke(BaseModel):
    id: int = Field(description="Index of the best joke, starting with 0", ge=0)


model = ChatAnthropic(model="claude-3-5-sonnet-20240620")

# Graph components: define the components that will make up the graph


# This will be the overall state of the main graph.
# It will contain a topic (which we expect the user to provide)
# and then will generate a list of subjects, and then a joke for
# each subject
class OverallState(TypedDict):
    topic: str
    subjects: list
    # Notice here we use the operator.add
    # This is because we want combine all the jokes we generate
    # from individual nodes back into one list - this is essentially
    # the "reduce" part
    jokes: Annotated[list, operator.add]
    best_selected_joke: str


# This will be the state of the node that we will "map" all
# subjects to in order to generate a joke
class JokeState(TypedDict):
    subject: str


# This is the function we will use to generate the subjects of the jokes
def generate_topics(state: OverallState):
    prompt = subjects_prompt.format(topic=state["topic"])
    response = model.with_structured_output(Subjects).invoke(prompt)
    return {"subjects": response.subjects}


# Here we generate a joke, given a subject
def generate_joke(state: JokeState):
    prompt = joke_prompt.format(subject=state["subject"])
    response = model.with_structured_output(Joke).invoke(prompt)
    return {"jokes": [response.joke]}


# Here we define the logic to map out over the generated subjects
# We will use this as an edge in the graph
def continue_to_jokes(state: OverallState):
    # We will return a list of `Send` objects
    # Each `Send` object consists of the name of a node in the graph
    # as well as the state to send to that node
    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]


# Here we will judge the best joke
def best_joke(state: OverallState):
    jokes = "\n\n".join(state["jokes"])
    prompt = best_joke_prompt.format(topic=state["topic"], jokes=jokes)
    response = model.with_structured_output(BestJoke).invoke(prompt)
    return {"best_selected_joke": state["jokes"][response.id]}


# Construct the graph: here we put everything together to construct our graph
graph = StateGraph(OverallState)
graph.add_node("generate_topics", generate_topics)
graph.add_node("generate_joke", generate_joke)
graph.add_node("best_joke", best_joke)
graph.add_edge(START, "generate_topics")
graph.add_conditional_edges("generate_topics", continue_to_jokes, ["generate_joke"])
graph.add_edge("generate_joke", "best_joke")
graph.add_edge("best_joke", END)
app = graph.compile()
```

```
from IPython.display import Image

Image(app.get_graph().draw_mermaid_png())
```

## Use the graph

```
# Call the graph: here we call it to generate a list of jokes
for s in app.stream({"topic": "animals"}):
    print(s)
```


```
{'generate_topics': {'subjects': ['Lions', 'Elephants', 'Penguins', 'Dolphins']}}
{'generate_joke': {'jokes': ["Why don't elephants use computers? They're afraid of the mouse!"]}}
{'generate_joke': {'jokes': ["Why don't dolphins use smartphones? Because they're afraid of phishing!"]}}
{'generate_joke': {'jokes': ["Why don't you see penguins in Britain? Because they're afraid of Wales!"]}}
{'generate_joke': {'jokes': ["Why don't lions like fast food? Because they can't catch it!"]}}
{'best_joke': {'best_selected_joke': "Why don't dolphins use smartphones? Because they're afraid of phishing!"}}
```

## How to create and control loops
When creating a graph with a loop, we require a mechanism for terminating execution. This is most commonly done by adding a conditional edge that routes to the END node once we reach some termination condition.

You can also set the graph recursion limit when invoking or streaming the graph. The recursion limit sets the number of supersteps that the graph is allowed to execute before it raises an error. Read more about the concept of recursion limits here.

Let's consider a simple graph with a loop to better understand how these mechanisms work.

## Summary
When creating a loop, you can include a conditional edge that specifies a termination condition:

```
builder = StateGraph(State)
builder.add_node(a)
builder.add_node(b)

def route(state: State) -> Literal["b", END]:
    if termination_condition(state):
        return END
    else:
        return "a"

builder.add_edge(START, "a")
builder.add_conditional_edges("a", route)
builder.add_edge("b", "a")
graph = builder.compile()
```

To control the recursion limit, specify "recursion_limit" in the config. This will raise a GraphRecursionError, which you can catch and handle:

```
from langgraph.errors import GraphRecursionError

try:
    graph.invoke(inputs, {"recursion_limit": 3})
except GraphRecursionError:
    print("Recursion Error")
```

## Define the graph
Let's define a graph with a simple loop. Note that we use a conditional edge to implement a termination condition.

```
import operator
from typing import Annotated, Literal

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    # The operator.add reducer fn makes this append-only
    aggregate: Annotated[list, operator.add]


def a(state: State):
    print(f'Node A sees {state["aggregate"]}')
    return {"aggregate": ["A"]}


def b(state: State):
    print(f'Node B sees {state["aggregate"]}')
    return {"aggregate": ["B"]}


# Define nodes
builder = StateGraph(State)
builder.add_node(a)
builder.add_node(b)


# Define edges
def route(state: State) -> Literal["b", END]:
    if len(state["aggregate"]) < 7:
        return "b"
    else:
        return END


builder.add_edge(START, "a")
builder.add_conditional_edges("a", route)
builder.add_edge("b", "a")
graph = builder.compile()
```

```
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))
```

This architecture is similar to a ReAct agent in which node "a" is a tool-calling model, and node "b" represents the tools.

In our route conditional edge, we specify that we should end after the "aggregate" list in the state passes a threshold length.

Invoking the graph, we see that we alternate between nodes "a" and "b" before terminating once we reach the termination condition.


```
graph.invoke({"aggregate": []})
```

```
Node A sees []
Node B sees ['A']
Node A sees ['A', 'B']
Node B sees ['A', 'B', 'A']
Node A sees ['A', 'B', 'A', 'B']
Node B sees ['A', 'B', 'A', 'B', 'A']
Node A sees ['A', 'B', 'A', 'B', 'A', 'B']
```

```
{'aggregate': ['A', 'B', 'A', 'B', 'A', 'B', 'A']}
```


## Impose a recursion limit

In some applications, we may not have a guarantee that we will reach a given termination condition. In these cases, we can set the graph's recursion limit. This will raise a GraphRecursionError after a given number of supersteps. We can then catch and handle this exception:

```
from langgraph.errors import GraphRecursionError

try:
    graph.invoke({"aggregate": []}, {"recursion_limit": 4})
except GraphRecursionError:
    print("Recursion Error")
```

```
Node A sees []
Node B sees ['A']
Node A sees ['A', 'B']
Node B sees ['A', 'B', 'A']
Recursion Error
```

## Loops with branches
To better understand how the recursion limit works, let's consider a more complex example. Below we implement a loop, but one step fans out into two nodes:

```
import operator
from typing import Annotated, Literal

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    aggregate: Annotated[list, operator.add]


def a(state: State):
    print(f'Node A sees {state["aggregate"]}')
    return {"aggregate": ["A"]}


def b(state: State):
    print(f'Node B sees {state["aggregate"]}')
    return {"aggregate": ["B"]}


def c(state: State):
    print(f'Node C sees {state["aggregate"]}')
    return {"aggregate": ["C"]}


def d(state: State):
    print(f'Node D sees {state["aggregate"]}')
    return {"aggregate": ["D"]}


# Define nodes
builder = StateGraph(State)
builder.add_node(a)
builder.add_node(b)
builder.add_node(c)
builder.add_node(d)


# Define edges
def route(state: State) -> Literal["b", END]:
    if len(state["aggregate"]) < 7:
        return "b"
    else:
        return END


builder.add_edge(START, "a")
builder.add_conditional_edges("a", route)
builder.add_edge("b", "c")
builder.add_edge("b", "d")
builder.add_edge(["c", "d"], "a")
graph = builder.compile()
```

```
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))
```

This graph looks complex, but can be conceptualized as loop of supersteps:

1. Node A
2. Node B
3. Nodes C and D
4. Node A
5. ...
We have a loop of four supersteps, where nodes C and D are executed concurrently.

Invoking the graph as before, we see that we complete two full "laps" before hitting the termination condition:

```
result = graph.invoke({"aggregate": []})
```

```
Node A sees []
Node B sees ['A']
Node D sees ['A', 'B']
Node C sees ['A', 'B']
Node A sees ['A', 'B', 'C', 'D']
Node B sees ['A', 'B', 'C', 'D', 'A']
Node D sees ['A', 'B', 'C', 'D', 'A', 'B']
Node C sees ['A', 'B', 'C', 'D', 'A', 'B']
Node A sees ['A', 'B', 'C', 'D', 'A', 'B', 'C', 'D']
```

However, if we set the recursion limit to four, we only complete one lap because each lap is four supersteps:

```
from langgraph.errors import GraphRecursionError

try:
    result = graph.invoke({"aggregate": []}, {"recursion_limit": 4})
except GraphRecursionError:
    print("Recursion Error")
```

```
Node A sees []
Node B sees ['A']
Node C sees ['A', 'B']
Node D sees ['A', 'B']
Node A sees ['A', 'B', 'C', 'D']
Recursion Error
```

## How to visualize your graph

## Set up Graph
You can visualize any arbitrary Graph, including StateGraph. Let's have some fun by drawing fractals :).

```
import random
from typing import Annotated, Literal

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]


class MyNode:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, state: State):
        return {"messages": [("assistant", f"Called node {self.name}")]}


def route(state) -> Literal["entry_node", "__end__"]:
    if len(state["messages"]) > 10:
        return "__end__"
    return "entry_node"


def add_fractal_nodes(builder, current_node, level, max_level):
    if level > max_level:
        return

    # Number of nodes to create at this level
    num_nodes = random.randint(1, 3)  # Adjust randomness as needed
    for i in range(num_nodes):
        nm = ["A", "B", "C"][i]
        node_name = f"node_{current_node}_{nm}"
        builder.add_node(node_name, MyNode(node_name))
        builder.add_edge(current_node, node_name)

        # Recursively add more nodes
        r = random.random()
        if r > 0.2 and level + 1 < max_level:
            add_fractal_nodes(builder, node_name, level + 1, max_level)
        elif r > 0.05:
            builder.add_conditional_edges(node_name, route, node_name)
        else:
            # End
            builder.add_edge(node_name, "__end__")


def build_fractal_graph(max_level: int):
    builder = StateGraph(State)
    entry_point = "entry_node"
    builder.add_node(entry_point, MyNode(entry_point))
    builder.add_edge(START, entry_point)

    add_fractal_nodes(builder, entry_point, 1, max_level)

    # Optional: set a finish point if required
    builder.add_edge(entry_point, END)  # or any specific node

    return builder.compile()


app = build_fractal_graph(3)
```

## Mermaid
We can also convert a graph class into Mermaid syntax.

```
print(app.get_graph().draw_mermaid())
```

```
%%{init: {'flowchart': {'curve': 'linear'}}}%%
graph TD;
    __start__([<p>__start__</p>]):::first
    entry_node(entry_node)
    node_entry_node_A(node_entry_node_A)
    node_entry_node_B(node_entry_node_B)
    node_node_entry_node_B_A(node_node_entry_node_B_A)
    node_node_entry_node_B_B(node_node_entry_node_B_B)
    node_node_entry_node_B_C(node_node_entry_node_B_C)
    __end__([<p>__end__</p>]):::last
    __start__ --> entry_node;
    entry_node --> __end__;
    entry_node --> node_entry_node_A;
    entry_node --> node_entry_node_B;
    node_entry_node_B --> node_node_entry_node_B_A;
    node_entry_node_B --> node_node_entry_node_B_B;
    node_entry_node_B --> node_node_entry_node_B_C;
    node_entry_node_A -.-> entry_node;
    node_entry_node_A -.-> __end__;
    node_node_entry_node_B_A -.-> entry_node;
    node_node_entry_node_B_A -.-> __end__;
    node_node_entry_node_B_B -.-> entry_node;
    node_node_entry_node_B_B -.-> __end__;
    node_node_entry_node_B_C -.-> entry_node;
    node_node_entry_node_B_C -.-> __end__;
    classDef default fill:#f2f0ff,line-height:1.2
    classDef first fill-opacity:0
    classDef last fill:#bfb6fc
```

## PNG
If preferred, we could render the Graph into a .png. Here we could use three options:

- Using Mermaid.ink API (does not require additional packages)
- Using Mermaid + Pyppeteer (requires pip install pyppeteer)
- Using graphviz (which requires pip install graphviz)

## Using Mermaid.Ink
By default, draw_mermaid_png() uses Mermaid.Ink's API to generate the diagram.

```
from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

display(
    Image(
        app.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
        )
    )
)
```

![Mermaid](image/Mermaid.png)

## Using Mermaid + Pyppeteer

```
%%capture --no-stderr
%pip install --quiet pyppeteer
%pip install --quiet nest_asyncio
```

```
import nest_asyncio

nest_asyncio.apply()  # Required for Jupyter Notebook to run async functions

display(
    Image(
        app.get_graph().draw_mermaid_png(
            curve_style=CurveStyle.LINEAR,
            node_colors=NodeStyles(first="#ffdfba", last="#baffc9", default="#fad7de"),
            wrap_label_n_words=9,
            output_file_path=None,
            draw_method=MermaidDrawMethod.PYPPETEER,
            background_color="white",
            padding=10,
        )
    )
)
```

![Pyppeteer](image/Pyppeteer.png)

## Using Graphviz

```
%%capture --no-stderr
%pip install pygraphviz
```

```
try:
    display(Image(app.get_graph().draw_png()))
except ImportError:
    print(
        "You likely need to install dependencies for pygraphviz, see more here https://github.com/pygraphviz/pygraphviz/blob/main/INSTALL.txt"
    )
```

![Graphviz](image/Graphviz.png)

# Fine-grained Control

## How to combine control flow and state updates with Command

It can be useful to combine control flow (edges) and state updates (nodes). For example, you might want to BOTH perform state updates AND decide which node to go to next in the SAME node. LangGraph provides a way to do so by returning a Command object from node functions:

```
def my_node(state: State) -> Command[Literal["my_other_node"]]:
    return Command(
        # state update
        update={"foo": "bar"},
        # control flow
        goto="my_other_node"
    )
```

If you are using subgraphs, you might want to navigate from a node within a subgraph to a different subgraph (i.e. a different node in the parent graph). To do so, you can specify graph=Command.PARENT in Command:

```
def my_node(state: State) -> Command[Literal["my_other_node"]]:
    return Command(
        update={"foo": "bar"},
        goto="other_subgraph",  # where `other_subgraph` is a node in the parent graph
        graph=Command.PARENT
    )
```

## How to add node retry policies
There are many use cases where you may wish for your node to have a custom retry policy, for example if you are calling an API, querying a database, or calling an LLM, etc.

```
%%capture --no-stderr
%pip install -U langgraph langchain_anthropic langchain_community
```

```
import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("ANTHROPIC_API_KEY")
```

In order to configure the retry policy, you have to pass the retry parameter to the add_node. The retry parameter takes in a RetryPolicy named tuple object. Below we instantiate a RetryPolicy object with the default parameters:

```
from langgraph.pregel import RetryPolicy

RetryPolicy()
```

```
RetryPolicy(initial_interval=0.5, backoff_factor=2.0, max_interval=128.0, max_attempts=3, jitter=True, retry_on=<function default_retry_on at 0x78b964b89940>)
```

By default, the retry_on parameter uses the default_retry_on function, which retries on any exception except for the following:

- ValueError
- TypeError
- ArithmeticError
- ImportError
- LookupError
- NameError
- SyntaxError
- RuntimeError
- ReferenceError
- StopIteration
- StopAsyncIteration
- OSError


In addition, for exceptions from popular http request libraries such as requests and httpx it only retries on 5xx status codes.

## Passing a retry policy to a node
Lastly, we can pass RetryPolicy objects when we call the add_node function. In the example below we pass two different retry policies to each of our nodes:

```
import operator
import sqlite3
from typing import Annotated, Sequence
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage

from langgraph.graph import END, StateGraph, START
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage

db = SQLDatabase.from_uri("sqlite:///:memory:")

model = ChatAnthropic(model_name="claude-2.1")


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


def query_database(state):
    query_result = db.run("SELECT * FROM Artist LIMIT 10;")
    return {"messages": [AIMessage(content=query_result)]}


def call_model(state):
    response = model.invoke(state["messages"])
    return {"messages": [response]}


# Define a new graph
builder = StateGraph(AgentState)
builder.add_node(
    "query_database",
    query_database,
    retry=RetryPolicy(retry_on=sqlite3.OperationalError),
)
builder.add_node("model", call_model, retry=RetryPolicy(max_attempts=5))
builder.add_edge(START, "model")
builder.add_edge("model", "query_database")
builder.add_edge("query_database", END)

graph = builder.compile()
```

## How to return state before hitting recursion limit
Setting the graph recursion limit can help you control how long your graph will stay running, but if the recursion limit is hit your graph returns an error - which may not be ideal for all use cases. Instead you may wish to return the value of the state just before the recursion limit is hit. This how-to will show you how to do this.

## Without returning state
We are going to define a dummy graph in this example that will always hit the recursion limit. First, we will implement it without returning the state and show that it hits the recursion limit. This graph is based on the ReAct architecture, but instead of actually making decisions and taking actions it just loops forever.

```
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph import START, END


class State(TypedDict):
    value: str
    action_result: str


def router(state: State):
    if state["value"] == "end":
        return END
    else:
        return "action"


def decision_node(state):
    return {"value": "keep going!"}


def action_node(state: State):
    # Do your action here ...
    return {"action_result": "what a great result!"}


workflow = StateGraph(State)
workflow.add_node("decision", decision_node)
workflow.add_node("action", action_node)
workflow.add_edge(START, "decision")
workflow.add_conditional_edges("decision", router, ["action", END])
workflow.add_edge("action", "decision")
app = workflow.compile()
```

```
from IPython.display import Image, display

display(Image(app.get_graph().draw_mermaid_png()))
```

Let's verify that our graph will always hit the recursion limit:

```
from langgraph.errors import GraphRecursionError

try:
    app.invoke({"value": "hi!"})
except GraphRecursionError:
    print("Recursion Error")
```

```
Recursion Error
```

## With returning state
To avoid hitting the recursion limit, we can introduce a new key to our state called remaining_steps. It will keep track of number of steps until reaching the recursion limit. We can then check the value of remaining_steps to determine whether we should terminate the graph execution and return the state to the user without causing the RecursionError.

To do so, we will use a special RemainingSteps annotation. Under the hood, it creates a special ManagedValue channel -- a state channel that will exist for the duration of our graph run and no longer.

Since our action node is going to always induce at least 2 extra steps to our graph (since the action node ALWAYS calls the decision node afterwards), we will use this channel to check if we are within 2 steps of the limit.

Now, when we run our graph we should receive no errors and instead get the last value of the state before the recursion limit was hit.

```
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from typing import Annotated

from langgraph.managed.is_last_step import RemainingSteps


class State(TypedDict):
    value: str
    action_result: str
    remaining_steps: RemainingSteps


def router(state: State):
    # Force the agent to end
    if state["remaining_steps"] <= 2:
        return END
    if state["value"] == "end":
        return END
    else:
        return "action"


def decision_node(state):
    return {"value": "keep going!"}


def action_node(state: State):
    # Do your action here ...
    return {"action_result": "what a great result!"}


workflow = StateGraph(State)
workflow.add_node("decision", decision_node)
workflow.add_node("action", action_node)
workflow.add_edge(START, "decision")
workflow.add_conditional_edges("decision", router, ["action", END])
workflow.add_edge("action", "decision")
app = workflow.compile()
```

```
app.invoke({"value": "hi!"})
```

```
{'value': 'keep going!', 'action_result': 'what a great result!'}
```

## Persistence
LangGraph Persistence makes it easy to persist state across graph runs (per-thread persistence) and across threads (cross-thread persistence). These how-to guides show how to add persistence to your graph.

## How to add thread-level persistence to your graph
Many AI applications need memory to share context across multiple interactions. In LangGraph, this kind of memory can be added to any StateGraph using thread-level persistence .

When creating any LangGraph graph, you can set it up to persist its state by adding a checkpointer when compiling the graph:

```
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph.compile(checkpointer=checkpointer)
```

**Note:** If you need memory that is shared across multiple conversations or users (cross-thread persistence), check out this how-to guide.

```
%%capture --no-stderr
%pip install --quiet -U langgraph langchain_anthropic
```

Next, we need to set API key for Anthropic (the LLM we will use).

```
import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("ANTHROPIC_API_KEY")
```

## Define graph
We will be using a single-node graph that calls a chat model.

Let's first define the model we'll be using:

```
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(model="claude-3-5-sonnet-20240620")
```

Now we can define our StateGraph and add our model-calling node:


```
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, MessagesState, START


def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}


builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")
graph = builder.compile()
```

If we try to use this graph, the context of the conversation will not be persisted across interactions:

```
input_message = {"role": "user", "content": "hi! I'm bob"}
for chunk in graph.stream({"messages": [input_message]}, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

input_message = {"role": "user", "content": "what's my name?"}
for chunk in graph.stream({"messages": [input_message]}, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
```

```
================================[1m Human Message [0m=================================

hi! I'm bob
==================================[1m Ai Message [0m==================================

Hello Bob! It's nice to meet you. How are you doing today? Is there anything I can help you with or would you like to chat about something in particular?
================================[1m Human Message [0m=================================

what's my name?
==================================[1m Ai Message [0m==================================

I apologize, but I don't have access to your personal information, including your name. I'm an AI language model designed to provide general information and answer questions to the best of my ability based on my training data. I don't have any information about individual users or their personal details. If you'd like to share your name, you're welcome to do so, but I won't be able to recall it in future conversations.
```

## Add persistence
To add in persistence, we need to pass in a Checkpointer when compiling the graph.

```
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
# If you're using LangGraph Cloud or LangGraph Studio, you don't need to pass the checkpointer when compiling the graph, since it's done automatically.
```

We can now interact with the agent and see that it remembers previous messages!

```
config = {"configurable": {"thread_id": "1"}}
input_message = {"role": "user", "content": "hi! I'm bob"}
for chunk in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
```

You can always resume previous threads:

```
input_message = {"role": "user", "content": "what's my name?"}
for chunk in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
```

```
================================[1m Human Message [0m=================================

what's my name?
==================================[1m Ai Message [0m==================================

Your name is Bob, as you introduced yourself at the beginning of our conversation.
```

If we want to start a new conversation, we can pass in a different thread_id. Poof! All the memories are gone!

```
input_message = {"role": "user", "content": "what's my name?"}
for chunk in graph.stream(
    {"messages": [input_message]},
    {"configurable": {"thread_id": "2"}},
    stream_mode="values",
):
    chunk["messages"][-1].pretty_print()
```

```
================================[1m Human Message [0m=================================

what's is my name?
==================================[1m Ai Message [0m==================================

I apologize, but I don't have access to your personal information, including your name. As an AI language model, I don't have any information about individual users unless it's provided within the conversation. If you'd like to share your name, you're welcome to do so, but otherwise, I won't be able to know or guess it.
```


## How to add thread-level persistence to a subgraph

```
%%capture --no-stderr
%pip install -U langgraph
```

## Define the graph with persistence
To add persistence to a graph with subgraphs, all you need to do is pass a checkpointer when compiling the parent graph. LangGraph will automatically propagate the checkpointer to the child subgraphs.

**Note:** You shouldn't provide a checkpointer when compiling a subgraph. Instead, you must define a single checkpointer that you pass to parent_graph.compile(), and LangGraph will automatically propagate the checkpointer to the child subgraphs. If you pass the checkpointer to the subgraph.compile(), it will simply be ignored. This also applies when you add a node function that invokes the subgraph.

Let's define a simple graph with a single subgraph node to show how to do this.

```
from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict


# subgraph


class SubgraphState(TypedDict):
    foo: str  # note that this key is shared with the parent graph state
    bar: str


def subgraph_node_1(state: SubgraphState):
    return {"bar": "bar"}


def subgraph_node_2(state: SubgraphState):
    # note that this node is using a state key ('bar') that is only available in the subgraph
    # and is sending update on the shared state key ('foo')
    return {"foo": state["foo"] + state["bar"]}


subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_node(subgraph_node_2)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
subgraph = subgraph_builder.compile()


# parent graph


class State(TypedDict):
    foo: str


def node_1(state: State):
    return {"foo": "hi! " + state["foo"]}


builder = StateGraph(State)
builder.add_node("node_1", node_1)
# note that we're adding the compiled subgraph as a node to the parent graph
builder.add_node("node_2", subgraph)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
```


We can now compile the graph with an in-memory checkpointer (MemorySaver).

```
checkpointer = MemorySaver()
# You must only pass checkpointer when compiling the parent graph.
# LangGraph will automatically propagate the checkpointer to the child subgraphs.
graph = builder.compile(checkpointer=checkpointer)
```

## Verify persistence works
Let's now run the graph and inspect the persisted state for both the parent graph and the subgraph to verify that persistence works. We should expect to see the final execution results for both the parent and subgraph in state.values.

```
config = {"configurable": {"thread_id": "1"}}
```

```
for _, chunk in graph.stream({"foo": "foo"}, config, subgraphs=True):
    print(chunk)
```

```
{'node_1': {'foo': 'hi! foo'}}
{'subgraph_node_1': {'bar': 'bar'}}
{'subgraph_node_2': {'foo': 'hi! foobar'}}
{'node_2': {'foo': 'hi! foobar'}}
```

We can now view the parent graph state by calling graph.get_state() with the same config that we used to invoke the graph.

```
graph.get_state(config).values
```

```
{'foo': 'hi! foobar'}
```

To view the subgraph state, we need to do two things:

1. Find the most recent config value for the subgraph
2. Use graph.get_state() to retrieve that value for the most recent subgraph config.

To find the correct config, we can examine the state history from the parent graph and find the state snapshot before we return results from node_2 (the node with subgraph):

```
state_with_subgraph = [
    s for s in graph.get_state_history(config) if s.next == ("node_2",)
][0]
```

The state snapshot will include the list of tasks to be executed next. When using subgraphs, the tasks will contain the config that we can use to retrieve the subgraph state:

```
subgraph_config = state_with_subgraph.tasks[0].state
subgraph_config
```

```
{'configurable': {'thread_id': '1',
  'checkpoint_ns': 'node_2:6ef111a6-f290-7376-0dfc-a4152307bc5b'}}
```

```
graph.get_state(subgraph_config).values
```

```
{'foo': 'hi! foobar', 'bar': 'bar'}
```

## How to add cross-thread persistence to your graph
In the previous guide you learned how to persist graph state across multiple interactions on a single thread. LangGraph also allows you to persist data across multiple threads. For instance, you can store information about users (their names or preferences) in a shared memory and reuse them in the new conversational threads.

In this guide, we will show how to construct and use a graph that has a shared memory implemented using the Store interface.

```
%%capture --no-stderr
%pip install -U langchain_openai langgraph
```

```
import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("ANTHROPIC_API_KEY")
_set_env("OPENAI_API_KEY")
```

## Define store
In this example we will create a graph that will be able to retrieve information about a user's preferences. We will do so by defining an InMemoryStore - an object that can store data in memory and query that data. We will then pass the store object when compiling the graph. This allows each node in the graph to access the store: when you define node functions, you can define store keyword argument, and LangGraph will automatically pass the store object you compiled the graph with.

When storing objects using the Store interface you define two things:

- the namespace for the object, a tuple (similar to directories)
- the object key (similar to filenames)

In our example, we'll be using ("memories", <user_id>) as namespace and random UUID as key for each new memory.

Importantly, to determine the user, we will be passing user_id via the config keyword argument of the node function.

Let's first define an InMemoryStore already populated with some memories about the users.


```
from langgraph.store.memory import InMemoryStore
from langchain_openai import OpenAIEmbeddings

in_memory_store = InMemoryStore(
    index={
        "embed": OpenAIEmbeddings(model="text-embedding-3-small"),
        "dims": 1536,
    }
)
```

## Create graph

```
import uuid
from typing import Annotated
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore


model = ChatAnthropic(model="claude-3-5-sonnet-20240620")


# NOTE: we're passing the Store param to the node --
# this is the Store we compile the graph with
def call_model(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    namespace = ("memories", user_id)
    memories = store.search(namespace, query=str(state["messages"][-1].content))
    info = "\n".join([d.value["data"] for d in memories])
    system_msg = f"You are a helpful assistant talking to the user. User info: {info}"

    # Store new memories if the user asks the model to remember
    last_message = state["messages"][-1]
    if "remember" in last_message.content.lower():
        memory = "User name is Bob"
        store.put(namespace, str(uuid.uuid4()), {"data": memory})

    response = model.invoke(
        [{"role": "system", "content": system_msg}] + state["messages"]
    )
    return {"messages": response}


builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")

# NOTE: we're passing the store object here when compiling the graph
graph = builder.compile(checkpointer=MemorySaver(), store=in_memory_store)
# If you're using LangGraph Cloud or LangGraph Studio, you don't need to pass the store or checkpointer when compiling the graph, since it's done automatically.
```


## Run the graph!
Now let's specify a user ID in the config and tell the model our name:

```
config = {"configurable": {"thread_id": "1", "user_id": "1"}}
input_message = {"role": "user", "content": "Hi! Remember: my name is Bob"}
for chunk in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
```

```
================================[1m Human Message [0m=================================

Hi! Remember: my name is Bob
==================================[1m Ai Message [0m==================================

Hello Bob! It's nice to meet you. I'll remember that your name is Bob. How can I assist you today?
```

```
config = {"configurable": {"thread_id": "2", "user_id": "1"}}
input_message = {"role": "user", "content": "what is my name?"}
for chunk in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
```

```
================================[1m Human Message [0m=================================

what is my name?
==================================[1m Ai Message [0m==================================

Your name is Bob.
```

We can now inspect our in-memory store and verify that we have in fact saved the memories for the user:

```
for memory in in_memory_store.search(("memories", "1")):
    print(memory.value)
```


```
{'data': 'User name is Bob'}
```

Let's now run the graph for another user to verify that the memories about the first user are self contained:



```
config = {"configurable": {"thread_id": "3", "user_id": "2"}}
input_message = {"role": "user", "content": "what is my name?"}
for chunk in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()
```

```
================================[1m Human Message [0m=================================

what is my name?
==================================[1m Ai Message [0m==================================

I apologize, but I don't have any information about your name. As an AI assistant, I don't have access to personal information about users unless it has been specifically shared in our conversation. If you'd like, you can tell me your name and I'll be happy to use it in our discussion.
```

# How to use Postgres checkpointer for persistence

When creating LangGraph agents, you can also set them up so that they persist their state. This allows you to do things like interact with an agent multiple times and have it remember previous interactions.


This how-to guide shows how to use Postgres as the backend for persisting checkpoint state using the **langgraph-checkpoint-postgres** library.

For demonstration purposes we add persistence to the pre-built create react agent.

In general, you can add a checkpointer to any custom graph that you build like this:

```
from langgraph.graph import StateGraph

builder = StateGraph(....)
# ... define the graph
checkpointer = # postgres checkpointer (see examples below)
graph = builder.compile(checkpointer=checkpointer)
...
```
## Setup
You will need access to a postgres instance. 
Next, let's install the required packages and set our API keys

```
%%capture --no-stderr
%pip install -U psycopg psycopg-pool langgraph langgraph-checkpoint-postgres
```

```
import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("OPENAI_API_KEY")
```

## Define model and tools for the graph

```
from typing import Literal

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")


tools = [get_weather]
model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
```

## Use sync connection
This sets up a synchronous connection to the database.

Synchronous connections execute operations in a blocking manner, meaning each operation waits for completion before moving to the next one. The DB_URI is the database connection URI, with the protocol used for connecting to a PostgreSQL database, authentication, and host where database is running. The connection_kwargs dictionary defines additional parameters for the database connection.

```
DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"
```

```
connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}
```

## With a connection pool
This manages a pool of reusable database connections: - Advantages: Efficient resource utilization, improved performance for frequent connections - Best for: Applications with many short-lived database operations

```
from psycopg_pool import ConnectionPool

with ConnectionPool(
    # Example configuration
    conninfo=DB_URI,
    max_size=20,
    kwargs=connection_kwargs,
) as pool:
    checkpointer = PostgresSaver(pool)

    # NOTE: you need to call .setup() the first time you're using your checkpointer
    checkpointer.setup()

    graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "1"}}
    res = graph.invoke({"messages": [("human", "what's the weather in sf")]}, config)
    checkpoint = checkpointer.get(config)
```


```
res
```

```
{'messages': [HumanMessage(content="what's the weather in sf", id='735b7deb-b0fe-4ad5-8920-2a3c69bbe9f7'),
  AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_lJHMDYgfgRdiEAGfFsEhqqKV', 'function': {'arguments': '{"city":"sf"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 57, 'total_tokens': 71}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-c56b3e04-08a9-4a59-b3f5-ee52d0ef0656-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'sf'}, 'id': 'call_lJHMDYgfgRdiEAGfFsEhqqKV', 'type': 'tool_call'}], usage_metadata={'input_tokens': 57, 'output_tokens': 14, 'total_tokens': 71}),
  ToolMessage(content="It's always sunny in sf", name='get_weather', id='0644bf7b-4d1b-4ebe-afa1-d2169ccce582', tool_call_id='call_lJHMDYgfgRdiEAGfFsEhqqKV'),
  AIMessage(content='The weather in San Francisco is always sunny!', response_metadata={'token_usage': {'completion_tokens': 10, 'prompt_tokens': 84, 'total_tokens': 94}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'stop', 'logprobs': None}, id='run-1ed9b8d0-9b50-4b87-b3a2-9860f51e9fd1-0', usage_metadata={'input_tokens': 84, 'output_tokens': 10, 'total_tokens': 94})]}
  ```

  ```
  checkpoint
  ```

  ```
  {'v': 1,
 'id': '1ef559b7-3b19-6ce8-8003-18d0f60634be',
 'ts': '2024-08-08T15:32:42.108605+00:00',
 'current_tasks': {},
 'pending_sends': [],
 'versions_seen': {'agent': {'tools': '00000000000000000000000000000004.022986cd20ae85c77ea298a383f69ba8',
   'start:agent': '00000000000000000000000000000002.d6f25946c3108fc12f27abbcf9b4cedc'},
  'tools': {'branch:agent:should_continue:tools': '00000000000000000000000000000003.065d90dd7f7cd091f0233855210bb2af'},
  '__input__': {},
  '__start__': {'__start__': '00000000000000000000000000000001.ab89befb52cc0e91e106ef7f500ea033'}},
 'channel_versions': {'agent': '00000000000000000000000000000005.065d90dd7f7cd091f0233855210bb2af',
  'tools': '00000000000000000000000000000005.',
  'messages': '00000000000000000000000000000005.b9adc75836c78af94af1d6811340dd13',
  '__start__': '00000000000000000000000000000002.',
  'start:agent': '00000000000000000000000000000003.',
  'branch:agent:should_continue:tools': '00000000000000000000000000000004.'},
 'channel_values': {'agent': 'agent',
  'messages': [HumanMessage(content="what's the weather in sf", id='735b7deb-b0fe-4ad5-8920-2a3c69bbe9f7'),
   AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_lJHMDYgfgRdiEAGfFsEhqqKV', 'function': {'arguments': '{"city":"sf"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 57, 'total_tokens': 71}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-c56b3e04-08a9-4a59-b3f5-ee52d0ef0656-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'sf'}, 'id': 'call_lJHMDYgfgRdiEAGfFsEhqqKV', 'type': 'tool_call'}], usage_metadata={'input_tokens': 57, 'output_tokens': 14, 'total_tokens': 71}),
   ToolMessage(content="It's always sunny in sf", name='get_weather', id='0644bf7b-4d1b-4ebe-afa1-d2169ccce582', tool_call_id='call_lJHMDYgfgRdiEAGfFsEhqqKV'),
   AIMessage(content='The weather in San Francisco is always sunny!', response_metadata={'token_usage': {'completion_tokens': 10, 'prompt_tokens': 84, 'total_tokens': 94}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'stop', 'logprobs': None}, id='run-1ed9b8d0-9b50-4b87-b3a2-9860f51e9fd1-0', usage_metadata={'input_tokens': 84, 'output_tokens': 10, 'total_tokens': 94})]}}
   ```

   ## With a connection
   This creates a single, dedicated connection to the database: - Advantages: Simple to use, suitable for longer transactions - Best for: Applications with fewer, longer-lived database operations

   ```
   from psycopg import Connection


with Connection.connect(DB_URI, **connection_kwargs) as conn:
    checkpointer = PostgresSaver(conn)
    # NOTE: you need to call .setup() the first time you're using your checkpointer
    # checkpointer.setup()
    graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "2"}}
    res = graph.invoke({"messages": [("human", "what's the weather in sf")]}, config)

    checkpoint_tuple = checkpointer.get_tuple(config)
```

```
checkpoint_tuple
```

```
CheckpointTuple(config={'configurable': {'thread_id': '2', 'checkpoint_ns': '', 'checkpoint_id': '1ef559b7-4650-6bfc-8003-1c5488f19318'}}, checkpoint={'v': 1, 'id': '1ef559b7-4650-6bfc-8003-1c5488f19318', 'ts': '2024-08-08T15:32:43.284551+00:00', 'current_tasks': {}, 'pending_sends': [], 'versions_seen': {'agent': {'tools': '00000000000000000000000000000004.022986cd20ae85c77ea298a383f69ba8', 'start:agent': '00000000000000000000000000000002.d6f25946c3108fc12f27abbcf9b4cedc'}, 'tools': {'branch:agent:should_continue:tools': '00000000000000000000000000000003.065d90dd7f7cd091f0233855210bb2af'}, '__input__': {}, '__start__': {'__start__': '00000000000000000000000000000001.ab89befb52cc0e91e106ef7f500ea033'}}, 'channel_versions': {'agent': '00000000000000000000000000000005.065d90dd7f7cd091f0233855210bb2af', 'tools': '00000000000000000000000000000005.', 'messages': '00000000000000000000000000000005.af9f229d2c4e14f4866eb37f72ec39f6', '__start__': '00000000000000000000000000000002.', 'start:agent': '00000000000000000000000000000003.', 'branch:agent:should_continue:tools': '00000000000000000000000000000004.'}, 'channel_values': {'agent': 'agent', 'messages': [HumanMessage(content="what's the weather in sf", id='7a14f96c-2d88-454f-9520-0e0287a4abbb'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_NcL4dBTYu4kSPGMKdxztdpjN', 'function': {'arguments': '{"city":"sf"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 57, 'total_tokens': 71}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-39adbf2c-36ef-40f6-9cad-8e1f8167fc19-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'sf'}, 'id': 'call_NcL4dBTYu4kSPGMKdxztdpjN', 'type': 'tool_call'}], usage_metadata={'input_tokens': 57, 'output_tokens': 14, 'total_tokens': 71}), ToolMessage(content="It's always sunny in sf", name='get_weather', id='c9f82354-3225-40a8-bf54-81f3e199043b', tool_call_id='call_NcL4dBTYu4kSPGMKdxztdpjN'), AIMessage(content='The weather in San Francisco is always sunny!', response_metadata={'token_usage': {'completion_tokens': 10, 'prompt_tokens': 84, 'total_tokens': 94}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'stop', 'logprobs': None}, id='run-83888be3-d681-42ca-ad67-e2f5ee8550de-0', usage_metadata={'input_tokens': 84, 'output_tokens': 10, 'total_tokens': 94})]}}, metadata={'step': 3, 'source': 'loop', 'writes': {'agent': {'messages': [AIMessage(content='The weather in San Francisco is always sunny!', response_metadata={'logprobs': None, 'model_name': 'gpt-4o-mini-2024-07-18', 'token_usage': {'total_tokens': 94, 'prompt_tokens': 84, 'completion_tokens': 10}, 'finish_reason': 'stop', 'system_fingerprint': 'fp_48196bc67a'}, id='run-83888be3-d681-42ca-ad67-e2f5ee8550de-0', usage_metadata={'input_tokens': 84, 'output_tokens': 10, 'total_tokens': 94})]}}}, parent_config={'configurable': {'thread_id': '2', 'checkpoint_ns': '', 'checkpoint_id': '1ef559b7-4087-681a-8002-88a5738f76f1'}}, pending_writes=[])
```


## With a connection string
This creates a connection based on a connection string: - Advantages: Simplicity, encapsulates connection details - Best for: Quick setup or when connection details are provided as a string

```
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "3"}}
    res = graph.invoke({"messages": [("human", "what's the weather in sf")]}, config)

    checkpoint_tuples = list(checkpointer.list(config))
```

```
checkpoint_tuples
```

```
[CheckpointTuple(config={'configurable': {'thread_id': '3', 'checkpoint_ns': '', 'checkpoint_id': '1ef559b7-5024-6476-8003-cf0a750e6b37'}}, checkpoint={'v': 1, 'id': '1ef559b7-5024-6476-8003-cf0a750e6b37', 'ts': '2024-08-08T15:32:44.314900+00:00', 'current_tasks': {}, 'pending_sends': [], 'versions_seen': {'agent': {'tools': '00000000000000000000000000000004.022986cd20ae85c77ea298a383f69ba8', 'start:agent': '00000000000000000000000000000002.d6f25946c3108fc12f27abbcf9b4cedc'}, 'tools': {'branch:agent:should_continue:tools': '00000000000000000000000000000003.065d90dd7f7cd091f0233855210bb2af'}, '__input__': {}, '__start__': {'__start__': '00000000000000000000000000000001.ab89befb52cc0e91e106ef7f500ea033'}}, 'channel_versions': {'agent': '00000000000000000000000000000005.065d90dd7f7cd091f0233855210bb2af', 'tools': '00000000000000000000000000000005.', 'messages': '00000000000000000000000000000005.3f8b8d9923575b911e17157008ab75ac', '__start__': '00000000000000000000000000000002.', 'start:agent': '00000000000000000000000000000003.', 'branch:agent:should_continue:tools': '00000000000000000000000000000004.'}, 'channel_values': {'agent': 'agent', 'messages': [HumanMessage(content="what's the weather in sf", id='5bf79d15-6332-4bf5-89bd-ee192b31ed84'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_9y3q1BiwW7zGh2gk2faInTRk', 'function': {'arguments': '{"city":"sf"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 57, 'total_tokens': 71}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_507c9469a1', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-2958adc7-f6a4-415d-ade1-5ee77e0b9276-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'sf'}, 'id': 'call_9y3q1BiwW7zGh2gk2faInTRk', 'type': 'tool_call'}], usage_metadata={'input_tokens': 57, 'output_tokens': 14, 'total_tokens': 71}), ToolMessage(content="It's always sunny in sf", name='get_weather', id='cac4f90a-dc3e-4bfa-940f-1c630289a583', tool_call_id='call_9y3q1BiwW7zGh2gk2faInTRk'), AIMessage(content='The weather in San Francisco is always sunny!', response_metadata={'token_usage': {'completion_tokens': 10, 'prompt_tokens': 84, 'total_tokens': 94}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'stop', 'logprobs': None}, id='run-97d3fb7a-3d2e-4090-84f4-dafdfe44553f-0', usage_metadata={'input_tokens': 84, 'output_tokens': 10, 'total_tokens': 94})]}}, metadata={'step': 3, 'source': 'loop', 'writes': {'agent': {'messages': [AIMessage(content='The weather in San Francisco is always sunny!', response_metadata={'logprobs': None, 'model_name': 'gpt-4o-mini-2024-07-18', 'token_usage': {'total_tokens': 94, 'prompt_tokens': 84, 'completion_tokens': 10}, 'finish_reason': 'stop', 'system_fingerprint': 'fp_48196bc67a'}, id='run-97d3fb7a-3d2e-4090-84f4-dafdfe44553f-0', usage_metadata={'input_tokens': 84, 'output_tokens': 10, 'total_tokens': 94})]}}}, parent_config={'configurable': {'thread_id': '3', 'checkpoint_ns': '', 'checkpoint_id': '1ef559b7-4b3d-6430-8002-b5c99d2eb4db'}}, pending_writes=None),
 CheckpointTuple(config={'configurable': {'thread_id': '3', 'checkpoint_ns': '', 'checkpoint_id': '1ef559b7-4b3d-6430-8002-b5c99d2eb4db'}}, checkpoint={'v': 1, 'id': '1ef559b7-4b3d-6430-8002-b5c99d2eb4db', 'ts': '2024-08-08T15:32:43.800857+00:00', 'current_tasks': {}, 'pending_sends': [], 'versions_seen': {'agent': {'start:agent': '00000000000000000000000000000002.d6f25946c3108fc12f27abbcf9b4cedc'}, 'tools': {'branch:agent:should_continue:tools': '00000000000000000000000000000003.065d90dd7f7cd091f0233855210bb2af'}, '__input__': {}, '__start__': {'__start__': '00000000000000000000000000000001.ab89befb52cc0e91e106ef7f500ea033'}}, 'channel_versions': {'agent': '00000000000000000000000000000004.', 'tools': '00000000000000000000000000000004.022986cd20ae85c77ea298a383f69ba8', 'messages': '00000000000000000000000000000004.1195f50946feaedb0bae1fdbfadc806b', '__start__': '00000000000000000000000000000002.', 'start:agent': '00000000000000000000000000000003.', 'branch:agent:should_continue:tools': '00000000000000000000000000000004.'}, 'channel_values': {'tools': 'tools', 'messages': [HumanMessage(content="what's the weather in sf", id='5bf79d15-6332-4bf5-89bd-ee192b31ed84'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_9y3q1BiwW7zGh2gk2faInTRk', 'function': {'arguments': '{"city":"sf"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 57, 'total_tokens': 71}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_507c9469a1', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-2958adc7-f6a4-415d-ade1-5ee77e0b9276-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'sf'}, 'id': 'call_9y3q1BiwW7zGh2gk2faInTRk', 'type': 'tool_call'}], usage_metadata={'input_tokens': 57, 'output_tokens': 14, 'total_tokens': 71}), ToolMessage(content="It's always sunny in sf", name='get_weather', id='cac4f90a-dc3e-4bfa-940f-1c630289a583', tool_call_id='call_9y3q1BiwW7zGh2gk2faInTRk')]}}, metadata={'step': 2, 'source': 'loop', 'writes': {'tools': {'messages': [ToolMessage(content="It's always sunny in sf", name='get_weather', id='cac4f90a-dc3e-4bfa-940f-1c630289a583', tool_call_id='call_9y3q1BiwW7zGh2gk2faInTRk')]}}}, parent_config={'configurable': {'thread_id': '3', 'checkpoint_ns': '', 'checkpoint_id': '1ef559b7-4b30-6078-8001-eaf8c9bd8844'}}, pending_writes=None),
 CheckpointTuple(config={'configurable': {'thread_id': '3', 'checkpoint_ns': '', 'checkpoint_id': '1ef559b7-4b30-6078-8001-eaf8c9bd8844'}}, checkpoint={'v': 1, 'id': '1ef559b7-4b30-6078-8001-eaf8c9bd8844', 'ts': '2024-08-08T15:32:43.795440+00:00', 'current_tasks': {}, 'pending_sends': [], 'versions_seen': {'agent': {'start:agent': '00000000000000000000000000000002.d6f25946c3108fc12f27abbcf9b4cedc'}, '__input__': {}, '__start__': {'__start__': '00000000000000000000000000000001.ab89befb52cc0e91e106ef7f500ea033'}}, 'channel_versions': {'agent': '00000000000000000000000000000003.065d90dd7f7cd091f0233855210bb2af', 'messages': '00000000000000000000000000000003.bab5fb3a70876f600f5f2fd46945ce5f', '__start__': '00000000000000000000000000000002.', 'start:agent': '00000000000000000000000000000003.', 'branch:agent:should_continue:tools': '00000000000000000000000000000003.065d90dd7f7cd091f0233855210bb2af'}, 'channel_values': {'agent': 'agent', 'messages': [HumanMessage(content="what's the weather in sf", id='5bf79d15-6332-4bf5-89bd-ee192b31ed84'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_9y3q1BiwW7zGh2gk2faInTRk', 'function': {'arguments': '{"city":"sf"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 57, 'total_tokens': 71}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_507c9469a1', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-2958adc7-f6a4-415d-ade1-5ee77e0b9276-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'sf'}, 'id': 'call_9y3q1BiwW7zGh2gk2faInTRk', 'type': 'tool_call'}], usage_metadata={'input_tokens': 57, 'output_tokens': 14, 'total_tokens': 71})], 'branch:agent:should_continue:tools': 'agent'}}, metadata={'step': 1, 'source': 'loop', 'writes': {'agent': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_9y3q1BiwW7zGh2gk2faInTRk', 'type': 'function', 'function': {'name': 'get_weather', 'arguments': '{"city":"sf"}'}}]}, response_metadata={'logprobs': None, 'model_name': 'gpt-4o-mini-2024-07-18', 'token_usage': {'total_tokens': 71, 'prompt_tokens': 57, 'completion_tokens': 14}, 'finish_reason': 'tool_calls', 'system_fingerprint': 'fp_507c9469a1'}, id='run-2958adc7-f6a4-415d-ade1-5ee77e0b9276-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'sf'}, 'id': 'call_9y3q1BiwW7zGh2gk2faInTRk', 'type': 'tool_call'}], usage_metadata={'input_tokens': 57, 'output_tokens': 14, 'total_tokens': 71})]}}}, parent_config={'configurable': {'thread_id': '3', 'checkpoint_ns': '', 'checkpoint_id': '1ef559b7-46d7-6116-8000-8976b7c89a2f'}}, pending_writes=None),
 CheckpointTuple(config={'configurable': {'thread_id': '3', 'checkpoint_ns': '', 'checkpoint_id': '1ef559b7-46d7-6116-8000-8976b7c89a2f'}}, checkpoint={'v': 1, 'id': '1ef559b7-46d7-6116-8000-8976b7c89a2f', 'ts': '2024-08-08T15:32:43.339573+00:00', 'current_tasks': {}, 'pending_sends': [], 'versions_seen': {'__input__': {}, '__start__': {'__start__': '00000000000000000000000000000001.ab89befb52cc0e91e106ef7f500ea033'}}, 'channel_versions': {'messages': '00000000000000000000000000000002.ba0c90d32863686481f7fe5eab9ecdf0', '__start__': '00000000000000000000000000000002.', 'start:agent': '00000000000000000000000000000002.d6f25946c3108fc12f27abbcf9b4cedc'}, 'channel_values': {'messages': [HumanMessage(content="what's the weather in sf", id='5bf79d15-6332-4bf5-89bd-ee192b31ed84')], 'start:agent': '__start__'}}, metadata={'step': 0, 'source': 'loop', 'writes': None}, parent_config={'configurable': {'thread_id': '3', 'checkpoint_ns': '', 'checkpoint_id': '1ef559b7-46ce-6c64-bfff-ef7fe2663573'}}, pending_writes=None),
 CheckpointTuple(config={'configurable': {'thread_id': '3', 'checkpoint_ns': '', 'checkpoint_id': '1ef559b7-46ce-6c64-bfff-ef7fe2663573'}}, checkpoint={'v': 1, 'id': '1ef559b7-46ce-6c64-bfff-ef7fe2663573', 'ts': '2024-08-08T15:32:43.336188+00:00', 'current_tasks': {}, 'pending_sends': [], 'versions_seen': {'__input__': {}}, 'channel_versions': {'__start__': '00000000000000000000000000000001.ab89befb52cc0e91e106ef7f500ea033'}, 'channel_values': {'__start__': {'messages': [['human', "what's the weather in sf"]]}}}, metadata={'step': -1, 'source': 'input', 'writes': {'messages': [['human', "what's the weather in sf"]]}}, parent_config=None, pending_writes=None)]
 ```

 ## Use async connection
 This sets up an asynchronous connection to the database.

Async connections allow non-blocking database operations. This means other parts of your application can continue running while waiting for database operations to complete. It's particularly useful in high-concurrency scenarios or when dealing with I/O-bound operations.


## With a connection pool

```
from psycopg_pool import AsyncConnectionPool

async with AsyncConnectionPool(
    # Example configuration
    conninfo=DB_URI,
    max_size=20,
    kwargs=connection_kwargs,
) as pool:
    checkpointer = AsyncPostgresSaver(pool)

    # NOTE: you need to call .setup() the first time you're using your checkpointer
    await checkpointer.setup()

    graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "4"}}
    res = await graph.ainvoke(
        {"messages": [("human", "what's the weather in nyc")]}, config
    )

    checkpoint = await checkpointer.aget(config)
```

```
checkpoint
```

```
{'v': 1,
 'id': '1ef559b7-5cc9-6460-8003-8655824c0944',
 'ts': '2024-08-08T15:32:45.640793+00:00',
 'current_tasks': {},
 'pending_sends': [],
 'versions_seen': {'agent': {'tools': '00000000000000000000000000000004.022986cd20ae85c77ea298a383f69ba8',
   'start:agent': '00000000000000000000000000000002.d6f25946c3108fc12f27abbcf9b4cedc'},
  'tools': {'branch:agent:should_continue:tools': '00000000000000000000000000000003.065d90dd7f7cd091f0233855210bb2af'},
  '__input__': {},
  '__start__': {'__start__': '00000000000000000000000000000001.0e148ae3debe753278387e84f786e863'}},
 'channel_versions': {'agent': '00000000000000000000000000000005.065d90dd7f7cd091f0233855210bb2af',
  'tools': '00000000000000000000000000000005.',
  'messages': '00000000000000000000000000000005.d869fc7231619df0db74feed624efe41',
  '__start__': '00000000000000000000000000000002.',
  'start:agent': '00000000000000000000000000000003.',
  'branch:agent:should_continue:tools': '00000000000000000000000000000004.'},
 'channel_values': {'agent': 'agent',
  'messages': [HumanMessage(content="what's the weather in nyc", id='d883b8a0-99de-486d-91a2-bcfa7f25dc05'),
   AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_H6TAYfyd6AnaCrkQGs6Q2fVp', 'function': {'arguments': '{"city":"nyc"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 15, 'prompt_tokens': 58, 'total_tokens': 73}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-6f542f84-ad73-444c-8ef7-b5ea75a2e09b-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'nyc'}, 'id': 'call_H6TAYfyd6AnaCrkQGs6Q2fVp', 'type': 'tool_call'}], usage_metadata={'input_tokens': 58, 'output_tokens': 15, 'total_tokens': 73}),
   ToolMessage(content='It might be cloudy in nyc', name='get_weather', id='c0e52254-77a4-4ea9-a2b7-61dd2d65ec68', tool_call_id='call_H6TAYfyd6AnaCrkQGs6Q2fVp'),
   AIMessage(content='The weather in NYC might be cloudy.', response_metadata={'token_usage': {'completion_tokens': 9, 'prompt_tokens': 88, 'total_tokens': 97}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'stop', 'logprobs': None}, id='run-977140d4-7582-40c3-b2b6-31b542c430a3-0', usage_metadata={'input_tokens': 88, 'output_tokens': 9, 'total_tokens': 97})]}}
   ```

## With a connection
   
```
   from psycopg import AsyncConnection

async with await AsyncConnection.connect(DB_URI, **connection_kwargs) as conn:
    checkpointer = AsyncPostgresSaver(conn)
    graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "5"}}
    res = await graph.ainvoke(
        {"messages": [("human", "what's the weather in nyc")]}, config
    )
    checkpoint_tuple = await checkpointer.aget_tuple(config)
```

```
checkpoint_tuple
```

```
CheckpointTuple(config={'configurable': {'thread_id': '5', 'checkpoint_ns': '', 'checkpoint_id': '1ef559b7-65b4-60ca-8003-1ef4b620559a'}}, checkpoint={'v': 1, 'id': '1ef559b7-65b4-60ca-8003-1ef4b620559a', 'ts': '2024-08-08T15:32:46.575814+00:00', 'current_tasks': {}, 'pending_sends': [], 'versions_seen': {'agent': {'tools': '00000000000000000000000000000004.022986cd20ae85c77ea298a383f69ba8', 'start:agent': '00000000000000000000000000000002.d6f25946c3108fc12f27abbcf9b4cedc'}, 'tools': {'branch:agent:should_continue:tools': '00000000000000000000000000000003.065d90dd7f7cd091f0233855210bb2af'}, '__input__': {}, '__start__': {'__start__': '00000000000000000000000000000001.0e148ae3debe753278387e84f786e863'}}, 'channel_versions': {'agent': '00000000000000000000000000000005.065d90dd7f7cd091f0233855210bb2af', 'tools': '00000000000000000000000000000005.', 'messages': '00000000000000000000000000000005.1557a6006d58f736d5cb2dd5c5f10111', '__start__': '00000000000000000000000000000002.', 'start:agent': '00000000000000000000000000000003.', 'branch:agent:should_continue:tools': '00000000000000000000000000000004.'}, 'channel_values': {'agent': 'agent', 'messages': [HumanMessage(content="what's the weather in nyc", id='935e7732-b288-49bd-9ec2-1f7610cc38cb'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_94KtjtPmsiaj7T8yXvL7Ef31', 'function': {'arguments': '{"city":"nyc"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 15, 'prompt_tokens': 58, 'total_tokens': 73}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-790c929a-7982-49e7-af67-2cbe4a86373b-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'nyc'}, 'id': 'call_94KtjtPmsiaj7T8yXvL7Ef31', 'type': 'tool_call'}], usage_metadata={'input_tokens': 58, 'output_tokens': 15, 'total_tokens': 73}), ToolMessage(content='It might be cloudy in nyc', name='get_weather', id='b2dc1073-abc4-4492-8982-434a7e32e445', tool_call_id='call_94KtjtPmsiaj7T8yXvL7Ef31'), AIMessage(content='The weather in NYC might be cloudy.', response_metadata={'token_usage': {'completion_tokens': 9, 'prompt_tokens': 88, 'total_tokens': 97}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'stop', 'logprobs': None}, id='run-7e8a7f16-d8e1-457a-89f3-192102396449-0', usage_metadata={'input_tokens': 88, 'output_tokens': 9, 'total_tokens': 97})]}}, metadata={'step': 3, 'source': 'loop', 'writes': {'agent': {'messages': [AIMessage(content='The weather in NYC might be cloudy.', response_metadata={'logprobs': None, 'model_name': 'gpt-4o-mini-2024-07-18', 'token_usage': {'total_tokens': 97, 'prompt_tokens': 88, 'completion_tokens': 9}, 'finish_reason': 'stop', 'system_fingerprint': 'fp_48196bc67a'}, id='run-7e8a7f16-d8e1-457a-89f3-192102396449-0', usage_metadata={'input_tokens': 88, 'output_tokens': 9, 'total_tokens': 97})]}}}, parent_config={'configurable': {'thread_id': '5', 'checkpoint_ns': '', 'checkpoint_id': '1ef559b7-62ae-6128-8002-c04af82bcd41'}}, pending_writes=[])
```

## With a connection string

```
async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
    graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "6"}}
    res = await graph.ainvoke(
        {"messages": [("human", "what's the weather in nyc")]}, config
    )
    checkpoint_tuples = [c async for c in checkpointer.alist(config)]
```

```
checkpoint_tuples
```

```
[CheckpointTuple(config={'configurable': {'thread_id': '6', 'checkpoint_ns': '', 'checkpoint_id': '1ef559b7-723c-67de-8003-63bd4eab35af'}}, checkpoint={'v': 1, 'id': '1ef559b7-723c-67de-8003-63bd4eab35af', 'ts': '2024-08-08T15:32:47.890003+00:00', 'current_tasks': {}, 'pending_sends': [], 'versions_seen': {'agent': {'tools': '00000000000000000000000000000004.022986cd20ae85c77ea298a383f69ba8', 'start:agent': '00000000000000000000000000000002.d6f25946c3108fc12f27abbcf9b4cedc'}, 'tools': {'branch:agent:should_continue:tools': '00000000000000000000000000000003.065d90dd7f7cd091f0233855210bb2af'}, '__input__': {}, '__start__': {'__start__': '00000000000000000000000000000001.0e148ae3debe753278387e84f786e863'}}, 'channel_versions': {'agent': '00000000000000000000000000000005.065d90dd7f7cd091f0233855210bb2af', 'tools': '00000000000000000000000000000005.', 'messages': '00000000000000000000000000000005.b6fe2a26011590cfe8fd6a39151a9e92', '__start__': '00000000000000000000000000000002.', 'start:agent': '00000000000000000000000000000003.', 'branch:agent:should_continue:tools': '00000000000000000000000000000004.'}, 'channel_values': {'agent': 'agent', 'messages': [HumanMessage(content="what's the weather in nyc", id='977ddb90-9991-44cb-9f73-361c6dd21396'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_QIFCuh4zfP9owpjToycJiZf7', 'function': {'arguments': '{"city":"nyc"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 15, 'prompt_tokens': 58, 'total_tokens': 73}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-47b10c48-4db3-46d8-b4fa-e021818e01c5-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'nyc'}, 'id': 'call_QIFCuh4zfP9owpjToycJiZf7', 'type': 'tool_call'}], usage_metadata={'input_tokens': 58, 'output_tokens': 15, 'total_tokens': 73}), ToolMessage(content='It might be cloudy in nyc', name='get_weather', id='798c520f-4f9a-4f6d-a389-da721eb4d4ce', tool_call_id='call_QIFCuh4zfP9owpjToycJiZf7'), AIMessage(content='The weather in NYC might be cloudy.', response_metadata={'token_usage': {'completion_tokens': 9, 'prompt_tokens': 88, 'total_tokens': 97}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'stop', 'logprobs': None}, id='run-4a34e05d-8bcf-41ad-adc3-715919fde64c-0', usage_metadata={'input_tokens': 88, 'output_tokens': 9, 'total_tokens': 97})]}}, metadata={'step': 3, 'source': 'loop', 'writes': {'agent': {'messages': [AIMessage(content='The weather in NYC might be cloudy.', response_metadata={'logprobs': None, 'model_name': 'gpt-4o-mini-2024-07-18', 'token_usage': {'total_tokens': 97, 'prompt_tokens': 88, 'completion_tokens': 9}, 'finish_reason': 'stop', 'system_fingerprint': 'fp_48196bc67a'}, id='run-4a34e05d-8bcf-41ad-adc3-715919fde64c-0', usage_metadata={'input_tokens': 88, 'output_tokens': 9, 'total_tokens': 97})]}}}, parent_config={'configurable': {'thread_id': '6', 'checkpoint_ns': '', 'checkpoint_id': '1ef559b7-6bf5-63c6-8002-ed990dbbc96e'}}, pending_writes=None),
 CheckpointTuple(config={'configurable': {'thread_id': '6', 'checkpoint_ns': '', 'checkpoint_id': '1ef559b7-6bf5-63c6-8002-ed990dbbc96e'}}, checkpoint={'v': 1, 'id': '1ef559b7-6bf5-63c6-8002-ed990dbbc96e', 'ts': '2024-08-08T15:32:47.231667+00:00', 'current_tasks': {}, 'pending_sends': [], 'versions_seen': {'agent': {'start:agent': '00000000000000000000000000000002.d6f25946c3108fc12f27abbcf9b4cedc'}, 'tools': {'branch:agent:should_continue:tools': '00000000000000000000000000000003.065d90dd7f7cd091f0233855210bb2af'}, '__input__': {}, '__start__': {'__start__': '00000000000000000000000000000001.0e148ae3debe753278387e84f786e863'}}, 'channel_versions': {'agent': '00000000000000000000000000000004.', 'tools': '00000000000000000000000000000004.022986cd20ae85c77ea298a383f69ba8', 'messages': '00000000000000000000000000000004.c9074f2a41f05486b5efb86353dc75c0', '__start__': '00000000000000000000000000000002.', 'start:agent': '00000000000000000000000000000003.', 'branch:agent:should_continue:tools': '00000000000000000000000000000004.'}, 'channel_values': {'tools': 'tools', 'messages': [HumanMessage(content="what's the weather in nyc", id='977ddb90-9991-44cb-9f73-361c6dd21396'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_QIFCuh4zfP9owpjToycJiZf7', 'function': {'arguments': '{"city":"nyc"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 15, 'prompt_tokens': 58, 'total_tokens': 73}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-47b10c48-4db3-46d8-b4fa-e021818e01c5-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'nyc'}, 'id': 'call_QIFCuh4zfP9owpjToycJiZf7', 'type': 'tool_call'}], usage_metadata={'input_tokens': 58, 'output_tokens': 15, 'total_tokens': 73}), ToolMessage(content='It might be cloudy in nyc', name='get_weather', id='798c520f-4f9a-4f6d-a389-da721eb4d4ce', tool_call_id='call_QIFCuh4zfP9owpjToycJiZf7')]}}, metadata={'step': 2, 'source': 'loop', 'writes': {'tools': {'messages': [ToolMessage(content='It might be cloudy in nyc', name='get_weather', id='798c520f-4f9a-4f6d-a389-da721eb4d4ce', tool_call_id='call_QIFCuh4zfP9owpjToycJiZf7')]}}}, parent_config={'configurable': {'thread_id': '6', 'checkpoint_ns': '', 'checkpoint_id': '1ef559b7-6be0-6926-8001-1a8ce73baf9e'}}, pending_writes=None),
 CheckpointTuple(config={'configurable': {'thread_id': '6', 'checkpoint_ns': '', 'checkpoint_id': '1ef559b7-6be0-6926-8001-1a8ce73baf9e'}}, checkpoint={'v': 1, 'id': '1ef559b7-6be0-6926-8001-1a8ce73baf9e', 'ts': '2024-08-08T15:32:47.223198+00:00', 'current_tasks': {}, 'pending_sends': [], 'versions_seen': {'agent': {'start:agent': '00000000000000000000000000000002.d6f25946c3108fc12f27abbcf9b4cedc'}, '__input__': {}, '__start__': {'__start__': '00000000000000000000000000000001.0e148ae3debe753278387e84f786e863'}}, 'channel_versions': {'agent': '00000000000000000000000000000003.065d90dd7f7cd091f0233855210bb2af', 'messages': '00000000000000000000000000000003.097b5407d709b297591f1ef5d50c8368', '__start__': '00000000000000000000000000000002.', 'start:agent': '00000000000000000000000000000003.', 'branch:agent:should_continue:tools': '00000000000000000000000000000003.065d90dd7f7cd091f0233855210bb2af'}, 'channel_values': {'agent': 'agent', 'messages': [HumanMessage(content="what's the weather in nyc", id='977ddb90-9991-44cb-9f73-361c6dd21396'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_QIFCuh4zfP9owpjToycJiZf7', 'function': {'arguments': '{"city":"nyc"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 15, 'prompt_tokens': 58, 'total_tokens': 73}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-47b10c48-4db3-46d8-b4fa-e021818e01c5-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'nyc'}, 'id': 'call_QIFCuh4zfP9owpjToycJiZf7', 'type': 'tool_call'}], usage_metadata={'input_tokens': 58, 'output_tokens': 15, 'total_tokens': 73})], 'branch:agent:should_continue:tools': 'agent'}}, metadata={'step': 1, 'source': 'loop', 'writes': {'agent': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_QIFCuh4zfP9owpjToycJiZf7', 'type': 'function', 'function': {'name': 'get_weather', 'arguments': '{"city":"nyc"}'}}]}, response_metadata={'logprobs': None, 'model_name': 'gpt-4o-mini-2024-07-18', 'token_usage': {'total_tokens': 73, 'prompt_tokens': 58, 'completion_tokens': 15}, 'finish_reason': 'tool_calls', 'system_fingerprint': 'fp_48196bc67a'}, id='run-47b10c48-4db3-46d8-b4fa-e021818e01c5-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'nyc'}, 'id': 'call_QIFCuh4zfP9owpjToycJiZf7', 'type': 'tool_call'}], usage_metadata={'input_tokens': 58, 'output_tokens': 15, 'total_tokens': 73})]}}}, parent_config={'configurable': {'thread_id': '6', 'checkpoint_ns': '', 'checkpoint_id': '1ef559b7-663d-60b4-8000-10a8922bffbf'}}, pending_writes=None),
 CheckpointTuple(config={'configurable': {'thread_id': '6', 'checkpoint_ns': '', 'checkpoint_id': '1ef559b7-663d-60b4-8000-10a8922bffbf'}}, checkpoint={'v': 1, 'id': '1ef559b7-663d-60b4-8000-10a8922bffbf', 'ts': '2024-08-08T15:32:46.631935+00:00', 'current_tasks': {}, 'pending_sends': [], 'versions_seen': {'__input__': {}, '__start__': {'__start__': '00000000000000000000000000000001.0e148ae3debe753278387e84f786e863'}}, 'channel_versions': {'messages': '00000000000000000000000000000002.2a79db8da664e437bdb25ea804457ca7', '__start__': '00000000000000000000000000000002.', 'start:agent': '00000000000000000000000000000002.d6f25946c3108fc12f27abbcf9b4cedc'}, 'channel_values': {'messages': [HumanMessage(content="what's the weather in nyc", id='977ddb90-9991-44cb-9f73-361c6dd21396')], 'start:agent': '__start__'}}, metadata={'step': 0, 'source': 'loop', 'writes': None}, parent_config={'configurable': {'thread_id': '6', 'checkpoint_ns': '', 'checkpoint_id': '1ef559b7-6637-6d4e-bfff-6cecf690c3cb'}}, pending_writes=None),
 CheckpointTuple(config={'configurable': {'thread_id': '6', 'checkpoint_ns': '', 'checkpoint_id': '1ef559b7-6637-6d4e-bfff-6cecf690c3cb'}}, checkpoint={'v': 1, 'id': '1ef559b7-6637-6d4e-bfff-6cecf690c3cb', 'ts': '2024-08-08T15:32:46.629806+00:00', 'current_tasks': {}, 'pending_sends': [], 'versions_seen': {'__input__': {}}, 'channel_versions': {'__start__': '00000000000000000000000000000001.0e148ae3debe753278387e84f786e863'}, 'channel_values': {'__start__': {'messages': [['human', "what's the weather in nyc"]]}}}, metadata={'step': -1, 'source': 'input', 'writes': {'messages': [['human', "what's the weather in nyc"]]}}, parent_config=None, pending_writes=None)]
 ```

 # How to use MongoDB checkpointer for persistence

 When creating LangGraph agents, you can also set them up so that they persist their state. This allows you to do things like interact with an agent multiple times and have it remember previous interactions.

This reference implementation shows how to use MongoDB as the backend for persisting checkpoint state using the langgraph-checkpoint-mongodb library.

For demonstration purposes we add persistence to a prebuilt ReAct agent.

In general, you can add a checkpointer to any custom graph that you build like this:

```
from langgraph.graph import StateGraph

builder = StateGraph(...)
# ... define the graph
checkpointer = # mongodb checkpointer (see examples below)
graph = builder.compile(checkpointer=checkpointer)
...
```

## Setup
To use the MongoDB checkpointer, you will need a MongoDB cluster. Follow this guide to create a cluster if you don't already have one.

Next, let's install the required packages and set our API keys

```
%%capture --no-stderr
%pip install -U pymongo langgraph langgraph-checkpoint-mongodb
```

```
import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("OPENAI_API_KEY")
```

```
OPENAI_API_KEY:  ········
```

## Define model and tools for the graph

```
from typing import Literal

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")


tools = [get_weather]
model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
```

## MongoDB checkpointer usage
## With a connection string

This creates a connection to MongoDB directly using the connection string of your cluster. This is ideal for use in scripts, one-off operations and short-lived applications.

```
from langgraph.checkpoint.mongodb import MongoDBSaver

MONGODB_URI = "localhost:27017"  # replace this with your connection string

with MongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer:
    graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "1"}}
    response = graph.invoke(
        {"messages": [("human", "what's the weather in sf")]}, config
    )
```

```
response
```

```
{'messages': [HumanMessage(content="what's the weather in sf", additional_kwargs={}, response_metadata={}, id='729afd6a-fdc0-4192-a255-1dac065c79b2'),
  AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_YqaO8oU3BhGmIz9VHTxqGyyN', 'function': {'arguments': '{"city":"sf"}', 'name': 'get_weather'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 57, 'total_tokens': 71, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_39a40c96a0', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-b45c0c12-c68e-4392-92dd-5d325d0a9f60-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'sf'}, 'id': 'call_YqaO8oU3BhGmIz9VHTxqGyyN', 'type': 'tool_call'}], usage_metadata={'input_tokens': 57, 'output_tokens': 14, 'total_tokens': 71, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}),
  ToolMessage(content="It's always sunny in sf", name='get_weather', id='0c72eb29-490b-44df-898f-8454c314eac1', tool_call_id='call_YqaO8oU3BhGmIz9VHTxqGyyN'),
  AIMessage(content='The weather in San Francisco is always sunny!', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 10, 'prompt_tokens': 84, 'total_tokens': 94, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_818c284075', 'finish_reason': 'stop', 'logprobs': None}, id='run-33f54c91-0ba9-48b7-9b25-5a972bbdeea9-0', usage_metadata={'input_tokens': 84, 'output_tokens': 10, 'total_tokens': 94, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})]}
  ```

  ## Using the MongoDB client
  This creates a connection to MongoDB using the MongoDB client. This is ideal for long-running applications since it allows you to reuse the client instance for multiple database operations without needing to reinitialize the connection each time.

  ```
  from pymongo import MongoClient

mongodb_client = MongoClient(MONGODB_URI)

checkpointer = MongoDBSaver(mongodb_client)
graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
config = {"configurable": {"thread_id": "2"}}
response = graph.invoke({"messages": [("user", "What's the weather in sf?")]}, config)
```

```
response
```

```
{'messages': [HumanMessage(content="What's the weather in sf?", additional_kwargs={}, response_metadata={}, id='4ce68bee-a843-4b08-9c02-7a0e3b010110'),
  AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_MvGxq9IU9wvW9mfYKSALHtGu', 'function': {'arguments': '{"city":"sf"}', 'name': 'get_weather'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 57, 'total_tokens': 71, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_6fc10e10eb', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-9712c5a4-376c-4812-a0c4-1b522334a59d-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'sf'}, 'id': 'call_MvGxq9IU9wvW9mfYKSALHtGu', 'type': 'tool_call'}], usage_metadata={'input_tokens': 57, 'output_tokens': 14, 'total_tokens': 71, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}),
  ToolMessage(content="It's always sunny in sf", name='get_weather', id='b4eed38d-bcaf-4497-ad08-f21ccd6a8c30', tool_call_id='call_MvGxq9IU9wvW9mfYKSALHtGu'),
  AIMessage(content='The weather in San Francisco is always sunny!', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 10, 'prompt_tokens': 84, 'total_tokens': 94, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_6fc10e10eb', 'finish_reason': 'stop', 'logprobs': None}, id='run-c6c4ad75-89ef-4b4f-9ca4-bd52ccb0729b-0', usage_metadata={'input_tokens': 84, 'output_tokens': 10, 'total_tokens': 94, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})]}
```

```
# Retrieve the latest checkpoint for the given thread ID
# To retrieve a specific checkpoint, pass the checkpoint_id in the config
checkpointer.get_tuple(config)
```

```
CheckpointTuple(config={'configurable': {'thread_id': '2', 'checkpoint_ns': '', 'checkpoint_id': '1efb8c75-9262-68b4-8003-1ac1ef198757'}}, checkpoint={'v': 1, 'ts': '2024-12-12T20:26:20.545003+00:00', 'id': '1efb8c75-9262-68b4-8003-1ac1ef198757', 'channel_values': {'messages': [HumanMessage(content="What's the weather in sf?", additional_kwargs={}, response_metadata={}, id='4ce68bee-a843-4b08-9c02-7a0e3b010110'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_MvGxq9IU9wvW9mfYKSALHtGu', 'function': {'arguments': '{"city":"sf"}', 'name': 'get_weather'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 57, 'total_tokens': 71, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_6fc10e10eb', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-9712c5a4-376c-4812-a0c4-1b522334a59d-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'sf'}, 'id': 'call_MvGxq9IU9wvW9mfYKSALHtGu', 'type': 'tool_call'}], usage_metadata={'input_tokens': 57, 'output_tokens': 14, 'total_tokens': 71, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}), ToolMessage(content="It's always sunny in sf", name='get_weather', id='b4eed38d-bcaf-4497-ad08-f21ccd6a8c30', tool_call_id='call_MvGxq9IU9wvW9mfYKSALHtGu'), AIMessage(content='The weather in San Francisco is always sunny!', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 10, 'prompt_tokens': 84, 'total_tokens': 94, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_6fc10e10eb', 'finish_reason': 'stop', 'logprobs': None}, id='run-c6c4ad75-89ef-4b4f-9ca4-bd52ccb0729b-0', usage_metadata={'input_tokens': 84, 'output_tokens': 10, 'total_tokens': 94, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})], 'agent': 'agent'}, 'channel_versions': {'__start__': 2, 'messages': 5, 'start:agent': 3, 'agent': 5, 'branch:agent:should_continue:tools': 4, 'tools': 5}, 'versions_seen': {'__input__': {}, '__start__': {'__start__': 1}, 'agent': {'start:agent': 2, 'tools': 4}, 'tools': {'branch:agent:should_continue:tools': 3}}, 'pending_sends': []}, metadata={'source': 'loop', 'writes': {'agent': {'messages': [AIMessage(content='The weather in San Francisco is always sunny!', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 10, 'prompt_tokens': 84, 'total_tokens': 94, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_6fc10e10eb', 'finish_reason': 'stop', 'logprobs': None}, id='run-c6c4ad75-89ef-4b4f-9ca4-bd52ccb0729b-0', usage_metadata={'input_tokens': 84, 'output_tokens': 10, 'total_tokens': 94, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})]}}, 'thread_id': '2', 'step': 3, 'parents': {}}, parent_config={'configurable': {'thread_id': '2', 'checkpoint_ns': '', 'checkpoint_id': '1efb8c75-8d89-6ffe-8002-84a4312c4fed'}}, pending_writes=[])
```

```
# Remember to close the connection after you're done
mongodb_client.close()
```

## Using an async connection
This creates a short-lived asynchronous connection to MongoDB.

Async connections allow non-blocking database operations. This means other parts of your application can continue running while waiting for database operations to complete. It's particularly useful in high-concurrency scenarios or when dealing with I/O-bound operations.

```
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver

async with AsyncMongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer:
    graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "3"}}
    response = await graph.ainvoke(
        {"messages": [("user", "What's the weather in sf?")]}, config
    )
```

```
response
```

```
{'messages': [HumanMessage(content="What's the weather in sf?", additional_kwargs={}, response_metadata={}, id='fed70fe6-1b2e-4481-9bfc-063df3b587dc'),
  AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_miRiF3vPQv98wlDHl6CeRxBy', 'function': {'arguments': '{"city":"sf"}', 'name': 'get_weather'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 57, 'total_tokens': 71, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_6fc10e10eb', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-7f2d5153-973e-4a9e-8b71-a77625c342cf-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'sf'}, 'id': 'call_miRiF3vPQv98wlDHl6CeRxBy', 'type': 'tool_call'}], usage_metadata={'input_tokens': 57, 'output_tokens': 14, 'total_tokens': 71, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}),
  ToolMessage(content="It's always sunny in sf", name='get_weather', id='49035e8e-8aee-4d9d-88ab-9a1bc10ecbd3', tool_call_id='call_miRiF3vPQv98wlDHl6CeRxBy'),
  AIMessage(content='The weather in San Francisco is always sunny!', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 10, 'prompt_tokens': 84, 'total_tokens': 94, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_6fc10e10eb', 'finish_reason': 'stop', 'logprobs': None}, id='run-9403d502-391e-4407-99fd-eec8ed184e50-0', usage_metadata={'input_tokens': 84, 'output_tokens': 10, 'total_tokens': 94, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})]}
```

## Using the async MongoDB client
This routes connections to MongoDB through an asynchronous MongoDB client.

```
from pymongo import AsyncMongoClient

async_mongodb_client = AsyncMongoClient(MONGODB_URI)

checkpointer = AsyncMongoDBSaver(async_mongodb_client)
graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
config = {"configurable": {"thread_id": "4"}}
response = await graph.ainvoke(
    {"messages": [("user", "What's the weather in sf?")]}, config
)
```

```
response
```

```
{'messages': [HumanMessage(content="What's the weather in sf?", additional_kwargs={}, response_metadata={}, id='58282e2b-4cc1-40a1-8e65-420a2177bbd6'),
  AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_SJFViVHl1tYTZDoZkNN3ePhJ', 'function': {'arguments': '{"city":"sf"}', 'name': 'get_weather'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 57, 'total_tokens': 71, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_bba3c8e70b', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-131af8c1-d388-4d7f-9137-da59ebd5fefd-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'sf'}, 'id': 'call_SJFViVHl1tYTZDoZkNN3ePhJ', 'type': 'tool_call'}], usage_metadata={'input_tokens': 57, 'output_tokens': 14, 'total_tokens': 71, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}),
  ToolMessage(content="It's always sunny in sf", name='get_weather', id='6090a56f-177b-4d3f-b16a-9c05f23800e3', tool_call_id='call_SJFViVHl1tYTZDoZkNN3ePhJ'),
  AIMessage(content='The weather in San Francisco is always sunny!', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 10, 'prompt_tokens': 84, 'total_tokens': 94, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_6fc10e10eb', 'finish_reason': 'stop', 'logprobs': None}, id='run-6ff5ddf5-6e13-4126-8df9-81c8638355fc-0', usage_metadata={'input_tokens': 84, 'output_tokens': 10, 'total_tokens': 94, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})]}
```

```
# Retrieve the latest checkpoint for the given thread ID
# To retrieve a specific checkpoint, pass the checkpoint_id in the config
latest_checkpoint = await checkpointer.aget_tuple(config)
print(latest_checkpoint)
```

```
CheckpointTuple(config={'configurable': {'thread_id': '4', 'checkpoint_ns': '', 'checkpoint_id': '1efb8c76-21f4-6d10-8003-9496e1754e93'}}, checkpoint={'v': 1, 'ts': '2024-12-12T20:26:35.599560+00:00', 'id': '1efb8c76-21f4-6d10-8003-9496e1754e93', 'channel_values': {'messages': [HumanMessage(content="What's the weather in sf?", additional_kwargs={}, response_metadata={}, id='58282e2b-4cc1-40a1-8e65-420a2177bbd6'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_SJFViVHl1tYTZDoZkNN3ePhJ', 'function': {'arguments': '{"city":"sf"}', 'name': 'get_weather'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 57, 'total_tokens': 71, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_bba3c8e70b', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-131af8c1-d388-4d7f-9137-da59ebd5fefd-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'sf'}, 'id': 'call_SJFViVHl1tYTZDoZkNN3ePhJ', 'type': 'tool_call'}], usage_metadata={'input_tokens': 57, 'output_tokens': 14, 'total_tokens': 71, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}), ToolMessage(content="It's always sunny in sf", name='get_weather', id='6090a56f-177b-4d3f-b16a-9c05f23800e3', tool_call_id='call_SJFViVHl1tYTZDoZkNN3ePhJ'), AIMessage(content='The weather in San Francisco is always sunny!', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 10, 'prompt_tokens': 84, 'total_tokens': 94, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_6fc10e10eb', 'finish_reason': 'stop', 'logprobs': None}, id='run-6ff5ddf5-6e13-4126-8df9-81c8638355fc-0', usage_metadata={'input_tokens': 84, 'output_tokens': 10, 'total_tokens': 94, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})], 'agent': 'agent'}, 'channel_versions': {'__start__': 2, 'messages': 5, 'start:agent': 3, 'agent': 5, 'branch:agent:should_continue:tools': 4, 'tools': 5}, 'versions_seen': {'__input__': {}, '__start__': {'__start__': 1}, 'agent': {'start:agent': 2, 'tools': 4}, 'tools': {'branch:agent:should_continue:tools': 3}}, 'pending_sends': []}, metadata={'source': 'loop', 'writes': {'agent': {'messages': [AIMessage(content='The weather in San Francisco is always sunny!', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 10, 'prompt_tokens': 84, 'total_tokens': 94, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_6fc10e10eb', 'finish_reason': 'stop', 'logprobs': None}, id='run-6ff5ddf5-6e13-4126-8df9-81c8638355fc-0', usage_metadata={'input_tokens': 84, 'output_tokens': 10, 'total_tokens': 94, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})]}}, 'thread_id': '4', 'step': 3, 'parents': {}}, parent_config={'configurable': {'thread_id': '4', 'checkpoint_ns': '', 'checkpoint_id': '1efb8c76-1c6c-6474-8002-9c2595cd481c'}}, pending_writes=[])
```

```
# Remember to close the connection after you're done
await async_mongodb_client.close()
```

## How to create a custom checkpointer using Redis
When creating LangGraph agents, you can also set them up so that they persist their state. This allows you to do things like interact with an agent multiple times and have it remember previous interactions.

This reference implementation shows how to use Redis as the backend for persisting checkpoint state. Make sure that you have Redis running on port 6379 for going through this guide.

For demonstration purposes we add persistence to the pre-built create react agent.

In general, you can add a checkpointer to any custom graph that you build like this:

```
from langgraph.graph import StateGraph

builder = StateGraph(....)
# ... define the graph
checkpointer = # redis checkpointer (see examples below)
graph = builder.compile(checkpointer=checkpointer)
...
```

## Setup
First, let's install the required packages and set our API keys

```
%%capture --no-stderr
%pip install -U redis langgraph langchain_openai
```

```
import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("OPENAI_API_KEY")
```

## Checkpointer implementation
## Define imports and helper functions

First, let's define some imports and shared utilities for both RedisSaver and AsyncRedisSaver

```
"""Implementation of a langgraph checkpoint saver using Redis."""
from contextlib import asynccontextmanager, contextmanager
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Iterator,
    List,
    Optional,
    Tuple,
)

from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    PendingWrite,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.base import SerializerProtocol
from redis import Redis
from redis.asyncio import Redis as AsyncRedis

REDIS_KEY_SEPARATOR = "$"


# Utilities shared by both RedisSaver and AsyncRedisSaver


def _make_redis_checkpoint_key(
    thread_id: str, checkpoint_ns: str, checkpoint_id: str
) -> str:
    return REDIS_KEY_SEPARATOR.join(
        ["checkpoint", thread_id, checkpoint_ns, checkpoint_id]
    )


def _make_redis_checkpoint_writes_key(
    thread_id: str,
    checkpoint_ns: str,
    checkpoint_id: str,
    task_id: str,
    idx: Optional[int],
) -> str:
    if idx is None:
        return REDIS_KEY_SEPARATOR.join(
            ["writes", thread_id, checkpoint_ns, checkpoint_id, task_id]
        )

    return REDIS_KEY_SEPARATOR.join(
        ["writes", thread_id, checkpoint_ns, checkpoint_id, task_id, str(idx)]
    )


def _parse_redis_checkpoint_key(redis_key: str) -> dict:
    namespace, thread_id, checkpoint_ns, checkpoint_id = redis_key.split(
        REDIS_KEY_SEPARATOR
    )
    if namespace != "checkpoint":
        raise ValueError("Expected checkpoint key to start with 'checkpoint'")

    return {
        "thread_id": thread_id,
        "checkpoint_ns": checkpoint_ns,
        "checkpoint_id": checkpoint_id,
    }


def _parse_redis_checkpoint_writes_key(redis_key: str) -> dict:
    namespace, thread_id, checkpoint_ns, checkpoint_id, task_id, idx = redis_key.split(
        REDIS_KEY_SEPARATOR
    )
    if namespace != "writes":
        raise ValueError("Expected checkpoint key to start with 'checkpoint'")

    return {
        "thread_id": thread_id,
        "checkpoint_ns": checkpoint_ns,
        "checkpoint_id": checkpoint_id,
        "task_id": task_id,
        "idx": idx,
    }


def _filter_keys(
    keys: List[str], before: Optional[RunnableConfig], limit: Optional[int]
) -> list:
    """Filter and sort Redis keys based on optional criteria."""
    if before:
        keys = [
            k
            for k in keys
            if _parse_redis_checkpoint_key(k.decode())["checkpoint_id"]
            < before["configurable"]["checkpoint_id"]
        ]

    keys = sorted(
        keys,
        key=lambda k: _parse_redis_checkpoint_key(k.decode())["checkpoint_id"],
        reverse=True,
    )
    if limit:
        keys = keys[:limit]
    return keys


def _load_writes(
    serde: SerializerProtocol, task_id_to_data: dict[tuple[str, str], dict]
) -> list[PendingWrite]:
    """Deserialize pending writes."""
    writes = [
        (
            task_id,
            data[b"channel"].decode(),
            serde.loads_typed((data[b"type"].decode(), data[b"value"])),
        )
        for (task_id, _), data in task_id_to_data.items()
    ]
    return writes


def _parse_redis_checkpoint_data(
    serde: SerializerProtocol,
    key: str,
    data: dict,
    pending_writes: Optional[List[PendingWrite]] = None,
) -> Optional[CheckpointTuple]:
    """Parse checkpoint data retrieved from Redis."""
    if not data:
        return None

    parsed_key = _parse_redis_checkpoint_key(key)
    thread_id = parsed_key["thread_id"]
    checkpoint_ns = parsed_key["checkpoint_ns"]
    checkpoint_id = parsed_key["checkpoint_id"]
    config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
        }
    }

    checkpoint = serde.loads_typed((data[b"type"].decode(), data[b"checkpoint"]))
    metadata = serde.loads(data[b"metadata"].decode())
    parent_checkpoint_id = data.get(b"parent_checkpoint_id", b"").decode()
    parent_config = (
        {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": parent_checkpoint_id,
            }
        }
        if parent_checkpoint_id
        else None
    )
    return CheckpointTuple(
        config=config,
        checkpoint=checkpoint,
        metadata=metadata,
        parent_config=parent_config,
        pending_writes=pending_writes,
    )
```

## RedisSaver
Below is an implementation of RedisSaver (for synchronous use of graph, i.e. .invoke(), .stream()). RedisSaver implements four methods that are required for any checkpointer:

- .put - Store a checkpoint with its configuration and metadata.
- .put_writes - Store intermediate writes linked to a checkpoint (i.e. pending writes).
- .get_tuple - Fetch a checkpoint tuple using for a given configuration (thread_id and checkpoint_id).
- .list - List checkpoints that match a given configuration and filter criteria.


```
class RedisSaver(BaseCheckpointSaver):
    """Redis-based checkpoint saver implementation."""

    conn: Redis

    def __init__(self, conn: Redis):
        super().__init__()
        self.conn = conn

    @classmethod
    @contextmanager
    def from_conn_info(cls, *, host: str, port: int, db: int) -> Iterator["RedisSaver"]:
        conn = None
        try:
            conn = Redis(host=host, port=port, db=db)
            yield RedisSaver(conn)
        finally:
            if conn:
                conn.close()

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to Redis.

        Args:
            config (RunnableConfig): The config to associate with the checkpoint.
            checkpoint (Checkpoint): The checkpoint to save.
            metadata (CheckpointMetadata): Additional metadata to save with the checkpoint.
            new_versions (ChannelVersions): New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = checkpoint["id"]
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")
        key = _make_redis_checkpoint_key(thread_id, checkpoint_ns, checkpoint_id)

        type_, serialized_checkpoint = self.serde.dumps_typed(checkpoint)
        serialized_metadata = self.serde.dumps(metadata)
        data = {
            "checkpoint": serialized_checkpoint,
            "type": type_,
            "metadata": serialized_metadata,
            "parent_checkpoint_id": parent_checkpoint_id
            if parent_checkpoint_id
            else "",
        }
        self.conn.hset(key, mapping=data)
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: List[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Store intermediate writes linked to a checkpoint.

        Args:
            config (RunnableConfig): Configuration of the related checkpoint.
            writes (Sequence[Tuple[str, Any]]): List of writes to store, each as (channel, value) pair.
            task_id (str): Identifier for the task creating the writes.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = config["configurable"]["checkpoint_id"]

        for idx, (channel, value) in enumerate(writes):
            key = _make_redis_checkpoint_writes_key(
                thread_id,
                checkpoint_ns,
                checkpoint_id,
                task_id,
                WRITES_IDX_MAP.get(channel, idx),
            )
            type_, serialized_value = self.serde.dumps_typed(value)
            data = {"channel": channel, "type": type_, "value": serialized_value}
            if all(w[0] in WRITES_IDX_MAP for w in writes):
                # Use HSET which will overwrite existing values
                self.conn.hset(key, mapping=data)
            else:
                # Use HSETNX which will not overwrite existing values
                for field, value in data.items():
                    self.conn.hsetnx(key, field, value)

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from Redis.

        This method retrieves a checkpoint tuple from Redis based on the
        provided config. If the config contains a "checkpoint_id" key, the checkpoint with
        the matching thread ID and checkpoint ID is retrieved. Otherwise, the latest checkpoint
        for the given thread ID is retrieved.

        Args:
            config (RunnableConfig): The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        checkpoint_key = self._get_checkpoint_key(
            self.conn, thread_id, checkpoint_ns, checkpoint_id
        )
        if not checkpoint_key:
            return None

        checkpoint_data = self.conn.hgetall(checkpoint_key)

        # load pending writes
        checkpoint_id = (
            checkpoint_id
            or _parse_redis_checkpoint_key(checkpoint_key)["checkpoint_id"]
        )
        pending_writes = self._load_pending_writes(
            thread_id, checkpoint_ns, checkpoint_id
        )
        return _parse_redis_checkpoint_data(
            self.serde, checkpoint_key, checkpoint_data, pending_writes=pending_writes
        )

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        # TODO: implement filtering
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from the database.

        This method retrieves a list of checkpoint tuples from Redis based
        on the provided config. The checkpoints are ordered by checkpoint ID in descending order (newest first).

        Args:
            config (RunnableConfig): The config to use for listing the checkpoints.
            filter (Optional[Dict[str, Any]]): Additional filtering criteria for metadata. Defaults to None.
            before (Optional[RunnableConfig]): If provided, only checkpoints before the specified checkpoint ID are returned. Defaults to None.
            limit (Optional[int]): The maximum number of checkpoints to return. Defaults to None.

        Yields:
            Iterator[CheckpointTuple]: An iterator of checkpoint tuples.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        pattern = _make_redis_checkpoint_key(thread_id, checkpoint_ns, "*")

        keys = _filter_keys(self.conn.keys(pattern), before, limit)
        for key in keys:
            data = self.conn.hgetall(key)
            if data and b"checkpoint" in data and b"metadata" in data:
                # load pending writes
                checkpoint_id = _parse_redis_checkpoint_key(key.decode())[
                    "checkpoint_id"
                ]
                pending_writes = self._load_pending_writes(
                    thread_id, checkpoint_ns, checkpoint_id
                )
                yield _parse_redis_checkpoint_data(
                    self.serde, key.decode(), data, pending_writes=pending_writes
                )

    def _load_pending_writes(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> List[PendingWrite]:
        writes_key = _make_redis_checkpoint_writes_key(
            thread_id, checkpoint_ns, checkpoint_id, "*", None
        )
        matching_keys = self.conn.keys(pattern=writes_key)
        parsed_keys = [
            _parse_redis_checkpoint_writes_key(key.decode()) for key in matching_keys
        ]
        pending_writes = _load_writes(
            self.serde,
            {
                (parsed_key["task_id"], parsed_key["idx"]): self.conn.hgetall(key)
                for key, parsed_key in sorted(
                    zip(matching_keys, parsed_keys), key=lambda x: x[1]["idx"]
                )
            },
        )
        return pending_writes

    def _get_checkpoint_key(
        self, conn, thread_id: str, checkpoint_ns: str, checkpoint_id: Optional[str]
    ) -> Optional[str]:
        """Determine the Redis key for a checkpoint."""
        if checkpoint_id:
            return _make_redis_checkpoint_key(thread_id, checkpoint_ns, checkpoint_id)

        all_keys = conn.keys(_make_redis_checkpoint_key(thread_id, checkpoint_ns, "*"))
        if not all_keys:
            return None

        latest_key = max(
            all_keys,
            key=lambda k: _parse_redis_checkpoint_key(k.decode())["checkpoint_id"],
        )
        return latest_key.decode()
```

## AsyncRedis
Below is a reference implementation of AsyncRedisSaver (for asynchronous use of graph, i.e. .ainvoke(), .astream()). AsyncRedisSaver implements four methods that are required for any async checkpointer:

- .aput - Store a checkpoint with its configuration and metadata.
- .aput_writes - Store intermediate writes linked to a checkpoint (i.e. pending writes).
- .aget_tuple - Fetch a checkpoint tuple using for a given configuration (thread_id and checkpoint_id).
- .alist - List checkpoints that match a given configuration and filter criteria.

```
class AsyncRedisSaver(BaseCheckpointSaver):
    """Async redis-based checkpoint saver implementation."""

    conn: AsyncRedis

    def __init__(self, conn: AsyncRedis):
        super().__init__()
        self.conn = conn

    @classmethod
    @asynccontextmanager
    async def from_conn_info(
        cls, *, host: str, port: int, db: int
    ) -> AsyncIterator["AsyncRedisSaver"]:
        conn = None
        try:
            conn = AsyncRedis(host=host, port=port, db=db)
            yield AsyncRedisSaver(conn)
        finally:
            if conn:
                await conn.aclose()

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to the database asynchronously.

        This method saves a checkpoint to Redis. The checkpoint is associated
        with the provided config and its parent config (if any).

        Args:
            config (RunnableConfig): The config to associate with the checkpoint.
            checkpoint (Checkpoint): The checkpoint to save.
            metadata (CheckpointMetadata): Additional metadata to save with the checkpoint.
            new_versions (ChannelVersions): New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = checkpoint["id"]
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")
        key = _make_redis_checkpoint_key(thread_id, checkpoint_ns, checkpoint_id)

        type_, serialized_checkpoint = self.serde.dumps_typed(checkpoint)
        serialized_metadata = self.serde.dumps(metadata)
        data = {
            "checkpoint": serialized_checkpoint,
            "type": type_,
            "checkpoint_id": checkpoint_id,
            "metadata": serialized_metadata,
            "parent_checkpoint_id": parent_checkpoint_id
            if parent_checkpoint_id
            else "",
        }

        await self.conn.hset(key, mapping=data)
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: List[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Store intermediate writes linked to a checkpoint asynchronously.

        This method saves intermediate writes associated with a checkpoint to the database.

        Args:
            config (RunnableConfig): Configuration of the related checkpoint.
            writes (Sequence[Tuple[str, Any]]): List of writes to store, each as (channel, value) pair.
            task_id (str): Identifier for the task creating the writes.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = config["configurable"]["checkpoint_id"]

        for idx, (channel, value) in enumerate(writes):
            key = _make_redis_checkpoint_writes_key(
                thread_id,
                checkpoint_ns,
                checkpoint_id,
                task_id,
                WRITES_IDX_MAP.get(channel, idx),
            )
            type_, serialized_value = self.serde.dumps_typed(value)
            data = {"channel": channel, "type": type_, "value": serialized_value}
            if all(w[0] in WRITES_IDX_MAP for w in writes):
                # Use HSET which will overwrite existing values
                await self.conn.hset(key, mapping=data)
            else:
                # Use HSETNX which will not overwrite existing values
                for field, value in data.items():
                    await self.conn.hsetnx(key, field, value)

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from Redis asynchronously.

        This method retrieves a checkpoint tuple from Redis based on the
        provided config. If the config contains a "checkpoint_id" key, the checkpoint with
        the matching thread ID and checkpoint ID is retrieved. Otherwise, the latest checkpoint
        for the given thread ID is retrieved.

        Args:
            config (RunnableConfig): The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        checkpoint_key = await self._aget_checkpoint_key(
            self.conn, thread_id, checkpoint_ns, checkpoint_id
        )
        if not checkpoint_key:
            return None
        checkpoint_data = await self.conn.hgetall(checkpoint_key)

        # load pending writes
        checkpoint_id = (
            checkpoint_id
            or _parse_redis_checkpoint_key(checkpoint_key)["checkpoint_id"]
        )
        pending_writes = await self._aload_pending_writes(
            thread_id, checkpoint_ns, checkpoint_id
        )
        return _parse_redis_checkpoint_data(
            self.serde, checkpoint_key, checkpoint_data, pending_writes=pending_writes
        )

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        # TODO: implement filtering
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncGenerator[CheckpointTuple, None]:
        """List checkpoints from Redis asynchronously.

        This method retrieves a list of checkpoint tuples from Redis based
        on the provided config. The checkpoints are ordered by checkpoint ID in descending order (newest first).

        Args:
            config (Optional[RunnableConfig]): Base configuration for filtering checkpoints.
            filter (Optional[Dict[str, Any]]): Additional filtering criteria for metadata.
            before (Optional[RunnableConfig]): If provided, only checkpoints before the specified checkpoint ID are returned. Defaults to None.
            limit (Optional[int]): Maximum number of checkpoints to return.

        Yields:
            AsyncIterator[CheckpointTuple]: An asynchronous iterator of matching checkpoint tuples.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        pattern = _make_redis_checkpoint_key(thread_id, checkpoint_ns, "*")
        keys = _filter_keys(await self.conn.keys(pattern), before, limit)
        for key in keys:
            data = await self.conn.hgetall(key)
            if data and b"checkpoint" in data and b"metadata" in data:
                checkpoint_id = _parse_redis_checkpoint_key(key.decode())[
                    "checkpoint_id"
                ]
                pending_writes = await self._aload_pending_writes(
                    thread_id, checkpoint_ns, checkpoint_id
                )
                yield _parse_redis_checkpoint_data(
                    self.serde, key.decode(), data, pending_writes=pending_writes
                )

    async def _aload_pending_writes(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> List[PendingWrite]:
        writes_key = _make_redis_checkpoint_writes_key(
            thread_id, checkpoint_ns, checkpoint_id, "*", None
        )
        matching_keys = await self.conn.keys(pattern=writes_key)
        parsed_keys = [
            _parse_redis_checkpoint_writes_key(key.decode()) for key in matching_keys
        ]
        pending_writes = _load_writes(
            self.serde,
            {
                (parsed_key["task_id"], parsed_key["idx"]): await self.conn.hgetall(key)
                for key, parsed_key in sorted(
                    zip(matching_keys, parsed_keys), key=lambda x: x[1]["idx"]
                )
            },
        )
        return pending_writes

    async def _aget_checkpoint_key(
        self, conn, thread_id: str, checkpoint_ns: str, checkpoint_id: Optional[str]
    ) -> Optional[str]:
        """Asynchronously determine the Redis key for a checkpoint."""
        if checkpoint_id:
            return _make_redis_checkpoint_key(thread_id, checkpoint_ns, checkpoint_id)

        all_keys = await conn.keys(
            _make_redis_checkpoint_key(thread_id, checkpoint_ns, "*")
        )
        if not all_keys:
            return None

        latest_key = max(
            all_keys,
            key=lambda k: _parse_redis_checkpoint_key(k.decode())["checkpoint_id"],
        )
        return latest_key.decode()
```

## Setup model and tools for the graph

```
from typing import Literal
from langchain_core.runnables import ConfigurableField
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")


tools = [get_weather]
model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
```

## Use sync connection

```
with RedisSaver.from_conn_info(host="localhost", port=6379, db=0) as checkpointer:
    graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "1"}}
    res = graph.invoke({"messages": [("human", "what's the weather in sf")]}, config)

    latest_checkpoint = checkpointer.get(config)
    latest_checkpoint_tuple = checkpointer.get_tuple(config)
    checkpoint_tuples = list(checkpointer.list(config))
```

```
latest_checkpoint
```

```
{'v': 1,
 'ts': '2024-08-09T01:56:48.328315+00:00',
 'id': '1ef55f2a-3614-69b4-8003-2181cff935cc',
 'channel_values': {'messages': [HumanMessage(content="what's the weather in sf", id='f911e000-75a1-41f6-8e38-77bb086c2ecf'),
   AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_l5e5YcTJDJYOdvi4scBy9n2I', 'function': {'arguments': '{"city":"sf"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 57, 'total_tokens': 71}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-4f1531f1-067c-4e16-8b62-7a6b663e93bd-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'sf'}, 'id': 'call_l5e5YcTJDJYOdvi4scBy9n2I', 'type': 'tool_call'}], usage_metadata={'input_tokens': 57, 'output_tokens': 14, 'total_tokens': 71}),
   ToolMessage(content="It's always sunny in sf", name='get_weather', id='e27bb3a1-1798-494a-b4ad-2deadda8b2bf', tool_call_id='call_l5e5YcTJDJYOdvi4scBy9n2I'),
   AIMessage(content='The weather in San Francisco is always sunny!', response_metadata={'token_usage': {'completion_tokens': 10, 'prompt_tokens': 84, 'total_tokens': 94}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'stop', 'logprobs': None}, id='run-ad546b5a-70ce-404e-9656-dcc6ecd482d3-0', usage_metadata={'input_tokens': 84, 'output_tokens': 10, 'total_tokens': 94})],
  'agent': 'agent'},
 'channel_versions': {'__start__': '00000000000000000000000000000002.',
  'messages': '00000000000000000000000000000005.16e98d6f7ece7598829eddf1b33a33c4',
  'start:agent': '00000000000000000000000000000003.',
  'agent': '00000000000000000000000000000005.065d90dd7f7cd091f0233855210bb2af',
  'branch:agent:should_continue:tools': '00000000000000000000000000000004.',
  'tools': '00000000000000000000000000000005.'},
 'versions_seen': {'__input__': {},
  '__start__': {'__start__': '00000000000000000000000000000001.ab89befb52cc0e91e106ef7f500ea033'},
  'agent': {'start:agent': '00000000000000000000000000000002.d6f25946c3108fc12f27abbcf9b4cedc',
   'tools': '00000000000000000000000000000004.022986cd20ae85c77ea298a383f69ba8'},
  'tools': {'branch:agent:should_continue:tools': '00000000000000000000000000000003.065d90dd7f7cd091f0233855210bb2af'}},
 'pending_sends': [],
 'current_tasks': {}}
```

```
latest_checkpoint_tuple
```

```
CheckpointTuple(config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef55f2a-3614-69b4-8003-2181cff935cc'}}, checkpoint={'v': 1, 'ts': '2024-08-09T01:56:48.328315+00:00', 'id': '1ef55f2a-3614-69b4-8003-2181cff935cc', 'channel_values': {'messages': [HumanMessage(content="what's the weather in sf", id='f911e000-75a1-41f6-8e38-77bb086c2ecf'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_l5e5YcTJDJYOdvi4scBy9n2I', 'function': {'arguments': '{"city":"sf"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 57, 'total_tokens': 71}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-4f1531f1-067c-4e16-8b62-7a6b663e93bd-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'sf'}, 'id': 'call_l5e5YcTJDJYOdvi4scBy9n2I', 'type': 'tool_call'}], usage_metadata={'input_tokens': 57, 'output_tokens': 14, 'total_tokens': 71}), ToolMessage(content="It's always sunny in sf", name='get_weather', id='e27bb3a1-1798-494a-b4ad-2deadda8b2bf', tool_call_id='call_l5e5YcTJDJYOdvi4scBy9n2I'), AIMessage(content='The weather in San Francisco is always sunny!', response_metadata={'token_usage': {'completion_tokens': 10, 'prompt_tokens': 84, 'total_tokens': 94}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'stop', 'logprobs': None}, id='run-ad546b5a-70ce-404e-9656-dcc6ecd482d3-0', usage_metadata={'input_tokens': 84, 'output_tokens': 10, 'total_tokens': 94})], 'agent': 'agent'}, 'channel_versions': {'__start__': '00000000000000000000000000000002.', 'messages': '00000000000000000000000000000005.16e98d6f7ece7598829eddf1b33a33c4', 'start:agent': '00000000000000000000000000000003.', 'agent': '00000000000000000000000000000005.065d90dd7f7cd091f0233855210bb2af', 'branch:agent:should_continue:tools': '00000000000000000000000000000004.', 'tools': '00000000000000000000000000000005.'}, 'versions_seen': {'__input__': {}, '__start__': {'__start__': '00000000000000000000000000000001.ab89befb52cc0e91e106ef7f500ea033'}, 'agent': {'start:agent': '00000000000000000000000000000002.d6f25946c3108fc12f27abbcf9b4cedc', 'tools': '00000000000000000000000000000004.022986cd20ae85c77ea298a383f69ba8'}, 'tools': {'branch:agent:should_continue:tools': '00000000000000000000000000000003.065d90dd7f7cd091f0233855210bb2af'}}, 'pending_sends': [], 'current_tasks': {}}, metadata={'source': 'loop', 'writes': {'agent': {'messages': [AIMessage(content='The weather in San Francisco is always sunny!', response_metadata={'token_usage': {'completion_tokens': 10, 'prompt_tokens': 84, 'total_tokens': 94}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'stop', 'logprobs': None}, id='run-ad546b5a-70ce-404e-9656-dcc6ecd482d3-0', usage_metadata={'input_tokens': 84, 'output_tokens': 10, 'total_tokens': 94})]}}, 'step': 3}, parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef55f2a-306f-6252-8002-47c2374ec1f2'}}, pending_writes=[])
```

## Use async connection

```
async with AsyncRedisSaver.from_conn_info(
    host="localhost", port=6379, db=0
) as checkpointer:
    graph = create_react_agent(model, tools=tools, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "2"}}
    res = await graph.ainvoke(
        {"messages": [("human", "what's the weather in nyc")]}, config
    )

    latest_checkpoint = await checkpointer.aget(config)
    latest_checkpoint_tuple = await checkpointer.aget_tuple(config)
    checkpoint_tuples = [c async for c in checkpointer.alist(config)]
```

```
latest_checkpoint
```

```
{'v': 1,
 'ts': '2024-08-09T01:56:49.503241+00:00',
 'id': '1ef55f2a-4149-61ea-8003-dc5506862287',
 'channel_values': {'messages': [HumanMessage(content="what's the weather in nyc", id='5a106e79-a617-4707-839f-134d4e4b762a'),
   AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_TvPLLyhuQQN99EcZc8SzL8x9', 'function': {'arguments': '{"city":"nyc"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 15, 'prompt_tokens': 58, 'total_tokens': 73}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-0d6fa3b4-cace-41a8-b025-d01d16f6bbe9-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'nyc'}, 'id': 'call_TvPLLyhuQQN99EcZc8SzL8x9', 'type': 'tool_call'}], usage_metadata={'input_tokens': 58, 'output_tokens': 15, 'total_tokens': 73}),
   ToolMessage(content='It might be cloudy in nyc', name='get_weather', id='922124bd-d3b0-4929-a996-a75d842b8b44', tool_call_id='call_TvPLLyhuQQN99EcZc8SzL8x9'),
   AIMessage(content='The weather in NYC might be cloudy.', response_metadata={'token_usage': {'completion_tokens': 9, 'prompt_tokens': 88, 'total_tokens': 97}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'stop', 'logprobs': None}, id='run-69a10e66-d61f-475e-b7de-a1ecd08a6c3a-0', usage_metadata={'input_tokens': 88, 'output_tokens': 9, 'total_tokens': 97})],
  'agent': 'agent'},
 'channel_versions': {'__start__': '00000000000000000000000000000002.',
  'messages': '00000000000000000000000000000005.2cb29d082da6435a7528b4c917fd0c28',
  'start:agent': '00000000000000000000000000000003.',
  'agent': '00000000000000000000000000000005.065d90dd7f7cd091f0233855210bb2af',
  'branch:agent:should_continue:tools': '00000000000000000000000000000004.',
  'tools': '00000000000000000000000000000005.'},
 'versions_seen': {'__input__': {},
  '__start__': {'__start__': '00000000000000000000000000000001.0e148ae3debe753278387e84f786e863'},
  'agent': {'start:agent': '00000000000000000000000000000002.d6f25946c3108fc12f27abbcf9b4cedc',
   'tools': '00000000000000000000000000000004.022986cd20ae85c77ea298a383f69ba8'},
  'tools': {'branch:agent:should_continue:tools': '00000000000000000000000000000003.065d90dd7f7cd091f0233855210bb2af'}},
 'pending_sends': [],
 'current_tasks': {}}
 ```

 ```
 latest_checkpoint_tuple
 ```

 ```
 CheckpointTuple(config={'configurable': {'thread_id': '2', 'checkpoint_ns': '', 'checkpoint_id': '1ef55f2a-4149-61ea-8003-dc5506862287'}}, checkpoint={'v': 1, 'ts': '2024-08-09T01:56:49.503241+00:00', 'id': '1ef55f2a-4149-61ea-8003-dc5506862287', 'channel_values': {'messages': [HumanMessage(content="what's the weather in nyc", id='5a106e79-a617-4707-839f-134d4e4b762a'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_TvPLLyhuQQN99EcZc8SzL8x9', 'function': {'arguments': '{"city":"nyc"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 15, 'prompt_tokens': 58, 'total_tokens': 73}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-0d6fa3b4-cace-41a8-b025-d01d16f6bbe9-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'nyc'}, 'id': 'call_TvPLLyhuQQN99EcZc8SzL8x9', 'type': 'tool_call'}], usage_metadata={'input_tokens': 58, 'output_tokens': 15, 'total_tokens': 73}), ToolMessage(content='It might be cloudy in nyc', name='get_weather', id='922124bd-d3b0-4929-a996-a75d842b8b44', tool_call_id='call_TvPLLyhuQQN99EcZc8SzL8x9'), AIMessage(content='The weather in NYC might be cloudy.', response_metadata={'token_usage': {'completion_tokens': 9, 'prompt_tokens': 88, 'total_tokens': 97}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'stop', 'logprobs': None}, id='run-69a10e66-d61f-475e-b7de-a1ecd08a6c3a-0', usage_metadata={'input_tokens': 88, 'output_tokens': 9, 'total_tokens': 97})], 'agent': 'agent'}, 'channel_versions': {'__start__': '00000000000000000000000000000002.', 'messages': '00000000000000000000000000000005.2cb29d082da6435a7528b4c917fd0c28', 'start:agent': '00000000000000000000000000000003.', 'agent': '00000000000000000000000000000005.065d90dd7f7cd091f0233855210bb2af', 'branch:agent:should_continue:tools': '00000000000000000000000000000004.', 'tools': '00000000000000000000000000000005.'}, 'versions_seen': {'__input__': {}, '__start__': {'__start__': '00000000000000000000000000000001.0e148ae3debe753278387e84f786e863'}, 'agent': {'start:agent': '00000000000000000000000000000002.d6f25946c3108fc12f27abbcf9b4cedc', 'tools': '00000000000000000000000000000004.022986cd20ae85c77ea298a383f69ba8'}, 'tools': {'branch:agent:should_continue:tools': '00000000000000000000000000000003.065d90dd7f7cd091f0233855210bb2af'}}, 'pending_sends': [], 'current_tasks': {}}, metadata={'source': 'loop', 'writes': {'agent': {'messages': [AIMessage(content='The weather in NYC might be cloudy.', response_metadata={'token_usage': {'completion_tokens': 9, 'prompt_tokens': 88, 'total_tokens': 97}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'stop', 'logprobs': None}, id='run-69a10e66-d61f-475e-b7de-a1ecd08a6c3a-0', usage_metadata={'input_tokens': 88, 'output_tokens': 9, 'total_tokens': 97})]}}, 'step': 3}, parent_config={'configurable': {'thread_id': '2', 'checkpoint_ns': '', 'checkpoint_id': '1ef55f2a-3d07-647e-8002-b5e4d28c00c9'}}, pending_writes=[])
 ```

 ```
 checkpoint_tuples
 ```

 ```
 [CheckpointTuple(config={'configurable': {'thread_id': '2', 'checkpoint_ns': '', 'checkpoint_id': '1ef55f2a-4149-61ea-8003-dc5506862287'}}, checkpoint={'v': 1, 'ts': '2024-08-09T01:56:49.503241+00:00', 'id': '1ef55f2a-4149-61ea-8003-dc5506862287', 'channel_values': {'messages': [HumanMessage(content="what's the weather in nyc", id='5a106e79-a617-4707-839f-134d4e4b762a'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_TvPLLyhuQQN99EcZc8SzL8x9', 'function': {'arguments': '{"city":"nyc"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 15, 'prompt_tokens': 58, 'total_tokens': 73}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-0d6fa3b4-cace-41a8-b025-d01d16f6bbe9-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'nyc'}, 'id': 'call_TvPLLyhuQQN99EcZc8SzL8x9', 'type': 'tool_call'}], usage_metadata={'input_tokens': 58, 'output_tokens': 15, 'total_tokens': 73}), ToolMessage(content='It might be cloudy in nyc', name='get_weather', id='922124bd-d3b0-4929-a996-a75d842b8b44', tool_call_id='call_TvPLLyhuQQN99EcZc8SzL8x9'), AIMessage(content='The weather in NYC might be cloudy.', response_metadata={'token_usage': {'completion_tokens': 9, 'prompt_tokens': 88, 'total_tokens': 97}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'stop', 'logprobs': None}, id='run-69a10e66-d61f-475e-b7de-a1ecd08a6c3a-0', usage_metadata={'input_tokens': 88, 'output_tokens': 9, 'total_tokens': 97})], 'agent': 'agent'}, 'channel_versions': {'__start__': '00000000000000000000000000000002.', 'messages': '00000000000000000000000000000005.2cb29d082da6435a7528b4c917fd0c28', 'start:agent': '00000000000000000000000000000003.', 'agent': '00000000000000000000000000000005.065d90dd7f7cd091f0233855210bb2af', 'branch:agent:should_continue:tools': '00000000000000000000000000000004.', 'tools': '00000000000000000000000000000005.'}, 'versions_seen': {'__input__': {}, '__start__': {'__start__': '00000000000000000000000000000001.0e148ae3debe753278387e84f786e863'}, 'agent': {'start:agent': '00000000000000000000000000000002.d6f25946c3108fc12f27abbcf9b4cedc', 'tools': '00000000000000000000000000000004.022986cd20ae85c77ea298a383f69ba8'}, 'tools': {'branch:agent:should_continue:tools': '00000000000000000000000000000003.065d90dd7f7cd091f0233855210bb2af'}}, 'pending_sends': [], 'current_tasks': {}}, metadata={'source': 'loop', 'writes': {'agent': {'messages': [AIMessage(content='The weather in NYC might be cloudy.', response_metadata={'token_usage': {'completion_tokens': 9, 'prompt_tokens': 88, 'total_tokens': 97}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'stop', 'logprobs': None}, id='run-69a10e66-d61f-475e-b7de-a1ecd08a6c3a-0', usage_metadata={'input_tokens': 88, 'output_tokens': 9, 'total_tokens': 97})]}}, 'step': 3}, parent_config={'configurable': {'thread_id': '2', 'checkpoint_ns': '', 'checkpoint_id': '1ef55f2a-3d07-647e-8002-b5e4d28c00c9'}}, pending_writes=None),
 CheckpointTuple(config={'configurable': {'thread_id': '2', 'checkpoint_ns': '', 'checkpoint_id': '1ef55f2a-3d07-647e-8002-b5e4d28c00c9'}}, checkpoint={'v': 1, 'ts': '2024-08-09T01:56:49.056860+00:00', 'id': '1ef55f2a-3d07-647e-8002-b5e4d28c00c9', 'channel_values': {'messages': [HumanMessage(content="what's the weather in nyc", id='5a106e79-a617-4707-839f-134d4e4b762a'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_TvPLLyhuQQN99EcZc8SzL8x9', 'function': {'arguments': '{"city":"nyc"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 15, 'prompt_tokens': 58, 'total_tokens': 73}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-0d6fa3b4-cace-41a8-b025-d01d16f6bbe9-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'nyc'}, 'id': 'call_TvPLLyhuQQN99EcZc8SzL8x9', 'type': 'tool_call'}], usage_metadata={'input_tokens': 58, 'output_tokens': 15, 'total_tokens': 73}), ToolMessage(content='It might be cloudy in nyc', name='get_weather', id='922124bd-d3b0-4929-a996-a75d842b8b44', tool_call_id='call_TvPLLyhuQQN99EcZc8SzL8x9')], 'tools': 'tools'}, 'channel_versions': {'__start__': '00000000000000000000000000000002.', 'messages': '00000000000000000000000000000004.07964a3a545f9ff95545db45a9753d11', 'start:agent': '00000000000000000000000000000003.', 'agent': '00000000000000000000000000000004.', 'branch:agent:should_continue:tools': '00000000000000000000000000000004.', 'tools': '00000000000000000000000000000004.022986cd20ae85c77ea298a383f69ba8'}, 'versions_seen': {'__input__': {}, '__start__': {'__start__': '00000000000000000000000000000001.0e148ae3debe753278387e84f786e863'}, 'agent': {'start:agent': '00000000000000000000000000000002.d6f25946c3108fc12f27abbcf9b4cedc'}, 'tools': {'branch:agent:should_continue:tools': '00000000000000000000000000000003.065d90dd7f7cd091f0233855210bb2af'}}, 'pending_sends': [], 'current_tasks': {}}, metadata={'source': 'loop', 'writes': {'tools': {'messages': [ToolMessage(content='It might be cloudy in nyc', name='get_weather', id='922124bd-d3b0-4929-a996-a75d842b8b44', tool_call_id='call_TvPLLyhuQQN99EcZc8SzL8x9')]}}, 'step': 2}, parent_config={'configurable': {'thread_id': '2', 'checkpoint_ns': '', 'checkpoint_id': '1ef55f2a-3cf9-6996-8001-88dab066840d'}}, pending_writes=None),
 CheckpointTuple(config={'configurable': {'thread_id': '2', 'checkpoint_ns': '', 'checkpoint_id': '1ef55f2a-3cf9-6996-8001-88dab066840d'}}, checkpoint={'v': 1, 'ts': '2024-08-09T01:56:49.051234+00:00', 'id': '1ef55f2a-3cf9-6996-8001-88dab066840d', 'channel_values': {'messages': [HumanMessage(content="what's the weather in nyc", id='5a106e79-a617-4707-839f-134d4e4b762a'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_TvPLLyhuQQN99EcZc8SzL8x9', 'function': {'arguments': '{"city":"nyc"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 15, 'prompt_tokens': 58, 'total_tokens': 73}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-0d6fa3b4-cace-41a8-b025-d01d16f6bbe9-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'nyc'}, 'id': 'call_TvPLLyhuQQN99EcZc8SzL8x9', 'type': 'tool_call'}], usage_metadata={'input_tokens': 58, 'output_tokens': 15, 'total_tokens': 73})], 'agent': 'agent', 'branch:agent:should_continue:tools': 'agent'}, 'channel_versions': {'__start__': '00000000000000000000000000000002.', 'messages': '00000000000000000000000000000003.cc96d93b1afbd1b69d53851320670b97', 'start:agent': '00000000000000000000000000000003.', 'agent': '00000000000000000000000000000003.065d90dd7f7cd091f0233855210bb2af', 'branch:agent:should_continue:tools': '00000000000000000000000000000003.065d90dd7f7cd091f0233855210bb2af'}, 'versions_seen': {'__input__': {}, '__start__': {'__start__': '00000000000000000000000000000001.0e148ae3debe753278387e84f786e863'}, 'agent': {'start:agent': '00000000000000000000000000000002.d6f25946c3108fc12f27abbcf9b4cedc'}}, 'pending_sends': [], 'current_tasks': {}}, metadata={'source': 'loop', 'writes': {'agent': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_TvPLLyhuQQN99EcZc8SzL8x9', 'function': {'arguments': '{"city":"nyc"}', 'name': 'get_weather'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 15, 'prompt_tokens': 58, 'total_tokens': 73}, 'model_name': 'gpt-4o-mini', 'system_fingerprint': 'fp_48196bc67a', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-0d6fa3b4-cace-41a8-b025-d01d16f6bbe9-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'nyc'}, 'id': 'call_TvPLLyhuQQN99EcZc8SzL8x9', 'type': 'tool_call'}], usage_metadata={'input_tokens': 58, 'output_tokens': 15, 'total_tokens': 73})]}}, 'step': 1}, parent_config={'configurable': {'thread_id': '2', 'checkpoint_ns': '', 'checkpoint_id': '1ef55f2a-36a6-6788-8000-9efe1769f8c1'}}, pending_writes=None),
 CheckpointTuple(config={'configurable': {'thread_id': '2', 'checkpoint_ns': '', 'checkpoint_id': '1ef55f2a-36a6-6788-8000-9efe1769f8c1'}}, checkpoint={'v': 1, 'ts': '2024-08-09T01:56:48.388067+00:00', 'id': '1ef55f2a-36a6-6788-8000-9efe1769f8c1', 'channel_values': {'messages': [HumanMessage(content="what's the weather in nyc", id='5a106e79-a617-4707-839f-134d4e4b762a')], 'start:agent': '__start__'}, 'channel_versions': {'__start__': '00000000000000000000000000000002.', 'messages': '00000000000000000000000000000002.a6994b785a651d88df51020401745af8', 'start:agent': '00000000000000000000000000000002.d6f25946c3108fc12f27abbcf9b4cedc'}, 'versions_seen': {'__input__': {}, '__start__': {'__start__': '00000000000000000000000000000001.0e148ae3debe753278387e84f786e863'}}, 'pending_sends': [], 'current_tasks': {}}, metadata={'source': 'loop', 'writes': None, 'step': 0}, parent_config={'configurable': {'thread_id': '2', 'checkpoint_ns': '', 'checkpoint_id': '1ef55f2a-36a3-6614-bfff-05dafa02b4d7'}}, pending_writes=None),
 CheckpointTuple(config={'configurable': {'thread_id': '2', 'checkpoint_ns': '', 'checkpoint_id': '1ef55f2a-36a3-6614-bfff-05dafa02b4d7'}}, checkpoint={'v': 1, 'ts': '2024-08-09T01:56:48.386807+00:00', 'id': '1ef55f2a-36a3-6614-bfff-05dafa02b4d7', 'channel_values': {'messages': [], '__start__': {'messages': [['human', "what's the weather in nyc"]]}}, 'channel_versions': {'__start__': '00000000000000000000000000000001.0e148ae3debe753278387e84f786e863'}, 'versions_seen': {'__input__': {}}, 'pending_sends': [], 'current_tasks': {}}, metadata={'source': 'input', 'writes': {'messages': [['human', "what's the weather in nyc"]]}, 'step': -1}, parent_config=None, pending_writes=None)]
 ```

 # How to add thread-level persistence (functional API)
 Many AI applications need memory to share context across multiple interactions on the same thread (e.g., multiple turns of a conversation). In LangGraph functional API, this kind of memory can be added to any entrypoint() workflow using thread-level persistence.

When creating a LangGraph workflow, you can set it up to persist its results by using a checkpointer:

1. Create an instance of a checkpointer:

```
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
```

2. Pass checkpointer instance to the entrypoint() decorator:

```
from langgraph.func import entrypoint

@entrypoint(checkpointer=checkpointer)
def workflow(inputs)
    ...
```

3. Optionally expose previous parameter in the workflow function signature:

```
@entrypoint(checkpointer=checkpointer)
def workflow(
    inputs,
    *,
    # you can optionally specify `previous` in the workflow function signature
    # to access the return value from the workflow as of the last execution
    previous
):
    previous = previous or []
    combined_inputs = previous + inputs
    result = do_something(combined_inputs)
    ...
```

4. Optionally choose which values will be returned from the workflow and which will be saved by the checkpointer as previous:

```
@entrypoint(checkpointer=checkpointer)
def workflow(inputs, *, previous):
    ...
    result = do_something(...)
    return entrypoint.final(value=result, save=combine(inputs, result))
```

## Setup

First we need to install the packages required

```
%%capture --no-stderr
%pip install --quiet -U langgraph langchain_anthropic
```

Next, we need to set API key for Anthropic (the LLM we will use).

```
import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("ANTHROPIC_API_KEY")
```

## Example: simple chatbot with short-term memory
We will be using a workflow with a single task that calls a chat model.

Let's first define the model we'll be using:

```
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(model="claude-3-5-sonnet-latest")
```

API Reference: ChatAnthropic

Now we can define our task and workflow. To add in persistence, we need to pass in a Checkpointer to the entrypoint() decorator.

```
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import MemorySaver


@task
def call_model(messages: list[BaseMessage]):
    response = model.invoke(messages)
    return response


checkpointer = MemorySaver()


@entrypoint(checkpointer=checkpointer)
def workflow(inputs: list[BaseMessage], *, previous: list[BaseMessage]):
    if previous:
        inputs = add_messages(previous, inputs)

    response = call_model(inputs).result()
    return entrypoint.final(value=response, save=add_messages(inputs, response))
```

If we try to use this workflow, the context of the conversation will be persisted across interactions:

We can now interact with the agent and see that it remembers previous messages!

```
config = {"configurable": {"thread_id": "1"}}
input_message = {"role": "user", "content": "hi! I'm bob"}
for chunk in workflow.stream([input_message], config, stream_mode="values"):
    chunk.pretty_print()
```

```
==================================[1m Ai Message [0m==================================

Hi Bob! I'm Claude. Nice to meet you! How are you today?
```

You can always resume previous threads:

```
input_message = {"role": "user", "content": "what's my name?"}
for chunk in workflow.stream([input_message], config, stream_mode="values"):
    chunk.pretty_print()
```

```
==================================[1m Ai Message [0m==================================

Your name is Bob.
```

If we want to start a new conversation, we can pass in a different thread_id. Poof! All the memories are gone!

```
input_message = {"role": "user", "content": "what's my name?"}
for chunk in workflow.stream(
    [input_message],
    {"configurable": {"thread_id": "2"}},
    stream_mode="values",
):
    chunk.pretty_print()
```

```
==================================[1m Ai Message [0m==================================

I don't know your name unless you tell me. Each conversation I have starts fresh, so I don't have access to any previous interactions or personal information unless you share it with me.
```

## How to add cross-thread persistence (functional API)
LangGraph allows you to persist data across different threads. For instance, you can store information about users (their names or preferences) in a shared (cross-thread) memory and reuse them in the new threads (e.g., new conversations).

When using the functional API, you can set it up to store and retrieve memories by using the Store interface:

1. Create an instance of a Store

```
from langgraph.store.memory import InMemoryStore, BaseStore

store = InMemoryStore()
```

2. Pass the store instance to the entrypoint() decorator and expose store parameter in the function signature:

```
from langgraph.func import entrypoint

@entrypoint(store=store)
def workflow(inputs: dict, store: BaseStore):
    my_task(inputs).result()
    ...
```

## Setup
First, let's install the required packages and set our API keys

```
%%capture --no-stderr
%pip install -U langchain_anthropic langchain_openai langgraph
```

```
import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("ANTHROPIC_API_KEY")
_set_env("OPENAI_API_KEY")
```

## Example: simple chatbot with long-term memory
## Define store

In this example we will create a workflow that will be able to retrieve information about a user's preferences. We will do so by defining an InMemoryStore - an object that can store data in memory and query that data.

When storing objects using the Store interface you define two things:

- the namespace for the object, a tuple (similar to directories)
- the object key (similar to filenames)

In our example, we'll be using ("memories", <user_id>) as namespace and random UUID as key for each new memory.

Importantly, to determine the user, we will be passing user_id via the config keyword argument of the node function.

Let's first define our store!


```
from langgraph.store.memory import InMemoryStore
from langchain_openai import OpenAIEmbeddings

in_memory_store = InMemoryStore(
    index={
        "embed": OpenAIEmbeddings(model="text-embedding-3-small"),
        "dims": 1536,
    }
)
```

## Create workflow

```
import uuid

from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage
from langgraph.func import entrypoint, task
from langgraph.graph import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore


model = ChatAnthropic(model="claude-3-5-sonnet-latest")


@task
def call_model(messages: list[BaseMessage], memory_store: BaseStore, user_id: str):
    namespace = ("memories", user_id)
    last_message = messages[-1]
    memories = memory_store.search(namespace, query=str(last_message.content))
    info = "\n".join([d.value["data"] for d in memories])
    system_msg = f"You are a helpful assistant talking to the user. User info: {info}"

    # Store new memories if the user asks the model to remember
    if "remember" in last_message.content.lower():
        memory = "User name is Bob"
        memory_store.put(namespace, str(uuid.uuid4()), {"data": memory})

    response = model.invoke([{"role": "system", "content": system_msg}] + messages)
    return response


# NOTE: we're passing the store object here when creating a workflow via entrypoint()
@entrypoint(checkpointer=MemorySaver(), store=in_memory_store)
def workflow(
    inputs: list[BaseMessage],
    *,
    previous: list[BaseMessage],
    config: RunnableConfig,
    store: BaseStore,
):
    user_id = config["configurable"]["user_id"]
    previous = previous or []
    inputs = add_messages(previous, inputs)
    response = call_model(inputs, store, user_id).result()
    return entrypoint.final(value=response, save=add_messages(inputs, response))
```

## Run the workflow!
Now let's specify a user ID in the config and tell the model our name:

```
config = {"configurable": {"thread_id": "1", "user_id": "1"}}
input_message = {"role": "user", "content": "Hi! Remember: my name is Bob"}
for chunk in workflow.stream([input_message], config, stream_mode="values"):
    chunk.pretty_print()
```

```
==================================[1m Ai Message [0m==================================

Hello Bob! Nice to meet you. I'll remember that your name is Bob. How can I help you today?
```

```
config = {"configurable": {"thread_id": "2", "user_id": "1"}}
input_message = {"role": "user", "content": "what is my name?"}
for chunk in workflow.stream([input_message], config, stream_mode="values"):
    chunk.pretty_print()
```

We can now inspect our in-memory store and verify that we have in fact saved the memories for the user:

```
for memory in in_memory_store.search(("memories", "1")):
    print(memory.value)
```

```
{'data': 'User name is Bob'}
```

Let's now run the workflow for another user to verify that the memories about the first user are self contained:

```
config = {"configurable": {"thread_id": "3", "user_id": "2"}}
input_message = {"role": "user", "content": "what is my name?"}
for chunk in workflow.stream([input_message], config, stream_mode="values"):
    chunk.pretty_print()
```

```
==================================[1m Ai Message [0m==================================

I don't have any information about your name. I can only see our current conversation without any prior context or personal details about you. If you'd like me to know your name, feel free to tell me!
```


# How to manage conversation history
One of the most common use cases for persistence is to use it to keep track of conversation history. This is great - it makes it easy to continue conversations. As conversations get longer and longer, however, this conversation history can build up and take up more and more of the context window. This can often be undesirable as it leads to more expensive and longer calls to the LLM, and potentially ones that error. In order to prevent this from happening, you need to properly manage the conversation history.

Note: this guide focuses on how to do this in LangGraph, where you can fully customize how this is done. If you want a more off-the-shelf solution, you can look into functionality provided in LangChain:


- How to filter messages
- How to trim messages

## Build the agent
Let's now build a simple ReAct style agent.


```
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode

memory = MemorySaver()


@tool
def search(query: str):
    """Call to surf the web."""
    # This is a placeholder for the actual implementation
    # Don't let the LLM know this though 😊
    return "It's sunny in San Francisco, but you better look out if you're a Gemini 😈."


tools = [search]
tool_node = ToolNode(tools)
model = ChatAnthropic(model_name="claude-3-haiku-20240307")
bound_model = model.bind_tools(tools)


def should_continue(state: MessagesState):
    """Return the next node to execute."""
    last_message = state["messages"][-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return END
    # Otherwise if there is, we continue
    return "action"


# Define the function that calls the model
def call_model(state: MessagesState):
    response = bound_model.invoke(state["messages"])
    # We return a list, because this will get added to the existing list
    return {"messages": response}


# Define a new graph
workflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.add_edge(START, "agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Next, we pass in the path map - all the possible nodes this edge could go to
    ["action", END],
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("action", "agent")

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile(checkpointer=memory)
```


```
from langchain_core.messages import HumanMessage

config = {"configurable": {"thread_id": "2"}}
input_message = HumanMessage(content="hi! I'm bob")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()


input_message = HumanMessage(content="what's my name?")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()
```

API Reference: HumanMessage

```
================================[1m Human Message [0m=================================

hi! I'm bob
==================================[1m Ai Message [0m==================================

Nice to meet you, Bob! As an AI assistant, I don't have a physical form, but I'm happy to chat with you and try my best to help out however I can. Please feel free to ask me anything, and I'll do my best to provide useful information or assistance.
================================[1m Human Message [0m=================================

what's my name?
==================================[1m Ai Message [0m==================================

You said your name is Bob, so that is the name I have for you.
```

## Filtering messages
The most straight-forward thing to do to prevent conversation history from blowing up is to filter the list of messages before they get passed to the LLM. This involves two parts: defining a function to filter messages, and then adding it to the graph. See the example below which defines a really simple filter_messages function and then uses it.

```
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import ToolNode

memory = MemorySaver()


@tool
def search(query: str):
    """Call to surf the web."""
    # This is a placeholder for the actual implementation
    # Don't let the LLM know this though 😊
    return "It's sunny in San Francisco, but you better look out if you're a Gemini 😈."


tools = [search]
tool_node = ToolNode(tools)
model = ChatAnthropic(model_name="claude-3-haiku-20240307")
bound_model = model.bind_tools(tools)


def should_continue(state: MessagesState):
    """Return the next node to execute."""
    last_message = state["messages"][-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return END
    # Otherwise if there is, we continue
    return "action"


def filter_messages(messages: list):
    # This is very simple helper function which only ever uses the last message
    return messages[-1:]


# Define the function that calls the model
def call_model(state: MessagesState):
    messages = filter_messages(state["messages"])
    response = bound_model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": response}


# Define a new graph
workflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.add_edge(START, "agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Next, we pass in the pathmap - all the possible nodes this edge could go to
    ["action", END],
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("action", "agent")

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile(checkpointer=memory)
```

```
from langchain_core.messages import HumanMessage

config = {"configurable": {"thread_id": "2"}}
input_message = HumanMessage(content="hi! I'm bob")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()

# This will now not remember the previous messages
# (because we set `messages[-1:]` in the filter messages argument)
input_message = HumanMessage(content="what's my name?")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()
```

API Reference: HumanMessage

```
================================[1m Human Message [0m=================================

hi! I'm bob
==================================[1m Ai Message [0m==================================

Nice to meet you, Bob! I'm Claude, an AI assistant created by Anthropic. It's a pleasure to chat with you. Feel free to ask me anything, I'm here to help!
================================[1m Human Message [0m=================================

what's my name?
==================================[1m Ai Message [0m==================================

I'm afraid I don't actually know your name. As an AI assistant, I don't have information about the specific identities of the people I talk to. I only know what is provided to me during our conversation.
```

# How to delete messages

One of the common states for a graph is a list of messages. Usually you only add messages to that state. However, sometimes you may want to remove messages (either by directly modifying the state or as part of the graph). To do that, you can use the RemoveMessage modifier. In this guide, we will cover how to do that.

The key idea is that each state key has a reducer key. This key specifies how to combine updates to the state. The default MessagesState has a messages key, and the reducer for that key accepts these RemoveMessage modifiers. That reducer then uses these RemoveMessage to delete messages from the key.

So note that just because your graph state has a key that is a list of messages, it doesn't mean that that this RemoveMessage modifier will work. You also have to have a reducer defined that knows how to work with this.

NOTE: Many models expect certain rules around lists of messages. For example, some expect them to start with a user message, others expect all messages with tool calls to be followed by a tool message. When deleting messages, you will want to make sure you don't violate these rules.


## Build the agent
Let's now build a simple ReAct style agent.

from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode

memory = MemorySaver()


@tool
def search(query: str):
    """Call to surf the web."""
    # This is a placeholder for the actual implementation
    # Don't let the LLM know this though 😊
    return "It's sunny in San Francisco, but you better look out if you're a Gemini 😈."


tools = [search]
tool_node = ToolNode(tools)
model = ChatAnthropic(model_name="claude-3-haiku-20240307")
bound_model = model.bind_tools(tools)


def should_continue(state: MessagesState):
    """Return the next node to execute."""
    last_message = state["messages"][-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return END
    # Otherwise if there is, we continue
    return "action"


# Define the function that calls the model
def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    # We return a list, because this will get added to the existing list
    return {"messages": response}


# Define a new graph
workflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.add_edge(START, "agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Next, we pass in the path map - all the possible nodes this edge could go to
    ["action", END],
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("action", "agent")

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile(checkpointer=memory)

```
from langchain_core.messages import HumanMessage

config = {"configurable": {"thread_id": "2"}}
input_message = HumanMessage(content="hi! I'm bob")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()


input_message = HumanMessage(content="what's my name?")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()
```

API Reference: HumanMessage

```
================================[1m Human Message [0m=================================

hi! I'm bob
==================================[1m Ai Message [0m==================================

It's nice to meet you, Bob! I'm an AI assistant created by Anthropic. I'm here to help out with any questions or tasks you might have. Please let me know if there's anything I can assist you with.
================================[1m Human Message [0m=================================

what's my name?
==================================[1m Ai Message [0m==================================

You said your name is Bob.
```

## Manually deleting messages
First, we will cover how to manually delete messages. Let's take a look at the current state of the thread:

```
messages = app.get_state(config).values["messages"]
messages
```

```
[HumanMessage(content="hi! I'm bob", additional_kwargs={}, response_metadata={}, id='db576005-3a60-4b3b-8925-dc602ac1c571'),
 AIMessage(content="It's nice to meet you, Bob! I'm an AI assistant created by Anthropic. I'm here to help out with any questions or tasks you might have. Please let me know if there's anything I can assist you with.", additional_kwargs={}, response_metadata={'id': 'msg_01BKAnYxmoC6bQ9PpCuHk8ZT', 'model': 'claude-3-haiku-20240307', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 12, 'output_tokens': 52}}, id='run-3a60c536-b207-4c56-98f3-03f94d49a9e4-0', usage_metadata={'input_tokens': 12, 'output_tokens': 52, 'total_tokens': 64}),
 HumanMessage(content="what's my name?", additional_kwargs={}, response_metadata={}, id='2088c465-400b-430b-ad80-fad47dc1f2d6'),
 AIMessage(content='You said your name is Bob.', additional_kwargs={}, response_metadata={'id': 'msg_013UWTLTzwZi81vke8mMQ2KP', 'model': 'claude-3-haiku-20240307', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 72, 'output_tokens': 10}}, id='run-3a6883be-0c52-4938-af98-e9e7476659eb-0', usage_metadata={'input_tokens': 72, 'output_tokens': 10, 'total_tokens': 82})]
 ```

 We can call update_state and pass in the id of the first message. This will delete that message.

 ```
 from langchain_core.messages import RemoveMessage

app.update_state(config, {"messages": RemoveMessage(id=messages[0].id)})
```

API Reference: RemoveMessage

```
{'configurable': {'thread_id': '2',
  'checkpoint_ns': '',
  'checkpoint_id': '1ef75157-f251-6a2a-8005-82a86a6593a0'}}
```

If we now look at the messages, we can verify that the first one was deleted.

```
messages = app.get_state(config).values["messages"]
messages
```

```
[AIMessage(content="It's nice to meet you, Bob! I'm Claude, an AI assistant created by Anthropic. How can I assist you today?", response_metadata={'id': 'msg_01XPSAenmSqK8rX2WgPZHfz7', 'model': 'claude-3-haiku-20240307', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 12, 'output_tokens': 32}}, id='run-1c69af09-adb1-412d-9010-2456e5a555fb-0', usage_metadata={'input_tokens': 12, 'output_tokens': 32, 'total_tokens': 44}),
 HumanMessage(content="what's my name?", id='f3c71afe-8ce2-4ed0-991e-65021f03b0a5'),
 AIMessage(content='Your name is Bob, as you introduced yourself at the beginning of our conversation.', response_metadata={'id': 'msg_01BPZdwsjuMAbC1YAkqawXaF', 'model': 'claude-3-haiku-20240307', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 52, 'output_tokens': 19}}, id='run-b2eb9137-2f4e-446f-95f5-3d5f621a2cf8-0', usage_metadata={'input_tokens': 52, 'output_tokens': 19, 'total_tokens': 71})]
 ```

 ## Programmatically deleting messages

 We can also delete messages programmatically from inside the graph. Here we'll modify the graph to delete any old messages (longer than 3 messages ago) at the end of a graph run.

 ```
 from langchain_core.messages import RemoveMessage
from langgraph.graph import END


def delete_messages(state):
    messages = state["messages"]
    if len(messages) > 3:
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:-3]]}


# We need to modify the logic to call delete_messages rather than end right away
def should_continue(state: MessagesState) -> Literal["action", "delete_messages"]:
    """Return the next node to execute."""
    last_message = state["messages"][-1]
    # If there is no function call, then we call our delete_messages function
    if not last_message.tool_calls:
        return "delete_messages"
    # Otherwise if there is, we continue
    return "action"


# Define a new graph
workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

# This is our new node we're defining
workflow.add_node(delete_messages)


workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
)
workflow.add_edge("action", "agent")

# This is the new edge we're adding: after we delete messages, we finish
workflow.add_edge("delete_messages", END)
app = workflow.compile(checkpointer=memory)
```


We can now try this out. We can call the graph twice and then check the state


```
from langchain_core.messages import HumanMessage

config = {"configurable": {"thread_id": "3"}}
input_message = HumanMessage(content="hi! I'm bob")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    print([(message.type, message.content) for message in event["messages"]])


input_message = HumanMessage(content="what's my name?")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    print([(message.type, message.content) for message in event["messages"]])
```

API Reference: HumanMessage

```
[('human', "hi! I'm bob")]
[('human', "hi! I'm bob"), ('ai', "Hello Bob! It's nice to meet you. I'm an AI assistant created by Anthropic. I'm here to help with any questions or tasks you might have. Please let me know how I can assist you.")]
[('human', "hi! I'm bob"), ('ai', "Hello Bob! It's nice to meet you. I'm an AI assistant created by Anthropic. I'm here to help with any questions or tasks you might have. Please let me know how I can assist you."), ('human', "what's my name?")]
[('human', "hi! I'm bob"), ('ai', "Hello Bob! It's nice to meet you. I'm an AI assistant created by Anthropic. I'm here to help with any questions or tasks you might have. Please let me know how I can assist you."), ('human', "what's my name?"), ('ai', 'You said your name is Bob, so that is the name I have for you.')]
[('ai', "Hello Bob! It's nice to meet you. I'm an AI assistant created by Anthropic. I'm here to help with any questions or tasks you might have. Please let me know how I can assist you."), ('human', "what's my name?"), ('ai', 'You said your name is Bob, so that is the name I have for you.')]
```


If we now check the state, we should see that it is only three messages long. This is because we just deleted the earlier messages - otherwise it would be four!

```
messages = app.get_state(config).values["messages"]
messages
```

```
[AIMessage(content="Hello Bob! It's nice to meet you. I'm an AI assistant created by Anthropic. I'm here to help with any questions or tasks you might have. Please let me know how I can assist you.", response_metadata={'id': 'msg_01XPEgPPbcnz5BbGWUDWTmzG', 'model': 'claude-3-haiku-20240307', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 12, 'output_tokens': 48}}, id='run-eded3820-b6a9-4d66-9210-03ca41787ce6-0', usage_metadata={'input_tokens': 12, 'output_tokens': 48, 'total_tokens': 60}),
 HumanMessage(content="what's my name?", id='a0ea2097-3280-402b-92e1-67177b807ae8'),
 AIMessage(content='You said your name is Bob, so that is the name I have for you.', response_metadata={'id': 'msg_01JGT62pxhrhN4SykZ57CSjW', 'model': 'claude-3-haiku-20240307', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 68, 'output_tokens': 20}}, id='run-ace3519c-81f8-45fe-a777-91f42d48b3a3-0', usage_metadata={'input_tokens': 68, 'output_tokens': 20, 'total_tokens': 88})]
 ```

 Remember, when deleting messages you will want to make sure that the remaining message list is still valid. This message list may actually not be - this is because it currently starts with an AI message, which some models do not allow.

 # How to add summary of the conversation history
 One of the most common use cases for persistence is to use it to keep track of conversation history. This is great - it makes it easy to continue conversations. As conversations get longer and longer, however, this conversation history can build up and take up more and more of the context window. This can often be undesirable as it leads to more expensive and longer calls to the LLM, and potentially ones that error. One way to work around that is to create a summary of the conversation to date, and use that with the past N messages. This guide will go through an example of how to do that.

 This will involve a few steps:


- Check if the conversation is too long (can be done by checking number of messages or length of messages)
- If yes, the create summary (will need a prompt for this)
- Then remove all except the last N messages

## Build the chatbot

```
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, RemoveMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START, END

memory = MemorySaver()


# We will add a `summary` attribute (in addition to `messages` key,
# which MessagesState already has)
class State(MessagesState):
    summary: str


# We will use this model for both the conversation and the summarization
model = ChatAnthropic(model_name="claude-3-haiku-20240307")


# Define the logic to call the model
def call_model(state: State):
    # If a summary exists, we add this in as a system message
    summary = state.get("summary", "")
    if summary:
        system_message = f"Summary of conversation earlier: {summary}"
        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        messages = state["messages"]
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# We now define the logic for determining whether to end or summarize the conversation
def should_continue(state: State) -> Literal["summarize_conversation", END]:
    """Return the next node to execute."""
    messages = state["messages"]
    # If there are more than six messages, then we summarize the conversation
    if len(messages) > 6:
        return "summarize_conversation"
    # Otherwise we can just end
    return END


def summarize_conversation(state: State):
    # First, we summarize the conversation
    summary = state.get("summary", "")
    if summary:
        # If a summary already exists, we use a different system prompt
        # to summarize it than if one didn't
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)
    # We now need to delete messages that we no longer want to show up
    # I will delete all but the last two messages, but you can change this
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}


# Define a new graph
workflow = StateGraph(State)

# Define the conversation node and the summarize node
workflow.add_node("conversation", call_model)
workflow.add_node(summarize_conversation)

# Set the entrypoint as conversation
workflow.add_edge(START, "conversation")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `conversation`.
    # This means these are the edges taken after the `conversation` node is called.
    "conversation",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)

# We now add a normal edge from `summarize_conversation` to END.
# This means that after `summarize_conversation` is called, we end.
workflow.add_edge("summarize_conversation", END)

# Finally, we compile it!
app = workflow.compile(checkpointer=memory)
```

## Using the graph
```
def print_update(update):
    for k, v in update.items():
        for m in v["messages"]:
            m.pretty_print()
        if "summary" in v:
            print(v["summary"])
```


```
from langchain_core.messages import HumanMessage

config = {"configurable": {"thread_id": "4"}}
input_message = HumanMessage(content="hi! I'm bob")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)

input_message = HumanMessage(content="what's my name?")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)

input_message = HumanMessage(content="i like the celtics!")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)
```


```
================================[1m Human Message [0m=================================

hi! I'm bob
==================================[1m Ai Message [0m==================================

It's nice to meet you, Bob! I'm an AI assistant created by Anthropic. How can I help you today?
================================[1m Human Message [0m=================================

what's my name?
==================================[1m Ai Message [0m==================================

Your name is Bob, as you told me at the beginning of our conversation.
================================[1m Human Message [0m=================================

i like the celtics!
==================================[1m Ai Message [0m==================================

That's great, the Celtics are a fun team to follow! Basketball is an exciting sport. Do you have a favorite Celtics player or a favorite moment from a Celtics game you've watched? I'd be happy to discuss the team and the sport with you.
```

We can see that so far no summarization has happened - this is because there are only six messages in the list.

```
values = app.get_state(config).values
values
```

Now let's send another message in

```
input_message = HumanMessage(content="i like how much they win")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)
```

```
================================[1m Human Message [0m=================================

i like how much they win
==================================[1m Ai Message [0m==================================

That's understandable, the Celtics have been one of the more successful NBA franchises over the years. Their history of winning championships is very impressive. It's always fun to follow a team that regularly competes for titles. What do you think has been the key to the Celtics' sustained success? Is there a particular era or team that stands out as your favorite?
================================[1m Remove Message [0m================================


================================[1m Remove Message [0m================================


================================[1m Remove Message [0m================================


================================[1m Remove Message [0m================================


================================[1m Remove Message [0m================================


================================[1m Remove Message [0m================================


Here is a summary of our conversation so far:

- You introduced yourself as Bob and said you like the Boston Celtics basketball team.
- I acknowledged that it's nice to meet you, Bob, and noted that you had shared your name earlier in the conversation.
- You expressed that you like how much the Celtics win, and I agreed that their history of sustained success and championship pedigree is impressive.
- I asked if you have a favorite Celtics player or moment that stands out to you, and invited further discussion about the team and the sport of basketball.
- The overall tone has been friendly and conversational, with me trying to engage with your interest in the Celtics by asking follow-up questions.
```

If we check the state now, we can see that we have a summary of the conversation, as well as the last two messages

```
values = app.get_state(config).values
values
```

```
{'messages': [HumanMessage(content='i like how much they win', id='bb916ce7-534c-4d48-9f92-e269f9dc4859'),
  AIMessage(content="That's understandable, the Celtics have been one of the more successful NBA franchises over the years. Their history of winning championships is very impressive. It's always fun to follow a team that regularly competes for titles. What do you think has been the key to the Celtics' sustained success? Is there a particular era or team that stands out as your favorite?", response_metadata={'id': 'msg_01B7TMagaM8xBnYXLSMwUDAG', 'model': 'claude-3-haiku-20240307', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 148, 'output_tokens': 82}}, id='run-c5aa9a8f-7983-4a7f-9c1e-0c0055334ac1-0')],
 'summary': "Here is a summary of our conversation so far:\n\n- You introduced yourself as Bob and said you like the Boston Celtics basketball team.\n- I acknowledged that it's nice to meet you, Bob, and noted that you had shared your name earlier in the conversation.\n- You expressed that you like how much the Celtics win, and I agreed that their history of sustained success and championship pedigree is impressive.\n- I asked if you have a favorite Celtics player or moment that stands out to you, and invited further discussion about the team and the sport of basketball.\n- The overall tone has been friendly and conversational, with me trying to engage with your interest in the Celtics by asking follow-up questions."}
 ```

 We can now resume having a conversation! Note that even though we only have the last two messages, we can still ask it questions about things mentioned earlier in the conversation (because we summarized those)

 ```
 input_message = HumanMessage(content="what's my name?")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)
```

```
================================[1m Human Message [0m=================================

what's my name?
==================================[1m Ai Message [0m==================================

In our conversation so far, you introduced yourself as Bob. I acknowledged that earlier when you had shared your name.
```

```
input_message = HumanMessage(content="what NFL team do you think I like?")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)
```

```
================================[1m Human Message [0m=================================

what NFL team do you think I like?
==================================[1m Ai Message [0m==================================

I don't actually have any information about what NFL team you might like. In our conversation so far, you've only mentioned that you're a fan of the Boston Celtics basketball team. I don't have any prior knowledge about your preferences for NFL teams. Unless you provide me with that information, I don't have a basis to guess which NFL team you might be a fan of.
```

```
input_message = HumanMessage(content="i like the patriots!")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_update(event)
```

```
================================[1m Human Message [0m=================================

i like the patriots!
==================================[1m Ai Message [0m==================================

Okay, got it! Thanks for sharing that you're also a fan of the New England Patriots in the NFL. That makes sense, given your interest in other Boston sports teams like the Celtics. The Patriots have also had a very successful run over the past couple of decades, winning multiple Super Bowls. It's fun to follow winning franchises like the Celtics and Patriots. Do you have a favorite Patriots player or moment that stands out to you?
================================[1m Remove Message [0m================================


================================[1m Remove Message [0m================================


================================[1m Remove Message [0m================================


================================[1m Remove Message [0m================================


================================[1m Remove Message [0m================================


================================[1m Remove Message [0m================================


Okay, extending the summary with the new information:

- You initially introduced yourself as Bob and said you like the Boston Celtics basketball team. 
- I acknowledged that and we discussed your appreciation for the Celtics' history of winning.
- You then asked what your name was, and I reminded you that you had introduced yourself as Bob earlier in the conversation.
- You followed up by asking what NFL team I thought you might like, and I explained that I didn't have any prior information about your NFL team preferences.
- You then revealed that you are also a fan of the New England Patriots, which made sense given your Celtics fandom.
- I responded positively to this new information, noting the Patriots' own impressive success and dynasty over the past couple of decades.
- I then asked if you have a particular favorite Patriots player or moment that stands out to you, continuing the friendly, conversational tone.

Overall, the discussion has focused on your sports team preferences, with you sharing that you are a fan of both the Celtics and the Patriots. I've tried to engage with your interests and ask follow-up questions to keep the dialogue flowing.
```

# Tool calling
## How to call tools using ToolNode
ToolNode is a LangChain Runnable that takes graph state (with a list of messages) as input and outputs state update with the result of tool calls. It is designed to work well out-of-box with LangGraph's prebuilt ReAct agent, but can also work with any StateGraph as long as its state has a messages key with an appropriate reducer (see MessagesState).


## Define tools
```
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from langgraph.prebuilt import ToolNode
```

```
@tool
def get_weather(location: str):
    """Call to get the current weather."""
    if location.lower() in ["sf", "san francisco"]:
        return "It's 60 degrees and foggy."
    else:
        return "It's 90 degrees and sunny."


@tool
def get_coolest_cities():
    """Get a list of coolest cities"""
    return "nyc, sf"
```

```
tools = [get_weather, get_coolest_cities]
tool_node = ToolNode(tools)
```

## Manually call ToolNode
ToolNode operates on graph state with a list of messages. It expects the last message in the list to be an AIMessage with tool_calls parameter.

Let's first see how to invoke the tool node manually:

```
message_with_single_tool_call = AIMessage(
    content="",
    tool_calls=[
        {
            "name": "get_weather",
            "args": {"location": "sf"},
            "id": "tool_call_id",
            "type": "tool_call",
        }
    ],
)

tool_node.invoke({"messages": [message_with_single_tool_call]})
```

```
{'messages': [ToolMessage(content="It's 60 degrees and foggy.", name='get_weather', tool_call_id='tool_call_id')]}
```

Note that typically you don't need to create AIMessage manually, and it will be automatically generated by any LangChain chat model that supports tool calling.

You can also do parallel tool calling using ToolNode if you pass multiple tool calls to AIMessage's tool_calls parameter:

```
message_with_multiple_tool_calls = AIMessage(
    content="",
    tool_calls=[
        {
            "name": "get_coolest_cities",
            "args": {},
            "id": "tool_call_id_1",
            "type": "tool_call",
        },
        {
            "name": "get_weather",
            "args": {"location": "sf"},
            "id": "tool_call_id_2",
            "type": "tool_call",
        },
    ],
)

tool_node.invoke({"messages": [message_with_multiple_tool_calls]})
```

```
{'messages': [ToolMessage(content='nyc, sf', name='get_coolest_cities', tool_call_id='tool_call_id_1'),
  ToolMessage(content="It's 60 degrees and foggy.", name='get_weather', tool_call_id='tool_call_id_2')]}
```

## Using with chat models

We'll be using a small chat model from Anthropic in our example. To use chat models with tool calling, we need to first ensure that the model is aware of the available tools. We do this by calling .bind_tools method on ChatAnthropic model



```
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode


model_with_tools = ChatAnthropic(
    model="claude-3-haiku-20240307", temperature=0
).bind_tools(tools)
```

```
model_with_tools.invoke("what's the weather in sf?").tool_calls
```

```
[{'name': 'get_weather',
  'args': {'location': 'San Francisco'},
  'id': 'toolu_01Fwm7dg1mcJU43Fkx2pqgm8',
  'type': 'tool_call'}]
```

As you can see, the AI message generated by the chat model already has tool_calls populated, so we can just pass it directly to ToolNode

```
tool_node.invoke({"messages": [model_with_tools.invoke("what's the weather in sf?")]})
```

```
{'messages': [ToolMessage(content="It's 60 degrees and foggy.", name='get_weather', tool_call_id='toolu_01LFvAVT3xJMeZS6kbWwBGZK')]}
```

## ReAct Agent
Next, let's see how to use ToolNode inside a LangGraph graph. Let's set up a graph implementation of the ReAct agent. This agent takes some query as input, then repeatedly call tools until it has enough information to resolve the query. We'll be using ToolNode and the Anthropic model with tools we just defined



```
from typing import Literal

from langgraph.graph import StateGraph, MessagesState, START, END


def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END


def call_model(state: MessagesState):
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


workflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_edge("tools", "agent")

app = workflow.compile()
```

```
from IPython.display import Image, display

try:
    display(Image(app.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass
```

```
# example with a single tool call
for chunk in app.stream(
    {"messages": [("human", "what's the weather in sf?")]}, stream_mode="values"
):
    chunk["messages"][-1].pretty_print()
```

```
================================[1m Human Message [0m=================================

what's the weather in sf?
==================================[1m Ai Message [0m==================================

[{'text': "Okay, let's check the weather in San Francisco:", 'type': 'text'}, {'id': 'toolu_01LdmBXYeccWKdPrhZSwFCDX', 'input': {'location': 'San Francisco'}, 'name': 'get_weather', 'type': 'tool_use'}]
Tool Calls:
  get_weather (toolu_01LdmBXYeccWKdPrhZSwFCDX)
 Call ID: toolu_01LdmBXYeccWKdPrhZSwFCDX
  Args:
    location: San Francisco
=================================[1m Tool Message [0m=================================
Name: get_weather

It's 60 degrees and foggy.
==================================[1m Ai Message [0m==================================

The weather in San Francisco is currently 60 degrees with foggy conditions.
```

```
# example with a multiple tool calls in succession

for chunk in app.stream(
    {"messages": [("human", "what's the weather in the coolest cities?")]},
    stream_mode="values",
):
    chunk["messages"][-1].pretty_print()
```

```
================================[1m Human Message [0m=================================

what's the weather in the coolest cities?
==================================[1m Ai Message [0m==================================

[{'text': "Okay, let's find out the weather in the coolest cities:", 'type': 'text'}, {'id': 'toolu_01LFZUWTccyveBdaSAisMi95', 'input': {}, 'name': 'get_coolest_cities', 'type': 'tool_use'}]
Tool Calls:
  get_coolest_cities (toolu_01LFZUWTccyveBdaSAisMi95)
 Call ID: toolu_01LFZUWTccyveBdaSAisMi95
  Args:
=================================[1m Tool Message [0m=================================
Name: get_coolest_cities

nyc, sf
==================================[1m Ai Message [0m==================================

[{'text': "Now let's get the weather for those cities:", 'type': 'text'}, {'id': 'toolu_01RHPQBhT1u6eDnPqqkGUpsV', 'input': {'location': 'nyc'}, 'name': 'get_weather', 'type': 'tool_use'}]
Tool Calls:
  get_weather (toolu_01RHPQBhT1u6eDnPqqkGUpsV)
 Call ID: toolu_01RHPQBhT1u6eDnPqqkGUpsV
  Args:
    location: nyc
=================================[1m Tool Message [0m=================================
Name: get_weather

It's 90 degrees and sunny.
==================================[1m Ai Message [0m==================================

[{'id': 'toolu_01W5sFGF8PfgYzdY4CqT5c6e', 'input': {'location': 'sf'}, 'name': 'get_weather', 'type': 'tool_use'}]
Tool Calls:
  get_weather (toolu_01W5sFGF8PfgYzdY4CqT5c6e)
 Call ID: toolu_01W5sFGF8PfgYzdY4CqT5c6e
  Args:
    location: sf
=================================[1m Tool Message [0m=================================
Name: get_weather

It's 60 degrees and foggy.
==================================[1m Ai Message [0m==================================

Based on the results, it looks like the weather in the coolest cities is:
- New York City: 90 degrees and sunny
- San Francisco: 60 degrees and foggy

So the weather in the coolest cities is a mix of warm and cool temperatures, with some sunny and some foggy conditions.
```

ToolNode can also handle errors during tool execution. You can enable / disable this by setting handle_tool_errors=True (enabled by default). See our guide on handling errors in ToolNode here

# How to integrate LangGraph into your React application
[React application](https://langchain-ai.github.io/langgraph/cloud/how-tos/use_stream_react/)


# Launch Local LangGraph Server
[LangGraph Server](https://langchain-ai.github.io/langgraph/tutorials/langgraph-platform/local-server/)