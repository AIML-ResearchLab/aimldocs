# Graph API

## What is Graphs?
LangGraph treats any AI workflow—like an agent reasoning, using tools, collaborating with other agents, or looping until a goal is reached—as a graph, not a sequence.

A graph is made up of:

- **Nodes** → steps that do work (LLM call, tool call, function, decision, API call, human approval, etc.)
- **Edges** → rules that decide where to go next based on the result or state

## 🔍 Why graphs?
Typical LLM pipelines (LangChain chains, simple agents) run in a linear flow:

```
User → LLM → Tool → LLM → Result
```

But real agent workflows are not linear. They often need:

- loops
- conditional decisions
- retries
- multi-step task decomposition
- multiple agents collaborating
- waiting for user input
- pausing + resuming long tasks


A graph can easily express all these patterns.


## 🧠 How LangGraph uses graphs

**1. Nodes = work**

Each node performs a meaningful step:

- Call an LLM
- Call a tool
- Summarize text
- Search the web
- Parse results
- Decide the next step


**2. Edges = transitions**

Edges define:

- What happens next?
- **Conditional routing** (if the tool failed → go to repair node)
- **Loops** (repeat analysis until done)
- **Branching** (select which agent handles a task)


**3. State = memory**

The shared state tracks:

- conversation history
- tool results
- agent decisions
- partial outputs


Nodes update the state and edges read it to make decisions.

## 📊 Visualizing the idea

![alt text](./image/langgraph1.png)

**This is a graph with:**

- **nodes:** A, B, C
- **edges:** transitions based on success/failure
- **loops:** from C → A → B until successful

## 💡 The big advantage
Using a graph lets agents behave like **deterministic programs**, not unpredictable black boxes.

- predictable sequences
- strict control over agent reasoning
- recoverable/resumable workflows
- multi-agent orchestration
- production-grade reliability

This is why LangGraph excels in agentic AI, workflow AI, copilots, RAG agents, and automated multi-step tasks.


**You define the behavior of your agents using ```three key components```:**

1. **State:** A shared data structure that represents the current snapshot of your application. It can be any data type, but is typically defined using a shared state schema.

2. **Nodes:** Functions that encode the logic of your agents. They receive the current state as input, perform some computation or side-effect, and return an updated state.

3. **Edges:** Functions that determine which Node to execute next based on the current state. They can be conditional branches or fixed transitions.

By composing **Nodes** and **Edges**, you can create complex, looping workflows that evolve the state over time. The real power, though, comes from how LangGraph manages that state.


**To emphasize:** Nodes and Edges are nothing more than functions – they can contain an LLM or just good ol’ code.

**In short:** nodes do the work, edges tell what to do next.


LangGraph’s underlying graph algorithm uses **message passing** to define a general program. When a Node completes its operation, it sends messages along one or more edges to other node(s). These recipient nodes then execute their functions, pass the resulting messages to the next set of nodes, and the process continues. Inspired by Google’s **Pregel** system, the program proceeds in discrete “super-steps.”

A super-step can be considered a single iteration over the graph nodes. Nodes that run in parallel are part of the same super-step, while nodes that run sequentially belong to separate super-steps. At the start of graph execution, all nodes begin in an **inactive** state. A node becomes **active** when it receives a new message (state) on any of its incoming edges (or “channels”). The **active** node then runs its function and responds with updates. At the end of each super-step, nodes with no incoming messages vote to **halt** by marking themselves as **inactive**. The graph execution terminates when all nodes are **inactive** and no messages are in transit.



## StateGraph

The **StateGraph** class is the main graph class to use. This is parameterized by a user defined State object.

## Compiling your graph

To build your graph, you first define the **state**, you then add **nodes** and **edges**, and then you compile it. What exactly is compiling your graph and why is it needed?

Compiling is a pretty simple step. It provides a few basic checks on the structure of your graph (no orphaned nodes, etc). It is also where you can specify runtime args like **checkpointers** and breakpoints. You compile your graph by just calling the **.compile** method:

```
graph = graph_builder.compile(...)
```

**Note:** You MUST compile your graph before you can use it.

## State
The first thing you do when you define a graph is define the **State** of the graph. The **State** consists of the **schema of the graph** as well as **reducer functions** which specify how to apply updates to the **state**. The schema of the **State** will be the input schema to all **Nodes** and **Edges** in the **graph**, and can be either a **TypedDict** or a **Pydantic** model. All **Nodes** will emit updates to the **State** which are then applied using the specified **reducer function**.


## Schema
- The main documented way to specify the schema of a graph is by using a **TypedDict**.
- If you want to provide default values in your state, use a **dataclass**.
- We also support using a Pydantic **BaseModel** as your graph state if you want recursive data validation (though note that Pydantic is less performant than a **TypedDict** or **dataclass**).
- By default, the graph will have the same input and output schemas. If you want to change this, you can also specify explicit input and output schemas directly. This is useful when you have a lot of keys, and some are explicitly for input and others for output.

## Multiple schemas
Typically, all graph nodes communicate with a single schema. This means that they will read and write to the same state channels. But, there are cases where we want more control over this:

   - Internal nodes can pass information that is not required in the graph’s input / output.
   - We may also want to use different input / output schemas for the graph. The output might, for example, only contain a single relevant output key.

It is possible to have nodes write to private state channels inside the graph for internal node communication. We can simply define a private schema, **PrivateState**.


It is also possible to define explicit input and output schemas for a graph. In these cases, we define an “internal” schema that contains all keys relevant to graph operations. But, we also define input and output schemas that are sub-sets of the “internal” schema to constrain the input and output of the graph. 

**Let’s look at an example:**

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

builder = StateGraph(OverallState,input_schema=InputState,output_schema=OutputState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", "node_3")
builder.add_edge("node_3", END)

graph = builder.compile()
graph.invoke({"user_input":"My"})
# {'graph_output': 'My name is Lance'}
```

There are two subtle and important points to note here:

1. We pass **state: InputState** as the input schema to **node_1** But, we write out to foo, a channel in **OverallState**.
How can we write out to a state channel that is not included in the input schema? This is because a node can write to any state channel in the graph state. The graph state is the union of the state channels defined at initialization, which includes OverallState and the filters **InputState** and **OutputState**.

2. We initialize the graph with:

```
StateGraph(
    OverallState,
    input_schema=InputState,
    output_schema=OutputState
)
```

So, how can we write to **PrivateState** in **node_2**? How does the graph gain access to this schema if it was not passed in the StateGraph initialization?

We can do this because _**nodes** can also declare additional state **channels_** as long as the state schema definition exists. In this case, the PrivateState schema is defined, so we can add bar as a new state channel in the graph and write to it.

## Reducers
Reducers are key to understanding how updates from nodes are applied to the **State**

Each key in the State has its own independent reducer function. If no reducer function is explicitly specified then it is assumed that all updates to that key should override it. There are a few different types of reducers, starting with the default type of reducer:

**Default Reducer**

These two examples show how to use the default reducer:

```
from typing_extensions import TypedDict

class State(TypedDict):
    foo: int
    bar: list[str]
```
In this example, no reducer functions are specified for any key. Let’s assume the input to the graph is:

```{"foo": 1, "bar": ["hi"]}``` . Let’s then assume the first **Node** returns ```{"foo": 2}```. This is treated as an update to the state. Notice that the Node does not need to return the whole State schema - just an update. After applying this update, the State would then be {"foo": 2, "bar": ["hi"]}.If the second node returns {"bar": ["bye"]} then the State would then be {"foo": 2, "bar": ["bye"]}

```
from typing import Annotated
from typing_extensions import TypedDict
from operator import add

class State(TypedDict):
    foo: int
    bar: Annotated[list[str], add]
```

In this example, we’ve used the ```Annotated``` type to specify a reducer function ```(operator.add)``` for the second key (bar). Note that the first key remains unchanged. Let’s assume the input to the graph is {"foo": 1, "bar": ["hi"]}. Let’s then assume the first Node returns {"foo": 2}. This is treated as an update to the state. Notice that the Node does not need to return the whole State schema - just an update. After applying this update, the State would then be {"foo": 2, "bar": ["hi"]}. If the second node returns {"bar": ["bye"]} then the State would then be {"foo": 2, "bar": ["hi", "bye"]}. Notice here that the bar key is updated by adding the two lists together.
​

## Overwrite
In some cases, you may want to bypass a reducer and directly ```overwrite``` a state value. LangGraph provides the ```Overwrite``` type for this purpose.


## Working with Messages in Graph State

**Why use messages?**
Most modern LLM providers have a chat model interface that accepts a list of messages as input. LangChain’s **chat model interface** in particular accepts a list of message objects as inputs. These messages come in a variety of forms such as **HumanMessage** (user input) or **AIMessage** (LLM response).


## Using Messages in your Graph
In many cases, it is helpful to store prior conversation history as a list of messages in your graph state. To do so, we can add a key (channel) to the graph state that stores a list of **Message** objects and annotate it with a reducer function

However, you might also want to manually update messages in your graph state (e.g. human-in-the-loop). If you were to use operator.add, the manual state updates you send to the graph would be appended to the existing list of messages, instead of updating existing messages. To avoid that, you need a reducer that can keep track of message IDs and overwrite existing messages, if updated. To achieve this, you can use the prebuilt add_messages function. For brand new messages, it will simply append to existing list, but it will also handle the updates for existing messages correctly.

## Serialization
In addition to keeping track of message IDs, the **add_messages** function will also try to deserialize messages into LangChain Message objects whenever a state update is received on the messages channel.

See more information on LangChain serialization/deserialization here. This allows sending graph inputs / state updates in the following format:

```
# this is supported
{"messages": [HumanMessage(content="message")]}

# and this is also supported
{"messages": [{"type": "human", "content": "message"}]}
```

Since the state updates are always deserialized into LangChain Messages when using add_messages, you should use dot notation to access message attributes, like ```state["messages"][-1].content```.


Below is an example of a graph that uses add_messages as its reducer function.

```
from langchain.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict

class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

## MessagesState
Since having a list of messages in your state is so common, there exists a prebuilt state called **MessagesState** which makes it easy to use messages. **MessagesState** is defined with a single messages key which is a list of **AnyMessage** objects and uses the **add_messages** reducer. Typically, there is more state to track than just messages, so we see people subclass this state and add more fields, like:






