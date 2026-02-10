# Streaming
LangGraph implements a streaming system to surface real-time updates. Streaming is crucial for enhancing the responsiveness of applications built on LLMs. By displaying output progressively, even before a complete response is ready, streaming significantly improves user experience (UX), particularly when dealing with the latency of LLMs.


What’s possible with LangGraph streaming:

- **Stream graph state** — get state updates / values with updates and values modes.
- **Stream subgraph outputs** — include outputs from both the parent graph and any nested subgraphs.
- **Stream LLM tokens** — capture token streams from anywhere: inside nodes, subgraphs, or tools.
- **Stream custom data** — send custom updates or progress signals directly from tool functions.
- **Use multiple streaming modes** — choose from values (full state), updates (state deltas), messages (LLM tokens + metadata), custom (arbitrary user data), or debug (detailed traces).

## Supported stream modes
Pass one or more of the following stream modes as a list to the stream or astream methods:

| **Mode**     | **Description**                                                                                                                                                           |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **values**   | Streams the full value of the state after each step of the graph.                                                                                                         |
| **updates**  | Streams the updates to the state after each step of the graph. If multiple updates occur in the same step (e.g., multiple nodes run), each update is streamed separately. |
| **custom**   | Streams custom data emitted from inside your graph nodes.                                                                                                                 |
| **messages** | Streams 2-tuples **(LLM token, metadata)** from any graph nodes where an LLM is invoked.                                                                                  |
| **debug**    | Streams **all available information** throughout the graph execution.                                                                                                     |


## Basic usage example
LangGraph graphs expose the ```stream ``` (sync) and ```astream``` (async) methods to yield streamed outputs as iterators.

```
for chunk in graph.stream(inputs, stream_mode="updates"):
    print(chunk)
```

```
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    topic: str
    joke: str

def refine_topic(state: State):
    return {"topic": state["topic"] + " and cats"}

def generate_joke(state: State):
    return {"joke": f"This is a joke about {state['topic']}"}

graph = (
    StateGraph(State)
    .add_node(refine_topic)
    .add_node(generate_joke)
    .add_edge(START, "refine_topic")
    .add_edge("refine_topic", "generate_joke")
    .add_edge("generate_joke", END)
    .compile()
)

# The stream() method returns an iterator that yields streamed outputs
for chunk in graph.stream(  
    {"topic": "ice cream"},
    # Set stream_mode="updates" to stream only the updates to the graph state after each node
    # Other stream modes are also available. See supported stream modes for details
    stream_mode="updates",  
):
    print(chunk)
```

## Stream multiple modes
You can pass a list as the stream_mode parameter to stream multiple modes at once.
The streamed outputs will be tuples of (mode, chunk) where mode is the name of the stream mode and chunk is the data streamed by that mode.

```
for mode, chunk in graph.stream(inputs, stream_mode=["updates", "custom"]):
    print(chunk)
```


## Stream graph state
Use the stream modes updates and values to stream the state of the graph as it executes.

- updates streams the updates to the state after each step of the graph.
- values streams the full value of the state after each step of the graph.


```
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
  topic: str
  joke: str


def refine_topic(state: State):
    return {"topic": state["topic"] + " and cats"}


def generate_joke(state: State):
    return {"joke": f"This is a joke about {state['topic']}"}

graph = (
  StateGraph(State)
  .add_node(refine_topic)
  .add_node(generate_joke)
  .add_edge(START, "refine_topic")
  .add_edge("refine_topic", "generate_joke")
  .add_edge("generate_joke", END)
  .compile()
)
```

**More info link:** https://docs.langchain.com/oss/python/langgraph/streaming#stream-graph-state


