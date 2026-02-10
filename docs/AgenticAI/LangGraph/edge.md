# Edges
Edges define how the logic is routed and how the graph decides to stop. This is a big part of how your agents work and how different nodes communicate with each other. There are a few key types of edges:

- **Normal Edges:** Go directly from one node to the next.
- **Conditional Edges:** Call a function to determine which node(s) to go to next.
- Entry Point: Which node to call first when user input arrives.

## Normal Edges
If you always want to go from node A to node B, you can use the add_edge method directly.


```
graph.add_edge("node_a", "node_b")
````
graph.add_edge("node_a", "node_b")
```



If you want to optionally route to one or more edges (or optionally terminate), you can use the add_conditional_edges method. This method accepts the name of a node and a “routing function” to call after that node is executed:

```
graph.add_conditional_edges("node_a", routing_function)
```

Similar to nodes, the routing_function accepts the current state of the graph and returns a value.
By default, the return value routing_function is used as the name of the node (or list of nodes) to send the state to next. All those nodes will be run in parallel as a part of the next superstep.
You can optionally provide a dictionary that maps the routing_function’s output to the name of the next node.

```
graph.add_conditional_edges("node_a", routing_function, {True: "node_b", False: "node_c"})
```

- Use Command instead of conditional edges if you want to combine state updates and routing in a single function.
​


## Entry point
The entry point is the first node(s) that are run when the graph starts. You can use the add_edge method from the virtual START node to the first node to execute to specify where to enter the graph.


```
from langgraph.graph import START

graph.add_edge(START, "node_a")````

```

## Conditional entry point
A conditional entry point lets you start at different nodes depending on custom logic. You can use add_conditional_edges from the virtual START node to accomplish this.

```
from langgraph.graph import START

graph.add_conditional_edges(START, routing_function)
```

You can optionally provide a dictionary that maps the routing_function’s output to the name of the next node.
```graph.add_conditional_edges(START, routing_function, {True: "node_b", False: "node_c"})```

## Send
By default, Nodes and Edges are defined ahead of time and operate on the same shared state. However, there can be cases where the exact edges are not known ahead of time and/or you may want different versions of State to exist at the same time. A common example of this is with map-reduce design patterns. In this design pattern, a first node may generate a list of objects, and you may want to apply some other node to all those objects. The number of objects may be unknown ahead of time (meaning the number of edges may not be known) and the input State to the downstream Node should be different (one for each generated object).

To support this design pattern, LangGraph supports returning Send objects from conditional edges. Send takes two arguments: first is the name of the node, and second is the state to pass to that node.

```
def continue_to_jokes(state: OverallState):
    return [Send("generate_joke", {"subject": s}) for s in state['subjects']]

graph.add_conditional_edges("node_a", continue_to_jokes)
```


## Command
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

With Command you can also achieve dynamic control flow behavior (identical to conditional edges):


```
Ask AI
def my_node(state: State) -> Command[Literal["my_other_node"]]:
    if state["foo"] == "bar":
        return Command(update={"foo": "baz"}, goto="my_other_node")

```

## When should I use Command instead of conditional edges?

- Use **Command** when you need to **both** update the graph state **and** route to a different node.
- For example, when implementing **multi-agent handoffs** where it’s important to route to a different agent and pass some information to that agent.
- Use **conditional edges** to route between nodes conditionally without updating the state.


## Navigating to a node in a parent graph
If you are using **subgraphs**, you might want to navigate from a node within a subgraph to a different subgraph (i.e. a different node in the parent graph). To do so, you can specify **```graph=Command.PARENT```** in ```Command```:

```
def my_node(state: State) -> Command[Literal["other_subgraph"]]:
    return Command(
        update={"foo": "bar"},
        goto="other_subgraph",  # where `other_subgraph` is a node in the parent graph
        graph=Command.PARENT
    )
```

**Note:** Setting graph to Command.PARENT will navigate to the closest parent graph.

When you send updates from a subgraph node to a parent graph node for a key that’s shared by both parent and subgraph state schemas, you must define a reducer for the key you’re updating in the parent graph state. 


## Using inside tools
A common use case is updating graph state from inside a tool. For example, in a customer support application you might want to look up customer information based on their account number or ID in the beginning of the conversation.

## Human-in-the-loop
**Command** is an important part of human-in-the-loop workflows: when using **interrupt()** to collect user input, Command is then used to supply the input and resume execution via **Command(resume="User input")**


## Graph migrations
LangGraph can easily handle migrations of graph definitions (nodes, edges, and state) even when using a checkpointer to track state.

## Runtime context
When creating a graph, you can specify a context_schema for runtime context passed to nodes. This is useful for passing information to nodes that is not part of the graph state. For example, you might want to pass dependencies such as model name or a database connection.


```
@dataclass
class ContextSchema:
    llm_provider: str = "openai"

graph = StateGraph(State, context_schema=ContextSchema)
```

You can then pass this context into the graph using the context parameter of the invoke method.

```
graph.invoke(inputs, context={"llm_provider": "anthropic"})
```

You can then access and use this context inside a node or conditional edge:

```
from langgraph.runtime import Runtime

def node_a(state: State, runtime: Runtime[ContextSchema]):
    llm = get_llm(runtime.context.llm_provider)
    # ...
```

## Recursion limit
The recursion limit sets the maximum number of super-steps the graph can execute during a single execution. Once the limit is reached, LangGraph will raise GraphRecursionError. By default this value is set to 25 steps. The recursion limit can be set on any graph at runtime, and is passed to invoke/stream via the config dictionary. Importantly, recursion_limit is a standalone config key and should not be passed inside the configurable key as all other user-defined configuration. See the example below:

```
graph.invoke(inputs, config={"recursion_limit": 5}, context={"llm": "anthropic"})
```

## Accessing and handling the recursion counter
The current step counter is accessible in config["metadata"]["langgraph_step"] within any node, allowing for proactive recursion handling before hitting the recursion limit. This enables you to implement graceful degradation strategies within your graph logic.
​
How it works
The step counter is stored in config["metadata"]["langgraph_step"]. The recursion limit check follows the logic: step > stop where stop = step + recursion_limit + 1. When the limit is exceeded, LangGraph raises a GraphRecursionError.


## Accessing the current step counter
You can access the current step counter within any node to monitor execution progress.

```
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

def my_node(state: dict, config: RunnableConfig) -> dict:
    current_step = config["metadata"]["langgraph_step"]
    print(f"Currently on step: {current_step}")
    return state
```

## Proactive recursion handling
You can check the step counter and proactively route to a different node before hitting the limit. This allows for graceful degradation within your graph.

```
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END

def reasoning_node(state: dict, config: RunnableConfig) -> dict:
    current_step = config["metadata"]["langgraph_step"]
    recursion_limit = config["recursion_limit"]  # always present, defaults to 25

    # Check if we're approaching the limit (e.g., 80% threshold)
    if current_step >= recursion_limit * 0.8:
        return {
            **state,
            "route_to": "fallback",
            "reason": "Approaching recursion limit"
        }

    # Normal processing
    return {"messages": state["messages"] + ["thinking..."]}

def fallback_node(state: dict, config: RunnableConfig) -> dict:
    """Handle cases where recursion limit is approaching"""
    return {
        **state,
        "messages": state["messages"] + ["Reached complexity limit, providing best effort answer"]
    }

def route_based_on_state(state: dict) -> str:
    if state.get("route_to") == "fallback":
        return "fallback"
    elif state.get("done"):
        return END
    return "reasoning"

# Build graph
graph = StateGraph(dict)
graph.add_node("reasoning", reasoning_node)
graph.add_node("fallback", fallback_node)
graph.add_conditional_edges("reasoning", route_based_on_state)
graph.add_edge("fallback", END)
graph.set_entry_point("reasoning")

app = graph.compile()
```

## Proactive vs reactive approaches
There are two main approaches to handling recursion limits: proactive (monitoring within the graph) and reactive (catching errors externally).

```
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.errors import GraphRecursionError

# Proactive Approach (recommended)
def agent_with_monitoring(state: dict, config: RunnableConfig) -> dict:
    """Proactively monitor and handle recursion within the graph"""
    current_step = config["metadata"]["langgraph_step"]
    recursion_limit = config["recursion_limit"]

    # Early detection - route to internal handling
    if current_step >= recursion_limit - 2:  # 2 steps before limit
        return {
            **state,
            "status": "recursion_limit_approaching",
            "final_answer": "Reached iteration limit, returning partial result"
        }

    # Normal processing
    return {"messages": state["messages"] + [f"Step {current_step}"]}

# Reactive Approach (fallback)
try:
    result = graph.invoke(initial_state, {"recursion_limit": 10})
except GraphRecursionError as e:
    # Handle externally after graph execution fails
    result = fallback_handler(initial_state)
```

## Visualization
It’s often nice to be able to visualize graphs, especially as they get more complex. LangGraph comes with several built-in ways to visualize graphs. See

## Visualize your graph
https://docs.langchain.com/oss/python/langgraph/use-graph-api#visualize-your-graph



