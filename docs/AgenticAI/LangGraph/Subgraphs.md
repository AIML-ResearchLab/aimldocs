# Subgraphs
A subgraph is a ```graph``` that is used as a ```node``` in another graph.

Subgraphs are useful for:

- Building ```multi-agent systems```
- Re-using a set of nodes in multiple graphs
- Distributing development: when you want different teams to work on different parts of the graph independently, you can define each part as a subgraph, and as long as the subgraph interface (the input and output schemas) is respected, the parent graph can be built without knowing any details of the subgraph


When adding subgraphs, you need to define how the parent graph and the subgraph communicate:

- **```Invoke a graph from a node```** — subgraphs are called from inside a node in the parent graph
- **```Add a graph as a node```** — a subgraph is added directly as a node in the parent and ```shares state keys``` with the parent


## Setup

```
npm install @langchain/langgraph
```

## Invoke a graph from a node

A simple way to implement a subgraph is to invoke a graph from inside the node of another graph. In this case subgraphs can have ```completely different schemas``` from the parent graph (no shared keys). For example, you might want to keep a private message history for each of the agents in a ```multi-agent``` system.

If that’s the case for your application, you need to define a node ```function that invokes the subgraph```. This function needs to transform the input (parent) state to the subgraph state before invoking the subgraph, and transform the results back to the parent state before returning the state update from the node.

```
import { StateGraph, START } from "@langchain/langgraph";
import * as z from "zod";

const SubgraphState = z.object({
  bar: z.string(),
});

// Subgraph
const subgraphBuilder = new StateGraph(SubgraphState)
  .addNode("subgraphNode1", (state) => {
    return { bar: "hi! " + state.bar };
  })
  .addEdge(START, "subgraphNode1");

const subgraph = subgraphBuilder.compile();

// Parent graph
const State = z.object({
  foo: z.string(),
});

// Transform the state to the subgraph state and back
const builder = new StateGraph(State)
  .addNode("node1", async (state) => {
    const subgraphOutput = await subgraph.invoke({ bar: state.foo });
    return { foo: subgraphOutput.bar };
  })
  .addEdge(START, "node1");

const graph = builder.compile();
```

## Full example: different state schemas

```
import { StateGraph, START } from "@langchain/langgraph";
import * as z from "zod";

// Define subgraph
const SubgraphState = z.object({
  // note that none of these keys are shared with the parent graph state
  bar: z.string(),
  baz: z.string(),
});

const subgraphBuilder = new StateGraph(SubgraphState)
  .addNode("subgraphNode1", (state) => {
    return { baz: "baz" };
  })
  .addNode("subgraphNode2", (state) => {
    return { bar: state.bar + state.baz };
  })
  .addEdge(START, "subgraphNode1")
  .addEdge("subgraphNode1", "subgraphNode2");

const subgraph = subgraphBuilder.compile();

// Define parent graph
const ParentState = z.object({
  foo: z.string(),
});

const builder = new StateGraph(ParentState)
  .addNode("node1", (state) => {
    return { foo: "hi! " + state.foo };
  })
  .addNode("node2", async (state) => {
    const response = await subgraph.invoke({ bar: state.foo });   
    return { foo: response.bar };   
  })
  .addEdge(START, "node1")
  .addEdge("node1", "node2");

const graph = builder.compile();

for await (const chunk of await graph.stream(
  { foo: "foo" },
  { subgraphs: true }
)) {
  console.log(chunk);
}
```

## Full example: different state schemas (two levels of subgraphs)

- This is an example with two levels of **subgraphs:** ```parent``` -> ```child``` -> ```grandchild```.

```
import { StateGraph, START, END } from "@langchain/langgraph";
import * as z from "zod";

// Grandchild graph
const GrandChildState = z.object({
  myGrandchildKey: z.string(),
});

const grandchild = new StateGraph(GrandChildState)
  .addNode("grandchild1", (state) => {
    // NOTE: child or parent keys will not be accessible here
    return { myGrandchildKey: state.myGrandchildKey + ", how are you" };
  })
  .addEdge(START, "grandchild1")
  .addEdge("grandchild1", END);

const grandchildGraph = grandchild.compile();

// Child graph
const ChildState = z.object({
  myChildKey: z.string(),
});

const child = new StateGraph(ChildState)
  .addNode("child1", async (state) => {
    // NOTE: parent or grandchild keys won't be accessible here
    const grandchildGraphInput = { myGrandchildKey: state.myChildKey };   
    const grandchildGraphOutput = await grandchildGraph.invoke(grandchildGraphInput);
    return { myChildKey: grandchildGraphOutput.myGrandchildKey + " today?" };   
  })   
  .addEdge(START, "child1")
  .addEdge("child1", END);

const childGraph = child.compile();

// Parent graph
const ParentState = z.object({
  myKey: z.string(),
});

const parent = new StateGraph(ParentState)
  .addNode("parent1", (state) => {
    // NOTE: child or grandchild keys won't be accessible here
    return { myKey: "hi " + state.myKey };
  })
  .addNode("child", async (state) => {
    const childGraphInput = { myChildKey: state.myKey };   
    const childGraphOutput = await childGraph.invoke(childGraphInput);
    return { myKey: childGraphOutput.myChildKey };   
  })   
  .addNode("parent2", (state) => {
    return { myKey: state.myKey + " bye!" };
  })
  .addEdge(START, "parent1")
  .addEdge("parent1", "child")
  .addEdge("child", "parent2")
  .addEdge("parent2", END);

const parentGraph = parent.compile();

for await (const chunk of await parentGraph.stream(
  { myKey: "Bob" },
  { subgraphs: true }
)) {
  console.log(chunk);
}
```

## Add a graph as a node
When the parent graph and subgraph can communicate over a shared state key (channel) in the ```schema```, you can add a graph as a ```node``` in another graph. For example, in ```multi-agent``` systems, the agents often communicate over a shared ```messages``` key.

![alt text](./image/Langgraph8.png)

If your subgraph shares state keys with the parent graph, you can follow these steps to add it to your graph:

1. Define the subgraph workflow (```subgraphBuilder``` in the example below) and compile it
2. Pass compiled subgraph to the ```.addNode``` method when defining the parent graph workflow

```
import { StateGraph, START } from "@langchain/langgraph";
import * as z from "zod";

const State = z.object({
  foo: z.string(),
});

// Subgraph
const subgraphBuilder = new StateGraph(State)
  .addNode("subgraphNode1", (state) => {
    return { foo: "hi! " + state.foo };
  })
  .addEdge(START, "subgraphNode1");

const subgraph = subgraphBuilder.compile();

// Parent graph
const builder = new StateGraph(State)
  .addNode("node1", subgraph)
  .addEdge(START, "node1");

const graph = builder.compile();
```

## ull example: shared state schemas

```
import { StateGraph, START } from "@langchain/langgraph";
import * as z from "zod";

// Define subgraph
const SubgraphState = z.object({
  foo: z.string(),    
  bar: z.string(),    
});

const subgraphBuilder = new StateGraph(SubgraphState)
  .addNode("subgraphNode1", (state) => {
    return { bar: "bar" };
  })
  .addNode("subgraphNode2", (state) => {
    // note that this node is using a state key ('bar') that is only available in the subgraph
    // and is sending update on the shared state key ('foo')
    return { foo: state.foo + state.bar };
  })
  .addEdge(START, "subgraphNode1")
  .addEdge("subgraphNode1", "subgraphNode2");

const subgraph = subgraphBuilder.compile();

// Define parent graph
const ParentState = z.object({
  foo: z.string(),
});

const builder = new StateGraph(ParentState)
  .addNode("node1", (state) => {
    return { foo: "hi! " + state.foo };
  })
  .addNode("node2", subgraph)
  .addEdge(START, "node1")
  .addEdge("node1", "node2");

const graph = builder.compile();

for await (const chunk of await graph.stream({ foo: "foo" })) {
  console.log(chunk);
}
```

## Add persistence
You only need to **provide the checkpointer when compiling the parent graph**.
LangGraph will automatically propagate the checkpointer to the child subgraphs.

```
import { StateGraph, START, MemorySaver } from "@langchain/langgraph";
import * as z from "zod";

const State = z.object({
  foo: z.string(),
});

// Subgraph
const subgraphBuilder = new StateGraph(State)
  .addNode("subgraphNode1", (state) => {
    return { foo: state.foo + "bar" };
  })
  .addEdge(START, "subgraphNode1");

const subgraph = subgraphBuilder.compile();

// Parent graph
const builder = new StateGraph(State)
  .addNode("node1", subgraph)
  .addEdge(START, "node1");

const checkpointer = new MemorySaver();
const graph = builder.compile({ checkpointer });
```

If you want the subgraph to ```have its own memory```, you can compile it with the appropriate checkpointer option. This is useful in ```multi-agent``` systems, if you want agents to keep track of their internal message histories:

```
const subgraphBuilder = new StateGraph(...)
const subgraph = subgraphBuilder.compile({ checkpointer: true });
```

## View subgraph state
When you enable ```persistence```, you can ```inspect the graph state``` (checkpoint) via the appropriate method. To view the subgraph state, you can use the subgraphs option.

You can inspect the graph state via ```graph.getState(config)```. To view the subgraph state, you can use ```graph.getState(config, { subgraphs: true })```.

## View interrupted subgraph state

```
import { StateGraph, START, MemorySaver, interrupt, Command } from "@langchain/langgraph";
import * as z from "zod";

const State = z.object({
  foo: z.string(),
});

// Subgraph
const subgraphBuilder = new StateGraph(State)
  .addNode("subgraphNode1", (state) => {
    const value = interrupt("Provide value:");
    return { foo: state.foo + value };
  })
  .addEdge(START, "subgraphNode1");

const subgraph = subgraphBuilder.compile();

// Parent graph
const builder = new StateGraph(State)
  .addNode("node1", subgraph)
  .addEdge(START, "node1");

const checkpointer = new MemorySaver();
const graph = builder.compile({ checkpointer });

const config = { configurable: { thread_id: "1" } };

await graph.invoke({ foo: "" }, config);
const parentState = await graph.getState(config);
const subgraphState = (await graph.getState(config, { subgraphs: true })).tasks[0].state;   

// resume the subgraph
await graph.invoke(new Command({ resume: "bar" }), config);
```

## Stream subgraph outputs
To include outputs from subgraphs in the streamed outputs, you can set the subgraphs option in the stream method of the parent graph. This will stream outputs from both the parent graph and any subgraphs.

```
for await (const chunk of await graph.stream(
  { foo: "foo" },
  {
    subgraphs: true,   
    streamMode: "updates",
  }
)) {
  console.log(chunk);
}
```

1. Set **subgraphs:** ```true``` to stream outputs from subgraphs.

## Stream from subgraphs

```
import { StateGraph, START } from "@langchain/langgraph";
import * as z from "zod";

// Define subgraph
const SubgraphState = z.object({
  foo: z.string(),
  bar: z.string(),
});

const subgraphBuilder = new StateGraph(SubgraphState)
  .addNode("subgraphNode1", (state) => {
    return { bar: "bar" };
  })
  .addNode("subgraphNode2", (state) => {
    // note that this node is using a state key ('bar') that is only available in the subgraph
    // and is sending update on the shared state key ('foo')
    return { foo: state.foo + state.bar };
  })
  .addEdge(START, "subgraphNode1")
  .addEdge("subgraphNode1", "subgraphNode2");

const subgraph = subgraphBuilder.compile();

// Define parent graph
const ParentState = z.object({
  foo: z.string(),
});

const builder = new StateGraph(ParentState)
  .addNode("node1", (state) => {
    return { foo: "hi! " + state.foo };
  })
  .addNode("node2", subgraph)
  .addEdge(START, "node1")
  .addEdge("node1", "node2");

const graph = builder.compile();

for await (const chunk of await graph.stream(
  { foo: "foo" },
  {
    streamMode: "updates",
    subgraphs: true,   
  }
)) {
  console.log(chunk);
}
```

