# Use time-travel

When working with non-deterministic systems that make model-based decisions (e.g., agents powered by LLMs), it can be useful to examine their decision-making process in detail:

1. **Understand reasoning:** Analyze the steps that led to a successful result.
2. **Debug mistakes:** Identify where and why errors occurred.
3. **Explore alternatives:** Test different paths to uncover better solutions.


LangGraph provides **time travel** functionality to support these use cases. Specifically, you can resume execution from a prior checkpoint — either replaying the same state or modifying it to explore alternatives.In all cases, resuming past execution produces a new fork in the history.

To use **time-travel** in LangGraph:


1. **Run the graph** with initial inputs using ```invoke``` or ```stream``` methods.
2. **Identify a checkpoint in an existing thread:** Use the ```getStateHistory``` method to retrieve the execution history for a specific ```thread_id``` and locate the desired ```checkpoint_id. Alternatively, set a breakpoint before the node(s) where you want execution to pause. You can then find the most recent checkpoint recorded up to that breakpoint.

3. **Update the graph state (optional):** Use the updateState method to modify the graph’s state at the checkpoint and resume execution from alternative state.
4. **Resume execution from the checkpoint:** Use the invoke or stream methods with an input of null and a configuration containing the appropriate thread_id and checkpoint_id.

```
import { v4 as uuidv4 } from "uuid";
import * as z from "zod";
import { StateGraph, START, END } from "@langchain/langgraph";
import { ChatAnthropic } from "@langchain/anthropic";
import { MemorySaver } from "@langchain/langgraph";

const State = z.object({
  topic: z.string().optional(),
  joke: z.string().optional(),
});

const model = new ChatAnthropic({
  model: "claude-sonnet-4-5-20250929",
  temperature: 0,
});

// Build workflow
const workflow = new StateGraph(State)
  // Add nodes
  .addNode("generateTopic", async (state) => {
    // LLM call to generate a topic for the joke
    const msg = await model.invoke("Give me a funny topic for a joke");
    return { topic: msg.content };
  })
  .addNode("writeJoke", async (state) => {
    // LLM call to write a joke based on the topic
    const msg = await model.invoke(`Write a short joke about ${state.topic}`);
    return { joke: msg.content };
  })
  // Add edges to connect nodes
  .addEdge(START, "generateTopic")
  .addEdge("generateTopic", "writeJoke")
  .addEdge("writeJoke", END);

// Compile
const checkpointer = new MemorySaver();
const graph = workflow.compile({ checkpointer });
```

**1. Run the graph**

```
const config = {
  configurable: {
    thread_id: uuidv4(),
  },
};

const state = await graph.invoke({}, config);

console.log(state.topic);
console.log();
console.log(state.joke);
```

**2. Identify a checkpoint**

```
// The states are returned in reverse chronological order.
const states = [];
for await (const state of graph.getStateHistory(config)) {
  states.push(state);
}

for (const state of states) {
  console.log(state.next);
  console.log(state.config.configurable?.checkpoint_id);
  console.log();
}
```

```
// This is the state before last (states are listed in chronological order)
const selectedState = states[1];
console.log(selectedState.next);
console.log(selectedState.values);
```

**3. Update the state**

updateState will create a new checkpoint. The new checkpoint will be associated with the same thread, but a new checkpoint ID.

```
const newConfig = await graph.updateState(selectedState.config, {
  topic: "chickens",
});
console.log(newConfig);
```

**4. Resume execution from the checkpoint**

```
await graph.invoke(null, newConfig);
```

