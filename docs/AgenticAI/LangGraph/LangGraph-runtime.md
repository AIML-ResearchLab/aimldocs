# LangGraph runtime

**Pregel** implements LangGraph’s runtime, managing the execution of LangGraph applications.

Compiling a ```StateGraph``` or creating an ```entrypoint``` produces a ```Pregel``` instance that can be invoked with input.


**Note:** The ```Pregel``` runtime is named after ```Google’s Pregel algorithm```, which describes an efficient method for large-scale parallel computation using graphs.

## Overview
In LangGraph, Pregel combines ```actors``` and ```channels``` into a single application. ```Actors``` read data from channels and write data to channels. Pregel organizes the execution of the application into multiple steps, following the ```Pregel Algorithm/Bulk Synchronous Parallel``` model.

Each step consists of three phases:

- **Plan:** Determine which ```actors``` to execute in this step. For example, in the first step, select the ```actors``` that subscribe to the special ```input``` channels; in subsequent steps, select the ```actors``` that subscribe to channels updated in the previous step.

- **Execution:** Execute all selected ```actors``` in parallel, until all complete, or one fails, or a timeout is reached. During this phase, channel updates are invisible to actors until the next step.

- **Update:** Update the channels with the values written by the actors in this step.

Repeat until no actors are selected for execution, or a maximum number of steps is reached.


## Actors
An ```actor``` is a ```PregelNode```. It subscribes to channels, reads data from them, and writes data to them. It can be thought of as an actor in the Pregel algorithm. PregelNodes implement LangChain’s Runnable interface.

## Channels
Channels are used to communicate between actors (PregelNodes). Each channel has a value type, an update type, and an update function – which takes a sequence of updates and modifies the stored value. Channels can be used to send data from one chain to another, or to send data from a chain to itself in a future step. LangGraph provides a number of built-in channels:

- **LastValue:** The default channel, stores the last value sent to the channel, useful for input and output values, or for sending data from one step to the next.

- **Topic:** A configurable PubSub Topic, useful for sending multiple values between actors, or for accumulating output. Can be configured to deduplicate values or to accumulate values over the course of multiple steps.

- **BinaryOperatorAggregate:** stores a persistent value, updated by applying a binary operator to the current value and each update sent to the channel, useful for computing aggregates over multiple steps; e.g.,```total = BinaryOperatorAggregate(int, operator.add)```

## Examples
While most users will interact with Pregel through the ```StateGraph API``` or the ```entrypoint ```decorator, it is possible to interact with Pregel directly.

Below are a few different examples to give  a sense of the Pregel API.

## Single node

```
import { EphemeralValue } from "@langchain/langgraph/channels";
import { Pregel, NodeBuilder } from "@langchain/langgraph/pregel";

const node1 = new NodeBuilder()
  .subscribeOnly("a")
  .do((x: string) => x + x)
  .writeTo("b");

const app = new Pregel({
  nodes: { node1 },
  channels: {
    a: new EphemeralValue<string>(),
    b: new EphemeralValue<string>(),
  },
  inputChannels: ["a"],
  outputChannels: ["b"],
});

await app.invoke({ a: "foo" });
```

## Multiple nodes

```
import { LastValue, EphemeralValue } from "@langchain/langgraph/channels";
import { Pregel, NodeBuilder } from "@langchain/langgraph/pregel";

const node1 = new NodeBuilder()
  .subscribeOnly("a")
  .do((x: string) => x + x)
  .writeTo("b");

const node2 = new NodeBuilder()
  .subscribeOnly("b")
  .do((x: string) => x + x)
  .writeTo("c");

const app = new Pregel({
  nodes: { node1, node2 },
  channels: {
    a: new EphemeralValue<string>(),
    b: new LastValue<string>(),
    c: new EphemeralValue<string>(),
  },
  inputChannels: ["a"],
  outputChannels: ["b", "c"],
});

await app.invoke({ a: "foo" });
```

## Topic

```
import { EphemeralValue, Topic } from "@langchain/langgraph/channels";
import { Pregel, NodeBuilder } from "@langchain/langgraph/pregel";

const node1 = new NodeBuilder()
  .subscribeOnly("a")
  .do((x: string) => x + x)
  .writeTo("b", "c");

const node2 = new NodeBuilder()
  .subscribeTo("b")
  .do((x: { b: string }) => x.b + x.b)
  .writeTo("c");

const app = new Pregel({
  nodes: { node1, node2 },
  channels: {
    a: new EphemeralValue<string>(),
    b: new EphemeralValue<string>(),
    c: new Topic<string>({ accumulate: true }),
  },
  inputChannels: ["a"],
  outputChannels: ["c"],
});

await app.invoke({ a: "foo" });
```

## BinaryOperatorAggregate

This example demonstrates how to use the BinaryOperatorAggregate channel to implement a reducer.

```
import { EphemeralValue, BinaryOperatorAggregate } from "@langchain/langgraph/channels";
import { Pregel, NodeBuilder } from "@langchain/langgraph/pregel";

const node1 = new NodeBuilder()
  .subscribeOnly("a")
  .do((x: string) => x + x)
  .writeTo("b", "c");

const node2 = new NodeBuilder()
  .subscribeOnly("b")
  .do((x: string) => x + x)
  .writeTo("c");

const reducer = (current: string, update: string) => {
  if (current) {
    return current + " | " + update;
  } else {
    return update;
  }
};

const app = new Pregel({
  nodes: { node1, node2 },
  channels: {
    a: new EphemeralValue<string>(),
    b: new EphemeralValue<string>(),
    c: new BinaryOperatorAggregate<string>({ operator: reducer }),
  },
  inputChannels: ["a"],
  outputChannels: ["c"],
});

await app.invoke({ a: "foo" });
```

## Cycle
This example demonstrates how to introduce a cycle in the graph, by having a chain write to a channel it subscribes to. Execution will continue until a null value is written to the channel.

```
import { EphemeralValue } from "@langchain/langgraph/channels";
import { Pregel, NodeBuilder, ChannelWriteEntry } from "@langchain/langgraph/pregel";

const exampleNode = new NodeBuilder()
  .subscribeOnly("value")
  .do((x: string) => x.length < 10 ? x + x : null)
  .writeTo(new ChannelWriteEntry("value", { skipNone: true }));

const app = new Pregel({
  nodes: { exampleNode },
  channels: {
    value: new EphemeralValue<string>(),
  },
  inputChannels: ["value"],
  outputChannels: ["value"],
});

await app.invoke({ value: "a" });
```


