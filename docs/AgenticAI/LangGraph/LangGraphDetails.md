# Graphs
At its core, LangGraph models agent workflows as graphs. You define the behavior of your agents using three key components:

1. **Tools**
2. **LLM**
3. **State**
4. **Memory**
5. **Nodes**
6. **Edges**
7. **Compile**
8. **Invoke**
9. **ZOD**
10. **TOON**
11. **Annotation**
12. **Messages**
13. **Reducers**
14. **Serialization**
15. **START Node**
16. **END Node**
17. **Node Caching**
18. **Normal Edges**
19. **Conditional Edges**

| Layer | LangGraph Feature / Function | Purpose | Mandatory in Prod | Notes / Best Practice |
|------|-----------------------------|---------|------------------|----------------------|
| **Graph Definition** | `StateGraph` | Defines multi-agent workflow | ✅ Yes | Always use typed state |
|  | `add_node()` | Register agent / tool nodes | ✅ Yes | One responsibility per node |
|  | `add_edge()` | Static routing between agents | ✅ Yes | Avoid hard-coded branching in agents |
|  | `add_conditional_edges()` | Dynamic routing | ✅ Yes | Use rule-first, LLM-second |
|  | `set_entry_point()` | Define start node | ✅ Yes | Single entry point only |
|  | `set_finish_point()` | Define completion node | ✅ Yes | Required for safe termination |
| **State Management** | Typed State (Pydantic / TypedDict) | Shared agent state | ✅ Yes | Never use raw dict |
|  | Partial State Updates | Update only owned fields | ✅ Yes | Prevent state corruption |
|  | State History | Track decisions | ⚠️ Recommended | Required for audit |
| **Agent Execution** | Agent Node (LLM) | Reasoning & planning | ✅ Yes | Separate planner vs executor |
|  | Tool Node | External actions | ✅ Yes | All side-effects via tools |
|  | Subgraph | Reusable agent flows | ⚠️ Recommended | Use for SOPs / runbooks |
| **Control Flow** | Conditional Routing | Decision-based flow | ✅ Yes | Deterministic > LLM |
|  | Loop Control | Iterative reasoning | ⚠️ Recommended | Always cap iterations |
|  | `langgraph_step` | Proactive recursion guard | ✅ Yes | Prevent infinite loops |
| **Error Handling** | Failure Edges | Graceful failure routing | ✅ Yes | No uncaught exceptions |
|  | Retry Logic | Controlled retries | ⚠️ Recommended | Prefer Temporal for retries |
|  | `GraphRecursionError` | Loop overflow detection | ⚠️ Backup only | Never primary control |
| **Human-in-the-Loop** | Interrupt / Pause Node | Approval gating | ⚠️ Recommended | Mandatory for prod changes |
|  | Resume Execution | Continue after approval | ⚠️ Recommended | Preserve state |
| **Memory & Knowledge** | Vector Store Access | Semantic KB | ⚠️ Recommended | Read-only by default |
|  | Graph DB Access | Relationship reasoning | ⚠️ Recommended | SOP → action mapping |
|  | SQL / Audit DB | Execution logs | ✅ Yes | Compliance requirement |
| **Observability** | Node-level Tracing | Debugging | ✅ Yes | Required for RCA |
|  | Execution Metadata | Time, tokens, agent | ⚠️ Recommended | Cost & performance |
| **Security & Safety** | Tool Whitelisting | Prevent misuse | ✅ Yes | Explicit allowlist |
|  | Input Validation (Zod/Pydantic) | Safe execution | ✅ Yes | LLM output never trusted |
|  | RBAC per Agent | Least privilege | ⚠️ Recommended | Prod requirement |
| **Scalability** | Async Execution | Parallel agents | ⚠️ Recommended | Fan-out/fan-in patterns |
|  | Idempotent Tools | Safe retries | ✅ Yes | Mandatory with Temporal |
| **Termination** | Completion Node | Clean exit | ✅ Yes | No dangling execution |
|  | Fallback Node | Safe failure end | ✅ Yes | Never crash silently |
| **Workflow Durability** | Temporal Integration | Long-running workflows | ⚠️ Recommended | Mandatory for prod ops |
| **Testing & Validation** | Dry-run Mode | No side-effects | ⚠️ Recommended | Mandatory before prod |
|  | Golden Path Tests | Known flows | ✅ Yes | Regression safety |


# State
A shared data structure that represents the current snapshot of your application. It can be any data type, but is typically defined using a shared state schema.

- The first thing you do when you define a graph is define the State of the graph.

## Schema
- The main documented way to specify the schema of a graph is by using Zod schemas.
- By default, the graph will have the same **input** and **output** schemas.
- If you want to change this, you can also specify explicit **input** and **output** schemas directly.

## Multiple schemas
- Typically, all graph nodes communicate with a single schema.
- This means that they will read and write to the same state channels.
- But, there are cases where we want more control over this:

    - Internal nodes can pass information that is not required in the graph’s input / output.
    - We may also want to use different input / output schemas for the graph. The output might, for example, only contain a single relevant output key.

- It is possible to have nodes write to private state channels inside the graph for internal node communication. We can simply define a private schema, **PrivateState**.

- It is also possible to define explicit input and output schemas for a graph. In these cases, we define an “internal” schema that contains all keys relevant to graph operations.

**example:**

```
const InputState = z.object({
  userInput: z.string(),
});

const OutputState = z.object({
  graphOutput: z.string(),
});

const OverallState = z.object({
  foo: z.string(),
  userInput: z.string(),
  graphOutput: z.string(),
});

const PrivateState = z.object({
  bar: z.string(),
});

const graph = new StateGraph({
  state: OverallState,
  input: InputState,
  output: OutputState,
})
  .addNode("node1", (state) => {
    // Write to OverallState
    return { foo: state.userInput + " name" };
  })
  .addNode("node2", (state) => {
    // Read from OverallState, write to PrivateState
    return { bar: state.foo + " is" };
  })
  .addNode(
    "node3",
    (state) => {
      // Read from PrivateState, write to OutputState
      return { graphOutput: state.bar + " Lance" };
    },
    { input: PrivateState }
  )
  .addEdge(START, "node1")
  .addEdge("node1", "node2")
  .addEdge("node2", "node3")
  .addEdge("node3", END)
  .compile();

await graph.invoke({ userInput: "My" });
// { graphOutput: 'My name is Lance' }
```

1. We pass **state** as the **input** schema to **node1**.
2. But, we write out to **foo**, a channel in **OverallState**.
3. How can we write out to a **state** channel that is not included in the **input** schema?
4. This is because a node can write to any state channel in the graph state.
5. The graph state is the **union** of the state channels defined at initialization, which includes **OverallState** and the filters **InputState** and **OutputState**.
6. We initialize the graph with **StateGraph({ state: OverallState, input: InputState, output: OutputState })**.
7. So, how can we write to **PrivateState** in **node2**?
8. How does the graph gain access to this schema if it was not passed in the **StateGraph** initialization?
9. We can do this because nodes can also declare additional state channels as long as the state schema definition exists.
10. In this case, the **PrivateState** schema is defined, so we can add **bar** as a new state channel in the graph and write to it.

## Reducers

- Reducers are key to understanding how updates from nodes are applied to the **State**.
- Each key in the State has its own independent reducer function.
- If no reducer function is explicitly specified then it is assumed that all updates to that key should override it.
- There are a few different types of reducers, starting with the default type of reducer:

**Default Reducer**

These two examples show how to use the default reducer:

```
const State = z.object({
  foo: z.number(),
  bar: z.array(z.string()),
});
```

- In this example, no reducer functions are specified for any key. Let’s assume the input to the graph is:

```{ foo: 1, bar: ["hi"] }`` 

- Let’s then assume the first **Node** returns ```{ foo: 2 }```
- This is treated as an update to the state.
- Notice that the **Node** does not need to return the whole **State** schema - just an update.
- After applying this update, the **State** would then be ```{ foo: 2, bar: ["hi"] }```.
- If the **second node** returns ```{ bar: ["bye"] }``` then the **State** would then be ```{ foo: 2, bar: ["bye"] }```.

**Example**

```
import * as z from "zod";
import { registry } from "@langchain/langgraph/zod";

const State = z.object({
  foo: z.number(),
  bar: z.array(z.string()).register(registry, {
    reducer: {
      fn: (x, y) => x.concat(y),
    },
    default: () => [] as string[],
  }),
});
```

- In this example, we’ve used **Zod 4 registries(```foo, bar, fn, default```)** to specify a **reducer** function for the **second key ```(bar)```**.
- Note that the first key **foo** remains unchanged.
- Let’s assume the input to the graph is **```{ foo: 1, bar: ["hi"] }```**.
- Let’s then assume the first Node returns **```{ foo: 2 }```**,
- This is treated as an update to the state.
- Notice that the Node does not need to return the whole **State** schema - just an update.
- After applying this update, the **State** would then be **```{ foo: 2, bar: ["hi"] }```**.
- If the **second node** returns **```{ bar: ["bye"] }```** then the **State** would then be **```{ foo: 2, bar: ["hi", "bye"] }```**.
- Notice here that the bar key is updated by adding the two arrays together.




## Difference Between Zod Schema and Annotation Schema

- **Zod** = Executable runtime validation
- **Annotation schema** = Declarative metadata / hints

## Zod Schema?

Zod is a runtime schema validation library (TypeScript-first).

- Validates data at runtime
- Parses & sanitizes input
- Throws or returns structured errors
- Is executable code 

**What Zod Does**

✔ Ensures data is valid
✔ Blocks bad input
✔ Produces structured errors
✔ Enforces constraints

**Example (Zod)**

```
import { z } from "zod";

const CreateVMRequest = z.object({
  region: z.string(),
  vmSize: z.enum(["small", "medium", "large"]),
  costCenter: z.string().regex(/^CC-\d+$/),
});
```

## What is an Annotation Schema?

An annotation schema is descriptive metadata attached to fields.

- Documents intent
- Guides LLM behavior
- Does NOT enforce anything
- Is not executable validation

**What Annotations Do**

✔ Explain meaning
✔ Improve prompts
✔ Help LLM reasoning
❌ Do NOT block bad data

**Example (Annotation)**

```
{
  "region": {
    "type": "string",
    "description": "Cloud region for VM provisioning",
    "example": "eastus"
  },
  "vmSize": {
    "type": "string",
    "description": "Size of the virtual machine"
  }
}
```

## ❌ Annotation-Only = Dangerous
If your agent relies only on annotations:

```
LLM says:
"vmSize": "super-large-quantum"
```
No system stops it.

## ✅ Zod = Safety Gate
With Zod:

```
Invalid input → Execution stops → Error handled
```

This is non-negotiable in:

- Cloud provisioning
- Auto-remediation
- Financial workflows
- Compliance systems

## How They Work Together (Best Practice)

**Enterprise Pattern**

```
Annotation → LLM understands intent
Zod        → System enforces reality
```

**Example: Agent Tool Definition**

```
const CreateVMInput = z.object({
  region: z.string().describe("Azure region like eastus"),
  vmSize: z.enum(["B2s", "D4s", "E8s"])
    .describe("Allowed VM sizes"),
});
```

Here:

- ```.describe()``` = Annotation
- ```z.object()``` = Enforcement

## In LangGraph / Agentic Frameworks

**Where Zod is Used?**

- Tool inputs
- Agent state
- API contracts
- Human approvals
- Safety checks

**Where Annotations are Used**

- Prompt construction
- Reasoning hints
- Agent instructions
- UI auto-generation

## What is TOON?

**TOON (Typed Object Oriented Notation)** is a **LLM output-constraining schema**.

- Guides LLMs to produce structured JSON
- Reduces hallucinations
- Improves determinism
- Works inside prompts / agent reasoning

⚠️ **TOON does NOT execute validation logic at runtime** like Zod.

**Example (TOON)**

```
{
  "type": "object",
  "properties": {
    "region": {
      "type": "string",
      "description": "Azure region"
    },
    "vmSize": {
      "type": "string",
      "enum": ["B2s", "D4s", "E8s"]
    }
  },
  "required": ["region", "vmSize"]
}
```

**What TOON Does**

✔ Forces structured output
✔ Limits LLM choices
✔ Improves reliability
❌ Does NOT stop invalid runtime values


## What is JSON?

**JSON (JavaScript Object Notation)** is a **data interchange format**.

- Represents structured data
- Is language-agnostic
- Has **NO rules, NO validation, NO meaning by default**

**Example (Pure JSON)**

```
{
  "region": "eastus",
  "vmSize": "D4s"
}
```

**What JSON Does**

✔ Stores data
✔ Transfers data
❌ No validation
❌ No constraints
❌ No semantics

JSON is just a **container**, not a guardrail.

## What is Temporal?

**Temporal** is a **distributed workflow orchestration engine** designed for:

- Long-running workflows
- Exactly-once execution
- Automatic retries
- State durability
- Fault tolerance

**Temporal guarantees that:**

```Your workflow will complete correctly, even if everything crashes.```

**What Temporal Does**

✔ Persists workflow state
✔ Handles crashes & restarts
✔ Supports long waits (days/months)
✔ Handles retries & backoff
✔ Supports signals & timers
✔ Provides full audit history

**Temporal Example (Simplified)**

```
@workflow.defn
class ProvisionVMWorkflow:

    @workflow.run
    async def run(self, request):
        await workflow.execute_activity(validate_input, request)
        await workflow.execute_activity(create_vm, request)
```

**Temporal ensures:**

✔ No double execution
✔ Resume after crash
✔ Full traceability


## What is Pydantic?

**Pydantic** is a **Python runtime schema validation library**.

- Validates data at runtime
- Enforces types
- Converts raw JSON into safe Python objects
- Raises structured validation errors

**What Pydantic Does**

✔ Validates LLM output
✔ Blocks invalid input
✔ Enforces constraints
✔ Provides type-safe objects
✔ Essential for AI agents

**Pydantic Example**

```
from pydantic import BaseModel, Field

class VMRequest(BaseModel):
    region: str = Field(description="Azure region")
    vm_size: str = Field(pattern="^(B2s|D4s|E8s)$")
```

If invalid data arrives:

```
ValidationError → Execution stops
```

## ❌ Without Pydantic

- LLM outputs garbage
- Tools crash
- Security risk

## ❌ Without Temporal

- Workflow dies on crash
- Manual retries
- No audit trail

## Ultimate Comparison Matrix (Executive View)

| Layer               | Preferred Choice  |
| ------------------- | ----------------- |
| Data format         | JSON              |
| Documentation       | Annotation        |
| LLM output shaping  | TOON              |
| Runtime safety      | Zod / Pydantic    |
| Agent state         | LangGraph State   |
| Workflow durability | Temporal          |
| Knowledge           | Vector + Graph DB |

## LLM Prompt Schema vs Hard Schema

| Aspect           | Prompt Schema | Zod/Pydantic |
| ---------------- | ------------- | ------------ |
| Enforced by code | ❌             | ✅            |
| Reliable         | ❌             | ✅            |
| Explainable      | ⚠️            | ✅            |
| Safe for prod    | ❌             | ✅            |

## Vector DB vs Graph DB vs SQL (Knowledge Layer)

| Use Case        | Vector DB | Graph DB | SQL |
| --------------- | --------- | -------- | --- |
| Semantic search | ✅         | ❌        | ❌   |
| Relationships   | ❌         | ✅        | ⚠️  |
| Transactions    | ❌         | ❌        | ✅   |
| Agent memory    | ✅         | ✅        | ⚠️  |


## Temporal vs Cron vs Celery (Workflow Safety)

| Feature           | Temporal | Cron | Celery |
| ----------------- | -------- | ---- | ------ |
| Durable workflows | ✅        | ❌    | ⚠️     |
| Long-running      | ✅        | ❌    | ⚠️     |
| Retry control     | ✅        | ⚠️   | ⚠️     |
| Audit & replay    | ✅        | ❌    | ❌      |
| Agentic AI fit    | ✅        | ❌    | ⚠️     |

## LangGraph State vs Plain JSON State

| Aspect            | LangGraph State | Plain JSON |
| ----------------- | --------------- | ---------- |
| Typed             | ✅               | ❌          |
| Enforced          | ✅               | ❌          |
| State transitions | Controlled      | Free-form  |
| Audit-ready       | ✅               | ❌          |
| Production-safe   | ✅               | ❌          |


## Zod vs Pydantic vs Marshmallow (Runtime Validation)

| Feature            | Zod (TS) | Pydantic (Py) | Marshmallow |
| ------------------ | -------- | ------------- | ----------- |
| Runtime validation | ✅        | ✅             | ✅           |
| LLM friendly       | ❌        | ❌             | ❌           |
| LangGraph fit      | ✅        | ✅             | ⚠️          |
| Typed outputs      | ✅        | ✅             | ⚠️          |
| Best for agents    | ✅        | ✅             | ❌           |

## TOON vs JSON Schema vs OpenAI Function Calling

| Feature               | TOON | JSON Schema | Function Calling |
| --------------------- | ---- | ----------- | ---------------- |
| LLM structured output | ✅    | ✅           | ✅                |
| Enum constraints      | ✅    | ✅           | ✅                |
| Runtime enforcement   | ❌    | ❌           | ❌                |
| Prompt-native         | ✅    | ⚠️          | ❌                |
| Vendor lock-in        | ❌    | ❌           | ⚠️               |

## Annotation vs OpenAPI vs Swagger

| Aspect             | Annotation     | OpenAPI      | Swagger   |
| ------------------ | -------------- | ------------ | --------- |
| Purpose            | Explain fields | API contract | UI + docs |
| Runtime validation | ❌              | ⚠️           | ❌         |
| LLM usability      | ✅              | ⚠️           | ❌         |
| Tool generation    | ❌              | ✅            | ✅         |
| Enterprise APIs    | ❌              | ✅            | ✅         |


## Full Comparison Table

| Category                        | JSON        | Annotation     | TOON           | JSON Schema    | OpenAI Function Calling | Zod                     | Pydantic                | LangGraph State     | Temporal         |
| ------------------------------- | ----------- | -------------- | -------------- | -------------- | ----------------------- | ----------------------- | ----------------------- | ------------------- | ---------------- |
| **Primary Role**                | Data format | Meaning / Docs | LLM constraint | Structure spec | LLM tool binding        | Runtime validation (TS) | Runtime validation (Py) | Agent state control | Durable workflow |
| **Executable**                  | ❌           | ❌              | ❌              | ❌              | ❌                       | ✅                       | ✅                       | ✅                   | ✅                |
| **Validates at runtime**        | ❌           | ❌              | ❌              | ❌              | ❌                       | ✅                       | ✅                       | ⚠️                  | ❌                |
| **Blocks bad execution**        | ❌           | ❌              | ❌              | ❌              | ❌                       | ✅                       | ✅                       | ⚠️                  | ❌                |
| **Guides LLM output**           | ❌           | ✅              | ✅              | ⚠️             | ✅                       | ❌                       | ❌                       | ❌                   | ❌                |
| **Prevents hallucination**      | ❌           | ❌              | ⚠️             | ⚠️             | ⚠️                      | ✅                       | ✅                       | ⚠️                  | ❌                |
| **LLM-native**                  | ❌           | ⚠️             | ✅              | ⚠️             | ✅                       | ❌                       | ❌                       | ❌                   | ❌                |
| **Type-safe objects**           | ❌           | ❌              | ❌              | ❌              | ❌                       | ✅                       | ✅                       | ✅                   | ❌                |
| **Agent state safety**          | ❌           | ❌              | ❌              | ❌              | ❌                       | ✅                       | ✅                       | ✅                   | ❌                |
| **Tool input safety**           | ❌           | ❌              | ⚠️             | ⚠️             | ⚠️                      | ✅                       | ✅                       | ❌                   | ❌                |
| **Long-running workflows**      | ❌           | ❌              | ❌              | ❌              | ❌                       | ❌                       | ❌                       | ⚠️                  | ✅                |
| **Crash recovery**              | ❌           | ❌              | ❌              | ❌              | ❌                       | ❌                       | ❌                       | ❌                   | ✅                |
| **Retries & backoff**           | ❌           | ❌              | ❌              | ❌              | ❌                       | ❌                       | ❌                       | ❌                   | ✅                |
| **Human-in-loop support**       | ❌           | ❌              | ❌              | ❌              | ⚠️                      | ❌                       | ❌                       | ✅                   | ✅                |
| **Audit & replay**              | ❌           | ❌              | ❌              | ❌              | ❌                       | ❌                       | ❌                       | ⚠️                  | ✅                |
| **Security-critical safe**      | ❌           | ❌              | ❌              | ❌              | ❌                       | ✅                       | ✅                       | ⚠️                  | ✅                |
| **Enterprise production-ready** | ❌           | ❌              | ⚠️             | ⚠️             | ⚠️                      | ✅                       | ✅                       | ✅                   | ✅                |


## Messages in Graph State

**Why use messages?**

- Most modern LLM providers have a chat model interface that accepts a list of messages as input.
- LangChain’s **chat model interface** in particular accepts a list of message objects as inputs.
- These messages come in a variety of forms such as **HumanMessage** ```(user input)``` or **AIMessage** ```(LLM response)```.

**Using Messages in your Graph**

- In many cases, it is helpful to ```store prior conversation history``` as a ```list of messages``` in your ```graph state```.
- To do so, we can add a ```key (channel)``` to the ```graph state``` that stores a ```list of Message objects and annotate``` it with a ```reducer``` function.
- The reducer function is vital to telling the graph how to update the list of Message objects in the state with each state update (for example, when a node sends an update).
- If you don’t specify a reducer, every state update will overwrite the list of messages with the most recently provided value.
-  If you wanted to simply append messages to the existing list, you could use a function that concatenates arrays as a reducer.

- However, you might also want to manually update messages in your graph state (e.g. human-in-the-loop).
- If you were to use a simple concatenation function, the manual state updates you send to the graph would be appended to the existing list of messages, instead of updating existing messages. 
- To avoid that, you need a reducer that can keep track of message IDs and overwrite existing messages, if updated. To achieve this, you can use the prebuilt **messagesStateReducer** function or **MessagesZodMeta** when state schema is defined with Zod.
- For brand new messages, it will simply append to existing list, but it will also handle the updates for existing messages correctly.

## Serialization
In addition to keeping track of message IDs, **MessagesZodMeta** will also try to deserialize messages into LangChain **Message** objects whenever a state update is received on the **messages** channel. This allows sending graph inputs / state updates in the following format:

```
// this is supported
{
  messages: [new HumanMessage("message")];
}

// and this is also supported
{
  messages: [{ role: "human", content: "message" }];
}
```

Since the state updates are always deserialized into LangChain **Messages** when using **MessagesZodMeta**, you should use dot notation to access message attributes, like **```state.messages[state.messages.length - 1].content```**. Below is an example of a graph that uses **MessagesZodMeta**:

```
import { StateGraph, MessagesZodMeta } from "@langchain/langgraph";
import { registry } from "@langchain/langgraph/zod";
import * as z from "zod";

const MessagesZodState = z.object({
  messages: z
    .array(z.custom<BaseMessage>())
    .register(registry, MessagesZodMeta),
});

const graph = new StateGraph(MessagesZodState)
  ...
```

- **MessagesZodState** is defined with a single **messages** key which is a list of **BaseMessage** objects and uses the appropriate reducer.
- Typically, there is more state to track than just messages, so we see people extend this state and add more fields, like:

```
const State = z.object({
  messages: z
    .array(z.custom<BaseMessage>())
    .register(registry, MessagesZodMeta),
  documents: z.array(z.string()),
});
```

## Nodes

In **LangGraph**, **nodes** are typically ```functions (sync or async)``` that accept the following arguments:

1. **state** – The **state** of the **graph**
2. **config** – A **RunnableConfig** object that contains configuration information like **thread_id** and tracing information like **tags**

- You can **add** **nodes** to a **graph** using the **addNode** method.

```
import { StateGraph } from "@langchain/langgraph";
import { RunnableConfig } from "@langchain/core/runnables";
import * as z from "zod";

const State = z.object({
  input: z.string(),
  results: z.string(),
});

const builder = new StateGraph(State);
  .addNode("myNode", (state, config) => {
    console.log("In node: ", config?.configurable?.user_id);
    return { results: `Hello, ${state.input}!` };
  })
  addNode("otherNode", (state) => {
    return state;
  })
  ...
```

- Behind the scenes, functions are converted to **RunnableLambda**, which add batch and async support to your function, along with native tracing and debugging.
- If you add a node to a graph without specifying a name, it will be given a default name equivalent to the function name.

```
builder.addNode(myNode);
// You can then create edges to/from this node by referencing it as `"myNode"`
```

## START Node

The **START Node** is a special node that represents the node that sends user input to the graph. The main purpose for referencing this node is to determine which nodes should be called first.

```
import { START } from "@langchain/langgraph";

graph.addEdge(START, "nodeA");
```

## END Node

The **END Node** is a special node that represents a terminal node. This node is referenced when you want to denote which edges have no actions after they are done.

```
import { END } from "@langchain/langgraph";

graph.addEdge("nodeA", END);
```

## Node Caching

LangGraph supports **caching** of tasks/nodes based on the input to the node. To use caching:

- Specify a ```cache``` when **compiling** a graph (or specifying an ```entrypoint```)
- Specify a cache policy for nodes. Each cache policy supports:
- **keyFunc**, which is used to generate a cache key based on the input to a node.
- **ttl**, the time to live for the cache in seconds. If not specified, the cache will never expire.

```
import { StateGraph, MessagesZodMeta } from "@langchain/langgraph";
import { registry } from "@langchain/langgraph/zod";
import * as z from "zod";
import { InMemoryCache } from "@langchain/langgraph-checkpoint";

const MessagesZodState = z.object({
  messages: z
    .array(z.custom<BaseMessage>())
    .register(registry, MessagesZodMeta),
});

const graph = new StateGraph(MessagesZodState)
  .addNode(
    "expensive_node",
    async () => {
      // Simulate an expensive operation
      await new Promise((resolve) => setTimeout(resolve, 3000));
      return { result: 10 };
    },
    { cachePolicy: { ttl: 3 } }
  )
  .addEdge(START, "expensive_node")
  .compile({ cache: new InMemoryCache() });

await graph.invoke({ x: 5 }, { streamMode: "updates" });   
// [{"expensive_node": {"result": 10}}]
await graph.invoke({ x: 5 }, { streamMode: "updates" });   
// [{"expensive_node": {"result": 10}, "__metadata__": {"cached": true}}]
```

## Edges

- Edges define how the logic is routed and how the graph decides to stop. 
- This is a big part of how your agents work and how different nodes communicate with each other. There are a few key types of edges:

1. **Normal Edges:** Go directly from one node to the next.
2. **Conditional Edges:** Call a function to determine which node(s) to go to next.
3. **Entry Point:** Which node to call first when user input arrives.
4. **Conditional Entry Point:** Call a function to determine which node(s) to call first when user input arrives.

A node can have multiple outgoing edges. If a node has multiple outgoing edges, all of those destination nodes will be executed in parallel


**Normal Edges**

- If you always want to go from node A to node B, you can use the addEdge method directly.

```graph.addEdge("nodeA", "nodeB");```


**Conditional Edges**

If you want to **optionally** route to one or more edges (or optionally terminate), you can use the **addConditionalEdges** method. This method accepts the name of a node and a “routing function” to call after that node is executed:

```graph.addConditionalEdges("nodeA", routingFunction);```

Similar to nodes, the **routingFunction** accepts the current **state** of the graph and returns a value.

By default, the return value **routingFunction** is used as the name of the node (or list of nodes) to send the state to next. All those nodes will be run in parallel as a part of the next superstep.

You can optionally provide an object that maps the **routingFunction’s** output to the name of the next node.

```
graph.addConditionalEdges("nodeA", routingFunction, {
  true: "nodeB",
  false: "nodeC",
});
```

## Entry point

The entry point is the first node(s) that are run when the graph starts. You can use the **addEdge** method from the virtual START node to the first node to execute to specify where to enter the graph.

```
import { START } from "@langchain/langgraph";

graph.addEdge(START, "nodeA");
```

## Conditional entry point

A conditional entry point lets you start at different nodes depending on custom logic. You can use **addConditionalEdges** from the virtual **START** node to accomplish this.

```
import { START } from "@langchain/langgraph";

graph.addConditionalEdges(START, routingFunction);
```

You can optionally provide an object that maps the **routingFunction’s** output to the name of the next node.

```
graph.addConditionalEdges(START, routingFunction, {
  true: "nodeB",
  false: "nodeC",
});
```

## Send
By default, **Nodes** and **Edges** are defined ahead of time and operate on the same shared state. However, there can be cases where the exact edges are not known ahead of time and/or you may want different versions of **State** to exist at the same time. A common example of this is with map-reduce design patterns. In this design pattern, a first node may generate a list of objects, and you may want to apply some other node to all those objects. The number of objects may be unknown ahead of time (meaning the number of edges may not be known) and the input **State** to the downstream Node should be different (one for each generated object).

To support this design pattern, LangGraph supports returning **Send** objects from conditional edges. **Send** takes two arguments: first is the name of the node, and second is the state to pass to that node.

```
import { Send } from "@langchain/langgraph";

graph.addConditionalEdges("nodeA", (state) => {
  return state.subjects.map((subject) => new Send("generateJoke", { subject }));
});
```

## Command
It can be useful to combine control flow (edges) and state updates (nodes). For example, you might want to BOTH perform state updates AND decide which node to go to next in the SAME node. LangGraph provides a way to do so by returning a **Command** object from node functions:

```
import { Command } from "@langchain/langgraph";

graph.addNode("myNode", (state) => {
  return new Command({
    update: { foo: "bar" },
    goto: "myOtherNode",
  });
});
```

With Command you can also achieve **dynamic** control flow behavior (identical to **conditional edges**):

```
import { Command } from "@langchain/langgraph";

graph.addNode("myNode", (state) => {
  if (state.foo === "bar") {
    return new Command({
      update: { foo: "baz" },
      goto: "myOtherNode",
    });
  }
});
```

When using **Command** in your node functions, you must add the ends parameter when adding the node to specify which nodes it can route to:

```
builder.addNode("myNode", myNode, {
  ends: ["myOtherNode", END],
});
```

## When should I use Command instead of conditional edges?

- Use **Command** when you need to **both** update the graph state **and** route to a different node.
- For example, when implementing **multi-agent handoffs** where it’s important to route to a different agent and pass some information to that agent.
- Use **conditional edges** to route between nodes conditionally without updating the state.

## Navigating to a node in a parent graph
If you are using **subgraphs**, you might want to navigate from a node within a subgraph to a different subgraph (i.e. a different node in the parent graph). To do so, you can specify **graph: Command.PARENT** in **Command**:

```
import { Command } from "@langchain/langgraph";

graph.addNode("myNode", (state) => {
  return new Command({
    update: { foo: "bar" },
    goto: "otherSubgraph", // where `otherSubgraph` is a node in the parent graph
    graph: Command.PARENT,
  });
});
```

Setting graph to ```Command.PARENT``` will navigate to the closest parent graph.
When you send updates from a subgraph node to a parent graph node for a key that’s shared by both parent and subgraph ```state schemas```, you must define a ```reducer``` for the key you’re updating in the parent graph state.

- This is particularly useful when implementing **multi-agent handoffs**.

## Using inside tools
A common use case is updating graph state from inside a tool. For example, in a customer support application you might want to look up customer information based on their account number or ID in the beginning of the conversation.

## Human-in-the-loop
Command is an important part of human-in-the-loop workflows: when using ```interrupt()``` to collect user input, Command is then used to supply the input and resume execution via new ```Command({ resume: "User input" })```

## Graph migrations
LangGraph can easily handle migrations of graph definitions (nodes, edges, and state) even when using a checkpointer to track state.

## Runtime context
- When creating a graph, you can specify a **contextSchema** for runtime context passed to nodes. 
- This is useful for passing information to nodes that is not part of the graph state. For example, you might want to pass dependencies such as model name or a database connection.

```
import * as z from "zod";

const ContextSchema = z.object({
  llm: z.union([z.literal("openai"), z.literal("anthropic")]),
});

const graph = new StateGraph(State, ContextSchema);
```

You can then pass this configuration into the graph using the context property.

```
const config = { context: { llm: "anthropic" } };

await graph.invoke(inputs, config);
```

You can then access and use this context inside a node or conditional edge:

```
import { Runtime } from "@langchain/langgraph";
import * as z from "zod";

const nodeA = (
  state: z.infer<typeof State>,
  runtime: Runtime<z.infer<typeof ContextSchema>>,
) => {
  const llm = getLLM(runtime.context?.llm);
  // ...
};
```

```
graph.addNode("myNode", (state, runtime) => {
  const llmType = runtime.context?.llm || "openai";
  const llm = getLLM(llmType);
  return { results: `Hello, ${state.input}!` };
});
```

## Recursion limit

- The recursion limit sets the maximum number of **super-steps** the graph can execute during a single execution.
- Once the limit is reached, LangGraph will raise **GraphRecursionError**.
- By default this value is set to 25 steps.
- The recursion limit can be set on any graph at runtime, and is passed to ```invoke/stream``` via the config object. 
- Importantly, ```recursionLimit``` is a standalone ```config``` key and should not be passed inside the ```configurable``` key as all other user-defined configuration. See the example below:

```
await graph.invoke(inputs, {
  recursionLimit: 5,
  context: { llm: "anthropic" },
});
```

## Accessing and handling the recursion counter
The current step counter is accessible in config.metadata.langgraph_step within any node, allowing for proactive recursion handling before hitting the recursion limit. This enables you to implement graceful degradation strategies within your graph logic.

**How it works**

The step counter is stored in ```config.metadata.langgraph_step```. The recursion limit check follows the logic: ```step > stop``` where ```stop = step + recursionLimit + 1```. When the limit is exceeded, LangGraph raises a ```GraphRecursionError```.

## Accessing the current step counter
You can access the current step counter within any node to monitor execution progress.

```
import { RunnableConfig } from "@langchain/core/runnables";
import { StateGraph } from "@langchain/langgraph";

async function myNode(state: any, config: RunnableConfig): Promise<any> {
  const currentStep = config.metadata?.langgraph_step;
  console.log(`Currently on step: ${currentStep}`);
  return state;
}
```

## Proactive recursion handling

You can check the step counter and proactively route to a different node before hitting the limit. This allows for graceful degradation within your graph.

```
import { RunnableConfig } from "@langchain/core/runnables";
import { StateGraph, END } from "@langchain/langgraph";

interface State {
  messages: string[];
  route_to?: string;
  reason?: string;
  done?: boolean;
}

async function reasoningNode(
  state: State,
  config: RunnableConfig
): Promise<Partial<State>> {
  const currentStep = config.metadata?.langgraph_step ?? 0;
  const recursionLimit = config.recursionLimit!; // always present, defaults to 25

  // Check if we're approaching the limit (e.g., 80% threshold)
  if (currentStep >= recursionLimit * 0.8) {
    return {
      ...state,
      route_to: "fallback",
      reason: "Approaching recursion limit"
    };
  }

  // Normal processing
  return {
    messages: [...state.messages, "thinking..."]
  };
}

async function fallbackNode(
  state: State,
  config: RunnableConfig
): Promise<Partial<State>> {
  return {
    ...state,
    messages: [
      ...state.messages,
      "Reached complexity limit, providing best effort answer"
    ]
  };
}

function routeBasedOnState(state: State): string {
  if (state.route_to === "fallback") {
    return "fallback";
  } else if (state.done) {
    return END;
  }
  return "reasoning";
}

// Build graph
const graph = new StateGraph<State>({ channels: {} })
  .addNode("reasoning", reasoningNode)
  .addNode("fallback", fallbackNode)
  .addConditionalEdges("reasoning", routeBasedOnState)
  .addEdge("fallback", END);

const app = graph.compile();
```

## Proactive vs reactive approaches
There are two main approaches to handling recursion limits: proactive (monitoring within the graph) and reactive (catching errors externally).

```
import { RunnableConfig } from "@langchain/core/runnables";
import { StateGraph, END } from "@langchain/langgraph";
import { GraphRecursionError } from "@langchain/langgraph";

interface State {
  messages: string[];
  status?: string;
  final_answer?: string;
}

// Proactive Approach (recommended)
async function agentWithMonitoring(
  state: State,
  config: RunnableConfig
): Promise<Partial<State>> {
  const currentStep = config.metadata?.langgraph_step ?? 0;
  const recursionLimit = config.recursionLimit!;

  // Early detection - route to internal handling
  if (currentStep >= recursionLimit - 2) { // 2 steps before limit
    return {
      ...state,
      status: "recursion_limit_approaching",
      final_answer: "Reached iteration limit, returning partial result"
    };
  }

  // Normal processing
  return {
    messages: [...state.messages, `Step ${currentStep}`]
  };
}

// Reactive Approach (fallback)
try {
  const result = await graph.invoke(initialState, { recursionLimit: 10 });
} catch (error) {
  if (error instanceof GraphRecursionError) {
    // Handle externally after graph execution fails
    const result = await fallbackHandler(initialState);
  }
}
```

The key differences between these approaches are:

| Approach | Detection | Handling | Control Flow |
|--------|-----------|----------|--------------|
| **Proactive (using `langgraph_step`)** | Before limit reached | Inside graph via conditional routing | Graph continues to completion node |
| **Reactive (catching `GraphRecursionError`)** | After limit exceeded | Outside graph in `try/catch` | Graph execution terminated |


**Proactive advantages:**

- Graceful degradation within the graph
- Can save intermediate state in checkpoints
- Better user experience with partial results
- Graph completes normally (no exception)

**Reactive advantages:**

- Simpler implementation
- No need to modify graph logic
- Centralized error handling

## Other available metadata

Along with ```langgraph_step```, the following metadata is also available in ```config.metadata```:

```
async function inspectMetadata(
  state: any,
  config: RunnableConfig
): Promise<any> {
  const metadata = config.metadata;

  console.log(`Step: ${metadata?.langgraph_step}`);
  console.log(`Node: ${metadata?.langgraph_node}`);
  console.log(`Triggers: ${metadata?.langgraph_triggers}`);
  console.log(`Path: ${metadata?.langgraph_path}`);
  console.log(`Checkpoint NS: ${metadata?.langgraph_checkpoint_ns}`);

  return state;
}
```

## Visualization
It’s often nice to be able to visualize graphs, especially as they get more complex. LangGraph comes with several built-in ways to visualize graphs.

Here we demonstrate how to visualize the graphs you create.

You can visualize any arbitrary **Graph**, including **StateGraph**.

Let’s create a simple example graph to demonstrate visualization.

```
import { StateGraph, START, END, MessagesZodMeta } from "@langchain/langgraph";
import { BaseMessage } from "@langchain/core/messages";
import { registry } from "@langchain/langgraph/zod";
import * as z from "zod";

const State = z.object({
  messages: z
    .array(z.custom<BaseMessage>())
    .register(registry, MessagesZodMeta),
  value: z.number().register(registry, {
    reducer: {
      fn: (x, y) => x + y,
    },
  }),
});

const app = new StateGraph(State)
  .addNode("node1", (state) => {
    return { value: state.value + 1 };
  })
  .addNode("node2", (state) => {
    return { value: state.value * 2 };
  })
  .addEdge(START, "node1")
  .addConditionalEdges("node1", (state) => {
    if (state.value < 10) {
      return "node2";
    }
    return END;
  })
  .addEdge("node2", "node1")
  .compile();
  ```

  ## Mermaid
  We can also convert a graph class into Mermaid syntax.

  ```
  const drawableGraph = await app.getGraphAsync();
console.log(drawableGraph.drawMermaid());
```

```
%%{init: {'flowchart': {'curve': 'linear'}}}%%
graph TD;
    tart__([<p>__start__</p>]):::first
    e1(node1)
    e2(node2)
    nd__([<p>__end__</p>]):::last
    tart__ --> node1;
    e1 -.-> node2;
    e1 -.-> __end__;
    e2 --> node1;
    ssDef default fill:#f2f0ff,line-height:1.2
    ssDef first fill-opacity:0
    ssDef last fill:#bfb6fc
```

## PNG
If preferred, we could render the Graph into a .png. This uses the Mermaid.ink API to generate the diagram.

```
import * as fs from "node:fs/promises";

const drawableGraph = await app.getGraphAsync();
const image = await drawableGraph.drawMermaidPng();
const imageBuffer = new Uint8Array(await image.arrayBuffer());

await fs.writeFile("graph.png", imageBuffer);
```

## Functional API vs Graph API

| Aspect | Functional API | Graph API |
|------|----------------|-----------|
| **Programming Model** | Linear function calls | Node–edge graph execution |
| **Control Flow** | Hard-coded in call order | Explicit, visualized flow |
| **Execution Style** | Sequential | Conditional, looping, parallel |
| **State Management** | Passed manually between functions | Shared, managed graph state |
| **Dynamic Routing** | Difficult (if/else in code) | Native via conditional edges |
| **Loops / Recursion** | Manual, error-prone | First-class support |
| **Parallelism** | Limited, manual | Native (fan-out / fan-in) |
| **Error Handling** | try/except per function | Dedicated failure edges |
| **Human-in-Loop** | Hard to pause/resume | Native pause, resume, approve |
| **Observability** | Logs scattered | Node-level tracing |
| **Reusability** | Function-level reuse | Subgraphs & reusable patterns |
| **Scalability** | Code complexity grows fast | Graph scales cleanly |
| **Determinism** | Implicit | Explicit and auditable |
| **Visualization** | ❌ Not visible | ✅ Graph is inspectable |
| **Agentic AI Fit** | ⚠️ Simple agents only | ✅ Multi-agent workflows |
| **Best Use Cases** | Simple pipelines, utilities | Agent orchestration, workflows |


