# Agentic AI Design Patterns

## 1️⃣ Single-Agent Patterns (Foundation)
![alt text](./image/dp1.png)

## 2️⃣ Planning & Reasoning Patterns
![alt text](./image/dp2.png)

## 3️⃣ Multi-Agent Collaboration Patterns
![alt text](./image/dp3.png)

## 4️⃣ Control & Orchestration Patterns
![alt text](./image/dp4.png)

## 5️⃣ Autonomy & Safety Patterns
![alt text](./image/dp5.png)

## 6️⃣ Learning & Adaptation Patterns
![alt text](./image/dp6.png)

## 7️⃣ Knowledge & Discovery Patterns
![alt text](./image/dp7.png)

## 8️⃣ Cost & Performance Patterns
![alt text](./image/dp8.png)

## 9️⃣ Observability & Reliability Patterns
![alt text](./image/dp9.png)

## 🔟 Enterprise & Platform Patterns
![alt text](./image/dp10.png)


## Mapping Agentic AI Patterns to LangGraph vs CrewAI

## 🧩 Pattern → Framework Mapping Table


| Pattern              | LangGraph          | CrewAI       | Notes                           |
| -------------------- | ------------------ | ------------ | ------------------------------- |
| Reactive Agent       | ✅ Node             | ✅ Agent      | Simple Q&A                      |
| Tool-Using Agent     | ✅ Tool Node        | ✅ Tools      | MCP fits both                   |
| ReAct                | ✅ Native           | ⚠️ Partial   | LangGraph better control        |
| RAG Agent            | ✅ Native           | ✅ Native     | Both strong                     |
| Stateful Agent       | ✅ Native State     | ⚠️ Limited   | LangGraph excels                |
| Planner–Executor     | ✅ Best Fit         | ⚠️ Manual    | LangGraph designed for this     |
| Tree-of-Thought      | ✅ Supported        | ❌ Not native | Needs graph branching           |
| Graph-of-Thought     | ✅ Native           | ❌ No         | LangGraph exclusive             |
| Manager–Worker       | ✅ Supervisor Graph | ✅ Crew       | Both strong                     |
| Specialist Swarm     | ✅ Nodes            | ✅ Agents     | CrewAI very natural             |
| Debate / Consensus   | ✅ Graph            | ✅ Crew       | CrewAI simpler                  |
| Critic–Generator     | ✅ Graph            | ✅ Crew       | Both good                       |
| Event-Driven Agents  | ✅ Excellent        | ❌ Limited    | LangGraph preferred             |
| Policy-Driven Flow   | ✅ Native           | ⚠️ External  | LangGraph integrates governance |
| Human-in-the-Loop    | ✅ Native           | ⚠️ Manual    | LangGraph safer                 |
| Auto-Remediation     | ✅ Best             | ⚠️ Risky     | Needs guardrails                |
| Registry & Discovery | ✅ Native           | ⚠️ External  | LangGraph aligns with A2A       |
| Observability-First  | ✅ Built-in         | ❌ Limited    | LangGraph enterprise ready      |


## Recommended Patterns per Use Case

## 🏦 Enterprise / Banking / Regulated Systems

| Use Case         | Recommended Patterns                                  | Framework |
| ---------------- | ----------------------------------------------------- | --------- |
| Auto-Remediation | Planner–Executor, SOP-Driven, Policy-Controlled, HITL | LangGraph |
| Incident RCA     | Specialist Swarm, Graph-of-Thought, RAG               | LangGraph |
| Compliance QA    | RAG, Governance-Driven                                | LangGraph |
| Audit Workflows  | Trace-First, Event-Driven                             | LangGraph |



## 🧠 Knowledge & Productivity

| Use Case               | Recommended Patterns     | Framework |
| ---------------------- | ------------------------ | --------- |
| Document Summarization | RAG, Critic–Generator    | CrewAI    |
| Research Assistant     | Debate, Specialist Swarm | CrewAI    |
| SOP Search             | Hybrid Discovery, RAG    | Either    |
| Q&A Bot                | Reactive, Tool-Using     | Either    |



## ⚙️ DevOps / Platform Engineering

| Use Case                | Recommended Patterns        | Framework |
| ----------------------- | --------------------------- | --------- |
| CI/CD Automation        | Event-Driven, State Machine | LangGraph |
| Cloud Provisioning      | Planner–Executor            | LangGraph |
| Infra Cost Optimization | Cost-Aware Routing          | LangGraph |



## 🧠 Innovation / POCs

| Use Case        | Recommended Patterns | Framework |
| --------------- | -------------------- | --------- |
| Idea Generation | Swarm, Debate        | CrewAI    |
| Brainstorming   | Peer-to-Peer         | CrewAI    |
| Hackathon Bots  | Minimal Agents       | CrewAI    |



## 🧪 LangGraph vs CrewAI (One-Slide Answer)

| Dimension        | LangGraph           | CrewAI          |
| ---------------- | ------------------- | --------------- |
| Control Flow     | Deterministic Graph | Sequential      |
| Governance       | Strong              | Weak            |
| State Management | Native              | Limited         |
| Multi-Agent      | Graph-based         | Role-based      |
| Safety           | High                | Medium          |
| Production Ready | ✅ Yes               | ⚠️ Partial      |
| Best For         | Enterprise AI       | Reasoning Teams |



## Anti-Patterns to Avoid (CRITICAL) 🚨

## ❌ Agentic AI Anti-Patterns

| Anti-Pattern               | Why It’s Dangerous      |
| -------------------------- | ----------------------- |
| Single mega-agent          | No control, no audit    |
| No governance              | Compliance failure      |
| Unbounded autonomy         | Production risk         |
| No observability           | Silent failures         |
| Tool access without policy | Security breach         |
| No fallback                | Infinite loops          |
| No versioning              | Irreproducible behavior |
| Prompt-only logic          | Fragile systems         |
| No cost controls           | Budget explosion        |
| Direct prod execution      | Catastrophic failures   |


## Agentic AI Patterns list

| #  | Pattern Name                | Category             | Purpose                          | When Used                 |
| -- | --------------------------- | -------------------- | -------------------------------- | ------------------------- |
| 1  | Chain-of-Thought (CoT)      | Planning & Reasoning | Step-by-step reasoning           | Complex analysis, RCA     |
| 2  | Tree-of-Thoughts (ToT)      | Planning & Reasoning | Explore multiple solution paths  | High-uncertainty problems |
| 3  | Graph-of-Thoughts (GoT)     | Planning & Reasoning | Reason over dependency graphs    | Service dependency RCA    |
| 4  | ReAct (Reason+Act)          | Planning & Reasoning | Alternate thinking and tool use  | Investigation workflows   |
| 5  | Plan–Act–Reflect            | Planning & Reasoning | Iterative improvement loop       | Autonomous remediation    |
| 6  | Reflexion                   | Planning & Reasoning | Self-critique and retry          | Low-confidence outputs    |
| 7  | Hypothesis Testing          | Planning & Reasoning | Validate multiple root causes    | Incident diagnosis        |
| 8  | Goal Decomposition          | Planning & Reasoning | Break into sub-tasks             | Multi-step automation     |
| 9  | Constraint-Aware Planning   | Planning & Reasoning | Respect policy/cost/risk limits  | Prod-safe automation      |
| 10 | Orchestrator–Worker         | Multi-Agent          | Central planner with specialists | Enterprise workflows      |
| 11 | Planner–Executor            | Multi-Agent          | Plan first, then execute         | Deterministic flows       |
| 12 | Critic–Generator            | Multi-Agent          | Validate generated outputs       | Change safety checks      |
| 13 | Debate Pattern              | Multi-Agent          | Competing solutions selection    | High-risk decisions       |
| 14 | Specialist Swarm            | Multi-Agent          | Domain agents collaborate        | Network/DB/Cloud RCA      |
| 15 | Hierarchical Agents         | Multi-Agent          | Manager → team → tools           | Large-scale systems       |
| 16 | Blackboard                  | Multi-Agent          | Shared working memory            | Cross-agent context       |
| 17 | Peer-to-Peer Agents         | Multi-Agent          | Direct negotiation               | Decentralized systems     |
| 18 | Workflow Graph              | Orchestration        | Stateful branching workflows     | LangGraph pipelines       |
| 19 | Event-Driven Agents         | Orchestration        | Trigger on alerts/events         | Monitoring, scaling       |
| 20 | Saga Pattern                | Orchestration        | Multi-step with rollback         | Patch, infra changes      |
| 21 | Checkpoint & Resume         | Orchestration        | Persist state across failures    | Long-running tasks        |
| 22 | Human Approval Gate         | Orchestration        | Pause for human review           | High-risk actions         |
| 23 | Policy-Based Routing        | Orchestration        | Route by risk/complexity         | Complexity router         |
| 24 | Circuit Breaker             | Orchestration        | Stop runaway loops               | Tool safety               |
| 25 | Guardrails                  | Safety               | Pre/post validation              | All prod actions          |
| 26 | Risk-Tiered Autonomy        | Safety               | Auto vs human control            | SRE automation            |
| 27 | Tool Permission Scoping     | Safety               | Limit tool access                | Security control          |
| 28 | Simulation/Dry Run          | Safety               | Test before execution            | Infra changes             |
| 29 | Confidence Thresholding     | Safety               | Execute above score              | Auto-remediation          |
| 30 | Kill Switch                 | Safety               | Emergency stop                   | Unsafe behavior           |
| 31 | RAG                         | Knowledge            | Retrieve docs/runbooks           | Known fixes               |
| 32 | Knowledge Graph Reasoning   | Knowledge            | Dependency + blast radius        | Impact analysis           |
| 33 | Semantic Memory             | Knowledge            | Store past learnings             | Repeated incidents        |
| 34 | Episodic Memory             | Knowledge            | Store execution history          | Auditing, learning        |
| 35 | Tool Discovery              | Knowledge            | Dynamic tool lookup              | Plugin ecosystems         |
| 36 | Context Optimization        | Knowledge            | Load relevant context only       | Token reduction           |
| 37 | Toolformer                  | Tool Usage           | LLM decides tool calls           | Flexible workflows        |
| 38 | Function Calling            | Tool Usage           | Structured API execution         | Deterministic actions     |
| 39 | Tool Chaining               | Tool Usage           | Multi-tool pipelines             | Diagnostics → fix         |
| 40 | Parallel Tool Execution     | Tool Usage           | Run tools concurrently           | Faster RCA                |
| 41 | Fallback Tool Strategy      | Tool Usage           | Alternate tool on failure        | Resilience                |
| 42 | Model Routing               | Cost & Performance   | Cheap vs powerful model          | Cost optimization         |
| 43 | Token Budgeting             | Cost & Performance   | Limit reasoning depth            | FinOps control            |
| 44 | Caching/Memoization         | Cost & Performance   | Reuse prior results              | Repeated tasks            |
| 45 | Batch Inference             | Cost & Performance   | Process tasks together           | High-volume alerts        |
| 46 | Early Exit                  | Cost & Performance   | Stop when confident              | Low-complexity cases      |
| 47 | Selective Reasoning         | Cost & Performance   | Use ToT only if needed           | Cost + latency            |
| 48 | Agent Telemetry             | Observability        | Track decisions/tools            | Performance monitoring    |
| 49 | Traceable Reasoning Logs    | Observability        | Full audit trail                 | Compliance                |
| 50 | Outcome-Based KPIs          | Observability        | Measure MTTR, success            | Value tracking            |
| 51 | Feedback Learning Loop      | Observability        | Improve from outcomes            | Continuous tuning         |
| 52 | Drift Detection             | Observability        | Detect performance decay         | Model health              |
| 53 | Multi-Tenant Isolation      | Enterprise           | Tenant-specific memory/policy    | SaaS platforms            |
| 54 | RBAC/ABAC Enforcement       | Enterprise           | Role-based access                | Governance                |
| 55 | Policy-as-Code              | Enterprise           | Centralized control              | Compliance                |
| 56 | Compliance Evidence Gen     | Enterprise           | Auto audit logs                  | Regulated env             |
| 57 | FinOps Cost Tracking        | Enterprise           | Cost per task/agent              | Budget control            |
| 58 | Registry & Discovery        | Enterprise           | Catalog agents/tools             | Platform scale            |
| 59 | Reinforcement Learning Loop | Learning             | Improve via feedback             | Optimization              |
| 60 | Runbook Mining              | Learning             | Convert manual → auto            | SRE automation            |
| 61 | Continuous Evaluation       | Learning             | Shadow → canary → prod           | Safe rollout              |
| 62 | Meta-Agent Optimization     | Learning             | Agents tuning agents             | Platform efficiency       |
| 63 | Working Memory              | Memory               | Session context                  | Active task state         |
| 64 | Long-Term Vector Memory     | Memory               | Semantic retrieval               | Knowledge reuse           |
| 65 | Structured State Store      | Memory               | Workflow state                   | LangGraph state           |
| 66 | Time-Weighted Memory        | Memory               | Recent > old context             | Incident timelines        |
| 67 | Complexity Router           | SRE-Specific         | Simple vs complex path           | Cost control              |
| 68 | Correlation Graph           | SRE-Specific         | Merge alerts                     | Noise reduction           |
| 69 | Blast Radius Estimation     | SRE-Specific         | Impact scoring                   | Change safety             |
| 70 | Autonomous Remediation Loop | SRE-Specific         | Diagnose → fix → validate        | Auto-healing              |
| 71 | Safe Rollback               | SRE-Specific         | Revert failed actions            | Change management         |
| 72 | Versioned Agents            | Lifecycle            | Track behavior per version       | Governance                |
| 73 | Canary Agents               | Lifecycle            | Test on subset                   | Safe deployment           |
| 74 | Blue-Green Agents           | Lifecycle            | Zero-downtime upgrades           | Platform ops              |
| 75 | Feature Flags for Autonomy  | Lifecycle            | Toggle automation level          | Gradual rollout           |


# 🧠 Agentic AI Enterprise Patterns — Single Master Table
| Pattern                        | 16-Layer Mapping          | LLM vs Non-LLM | Primary Agent Role        |
| ------------------------------ | ------------------------- | -------------- | ------------------------- |
| Chain-of-Thought (CoT)         | Agent Runtime             | LLM Mandatory  | Planner                   |
| Tree-of-Thought (ToT)          | Agent Runtime             | LLM Mandatory  | Planner                   |
| ReAct (Reason+Act)             | Runtime + Tools           | Hybrid         | Planner → Executor        |
| Plan-Act                       | Orchestration (LangGraph) | Non-LLM        | Orchestrator              |
| Plan-Act-Reflect               | Orchestrator + Reflection | Hybrid         | Orchestrator + Reflection |
| Reflection / Self-Critique     | Reflection Layer          | LLM Mandatory  | Reflection                |
| Critic-Planner (Maker-Checker) | Governance                | Hybrid         | Validator                 |
| Reasoner-Planner-Executor      | Control Plane             | Hybrid         | Orchestrator              |
| World-Model State Tracking     | State & Memory            | Non-LLM        | Orchestrator              |

# Multi-Agent Collaboration
| Pattern               | 16-Layer Mapping     | LLM vs Non-LLM | Primary Agent Role |
| --------------------- | -------------------- | -------------- | ------------------ |
| Hierarchical Agents   | Control Plane        | Non-LLM        | Orchestrator       |
| Peer-to-Peer Agents   | Agent Runtime        | Hybrid         | Planner            |
| Role-Based Agents     | Registry & Discovery | Non-LLM        | Orchestrator       |
| Debate / Consensus    | Governance           | LLM Mandatory  | Validator          |
| Swarm Parallel Agents | Runtime              | Non-LLM        | Executor           |
| Agent Mesh (A2A Bus)  | Integration Layer    | Non-LLM        | Orchestrator       |

# Orchestration & Control
| Pattern                        | 16-Layer Mapping | LLM vs Non-LLM | Primary Agent Role |
| ------------------------------ | ---------------- | -------------- | ------------------ |
| Sequential Orchestration (DAG) | Control Plane    | Non-LLM        | Orchestrator       |
| Parallel Orchestration         | Control Plane    | Non-LLM        | Orchestrator       |
| Dynamic Routing                | Control Plane    | Hybrid         | Orchestrator       |
| Handoff Pattern                | Control Plane    | Non-LLM        | Orchestrator       |
| Group Chat Orchestration       | Runtime          | LLM Mandatory  | Planner            |
| Event-Driven Agents            | Input Layer      | Non-LLM        | Orchestrator       |
| Workflow Graph (LangGraph)     | Control Plane    | Non-LLM        | Orchestrator       |
| Control-Plane-as-Tool          | Tools Layer      | Non-LLM        | Executor           |

# Tooling & Action
| Pattern             | 16-Layer Mapping    | LLM vs Non-LLM | Primary Agent Role |
| ------------------- | ------------------- | -------------- | ------------------ |
| Tool-Use Selection  | Tools & Integration | Hybrid         | Planner            |
| Function Calling    | LLM Abstraction     | Hybrid         | Planner            |
| Skill Library       | Registry            | Non-LLM        | Orchestrator       |
| Runbook Automation  | Execution Layer     | Non-LLM        | Executor           |
| Capability Routing  | Registry            | Non-LLM        | Orchestrator       |
| Sandboxed Execution | Security Layer      | Non-LLM        | Executor           |

# Memory & Knowledge
| Pattern                    | 16-Layer Mapping | LLM vs Non-LLM | Primary Agent Role |
| -------------------------- | ---------------- | -------------- | ------------------ |
| Short-Term Memory          | State Layer      | Non-LLM        | Orchestrator       |
| Long-Term Memory           | Memory DB        | Non-LLM        | Orchestrator       |
| RAG Retrieval              | Knowledge Layer  | Hybrid         | Planner            |
| Episodic Memory            | State Layer      | Non-LLM        | Reflection         |
| Semantic Cache             | FinOps Layer     | Non-LLM        | Orchestrator       |
| Stateful Agent Checkpoints | Orchestrator     | Non-LLM        | Orchestrator       |

# Governance, Safety, HITL
| Pattern                         | 16-Layer Mapping | LLM vs Non-LLM | Primary Agent Role |
| ------------------------------- | ---------------- | -------------- | ------------------ |
| Human-in-the-Loop (HITL)        | HITL Layer       | Non-LLM        | Validator          |
| Policy Guardrails (OPA/Cedar)   | Policy Layer     | Non-LLM        | Validator          |
| RBAC/ABAC for Agents            | Security Layer   | Non-LLM        | Validator          |
| Risk-Based Autonomy             | Governance       | Hybrid         | Validator          |
| Audit Trail                     | Audit Layer      | Non-LLM        | Orchestrator       |
| Explainability (Evidence-First) | Governance       | Hybrid         | Reflection         |

# Observability & Evaluation
| Pattern                        | 16-Layer Mapping | LLM vs Non-LLM | Primary Agent Role |
| ------------------------------ | ---------------- | -------------- | ------------------ |
| Telemetry-Driven Agents        | Input Layer      | Non-LLM        | Orchestrator       |
| Outcome-Based Evaluation (KPI) | Observability    | Non-LLM        | Reflection         |
| Confidence Scoring             | Reflection       | Hybrid         | Reflection         |
| Canary Execution               | Deployment       | Non-LLM        | Executor           |
| Feedback Loop Learning         | Learning Layer   | Hybrid         | Reflection         |
| Digital Twin Simulation        | Governance       | Non-LLM        | Validator          |

# Cost & Performance
| Pattern                   | 16-Layer Mapping | LLM vs Non-LLM | Primary Agent Role |
| ------------------------- | ---------------- | -------------- | ------------------ |
| Model Routing             | LLM Abstraction  | Non-LLM        | Orchestrator       |
| Adaptive Autonomy         | Control Plane    | Hybrid         | Orchestrator       |
| Batch Reasoning           | Runtime          | Non-LLM        | Executor           |
| Caching (Semantic/Output) | State Layer      | Non-LLM        | Orchestrator       |
| Elastic Agent Scaling     | Deployment       | Non-LLM        | Executor           |

# Lifecycle & Deployment
| Pattern                    | 16-Layer Mapping | LLM vs Non-LLM | Primary Agent Role |
| -------------------------- | ---------------- | -------------- | ------------------ |
| Agent Registry & Discovery | Registry Layer   | Non-LLM        | Orchestrator       |
| Versioned Agents           | Lifecycle Layer  | Non-LLM        | Orchestrator       |
| Blue-Green Deployment      | Deployment       | Non-LLM        | Executor           |
| Shadow Mode Agents         | Observability    | Non-LLM        | Reflection         |
| Policy-as-Code             | Governance       | Non-LLM        | Validator          |
| Continuous Evaluation (CE) | MLOps            | Hybrid         | Reflection         |

# Communication & Transactions
| Pattern                        | 16-Layer Mapping  | LLM vs Non-LLM | Primary Agent Role |
| ------------------------------ | ----------------- | -------------- | ------------------ |
| Message Bus (Kafka/Pulsar)     | Integration Layer | Non-LLM        | Orchestrator       |
| Contract-Based Messaging (A2A) | Integration       | Non-LLM        | Orchestrator       |
| Event Sourcing                 | State Layer       | Non-LLM        | Orchestrator       |
| Saga Pattern                   | Orchestrator      | Non-LLM        | Orchestrator       |


# ✅ Single Master Table (Enterprise)
| Pattern                         | When to Use (Use Case)                        | Example (How to use)                                                           | Planner Agent                         | Executor Agent                       | Validator Agent                              | Reflection Agent                         |
| ------------------------------- | --------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------- | ------------------------------------ | -------------------------------------------- | ---------------------------------------- |
| Chain-of-Thought (CoT)          | RCA, risk reasoning, policy interpretation    | Planner reasons through evidence (metrics/logs) to propose safest remediation  | **LLM Required**                      | N/A                                  | N/A                                          | **LLM Optional** (quality review)        |
| Tree-of-Thought (ToT)           | Multiple remediation options                  | Generate 3–5 options (restart/scale/rollback), score and pick best             | **LLM Required**                      | N/A                                  | N/A                                          | **LLM Required** (branch outcome review) |
| ReAct (Reason+Act)              | Investigation requiring tools                 | Query Prometheus → logs → CMDB → decide action                                 | **LLM Required**                      | **Non-LLM** (tool calls)             | **Non-LLM** (policy gate)                    | **LLM Required** (did it work?)          |
| Plan-Act                        | Approved runbooks, patch workflows            | Orchestrator runs deterministic steps: drain→patch→verify                      | **LLM Optional** (plan draft)         | **Non-LLM Required**                 | **Non-LLM Required**                         | **Hybrid** (Non-LLM KPI + LLM summary)   |
| Plan-Act-Reflect                | Continuous improvement automation             | Execute fix → reflect → update future decision policy                          | **LLM Required**                      | **Non-LLM Required**                 | **Non-LLM Required**                         | **LLM Required**                         |
| Reflection / Self-Critique      | Reduce hallucination, validate outputs        | Reflection agent checks proposed action against evidence + constraints         | **LLM Optional**                      | N/A                                  | **LLM Optional** (critique) + Non-LLM policy | **LLM Required**                         |
| Critic-Planner (Maker-Checker)  | High-risk actions (DB restart, prod rollback) | Planner proposes → validator rejects if outside window → propose alternate     | **LLM Required**                      | **Non-LLM** (if approved)            | **Non-LLM Required** + **LLM Optional**      | **LLM Required**                         |
| Reasoner-Planner-Executor       | Enterprise separation of duties               | Planner decides “scale to 10” → executor calls K8s → orchestrator tracks state | **LLM Required**                      | **Non-LLM Required**                 | **Non-LLM Required**                         | **LLM Required**                         |
| World-Model / State Tracking    | Long-running incidents, resumable flows       | Maintain incident state + dependencies; resume after failure                   | **LLM Optional** (interpret)          | **Non-LLM Required** (state updates) | **Non-LLM**                                  | **LLM Optional** (drift insights)        |
| Hierarchical Multi-Agent        | SecOps/SRE/FinOps under one orchestrator      | Orchestrator routes work to domain agents                                      | **LLM Optional** (task decomposition) | **Non-LLM**                          | **Non-LLM**                                  | **LLM Optional**                         |
| Peer-to-Peer Agents             | Cross-domain collaboration                    | SRE shares latency findings with FinOps agent                                  | **LLM Required** (reasoning exchange) | **Non-LLM**                          | **Non-LLM**                                  | **LLM Optional**                         |
| Role-Based Agents               | Clear ownership boundaries                    | Patch agent only patches; Scaling agent only scales                            | **Non-LLM** (routing/registry)        | **Non-LLM**                          | **Non-LLM Required** (RBAC)                  | **LLM Optional**                         |
| Debate / Consensus              | Critical decisions requiring high confidence  | 2–3 planners debate; consensus agent chooses                                   | **LLM Required**                      | N/A                                  | **LLM Required** + Non-LLM policy            | **LLM Required**                         |
| Swarm Parallelization           | Fleet tasks at scale                          | Check 500 nodes patch status in parallel                                       | **Non-LLM**                           | **Non-LLM Required**                 | **Non-LLM**                                  | **LLM Optional**                         |
| Agent Mesh / A2A Messaging      | Decoupled agent communication                 | Agents communicate via contracts on message bus                                | **Non-LLM**                           | **Non-LLM Required**                 | **Non-LLM Required** (schema)                | **LLM Optional**                         |
| Sequential Orchestration (DAG)  | Runbooks, controlled flows                    | LangGraph executes step-by-step with retries/rollback                          | **Non-LLM**                           | **Non-LLM Required**                 | **Non-LLM Required**                         | **LLM Optional**                         |
| Parallel Orchestration          | Faster diagnosis/execution                    | Run checks for CPU/mem/disk/net concurrently                                   | **Non-LLM**                           | **Non-LLM Required**                 | **Non-LLM**                                  | **LLM Optional**                         |
| Dynamic Routing                 | Choose best agent/tool dynamically            | Classify incident → route to Network vs App agent                              | **Hybrid** (LLM classify)             | **Non-LLM Required**                 | **Non-LLM Required**                         | **LLM Optional**                         |
| Handoff / Escalation            | Risk threshold exceeded                       | Auto-heal fails → escalate to human approval                                   | **LLM Optional** (explain)            | **Non-LLM**                          | **Non-LLM Required** (HITL gate)             | **LLM Required**                         |
| Group Chat Orchestration        | Complex reasoning tasks                       | Planner+Risk+Cost agents collaborate                                           | **LLM Required**                      | N/A                                  | **LLM Optional** + policy                    | **LLM Required**                         |
| Event-Driven Agents             | Real-time ops automation                      | Kafka event “pod crash” triggers flow                                          | **Non-LLM**                           | **Non-LLM Required**                 | **Non-LLM Required**                         | **LLM Optional**                         |
| Tool-Use Selection              | Many possible tools/APIs                      | Choose between SSM / Terraform / K8s / ITSM                                    | **LLM Required**                      | **Non-LLM Required**                 | **Non-LLM Required**                         | **LLM Optional**                         |
| Function Calling (Structured)   | Deterministic tool invocation                 | LLM outputs JSON action contract for executor                                  | **LLM Required**                      | **Non-LLM Required**                 | **Non-LLM Required** (schema)                | **LLM Optional**                         |
| Skill Library                   | Standard reusable actions                     | “restart_service_v3”, “apply_patch_v2” skills                                  | **Non-LLM**                           | **Non-LLM Required**                 | **Non-LLM Required**                         | **LLM Optional**                         |
| Runbook Automation              | Approved self-heal actions                    | Restart service, clear queue, scale within blast radius                        | N/A                                   | **Non-LLM Required**                 | **Non-LLM Required**                         | **LLM Optional**                         |
| Short-Term Memory               | Maintain context per incident                 | Store current metrics/log pointers/session state                               | N/A                                   | **Non-LLM Required**                 | N/A                                          | **LLM Optional**                         |
| Long-Term Memory                | Learn from past incidents                     | Store successful fixes/outcomes; retrieve later                                | **Hybrid** (LLM uses)                 | **Non-LLM Required**                 | **Non-LLM**                                  | **LLM Required**                         |
| RAG Grounding                   | Need factual/policy grounding                 | Retrieve policy/runbook/CMDB docs → synthesize decision                        | **LLM Required**                      | **Non-LLM** (retrieval)              | **Non-LLM Required** (policy)                | **LLM Optional**                         |
| Episodic Memory                 | Outcome-based learning                        | “Restart fixed it” stored and ranked next time                                 | **Hybrid**                            | **Non-LLM Required**                 | N/A                                          | **LLM Required**                         |
| Semantic / Output Cache         | Reduce cost/latency                           | Cache repeated classifications and answers                                     | N/A                                   | **Non-LLM Required**                 | N/A                                          | **LLM Optional**                         |
| Human-in-the-Loop (HITL)        | High blast radius changes                     | Require approval for DB restart / prod rollback                                | **LLM Optional** (explain)            | N/A                                  | **Non-LLM Required** (approval)              | **LLM Required**                         |
| Policy Guardrails (OPA/Cedar)   | Enforce compliance and safety                 | Block action outside window/without ticket                                     | N/A                                   | N/A                                  | **Non-LLM Required**                         | **LLM Optional**                         |
| RBAC/ABAC                       | Least privilege                               | Patch agent cannot modify IAM; FinOps cannot restart prod                      | N/A                                   | N/A                                  | **Non-LLM Required**                         | N/A                                      |
| Risk-Based Autonomy             | Auto vs manual by risk score                  | Low risk auto-heal; high risk HITL                                             | **Hybrid** (LLM risk)                 | **Non-LLM Required**                 | **Non-LLM Required**                         | **LLM Required**                         |
| Audit Trail                     | SOC2/ISO evidence                             | Immutable log of decisions/actions/timestamps                                  | N/A                                   | **Non-LLM Required**                 | **Non-LLM Required**                         | **LLM Optional** (summaries)             |
| Explainability (Evidence-First) | Trust & governance                            | Output: evidence → policy → action → expected impact                           | **LLM Required**                      | N/A                                  | **Non-LLM** (check)                          | **LLM Required**                         |
| Outcome-Based Evaluation (KPI)  | Measure MTTR/availability value               | Compare before/after MTTR; autonomous resolution ratio                         | N/A                                   | **Non-LLM Required**                 | **Non-LLM**                                  | **LLM Required** (insights)              |
| Confidence Scoring              | Auto-escalation control                       | Low confidence → escalate to human                                             | **Hybrid**                            | N/A                                  | **Non-LLM Required** (threshold)             | **LLM Required** (calibrate)             |
| Canary Execution                | Safe changes                                  | Patch 5% nodes → validate → expand rollout                                     | N/A                                   | **Non-LLM Required**                 | **Non-LLM Required**                         | **LLM Optional**                         |
| Digital Twin / Simulation       | Predict impact before action                  | Simulate scale-in effect on latency before doing it                            | **LLM Optional**                      | **Non-LLM Required**                 | **Non-LLM Required**                         | **LLM Optional**                         |
| Model Routing                   | Reduce LLM cost                               | Small model classify; large model plan only when needed                        | **Non-LLM Required**                  | N/A                                  | **Non-LLM**                                  | **LLM Optional**                         |
| Adaptive Autonomy               | Save cost on simple cases                     | Simple alert → single-step plan; complex → ToT                                 | **Hybrid**                            | **Non-LLM Required**                 | **Non-LLM Required**                         | **LLM Required**                         |
| Batch Reasoning                 | High volume similar alerts                    | Group 100 similar alerts → one diagnosis plan                                  | **Hybrid**                            | **Non-LLM Required**                 | **Non-LLM**                                  | **LLM Optional**                         |
| Elastic Agent Scaling           | Scale platform runtime                        | Scale executor pods based on queue depth                                       | N/A                                   | **Non-LLM Required**                 | **Non-LLM**                                  | **LLM Optional**                         |
| Registry & Discovery            | Enterprise catalog                            | Find correct agent/skill/tool by metadata                                      | N/A                                   | **Non-LLM Required**                 | **Non-LLM Required**                         | **LLM Optional**                         |
| Versioned Agents                | Change control                                | Deploy v2 agent with rollback                                                  | N/A                                   | **Non-LLM Required**                 | **Non-LLM Required**                         | **LLM Optional**                         |
| Blue-Green Deployment           | Safe upgrades                                 | Switch traffic from v1 agent to v2                                             | N/A                                   | **Non-LLM Required**                 | **Non-LLM Required**                         | **LLM Optional**                         |
| Shadow Mode                     | Validate before autonomy                      | Run AI decisions in parallel; no execution                                     | **LLM Required**                      | N/A                                  | **Non-LLM Required**                         | **LLM Required**                         |
| Policy-as-Code                  | GitOps governance                             | PR to change autonomy thresholds                                               | N/A                                   | N/A                                  | **Non-LLM Required**                         | N/A                                      |
| Continuous Evaluation           | Regression tests for agents                   | Replay incident dataset; score accuracy                                        | **LLM Optional**                      | **Non-LLM Required**                 | **Non-LLM Required**                         | **LLM Required**                         |
| Message Bus                     | Event backbone                                | Alerts and agent messages through Kafka topics                                 | N/A                                   | **Non-LLM Required**                 | **Non-LLM Required**                         | **LLM Optional**                         |
| Contract Messaging              | Typed A2A communication                       | JSON schema enforced between agents                                            | N/A                                   | **Non-LLM Required**                 | **Non-LLM Required**                         | **LLM Optional**                         |
| Event Sourcing                  | Replayability                                 | Re-run incident flow for audits                                                | N/A                                   | **Non-LLM Required**                 | **Non-LLM**                                  | **LLM Optional**                         |
| Saga Pattern                    | Multi-step rollback safety                    | Patch fails → rollback → restore traffic                                       | **LLM Optional** (plan)               | **Non-LLM Required**                 | **Non-LLM Required**                         | **LLM Optional**                         |
