# Core application - Python SDK

## Develop a basic Workflow

How to develop a basic Workflow using the Temporal Python SDK.

Workflows are the fundamental unit of a Temporal Application, and it all starts with the development of a Workflow Definition.

In the Temporal Python SDK programming model, **`Workflows are defined as classes`**.

Specify the **`@workflow.defn`** decorator on the Workflow class to identify a Workflow.

Use the **`@workflow.run`** to mark the entry point method to be invoked. This must be set on one asynchronous method defined on the same class as @workflow.defn. Run methods have positional parameters.

```
from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from your_activities_dacx import your_activity
    from your_dataobject_dacx import YourParams

"""dacx
To spawn an Activity Execution, use the [`execute_activity()`](https://python.temporal.io/temporalio.workflow.html#execute_activity) operation from within your Workflow Definition.

`execute_activity()` is a shortcut for [`start_activity()`](https://python.temporal.io/temporalio.workflow.html#start_activity) that waits on its result.

To get just the handle to wait and cancel separately, use `start_activity()`.
In most cases, use `execute_activity()` unless advanced task capabilities are needed.

A single argument to the Activity is positional. Multiple arguments are not supported in the type-safe form of `start_activity()` or `execute_activity()` and must be supplied by the `args` keyword argument.
dacx"""

"""dacx
You can customize the Workflow name with a custom name in the decorator argument. For example, `@workflow.defn(name="your-workflow-name")`. If the name parameter is not specified, the Workflow name defaults to the function name.
dacx"""

"""dacx
In the Temporal Python SDK programming model, Workflows are defined as classes.

Specify the `@workflow.defn` decorator on the Workflow class to identify a Workflow.

Use the `@workflow.run` to mark the entry point method to be invoked. This must be set on one asynchronous method defined on the same class as `@workflow.defn`. Run methods have positional parameters.
dacx"""

"""dacx
To return a value of the Workflow, use `return` to return an object.

To return the results of a Workflow Execution, use either `start_workflow()` or `execute_workflow()` asynchronous methods.
dacx"""

"""dacx
Use [`start_activity()`](https://python.temporal.io/temporalio.workflow.html#start_activity) to start an Activity and return its handle, [`ActivityHandle`](https://python.temporal.io/temporalio.workflow.ActivityHandle.html). Use [`execute_activity()`](https://python.temporal.io/temporalio.workflow.html#execute_activity) to return the results.

You must provide either `schedule_to_close_timeout` or `start_to_close_timeout`.

`execute_activity()` is a shortcut for `await start_activity()`. An asynchronous `execute_activity()` helper is provided which takes the same arguments as `start_activity()` and `await`s on the result. `execute_activity()` should be used in most cases unless advanced task capabilities are needed.
dacx"""


@workflow.defn(name="YourWorkflow")
class YourWorkflow:
    @workflow.run
    async def run(self, name: str) -> str:
        return await workflow.execute_activity(
            your_activity,
            YourParams("Hello", name),
            start_to_close_timeout=timedelta(seconds=10),
        )


""" @dacx
id: how-to-spawn-an-activity-execution-in-python
title: How to spawn an Activity Execution in Python
label: Activity Execution
description: Use the `execute_activity()` operation from within your Workflow Definition.
lines: 3, 9-18, 47-55
@dacx """


""" @dacx
id: how-to-customize-workflow-type-in-python
title: How to customize Workflow types in Python
label: Customize Workflow types
description: Customize Workflow types.
lines: 3, 20-22, 47-55
@dacx """

""" @dacx
id: how-to-develop-a-workflow-definition-in-python
title: How to develop a Workflow Definition in Python
label: Develop a Workflow Definition
description: To develop a Workflow Definition, specify the `@workflow.defn` decorator on the Workflow class and use `@workflow.run` to mark the entry point.
lines: 3, 24-30, 47-55
@dacx """


""" @dacx
id: how-to-define-workflow-return-values-in-python
title: How to define Workflow return values
label: Define Workflow return values
description: Define Workflow return values.
tags:
 - workflow return values
lines: 3, 32-36, 47-55
@dacx """


""" @dacx
id: how-to-get-the-result-of-an-activity-execution-in-python
title: How to get the result of an Activity Execution in Python
label: Get the result of an Activity Execution
description: Get the result of an Activity Execution.
tags:
 - activity execution
 - python sdk
 - code sample
lines: 3, 38-44, 47-55
@dacx """
```

## Define Workflow parameters

**How to define Workflow parameters using the Temporal Python SDK.**

Temporal Workflows may have any number of custom parameters. However, we strongly recommend that objects are used as parameters, so that the object's individual fields may be altered without breaking the signature of the Workflow. All Workflow Definition parameters must be serializable.



`Temporal Workflows can take any number of parameters
But you should pass a single object (DTO), not many primitive arguments
And everything must be serializable`

## 1️⃣ What are “Workflow parameters”?

In Temporal, when you **start a Workflow**, you pass **input parameters**.

**Example (Python):**

```
@workflow.defn
class ScanWorkflow:
    @workflow.run
    async def run(self, account_id: str, region: str):
        ...
```

**Here:**

- `account_id`
- `region`

are **workflow parameters**.

## 2️⃣ What does “signature breaking” mean?

Let’s say you deploy this workflow:

`async def run(self, account_id: str, region: str):`

Now `later`, you decide you need one more parameter:

`async def run(self, account_id: str, region: str, severity: str):`

**🚨 Problem:**

- Existing running workflows
- Stored execution history
- Replays

**👉 Signature changed → old executions can break**

This is what `“breaking the signature”` means.


## 3️⃣ Why passing many parameters is risky ❌

**Bad practice:**

```
async def run(
    self,
    account_id: str,
    region: str,
    severity: str,
    notify: bool,
    approval_required: bool,
):
```

If you:

- Add
- Remove
- Reorder
- parameters

You risk:

- Workflow replay failures
- Versioning headaches
- Backward incompatibility

## 4️⃣ Recommended approach: use an object (DTO) ✅

Instead, use **one object parameter**.

**Example (Python dataclass)**

```
from dataclasses import dataclass

@dataclass
class ScanInput:
    account_id: str
    region: str
    severity: str | None = None
    notify: bool = False
    approval_required: bool = True
```

**Workflow:**

```
@workflow.defn
class ScanWorkflow:
    @workflow.run
    async def run(self, input: ScanInput):
        ...
```

## 5️⃣ Why this is better (key reason)

Now, in the future, you can:

```
@dataclass
class ScanInput:
    account_id: str
    region: str
    severity: str | None = None
    notify: bool = False
    approval_required: bool = True
    remediation_strategy: str | None = None  # NEW FIELD
```

- ✅ Existing workflows still work
- ✅ No signature change
- ✅ Safe evolution
- ✅ Backward compatibility

## 6️⃣ What does “All Workflow parameters must be serializable” mean?

**Temporal:**

- Stores workflow inputs
- Replays workflows
- Persists state

So parameters must be:

- JSON-serializable
- Or Protobuf-serializable

**✅ Good types**

- `str`
- `int`
- `float`
- `bool`
- `dict`
- `list`
- `dataclass`
- `Pydantic model`

**❌ Bad (NOT serializable)**

- Open file handles
- Database connections
- Thread locks
- Sockets
- Lambdas / functions
- Class instances with live state

## 8️⃣ Why Temporal is strict about this

**Temporal guarantees:**

- Deterministic execution
- Crash recovery
- Replay correctness

That’s why:

- Inputs must be immutable data
- No external side effects in Workflow code
- No non-serializable objects

## 1️⃣1️⃣ Best practice checklist

✔ One input object (DTO)
✔ Default values for new fields
✔ Only serializable data
✔ No live connections
✔ Activities do the real work
✔ Workflow = orchestration only

## ❌ Examples of NON-serializable objects (DO NOT USE)

**1️⃣ Database connections**

```
async def run(self, db_conn):
    ...
```

Why ❌

- Live network connection
- Cannot be serialized
- Breaks replay


**2️⃣ Cloud SDK clients (AWS, Azure, GCP)**

```
import boto3

client = boto3.client("ec2")

async def run(self, ec2_client):
    ...
```

Why ❌

- Holds sockets, credentials, state
- Non-deterministic

**3️⃣ Open file handles**

```
file = open("data.txt")

async def run(self, f):
    ...
```

Why ❌

- OS resource
- Cannot be restored on replay

**4️⃣ Threads, locks, executors**

```
from threading import Lock

lock = Lock()

async def run(self, lock):
    ...
```

Why ❌

- Runtime-only objects
- No serialization format

**5️⃣ Functions / lambdas**

```
async def run(self, callback):
    callback()
```

Why ❌

- Code references cannot be serialized
- Replay unsafe

**6️⃣ Sockets / HTTP sessions**

```
import requests

session = requests.Session()

async def run(self, session):
    ...
```

Why ❌

- Connection state
- Side effects

**7️⃣ ORM sessions (SQLAlchemy)**

```
async def run(self, session):
    session.query(...)
```

Why ❌

- Connection + transaction state
- Replay impossible


## ✅ Serializable objects (SAFE)

**1️⃣ Primitives**

```
async def run(self, account_id: str, region: str):
    ...
```

**2️⃣ Dict / List**

```
async def run(self, config: dict):
    ...
```

**3️⃣ Dataclasses (BEST PRACTICE)**

```
from dataclasses import dataclass

@dataclass
class ScanInput:
    account_id: str
    region: str
    severity: str | None = None
```

```
async def run(self, input: ScanInput):
    ...
```


**4️⃣ Pydantic models**

```
from pydantic import BaseModel

class ScanInput(BaseModel):
    account_id: str
    region: str
```

**5️⃣ Enums**

```
from enum import Enum

class ScanType(str, Enum):
    FULL = "full"
    QUICK = "quick"
```

## ❌ WRONG (logic + side effects inside Workflow)

```
@workflow.run
async def run(self, input):
    client = boto3.client("ec2")
    client.describe_instances()
```

## ✅ RIGHT (Workflow orchestrates, Activity executes)

```
@workflow.run
async def run(self, input: ScanInput):
    await workflow.execute_activity(
        scan_ec2_activity,
        input.account_id,
        input.region,
        start_to_close_timeout=timedelta(minutes=5),
    )
```

```
@activity.defn
async def scan_ec2_activity(account_id: str, region: str):
    client = boto3.client("ec2")
    return client.describe_instances()
```


Workflow parameters are the method parameters of the singular method decorated with `@workflow.run`. These can be any data type Temporal can convert, including `dataclasses` when properly type-annotated. Technically this can be multiple parameters, but Temporal strongly encourages a single `dataclass` parameter containing all input fields.


## Define Workflow return parameters

**How to define Workflow return parameters using the Temporal Python SDK.**

Workflow return values must also be serializable. Returning results, returning errors, or throwing exceptions is fairly idiomatic in each language that is supported. However, Temporal APIs that must be used to get the result of a Workflow Execution will only ever receive one of either the result or the error.


To return a value of the Workflow, use `return` to return an object.

To return the results of a Workflow Execution, use either `start_workflow()` or `execute_workflow()` asynchronous methods.

```
from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from your_activities_dacx import your_activity
    from your_dataobject_dacx import YourParams

"""dacx
To spawn an Activity Execution, use the [`execute_activity()`](https://python.temporal.io/temporalio.workflow.html#execute_activity) operation from within your Workflow Definition.

`execute_activity()` is a shortcut for [`start_activity()`](https://python.temporal.io/temporalio.workflow.html#start_activity) that waits on its result.

To get just the handle to wait and cancel separately, use `start_activity()`.
In most cases, use `execute_activity()` unless advanced task capabilities are needed.

A single argument to the Activity is positional. Multiple arguments are not supported in the type-safe form of `start_activity()` or `execute_activity()` and must be supplied by the `args` keyword argument.
dacx"""

"""dacx
You can customize the Workflow name with a custom name in the decorator argument. For example, `@workflow.defn(name="your-workflow-name")`. If the name parameter is not specified, the Workflow name defaults to the function name.
dacx"""

"""dacx
In the Temporal Python SDK programming model, Workflows are defined as classes.

Specify the `@workflow.defn` decorator on the Workflow class to identify a Workflow.

Use the `@workflow.run` to mark the entry point method to be invoked. This must be set on one asynchronous method defined on the same class as `@workflow.defn`. Run methods have positional parameters.
dacx"""

"""dacx
To return a value of the Workflow, use `return` to return an object.

To return the results of a Workflow Execution, use either `start_workflow()` or `execute_workflow()` asynchronous methods.
dacx"""

"""dacx
Use [`start_activity()`](https://python.temporal.io/temporalio.workflow.html#start_activity) to start an Activity and return its handle, [`ActivityHandle`](https://python.temporal.io/temporalio.workflow.ActivityHandle.html). Use [`execute_activity()`](https://python.temporal.io/temporalio.workflow.html#execute_activity) to return the results.

You must provide either `schedule_to_close_timeout` or `start_to_close_timeout`.

`execute_activity()` is a shortcut for `await start_activity()`. An asynchronous `execute_activity()` helper is provided which takes the same arguments as `start_activity()` and `await`s on the result. `execute_activity()` should be used in most cases unless advanced task capabilities are needed.
dacx"""


@workflow.defn(name="YourWorkflow")
class YourWorkflow:
    @workflow.run
    async def run(self, name: str) -> str:
        return await workflow.execute_activity(
            your_activity,
            YourParams("Hello", name),
            start_to_close_timeout=timedelta(seconds=10),
        )


""" @dacx
id: how-to-spawn-an-activity-execution-in-python
title: How to spawn an Activity Execution in Python
label: Activity Execution
description: Use the `execute_activity()` operation from within your Workflow Definition.
lines: 3, 9-18, 47-55
@dacx """


""" @dacx
id: how-to-customize-workflow-type-in-python
title: How to customize Workflow types in Python
label: Customize Workflow types
description: Customize Workflow types.
lines: 3, 20-22, 47-55
@dacx """

""" @dacx
id: how-to-develop-a-workflow-definition-in-python
title: How to develop a Workflow Definition in Python
label: Develop a Workflow Definition
description: To develop a Workflow Definition, specify the `@workflow.defn` decorator on the Workflow class and use `@workflow.run` to mark the entry point.
lines: 3, 24-30, 47-55
@dacx """


""" @dacx
id: how-to-define-workflow-return-values-in-python
title: How to define Workflow return values
label: Define Workflow return values
description: Define Workflow return values.
tags:
 - workflow return values
lines: 3, 32-36, 47-55
@dacx """


""" @dacx
id: how-to-get-the-result-of-an-activity-execution-in-python
title: How to get the result of an Activity Execution in Python
label: Get the result of an Activity Execution
description: Get the result of an Activity Execution.
tags:
 - activity execution
 - python sdk
 - code sample
lines: 3, 38-44, 47-55
@dacx """
```


## Customize your Workflow Type

**How to customize your Workflow Type using the Temporal Python SDK.**

Workflows have a Type that are referred to as the Workflow name.

The following examples demonstrate how to set a custom name for your Workflow Type.

You can customize the Workflow name with a custom name in the decorator argument. For example, `@workflow.defn(name="your-workflow-name")`. If the name parameter is not specified, the Workflow name defaults to the unqualified class name.

```
from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from your_activities_dacx import your_activity
    from your_dataobject_dacx import YourParams

"""dacx
To spawn an Activity Execution, use the [`execute_activity()`](https://python.temporal.io/temporalio.workflow.html#execute_activity) operation from within your Workflow Definition.

`execute_activity()` is a shortcut for [`start_activity()`](https://python.temporal.io/temporalio.workflow.html#start_activity) that waits on its result.

To get just the handle to wait and cancel separately, use `start_activity()`.
In most cases, use `execute_activity()` unless advanced task capabilities are needed.

A single argument to the Activity is positional. Multiple arguments are not supported in the type-safe form of `start_activity()` or `execute_activity()` and must be supplied by the `args` keyword argument.
dacx"""

"""dacx
You can customize the Workflow name with a custom name in the decorator argument. For example, `@workflow.defn(name="your-workflow-name")`. If the name parameter is not specified, the Workflow name defaults to the function name.
dacx"""

"""dacx
In the Temporal Python SDK programming model, Workflows are defined as classes.

Specify the `@workflow.defn` decorator on the Workflow class to identify a Workflow.

Use the `@workflow.run` to mark the entry point method to be invoked. This must be set on one asynchronous method defined on the same class as `@workflow.defn`. Run methods have positional parameters.
dacx"""

"""dacx
To return a value of the Workflow, use `return` to return an object.

To return the results of a Workflow Execution, use either `start_workflow()` or `execute_workflow()` asynchronous methods.
dacx"""

"""dacx
Use [`start_activity()`](https://python.temporal.io/temporalio.workflow.html#start_activity) to start an Activity and return its handle, [`ActivityHandle`](https://python.temporal.io/temporalio.workflow.ActivityHandle.html). Use [`execute_activity()`](https://python.temporal.io/temporalio.workflow.html#execute_activity) to return the results.

You must provide either `schedule_to_close_timeout` or `start_to_close_timeout`.

`execute_activity()` is a shortcut for `await start_activity()`. An asynchronous `execute_activity()` helper is provided which takes the same arguments as `start_activity()` and `await`s on the result. `execute_activity()` should be used in most cases unless advanced task capabilities are needed.
dacx"""


@workflow.defn(name="YourWorkflow")
class YourWorkflow:
    @workflow.run
    async def run(self, name: str) -> str:
        return await workflow.execute_activity(
            your_activity,
            YourParams("Hello", name),
            start_to_close_timeout=timedelta(seconds=10),
        )


""" @dacx
id: how-to-spawn-an-activity-execution-in-python
title: How to spawn an Activity Execution in Python
label: Activity Execution
description: Use the `execute_activity()` operation from within your Workflow Definition.
lines: 3, 9-18, 47-55
@dacx """


""" @dacx
id: how-to-customize-workflow-type-in-python
title: How to customize Workflow types in Python
label: Customize Workflow types
description: Customize Workflow types.
lines: 3, 20-22, 47-55
@dacx """

""" @dacx
id: how-to-develop-a-workflow-definition-in-python
title: How to develop a Workflow Definition in Python
label: Develop a Workflow Definition
description: To develop a Workflow Definition, specify the `@workflow.defn` decorator on the Workflow class and use `@workflow.run` to mark the entry point.
lines: 3, 24-30, 47-55
@dacx """


""" @dacx
id: how-to-define-workflow-return-values-in-python
title: How to define Workflow return values
label: Define Workflow return values
description: Define Workflow return values.
tags:
 - workflow return values
lines: 3, 32-36, 47-55
@dacx """


""" @dacx
id: how-to-get-the-result-of-an-activity-execution-in-python
title: How to get the result of an Activity Execution in Python
label: Get the result of an Activity Execution
description: Get the result of an Activity Execution.
tags:
 - activity execution
 - python sdk
 - code sample
lines: 3, 38-44, 47-55
@dacx """
```


## Develop Workflow logic

**How to develop Workflow logic using the Temporal Python SDK.**

Workflow logic is constrained by deterministic execution requirements. Therefore, each language is limited to the use of certain idiomatic techniques. However, each Temporal SDK provides a set of APIs that can be used inside your Workflow to interact with external (to the Workflow) application code.

Workflow code must be deterministic. This means:

- no threading
- no randomness
- no external calls to processes
- no network I/O
- no global state mutation
- no system date or time


All API safe for Workflows used in the `temporalio.workflow` must run in the implicit `asyncio event loop` and be deterministic.

## Develop a basic Activity

**How to develop a basic Activity using the Temporal Python SDK.**

One of the primary things that Workflows do is orchestrate the execution of Activities. An Activity is a normal function or method execution that's intended to execute a single, well-defined action (either short or long-running), such as querying a database, calling a third-party API, or transcoding a media file. An Activity can interact with world outside the Temporal Platform or use a Temporal Client to interact with a Temporal Service. For the Workflow to be able to execute the Activity, we must define the `Activity Definition`.

You can develop an Activity Definition by using the `@activity.defn decorator`. Register the function as an Activity with a custom name through a decorator argument, for example `@activity.defn(name="your_activity")`.

**The Temporal Python SDK supports multiple ways of implementing an Activity:**

- Asynchronously using `asyncio`
- Synchronously multithreaded using `concurrent.futures.ThreadPoolExecutor`
- Synchronously multiprocess using `concurrent.futures.ProcessPoolExecutor and multiprocessing.managers.SyncManager`

```
from temporalio import activity
from your_dataobject_dacx import YourParams

"""dacx
You can develop an Activity Definition by using the `@activity.defn` decorator.
Register the function as an Activity with a custom name through a decorator argument, for example `@activity.defn(name="your_activity")`.

:::note

The Temporal Python SDK supports multiple ways of implementing an Activity:
- Asynchronously using [`asyncio`](https://docs.python.org/3/library/asyncio.html)
- Synchronously multithreaded using [`concurrent.futures.ThreadPoolExecutor`](https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor)
- Synchronously multiprocess using [`concurrent.futures.ProcessPoolExecutor`](https://docs.python.org/3/library/concurrent.futures.html#processpoolexecutor) and [`multiprocessing.managers.SyncManager`](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.managers.SyncManager)

Blocking the async event loop in Python would turn your asynchronous program into a synchronous program that executes serially, defeating the entire purpose of using `asyncio`.
This can also lead to potential deadlock, and unpredictable behavior that causes tasks to be unable to execute.
Debugging these issues can be difficult and time consuming, as locating the source of the blocking call might not always be immediately obvious.

Due to this, consider not make blocking calls from within an asynchronous Activity, or use an async safe library to perform
these actions.
If you must use a blocking library, consider using a synchronous Activity instead.

:::
dacx"""

"""dacx
Activity parameters are the function parameters of the function decorated with `@activity.defn`.
These can be any data type Temporal can convert, including dataclasses when properly type-annotated.
Technically this can be multiple parameters, but Temporal strongly encourages a single dataclass parameter containing all input fields.
dacx"""

"""dacx
An Activity Execution can return inputs and other Activity values.

The following example defines an Activity that takes a string as input and returns a string.
dacx"""

"""dacx
You can customize the Activity name with a custom name in the decorator argument. For example, `@activity.defn(name="your-activity")`.
If the name parameter is not specified, the Activity name defaults to the function name.
dacx"""


@activity.defn(name="your_activity")
async def your_activity(input: YourParams) -> str:
    return f"{input.greeting}, {input.name}!"


""" @dacx
id: how-to-develop-an-activity-definition-in-python
title: How to develop an Activity Definition in Python
label: Activity Definition
description: You can develop an Activity Definition by using the `@activity.defn` decorator.
tags:
 - python sdk
 - code sample
 - activity definition
lines: 1, 4-24, 44-46
@dacx """

""" @dacx
id: how-to-define-activity-parameters-in-python
title: How to do define Activity parameters in Python
label: Activity parameters
description: Activity parameters are the function parameters of the function decorated with `@activity.defn`.
tags:
 - activity execution
 - python sdk
 - code sample
lines: 1-3, 26-30, 44-46
@dacx """

""" @dacx
id: how-to-define-activity-return-values-in-python
title: How to define Activity return values in Python
label: Activity return values
description: To return a value of the Workflow, use `return` to return an object.
tags:
 - activity execution
 - python sdk
 - code sample
lines: 32-36, 44-46
@dacx """

""" @dacx
id: how-to-customize-activity-type-in-python
title: How to customize Activity Type in Python
label: Customize Activity Type
description: Customize your Activity Type.
tags:
 - activity execution
 - python sdk
 - code sample
lines: 38-41, 44-46
@dacx """
```

## Develop Activity Parameters

**How to develop Activity Parameters using the Temporal Python SDK.**

There is no explicit limit to the total number of parameters that an Activity Definition may support. However, there is a limit to the total size of the data that ends up encoded into a gRPC message Payload.

A single argument is limited to a maximum size of 2 MB. And the total size of a gRPC message, which includes all the arguments, is limited to a maximum of 4 MB.

Also, keep in mind that all Payload data is recorded in the `Workflow Execution Event History` and large Event Histories can affect Worker performance. This is because the entire Event History could be transferred to a Worker Process with a `Workflow Task`.

Some SDKs require that you pass context objects, others do not. When it comes to your application data—that is, data that is serialized and encoded into a Payload—we recommend that you use a single object as an argument that wraps the application data passed to Activities. This is so that you can change what data is passed to the Activity without breaking a function or method signature.


Activity parameters are the function parameters of the function decorated with `@activity.defn```. These can be any data type Temporal can convert, including dataclasses when properly type-annotated. Technically this can be multiple parameters, but Temporal strongly encourages a single dataclass parameter containing all input fields.

```
from temporalio import activity
from your_dataobject_dacx import YourParams

"""dacx
You can develop an Activity Definition by using the `@activity.defn` decorator.
Register the function as an Activity with a custom name through a decorator argument, for example `@activity.defn(name="your_activity")`.

:::note

The Temporal Python SDK supports multiple ways of implementing an Activity:
- Asynchronously using [`asyncio`](https://docs.python.org/3/library/asyncio.html)
- Synchronously multithreaded using [`concurrent.futures.ThreadPoolExecutor`](https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor)
- Synchronously multiprocess using [`concurrent.futures.ProcessPoolExecutor`](https://docs.python.org/3/library/concurrent.futures.html#processpoolexecutor) and [`multiprocessing.managers.SyncManager`](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.managers.SyncManager)

Blocking the async event loop in Python would turn your asynchronous program into a synchronous program that executes serially, defeating the entire purpose of using `asyncio`.
This can also lead to potential deadlock, and unpredictable behavior that causes tasks to be unable to execute.
Debugging these issues can be difficult and time consuming, as locating the source of the blocking call might not always be immediately obvious.

Due to this, consider not make blocking calls from within an asynchronous Activity, or use an async safe library to perform
these actions.
If you must use a blocking library, consider using a synchronous Activity instead.

:::
dacx"""

"""dacx
Activity parameters are the function parameters of the function decorated with `@activity.defn`.
These can be any data type Temporal can convert, including dataclasses when properly type-annotated.
Technically this can be multiple parameters, but Temporal strongly encourages a single dataclass parameter containing all input fields.
dacx"""

"""dacx
An Activity Execution can return inputs and other Activity values.

The following example defines an Activity that takes a string as input and returns a string.
dacx"""

"""dacx
You can customize the Activity name with a custom name in the decorator argument. For example, `@activity.defn(name="your-activity")`.
If the name parameter is not specified, the Activity name defaults to the function name.
dacx"""


@activity.defn(name="your_activity")
async def your_activity(input: YourParams) -> str:
    return f"{input.greeting}, {input.name}!"


""" @dacx
id: how-to-develop-an-activity-definition-in-python
title: How to develop an Activity Definition in Python
label: Activity Definition
description: You can develop an Activity Definition by using the `@activity.defn` decorator.
tags:
 - python sdk
 - code sample
 - activity definition
lines: 1, 4-24, 44-46
@dacx """

""" @dacx
id: how-to-define-activity-parameters-in-python
title: How to do define Activity parameters in Python
label: Activity parameters
description: Activity parameters are the function parameters of the function decorated with `@activity.defn`.
tags:
 - activity execution
 - python sdk
 - code sample
lines: 1-3, 26-30, 44-46
@dacx """

""" @dacx
id: how-to-define-activity-return-values-in-python
title: How to define Activity return values in Python
label: Activity return values
description: To return a value of the Workflow, use `return` to return an object.
tags:
 - activity execution
 - python sdk
 - code sample
lines: 32-36, 44-46
@dacx """

""" @dacx
id: how-to-customize-activity-type-in-python
title: How to customize Activity Type in Python
label: Customize Activity Type
description: Customize your Activity Type.
tags:
 - activity execution
 - python sdk
 - code sample
lines: 38-41, 44-46
@dacx """
```

## Define Activity return values

**How to define Activity return values using the Temporal Python SDK.**

All data returned from an Activity must be serializable.

Activity return values are subject to payload size limits in Temporal. The default payload size limit is 2MB, and there is a hard limit of 4MB for any gRPC message size in the Event History transaction (see Cloud limits here). Keep in mind that all return values are recorded in a Workflow Execution Event History.


An Activity Execution can return inputs and other Activity values.

The following example defines an Activity that takes a string as input and returns a string.

```
from temporalio import activity
from your_dataobject_dacx import YourParams

"""dacx
You can develop an Activity Definition by using the `@activity.defn` decorator.
Register the function as an Activity with a custom name through a decorator argument, for example `@activity.defn(name="your_activity")`.

:::note

The Temporal Python SDK supports multiple ways of implementing an Activity:
- Asynchronously using [`asyncio`](https://docs.python.org/3/library/asyncio.html)
- Synchronously multithreaded using [`concurrent.futures.ThreadPoolExecutor`](https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor)
- Synchronously multiprocess using [`concurrent.futures.ProcessPoolExecutor`](https://docs.python.org/3/library/concurrent.futures.html#processpoolexecutor) and [`multiprocessing.managers.SyncManager`](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.managers.SyncManager)

Blocking the async event loop in Python would turn your asynchronous program into a synchronous program that executes serially, defeating the entire purpose of using `asyncio`.
This can also lead to potential deadlock, and unpredictable behavior that causes tasks to be unable to execute.
Debugging these issues can be difficult and time consuming, as locating the source of the blocking call might not always be immediately obvious.

Due to this, consider not make blocking calls from within an asynchronous Activity, or use an async safe library to perform
these actions.
If you must use a blocking library, consider using a synchronous Activity instead.

:::
dacx"""

"""dacx
Activity parameters are the function parameters of the function decorated with `@activity.defn`.
These can be any data type Temporal can convert, including dataclasses when properly type-annotated.
Technically this can be multiple parameters, but Temporal strongly encourages a single dataclass parameter containing all input fields.
dacx"""

"""dacx
An Activity Execution can return inputs and other Activity values.

The following example defines an Activity that takes a string as input and returns a string.
dacx"""

"""dacx
You can customize the Activity name with a custom name in the decorator argument. For example, `@activity.defn(name="your-activity")`.
If the name parameter is not specified, the Activity name defaults to the function name.
dacx"""


@activity.defn(name="your_activity")
async def your_activity(input: YourParams) -> str:
    return f"{input.greeting}, {input.name}!"


""" @dacx
id: how-to-develop-an-activity-definition-in-python
title: How to develop an Activity Definition in Python
label: Activity Definition
description: You can develop an Activity Definition by using the `@activity.defn` decorator.
tags:
 - python sdk
 - code sample
 - activity definition
lines: 1, 4-24, 44-46
@dacx """

""" @dacx
id: how-to-define-activity-parameters-in-python
title: How to do define Activity parameters in Python
label: Activity parameters
description: Activity parameters are the function parameters of the function decorated with `@activity.defn`.
tags:
 - activity execution
 - python sdk
 - code sample
lines: 1-3, 26-30, 44-46
@dacx """

""" @dacx
id: how-to-define-activity-return-values-in-python
title: How to define Activity return values in Python
label: Activity return values
description: To return a value of the Workflow, use `return` to return an object.
tags:
 - activity execution
 - python sdk
 - code sample
lines: 32-36, 44-46
@dacx """

""" @dacx
id: how-to-customize-activity-type-in-python
title: How to customize Activity Type in Python
label: Customize Activity Type
description: Customize your Activity Type.
tags:
 - activity execution
 - python sdk
 - code sample
lines: 38-41, 44-46
@dacx """
```

## Customize your Activity Type

**How to customize your Activity Type**

Activities have a Type that are referred to as the Activity name. The following examples demonstrate how to set a custom name for your Activity Type.

You can customize the Activity name with a custom name in the decorator argument. For example, `@activity.defn(name="your-activity")`. If the name parameter is not specified, the Activity name defaults to the function name.


```
from temporalio import activity
from your_dataobject_dacx import YourParams

"""dacx
You can develop an Activity Definition by using the `@activity.defn` decorator.
Register the function as an Activity with a custom name through a decorator argument, for example `@activity.defn(name="your_activity")`.

:::note

The Temporal Python SDK supports multiple ways of implementing an Activity:
- Asynchronously using [`asyncio`](https://docs.python.org/3/library/asyncio.html)
- Synchronously multithreaded using [`concurrent.futures.ThreadPoolExecutor`](https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor)
- Synchronously multiprocess using [`concurrent.futures.ProcessPoolExecutor`](https://docs.python.org/3/library/concurrent.futures.html#processpoolexecutor) and [`multiprocessing.managers.SyncManager`](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.managers.SyncManager)

Blocking the async event loop in Python would turn your asynchronous program into a synchronous program that executes serially, defeating the entire purpose of using `asyncio`.
This can also lead to potential deadlock, and unpredictable behavior that causes tasks to be unable to execute.
Debugging these issues can be difficult and time consuming, as locating the source of the blocking call might not always be immediately obvious.

Due to this, consider not make blocking calls from within an asynchronous Activity, or use an async safe library to perform
these actions.
If you must use a blocking library, consider using a synchronous Activity instead.

:::
dacx"""

"""dacx
Activity parameters are the function parameters of the function decorated with `@activity.defn`.
These can be any data type Temporal can convert, including dataclasses when properly type-annotated.
Technically this can be multiple parameters, but Temporal strongly encourages a single dataclass parameter containing all input fields.
dacx"""

"""dacx
An Activity Execution can return inputs and other Activity values.

The following example defines an Activity that takes a string as input and returns a string.
dacx"""

"""dacx
You can customize the Activity name with a custom name in the decorator argument. For example, `@activity.defn(name="your-activity")`.
If the name parameter is not specified, the Activity name defaults to the function name.
dacx"""


@activity.defn(name="your_activity")
async def your_activity(input: YourParams) -> str:
    return f"{input.greeting}, {input.name}!"


""" @dacx
id: how-to-develop-an-activity-definition-in-python
title: How to develop an Activity Definition in Python
label: Activity Definition
description: You can develop an Activity Definition by using the `@activity.defn` decorator.
tags:
 - python sdk
 - code sample
 - activity definition
lines: 1, 4-24, 44-46
@dacx """

""" @dacx
id: how-to-define-activity-parameters-in-python
title: How to do define Activity parameters in Python
label: Activity parameters
description: Activity parameters are the function parameters of the function decorated with `@activity.defn`.
tags:
 - activity execution
 - python sdk
 - code sample
lines: 1-3, 26-30, 44-46
@dacx """

""" @dacx
id: how-to-define-activity-return-values-in-python
title: How to define Activity return values in Python
label: Activity return values
description: To return a value of the Workflow, use `return` to return an object.
tags:
 - activity execution
 - python sdk
 - code sample
lines: 32-36, 44-46
@dacx """

""" @dacx
id: how-to-customize-activity-type-in-python
title: How to customize Activity Type in Python
label: Customize Activity Type
description: Customize your Activity Type.
tags:
 - activity execution
 - python sdk
 - code sample
lines: 38-41, 44-46
@dacx """
```


## Start an Activity Execution

**How to start an Activity Execution using the Temporal Python SDK.**

Calls to spawn `Activity Executions` are written within a `Workflow Definition`. The call to spawn an Activity Execution generates the `ScheduleActivityTask` Command. This results in the set of three `Activity Task` related Events (`ActivityTaskScheduled`, `ActivityTaskStarted`, and `ActivityTask[Closed]`)in your Workflow Execution Event History.


To spawn an Activity Execution, use the `execute_activity()` operation from within your Workflow Definition.

`execute_activity()` is a shortcut for `start_activity()` that waits on its result.

To get just the handle to wait and cancel separately, use `start_activity()`. In most cases, use `execute_activity()` unless advanced task capabilities are needed.

A single argument to the Activity is positional. Multiple arguments are not supported in the type-safe form of `start_activity()` or `execute_activity()` and must be supplied by the args keyword argument.

```
from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from your_activities_dacx import your_activity
    from your_dataobject_dacx import YourParams

"""dacx
To spawn an Activity Execution, use the [`execute_activity()`](https://python.temporal.io/temporalio.workflow.html#execute_activity) operation from within your Workflow Definition.

`execute_activity()` is a shortcut for [`start_activity()`](https://python.temporal.io/temporalio.workflow.html#start_activity) that waits on its result.

To get just the handle to wait and cancel separately, use `start_activity()`.
In most cases, use `execute_activity()` unless advanced task capabilities are needed.

A single argument to the Activity is positional. Multiple arguments are not supported in the type-safe form of `start_activity()` or `execute_activity()` and must be supplied by the `args` keyword argument.
dacx"""

"""dacx
You can customize the Workflow name with a custom name in the decorator argument. For example, `@workflow.defn(name="your-workflow-name")`. If the name parameter is not specified, the Workflow name defaults to the function name.
dacx"""

"""dacx
In the Temporal Python SDK programming model, Workflows are defined as classes.

Specify the `@workflow.defn` decorator on the Workflow class to identify a Workflow.

Use the `@workflow.run` to mark the entry point method to be invoked. This must be set on one asynchronous method defined on the same class as `@workflow.defn`. Run methods have positional parameters.
dacx"""

"""dacx
To return a value of the Workflow, use `return` to return an object.

To return the results of a Workflow Execution, use either `start_workflow()` or `execute_workflow()` asynchronous methods.
dacx"""

"""dacx
Use [`start_activity()`](https://python.temporal.io/temporalio.workflow.html#start_activity) to start an Activity and return its handle, [`ActivityHandle`](https://python.temporal.io/temporalio.workflow.ActivityHandle.html). Use [`execute_activity()`](https://python.temporal.io/temporalio.workflow.html#execute_activity) to return the results.

You must provide either `schedule_to_close_timeout` or `start_to_close_timeout`.

`execute_activity()` is a shortcut for `await start_activity()`. An asynchronous `execute_activity()` helper is provided which takes the same arguments as `start_activity()` and `await`s on the result. `execute_activity()` should be used in most cases unless advanced task capabilities are needed.
dacx"""


@workflow.defn(name="YourWorkflow")
class YourWorkflow:
    @workflow.run
    async def run(self, name: str) -> str:
        return await workflow.execute_activity(
            your_activity,
            YourParams("Hello", name),
            start_to_close_timeout=timedelta(seconds=10),
        )


""" @dacx
id: how-to-spawn-an-activity-execution-in-python
title: How to spawn an Activity Execution in Python
label: Activity Execution
description: Use the `execute_activity()` operation from within your Workflow Definition.
lines: 3, 9-18, 47-55
@dacx """


""" @dacx
id: how-to-customize-workflow-type-in-python
title: How to customize Workflow types in Python
label: Customize Workflow types
description: Customize Workflow types.
lines: 3, 20-22, 47-55
@dacx """

""" @dacx
id: how-to-develop-a-workflow-definition-in-python
title: How to develop a Workflow Definition in Python
label: Develop a Workflow Definition
description: To develop a Workflow Definition, specify the `@workflow.defn` decorator on the Workflow class and use `@workflow.run` to mark the entry point.
lines: 3, 24-30, 47-55
@dacx """


""" @dacx
id: how-to-define-workflow-return-values-in-python
title: How to define Workflow return values
label: Define Workflow return values
description: Define Workflow return values.
tags:
 - workflow return values
lines: 3, 32-36, 47-55
@dacx """


""" @dacx
id: how-to-get-the-result-of-an-activity-execution-in-python
title: How to get the result of an Activity Execution in Python
label: Get the result of an Activity Execution
description: Get the result of an Activity Execution.
tags:
 - activity execution
 - python sdk
 - code sample
lines: 3, 38-44, 47-55
@dacx """
```

## Set the required Activity Timeouts

**How to set the required Activity Timeouts using the Temporal Python SDK.**

Activity Execution semantics rely on several parameters. The only required value that needs to be set is either a `Schedule-To-Close Timeout` or a `Start-To-Close Timeout`. These values are set in the Activity Options.

Activity options are set as keyword arguments after the Activity arguments.

Available timeouts are:

- schedule_to_close_timeout
- schedule_to_start_timeout
- start_to_close_timeout

```
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from activities import your_activity, YourParams

"""dacx
Activity options are set as keyword arguments after the Activity arguments.

Available timeouts are:

- schedule_to_close_timeout
- schedule_to_start_timeout
- start_to_close_timeout
dacx"""

"""dacx
To create an Activity Retry Policy in Python, set the [RetryPolicy](https://python.temporal.io/temporalio.common.RetryPolicy.html) class within the [`start_activity()`](https://python.temporal.io/temporalio.workflow.html#start_activity) or [`execute_activity()`](https://python.temporal.io/temporalio.workflow.html#execute_activity) function.
dacx"""


@workflow.defn
class YourWorkflow:
    @workflow.run
    async def run(self, greeting: str) -> list[str]:
        activity_timeout_result = await workflow.execute_activity(
            your_activity,
            YourParams(greeting, "Activity Timeout option"),
            # Activity Execution Timeout
            start_to_close_timeout=timedelta(seconds=10),
            # schedule_to_start_timeout=timedelta(seconds=10),
            # schedule_to_close_timeout=timedelta(seconds=10),
        )
        activity_result = await workflow.execute_activity(
            your_activity,
            YourParams(greeting, "Retry Policy options"),
            start_to_close_timeout=timedelta(seconds=10),
            # Retry Policy
            retry_policy=RetryPolicy(
                backoff_coefficient=2.0,
                maximum_attempts=5,
                initial_interval=timedelta(seconds=1),
                maximum_interval=timedelta(seconds=2),
                # non_retryable_error_types=["ValueError"],
            ),
        )
        return activity_timeout_result, activity_result


""" @dacx
id: how-to-set-activity-timeouts-in-python
title: How to set Activity Timeouts in Python
label: Set Activity Timeouts
description: Set Activity timeouts from within your Workflow Definition.
tags:
 - activity
 - timeout
 - python sdk
 - code sample
lines: 9-17, 28-35
@dacx """

""" @dacx
id: how-to-set-an-activity-retry-policy-in-python
title: How to set an Activity Retry Policy in Python
label: Retry Policy
description: Create an instance of an Activity Retry Policy in Python.
tags:
 - activity
 - retry policy
 - python sdk
 - code sample
lines: 19-21, 4, 36-48
@dacx """
```

## Get the results of an Activity Execution

**How to get the results of an Activity Execution using the Temporal Python SDK.**

The call to spawn an Activity Execution generates the ScheduleActivityTask Command and provides the Workflow with an Awaitable. Workflow Executions can either block progress until the result is available through the Awaitable or continue progressing, making use of the result when it becomes available.

Use `start_activity()` to start an Activity and return its handle, `ActivityHandle`. Use `execute_activity()` to return the results.

You must provide either `schedule_to_close_timeout` or `start_to_close_timeout`.

`execute_activity()` is a shortcut for await `start_activity()`. An asynchronous `execute_activity()` helper is provided which takes the same arguments as `start_activity()` and awaits on the `result. execute_activity()` should be used in most cases unless advanced task capabilities are needed.


```
from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from your_activities_dacx import your_activity
    from your_dataobject_dacx import YourParams

"""dacx
To spawn an Activity Execution, use the [`execute_activity()`](https://python.temporal.io/temporalio.workflow.html#execute_activity) operation from within your Workflow Definition.

`execute_activity()` is a shortcut for [`start_activity()`](https://python.temporal.io/temporalio.workflow.html#start_activity) that waits on its result.

To get just the handle to wait and cancel separately, use `start_activity()`.
In most cases, use `execute_activity()` unless advanced task capabilities are needed.

A single argument to the Activity is positional. Multiple arguments are not supported in the type-safe form of `start_activity()` or `execute_activity()` and must be supplied by the `args` keyword argument.
dacx"""

"""dacx
You can customize the Workflow name with a custom name in the decorator argument. For example, `@workflow.defn(name="your-workflow-name")`. If the name parameter is not specified, the Workflow name defaults to the function name.
dacx"""

"""dacx
In the Temporal Python SDK programming model, Workflows are defined as classes.

Specify the `@workflow.defn` decorator on the Workflow class to identify a Workflow.

Use the `@workflow.run` to mark the entry point method to be invoked. This must be set on one asynchronous method defined on the same class as `@workflow.defn`. Run methods have positional parameters.
dacx"""

"""dacx
To return a value of the Workflow, use `return` to return an object.

To return the results of a Workflow Execution, use either `start_workflow()` or `execute_workflow()` asynchronous methods.
dacx"""

"""dacx
Use [`start_activity()`](https://python.temporal.io/temporalio.workflow.html#start_activity) to start an Activity and return its handle, [`ActivityHandle`](https://python.temporal.io/temporalio.workflow.ActivityHandle.html). Use [`execute_activity()`](https://python.temporal.io/temporalio.workflow.html#execute_activity) to return the results.

You must provide either `schedule_to_close_timeout` or `start_to_close_timeout`.

`execute_activity()` is a shortcut for `await start_activity()`. An asynchronous `execute_activity()` helper is provided which takes the same arguments as `start_activity()` and `await`s on the result. `execute_activity()` should be used in most cases unless advanced task capabilities are needed.
dacx"""


@workflow.defn(name="YourWorkflow")
class YourWorkflow:
    @workflow.run
    async def run(self, name: str) -> str:
        return await workflow.execute_activity(
            your_activity,
            YourParams("Hello", name),
            start_to_close_timeout=timedelta(seconds=10),
        )


""" @dacx
id: how-to-spawn-an-activity-execution-in-python
title: How to spawn an Activity Execution in Python
label: Activity Execution
description: Use the `execute_activity()` operation from within your Workflow Definition.
lines: 3, 9-18, 47-55
@dacx """


""" @dacx
id: how-to-customize-workflow-type-in-python
title: How to customize Workflow types in Python
label: Customize Workflow types
description: Customize Workflow types.
lines: 3, 20-22, 47-55
@dacx """

""" @dacx
id: how-to-develop-a-workflow-definition-in-python
title: How to develop a Workflow Definition in Python
label: Develop a Workflow Definition
description: To develop a Workflow Definition, specify the `@workflow.defn` decorator on the Workflow class and use `@workflow.run` to mark the entry point.
lines: 3, 24-30, 47-55
@dacx """


""" @dacx
id: how-to-define-workflow-return-values-in-python
title: How to define Workflow return values
label: Define Workflow return values
description: Define Workflow return values.
tags:
 - workflow return values
lines: 3, 32-36, 47-55
@dacx """


""" @dacx
id: how-to-get-the-result-of-an-activity-execution-in-python
title: How to get the result of an Activity Execution in Python
label: Get the result of an Activity Execution
description: Get the result of an Activity Execution.
tags:
 - activity execution
 - python sdk
 - code sample
lines: 3, 38-44, 47-55
@dacx """
```

## Run a Worker Process

**How to run a Worker Process using the Temporal Python SDK.**

The `Worker Process` is where Workflow Functions and Activity Functions are executed.

- Each `Worker Entity` in the Worker Process must register the exact Workflow Types and Activity Types it may execute.
- Each Worker Entity must also associate itself with exactly one `Task Queue`.
- Each Worker Entity polling the same Task Queue must be registered with the same Workflow Types and Activity Types.
- A Worker Entity is the component within a Worker Process that listens to a specific Task Queue.
- Although multiple Worker Entities can be in a single Worker Process, a single Worker Entity Worker Process may be perfectly sufficient. 
- A Worker Entity contains a Workflow Worker and/or an Activity Worker, which makes progress on Workflow Executions and Activity Executions, respectively.
- To develop a Worker, use the Worker() constructor and add your Client, Task Queue, Workflows, and Activities as arguments. The following code example creates a Worker that polls for tasks from the Task Queue and executes the Workflow. When a Worker is created, it accepts a list of Workflows in the workflows parameter, a list of Activities in the activities parameter, or both.


```
import asyncio

from temporalio.client import Client
from temporalio.worker import Worker
from your_activities_dacx import your_activity
from your_workflows_dacx import YourWorkflow

"""dacx
To develop a Worker, use the `Worker()` constructor and add your Client, Task Queue, Workflows, and Activities as arguments.
The following code example creates a Worker that polls for tasks from the Task Queue and executes the Workflow.
When a Worker is created, it accepts a list of Workflows in the workflows parameter, a list of Activities in the activities parameter, or both.
dacx"""

"""dacx
When a `Worker` is created, it accepts a list of Workflows in the `workflows` parameter, a list of Activities in the `activities` parameter, or both.
dacx"""


async def main():
    client = await Client.connect("localhost:7233")
    worker = Worker(
        client,
        task_queue="your-task-queue",
        workflows=[YourWorkflow],
        activities=[your_activity],
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())


""" @dacx
id: how-to-develop-a-worker-program-in-python
title: How to develop a Worker Program in Python
label: Worker Program
description: Create a new instance of a Worker.
tags:
 - worker
 - python sdk
 - code sample
lines: 3-4, 8-12, 19-31
@dacx """

""" @dacx
id: how-to-register-types-with-a-worker-in-python
title: How to register types with a Worker in Python
label: Register types with a Worker
description: Register types with a Worker.
tags:
 - worker
 - python sdk
 - code sample
lines: 14-16, 19-31
@dacx """
```

## Register types

**How to register types using the Temporal Python SDK.**

All Workers listening to the same Task Queue name must be registered to handle the exact same Workflows Types and Activity Types.

When a `Worker` is created, it accepts a list of `Workflows` in the `workflows` parameter, a list of Activities in the activities parameter, or both.

```
import asyncio

from temporalio.client import Client
from temporalio.worker import Worker
from your_activities_dacx import your_activity
from your_workflows_dacx import YourWorkflow

"""dacx
To develop a Worker, use the `Worker()` constructor and add your Client, Task Queue, Workflows, and Activities as arguments.
The following code example creates a Worker that polls for tasks from the Task Queue and executes the Workflow.
When a Worker is created, it accepts a list of Workflows in the workflows parameter, a list of Activities in the activities parameter, or both.
dacx"""

"""dacx
When a `Worker` is created, it accepts a list of Workflows in the `workflows` parameter, a list of Activities in the `activities` parameter, or both.
dacx"""


async def main():
    client = await Client.connect("localhost:7233")
    worker = Worker(
        client,
        task_queue="your-task-queue",
        workflows=[YourWorkflow],
        activities=[your_activity],
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())


""" @dacx
id: how-to-develop-a-worker-program-in-python
title: How to develop a Worker Program in Python
label: Worker Program
description: Create a new instance of a Worker.
tags:
 - worker
 - python sdk
 - code sample
lines: 3-4, 8-12, 19-31
@dacx """

""" @dacx
id: how-to-register-types-with-a-worker-in-python
title: How to register types with a Worker in Python
label: Register types with a Worker
description: Register types with a Worker.
tags:
 - worker
 - python sdk
 - code sample
lines: 14-16, 19-31
@dacx """
```

