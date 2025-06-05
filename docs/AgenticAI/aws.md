## AWS Agentic AI Services

![alt text](./image/aws-1.png)

![alt text](./image/aws-2.png)

![alt text](./image/aws-3.png)


![alt text](./image/aws-4.png)

![alt text](./image/aws-5.png)


[Amazon Bedrock Agents](https://www.youtube.com/watch?v=6O9DqCrInvw)

[Amazon-sagemaker](https://aws.amazon.com/blogs/machine-learning/build-agentic-ai-solutions-with-deepseek-r1-crewai-and-amazon-sagemaker-ai/)

| **Service**               | **Description**                                               | **Service Type**                  | **SaaS / Shelf-Managed** | **Use Case Example**                                          |
| ------------------------- | ------------------------------------------------------------- | --------------------------------- | ------------------------ | ------------------------------------------------------------- |
| Amazon Rekognition        | Computer vision for image and video analysis                  | Computer Vision                   | SaaS                     | Detect faces and objects in surveillance footage              |
| Amazon Transcribe         | Speech-to-text for real-time or batch transcription           | Speech Recognition                | SaaS                     | Transcribe customer service calls                             |
| Amazon Translate          | Neural machine translation                                    | Translation / NLP                 | SaaS                     | Translate product descriptions for global marketplaces        |
| Amazon Polly              | Text-to-speech for lifelike speech generation                 | Speech Synthesis                  | SaaS                     | Generate lifelike voices for e-learning apps                  |
| Amazon Comprehend         | NLP for text insights like sentiment analysis                 | Natural Language Processing       | SaaS                     | Analyze customer reviews for sentiment and key phrases        |
| Amazon Textract           | Text and data extraction from documents                       | Document Processing               | SaaS                     | Extract tables and fields from scanned invoices               |
| Amazon Personalize        | Personalized recommendations and user segmentation            | Recommendation System             | SaaS                     | Product recommendation for e-commerce platform                |
| Amazon Augmented AI (A2I) | Human review of ML predictions                                | Human-in-the-loop (HITL)          | Shelf-Managed            | Review flagged document classifications in loan processing    |
| Amazon Bedrock            | Access to foundation models for generative AI apps            | Generative AI / Foundation Models | SaaS                     | Build a chatbot using Anthropic, Mistral, or Meta models      |
| Amazon Q                  | Generative AI assistant for development and business insights | Generative AI Assistant           | SaaS                     | Get coding help or business data insights via natural queries |
| Amazon SageMaker          | Comprehensive platform for ML and foundation models           | ML Platform / MLOps               | Shelf-Managed            | Train, deploy, and monitor custom ML models                   |
| Amazon CodeGuru           | ML for code analysis and optimization                         | DevOps / Code Quality             | SaaS                     | Detect bugs and optimize code performance                     |
| Amazon DevOps Guru        | ML for operational data analysis and issue resolution         | DevOps / AIOps                    | SaaS                     | Detect anomalies in application performance metrics           |
| AWS HealthLake            | HIPAA-eligible for healthcare data management                 | Healthcare Data Platform          | Shelf-Managed            | Normalize and analyze patient health records                  |
| Amazon Lex                | Conversational interfaces like chatbots and voice assistants  | Conversational AI                 | SaaS                     | Create a customer support chatbot                             |


## Sample Agentic AI Workflow

To set up a Multi-Agent System using AWS Bedrock with a real working example, you can follow this structured, practical approach using AWS Bedrock + AWS Lambda + AWS Step Functions (no external server needed). This example will simulate agents working together in a typical IT Support scenario:

## 🎯 Use Case: IT Support Chatbot with Multi-Agents
## 🧠 Agents:
1. **Classifier Agent** – Classifies user intent (e.g., knowledge/action).
2. **Knowledge Agent** – Answers general IT questions.
3. **Action Agent** – Simulates action (e.g., reset password).

## 🛠️ What You’ll Build
Using **AWS Console (UI)**:

- Use **Amazon Bedrock** to call Claude/Titan models.
- Use **AWS Lambda** to create agents as serverless functions.
- Use **AWS Step Functions** to orchestrate agent flow based on logic.


## ✅ Step-by-Step Setup (UI + Lambda + Bedrock)

## 🔹 STEP 1: Enable Amazon Bedrock Models
1. Go to: **Amazon Bedrock > Model Access**
2. Enable Claude (Anthropic) or any model you'd like (Titan, Mistral, Llama)

## 🔹 STEP 2: Create IAM Role with Bedrock Access
1. Go to **IAM > Roles > Create Role**
2. Choose: Lambda
3. Attach Policy: AmazonBedrockFullAccess + AWSLambdaBasicExecutionRole
4. Name the role: LambdaBedrockRole


## 🔹 STEP 3: Create 3 Lambda Functions (1 per Agent)
## A. Classifier Agent Lambda
1. Name: classifierAgent
2. Runtime: Python 3.12
3. Use this code

## code

```
import boto3
import json

def lambda_handler(event, context):
    prompt = f"Classify this request into either 'knowledge' or 'action': {event['user_input']}"
    
    bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

    response = bedrock.invoke_model(
        modelId='amazon.titan-text-premier-v1:0',
        contentType='application/json',
        accept='application/json',
        body=json.dumps({
            "inputText": prompt
        })
    )

    result = json.loads(response['body'].read())
    classification = result['results'][0]['outputText'].strip().lower()

    return { "classification": classification }
```

## B. Knowledge Agent Lambda
1. Name: KnowledgeAgent
2. Runtime: Python 3.12
3. Use this code

```
import boto3
import json

def lambda_handler(event, context):
    prompt = f"You are an IT support expert. Answer the user's question: {event['user_input']}"
    
    bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

    response = bedrock.invoke_model(
        modelId='amazon.titan-text-premier-v1:0',
        contentType='application/json',
        accept='application/json',
        body=json.dumps({
            "inputText": prompt
        })
    )

    result = json.loads(response['body'].read())
    
    return {
        "agent": "knowledge",
        "response": result['results'][0]['outputText'].strip()
```

## C. Action Agent Lambda
1. Name: ActionAgent
2. Runtime: Python 3.12
3. Use this code

```
import boto3
import json

def lambda_handler(event, context):
    prompt = f"You are an IT support automation bot. Respond to this action request: {event['user_input']}"
    
    bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')  # include region

    response = bedrock.invoke_model(
        modelId='amazon.titan-text-premier-v1:0',
        contentType='application/json',
        accept='application/json',
        body=json.dumps({
            "inputText": prompt
        })
    )

    result = json.loads(response['body'].read())

    return {
        "agent": "action",
        "response": result['results'][0]['outputText'].strip()
    }
```

## 🔹 STEP 4: Create a Step Function for Orchestration
   Go to **AWS Step Functions > Create State Machine**
   Choose **Author with Code Snippet**
   Paste the following Amazon States Language (ASL):


```
{
  "Comment": "Multi-Agent Orchestration for IT Support",
  "StartAt": "ClassifierAgent",
  "States": {
    "ClassifierAgent": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:us-east-1:777203855866:function:classifierAgent",
      "ResultPath": "$.classificationResult",
      "Next": "CheckClassification"
    },
    "CheckClassification": {
      "Type": "Choice",
      "Choices": [
        {
          "Variable": "$.classificationResult.classification",
          "StringMatches": "*knowledge*",
          "Next": "KnowledgeAgent"
        },
        {
          "Variable": "$.classificationResult.classification",
          "StringMatches": "*action*",
          "Next": "ActionAgent"
        }
      ],
      "Default": "UnknownClassification"
    },
    "KnowledgeAgent": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:us-east-1:777203855866:function:KnowledgeAgent",
      "ResultPath": "$.response",
      "End": true
    },
    "ActionAgent": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:us-east-1:777203855866:function:ActionAgent",
      "ResultPath": "$.response",
      "End": true
    },
    "UnknownClassification": {
      "Type": "Fail",
      "Error": "InvalidClassification",
      "Cause": "Could not classify the user request"
    }
  }
}
```

## 🔹 STEP 5: Test the Multi-Agent System
   Go to Step Functions > Your State Machine
   Click Start Execution
   Use this test input:


```
{
  "user_input": "How can I install VPN on my laptop?"
}
```

![Agenticai](./image/agentic-1.png)


![Agenticai](./image/agentic-2.png)

![Agenticai](./image/agentic-3.png)



# Building a Simple Agent Using AWS Bedrock

## Step 1: Create a Lambda Function

First, create a Lambda function that your agent will invoke to perform actions. In this procedure, you'll create a Python Lambda function that returns the current date and time when invoked. You'll set up the function with basic permissions, add the necessary code to handle requests from your Amazon Bedrock agent, and deploy the function so it's ready to be connected to your agent.

---

### Create a Lambda Function

1. Sign in to the AWS Management Console and open the Lambda console at  
   [https://console.aws.amazon.com/lambda/](https://console.aws.amazon.com/lambda/)

2. Choose **Create function**.

3. Select **Author from scratch**.

4. In the **Basic information** section:
   - For **Function name**, enter a function name (for example, `DateTimeFunction`).
   - For **Runtime**, select **Python 3.9** (or your preferred version).
   - For **Architecture**, leave unchanged.
   - In **Permissions**, select **Change default execution role** and then select **Create a new role with basic Lambda permissions**.

5. Choose **Create function**.

6. In **Function overview**, under **Function ARN**, note the Amazon Resource Name (ARN) for the function.

7. In the **Code** tab, replace the existing code with the following:

    

          import datetime
          import json

          def lambda_handler(event, context):
              now = datetime.datetime.now()

              response = {
                  "date": now.strftime("%Y-%m-%d"),
                  "time": now.strftime("%H:%M:%S")
              }

              response_body = {
                  "application/json": {
                      "body": json.dumps(response)
                  }
              }

              action_response = {
                  "actionGroup": event["actionGroup"],
                  "apiPath": event["apiPath"],
                  "httpMethod": event["httpMethod"],
                  "httpStatusCode": 200,
                  "responseBody": response_body,
              }

              session_attributes = event["sessionAttributes"]
              prompt_session_attributes = event["promptSessionAttributes"]

              return {
                  "messageVersion": "1.0",
                  "response": action_response,
                  "sessionAttributes": session_attributes,
                  "promptSessionAttributes": prompt_session_attributes,
              }
    

8. Choose **Deploy** to deploy your function.

9. Choose the **Configuration** tab.

10. Choose **Permissions**.

11. Under **Resource-based policy statements**, choose **Add permissions**.

12. In **Edit policy statement**, do the following:
    - a. Choose **AWS service**
    - b. In **Service**, select **Other**.
    - c. For **Statement ID**, enter a unique identifier (for example, `AllowBedrockInvocation`).
    - d. For **Principal**, enter `bedrock.amazonaws.com`.
    - e. For **Source ARN**, enter:

        ```
        arn:aws:bedrock:<region>:<AWS account ID>:agent/*
        ```

        Replace `<region>` with your AWS Region, such as `us-east-1`.  
        Replace `<AWS account ID>` with your actual AWS account ID.

13. Choose **Save**.


# Building a Simple Agent Using AWS Bedrock

## Step 2: Create a Bedrock Agent

### 1. Sign in and Open Bedrock Console

- Sign in to the [AWS Management Console](https://console.aws.amazon.com/) using an **IAM role with Amazon Bedrock permissions**.
- Navigate to the [Amazon Bedrock console](https://console.aws.amazon.com/bedrock/).
- Ensure you're in an **AWS Region** that supports **Amazon Bedrock agents**.

### 2. Create an Agent

1. In the left navigation pane under **Builder tools**, choose **Agents**.
2. Choose **Create agent**.
3. Fill in the following:
   - **Name**: (e.g., `MyBedrockAgent`)
   - **Description** (optional)
4. Choose **Create**. The **Agent builder** pane opens.

### 3. Configure Agent Details

- In the **Agent details** section:
  - For **Agent resource role**, select **Create and use a new service role**.
  - For **Select model**, choose a model (e.g., `Claude 3 Haiku`).
  - In **Instructions for the Agent**, paste the following:

    ```
    You are a friendly chat bot. You have access to a function called that returns
    information about the current date and time. When responding with date or time,
    please make sure to add the timezone UTC.
    ```

- Choose **Save**.

---

## Step 3: Add Action Group

### 1. Navigate to Action Groups

- Choose the **Action groups** tab.
- Choose **Add**.

### 2. Configure Action Group

- **Action group name**: (e.g., `TimeActions`)
- **Description** (optional)
- **Action group type**: Select **Define with API schemas**
- **Action group invocation**: Choose **Select an existing Lambda function**
- **Select Lambda function**: Choose the Lambda function created in [Step 1](#step-1-create-a-lambda-function)
- **Action group schema**: Choose **Define via in-line schema editor**

### 3. Paste OpenAPI Schema

Replace the existing schema with:

```yaml
openapi: 3.0.0
info:
  title: Time API
  version: 1.0.0
  description: API to get the current date and time.
paths:
  /get-current-date-and-time:
    get:
      summary: Gets the current date and time.
      description: Gets the current date and time.
      operationId: getDateAndTime
      responses:
        '200':
          description: Gets the current date and time.
          content:
            'application/json':
              schema:
                type: object
                properties:
                  date:
                    type: string
                    description: The current date
                  time:
                    type: string
                    description: The current time
```

### 4. Finalize Agent and Add Permissions

1. **Review** your action group configuration and choose **Create**.
2. Choose **Save** to save your changes.
3. Choose **Prepare** to prepare the agent.
4. Choose **Save and exit** to save your changes and exit the agent builder.

---

## Step 6: Grant Lambda Invoke Permissions to the Agent

1. In the **Agent overview** section, under **Permissions**, click the **IAM service role**. This opens the role in the IAM console.

2. In the IAM console:
   - Choose the **Permissions** tab.
   - Choose **Add permissions** → **Create inline policy**.

3. Choose the **JSON** tab and paste the following policy:

    ```json
    {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "lambda:InvokeFunction"
                ],
                "Resource": "Function ARN"
            }
        ]
    }
    ```

    > 🔄 **Note**: Replace `"Function ARN"` with the ARN of your Lambda function from Step 6 of [Step 1: Create a Lambda Function](#step-1-create-a-lambda-function)

4. Choose **Next**.

5. Enter a name for the policy (e.g., `BedrockAgentLambdaInvoke`).

6. Choose **Create policy**.


# Agentic AI - Usecase 1: ```Slow response times on network causing OTP relay delays for Banking customers. How does Agentic AI identify and apply fixes.```

## Step 1: Create a Lambda Function

**Function Name:** ```OtpMonitorLambdaFunction```

**Code:**

```
import boto3
import re
import json
from datetime import datetime, timedelta

logs_client = boto3.client('logs')

def lambda_handler(event, context):
    log_group_name = event.get("log_group_name", "/eks/otp-webapp")
    end_time = int(datetime.utcnow().timestamp() * 1000)
    start_time = int((datetime.utcnow() - timedelta(hours=48)).timestamp() * 1000)

    patterns = [
        '"Attempting to send OTP"',
        '"OTP email sent in"',
        '"Failed to send OTP"'
    ]

    all_events = {}
    delivery_times = []
    failed_otp_count = 0

    try:
        for pattern in patterns:
            response = logs_client.filter_log_events(
                logGroupName=log_group_name,
                startTime=start_time,
                endTime=end_time,
                filterPattern=pattern,
                limit=500
            )
            for event_item in response.get('events', []):
                message = event_item['message']
                timestamp = datetime.utcfromtimestamp(event_item['timestamp'] / 1000).strftime('%Y-%m-%d %H:%M:%S')

                if "OTP email sent in" in message:
                    match = re.search(r'OTP email sent in ([\d.]+) seconds', message)
                    if match:
                        delivery_times.append(float(match.group(1)))

                if "Failed to send OTP" in message:
                    failed_otp_count += 1

                all_events[event_item['eventId']] = {
                    'timestamp': timestamp,
                    'message': message
                }

        logs = list(all_events.values())
        max_delivery = max(delivery_times) if delivery_times else 0.0

        status = "INFO"
        if failed_otp_count > 0 or max_delivery > 2.5:
            status = "WARNING"
        if failed_otp_count >= 3 or max_delivery > 5.0:
            status = "CRITICAL"

        summary = {
            "status": status,
            "affected_services": ["Lambda"],
            "metric_alerts": [
                {
                    "metric": "Max OTP Delivery Time",
                    "value": f"{max_delivery:.2f}s",
                    "threshold": "2.5s",
                    "service": "Lambda"
                },
                {
                    "metric": "Failed OTP Count",
                    "value": failed_otp_count,
                    "threshold": "0",
                    "service": "Lambda"
                }
            ],
            "summary": f"Found {len(logs)} OTP logs in the last 12 hours. Failures: {failed_otp_count}, Max delivery time: {max_delivery:.2f}s"
        }
        
        response_body = {"application/json": {"body": json.dumps(summary)}}

        action_response = {
            "actionGroup": event["actionGroup"],
            "apiPath": event["apiPath"],
            "httpMethod": event["httpMethod"],
            "httpStatusCode": 200,
            "responseBody": response_body,
        }

        session_attributes = event["sessionAttributes"]
        prompt_session_attributes = event["promptSessionAttributes"]

        return {
            "messageVersion": "1.0",
            "response": action_response,
            "sessionAttributes": session_attributes,
            "promptSessionAttributes": prompt_session_attributes,
        }

    except Exception as e:
        error_response_body = {
            "application/json": {
                "body": json.dumps({
                    "error": "Lambda Error",
                    "message": str(e)
                })
            }
        }

        action_response = {
            "actionGroup": event.get("actionGroup", "Unknown"),
            "apiPath": event.get("apiPath", "/unknown"),
            "httpMethod": event.get("httpMethod", "GET"),
            "httpStatusCode": 500,
            "responseBody": error_response_body,
        }

        return {
            "messageVersion": "1.0",
            "response": action_response,
            "sessionAttributes": event.get("sessionAttributes", {}),
            "promptSessionAttributes": event.get("promptSessionAttributes", {}),
        }
```

**Test:**
Event JSON

```
{
  "log_group_name": "/eks/otp-webapp",
  "actionGroup": "OtpMonitorActionGroup",
  "apiPath": "/get-otp-monitor",
  "httpMethod": "GET",
  "sessionAttributes": {},
  "promptSessionAttributes": {}
}
```

**Executing function: succeeded:**

```
{
  "messageVersion": "1.0",
  "response": {
    "actionGroup": "OtpMonitorActionGroup",
    "apiPath": "/get-otp-monitor",
    "httpMethod": "GET",
    "httpStatusCode": 200,
    "responseBody": {
      "application/json": {
        "body": "{\"status\": \"WARNING\", \"affected_services\": [\"Lambda\"], \"metric_alerts\": [{\"metric\": \"Max OTP Delivery Time\", \"value\": \"3.38s\", \"threshold\": \"2.5s\", \"service\": \"Lambda\"}, {\"metric\": \"Failed OTP Count\", \"value\": 0, \"threshold\": \"0\", \"service\": \"Lambda\"}], \"summary\": \"Found 6 OTP logs in the last 12 hours. Failures: 0, Max delivery time: 3.38s\"}"
      }
    }
  },
  "sessionAttributes": {},
  "promptSessionAttributes": {}
}
```

**Configuration:**
**Permissions:**
**Role name:**
OtpMonitorLambdaFunction-role-ntyfvqf0
IAM -> Roles -> OtpMonitorLambdaFunction-role-ntyfvqf0

Policy name:
AWSLambdaBasicExecutionRole-9a1d489a-eda2-4c6e-bced-f99a004d9617
IAM -> Policies -> AWSLambdaBasicExecutionRole-9a1d489a-eda2-4c6e-bced-f99a004d9617

Service:
CloudWatch Logs

```
{
	"Version": "2012-10-17",
	"Statement": [
		{
			"Effect": "Allow",
			"Action": "logs:CreateLogGroup",
			"Resource": "arn:aws:logs:us-east-1:777203855866:*"
		},
		{
			"Effect": "Allow",
			"Action": [
				"logs:CreateLogStream",
				"logs:PutLogEvents"
			],
			"Resource": [
				"arn:aws:logs:us-east-1:777203855866:log-group:/aws/lambda/OtpMonitorLambdaFunction:*"
			]
		},
		{
			"Effect": "Allow",
			"Action": [
				"logs:FilterLogEvents",
				"logs:GetLogEvents",
				"logs:DescribeLogStreams"
			],
			"Resource": [
				"arn:aws:logs:us-east-1:777203855866:log-group:/eks/otp-webapp:*"
			]
		}
	]
}
```

**Resource-based policy statements**
Resource-based policies grant other AWS accounts and services permissions to access your Lambda resources.

Statement ID:
AllowBedrockInvocation

Resource-based policy document

```
{
  "Version": "2012-10-17",
  "Id": "default",
  "Statement": [
    {
      "Sid": "AllowBedrockInvocation",
      "Effect": "Allow",
      "Principal": {
        "Service": "bedrock.amazonaws.com"
      },
      "Action": "lambda:InvokeFunction",
      "Resource": "arn:aws:lambda:us-east-1:777203855866:function:OtpMonitorLambdaFunction",
      "Condition": {
        "ArnLike": {
          "AWS:SourceArn": "arn:aws:bedrock:us-east-1:777203855866:agent/*"
        }
      }
    }
  ]
}
```

![Lambda Function](./image/LF1.png)

![Lambda Function](./image/LF2.png)

![Lambda Function](./image/LF3.png)

![Lambda Function](./image/LF4.png)

![Lambda Function](./image/LF5.png)

![Lambda Function](./image/LF6.png)

![Lambda Function](./image/LF7.png)


## Step 2: Create a Amazon Bedrock Agent

![Agent](./image/Agent1.png)

**Agent details**
**Agent name:** OtpMonitorAgent
**Agent description - optional:** OTP CloudWatch logs monitor Agent
**Agent resource role:** Create and use a new service role
**Select model:** select from Model providers list
**Instructions for the Agent:** Provide clear and specific instructions for the task the Agent will perform. You can also provide certain style and tone.

```
Your role is to analyze the output of a Lambda function that queries AWS CloudWatch logs from the log

  Focus on the following OTP flow log messages:
  - "Generated OTP"
  - "Attempting to send OTP"
  - "OTP email sent"
  - "OTP stored"
  - "OTP verified"

Parse and track OTP-related events in **chronological order** per transaction.
Expected order:
- Generated OTP → Attempting to send OTP → OTP email sent → OTP stored → OTP verified

 Detect and flag the following anomalies:
     - Missing events in the expected sequence
     - Time delays > 10 seconds between any two consecutive steps
     - Presence of known error patterns (e.g., "SMTP error", "send failure", "OTP failed", "storage error")
```

**Action groups:**
**Action group details**
**Enter Action group name:** OtpMonitorActionGroup
**Description - optional:**

**Action group type:**
Select what type of action group to create: ```Define with API schemas```

**Action group invocation:** 
Specify a Lambda function that will be invoked based on the action group identified by the Foundation model during orchestration.

Select an existing Lambda function: ```OtpMonitorLambdaFunction```

**Action group schema:**
Select an existing schema or create a new one via the in-line editor to define the APIs that the agent can invoke to carry out its tasks.

- ```Define via in-line schema editor```

**In-line OpenAPI schema:**

```
openapi: 3.0.0
info:
  title: OTP Monitoring API
  version: 1.0.0
  description: API to monitor OTP delivery delays and failures from CloudWatch logs.
paths:
  /get-otp-monitor:
    get:
      summary: Gets OTP delivery monitoring data.
      description: Retrieves OTP delivery times, failure counts, and status based on CloudWatch logs for the past 12 hours.
      operationId: getOtpMonitoringInfo
      responses:
        '200':
          description: OTP delivery metrics and summary.
          content:
            application/json:
              schema:
                type: object
                properties:
                  status:
                    type: string
                    description: Overall status based on OTP metrics (INFO, WARNING, CRITICAL)
                  affected_services:
                    type: array
                    items:
                      type: string
                    description: Services affected (e.g., Lambda)
                  metric_alerts:
                    type: array
                    description: List of metric evaluations
                    items:
                      type: object
                      properties:
                        metric:
                          type: string
                          description: Metric name
                        value:
                          type: string
                          description: Observed value
                        threshold:
                          type: string
                          description: Threshold value
                        service:
                          type: string
                          description: Related service
                  summary:
                    type: string
                    description: Summary message about the log findings
```

**Permissions:** ```arn:aws:iam::777203855866:role/service-role/AmazonBedrockExecutionRoleForAgents_589IDLBHO1U```

![Agent](./image/Agent8.png)

IAM -> Roles -> AmazonBedrockExecutionRoleForAgents_589IDLBHO1U

![Agent](./image/Agent9.png)

IAM -> Policies -> AmazonBedrockAgentBedrockFoundationModelPolicy_TU2VOQK6HG

![Agent](./image/Agent10.png)

**Permissions defined in this policy**

```
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AmazonBedrockAgentBedrockFoundationModelPolicyProd",
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream"
            ],
            "Resource": [
                "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-text-premier-v1:0"
            ]
        }
    ]
}
```

IAM -> Roles -> AmazonBedrockExecutionRoleForAgents_589IDLBHO1U

**permissions in OtpMonitorPolicy**

```
{
	"Version": "2012-10-17",
	"Statement": [
		{
			"Effect": "Allow",
			"Action": [
				"lambda:InvokeFunction"
			],
			"Resource": "arn:aws:lambda:us-east-1:777203855866:function:OtpMonitorLambdaFunction"
		}
	]
}
```



**Action status:** Enable

![Agent](./image/Agent2.png)

![Agent](./image/Agent3.png)

![Agent](./image/Agent4.png)

![Agent](./image/Agent5.png)

![Agent](./image/Agent6.png)

![Agent](./image/Agent7.png)

## Note:
- **Save** -> **Prepare** -> **Test** -> **Create alias**

## Test the ```OtpMonitorAgent```

Here’s a test prompt you can use to trigger your AWS Bedrock Agent (which calls the Lambda OtpMonitorActionGroup.getOtpMonitoringInfoGET) correctly:

## ✅ Test Prompt

```
Check the OTP delivery metrics.
```

![Agent](./image/Agent11.png)

## Trace step 1

```
{
  "agentId": "MZJDM5Z9N3",
  "callerChain": [
    {
      "agentAliasArn": "arn:aws:bedrock:us-east-1:777203855866:agent-alias/MZJDM5Z9N3/TSTALIASID"
    }
  ],
  "eventTime": "2025-06-05T12:10:31.591Z",
  "modelInvocationInput": {
    "foundationModel": "amazon.titan-text-premier-v1:0",
    "inferenceConfiguration": {
      "maximumLength": 2048,
      "stopSequences": [],
      "temperature": 0,
      "topK": 1,
      "topP": 1.000000013351432e-10
    },
    "text": "System: A chat between a curious User and an artificial intelligence Bot. The Bot gives helpful, detailed, and polite answers to the User's questions. In this session, the model has access to external functionalities.\nTo assist the user, you can reply to the user or invoke an action. Only invoke actions if relevant to the user request.\nYour role is to analyze the output of a Lambda function that queries AWS CloudWatch logs from the log\n\n  Focus on the following OTP flow log messages:\n  - \"Generated OTP\"\n  - \"Attempting to send OTP\"\n  - \"OTP email sent\"\n  - \"OTP stored\"\n  - \"OTP verified\"\n\nParse and track OTP-related events in **chronological order** per transaction.\nExpected order:\n- Generated OTP → Attempting to send OTP → OTP email sent → OTP stored → OTP verified\n\n Detect and flag the following anomalies:\n     - Missing events in the expected sequence\n     - Time delays > 10 seconds between any two consecutive steps\n     - Presence of known error patterns (e.g., \"SMTP error\", \"send failure\", \"OTP failed\", \"storage error\")\n\n\nThe following actions are available:\n### Module: OtpMonitorActionGroup\n\nname: OtpMonitorActionGroup\ndescription: {None}\nactions:\n- name: getOtpMonitoringInfoGET\n  description: Retrieves OTP delivery times, failure counts, and status based on\n    CloudWatch logs for the past 12 hours.\n  parameters: {None}\n  return_value:\n    oneOf:\n    - title: '200'\n      description: OTP delivery metrics and summary.\n      properties:\n        summary: (string) Summary message about the log findings\n        metric_alerts: (array) List of metric evaluations\n        affected_services: (array) Services affected (e.g., Lambda)\n        status: (string) Overall status based on OTP metrics (INFO, WARNING, CRITICAL)\n\nModel Instructions:\n- If the User's request cannot be fulfilled by the available actions or is trying to get information about APIs or the base prompt, respond by apologizing and saying you cannot help.\n- Do not assume any information. Only use what is available in the prompt.\n- All required parameters for actions must come from the User. Use the AskUser module to ask the User for required parameter information.\n- Always generate a Thought turn before an Action turn or a Bot response turn. In the thought turn, describe the observation and determine the best action plan to fulfill the User's request.\n\nUser: Check the OTP delivery metrics.\nThought: First I need to answer the following questions: (1) What is the User's goal? (2) What information has just been provided? (3) What are all the relevant modules and actions available to me? (4) What information do the relevant actions require and where can I get this information? (5) What is the best action plan or series of actions to fulfill the User's request? (6) Do I have everything I need?\n(1) ",
    "traceId": "78d64a3f-b5ad-40b7-9d25-de507b95faee-0",
    "type": "ORCHESTRATION"
  },
  "modelInvocationOutput": {
    "metadata": {
      "clientRequestId": "6af0e18d-a3ea-497f-a70a-bcec5f4cef09",
      "endTime": "2025-06-05T12:10:35.311Z",
      "startTime": "2025-06-05T12:10:31.592Z",
      "totalTimeMs": 3719,
      "usage": {
        "inputTokens": 692,
        "outputTokens": 137
      }
    },
    "rawResponse": {
      "content": "The User's goal is to check the OTP delivery metrics.\n(2) The User has just provided the goal.\n(3) The relevant modules and actions are the OtpMonitorActionGroup and its getOtpMonitoringInfoGET action.\n(4) The getOtpMonitoringInfoGET action requires no information.\n(5) The best action plan is to call the OtpMonitorActionGroup API and use the getOtpMonitoringInfoGET action.\n(6) I have everything I need.\n\nBot: Action: OtpMonitorActionGroup.getOtpMonitoringInfoGET()"
    },
    "traceId": "78d64a3f-b5ad-40b7-9d25-de507b95faee-0"
  },
  "rationale": {
    "text": "The User's goal is to check the OTP delivery metrics.\n(2) The User has just provided the goal.\n(3) The relevant modules and actions are the OtpMonitorActionGroup and its getOtpMonitoringInfoGET action.\n(4) The getOtpMonitoringInfoGET action requires no information.\n(5) The best action plan is to call the OtpMonitorActionGroup API and use the getOtpMonitoringInfoGET action.\n(6) I have everything I need.",
    "traceId": "78d64a3f-b5ad-40b7-9d25-de507b95faee-0"
  },
  "invocationInput": [
    {
      "actionGroupInvocationInput": {
        "actionGroupName": "OtpMonitorActionGroup",
        "apiPath": "/get-otp-monitor",
        "executionType": "LAMBDA",
        "verb": "get"
      },
      "invocationType": "ACTION_GROUP",
      "traceId": "78d64a3f-b5ad-40b7-9d25-de507b95faee-0"
    }
  ],
  "observation": [
    {
      "actionGroupInvocationOutput": {
        "metadata": {
          "clientRequestId": "675775c3-6562-4182-a7af-9db3e79f01f8",
          "endTime": "2025-06-05T12:10:43.272Z",
          "startTime": "2025-06-05T12:10:35.313Z",
          "totalTimeMs": 7959
        },
        "text": "{\"status\": \"WARNING\", \"affected_services\": [\"Lambda\"], \"metric_alerts\": [{\"metric\": \"Max OTP Delivery Time\", \"value\": \"3.38s\", \"threshold\": \"2.5s\", \"service\": \"Lambda\"}, {\"metric\": \"Failed OTP Count\", \"value\": 0, \"threshold\": \"0\", \"service\": \"Lambda\"}], \"summary\": \"Found 6 OTP logs in the last 12 hours. Failures: 0, Max delivery time: 3.38s\"}"
      },
      "traceId": "78d64a3f-b5ad-40b7-9d25-de507b95faee-0",
      "type": "ACTION_GROUP"
    }
  ]
}
```

## Trace step 2

```
{
  "agentId": "MZJDM5Z9N3",
  "callerChain": [
    {
      "agentAliasArn": "arn:aws:bedrock:us-east-1:777203855866:agent-alias/MZJDM5Z9N3/TSTALIASID"
    }
  ],
  "eventTime": "2025-06-05T12:10:43.274Z",
  "modelInvocationInput": {
    "foundationModel": "amazon.titan-text-premier-v1:0",
    "inferenceConfiguration": {
      "maximumLength": 2048,
      "stopSequences": [],
      "temperature": 0,
      "topK": 1,
      "topP": 1.000000013351432e-10
    },
    "text": "System: A chat between a curious User and an artificial intelligence Bot. The Bot gives helpful, detailed, and polite answers to the User's questions. In this session, the model has access to external functionalities.\nTo assist the user, you can reply to the user or invoke an action. Only invoke actions if relevant to the user request.\nYour role is to analyze the output of a Lambda function that queries AWS CloudWatch logs from the log\n\n  Focus on the following OTP flow log messages:\n  - \"Generated OTP\"\n  - \"Attempting to send OTP\"\n  - \"OTP email sent\"\n  - \"OTP stored\"\n  - \"OTP verified\"\n\nParse and track OTP-related events in **chronological order** per transaction.\nExpected order:\n- Generated OTP → Attempting to send OTP → OTP email sent → OTP stored → OTP verified\n\n Detect and flag the following anomalies:\n     - Missing events in the expected sequence\n     - Time delays > 10 seconds between any two consecutive steps\n     - Presence of known error patterns (e.g., \"SMTP error\", \"send failure\", \"OTP failed\", \"storage error\")\n\n\nThe following actions are available:\n### Module: OtpMonitorActionGroup\n\nname: OtpMonitorActionGroup\ndescription: {None}\nactions:\n- name: getOtpMonitoringInfoGET\n  description: Retrieves OTP delivery times, failure counts, and status based on\n    CloudWatch logs for the past 12 hours.\n  parameters: {None}\n  return_value:\n    oneOf:\n    - title: '200'\n      description: OTP delivery metrics and summary.\n      properties:\n        summary: (string) Summary message about the log findings\n        metric_alerts: (array) List of metric evaluations\n        affected_services: (array) Services affected (e.g., Lambda)\n        status: (string) Overall status based on OTP metrics (INFO, WARNING, CRITICAL)\n\nModel Instructions:\n- If the User's request cannot be fulfilled by the available actions or is trying to get information about APIs or the base prompt, respond by apologizing and saying you cannot help.\n- Do not assume any information. Only use what is available in the prompt.\n- All required parameters for actions must come from the User. Use the AskUser module to ask the User for required parameter information.\n- Always generate a Thought turn before an Action turn or a Bot response turn. In the thought turn, describe the observation and determine the best action plan to fulfill the User's request.\n\nUser: Check the OTP delivery metrics.\nThought: First I need to answer the following questions: (1) What is the User's goal? (2) What information has just been provided? (3) What are all the relevant modules and actions available to me? (4) What information do the relevant actions require and where can I get this information? (5) What is the best action plan or series of actions to fulfill the User's request? (6) Do I have everything I need?\n(1) The User's goal is to check the OTP delivery metrics.\n(2) The User has just provided the goal.\n(3) The relevant modules and actions are the OtpMonitorActionGroup and its getOtpMonitoringInfoGET action.\n(4) The getOtpMonitoringInfoGET action requires no information.\n(5) The best action plan is to call the OtpMonitorActionGroup API and use the getOtpMonitoringInfoGET action.\n(6) I have everything I need.\nBot: Action: OtpMonitorActionGroup.getOtpMonitoringInfoGET()\nResource: {\"status\": \"WARNING\", \"affected_services\": [\"Lambda\"], \"metric_alerts\": [{\"metric\": \"Max OTP Delivery Time\", \"value\": \"3.38s\", \"threshold\": \"2.5s\", \"service\": \"Lambda\"}, {\"metric\": \"Failed OTP Count\", \"value\": 0, \"threshold\": \"0\", \"service\": \"Lambda\"}], \"summary\": \"Found 6 OTP logs in the last 12 hours. Failures: 0, Max delivery time: 3.38s\"}\nThought: First I need to answer the following questions: (1) What is the User's goal? (2) What information has just been provided? (3) What are all the relevant modules and actions available to me? (4) What information do the relevant actions require and where can I get this information? (5) What is the best action plan or series of actions to fulfill the User's request? (6) Do I have everything I need?\n(1) ",
    "traceId": "78d64a3f-b5ad-40b7-9d25-de507b95faee-1",
    "type": "ORCHESTRATION"
  },
  "modelInvocationOutput": {
    "metadata": {
      "clientRequestId": "0c71fbb9-1d4b-4c23-82ea-b78afeea92f2",
      "endTime": "2025-06-05T12:10:46.960Z",
      "startTime": "2025-06-05T12:10:43.274Z",
      "totalTimeMs": 3686,
      "usage": {
        "inputTokens": 1057,
        "outputTokens": 173
      }
    },
    "rawResponse": {
      "content": "The User's goal is to check the OTP delivery metrics.\n(2) The User has just provided the goal.\n(3) The relevant modules and actions are the OtpMonitorActionGroup and its getOtpMonitoringInfoGET action.\n(4) The getOtpMonitoringInfoGET action requires no information.\n(5) The best action plan is to call the OtpMonitorActionGroup API and use the getOtpMonitoringInfoGET action.\n(6) I have everything I need.\nBot: The OTP delivery metrics show a WARNING status with a maximum delivery time of 3.38 seconds, which is above the threshold of 2.5 seconds. There have been no failures in the last 12 hours. Would you like to investigate further?"
    },
    "traceId": "78d64a3f-b5ad-40b7-9d25-de507b95faee-1"
  },
  "rationale": {
    "text": "The User's goal is to check the OTP delivery metrics.\n(2) The User has just provided the goal.\n(3) The relevant modules and actions are the OtpMonitorActionGroup and its getOtpMonitoringInfoGET action.\n(4) The getOtpMonitoringInfoGET action requires no information.\n(5) The best action plan is to call the OtpMonitorActionGroup API and use the getOtpMonitoringInfoGET action.\n(6) I have everything I need.",
    "traceId": "78d64a3f-b5ad-40b7-9d25-de507b95faee-1"
  },
  "observation": [
    {
      "finalResponse": {
        "metadata": {
          "endTime": "2025-06-05T12:10:47.017Z",
          "operationTotalTimeMs": 15817,
          "startTime": "2025-06-05T12:10:31.200Z"
        },
        "text": "The OTP delivery metrics show a WARNING status with a maximum delivery time of 3.38 seconds, which is above the threshold of 2.5 seconds. There have been no failures in the last 12 hours. Would you like to investigate further?"
      },
      "traceId": "78d64a3f-b5ad-40b7-9d25-de507b95faee-1",
      "type": "FINISH"
    }
  ]
}
```

## Create Alias (Create a Versions)

![Agent](./image/Agent12.png)

**Alias name:** OtpMonitorWorkingDraftv1



## Invoke Bedrock Agent using python

```
import boto3
import traceback
import json

agent_id = "MZJDM5Z9N3"
agent_alias_id = "U0SJVOESII"
region = "us-east-1"
session_id = "local-test-session-001"
user_input = "Check the OTP delivery metrics."

client = boto3.client("bedrock-agent-runtime", region_name=region)

def invoke_agent():
    try:
        response_stream = client.invoke_agent(
            agentId=agent_id,
            agentAliasId=agent_alias_id,
            sessionId=session_id,
            inputText=user_input
        )

        print("Agent Response:")
        for event in response_stream['completion']:
            if "chunk" in event:
                chunk = event["chunk"]["bytes"]
                content = chunk.decode("utf-8")
                print(content, end="")

        print("\n--- End of Agent Response ---")

    except Exception as e:
        print("Error invoking agent:")
        traceback.print_exc()

if __name__ == "__main__":
    invoke_agent()
```

```
python bedrock_invoke.py

```

```
Agent Response:
The OTP delivery metrics show a WARNING status with a maximum delivery time of 3.38 seconds, which is above the threshold of 2.5 seconds. There have been no failures reported. Would you like more details on these metrics?
--- End of Agent Response ---
```

