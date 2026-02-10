# Azure OpenAI in Microsoft Foundry models

Azure OpenAI is powered by a diverse set of models with different capabilities and price points.Model availability varies by region and cloud.

| **Model Category**                      | **Description** |
|----------------------------------------|-----------------|
| **GPT-5.1 series**                     | NEW gpt-5.1, gpt-5.1-chat, gpt-5.1-codex, gpt-5.1-codex-mini |
| **Sora**                                | NEW sora-2 |
| **GPT-5 series**                        | gpt-5, gpt-5-mini, gpt-5-nano, gpt-5-chat |
| **gpt-oss**                              | Open-weight reasoning models |
| **codex-mini**                          | Fine-tuned version of o4-mini |
| **GPT-4.1 series**                      | gpt-4.1, gpt-4.1-mini, gpt-4.1-nano |
| **computer-use-preview**               | Experimental model for the Responses API computer-use tool |
| **o-series models**                    | Reasoning models with advanced problem solving and increased focus and capability |
| **GPT-4o, GPT-4o mini, GPT-4 Turbo**   | Azure OpenAI multimodal models (text + images) |
| **GPT-4**                               | Improved over GPT-3.5; understands and generates natural language and code |
| **GPT-3.5**                             | Improved over GPT-3; understands and generates natural language and code |
| **Embeddings**                          | Models that convert text into vector representations for similarity tasks |
| **Image generation**                    | Models that generate original images from text |
| **Video generation**                    | Models that generate original video scenes from text |
| **Audio**                               | Speech-to-text, translation, and text-to-speech models; GPT-4o supports low-latency voice interactions |

# 🎯 Simplified Architecture-Friendly Domain View (recommended for Agent Design)

| Azure Domain              | What Agents You Would Create                |
| ------------------------- | ------------------------------------------- |
| **Identity**              | IdentityAgent, RBACAgent                    |
| **Network**               | NetworkAgent, DNSAgent, FirewallAgent       |
| **Compute**               | ComputeAgent, VMAgent, AppServiceAgent      |
| **Containers**            | AKSAgent, ContainerRegistryAgent            |
| **Storage**               | StorageAgent                                |
| **Database**              | SQLAgent, CosmosAgent, PostgresAgent        |
| **Security**              | KeyVaultAgent, PolicyAgent, DefenderAgent   |
| **Governance/Management** | ResourceGroupAgent, TaggingAgent, CostAgent |
| **Monitoring**            | MonitorAgent, LogAnalyticsAgent             |
| **DevOps**                | TerraformAgent, PipelineAgent               |
| **Integration**           | APIMAgent, ServiceBusAgent                  |
| **AI/Analytics**          | OpenAIAgent, AMLAgent                       |
| **BCDR**                  | BackupAgent, RecoveryAgent                  |

# ✅ MASTER LIST — Azure Domains (Top-Level Enterprise Architecture View)
These are the canonical Azure domains recognized across Microsoft Cloud Adoption Framework (CAF), Landing Zones, Enterprise-Scale Architecture, and Well-Architected Framework.


## 1. Identity & Access Management (IAM)

- Azure Active Directory / Entra ID
- Conditional Access
- Privileged Identity Management (PIM)
- Managed Identities
- RBAC, Custom Roles

## 2. Network & Connectivity

- Virtual Networks (VNet)
- Subnets / NSGs
- Azure Firewall / WAF
- Application Gateway
- VPN Gateway
- ExpressRoute
- Load Balancers
- Private Link / Private Endpoints
- DNS / Private DNS

# 3. Compute

- Azure Virtual Machines
- VM Scale Sets (VMSS)
- Azure App Service
- Azure Functions
- Azure Batch
- Azure Container Instances (ACI)
- Azure Container Apps

