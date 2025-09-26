# Setting up a RunAI container for URSA

**This documentation assumes you are using Run:AI v2**

## What is RunAI?

Run:AI is a platform designed to optimize and manage the use of artificial intelligence (AI) infrastructure, particularly in environments where teams train and deploy machine learning models at scale. It provides a layer of orchestration on top of GPUs and other computing resources, allowing organizations to maximize utilization and efficiency. With features like workload scheduling, resource allocation, and elastic scaling, Run:AI ensures that data scientists and researchers can access the computational power they need without bottlenecks or underutilization. Essentially, it enables organizations to treat their AI compute as a shared pool of resources, improving productivity, collaboration, and cost-effectiveness.


## Why use RunAI for URSA?

### Containerization:
Run:AI leverages containerization to make AI workloads more portable, reproducible, and easier to manage. By running experiments and models inside containers (such as Docker), researchers can ensure that dependencies, libraries, and runtime environments are consistent across different systems. This reduces the "it works on my machine" problem and simplifies scaling AI workloads across clusters of GPUs and CPUs. Containerization also integrates naturally with Kubernetes, which Run:AI uses as its orchestration backbone, giving users the ability to schedule and manage workloads dynamically across shared infrastructure.

### Security:
On the security side, containerization also plays a key role. Containers provide workload isolation, ensuring that experiments and models run in secure, sandboxed environments that reduce the risk of interference or cross-contamination between users. Run:AI builds on Kubernetes’ security features, including role-based access control (RBAC), namespace segregation, and fine-grained permission management. This allows organizations to enforce governance, protect sensitive datasets, and ensure compliance with industry regulations. Additionally, by abstracting GPU resources behind a scheduling and orchestration layer, Run:AI reduces the need for direct low-level access to hardware, adding another layer of control and security.


## What do I do with these files?
First, you will need to talk to your RunAI admins to give you your RunAI account and access. Follow the RunAI instructions on how to install it on your system here: https://run-ai-docs.nvidia.com/saas/reference/cli/install-cli 

Once you have completed that step, you will need to login to RunAI with `runai login remote browser` (if you are using CLI). The system will then ask you to copy a string to your terminal to complete authentication.

Next, you can conviniently use the `sleep_inf.sh` and `exec_container.sh` scripts to start your workload, and then interact with it inside a shell instance. You could do this all by yourself too, and use the `.sh` files as a guide. Note that you will need to include your RunAI account/project variables in a `.env`. Included in this folder is a template you can follow as a reference.

Then from there, you can install URSA inside the RunAI container. You can use `isolation_summary.sh` to check for isolation inside the container.
