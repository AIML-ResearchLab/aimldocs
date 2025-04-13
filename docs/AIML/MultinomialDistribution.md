# 🎯 What is a Multinomial Distribution?
The **Multinomial Distribution** is a generalization of the **Binomial Distribution**. While the Binomial Distribution deals with **binary outcomes** (e.g., success/failure), the **Multinomial Distribution** handles scenarios with **more than two possible outcomes**.

# 🧮 Definition
The multinomial distribution gives the **probability of counts** for each possible outcome when you perform a **fixed number of independent experiments**, each with **multiple outcomes**.

**Parameters:**
- **n:** Number of trials (e.g., total votes, total tosses)

- **k:** Number of possible outcomes per trial (e.g., categories)

- **p₁, p₂, ..., pₖ:** Probabilities of each outcome (must sum to 1)


![MD](./img/image19.png)

✅ Real-Life Example

**🗳️ Election Voting**
Let’s say there are 3 political parties: A, B, and C.
    - 100 people vote.
    - Probability of voting:
            - Party A: 0.4
            - Party B: 0.35
            - Party C: 0.25

You want to find the probability that:
    - 40 votes for A
    - 35 votes for B
    - 25 votes for C



# 🧠 Use Case in AI/ML

## 🏷️ Text Classification (NLP)
Multinomial distribution is the foundation of the Multinomial Naive Bayes algorithm, which is widely used in NLP tasks such as:

- Spam Detection
- Sentiment Analysis
- Topic Classification

Each word in a document is considered as a trial, and the probability of each word belonging to a particular class (like spam or not spam) is calculated using the multinomial model.


