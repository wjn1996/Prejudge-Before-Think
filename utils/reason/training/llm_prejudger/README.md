基于bootstrapping的数据，训练一个prejudger模型，用来预判LLM在推理过程中哪些步骤开始可能会陷入错误。

LLM Prejudger与LLM Critique的区别在于：
- LLM Critique：检测当前步骤是否正确。如果不正确，那么当前步骤需要correct或rollback；
- LLM Prejudger：检测当前步骤开始是否可能会存在错误的路径。如果一个step具有预判价值，则还需要给出预判理由和避错信息。

