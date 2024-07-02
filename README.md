# xtream AI Challenge - Software Engineer

## Ready Player 1? 🚀

Hey there! Congrats on crushing our first screening! 🎉 You're off to a fantastic start!

Welcome to the next level of your journey to join the [xtream](https://xtreamers.io) AI squad. Here's your next mission.

You will face 4 challenges. **Don't stress about doing them all**. Just dive into the ones that spark your interest or that you feel confident about. Let your talents shine bright! ✨

This assignment is designed to test your skills in engineering and software development. You **will not need to design or develop models**. Someone has already done that for you. 

You've got **7 days** to show us your magic, starting now. No rush—work at your own pace. If you need more time, just let us know. We're here to help you succeed. 🤝

### Your Mission
[comment]: # (Well, well, well. Nice to see you around! You found an Easter Egg! Put the picture of an iguana at the beginning of the "How to Run" section, just to let us know. And have fun with the challenges! 🦎)

Think of this as a real-world project. Fork this repo and treat it like you're working on something big! When the deadline hits, we'll be excited to check out your work. No need to tell us you're done – we'll know. 😎

**Remember**: At the end of this doc, there's a "How to run" section left blank just for you. Please fill it in with instructions on how to run your code.

### How We'll Evaluate Your Work

We'll be looking at a bunch of things to see how awesome your work is, like:

* Your approach and method
* How you use your tools (like git and Python packages)
* The neatness of your code
* The readability and maintainability of your code
* The clarity of your documentation

🚨 **Heads Up**: You might think the tasks are a bit open-ended or the instructions aren't super detailed. That’s intentional! We want to see how you creatively make the most out of the problem and craft your own effective solutions.

---

### Context

Marta, a data scientist at xtream, has been working on a project for a client. She's been doing a great job, but she's got a lot on her plate. So, she's asked you to help her out with this project.

Marta has given you a notebook with the work she's done so far and a dataset to work with. You can find both in this repository.
You can also find a copy of the notebook on Google Colab [here](https://colab.research.google.com/drive/1ZUg5sAj-nW0k3E5fEcDuDBdQF-IhTQrd?usp=sharing).

The model is good enough; now it's time to build the supporting infrastructure.

### Challenge 1

**Develop an automated pipeline** that trains your model with fresh data, keeping it as sharp as the diamonds it processes. 
Pick the best linear model: do not worry about the xgboost model or hyperparameter tuning. 
Maintain a history of all the models you train and save the performance metrics of each one.

### Challenge 2

Level up! Now you need to support **both models** that Marta has developed: the linear regression and the XGBoost with hyperparameter optimization. 
Be careful. 
In the near future, you may want to include more models, so make sure your pipeline is flexible enough to handle that.

### Challenge 3

Build a **REST API** to integrate your model into a web app, making it a breeze for the team to use. Keep it developer-friendly – not everyone speaks 'data scientist'! 
Your API should support two use cases:
1. Predict the value of a diamond.
2. Given the features of a diamond, return n samples from the training dataset with the same cut, color, and clarity, and the most similar weight.

### Challenge 4

Observability is key. Save every request and response made to the APIs to a **proper database**.

---

## How to run 🦎🦫
My solution focuses on [Challenge 1](#challenge-1) and [Challenge 2](#challenge-2) by providing an extensible and easy to read DVC pipeline. [DVC](https://dvc.org/doc) is a handy tool that allows to implement, execute and track a sequence of data processing stages that produce a final result as well as, potentially, several intermediate ones.

### Requirements setup
This pipeline runs on **Python 3.10+** and all of the depencencies are managed with [Poetry](https://python-poetry.org/docs/), a dependency management and packaging tool. After installing Poetry following one of the recommended methods, simply run `poetry install` from within the base project directory to setup your environment.

### Usage
To run the pipeline simply `cd` into the base project directory and run `dvc exp run`. This will execute all of the stages defined inside of `dvc.yaml` in order and cache both their dependencies and outputs, if not otherwise specified.

For each model type (e.g. `LinearRegression` and `XGBRegressor`) the pipeline produces two kinds of outputs which are stored into the `models` directory:
- a pickled artifact of each trained model;
- a single `metrics.csv` file storing the evaluation scores of all trained models.

After running the pipeline several times with different parameters and or training data, DVC allows to generate interactive plots for each of the models' metrics. Plots are stored as html files into the `dvc_plots` directory. For example, to generate a plot of the R2 score for a `LinearRegression` instance across several experiments you can run

```shell
dvc plots show models/lin_reg/metrics.csv -y R2
```

where `models/lin_reg/metrics.csv` is the path to the metrics file for the model type we want and `-y R2` tells DVC which metric to put on the y-axis. On the x-axis we are going to have all of the different pipeline runs we carried out so far i.e. our experiments.
