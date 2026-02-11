\# How to download the dataset



This repository does not include the dataset.



We use:



140K Real vs Fake Faces

https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces



\## Step 1 — Get Kaggle API



1\. Login to Kaggle

2\. Go to https://www.kaggle.com/settings

3\. Click \*\*Create New API Token\*\*

4\. Download `kaggle.json`



\## Step 2 — Place kaggle.json



Create this folder:



```

C:\\Users\\YOUR\_USERNAME\\.kaggle\\

```



Put `kaggle.json` inside it.



\## Step 3 — Download dataset



Open terminal inside the project and run:



```

kaggle datasets download -d xhlulu/140k-real-and-fake-faces

```



\## Step 4 — Extract



```

tar -xf 140k-real-and-fake-faces.zip -C dataset

```



After extraction:



```

dataset/real\_vs\_fake/real-vs-fake/train

dataset/real\_vs\_fake/real-vs-fake/valid

dataset/real\_vs\_fake/real-vs-fake/test

```

