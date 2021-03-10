```python
import os, glob
```


```python
GH_USER="junha_lee"
PC_USER="junha"
POST_PATH="/Users/"+ PC_USER+"/document/github/"+GH_USER+".github.io/_posts" # <username>.github.io repository 가 있는 주소로 변경
IMG_PATH="/Users/"+ PC_USER+"/document/github/"+GH_USER+".github.io/images"  # <username>.github.io repository 가 있는 주소로 변경
```


```python
os.listdir()
condition = '*.ipynb'
pynbfiles = glob.glob(condition)
```


```python
for i in range(len(pynbfiles)):
    os.system("jupyter nbconvert --to markdown "+ pynbfiles[i] )
```


```python
os.listdir()
condition = '*files'
files = glob.glob(condition)
```


```python
for i in range(len(files)):
    os.system("move "+files[i]+r" C:\Users\junha\Documents\GitHub\junha-lee.github.io\assets\images")
    f = open(files[i][:-6]+'.md', 'r+', encoding='UTF8')
    line = f.read().replace('![png](', '![png](https://raw.githubusercontent.com/junha-lee/junha-lee.github.io/main/assets/images/')
    f.close()
    f2 = open(files[i][:-6]+'.md', 'w', encoding='UTF8')
    f2.write(line)
    f2.close()
```


```python
os.listdir()
condition = '*.md'
mdfiles = glob.glob(condition)
```


```python
for i in range(len(mdfiles)):
    os.system("move "+mdfiles[i]+r" C:\Users\junha\Documents\GitHub\junha-lee.github.io\_posts")
```


```python

```
