---
{"dg-publish":true,"permalink":"/AboutPython/env/2) Venv 에서 가상환경 생성/","tags":["python"],"noteIcon":""}
---

venv 모듈의 경우 온라인 오프라인 설치가 따로 정해져 있지는 않다. (파이썬 내장 가상환경이기 떄문) 
1. 파이썬 을 로컬 PC에 설치한다. 
2. 
```cmd 
# CMD 창을 연후에 가상환경을 만들 폴더로 이동한다.  
cd [가상환경 설치경로]

# 이후 아래 명령어 입력 
python -m venv [가상환경이름]  ## 최신버젼
py -3.11 -m venv [가상환경이름] ##이때 설치하고잫하는 파이썬 버젼이 깔려있어야 가능.

```

가상환경 배포 
설치된 가상환경을 그대로 압축했다가 새로운 경로에 풀면 된다. 
